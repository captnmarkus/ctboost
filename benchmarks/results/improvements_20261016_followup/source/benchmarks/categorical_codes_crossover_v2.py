"""Predeclared synthetic crossover study for inference dictionary dispatch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def runtime(args):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS"):
        os.environ[key] = "1"
    sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(args.package))
    import ctboost
    if not Path(ctboost.__file__).resolve().is_relative_to(args.package):
        raise ValueError("Unexpected native package")
    spec = importlib.util.spec_from_file_location("dictionary_candidate", args.helper)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    record = {"python": platform.python_version(), "package": str(Path(ctboost.__file__).resolve()),
              "native_sha256": digest(ctboost._core.__file__), "build": ctboost.build_info(),
              "pipeline_sha256": digest(Path(ctboost.__file__).parent / "feature_pipeline.py"),
              "helper_sha256": digest(args.helper), "versions": {key: importlib.metadata.version(key) for key in ("numpy", "pandas", "psutil")}}
    return ctboost, candidate, record


def prepare(args):
    _, _, record = runtime(args)
    if args.output.exists():
        raise ValueError("Preserve existing studies")
    plan = {"protocol": "categorical_dictionary_crossover_v2", "script_sha256": digest(__file__),
        "runtime": record, "seed": 20261016, "widths": [1, 32, 512], "cardinalities": [2, 8, 32],
        "distributions": ["uniform", "skewed_85pct_first"], "batch_sizes": [128, 256, 512, 1000, 2048],
        "training_rows": 2048, "missing_probability": .02,
        "profiles": ["categorical_only", "scalar_mean_frequency", "multiclass_mean"],
        "pipeline": {"ctr_prior_strength": 1.0, "labels": "synthetic independent continuous regression or3class float32; no real quality data"},
        "input_policy": "Each shape/distribution has deterministic independent training and request draws. All batches use prefixes of the same2048-row request. Both arms share identical frames and fitted state.",
        "scope": "FeaturePipeline.transform_array plus fresh shallow DataFrame copy per call; no tree inference or benchmark normalization adapter. Code plan derived once outside measured calls. Candidate helper is identical to frozen broad-dispatch prototype.",
        "passes": [{"arm": "baseline", "order": "forward"}, {"arm": "dictionary", "order": "reverse"},
                   {"arm": "dictionary", "order": "forward"}, {"arm": "baseline", "order": "reverse"}],
        "measurement": {"cpu": 6, "threads": 1, "warmups": 1, "calibration_calls": 1, "blocks": 3,
                        "target_block_seconds": .03, "minimum_calls": 1, "maximum_calls": 50},
        "guard_selection": {"candidate_minimum_rows": [128, 256, 512, 1000, 2048],
            "rule": "Choose the lowest threshold for which EVERY shape/cardinality/distribution at every tested batch>=threshold has pooled median speedup>=1.05 and both paired-order speedups>=1.00. Otherwise retain no automatic dictionary dispatch. No fixture-specific width/cardinality thresholds.",
            "independent_real_replication": "Before production, freeze selected row threshold and compare all5 saved real fixtures in balanced order at1,64,128,256,512,1000,2048 rows. Preserve exact transform/prediction parity. Require each eligible row>=threshold to improve in both orderings, aggregate raw speedup>=1, and fallback controls without repeated>3pct slowdown. A failure rejects this guard rather than tuning it from real fixtures."},
        "limitations": "Local CPU/pandas2 object-string categorical serving experiment, not accuracy or statistical generalization evidence; sparse/highcardinality/mixed/custom inputs remain fallback."}
    write(args.output / "plan.json", plan)
    print(json.dumps({"plan_sha256": digest(args.output / "plan.json"), "cases": 270, "passes": 4}))


def measure(call, plan):
    for _ in range(plan["warmups"]):
        call()
    start = time.perf_counter()
    call()
    elapsed = time.perf_counter() - start
    count = min(plan["maximum_calls"], max(plan["minimum_calls"], int(plan["target_block_seconds"] / max(elapsed, 1e-8))))
    blocks = []
    for _ in range(plan["blocks"]):
        start = time.perf_counter()
        for _ in range(count):
            call()
        blocks.append(1000 * (time.perf_counter() - start) / count)
    return {"median_ms": statistics.median(blocks), "blocks_ms": blocks, "calls_per_block": count}


def fixture(ctboost, candidate, plan, width, cardinality, distribution, profile, index):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(plan["seed"] + index * 1009)
    categories = ["value_" + str(i) for i in range(cardinality)]
    probability = None
    if distribution != "uniform":
        probability = np.array([.85] + [.15 / (cardinality - 1)] * (cardinality - 1))

    def draw(rows):
        codes = rng.choice(cardinality, size=(rows, width), p=probability).astype(np.int8)
        codes[rng.random((rows, width)) < plan["missing_probability"]] = -1
        frame = pd.DataFrame({f"f{column}": pd.Categorical.from_codes(codes[:, column], categories=categories)
                              for column in range(width)})
        return frame, hashlib.sha256(codes.tobytes()).hexdigest()

    train, train_hash = draw(plan["training_rows"])
    request, request_hash = draw(max(plan["batch_sizes"]))
    labels = (rng.integers(0, 3, len(train)) if profile == "multiclass_mean" else rng.standard_normal(len(train))).astype(np.float32)
    controls = {"ordered_ctr": profile != "categorical_only"}
    if profile == "scalar_mean_frequency":
        controls["simple_ctr"] = ["Mean", "Frequency"]
    pipeline = ctboost.FeaturePipeline(cat_features=list(train.columns), **controls,
                                      ctr_prior_strength=plan["pipeline"]["ctr_prior_strength"]).fit(train, labels)
    code_plan = candidate.build_code_plan(pipeline.to_state())
    if code_plan is None:
        raise ValueError("Synthetic fixture is unexpectedly ineligible")
    return pipeline, request, code_plan, {"training_codes_sha256": train_hash,
        "request_codes_sha256": request_hash, "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "output_columns": len(code_plan[1])}


def run(args):
    plan = json.loads((args.output / "plan.json").read_text())
    ctboost, candidate, actual = runtime(args)
    import numpy as np
    import psutil
    if actual != plan["runtime"] or digest(__file__) != plan["script_sha256"]:
        raise ValueError("Frozen source/runtime changed")
    if (args.output / "started.json").exists():
        raise ValueError("A started study is terminal; preserve it")
    psutil.Process().cpu_affinity([plan["measurement"]["cpu"]])
    write(args.output / "started.json", {"plan_sha256": digest(args.output / "plan.json")})
    cases, fixture_receipts = [], []
    index = 0
    for width in plan["widths"]:
        for cardinality in plan["cardinalities"]:
            for distribution in plan["distributions"]:
                for profile in plan["profiles"]:
                    pipeline, frame, code_plan, receipt = fixture(ctboost, candidate, plan, width, cardinality, distribution, profile, index)
                    fixture_receipts.append({"width": width, "cardinality": cardinality, "distribution": distribution, "profile": profile, **receipt})
                    for rows in plan["batch_sizes"]:
                        batch = frame.iloc[:rows]
                        expected = pipeline.transform_array(batch.copy(deep=False))
                        result = candidate.try_code_transform(pipeline, batch.copy(deep=False), code_plan)
                        if result is None:
                            raise ValueError("Unexpected candidate fallback")
                        np.testing.assert_array_equal(result[0].view(np.uint32), expected[0].view(np.uint32))
                        if result[1:] != expected[1:]:
                            raise AssertionError("Transformed metadata changed")
                        cases.append({"key": f"w{width}-k{cardinality}-{distribution}-{profile}-n{rows}", "width": width,
                            "cardinality": cardinality, "distribution": distribution, "profile": profile, "rows": rows,
                            "pipeline": pipeline, "batch": batch, "code_plan": code_plan})
                index += 1
    write(args.output / "fixtures.json", {"fixtures": fixture_receipts, "parity_cases": len(cases), "exact_float32_bits": True})
    passes = []
    for pass_index, policy in enumerate(plan["passes"]):
        records = []
        for case in (cases if policy["order"] == "forward" else list(reversed(cases))):
            if policy["arm"] == "baseline":
                call = lambda c=case: c["pipeline"].transform_array(c["batch"].copy(deep=False))
            else:
                call = lambda c=case: candidate.try_code_transform(c["pipeline"], c["batch"].copy(deep=False), c["code_plan"])
            record = {key: case[key] for key in ("key", "width", "cardinality", "distribution", "profile", "rows")}
            record.update(measure(call, plan["measurement"]))
            records.append(record)
        document = {"pass": pass_index, **policy, "records": records}
        write(args.output / f"pass_{pass_index}.json", document)
        passes.append(document)
        print(json.dumps({"pass_completed": pass_index, "arm": policy["arm"], "cases": len(records)}), flush=True)
    indexed = [{entry["key"]: entry for entry in document["records"]} for document in passes]
    comparisons = []
    for case in cases:
        key = case["key"]
        baseline = statistics.median(indexed[0][key]["blocks_ms"] + indexed[3][key]["blocks_ms"])
        proposed = statistics.median(indexed[1][key]["blocks_ms"] + indexed[2][key]["blocks_ms"])
        ratios = [indexed[0][key]["median_ms"] / indexed[1][key]["median_ms"],
                  indexed[3][key]["median_ms"] / indexed[2][key]["median_ms"]]
        comparisons.append({name: case[name] for name in ("key", "width", "cardinality", "distribution", "profile", "rows")} |
            {"baseline_ms": baseline, "dictionary_ms": proposed, "speedup": baseline / proposed, "ordering_speedups": ratios})
    choices = []
    for threshold in plan["guard_selection"]["candidate_minimum_rows"]:
        selected = [row for row in comparisons if row["rows"] >= threshold]
        failed = [row["key"] for row in selected if row["speedup"] < 1.05 or min(row["ordering_speedups"]) < 1]
        choices.append({"minimum_rows": threshold, "cases": len(selected), "passed": not failed, "failed_cases": failed})
    threshold = next((choice["minimum_rows"] for choice in choices if choice["passed"]), None)
    write(args.output / "comparison.json", {"plan_sha256": digest(args.output / "plan.json"), "rows": comparisons,
        "guard_choices": choices, "selected_minimum_rows": threshold, "requires_independent_real_replication": threshold is not None})
    print(json.dumps({"selected_minimum_rows": threshold, "cases": len(comparisons)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--package", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--helper", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--output", type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.stage](args)
