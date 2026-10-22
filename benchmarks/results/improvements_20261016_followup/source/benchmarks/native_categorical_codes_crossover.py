"""Predeclared, separate-runtime synthetic crossover for categorical codes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def runtime(package):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS"):
        os.environ[key] = "1"
    sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(package))
    import ctboost
    if not Path(ctboost.__file__).resolve().is_relative_to(package):
        raise ValueError("Unexpected native package")
    helper = package / "ctboost" / "_native_categorical_codes_prototype.py"
    record = {"python": platform.python_version(), "package": str(Path(ctboost.__file__).resolve()),
              "native_sha256": digest(ctboost._core.__file__), "build": ctboost.build_info(),
              "pipeline_sha256": digest(package / "ctboost" / "feature_pipeline.py"),
              "helper_sha256": digest(helper) if helper.exists() else None,
              "build_receipt_sha256": digest(package / "build_receipt.json"),
              "versions": {key: importlib.metadata.version(key) for key in ("numpy", "pandas", "psutil")}}
    return ctboost, record


def command(args, stage, *extra):
    return [sys.executable, "-I", str(Path(__file__).resolve()), stage,
            "--baseline", str(args.baseline), "--candidate", str(args.candidate),
            "--output", str(args.output), "--artifacts", str(args.artifacts), *map(str, extra)]


def probe(args):
    _, record = runtime(args.baseline if args.arm == "baseline" else args.candidate)
    print(json.dumps(record))


def prepare(args):
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous studies; choose fresh paths")
    records = {arm: json.loads(subprocess.check_output(command(args, "probe", "--arm", arm), text=True))
               for arm in ("baseline", "candidate")}
    if records["baseline"]["versions"] != records["candidate"]["versions"]:
        raise ValueError("Unmatched dependency versions")
    plan = {"protocol": "native_categorical_dictionary_crossover_v1", "script_sha256": digest(__file__),
        "runtime": records, "seed": 20261016, "widths": [1, 32, 512], "cardinalities": [2, 8, 32],
        "distributions": ["uniform", "skewed_85pct_first"], "batch_sizes": [128, 256, 512, 1000, 2048],
        "training_rows": 2048, "missing_probability": .02,
        "profiles": ["categorical_only", "scalar_mean", "scalar_mean_frequency", "multiclass_mean"],
        "pipeline": {"ctr_prior_strength": 1.0, "labels": "Independent synthetic continuous regression or3class float32"},
        "input_policy": "Each width/cardinality/distribution shares the same training and request category draws across all4 output profiles; batches are prefixes of one2048-row request. Baseline creates fitted states and expected float32 bit hashes once. Both isolated packages restore exactly those states.",
        "scope": "Entire public FeaturePipeline.transform_array plus fresh shallow DataFrame copy per call, in separate actual composed/native-package processes. Fitting, restore, hashing, exact parity and imports are outside timings. No tree inference, accuracy data or benchmark normalization adapter in this synthetic phase.",
        "passes": [{"arm": "baseline", "order": "forward"}, {"arm": "candidate", "order": "reverse"},
                   {"arm": "candidate", "order": "forward"}, {"arm": "baseline", "order": "reverse"}],
        "measurement": {"cpu": 6, "threads": 1, "warmups": 1, "calibration_calls": 1, "blocks": 3,
                        "target_block_seconds": .01, "minimum_calls": 1, "maximum_calls": 50,
                        "maximum_total_run_seconds": 900},
        "guard_selection": {"candidate_minimum_rows": [128, 256, 512, 1000, 2048],
            "minimum_pooled_speedup": 1.10, "minimum_each_ordering_speedup": 1.0,
            "rule": "Choose the lowest universal row threshold where EVERY profile/width/cardinality/distribution at EVERY tested batch>=threshold has pooled median speedup>=1.10 and both paired-order speedups>=1.0. Prefer it if<=1000. Otherwise use the same rule independently in four predeclared metadata buckets, with unqualified buckets falling back. No additional buckets are invented from results.",
            "metadata_buckets": {"uniform_outputs_per_input": ["1", ">=2"], "maximum_dictionary_size": ["<=8", "9..32"],
                "rule": "Require full source coverage with equal output multiplicity across all inputs; mixed/partial/unknown plans fall back. Each bucket threshold must pass every synthetic width, distribution, cardinality and output profile in that bucket at every tested batch above threshold."},
            "fallback_controls": "At1000rows, test object-string frames and all-categorical frames with33categories only in the last column, for widths1/32/512 and multiclass-Mean output profile. Baseline/current candidate exact parity and timing are recorded separately. A deployed guard must not introduce repeated>3pct slowdown on these synthetic fallback controls; initial controls diagnose the broad candidate and do not select row thresholds.",
            "independent_real_replication": "Freeze the synthetic threshold before looking at additional real batch timings. Compare all5 saved real fixtures at1,64,128,256,512,1000,2048 in balanced order with exact transform/prediction bits. Each eligible tested batch must improve in both orderings; overall raw geometric speedup>=1 and fallback controls must have no repeated>3pct regression. Failure rejects the guard rather than retuning from real fixtures."},
        "limitations": "Finite local CPU/pandas object-string categorical serving experiment, not a universal speed or accuracy guarantee. One-row behavior stays on existing path; custom/mixed/high-cardinality inputs fall back. Exact parity hashes cover every synthetic case in every runtime/pass."}
    args.artifacts.mkdir(parents=True)
    write(args.output / "plan.json", plan)
    print(json.dumps({"plan_sha256": digest(args.output / "plan.json"), "eligible_cases": 360, "fallback_controls": 6, "passes": 4}))


def checked(args, arm):
    plan = read(args.output / "plan.json")
    ctboost, actual = runtime(args.baseline if arm == "baseline" else args.candidate)
    if actual != plan["runtime"][arm] or digest(__file__) != plan["script_sha256"]:
        raise ValueError("Frozen source/runtime changed")
    import psutil
    psutil.Process().cpu_affinity([plan["measurement"]["cpu"]])
    return ctboost, plan


def exact_signature(result):
    import numpy as np
    array, categories, names = result
    if array.dtype != np.float32:
        raise AssertionError("Expected float32 transform")
    return {"shape": list(array.shape), "float32_bits_sha256": hashlib.sha256(array.tobytes(order="F")).hexdigest(),
            "categorical_indices": list(categories), "output_names": list(names)}


def make_fixture(ctboost, plan, width, cardinality, distribution, profile, index):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(plan["seed"] + index * 1009)
    categories = ["value_" + str(i) for i in range(cardinality)]
    probability = None if distribution == "uniform" else np.array([.85] + [.15 / (cardinality - 1)] * (cardinality - 1))

    def draw(rows):
        codes = rng.choice(cardinality, size=(rows, width), p=probability).astype(np.int8)
        codes[rng.random((rows, width)) < plan["missing_probability"]] = -1
        frame = pd.DataFrame({f"f{column}": pd.Categorical.from_codes(codes[:, column], categories=categories)
                              for column in range(width)})
        return frame, hashlib.sha256(codes.tobytes()).hexdigest()

    training, train_hash = draw(plan["training_rows"])
    request, request_hash = draw(max(plan["batch_sizes"]))
    labels = (rng.integers(0, 3, len(training)) if profile == "multiclass_mean" else rng.standard_normal(len(training))).astype(np.float32)
    controls = {"ordered_ctr": profile != "categorical_only"}
    if profile == "scalar_mean_frequency":
        controls["simple_ctr"] = ["Mean", "Frequency"]
    pipeline = ctboost.FeaturePipeline(cat_features=list(training.columns), **controls,
        ctr_prior_strength=plan["pipeline"]["ctr_prior_strength"]).fit(training, labels)
    expected = {str(rows): exact_signature(pipeline.transform_array(request.iloc[:rows].copy(deep=False)))
                for rows in plan["batch_sizes"]}
    state = pipeline.to_state()
    output_count = len(state["output_feature_names_"])
    if output_count % width:
        raise ValueError("Expected uniform complete output profile")
    return {"state": state, "request": request, "expected": expected}, {
        "training_codes_sha256": train_hash, "request_codes_sha256": request_hash,
        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "output_columns": output_count, "outputs_per_input": output_count // width}


def fixtures(args):
    ctboost, plan = checked(args, "baseline")
    if (args.output / "fixtures.json").exists() or (args.artifacts / "fixtures").exists():
        raise ValueError("Preserve existing fixture attempts")
    directory = args.artifacts / "fixtures"
    directory.mkdir()
    receipts, index = [], 0
    for width in plan["widths"]:
        for cardinality in plan["cardinalities"]:
            for distribution in plan["distributions"]:
                for profile in plan["profiles"]:
                    key = f"w{width}-k{cardinality}-{distribution}-{profile}"
                    payload, receipt = make_fixture(ctboost, plan, width, cardinality, distribution, profile, index)
                    path = directory / (key + ".pkl")
                    with path.open("wb") as stream:
                        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    receipts.append({"key": key, "width": width, "cardinality": cardinality, "distribution": distribution,
                        "profile": profile, "path": str(path), "sha256": digest(path), "eligible": True,
                        "batch_sizes": plan["batch_sizes"], **receipt})
                    if cardinality == 32 and distribution == "uniform" and profile == "multiclass_mean":
                        for kind in ("late_33_categories", "object_strings"):
                            fallback_key = key + "-" + kind
                            request = payload["request"].copy()
                            if kind == "late_33_categories":
                                column = request.columns[-1]
                                request[column] = request[column].cat.add_categories(["unused_extra_category"])
                            else:
                                request = request.astype(object)
                            pipeline = ctboost.FeaturePipeline.from_state(payload["state"])
                            fallback = {"state": payload["state"], "request": request,
                                "expected": {"1000": exact_signature(pipeline.transform_array(request.iloc[:1000].copy(deep=False)))}}
                            fallback_path = directory / (fallback_key + ".pkl")
                            with fallback_path.open("wb") as stream:
                                pickle.dump(fallback, stream, protocol=pickle.HIGHEST_PROTOCOL)
                            receipts.append({"key": fallback_key, "width": width, "cardinality": 33 if kind == "late_33_categories" else 32,
                                "distribution": distribution, "profile": profile, "path": str(fallback_path),
                                "sha256": digest(fallback_path), "eligible": False, "fallback_kind": kind,
                                "batch_sizes": [1000], **receipt})
                index += 1
    write(args.output / "fixtures.json", {"plan_sha256": digest(args.output / "plan.json"), "fixtures": receipts,
        "expected_cases": sum(len(receipt["batch_sizes"]) for receipt in receipts)})
    print(json.dumps({"fixtures": len(receipts), "expected_cases": 366}), flush=True)


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


def timed_pass(args):
    plan = read(args.output / "plan.json")
    policy = plan["passes"][args.pass_index]
    ctboost, plan = checked(args, policy["arm"])
    path = args.output / f"pass_{args.pass_index}.json"
    if path.exists():
        raise ValueError("Preserve existing timed pass")
    document = read(args.output / "fixtures.json")
    if document["plan_sha256"] != digest(args.output / "plan.json"):
        raise ValueError("Fixture plan changed")
    cases = []
    for receipt in document["fixtures"]:
        if digest(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Fixture changed")
        with Path(receipt["path"]).open("rb") as stream:
            payload = pickle.load(stream)
        pipeline = ctboost.FeaturePipeline.from_state(payload["state"])
        if policy["arm"] == "candidate" and pipeline._categorical_code_plan is None:
            raise ValueError("Unexpected dictionary ineligibility")
        for rows in receipt["batch_sizes"]:
            batch = payload["request"].iloc[:rows]
            if exact_signature(pipeline.transform_array(batch.copy(deep=False))) != payload["expected"][str(rows)]:
                raise AssertionError("Synthetic transform bits or metadata changed")
            cases.append({**{key: receipt[key] for key in ("key", "width", "cardinality", "distribution", "profile", "output_columns", "outputs_per_input", "eligible")},
                          "key": receipt["key"] + f"-n{rows}", "rows": rows, "pipeline": pipeline, "batch": batch})
    records = []
    for case in (cases if policy["order"] == "forward" else list(reversed(cases))):
        call = lambda c=case: c["pipeline"].transform_array(c["batch"].copy(deep=False))
        record = {key: value for key, value in case.items() if key not in ("pipeline", "batch")}
        record.update(measure(call, plan["measurement"]))
        records.append(record)
    write(path, {"plan_sha256": digest(args.output / "plan.json"), "fixtures_sha256": digest(args.output / "fixtures.json"),
        "pass": args.pass_index, **policy, "exact_float32_bits_and_metadata": True, "records": records})
    print(json.dumps({"pass_completed": args.pass_index, "arm": policy["arm"], "cases": len(records)}), flush=True)


def run(args):
    plan = read(args.output / "plan.json")
    if digest(__file__) != plan["script_sha256"] or (args.output / "started.json").exists():
        raise ValueError("Changed or already-started study; preserve it")
    write(args.output / "started.json", {"plan_sha256": digest(args.output / "plan.json")})
    deadline = time.monotonic() + plan["measurement"]["maximum_total_run_seconds"]
    subprocess.run(command(args, "fixtures"), check=True, timeout=max(1, deadline - time.monotonic()))
    for index in range(len(plan["passes"])):
        subprocess.run(command(args, "pass", "--pass-index", index), check=True, timeout=max(1, deadline - time.monotonic()))
    passes = [read(args.output / f"pass_{index}.json") for index in range(4)]
    indexed = [{entry["key"]: entry for entry in document["records"]} for document in passes]
    comparisons = []
    for key, case in indexed[0].items():
        baseline = statistics.median(indexed[0][key]["blocks_ms"] + indexed[3][key]["blocks_ms"])
        proposed = statistics.median(indexed[1][key]["blocks_ms"] + indexed[2][key]["blocks_ms"])
        ratios = [indexed[0][key]["median_ms"] / indexed[1][key]["median_ms"],
                  indexed[3][key]["median_ms"] / indexed[2][key]["median_ms"]]
        comparisons.append({name: case[name] for name in ("key", "width", "cardinality", "distribution", "profile", "rows", "output_columns", "outputs_per_input", "eligible")} |
            {"baseline_ms": baseline, "candidate_ms": proposed, "speedup": baseline / proposed, "ordering_speedups": ratios})
    def thresholds(rows):
        choices = []
        for threshold in plan["guard_selection"]["candidate_minimum_rows"]:
            selected = [row for row in rows if row["rows"] >= threshold]
            failed = [row["key"] for row in selected if row["speedup"] < plan["guard_selection"]["minimum_pooled_speedup"]
                      or min(row["ordering_speedups"]) < plan["guard_selection"]["minimum_each_ordering_speedup"]]
            choices.append({"minimum_rows": threshold, "cases": len(selected), "passed": bool(selected) and not failed, "failed_cases": failed})
        return choices, next((choice["minimum_rows"] for choice in choices if choice["passed"]), None)
    eligible = [row for row in comparisons if row["eligible"]]
    choices, threshold = thresholds(eligible)
    buckets = []
    for multiplicity in ("1", ">=2"):
        for cardinality in ("<=8", "9..32"):
            rows = [row for row in eligible if ((row["outputs_per_input"] == 1) == (multiplicity == "1"))
                    and ((row["cardinality"] <= 8) == (cardinality == "<=8"))]
            bucket_choices, selected = thresholds(rows)
            buckets.append({"uniform_outputs_per_input": multiplicity, "maximum_dictionary_size": cardinality,
                            "guard_choices": bucket_choices, "selected_minimum_rows": selected})
    mode = "universal" if threshold is not None and threshold <= 1000 else "metadata_buckets"
    selected_guard = {"mode": mode, "minimum_rows": threshold} if mode == "universal" else {"mode": mode, "buckets": buckets}
    has_guard = threshold is not None if mode == "universal" else any(bucket["selected_minimum_rows"] is not None for bucket in buckets)
    write(args.output / "comparison.json", {"plan_sha256": digest(args.output / "plan.json"), "rows": comparisons,
        "guard_choices": choices, "selected_guard": selected_guard,
        "requires_independent_real_replication": has_guard,
        "fallback_control_repeated_regressions": [row["key"] for row in comparisons if not row["eligible"]
            and max(row["ordering_speedups"]) < 1 / 1.03]})
    print(json.dumps({"guard_mode": mode, "universal_threshold": threshold,
        "bucket_thresholds": [{key: value for key, value in bucket.items() if key != "guard_choices"} for bucket in buckets],
        "cases": len(comparisons)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "prepare", "fixtures", "pass", "run"))
    for name in ("baseline", "candidate", "output", "artifacts"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--pass-index", type=int, choices=range(4))
    args = parser.parse_args()
    {"probe": probe, "prepare": prepare, "fixtures": fixtures, "pass": timed_pass, "run": run}[args.stage](args)
