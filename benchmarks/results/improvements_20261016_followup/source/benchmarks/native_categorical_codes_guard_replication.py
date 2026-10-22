"""Frozen validation of the width32/rows1000 categorical-code dispatch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("crossover", HERE / "native_categorical_codes_crossover.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def command(args, stage, *extra):
    return [sys.executable, "-I", str(Path(__file__).resolve()), stage,
        "--baseline", str(args.baseline), "--candidate", str(args.candidate), "--output", str(args.output),
        "--synthetic", str(args.synthetic), "--references", str(args.references), "--scout", str(args.scout), *map(str, extra)]


def probe(args):
    _, record = study.runtime(args.baseline if args.arm == "baseline" else args.candidate)
    print(json.dumps(record))


def prepare(args):
    if args.output.exists():
        raise ValueError("Preserve previous studies")
    previous = study.read(args.synthetic / "comparison.json")
    if previous["selected_guard"] != {"mode": "universal", "minimum_rows": 1000}:
        raise ValueError("Unexpected prior row-threshold decision")
    baseline = study.read(args.references / "baseline-reference.json")
    candidate = study.read(args.references / "guarded-parity.json")
    if len(baseline["records"]) != 35 or len(candidate["records"]) != 35:
        raise ValueError("Expected35real parity cases before freezing")
    dependencies = {str(path): study.digest(path) for path in (
        HERE / "native_categorical_codes_crossover.py", args.synthetic / "plan.json",
        args.synthetic / "comparison.json", args.synthetic / "fixtures.json",
        args.references / "baseline-reference.json", args.references / "guarded-parity.json")}
    records = {arm: json.loads(subprocess.check_output(command(args, "probe", "--arm", arm), text=True))
               for arm in ("baseline", "candidate")}
    plan = {"protocol": "native_codes_guard_replication_v1", "script_sha256": study.digest(__file__),
        "dependencies": dependencies, "runtime": records,
        "guard": {"minimum_fitted_input_columns": 32, "minimum_request_rows": 1000,
                  "column_names": "Exact pandas.Index/objectdtype with exact builtinstr labels",
                  "fallback": "Public data.dtypes preflight rejects unsupported dtypes/cardinality beforeSeriescreation"},
        "selection_history": "Rows1000 chosen by preregistered360-case synthetic crossover. Its broad candidate failed four ofsix fallback controls; preserved unchanged. Width32 is the smallest studied nontrivial width, added conservatively before this replication, not a universal optimal cutoff. No thresholds selected here.",
        "synthetic": "Reuse all96 eligible width32/512 cases at1000/2048rows and allsix original late33/object fallback controls. Same serialized states and bit signatures; no fitting.",
        "real": {"cases": ["hiva_agnostic", "coil2000_insurance_policies", "splice", "GiveMeSomeCredit", "SDSS17"],
                 "rows": [1, 64, 128, 256, 512, 1000, 2048], "eligible_cases": ["hiva_agnostic", "splice"]},
        "scope": "Synthetic public transform_array plus fresh shallowcopy; real raw prediction including the existing normalize_tabarena_frame adapter, fresh percall. Native/prepared prediction and every transform/prediction bit check outsidetiming. Separate actualpackage processes inABBA with reversedcaseorder.",
        "passes": [{"arm": "baseline", "order": "forward"}, {"arm": "candidate", "order": "reverse"},
                   {"arm": "candidate", "order": "forward"}, {"arm": "baseline", "order": "reverse"}],
        "measurement": {"cpu": 6, "threads": 1, "warmups": 1, "calibration_calls": 1, "blocks": 3,
                        "target_block_seconds": .01, "minimum_calls": 1, "maximum_calls": 50,
                        "maximum_total_run_seconds": 900},
        "acceptance": {"synthetic_eligible": "Every case pooledspeedup>=1.10 and bothpairedorderings>=1.0",
            "real_eligible": "Every eligiblecase/batch improves inbothorderings; all35rawcases geometricmeanspeedup>=1",
            "fallback": "No synthetic or realfallback case regresses>3percent inbothorderings",
            "failure": "Preserve evidence and reject this candidate; no fixture-specific thresholdretuning"},
        "limitations": "Finite local serving evidence with existing developmentinputs only; no accuracy claim. Rows beyondcachedinputs arecycled. Width/range conservatism deliberately givesup some smaller eligiblegains."}
    study.write(args.output / "plan.json", plan)
    print(json.dumps({"plan_sha256": study.digest(args.output / "plan.json"), "synthetic": 102, "real": 35}))


def checked(args, arm):
    plan = study.read(args.output / "plan.json")
    if study.digest(__file__) != plan["script_sha256"]:
        raise ValueError("Frozen runner changed")
    for path, expected in plan["dependencies"].items():
        if study.digest(path) != expected:
            raise ValueError("Frozen evidence changed")
    ctboost, actual = study.runtime(args.baseline if arm == "baseline" else args.candidate)
    if actual != plan["runtime"][arm]:
        raise ValueError("Frozen runtime changed")
    import psutil
    psutil.Process().cpu_affinity([plan["measurement"]["cpu"]])
    return ctboost, plan


def synthetic_cases(args, ctboost):
    cases = []
    for receipt in study.read(args.synthetic / "fixtures.json")["fixtures"]:
        if receipt["eligible"] and receipt["width"] < 32:
            continue
        if study.digest(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Synthetic fixture changed")
        with Path(receipt["path"]).open("rb") as stream:
            payload = pickle.load(stream)
        pipeline = ctboost.FeaturePipeline.from_state(payload["state"])
        for rows in receipt["batch_sizes"]:
            if receipt["eligible"] and rows < 1000:
                continue
            batch = payload["request"].iloc[:rows]
            call = lambda p=pipeline, b=batch: p.transform_array(b.copy(deep=False))
            if study.exact_signature(call()) != payload["expected"][str(rows)]:
                raise AssertionError("Synthetic bits or metadata changed")
            cases.append({"key": "synthetic-" + receipt["key"] + f"-n{rows}", "kind": "synthetic",
                "eligible": receipt["eligible"], "rows": rows, "width": receipt["width"], "call": call})
    if len(cases) != 102:
        raise ValueError("Unexpected synthetic validation population")
    return cases


def real_cases(args, plan):
    import numpy as np
    sys.path.append(str(HERE.parent))
    from benchmarks.inference_real_followup import CompatibleUnpickler
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    references = {(row["case"], row["rows"]): row for row in study.read(args.references / "baseline-reference.json")["records"]}
    cases = []
    for name in plan["real"]["cases"]:
        model_path = args.scout / "fits" / name / "ctboost_default" / "model.pkl"
        data_path = args.scout / "data" / name / "development.pkl"
        first_reference = references[(name, plan["real"]["rows"][0])]
        if study.digest(model_path) != first_reference["model_sha256"] or study.digest(data_path) != first_reference["input_sha256"]:
            raise ValueError("Real fixture changed")
        with model_path.open("rb") as stream:
            payload = CompatibleUnpickler(stream).load()
        with data_path.open("rb") as stream:
            source = CompatibleUnpickler(stream).load()["X"]
        model, categorical = payload["model"], payload["categorical_columns"]
        raw = source.iloc[np.arange(max(plan["real"]["rows"])) % len(source)].copy().reset_index(drop=True)
        transform_reference = args.references / "reference" / f"{name}-transformed.npy"
        if study.digest(transform_reference) != first_reference["transformed_reference_sha256"]:
            raise ValueError("Transformed reference changed")
        normalize = lambda b, c=categorical: normalize_tabarena_frame(b, categorical_columns=c)[0]
        predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        native = model.get_booster()._handle
        for rows in plan["real"]["rows"]:
            batch = raw.iloc[:rows]
            normalized = normalize(batch)
            transformed = model._feature_pipeline.transform_array(normalized)[0]
            expected_transform = np.load(transform_reference)[:rows]
            np.testing.assert_array_equal(np.ascontiguousarray(transformed).view(np.uint32), np.ascontiguousarray(expected_transform).view(np.uint32))
            prepared = model._transform_prediction_pool(normalized)
            reference_path = args.references / "reference" / f"{name}-{rows}.npy"
            if study.digest(reference_path) != references[(name, rows)]["expected_sha256"]:
                raise ValueError("Prediction reference changed")
            expected = np.load(reference_path)
            for actual in (predict(normalized), predict(prepared)):
                np.testing.assert_array_equal(np.ascontiguousarray(actual).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8))
            np.testing.assert_array_equal(np.asarray(native.predict(prepared._handle)).view(np.uint32),
                                          np.asarray(native._predict_uncached(prepared._handle)).view(np.uint32))
            cases.append({"key": f"real-{name}-n{rows}", "kind": "real", "case": name, "rows": rows,
                "eligible": name in plan["real"]["eligible_cases"] and rows >= 1000,
                "call": lambda b=batch, n=normalize, p=predict: p(n(b))})
    return cases


def timed_pass(args):
    policy = study.read(args.output / "plan.json")["passes"][args.pass_index]
    ctboost, plan = checked(args, policy["arm"])
    output = args.output / f"pass_{args.pass_index}.json"
    if output.exists():
        raise ValueError("Preserve completed pass")
    cases = synthetic_cases(args, ctboost) + real_cases(args, plan)
    records = []
    for case in cases if policy["order"] == "forward" else list(reversed(cases)):
        records.append({**{key: value for key, value in case.items() if key != "call"},
                        **study.measure(case["call"], plan["measurement"])})
    study.write(output, {"pass": args.pass_index, **policy, "plan_sha256": study.digest(args.output / "plan.json"),
        "all_exact_transform_and_prediction_checks_passed": True, "records": records})
    print(json.dumps({"pass_completed": args.pass_index, "arm": policy["arm"], "cases": len(records)}), flush=True)


def run(args):
    import math
    plan = study.read(args.output / "plan.json")
    if study.digest(__file__) != plan["script_sha256"] or (args.output / "started.json").exists():
        raise ValueError("Changed or already-started study")
    study.write(args.output / "started.json", {"plan_sha256": study.digest(args.output / "plan.json")})
    deadline = time.monotonic() + plan["measurement"]["maximum_total_run_seconds"]
    for index in range(4):
        subprocess.run(command(args, "pass", "--pass-index", index), check=True, timeout=max(1, deadline - time.monotonic()))
    indexed = [{row["key"]: row for row in study.read(args.output / f"pass_{index}.json")["records"]} for index in range(4)]
    rows = []
    for key, case in indexed[0].items():
        baseline = statistics.median(indexed[0][key]["blocks_ms"] + indexed[3][key]["blocks_ms"])
        candidate = statistics.median(indexed[1][key]["blocks_ms"] + indexed[2][key]["blocks_ms"])
        paired = [indexed[0][key]["median_ms"] / indexed[1][key]["median_ms"], indexed[3][key]["median_ms"] / indexed[2][key]["median_ms"]]
        rows.append({**{k: v for k, v in case.items() if k not in ("median_ms", "blocks_ms", "calls_per_block")},
                     "baseline_ms": baseline, "candidate_ms": candidate, "speedup": baseline / candidate, "ordering_speedups": paired})
    failures = []
    for row in rows:
        if row["eligible"]:
            if min(row["ordering_speedups"]) < 1 or (row["kind"] == "synthetic" and row["speedup"] < 1.1):
                failures.append(row["key"])
        elif max(row["ordering_speedups"]) < 1 / 1.03:
            failures.append(row["key"])
    raw_geomean = math.exp(statistics.mean(math.log(row["speedup"]) for row in rows if row["kind"] == "real"))
    passed = not failures and raw_geomean >= 1
    study.write(args.output / "comparison.json", {"plan_sha256": study.digest(args.output / "plan.json"), "rows": rows,
        "passed": passed, "failed_cases": failures, "real_raw_geomean_speedup": raw_geomean})
    print(json.dumps({"passed": passed, "failed_cases": failures, "real_raw_geomean_speedup": raw_geomean}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "prepare", "pass", "run"))
    for name in ("baseline", "candidate", "output", "synthetic", "references", "scout"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--arm", choices=("baseline", "candidate"))
    parser.add_argument("--pass-index", type=int, choices=range(4))
    args = parser.parse_args()
    {"probe": probe, "prepare": prepare, "pass": timed_pass, "run": run}[args.stage](args)
