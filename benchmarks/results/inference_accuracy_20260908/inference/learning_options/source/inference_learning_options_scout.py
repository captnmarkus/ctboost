"""Timing-only follow-up for the existing grouped and joint-feature models.

No fitting, model selection, scoring, or confirmation-role access. The fixed
family mapping is exploratory, chosen after the earlier development studies.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def start(cpus):
    for key in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import ctboost  # -I must resolve the installed wheel before exposing ROOT.
    sys.path.insert(0, str(ROOT))
    from benchmarks import inference_tabarena_scout as base
    runtime = base.bootstrap(cpus, 1)
    runtime["pipeline_sha256"] = base.digest(Path(ctboost.__file__).parent / "feature_pipeline.py")
    return base, runtime


def prepare(args, base, runtime):
    original = base.checked_plan(args.original)
    joint = base.read(args.joint / "plan.json")
    cases = []
    files = {str(args.original / "plan.json"): base.digest(args.original / "plan.json"),
             str(Path(original["scout"]) / "plan.json"): original["scout_plan_sha256"],
             str(args.joint / "plan.json"): base.digest(args.joint / "plan.json"),
             str(Path(base.__file__)): base.digest(base.__file__),
             str(Path(__file__)): base.digest(__file__)}
    for name, digest in original["source_sha256"].items():
        files[str(ROOT / name)] = digest
    for case in original["datasets"]:
        name = case["dataset_name"]
        arm = "joint_feature_scalar_cut" if case["problem_type"] == "multiclass" else "ctboost_grouped8"
        source = args.joint if case["problem_type"] == "multiclass" else Path(original["scout"])
        directory = source / "fits" / name / arm
        record = base.read(directory / "result.json")
        assert record["status"] == "ok" and record["resource_failure"] is None
        assert record["plan_sha256"] == base.digest(source / "plan.json")
        if arm == "joint_feature_scalar_cut":
            assert record["parameters"]["multiclass_split_score"] == "scalar"
            assert record["parameters"]["multiclass_feature_test"] == "joint"
        else:
            assert record["parameters"]["feature_test"] == "grouped"
        assert base.digest(directory / "model.pkl") == record["model_sha256"]
        assert base.digest(directory / "development.npz") == record["development_sha256"]
        data = source / "data" / name / "development.pkl"
        assert base.digest(data) == case["roles"]["development"]["sha256"]
        for path in (directory / "result.json", directory / "model.pkl", directory / "development.npz", data):
            files[str(path)] = base.digest(path)
        cat = args.original / "catboost" / name
        cat_record = base.read(cat / "result.json")
        assert cat_record["status"] == "ok" and cat_record["resource_failure"] is None
        assert cat_record["plan_sha256"] == base.digest(args.original / "plan.json")
        assert base.digest(cat / "model.pkl") == cat_record["model_sha256"]
        for path in (cat / "result.json", cat / "model.pkl"):
            files[str(path)] = base.digest(path)
        reference = args.original / "measurements/baseline-2" / f"{name}.npz"
        files[str(reference)] = base.digest(reference)
        cases.append({"dataset_name": name, "problem_type": case["problem_type"],
                      "ctboost_arm": arm, "ctboost_source": str(directory), "catboost_source": str(cat),
                      "development_path": str(data), "development_rows": case["roles"]["development"]["rows"],
                      "reference_runtime": joint["native_sha256"] if arm == "joint_feature_scalar_cut" else original["public_runtime"]["native_sha256"]})
    assert len(cases) == 14
    args.output.mkdir(parents=True, exist_ok=False)
    base.write(args.output / "plan.json", {
        "protocol_id": "inference_learning_options_20260908_v1", "created_at": base.now(),
        "selection": "Fixed family mapping after development: grouped8 for all6 binary/regression tasks; grouped8+joint feature test with scalar cut for all8 multiclass tasks. No task-specific choice.",
        "limitations": "Timing-only exploratory follow-up, reused development rows/models. No new fitting, quality selection, confirmation access, Elo or official benchmark claim.",
        "runtime": runtime, "measurement_resources": {"cpus": [6], "threads": 1},
        "measurement": "Same1000-row cycling, normalization, model preprocessing, first-call cold and seven warm blocks as original inference scout; no bagging.",
        "original_inference": str(args.original), "files_sha256": files, "cases": cases,
    })


def load(plan, case, arm):
    source = Path(case[f"{arm}_source"])
    with (source / "model.pkl").open("rb") as handle:
        return pickle.load(handle)


def predictor(payload, frame, problem, arm):
    import numpy as np
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    model = payload["model"]
    if arm == "catboost":
        for method in ("predict", "predict_proba"):
            if hasattr(model.model, method):
                setattr(model.model, method, functools.partial(getattr(model.model, method), thread_count=1))

    def predict():
        normalized, _ = normalize_tabarena_frame(frame, categorical_columns=payload["categorical_columns"])
        values = np.asarray(model.predict(normalized) if problem == "regression" else model.predict_proba(normalized))
        return values[:, 1] if problem == "binary" and values.ndim == 2 else values
    return predict


def run(args, base, runtime):
    import numpy as np
    import psutil
    plan = base.read(args.output / "plan.json")
    for key in ("versions", "native_sha256", "pipeline_sha256"):
        if plan["runtime"][key] != runtime[key]:
            raise ValueError(f"Changed runtime: {key}")
    for path, digest in plan["files_sha256"].items():
        if base.digest(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    if args.mode == "measure":
        assert args.cpus == plan["measurement_resources"]["cpus"]
        preflight = base.read(args.output / "preflight.json")
        assert preflight["plan_sha256"] == base.digest(args.output / "plan.json")
        assert preflight["all14_exact"] and len(preflight["rows"]) == 14
    directory = args.output / args.mode
    directory.mkdir(exist_ok=False)
    rows = []
    for case in plan["cases"]:
        with Path(case["development_path"]).open("rb") as handle:
            role = pickle.load(handle)
        with np.load(Path(case["ctboost_source"]) / "development.npz", allow_pickle=False) as reference:
            expected = reference["predictions"].copy()
            np.testing.assert_array_equal(reference["outer_train_positions"], role["outer_train_positions"])
        if args.mode == "preflight":
            payload = load(plan, case, "ctboost")
            actual = predictor(payload, role["X"], case["problem_type"], "ctboost")()
            np.testing.assert_array_equal(actual, expected)
            rows.append({"dataset": case["dataset_name"], "arm": case["ctboost_arm"], "exact": True,
                         "prediction_sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()})
        else:
            positions = np.arange(1000) % len(role["X"])
            frame = role["X"].iloc[positions].copy().reset_index(drop=True)
            row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"],
                   "ctboost_arm": case["ctboost_arm"], "development_rows": len(role["X"]),
                   "timed_rows": 1000, "models": {}}
            arms = ["ctboost", "catboost"] if len(rows) % 2 == 0 else ["catboost", "ctboost"]
            predictions = {}
            for arm in arms:
                payload = load(plan, case, arm)
                predict = predictor(payload, frame, case["problem_type"], arm)
                started = time.perf_counter()
                prediction = predict()
                cold = (time.perf_counter() - started) * 1000
                if arm == "ctboost":
                    np.testing.assert_array_equal(prediction, expected[positions])
                else:
                    old = Path(plan["original_inference"]) / "measurements/baseline-2" / f"{case['dataset_name']}.npz"
                    with np.load(old, allow_pickle=False) as reference:
                        np.testing.assert_array_equal(prediction, reference["catboost_ag_default"])
                timing = base.measure_call(predict)
                np.testing.assert_array_equal(predict(), prediction)
                receipt = base.read(Path(case[f"{arm}_source"]) / "result.json")
                model = payload["model"]
                trees = int(model.get_booster()._handle.num_trees()) if arm == "ctboost" else int(model.model.tree_count_)
                row["models"][arm] = {**timing, "cold_ms": cold, "rounds": receipt["rounds"], "trees": trees,
                                      "model_sha256": receipt["model_sha256"], "rss_bytes_after_warmup": psutil.Process().memory_info().rss}
                predictions[arm] = prediction
            np.savez_compressed(directory / f"{case['dataset_name']}.npz", **predictions)
            row["prediction_sha256"] = base.digest(directory / f"{case['dataset_name']}.npz")
            row["reference_prediction_identity"] = True
            rows.append(row)
        base.write(directory / "progress.json", {"rows": rows})
        print(json.dumps({"mode": args.mode, "dataset": case["dataset_name"], "exact": True}), flush=True)
    base.write(args.output / f"{args.mode}.json", {"created_at": base.now(), "runtime": runtime,
               "plan_sha256": base.digest(args.output / "plan.json"), "all14_exact": len(rows) == 14, "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "measure"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--joint", type=Path)
    parser.add_argument("--cpus", type=lambda text: [int(value) for value in text.split(",")], default=[6])
    args = parser.parse_args()
    for name in ("output", "original", "joint"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    base, runtime = start(args.cpus)
    if args.mode == "prepare":
        if args.original is None or args.joint is None:
            parser.error("prepare requires --original and --joint")
        prepare(args, base, runtime)
    else:
        run(args, base, runtime)


if __name__ == "__main__":
    main()
