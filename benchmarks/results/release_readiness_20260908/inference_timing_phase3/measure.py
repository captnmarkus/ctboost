"""Measure saved models only; keep the original diagnostic and environments immutable."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".tmp/release-readiness-20260908"
ORIGINAL = ROOT / ".tmp/inference-accuracy-20260908/real-latency"
FINAL_PACKAGE = RELEASE / "release-env/Lib/site-packages/ctboost"
FINAL_WHEEL = RELEASE / "dist-portable/ctboost-0.1.61-cp312-cp312-win_amd64.whl"
FINAL_WHEEL_SHA = "b9fb7b598395a64380ee2f5a2395269ef567169edd151916790087dd71177f94"
FINAL_NATIVE_SHA = "5b91bb3139dd54ab05f5c8dea2b1bd1caabf0b9dcb2a2e76cafb01b30281e1b5"
ORDER = ("baseline", "final", "final", "baseline")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def helper():
    path = ROOT / "benchmarks/inference_tabarena_scout.py"
    assert sha(path) == read(ORIGINAL / "plan.json")["runner_sha256"]
    spec = importlib.util.spec_from_file_location("original_latency_helper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identities(original):
    result = {}
    for case in original["datasets"]:
        name = case["dataset_name"]
        scout = Path(original["scout"])
        paths = [scout / "data" / name / "development.pkl"]
        for location in (scout / "fits" / name / "ctboost_default", ORIGINAL / "catboost" / name):
            paths.extend(location / filename for filename in ("model.pkl", "result.json"))
        paths.extend(ORIGINAL / "measurements" / f"baseline-{repeat}" / f"{name}.npz"
                     for repeat in (1, 2))
        result[name] = {str(path): sha(path) for path in paths}
    return result


def prepare(output):
    helpers = helper()
    original = helpers.checked_plan(ORIGINAL)
    assert sha(FINAL_WHEEL) == FINAL_WHEEL_SHA
    output.mkdir(parents=True, exist_ok=False)
    helpers.write(output / "plan.json", {
        "protocol": "release_0161_saved_model_timing_phase3", "created_at": helpers.now(),
        "source_commit": "5780b41c7321c70e508dccf34e2f76f2503d1175",
        "adapter_sha256": sha(__file__), "original_plan_sha256": sha(ORIGINAL / "plan.json"),
        "original_runner_sha256": original["runner_sha256"],
        "original_source_sha256": original["source_sha256"],
        "order": list(ORDER), "cpu_affinity": [6], "threads": 1,
        "roles_opened": ["development"], "fit_models": False,
        "row_selection": "Same 1000 cyclic development rows as the original diagnostic",
        "original_runtime": original["public_runtime"],
        "final_package": str(FINAL_PACKAGE), "final_wheel": str(FINAL_WHEEL),
        "final_wheel_sha256": FINAL_WHEEL_SHA, "final_native_sha256": FINAL_NATIVE_SHA,
        "input_sha256": identities(original),
        "limitations": "Reused development data and unequal fitted tree counts; one-core local timing, no benchmark score or independent quality claim. Cold means first prediction after loading, not process startup. Four sequential passes expose control drift but do not eliminate shared-system noise.",
    })


def bootstrap(plan, arm):
    for name in ("CTBOOST_HIST_THREADS", "CTBOOST_NODE_HIST_THREADS", "OMP_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import psutil
    psutil.Process().cpu_affinity(plan["cpu_affinity"])
    if arm == "final":
        assert "ctboost" not in sys.modules
        spec = importlib.util.spec_from_file_location(
            "ctboost", FINAL_PACKAGE / "__init__.py", submodule_search_locations=[str(FINAL_PACKAGE)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["ctboost"] = module
        spec.loader.exec_module(module)
    import ctboost
    expected_version = "0.1.61" if arm == "final" else "0.1.60"
    expected_native = FINAL_NATIVE_SHA if arm == "final" else plan["original_runtime"]["native_sha256"]
    assert ctboost.__version__ == expected_version and sha(ctboost._core.__file__) == expected_native
    if arm == "final":
        assert Path(ctboost.__file__).resolve().parent == FINAL_PACKAGE.resolve()
        assert sha(FINAL_WHEEL) == FINAL_WHEEL_SHA
    versions = {name: importlib.metadata.version(name) for name in plan["original_runtime"]["versions"]}
    assert versions == plan["original_runtime"]["versions"], versions
    sys.path.insert(0, str(ROOT))
    return {
        "ctboost_module_version": ctboost.__version__, "ctboost_module_path": ctboost.__file__,
        "native_sha256": expected_native, "distribution_versions": versions,
        "metadata_note": "Only the final ctboost package is explicitly loaded; distribution metadata remains the baseline environment's 0.1.60. All other dependencies resolve from that unchanged environment.",
        "cpu_affinity": psutil.Process().cpu_affinity(), "threads": 1,
        "python_executable": sys.executable, "python_version": sys.version,
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_bytes": psutil.virtual_memory().total,
        "ctboost_python_sha256": {p.relative_to(Path(ctboost.__file__).parent).as_posix(): sha(p)
                                  for p in sorted(Path(ctboost.__file__).parent.rglob("*.py"))},
    }


def measure(output, index):
    helpers = helper()
    plan = read(output / "plan.json")
    assert plan["adapter_sha256"] == sha(__file__)
    assert plan["original_plan_sha256"] == sha(ORIGINAL / "plan.json")
    original = helpers.checked_plan(ORIGINAL)
    assert identities(original) == plan["input_sha256"]
    arm = plan["order"][index - 1]
    runtime = bootstrap(plan, arm)
    import numpy as np
    import psutil
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

    directory = output / f"pass-{index}-{arm}"
    directory.mkdir(exist_ok=False)
    report = {"started_at": helpers.now(), "pass": index, "arm": arm, "runtime": runtime,
              "plan_sha256": sha(output / "plan.json"), "adapter_sha256": sha(__file__), "rows": []}
    helpers.write(directory / "started.json", report)
    started_pass = time.perf_counter()
    try:
        for case in original["datasets"]:
            name = case["dataset_name"]
            role = helpers.load_role(original, case, "development")
            positions = np.arange(1000) % len(role["X"])
            batch = role["X"].iloc[positions].copy().reset_index(drop=True)
            row = {"dataset": name, "problem_type": case["problem_type"], "timed_rows": len(batch),
                   "development_rows": len(role["X"]), "input_sha256": plan["input_sha256"][name],
                   "cycle_index_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(), "models": {}}
            arms = ["ctboost_default", "catboost_ag_default"]
            if len(report["rows"]) % 2:
                arms.reverse()
            for model_arm in arms:
                source = (Path(original["scout"]) / "fits" / name / model_arm if model_arm == "ctboost_default"
                          else ORIGINAL / "catboost" / name)
                receipt = read(source / "result.json")
                expected_plan = original["scout_plan_sha256"] if model_arm == "ctboost_default" else sha(ORIGINAL / "plan.json")
                assert receipt["status"] == "ok" and not receipt.get("resource_failure")
                assert receipt["plan_sha256"] == expected_plan and receipt["model_sha256"] == sha(source / "model.pkl")
                payload = pickle.loads((source / "model.pkl").read_bytes())
                model = payload["model"]
                original_methods = {}
                if model_arm == "catboost_ag_default":
                    for method in ("predict", "predict_proba"):
                        if hasattr(model.model, method):
                            original_methods[method] = getattr(model.model, method)
                            setattr(model.model, method, functools.partial(getattr(model.model, method), thread_count=1))

                def predict():
                    frame, _ = normalize_tabarena_frame(batch, categorical_columns=payload["categorical_columns"])
                    prediction = np.asarray(model.predict(frame) if case["problem_type"] == "regression" else model.predict_proba(frame))
                    return prediction[:, 1] if case["problem_type"] == "binary" and prediction.ndim == 2 else prediction

                started = time.perf_counter()
                prediction = predict()
                cold_ms = (time.perf_counter() - started) * 1000
                assert np.isfinite(prediction).all()
                timing = helpers.measure_call(predict)
                repeated = predict()
                assert prediction.dtype == repeated.dtype and prediction.shape == repeated.shape
                assert prediction.tobytes() == repeated.tobytes()
                for repeat in (1, 2):
                    with np.load(ORIGINAL / "measurements" / f"baseline-{repeat}" / f"{name}.npz", allow_pickle=False) as reference:
                        expected = reference[model_arm]
                        assert expected.dtype == prediction.dtype and expected.shape == prediction.shape
                        assert expected.tobytes() == prediction.tobytes(), (name, model_arm)
                row["models"][model_arm] = {
                    **timing, "cold_ms": cold_ms, "rounds": receipt["rounds"],
                    "trees": int(model.get_booster()._handle.num_trees()) if model_arm == "ctboost_default" else int(model.model.tree_count_),
                    "model_sha256": receipt["model_sha256"], "prediction_dtype": str(prediction.dtype),
                    "prediction_shape": list(prediction.shape), "public_prediction_bitwise": True,
                    "prediction_values_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                    "rss_bytes_after_warmup": psutil.Process().memory_info().rss,
                }
                for method, original_method in original_methods.items():
                    setattr(model.model, method, original_method)
            report["rows"].append(row)
            helpers.write(directory / "progress.json", report)
            print(json.dumps({"pass": index, "arm": arm, "dataset": name}), flush=True)
        assert identities(original) == plan["input_sha256"]
        assert sha(sys.modules["ctboost"]._core.__file__) == runtime["native_sha256"]
        report.update(all14_complete=len(report["rows"]) == 14, finished_at=helpers.now(),
                      pass_wall_seconds=time.perf_counter() - started_pass)
        helpers.write(directory / "summary.json", report)
    except Exception:
        report["error"] = traceback.format_exc()
        helpers.write(directory / "failed.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pass-index", type=int, choices=range(1, 5))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output.resolve())
    else:
        if args.pass_index is None:
            parser.error("measure requires --pass-index")
        measure(args.output.resolve(), args.pass_index)


if __name__ == "__main__":
    main()
