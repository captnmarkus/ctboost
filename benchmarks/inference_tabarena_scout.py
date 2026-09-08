"""Real-data inference diagnostic on the 14 fixed development tasks.

Reuse the accuracy scout's train/stop/development roles and public CTBoost
models. Confirmation rows are never opened. Fit AutoGluon's default CatBoost
once; measure both libraries including normalization and model preprocessing.
This is a local latency diagnostic, not a TabArena score or admission result.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIT_CPUS = [3, 11]
PACKAGES = ("ctboost", "catboost", "autogluon.tabular", "autogluon.core", "numpy", "pandas", "scikit-learn")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bootstrap(cpus, threads):
    if threads < 1 or len(cpus) != len(set(cpus)) or not cpus:
        raise ValueError("Specify positive threads and distinct CPU IDs")
    for name in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    import psutil
    psutil.Process().cpu_affinity(cpus)
    import ctboost
    if Path(ctboost.__file__).resolve().is_relative_to(ROOT / "ctboost"):
        raise ValueError("Use an installed wheel with isolated Python -I")
    if ctboost.__version__ != "0.1.60":
        raise ValueError("This diagnostic compares public 0.1.60 and its inference-only candidate")
    sys.path.insert(0, str(ROOT))
    return {"versions": {p: importlib.metadata.version(p) for p in PACKAGES},
            "native_sha256": digest(ctboost._core.__file__), "ctboost_path": ctboost.__file__,
            "cpu_affinity": cpus, "threads": threads}


def checked_plan(output):
    plan = read(output / "plan.json")
    if plan["runner_sha256"] != digest(__file__):
        raise ValueError("Latency runner changed after preparation")
    if digest(Path(plan["scout"]) / "plan.json") != plan["scout_plan_sha256"]:
        raise ValueError("Accuracy scout plan changed")
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
    return plan


def load_role(plan, case, role):
    if role not in ("train", "stop", "development"):
        raise ValueError("Confirmation is outside this diagnostic")
    path = Path(plan["scout"]) / "data" / case["dataset_name"] / f"{role}.pkl"
    if digest(path) != case["roles"][role]["sha256"]:
        raise ValueError(f"Changed {role} data")
    with path.open("rb") as stream:
        return pickle.load(stream)


def measure_call(call):
    for _ in range(3):
        call()
    started = time.perf_counter()
    for _ in range(10):
        call()
    estimate = max((time.perf_counter() - started) / 10, 1e-6)
    count = min(1000, max(5, int(.1 / estimate)))
    samples = []
    for _ in range(7):
        started = time.perf_counter()
        for _ in range(count):
            call()
        samples.append((time.perf_counter() - started) * 1000 / count)
    return {"median_ms": statistics.median(samples), "block_ms": samples, "calls_per_block": count}


def fit(output, dataset):
    runtime = bootstrap(FIT_CPUS, 2)
    from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

    plan = checked_plan(output)
    if runtime["native_sha256"] != plan["public_runtime"]["native_sha256"]:
        raise ValueError("Prepare with the public baseline wheel")
    case = next(c for c in plan["datasets"] if c["dataset_name"] == dataset)
    directory = output / "catboost" / dataset
    directory.mkdir(parents=True, exist_ok=False)
    record = {"dataset": dataset, "status": "failed", "started_at": now(),
              "plan_sha256": digest(output / "plan.json"), "runtime": runtime}
    write(directory / "started.json", record)
    try:
        train, stop = load_role(plan, case, "train"), load_role(plan, case, "stop")
        X, categoricals = normalize_tabarena_frame(train["X"])
        X_stop, _ = normalize_tabarena_frame(stop["X"], categorical_columns=categoricals)
        model = CatBoostModel(path=str(directory / "ag"), name="CatBoost", problem_type=case["problem_type"],
                              eval_metric=case["eval_metric"], hyperparameters={"random_seed": 47})
        started = time.perf_counter()
        model.fit(X=X, y=train["y"], X_val=X_stop, y_val=stop["y"],
                  time_limit=300, num_cpus=2, num_gpus=0, verbosity=0)
        record.update(fit_seconds=time.perf_counter() - started, rounds=int(model.model.tree_count_),
                      parameters=model.params, trained_parameters=model.params_trained)
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump({"model": model, "categorical_columns": categoricals, "arm": "catboost_ag_default"}, stream, protocol=5)
        record.update(status="ok", model_sha256=digest(directory / "model.pkl"))
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = now()
    write(directory / "result.json", record)


def prepare(args):
    import psutil
    runtime = bootstrap(FIT_CPUS, 2)
    scout = read(args.scout / "plan.json")
    if len(scout["datasets"]) != 14 or runtime["native_sha256"] != scout["ctboost_native_sha256"]:
        raise ValueError("Require all 14 predeclared tasks and the same public runtime")
    args.output.mkdir(parents=True, exist_ok=True)
    # One controller owns this directory; OS releases the lock on termination.
    with (args.output / "controller.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (args.output / "plan.json").exists():
            dependencies = ["benchmarks/tabarena/ctboost_model.py", "benchmarks/tabarena/learning_options.py"]
            plan = {"protocol_id": "real_data_inference_scout_20260908_v1", "created_at": now(),
                    "runner_sha256": digest(__file__), "scout": str(args.scout),
                    "scout_plan_sha256": digest(args.scout / "plan.json"), "public_runtime": runtime,
                    "source_sha256": {p: digest(ROOT / p) for p in dependencies}, "datasets": scout["datasets"],
                    "fit_resources": {"seconds": 300, "hard_wall_seconds": 390, "rss_bytes": 8 * 1024**3,
                                      "workers": 1, "cpu_affinity": FIT_CPUS, "threads": 2},
                    "catboost": "AutoGluon defaults: iterations10000, learning_rate.05, adaptive early stopping; seed47.",
                    "measurement": "1000 rows by deterministic cycling/truncating the development role. Normalization and model preprocessing are timed; loading and row cycling are excluded. Cold means first predict after loading. No bagging. Warm blocks retained. Explicit CatBoost thread_count bound in memory because AG does not forward it.",
                    "limitations": "Reused development tasks, capped training panel; unequal actual tree counts/default learners; active background HPO contends for shared resources. No score, Elo, independent holdout or benchmark promotion claim."}
            write(args.output / "plan.json", plan)
        plan = checked_plan(args.output)
        if plan["scout"] != str(args.scout) or runtime != plan["public_runtime"]:
            raise ValueError("Prepare resume requires the same data and public runtime")
        write(args.output / "controller.json", {"pid": os.getpid(), "process_created": psutil.Process().create_time(),
                                               "started_at": now(), "plan_sha256": digest(args.output / "plan.json")})
        for case in plan["datasets"]:
            dataset = case["dataset_name"]
            directory = args.output / "catboost" / dataset
            if directory.exists():
                continue  # Never repeat a started attempt automatically.
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(args.output), "--dataset", dataset]
            log_path = args.output / "logs" / f"{dataset}.log"
            log_path.parent.mkdir(exist_ok=True)
            started, peak, reason = time.monotonic(), 0, None
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                owner = psutil.Process(child.pid)
                while child.poll() is None:
                    try:
                        peak = max(peak, sum(p.memory_info().rss for p in [owner] + owner.children(recursive=True)))
                    except psutil.NoSuchProcess:
                        pass
                    if peak > plan["fit_resources"]["rss_bytes"]:
                        reason = "memory_limit"
                    elif time.monotonic() - started > plan["fit_resources"]["hard_wall_seconds"]:
                        reason = "hard_timeout"
                    if reason:
                        try:
                            descendants = owner.children(recursive=True)
                        except psutil.NoSuchProcess:
                            descendants = []
                        for process in descendants + [owner]:
                            try:
                                process.kill()  # psutil checks PID reuse.
                            except psutil.NoSuchProcess:
                                pass
                        child.wait(timeout=15)
                        break
                    time.sleep(.25)
            record_path = directory / "result.json"
            record = read(record_path) if record_path.exists() else {
                "dataset": dataset, "status": "failed", "plan_sha256": digest(args.output / "plan.json")}
            record.update(peak_rss_bytes=peak, resource_failure=reason, exit_code=child.returncode)
            if reason or child.returncode != 0:
                record["status"] = "failed"
            write(record_path, record)
            print(json.dumps({"dataset": dataset, "status": record["status"], "rounds": record.get("rounds")}), flush=True)
        records = [read(args.output / "catboost" / c["dataset_name"] / "result.json") for c in plan["datasets"]
                   if (args.output / "catboost" / c["dataset_name"] / "result.json").exists()]
        write(args.output / "preparation.json", {"finished_at": now(), "expected": 14,
                                               "successful": sum(r["status"] == "ok" for r in records), "records": records})


def measure(args):
    runtime = bootstrap(args.cpus, args.threads)
    import numpy as np
    import psutil
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

    plan = checked_plan(args.output)
    for name, version in plan["public_runtime"]["versions"].items():
        if runtime["versions"][name] != version:
            raise ValueError(f"Changed dependency: {name}")
    if args.label == "baseline" and runtime["native_sha256"] != plan["public_runtime"]["native_sha256"]:
        raise ValueError("Baseline must use the original public wheel")
    directory = args.output / "measurements" / f"{args.label}-{args.repeat}"
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for case in plan["datasets"]:
        dataset = case["dataset_name"]
        role = load_role(plan, case, "development")
        positions = np.arange(1000) % len(role["X"])
        batch = role["X"].iloc[positions].copy().reset_index(drop=True)
        row = {"dataset": dataset, "problem_type": case["problem_type"], "development_rows": len(role["X"]),
               "timed_rows": len(batch), "cycle_index_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
               "train_rows": case["roles"]["train"]["rows"], "stop_rows": case["roles"]["stop"]["rows"],
               "models": {}}
        # Alternate library order across tasks, fixed across wheel measurements.
        arms = ["ctboost_default", "catboost_ag_default"]
        if len(results) % 2:
            arms.reverse()
        predictions = {}
        for arm in arms:
            source = (Path(plan["scout"]) / "fits" / dataset / arm if arm == "ctboost_default"
                      else args.output / "catboost" / dataset)
            receipt = read(source / "result.json")
            expected_plan = plan["scout_plan_sha256"] if arm == "ctboost_default" else digest(args.output / "plan.json")
            if receipt["status"] != "ok" or receipt.get("resource_failure") or receipt["plan_sha256"] != expected_plan:
                raise ValueError(f"Incomplete or invalid fitted model: {dataset}/{arm}")
            if receipt["model_sha256"] != digest(source / "model.pkl"):
                raise ValueError("Saved model changed")
            with (source / "model.pkl").open("rb") as stream:
                payload = pickle.load(stream)
            model = payload["model"]
            original_methods = {}
            if arm == "catboost_ag_default":
                for method in ("predict", "predict_proba"):
                    if hasattr(model.model, method):
                        original_methods[method] = getattr(model.model, method)
                        setattr(model.model, method, functools.partial(getattr(model.model, method), thread_count=args.threads))

            def predict():
                frame, _ = normalize_tabarena_frame(batch, categorical_columns=payload["categorical_columns"])
                prediction = np.asarray(model.predict(frame) if case["problem_type"] == "regression" else model.predict_proba(frame))
                return prediction[:, 1] if case["problem_type"] == "binary" and prediction.ndim == 2 else prediction

            started = time.perf_counter()
            prediction = predict()
            cold_ms = (time.perf_counter() - started) * 1000
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite development prediction")
            predictions[arm] = prediction
            timing = measure_call(predict)
            np.testing.assert_array_equal(predict(), prediction)
            trees = (int(model.get_booster()._handle.num_trees()) if arm == "ctboost_default"
                     else int(model.model.tree_count_))
            row["models"][arm] = {**timing, "cold_ms": cold_ms, "rounds": receipt["rounds"], "trees": trees,
                                    "model_sha256": receipt["model_sha256"], "rss_bytes_after_warmup": psutil.Process().memory_info().rss}
            for method, original in original_methods.items():
                setattr(model.model, method, original)
        reference_path = args.output / "measurements" / f"baseline-{args.repeat}" / f"{dataset}.npz"
        if args.label != "baseline":
            with np.load(reference_path, allow_pickle=False) as reference:
                for arm, prediction in predictions.items():
                    np.testing.assert_array_equal(prediction, reference[arm])
            row["public_prediction_identity"] = True
        np.savez_compressed(directory / f"{dataset}.npz", **predictions)
        row["prediction_sha256"] = digest(directory / f"{dataset}.npz")
        results.append(row)
        write(directory / "progress.json", {"rows": results})
        print(json.dumps({"measured": dataset, "label": args.label}), flush=True)
    write(directory / "summary.json", {"created_at": now(), "runtime": runtime, "plan_sha256": digest(args.output / "plan.json"),
                                       "runner_sha256": digest(__file__), "repeat": args.repeat,
                                       "all14_complete": len(results) == 14, "rows": results,
                                       "median_ctboost_to_catboost_ratio": statistics.median(r["models"]["ctboost_default"]["median_ms"] / r["models"]["catboost_ag_default"]["median_ms"] for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "fit", "measure"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scout", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--label", choices=("baseline", "candidate"))
    parser.add_argument("--cpus", type=lambda text: [int(v) for v in text.split(",")], default=[6])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == "prepare":
        if args.scout is None:
            parser.error("prepare requires --scout")
        args.scout = args.scout.resolve()
        prepare(args)
    elif args.mode == "fit":
        if not args.dataset:
            parser.error("fit requires --dataset")
        fit(args.output, args.dataset)
    else:
        if args.label is None:
            parser.error("measure requires --label")
        if args.repeat < 1:
            parser.error("repeat must be positive")
        measure(args)


if __name__ == "__main__":
    main()
