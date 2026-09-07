"""Calibrate local CPU fit concurrency using synthetic data only.

Run with the isolated benchmark environment's ``python -I`` and this file's
absolute path. Each layout runs eight identical fits twice, in reverse order
on the second pass. Imports/data generation and one warmup fit per worker precede
a synchronized fit barrier.
This measures local scheduling throughput, not TabArena scores or timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024 ** 3
LAYOUTS = ((1, 16), (2, 8), (4, 4), (8, 2), (2, 4), (4, 2))
THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
FIT_PARAMS = {
    "iterations": 50, "max_depth": 4, "max_bins": 64,
    "learning_rate": 0.1, "random_seed": 159, "task_type": "CPU",
    "verbose": False,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_affinity(allowed, workers, threads):
    """Allocate disjoint logical CPUs; reject oversubscription, never truncate."""
    allowed = sorted(set(allowed))
    if workers < 1 or threads < 1 or workers * threads > len(allowed):
        raise ValueError("Layout exceeds the available logical CPU affinity")
    return [allowed[i * threads:(i + 1) * threads] for i in range(workers)]


def worker_environment(threads):
    environment = os.environ.copy()
    environment.update({key: "1" for key in THREAD_ENV})
    environment["CTBOOST_HIST_THREADS"] = str(threads)
    environment["PYTHONHASHSEED"] = "159"
    return environment


def select_layout(records, memory_budget_bytes, repeats=2):
    """Select sustained fit throughput among complete, memory-safe layouts."""
    candidates = []
    names = sorted({record["layout"] for record in records})
    for name in names:
        rows = [record for record in records if record["layout"] == name]
        if len(rows) != repeats or any(row["status"] != "complete" for row in rows):
            continue
        if any(row["peak_worker_tree_rss_bytes"] > memory_budget_bytes for row in rows):
            continue
        elapsed = sum(row["fit_wall_seconds"] for row in rows)
        row = rows[0]
        candidates.append({
            "layout": name, "workers": row["workers"], "threads_per_worker": row["threads_per_worker"],
            "affinity": row["affinity"],
            "fits_per_second": sum(item["completed_fits"] for item in rows) / elapsed,
            "peak_worker_tree_rss_bytes": max(item["peak_worker_tree_rss_bytes"] for item in rows),
            "memory_budget_bytes": memory_budget_bytes,
            "memory_per_worker_bytes": memory_budget_bytes // row["workers"],
        })
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["fits_per_second"], -row["workers"]))


def run_worker(spec_path):
    import psutil

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    root = Path(spec["root"])
    identity = spec["identity"]
    psutil.Process().cpu_affinity(spec["affinity"])
    # Heavy imports must occur after affinity and thread environment are applied.
    import numpy as np
    import ctboost
    from ctboost import _core

    if ctboost.__version__ != "0.1.59":
        raise RuntimeError(f"Expected public CTBoost 0.1.59, found {ctboost.__version__}")
    rng = np.random.default_rng(159)
    features = rng.normal(size=(4000, 32)).astype(np.float32)
    target = (3 * features[:, 0] - 2 * features[:, 1]
              + features[:, 2] * features[:, 3] + rng.normal(size=4000)).astype(np.float32)
    data_hash = hashlib.sha256(features.tobytes() + target.tobytes()).hexdigest()
    # Load native runtime/lazy dispatch before timing; otherwise the first layout
    # can absorb one-time cold initialization that later layouts never pay.
    warmup = ctboost.CTBoostRegressor(**FIT_PARAMS)
    warmup.fit(features, target)
    warmup.predict(features[:64])
    del warmup
    write_json(root / f"{identity}.ready.json", {"pid": os.getpid()})
    while not (root / "go").exists():
        if time.time() >= spec["deadline_epoch"]:
            raise TimeoutError("Calibration barrier deadline expired")
        time.sleep(0.005)
    started = time.perf_counter()
    fits = []
    for _ in range(spec["fits"]):
        fit_started = time.perf_counter()
        model = ctboost.CTBoostRegressor(**FIT_PARAMS)
        model.fit(features, target)
        predictions = np.asarray(model.predict(features[:64]), dtype=np.float64)
        if not np.isfinite(predictions).all():
            raise RuntimeError("Synthetic fit produced non-finite predictions")
        fits.append({"seconds": time.perf_counter() - fit_started,
                     "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest()})
    finished = time.perf_counter()
    write_json(root / f"{identity}.result.json", {
        "elapsed_seconds": finished - started, "finished_perf_counter": finished, "fits": fits,
        "affinity": psutil.Process().cpu_affinity(), "histogram_threads": os.environ["CTBOOST_HIST_THREADS"],
        "provenance": {"ctboost_version": ctboost.__version__, "ctboost_module": ctboost.__file__,
                       "native_module": _core.__file__, "native_sha256": file_hash(_core.__file__),
                       "ctboost_build_info": ctboost.build_info(), "numpy_version": np.__version__,
                       "data_sha256": data_hash, "python_executable": sys.executable},
    })
    return 0


def run_layout(root, workers, threads, affinity, repeat, deadline, reserve_bytes):
    import psutil

    root.mkdir(parents=True, exist_ok=False)
    partitions = partition_affinity(affinity, workers, threads)
    record = {"layout": f"{workers}x{threads}", "workers": workers, "threads_per_worker": threads,
              "affinity": partitions, "repeat": repeat, "status": "initializing",
              "peak_worker_tree_rss_bytes": 0, "minimum_system_available_bytes": psutil.virtual_memory().available,
              "completed_fits": 0, "logs": []}
    processes, streams = [], []
    fit_started = None
    try:
        for index in range(workers):
            identity = f"worker-{index}"
            spec_path = root / f"{identity}.spec.json"
            write_json(spec_path, {"root": str(root), "identity": identity, "affinity": partitions[index],
                                   "fits": 8 // workers, "deadline_epoch": time.time() + max(0, deadline - time.monotonic())})
            log_path = root / f"{identity}.log"
            stream = log_path.open("w", encoding="utf-8")
            streams.append(stream)
            record["logs"].append(str(log_path))
            processes.append(subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(spec_path)],
                stdout=stream, stderr=subprocess.STDOUT, env=worker_environment(threads),
                cwd=str(root),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ))
        while True:
            available = psutil.virtual_memory().available
            record["minimum_system_available_bytes"] = min(record["minimum_system_available_bytes"], available)
            rss = 0
            for process in processes:
                try:
                    parent = psutil.Process(process.pid)
                    rss += parent.memory_info().rss
                    for child in parent.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass
            record["peak_worker_tree_rss_bytes"] = max(record["peak_worker_tree_rss_bytes"], rss)
            if available < reserve_bytes:
                record["status"] = "memory_guard"
                break
            if time.monotonic() >= deadline:
                record["status"] = "time_guard"
                break
            if any(process.poll() not in (None, 0) for process in processes):
                record["status"] = "worker_failed"
                break
            if fit_started is None and all((root / f"worker-{i}.ready.json").exists() for i in range(workers)):
                fit_started = time.perf_counter()
                (root / "go").touch()
            if all((root / f"worker-{i}.result.json").exists() for i in range(workers)):
                outputs = [json.loads((root / f"worker-{i}.result.json").read_text(encoding="utf-8"))
                           for i in range(workers)]
                # perf_counter uses a shared system monotonic clock on supported
                # CPython platforms. Exclude result hashing/JSON and monitor polling.
                record["fit_wall_seconds"] = max(output["finished_perf_counter"] for output in outputs) - fit_started
                record["completed_fits"] = sum(len(output["fits"]) for output in outputs)
                record["fit_seconds"] = [fit["seconds"] for output in outputs for fit in output["fits"]]
                record["prediction_sha256"] = sorted({fit["prediction_sha256"] for output in outputs for fit in output["fits"]})
                record["provenance"] = outputs[0]["provenance"]
                record["status"] = "complete"
                break
            time.sleep(0.05)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in streams:
            stream.close()
    write_json(root / "measurement.json", record)
    return record


def calibrate(output, max_seconds=285, reserve_gib=3):
    import psutil

    output = Path(output).resolve()
    allowed = psutil.Process().cpu_affinity()
    memory = psutil.virtual_memory()
    reserve_bytes = int(reserve_gib * GIB)
    budget = max(0, memory.available - reserve_bytes)
    started = time.monotonic()
    deadline = started + max_seconds
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    work_root = output.parent / "resource-calibration" / run_id
    layouts = [(workers, threads) for workers, threads in LAYOUTS if workers * threads <= len(allowed)]
    report = {
        "schema_version": 1, "synthetic_only": True, "status": "running", "run_id": run_id,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": psutil.cpu_count(logical=True), "physical_cores": psutil.cpu_count(logical=False),
                     "allowed_logical_cpu_affinity": allowed, "memory_total_bytes": memory.total,
                     "memory_initial_available_bytes": memory.available},
        "protocol": {"rows": 4000, "features": 32, "data_seed": 159, "fit_params": FIT_PARAMS,
                     "fits_per_layout_per_repeat": 8, "repeats": 2, "layout_order": [list(layouts), list(reversed(layouts))],
                     "unmeasured_warmup_fits_per_worker": 1,
                     "reserve_bytes": reserve_bytes, "max_seconds": max_seconds,
                     "blas_omp_threads": 1, "histogram_thread_environment": "CTBOOST_HIST_THREADS",
                     "memory_measurement": "Sum of sampled worker process-tree RSS, 50ms interval; transient peaks may be missed",
                     "timing": "Barrier release through final worker fit completion; imports, warmup, data generation, provenance hashing and monitor polling excluded",
                     "selection_limit": "Synthetic regression throughput only; real task memory and multiclass scaling can differ"},
        "harness_sha256": file_hash(__file__), "python": sys.version, "records": [], "selected": None,
    }
    write_json(output, report)
    for repeat, ordered in enumerate((layouts, list(reversed(layouts)))):
        for workers, threads in ordered:
            if time.monotonic() >= deadline or psutil.virtual_memory().available < reserve_bytes:
                report["status"] = "guard_stopped"
                break
            print(f"calibrating repeat={repeat} layout={workers}x{threads}", flush=True)
            record = run_layout(work_root / f"r{repeat}-{workers}x{threads}", workers, threads, allowed,
                                repeat, deadline, reserve_bytes)
            report["records"].append(record)
            print(f"status={record['status']} fit_wall_seconds={record.get('fit_wall_seconds')}", flush=True)
            write_json(output, report)
    report["selected"] = select_layout(report["records"], budget)
    report["elapsed_seconds"] = time.monotonic() - started
    if report["status"] == "running":
        report["status"] = "complete" if len(report["records"]) == 2 * len(layouts) and all(
            row["status"] == "complete" for row in report["records"]) else "incomplete"
    write_json(output, report)
    print(json.dumps({"status": report["status"], "selected": report["selected"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    return 0 if report["selected"] is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--max-seconds", type=float, default=285)
    parser.add_argument("--reserve-gib", type=float, default=3)
    args = parser.parse_args()
    if args.worker is not None:
        return run_worker(args.worker)
    if args.output is None:
        parser.error("--output is required")
    if not 0 < args.max_seconds <= 285 or args.reserve_gib < 3:
        parser.error("max-seconds must be in (0,285], reserve-gib must be at least 3")
    return calibrate(args.output, args.max_seconds, args.reserve_gib)


if __name__ == "__main__":
    raise SystemExit(main())
