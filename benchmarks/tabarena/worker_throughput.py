"""Separate synthetic ABCCBA throughput screen; never changes official run state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024**3
HERE = Path(__file__).resolve().parent
ORDER = ["A", "B", "C", "C", "B", "A"]


def helper(name):
    spec = importlib.util.spec_from_file_location(
        name, HERE / f"worker_throughput_{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def source_hashes():
    return {
        path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in (
            Path(__file__),
            HERE / "worker_throughput_workload.py",
            HERE / "worker_throughput_topology.py",
        )
    }


def layouts(topology):
    siblings = [
        [p["logical_cpu"] for p in core["processors"]] for core in topology["cores"]
    ]
    if (
        topology["allowed_logical_cpus"] != list(range(16))
        or len(siblings) != 8
        or any(len(pair) != 2 for pair in siblings)
    ):
        raise ValueError(
            "This screen requires the declared eight-core, sixteen-logical-CPU topology"
        )
    slots = {
        "A": [[i, i + 8] for i in range(8)],
        "B": [[i] for i in range(16)],
        "C": siblings,
    }
    topology_module = helper("topology")
    return {
        name: {
            "slots": pairs,
            "histogram_threads": len(pairs[0]),
            "core_membership": topology_module.validate_layout(topology, pairs),
        }
        for name, pairs in slots.items()
    }


def make_spec(count):
    import psutil

    workload = helper("workload")
    topology = helper("topology").collect_topology()
    return {
        "schema_version": 1,
        "order": ORDER,
        "jobs": workload.make_jobs(count),
        "workload": workload.specification(),
        "runtime": workload.runtime_provenance(),
        "topology": topology,
        "layouts": layouts(topology),
        "source_sha256": source_hashes(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "python": sys.executable,
        "guards": {
            "startup_seconds": 120,
            "arm_seconds": 1200,
            "physical_free_bytes": 4 * GIB,
            "sampling_seconds": 0.5,
        },
        "criteria": {
            "steady_ratio": 1.10,
            "total_ratio": 1.00,
            "rtol": 1e-12,
            "atol": 1e-12,
            "prefer_C_within_B": 0.05,
            "automatic_policy_change": False,
        },
    }


def require_drained(official):
    if official is None:
        return
    progress_path = official / "progress.json"
    progress = read(progress_path)
    if (
        not (official / "PAUSE").is_file()
        or time.time() - progress_path.stat().st_mtime > 30
        or progress.get("paused") is not True
        or progress.get("active") != []
        or progress.get("counts", {}).get("running") != 0
        or progress.get("counts", {}).get("launching", 0) != 0
    ):
        raise RuntimeError(
            "Official run must have a fresh paused, fully drained progress receipt"
        )


def claim(arm, jobs, slot):
    for job in jobs:
        try:
            with (arm / "claims" / f"{job['job_id']:03d}.json").open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump({"slot": slot, "pid": os.getpid(), "job": job}, stream)
            return job
        except FileExistsError:
            continue
    return None


def worker_main(arm, slot):
    import psutil

    plan = read(arm.parent / "spec.json")
    if source_hashes() != plan["source_sha256"]:
        raise RuntimeError("Screen source changed after specification freeze")
    layout = plan["layouts"][read(arm / "arm.json")["layout"]]
    affinity, threads = layout["slots"][slot], layout["histogram_threads"]
    psutil.Process().cpu_affinity(affinity)
    workload = helper("workload")
    workload.configure_threads(threads)
    runtime = workload.runtime_provenance()
    if runtime != plan["runtime"] or psutil.Process().cpu_affinity() != affinity:
        raise RuntimeError(
            "Worker runtime or affinity differs from frozen specification"
        )
    try:
        started = time.perf_counter()
        prepared = workload.prepare_profiles()
        preparation_seconds = time.perf_counter() - started
        warmup = workload.fit_job(
            workload.warmup_job(), prepared, histogram_threads=threads
        )
        write(
            arm / "ready" / f"{slot:02d}.json",
            {
                "slot": slot,
                "runtime": runtime,
                "affinity": affinity,
                "preparation_seconds": preparation_seconds,
                "warmup": warmup,
            },
        )
        while not (arm / "START").exists():
            time.sleep(0.02)
        while (job := claim(arm, plan["jobs"], slot)) is not None:
            result = workload.fit_job(job, prepared, histogram_threads=threads)
            result.update(slot=slot, finished_monotonic=time.perf_counter())
            write(arm / "results" / f"{job['job_id']:03d}.json", result)
    except BaseException as exc:
        write(
            arm / "errors" / f"{slot:02d}.json",
            {"slot": slot, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


def owns(process, identity):
    import psutil

    try:
        command = process.cmdline()
        return (
            process.create_time() == identity["create_time"]
            and command[1:] == identity["command"][1:]
            and os.path.normcase(os.path.abspath(command[0]))
            == os.path.normcase(os.path.abspath(identity["command"][0]))
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, IndexError):
        return False


def stop_owned(jobs):
    import psutil

    for job in jobs:
        if job["process"].poll() is not None:
            continue
        try:
            process = psutil.Process(job["process"].pid)
            if not owns(process, job["identity"]):
                continue
            process.suspend()
            for child in reversed(process.children(recursive=True)):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            job["process"].wait(timeout=10)
        except psutil.NoSuchProcess:
            pass


def run_arm(output, index, plan, official):
    import psutil

    require_drained(official)
    arm = output / f"{index:02d}-{ORDER[index]}"
    arm.mkdir()
    for directory in ("claims", "ready", "results", "errors", "logs"):
        (arm / directory).mkdir()
    record = {
        "index": index,
        "layout": ORDER[index],
        "status": "starting",
        "workers": [],
        "peak_tree_rss_bytes": 0,
        "minimum_system_free_bytes": psutil.virtual_memory().available,
    }
    write(arm / "arm.json", record)
    started, barrier, jobs = time.perf_counter(), None, []
    try:
        for slot in range(len(plan["layouts"][ORDER[index]]["slots"])):
            command = [
                sys.executable,
                "-I",
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                str(arm),
                "--slot",
                str(slot),
            ]
            with (arm / "logs" / f"{slot:02d}.log").open("wb") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            identity = {
                "pid": process.pid,
                "create_time": psutil.Process(process.pid).create_time(),
                "command": command,
            }
            jobs.append({"process": process, "identity": identity, "peak_rss_bytes": 0})
            record["workers"].append(identity)
        write(arm / "arm.json", record)
        while True:
            require_drained(official)
            free = psutil.virtual_memory().available
            record["minimum_system_free_bytes"] = min(
                free, record["minimum_system_free_bytes"]
            )
            total_rss = 0
            for job in jobs:
                if job["process"].poll() is None:
                    try:
                        process = psutil.Process(job["process"].pid)
                        if not owns(process, job["identity"]):
                            if barrier is not None and job["process"].poll() == 0:
                                continue
                            raise RuntimeError("Test process identity changed")
                        rss = 0
                        for member in [process, *process.children(recursive=True)]:
                            try:
                                rss += member.memory_info().rss
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        if barrier is not None and job["process"].poll() == 0:
                            continue
                        raise RuntimeError(
                            "Test process disappeared before successful completion"
                        ) from None
                    total_rss += rss
                    job["peak_rss_bytes"] = max(rss, job["peak_rss_bytes"])
                elif job["process"].returncode != 0 or barrier is None:
                    raise RuntimeError(
                        "Throughput worker failed; preserve its log and receipts"
                    )
            record["peak_tree_rss_bytes"] = max(
                total_rss, record["peak_tree_rss_bytes"]
            )
            if free < plan["guards"]["physical_free_bytes"]:
                raise RuntimeError("Physical free memory fell below four GiB")
            now = time.perf_counter()
            if (
                now - (barrier if barrier is not None else started)
                > plan["guards"]["arm_seconds" if barrier else "startup_seconds"]
            ):
                raise TimeoutError("Throughput arm exceeded its frozen time limit")
            if barrier is None and len(list((arm / "ready").glob("*.json"))) == len(
                jobs
            ):
                barrier = time.perf_counter()
                write(arm / "START", {"monotonic": barrier})
            if barrier is not None and all(job["process"].poll() == 0 for job in jobs):
                break
            time.sleep(plan["guards"]["sampling_seconds"])
        results = [read(path) for path in sorted((arm / "results").glob("*.json"))]
        if sorted(row["job_id"] for row in results) != [
            row["job_id"] for row in plan["jobs"]
        ]:
            raise RuntimeError("Arm did not finish exactly the frozen jobs")
        steady = max(row["finished_monotonic"] for row in results) - barrier
        total = time.perf_counter() - started
        record.update(
            status="complete",
            steady_seconds=steady,
            total_seconds=total,
            steady_fits_per_hour=len(results) * 3600 / steady,
            total_fits_per_hour=len(results) * 3600 / total,
        )
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_owned(jobs)
        record["worker_peak_rss_bytes"] = [job["peak_rss_bytes"] for job in jobs]
        write(arm / "arm.json", record)
    return record


def summarize(output, plan, arms):
    import numpy as np

    expected = {job["job_id"]: job for job in plan["jobs"]}
    reference, bitwise, allclose, maximum = {}, True, True, 0.0
    for arm in arms:
        rows = [
            read(path)
            for path in sorted(
                (output / f"{arm['index']:02d}-{arm['layout']}" / "results").glob(
                    "*.json"
                )
            )
        ]
        if len(rows) != len(expected) or {row["job_id"] for row in rows} != set(
            expected
        ):
            raise ValueError("Incomplete or duplicated job evidence")
        for row in rows:
            values = np.asarray(row["prediction_values"], dtype=np.float64)
            if (
                any(row[key] != value for key, value in expected[row["job_id"]].items())
                or row["iterations_trained"] != 128
                or row["status"] != "complete"
                or row["is_warmup"] is not False
                or row["histogram_threads"]
                != plan["layouts"][arm["layout"]]["histogram_threads"]
                or row["histogram_threads_env"]
                != str(plan["layouts"][arm["layout"]]["histogram_threads"])
                or row["spec_sha256"] != plan["runtime"]["spec_sha256"]
                or not np.isfinite(values).all()
                or values.shape[0] != 256
                or list(values.shape) != row["prediction_shape"]
                or hashlib.sha256(values.tobytes()).hexdigest()
                != row["prediction_sha256"]
            ):
                raise ValueError("Synthetic workload or prediction evidence differs")
            baseline = reference.setdefault(row["job_id"], row)
            if (
                row["data_sha256"] != baseline["data_sha256"]
                or row["prediction_shape"] != baseline["prediction_shape"]
            ):
                raise ValueError("Synthetic job data or prediction shape changed")
            baseline_values = np.asarray(baseline["prediction_values"])
            bitwise &= row["prediction_sha256"] == baseline["prediction_sha256"]
            allclose &= bool(
                np.allclose(values, baseline_values, rtol=1e-12, atol=1e-12)
            )
            maximum = max(maximum, float(np.max(np.abs(values - baseline_values))))
    comparison = {}
    memory_ok = all(arm["minimum_system_free_bytes"] >= 4 * GIB for arm in arms)
    for candidate, indices in {"B": (1, 4), "C": (2, 3)}.items():
        steady = [
            arms[i]["steady_fits_per_hour"] / arms[a]["steady_fits_per_hour"]
            for i, a in zip(indices, (0, 5))
        ]
        total = [
            arms[i]["total_fits_per_hour"] / arms[a]["total_fits_per_hour"]
            for i, a in zip(indices, (0, 5))
        ]
        comparison[candidate] = {
            "steady_ratios": steady,
            "total_ratios": total,
            "passes": allclose
            and memory_ok
            and min(steady) >= 1.10
            and min(total) >= 1.00,
            "geometric_mean_steady_fits_per_hour": math.sqrt(
                arms[indices[0]]["steady_fits_per_hour"]
                * arms[indices[1]]["steady_fits_per_hour"]
            ),
        }
    selected = "A"
    if comparison["B"]["passes"]:
        selected = "B"
    if comparison["C"]["passes"] and (
        selected == "A"
        or comparison["C"]["geometric_mean_steady_fits_per_hour"]
        >= 0.95 * comparison["B"]["geometric_mean_steady_fits_per_hour"]
    ):
        selected = "C"
    return {
        "spec_sha256": digest(plan),
        "arms": arms,
        "comparison": comparison,
        "prediction_allclose": allclose,
        "prediction_bitwise_equal": bitwise,
        "maximum_prediction_difference": maximum,
        "selected_layout": selected,
        "automatic_policy_change": False,
        "scope": "synthetic throughput only; no official scores",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int, default=32, choices=(32, 64))
    parser.add_argument("--official-local", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--slot", type=int)
    args = parser.parse_args()
    if args.worker:
        worker_main(args.worker, args.slot)
        return
    if args.output is None or not sys.flags.isolated:
        parser.error("Use python -I and --output NEWDIR")
    plan = make_spec(args.jobs)
    if args.prepare_only:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "spec.json", plan)
        return
    if args.spec is None or read(args.spec) != plan:
        parser.error(
            "Execution requires an unchanged frozen --spec from --prepare-only"
        )
    require_drained(args.official_local)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "STARTED.json").open("x", encoding="utf-8") as stream:
        json.dump({"spec_sha256": digest(plan)}, stream)
    write(args.output / "spec.json", plan)
    try:
        arms = [
            run_arm(args.output, index, plan, args.official_local) for index in range(6)
        ]
        write(args.output / "summary.json", summarize(args.output, plan, arms))
    except BaseException as exc:
        write(
            args.output / "failure.json",
            {"error": f"{type(exc).__name__}: {exc}", "selected_layout": "A"},
        )
        raise


if __name__ == "__main__":
    main()
