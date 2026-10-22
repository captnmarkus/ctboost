"""Supervise one immutable pass of the prospectively frozen final diagnostic.

Root must first record independent completion review and exclusive CPU release
in authorization.json. This wrapper does not select models from their scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
ROOT = ARCHIVE.parents[2]
POLICY = ARCHIVE / "inference/final_runner_preparation/selection_policy_v4.json"
POLICY_HASH = "7d8e71259b2dd6cb1e47b568e6899c3612a3fd6913e1da1703ec6124444168c2"
WATCHDOG = ARCHIVE / "release_preparation/installed_candidate_02/release_validation_watchdog.py"
WATCHDOG_HASH = "00b9f195b90285c0ca7fb89ff26dcbd679568a856954b61c3f2b52a5943445f8"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["reference", "pass-1", "pass-2", "pass-3"])
    args = parser.parse_args()
    authorization_path = HERE / "authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    assert authorization["quality_work_terminal"] is True
    assert authorization["exclusive_timing_released"] is True
    assert digest(POLICY) == POLICY_HASH
    assert digest(WATCHDOG) == WATCHDOG_HASH
    for name, expected in authorization["verified_files"].items():
        assert digest(name) == expected, name
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    branch = authorization["selected_policy"]
    assert branch in ["candidate_policy", "fallback_policy"]
    selected = policy[branch]
    runner = ROOT / policy["runner"]
    assert digest(runner) == policy["runner_sha256"]
    for name, expected in policy["runtime_provenance"].items():
        assert digest(name) == expected, name
    runtime = Path(policy["runtime_path"])
    build = json.loads((runtime / "build_receipt.json").read_text(encoding="utf-8"))
    assert digest(runtime / "ctboost/_core.cp312-win_amd64.pyd") == build["native_sha256"] == policy["runtime_native_sha256"]
    for name, expected in build["python_sha256"].items():
        assert digest(runtime / "ctboost" / name) == expected, name
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    receipt_path, log_path = HERE / f"{args.stage}-execution.json", HERE / f"{args.stage}.log"
    assert not any(path.exists() for path in [receipt_path, log_path, output / f"{args.stage}.json"])
    if args.stage.startswith("pass-"):
        prior = "reference" if args.stage == "pass-1" else f"pass-{int(args.stage[-1]) - 1}"
        previous = json.loads((HERE / f"{prior}-execution.json").read_text(encoding="utf-8"))
        assert previous["status"] == "passed" and previous["cleanup_complete"] is True
        assert previous["authorization_sha256"] == digest(authorization_path)
        assert previous["result_sha256"] == digest(output / f"{prior}.json")
        assert previous["wrapper_sha256"] == digest(__file__)
    plan_path = Path(selected["accuracy_results"]) / "plan.json"
    assert digest(plan_path) == selected["plan_sha256"]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    command = [plan["runtime"]["python"], "-I", str(runner),
        "--package-root", policy["runtime_path"],
        "--dependencies", str(ROOT / ".tmp/improvements-20261016/followup/deps"),
        "--accuracy-results", selected["accuracy_results"],
        "--accuracy-artifacts", selected["accuracy_artifacts"],
        "--output", str(output), "--label", args.stage,
        "--rows", *map(str, policy["batch_rows"])]
    if args.stage == "reference":
        command += ["--check-only"]
    else:
        command += ["--reference", str(output / "reference.json"),
                    "--order", *policy["library_orders"][int(args.stage[-1]) - 1]]
    available = psutil.virtual_memory().available
    assert available >= 12 * 1024**3, "Insufficient host headroom; no process started"
    spec = importlib.util.spec_from_file_location("timing_watchdog", WATCHDOG)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    psutil.Process().cpu_affinity([0])
    environment = os.environ.copy()
    for key in ["CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        environment[key] = "1"
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = watchdog.watch(process, maximum_seconds=1800,
                                maximum_rss_bytes=8 * 1024**3)
    passed = result["exit_code"] == 0 and result["resource_failure"] is None and result["cleanup_complete"]
    receipt = {"stage": args.stage, "status": "passed" if passed else "failed",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "command": command, "available_before_bytes": available,
        "authorization_sha256": digest(authorization_path), "policy_sha256": digest(POLICY),
        "wrapper_sha256": digest(__file__), "watchdog_sha256": digest(WATCHDOG),
        "log_sha256": digest(log_path), **result}
    if passed:
        receipt["result_sha256"] = digest(output / f"{args.stage}.json")
    write(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in ["stage", "status", "wall_seconds", "peak_rss_bytes", "cleanup_complete"]}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
