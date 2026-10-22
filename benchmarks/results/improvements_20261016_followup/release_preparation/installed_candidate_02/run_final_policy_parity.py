"""Check the tested release wheel against final saved models after timing ends."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import psutil

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
ROOT = ARCHIVE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    timing = ARCHIVE / "inference/final_execution_20261022"
    final_pass = read(timing / "pass-3-execution.json")
    assert final_pass["status"] == "passed" and final_pass["cleanup_complete"]
    assert final_pass["result_sha256"] == digest(timing / "results/pass-3.json")
    plan_path = HERE / "final_policy_parity_plan_20261022.json"
    assert digest(plan_path) == "01112b7d25da76e5dffc2a0fb69ef40fe0e6758f8663dfa57317393590b70096"
    plan = read(plan_path)
    command = plan["command"]
    assert digest(command[2]) == plan["runner_sha256"]
    output = Path(command[command.index("--output") + 1]) / "candidate02-check-only.json"
    receipt_path = HERE / "final_policy_parity_20261022.json"
    log_path = HERE / "final_policy_parity_20261022.log"
    reference_path = HERE / "final_policy_parity_reference_20261022.json"
    assert not any(p.exists() for p in [output, receipt_path, log_path, reference_path])
    for name, expected in plan["accuracy_bindings"].items():
        assert digest(ARCHIVE / "accuracy_wave18b_recovery" / name) == expected
    package = Path(command[command.index("--package-root") + 1])
    assert digest(package / "ctboost/_core.cp312-win_amd64.pyd") == plan["candidate_runtime_native_sha256"]
    available = psutil.virtual_memory().available
    assert available >= 12 * 1024**3
    psutil.Process().cpu_affinity([0])
    watchdog_path = HERE / "release_validation_watchdog.py"
    assert digest(watchdog_path) == "00b9f195b90285c0ca7fb89ff26dcbd679568a856954b61c3f2b52a5943445f8"
    spec = importlib.util.spec_from_file_location("parity_watchdog", watchdog_path)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    environment = os.environ.copy()
    for key in ["CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        environment[key] = "1"
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = watchdog.watch(process, maximum_seconds=600, maximum_rss_bytes=2 * 1024**3)
    success = result["exit_code"] == 0 and result["resource_failure"] is None and result["cleanup_complete"]
    receipt = {"status": "worker_passed" if success else "failed", "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(), "command": command,
        "plan_sha256": digest(plan_path), "runner_sha256": digest(__file__),
        "available_before_bytes": available, "log_sha256": digest(log_path), **result}
    if success:
        current, previous = read(output), read(timing / "results/reference.json")
        success = (current["check_only"] is True and len(current["records"]) == 108
            and current["records"] == previous["records"]
            and current["runtime"]["ctboost_native_sha256"] == plan["candidate_runtime_native_sha256"]
            and current["runtime"]["ctboost_build"]["version"] == "0.1.62")
        shutil.copyfile(output, reference_path)
        receipt.update(status="passed" if success else "record_comparison_failed",
            reference_sha256=digest(reference_path), timed_reference_sha256=digest(timing / "results/reference.json"),
            exact_model_and_batch_records=success, full_development_models=36, batch_records=108,
            scope="Correctness only; release0.1.62 runtime differs from the timed04ad runtime. No fits or accuracy scoring.")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k:receipt[k] for k in ["status", "wall_seconds", "peak_rss_bytes", "cleanup_complete"]}))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
