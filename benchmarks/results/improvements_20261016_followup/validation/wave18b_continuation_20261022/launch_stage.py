"""Launch one unchanged scientific stage and preserve its direct process receipt.

Every union worker is limited by the existing frozen dispatcher/watchdog. This
wrapper does not discover, terminate, or otherwise manage unrelated processes.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(r"C:\apps\ctboost")
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "benchmarks/results/improvements_20261016_followup"
RECOVERY = ARCHIVE / "accuracy_wave18b_recovery"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "select", "run_union", "development"))
    args = parser.parse_args()
    import psutil
    evidence = HERE / args.stage
    evidence.mkdir(exist_ok=False)
    plan = json.loads((RECOVERY / "recovery_plan.json").read_text())
    audit = HERE / "fold_terminal_audit.json"
    assert digest(audit) == "9295fd07e3402b610d8ae1402c78c1a561c37f7c430c91568b945aa640f518ae"
    resource = plan["resources"]
    psutil.Process().cpu_affinity([resource["cpu"]])
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    script = (ARCHIVE / "validation/wave18b_external_recovery/controller.py") if args.stage == "verify" else (ROOT / "benchmarks/fair_accuracy_current_round_cv.py")
    expected_script_hash = plan["script_sha256"] if args.stage == "verify" else plan["main_sha256"]
    assert digest(script) == expected_script_hash
    command = [sys.executable, "-I", str(script), args.stage]
    keys = tuple(plan["arguments"]) if args.stage == "verify" else ("output", "artifacts", "package_root", "dependencies", "previous")
    for key in keys:
        command += ["--"+key.replace("_", "-"), plan["arguments"][key]]
    receipt = {"started_at": now(), "stage": args.stage, "command": command,
        "launcher_sha256": digest(__file__), "script_sha256": digest(script),
        "fold_audit_sha256": digest(audit), "available_memory_before_bytes": psutil.virtual_memory().available,
        "cpu_affinity": psutil.Process().cpu_affinity(), "thread_environment": {k:env[k] for k in env if k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "resources": resource, "resource_enforcement": "Unchanged main dispatcher applies the frozen headroom gate and owned-process watchdog to every union worker. This wrapper only waits for its direct child; no PID lookup or unrelated process management."}
    write(evidence / "launch.json", receipt)
    start = time.perf_counter()
    log = evidence / "stdout.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        identity = {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()}
        write(evidence / "owned_direct_child.json", identity)
        exit_code = process.wait()
    receipt.update(finished_at=now(), elapsed_seconds=time.perf_counter()-start, exit_code=exit_code,
                   direct_child=identity, stdout_sha256=digest(log))
    write(evidence / "completion.json", receipt)
    print(json.dumps({"stage": args.stage, "exit_code": exit_code, "elapsed_seconds": receipt["elapsed_seconds"], "receipt": str(evidence / "completion.json")}))
    raise SystemExit(exit_code)

if __name__ == "__main__":
    main()
