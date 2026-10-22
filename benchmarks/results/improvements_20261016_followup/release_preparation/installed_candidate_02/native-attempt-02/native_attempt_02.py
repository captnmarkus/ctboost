"""Run embedded-Python native validation with the installed NumPy visible."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import psutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("validation", HERE / "validate_installed.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
out = v.OUT / "native-attempt-02"
out.mkdir(parents=True, exist_ok=False)
psutil.Process().cpu_affinity([0])
available = psutil.virtual_memory().available
receipt = {"status": "preflight", "started_at": v.now(), "available_before_bytes": available,
    "limits": {"minimum_available_bytes": v.MIN_AVAILABLE, "maximum_process_tree_rss_bytes": v.MAX_RSS,
               "maximum_seconds": v.MAX_SECONDS, "cpu_affinity": [0], "native_threads": 1},
    "runner_sha256": v.digest(Path(__file__)), "environment_helper_sha256": v.digest(HERE / "validate_installed.py"),
    "wheel_sha256": v.digest(v.WHEEL), "initial_attempt": "../native.json",
    "initial_attempt_sha256": v.digest(v.OUT / "native.json"),
    "change": "Explicit installed venv site-packages for the embedded interpreter's required NumPy import; no native or package changes.",
    "bare_base_numpy_spec": None, "no_latency_claim": True}
v.write(out / "receipt.json", receipt)
if available < v.MIN_AVAILABLE:
    receipt.update(status="deferred_insufficient_available_memory", finished_at=v.now())
    v.write(out / "receipt.json", receipt)
    raise SystemExit(75)
env = v.environment()
env["PYTHONHOME"] = sys.base_prefix
env["PYTHONPATH"] = str(v.ENV / "Lib/site-packages")
env["PYTHONNOUSERSITE"] = "1"
env["PATH"] = sys.base_prefix + os.pathsep + env["PATH"]
command = ["C:/Users/captn/miniconda3/Lib/site-packages/cmake/data/bin/ctest.exe", "--test-dir", str(v.BUILD),
           "-C", "Release", "--output-on-failure", "-V"]
watchdog_path = v.ROOT / ".tmp/improvements-20261016/release-wheel-cp312/release_build_watchdog.py"
spec = importlib.util.spec_from_file_location("watchdog", watchdog_path)
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)
receipt.update(status="running", command=command, cwd=str(HERE),
    environment={key: env[key] for key in ["PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PATH"]},
    executable_sha256=v.digest(v.BUILD / "ctboost_prediction_cache_smoke.exe"),
    watchdog_sha256=v.digest(watchdog_path))
shutil.copyfile(Path(__file__), out / Path(__file__).name)
with (out / "native.log").open("wb") as log:
    process = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    receipt.update(owned_process_pid=process.pid, owned_process_create_time=psutil.Process(process.pid).create_time())
    v.write(out / "receipt.json", receipt)
    print(json.dumps({"pid": process.pid, "started_at": receipt["started_at"], "available_before_bytes": available}), flush=True)
    result = watchdog.watch(process, maximum_seconds=v.MAX_SECONDS, maximum_rss_bytes=v.MAX_RSS)
text = (out / "native.log").read_text(encoding="utf-8", errors="replace")
receipt.update(result, finished_at=v.now(), log_sha256=v.digest(out / "native.log"))
receipt["status"] = "passed" if result["exit_code"] == 0 and result["cleanup_complete"] and not result["resource_failure"] and "checks=230 failures=0" in text else "failed"
receipt["native_checks"] = {"checks": 230, "failures": 0} if "checks=230 failures=0" in text else None
v.write(out / "receipt.json", receipt)
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt["status"] == "passed" else 1)
