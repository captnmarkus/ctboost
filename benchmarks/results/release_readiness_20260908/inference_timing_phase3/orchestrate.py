from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys

import psutil

ROOT = Path(r"C:\apps\ctboost")
OUTPUT = ROOT / ".tmp/release-readiness-20260908/timing-phase3"
ADAPTER = ROOT / "benchmarks/inference_release_timing.py"
PYTHON = ROOT / ".tmp/inference-accuracy-20260908/baseline-env/Scripts/python.exe"
EXPECTED_PLAN = "0abc3d88279266e183a79f8a1c2167229409db5a6a5268e4a16c58c144a5557e"


def write(receipt):
    path = OUTPUT / "orchestration.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


receipt = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "plan_sha256": EXPECTED_PLAN,
           "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "pid": psutil.Process().pid, "create_time": psutil.Process().create_time(), "passes": []}
assert not (OUTPUT / "orchestration.json").exists()
write(receipt)
for index, arm in enumerate(("baseline", "final", "final", "baseline"), start=1):
    assert hashlib.sha256((OUTPUT / "plan.json").read_bytes()).hexdigest() == EXPECTED_PLAN
    command = [str(PYTHON), "-I", str(ADAPTER), "measure", "--output", str(OUTPUT), "--pass-index", str(index)]
    with (OUTPUT / f"pass-{index}-{arm}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        record = {"index": index, "arm": arm, "command": command, "pid": process.pid,
                  "create_time": psutil.Process(process.pid).create_time()}
        receipt["passes"].append(record)
        write(receipt)
        record["exit_code"] = process.wait()
    write(receipt)
    print(json.dumps({"pass": index, "arm": arm, "exit_code": record["exit_code"]}), flush=True)
    if record["exit_code"] != 0:
        raise SystemExit(record["exit_code"])
receipt["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt["complete"] = True
write(receipt)
