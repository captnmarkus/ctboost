"""Reversibly activate verified candidates after preserving checkout binaries."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(r"C:/apps/ctboost")
WORK = ROOT / ".tmp/improvements-20261016/ctr-precision"
backup = WORK / "checkout-binary-backup"
backup.mkdir(exist_ok=False)
previous = {
    "cp310":"6df3e210c2086004519e7d2977e69db72d465e2e007799b9ba4288b5099d7479",
    "cp312":"94b2b0bf3f0b4863b8c27eb308a41354a92025b6b81f2c7f539c6a4923a7e233",
}
records = []
for abi, expected in previous.items():
    target = ROOT / f"ctboost/_core.{abi}-win_amd64.pyd"
    candidate = WORK / f"runtime-{abi}/ctboost" / target.name
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual == expected
    shutil.copy2(target, backup / target.name)
    assert hashlib.sha256((backup / target.name).read_bytes()).hexdigest() == expected
    record = {"abi":abi,"target":str(target),"backup":str(backup / target.name),
              "before_sha256":actual,"after_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest()}
    records.append(record)
for record in records:
    shutil.copy2(WORK / f"runtime-{record['abi']}/ctboost" / Path(record["target"]).name, record["target"])
    assert hashlib.sha256(Path(record["target"]).read_bytes()).hexdigest() == record["after_sha256"]
(WORK / "checkout-activation.json").write_text(json.dumps({"records":records,
    "frozen_accuracy_runtime_changed":False,"previous_isolated_runtimes_changed":False,
    "other_python_ABIs_changed":False},indent=2)+"\n")
print(json.dumps(records),flush=True)
