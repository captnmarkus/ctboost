"""Package an isolated runtime without touching any frozen installation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(r"C:/apps/ctboost")
OUT = ROOT / ".tmp/improvements-20261016/ctr-precision"
abi = sys.argv[1] if len(sys.argv) > 1 else "cp312"
assert abi in {"cp310", "cp312"}
target = OUT / f"runtime-{abi}" / "ctboost"
target.mkdir(parents=True, exist_ok=True)
hashes = {}
for source in (ROOT / "ctboost").rglob("*.py"):
    rel = source.relative_to(ROOT / "ctboost")
    destination = target / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    hashes[rel.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
native = OUT / f"build-{abi}/_core.{abi}-win_amd64.pyd"
shutil.copy2(native, target / native.name)
receipt = {"native_sha256":hashlib.sha256(native.read_bytes()).hexdigest(),
           "python_sha256":hashes, "native_source_manifest_sha256":hashlib.sha256((OUT / "native-source-hashes.json").read_bytes()).hexdigest(),
           "build_recipe_sha256":hashlib.sha256((OUT / f"build-{abi}.cmd").read_bytes()).hexdigest(),
           "baseline_source_manifest_sha256":hashlib.sha256((OUT / "baseline-source-hashes.json").read_bytes()).hexdigest(),
           "source_snapshot":str(OUT / "source"), "package":str(target.parent),
           "build_threads":2,"build_cpu_affinity":[0,2],"timing_claim":False,
           "scope":"CTR precision only; existing accuracy and production runtimes unchanged"}
(OUT / f"runtime-{abi}/build_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"native_sha256":receipt["native_sha256"],"package":receipt["package"]}))
