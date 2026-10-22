"""Archive provenance and compatibility evidence without copying real models."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(r"C:/apps/ctboost")
WORK = ROOT / ".tmp/improvements-20261016/ctr-precision"
OUT = ROOT / "benchmarks/results/improvements_20261016_followup/inference/ctr_precision"
OUT.mkdir(parents=True, exist_ok=True)
baseline = json.loads((WORK / "baseline-source-hashes.json").read_text())
compiled = json.loads((WORK / "native-source-hashes.json").read_text())
assert all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
           for relative, expected in compiled.items()), "Compiled native source changed"
sources = set(baseline) | {"tests/test_ctr_precision.py", "tests/test_feature_pipeline_ctr_inference.py",
                         "tests/test_distributed_training_raw_pipeline.py", "tests/conftest.py"}
patch = []
changes = {}
for relative in sorted(sources):
    current = ROOT / relative
    previous = WORK / "baseline-source" / relative
    if previous.exists():
        assert hashlib.sha256(previous.read_bytes()).hexdigest() == baseline[relative]
        before = previous.read_text(encoding="utf-8")
    elif relative in {"tests/test_ctr_precision.py", "tests/conftest.py"}:
        before = ""
    else:
        before = subprocess.check_output(["git","show",f"HEAD:{relative}"], cwd=ROOT).decode("utf-8")
    after = current.read_text(encoding="utf-8")
    patch.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
                                    fromfile=f"before/{relative}",tofile=f"after/{relative}"))
    changes[relative] = {"sha256":hashlib.sha256(current.read_bytes()).hexdigest(),
                         "baseline":baseline.get(relative, "new file" if not before else "unchanged HEAD source before this fix")}
(OUT / "ctr-only.patch").write_text("".join(patch),encoding="utf-8")
(OUT / "changed-source-hashes.json").write_text(json.dumps(changes,indent=2)+"\n")
for path in WORK.iterdir():
    if path.is_file() and path.suffix in {".py", ".json", ".log", ".cmd", ".txt"}:
        shutil.copy2(path,OUT / path.name)
shutil.copy2(WORK / "legacy-model.ctb", OUT / "legacy-model.ctb")
for abi in ("cp310","cp312"):
    shutil.copy2(WORK / f"runtime-{abi}/build_receipt.json",OUT / f"build-receipt-{abi}.json")
    cache = WORK / f"build-{abi}/CMakeCache.txt"
    flags = [line for line in cache.read_text().splitlines()
             if line.startswith(("CMAKE_CXX_FLAGS:","CMAKE_CXX_FLAGS_RELEASE:","Python_EXECUTABLE:","CTBOOST_ENABLE_CUDA:","CTBOOST_BUILD_NATIVE_TESTS:"))]
    (OUT / f"build-flags-{abi}.json").write_text(json.dumps(flags,indent=2)+"\n")
review = ROOT / "benchmarks/results/improvements_20261016_followup/validation/ctr_precision_independent_review.json"
shutil.copy2(review, OUT / "independent-review.json")
manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
            if p.is_file() and p.name != "manifest.json"}
(OUT / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"archive":str(OUT),"artifacts":len(manifest)}))
