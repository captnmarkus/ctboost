"""Snapshot the CTR-only candidate and prepare a separate native build."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(r"C:/apps/ctboost")
DEST = ROOT / ".tmp/improvements-20261016/ctr-precision"
source = DEST / "source"
assert not source.exists()
source.mkdir()
paths = [ROOT / "CMakeLists.txt"]
for directory in ("src", "include", "tests/native"):
    paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
hashes = {}
for path in paths:
    relative = path.relative_to(ROOT)
    target = source / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
(DEST / "native-source-hashes.json").write_text(json.dumps(hashes, indent=2)+"\n")
recipe = (ROOT / ".tmp/improvements-20261016/build-cp312.cmd").read_text()
recipe = recipe.replace("cmake -S C:\\apps\\ctboost -B", f"cmake -S {source} -B")
recipe = recipe.replace(r"C:\apps\ctboost\.tmp\improvements-20261016\build-cp312", str(DEST / "build-cp312"))
(DEST / "build-cp312.cmd").write_text(recipe)
print(json.dumps({"source_files":len(hashes), "source":str(source), "recipe":str(DEST / "build-cp312.cmd")}))
