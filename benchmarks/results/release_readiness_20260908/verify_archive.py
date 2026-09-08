"""Verify the retained release evidence without loading models or benchmark data."""

from pathlib import Path
import hashlib
import json


root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["algorithm"] == "sha256"
expected = {entry["path"]: entry for entry in manifest["files"]}
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path != root / "manifest.json"
    and "__pycache__" not in path.parts
}
assert actual == expected.keys(), {
    "missing": sorted(expected.keys() - actual),
    "unexpected": sorted(actual - expected.keys()),
}
for name, entry in expected.items():
    payload = (root / name).read_bytes()
    assert len(payload) == entry["bytes"], name
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"], name
print(f"Verified {len(expected)} archived files; no model or benchmark data loaded.")
