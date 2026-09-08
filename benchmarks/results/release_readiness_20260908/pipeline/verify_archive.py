"""Verify the archived pipeline evidence without importing optional libraries."""

import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
actual_paths = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts
}
assert actual_paths == set(manifest["files"]), "missing or unexpected archive members"
for relative, expected in manifest["files"].items():
    data = (root / relative).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected["sha256"], relative
    assert len(data) == expected["size_bytes"], relative
inventory = json.loads((root / "original-file-inventory.json").read_text(encoding="utf-8"))
for original in inventory["files"]:
    data = (root / original["archive_path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == original["sha256"], original["archive_path"]
    assert len(data) == original["size_bytes"], original["archive_path"]
print("Verified {} files, including {} unchanged originals.".format(len(actual_paths), len(inventory["files"])))
