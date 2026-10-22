"""Snapshot this study's implementation and hash its completed evidence archive."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ARCHIVE = Path(__file__).resolve().parent
ROOT = ARCHIVE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    changed = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT, text=True
    ).splitlines()
    paths = [ROOT / name for name in changed]
    paths += list((ROOT / "benchmarks").glob("fair_*.py"))
    paths += [ROOT / name for name in subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "tests", "ctboost"],
        cwd=ROOT, text=True).splitlines() if name.endswith(".py")]
    # Include new top-level runners and protocols without copying this archive's
    # own source snapshots back into itself.
    paths += [ROOT / name for name in subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "benchmarks"],
        cwd=ROOT, text=True).splitlines()
        if Path(name).parent == Path("benchmarks") and name.endswith((".py", ".json"))]
    paths += [ROOT / name for name in (
        "benchmarks/inference_real_followup.py",
        "benchmarks/serialization_load_scout.py",
        "benchmarks/fixed_leaf_refit.py",
        "benchmarks/categorical_codes_prototype.py",
        "tests/test_feature_pipeline_categorical_row.py",
        "agent.md",
    )]
    source_records = []
    for original in sorted(set(paths)):
        if not original.is_file():
            continue
        relative = original.relative_to(ROOT)
        destination = ARCHIVE / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        source_records.append({"path": relative.as_posix(), "sha256": digest(original)})
    (ARCHIVE / "source" / "tracked_changes.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    )
    records = []
    local_only_artifacts = []
    for path in sorted(ARCHIVE.rglob("*")):
        if not path.is_file() or path == ARCHIVE / "manifest.json" or "__pycache__" in path.parts:
            continue
        record = {"path": path.relative_to(ARCHIVE).as_posix(),
                  "bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix.lower() == ".npz":
            # Preserve frozen prediction provenance without publishing benchmark
            # labels and row identities carried by the STOP audit arrays.
            local_only_artifacts.append(record)
        else:
            records.append(record)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Public study evidence and source snapshots are listed in files. NPZ prediction archives remain local; their paths, sizes and exact hashes are listed separately in local_only_artifacts. Large local models/data and built native binaries are referenced by their separate receipts.",
        "local_only_reason": "NPZ archives contain benchmark STOP predictions, labels and row IDs; their contents are not included in the public archive. Nested frozen receipts remain unchanged.",
        "source_files": source_records,
        "files": records,
        "local_only_artifacts": local_only_artifacts,
    }
    (ARCHIVE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(records), "bytes": sum(item["bytes"] for item in records),
                      "local_only_artifacts": len(local_only_artifacts),
                      "local_only_bytes": sum(item["bytes"] for item in local_only_artifacts),
                      "source_files": len(source_records)}))


if __name__ == "__main__":
    main()
