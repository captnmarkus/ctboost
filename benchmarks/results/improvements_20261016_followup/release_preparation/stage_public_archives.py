"""Stage manifest-listed public evidence and verify the exact indexed bytes.

Run in the clean release worktree after copying both completed archives.
Receipts and pathspecs live outside the frozen archives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--receipt", type=lambda x: Path(x).resolve(), required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    assert not args.receipt.is_relative_to(args.release)
    expected = {}
    manifest_hashes = {}
    for archive in ["improvements_20261016", "improvements_20261016_followup"]:
        prefix = f"benchmarks/results/{archive}/"
        manifest_path = args.release / prefix / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        records = manifest.get("files")
        if records is None:
            records = [{"path": path, "sha256": value} for path, value in manifest["files_sha256"].items()]
        local_only = {row["path"] for row in manifest.get("local_only_artifacts", [])}
        manifest_hashes[archive] = hashlib.sha256(manifest_bytes).hexdigest()
        for row in records + [{"path": "manifest.json", "sha256": manifest_hashes[archive]}]:
            name = PurePosixPath(row["path"])
            assert not name.is_absolute() and ".." not in name.parts
            assert "\\" not in row["path"]
            assert name.suffix.lower() != ".npz" and row["path"] not in local_only
            relative = prefix + row["path"]
            assert relative not in expected
            assert hashlib.sha256((args.release / relative).read_bytes()).hexdigest() == row["sha256"]
            expected[relative] = row["sha256"]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    pathspec = args.receipt.with_suffix(".pathspec")
    with pathspec.open("xb") as stream:
        stream.write(b"\0".join(name.encode("utf-8") for name in sorted(expected)) + b"\0")
    subprocess.run(["git", "add", "-f", "--pathspec-file-nul", f"--pathspec-from-file={pathspec}"],
                   cwd=args.release, check=True)
    index = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=args.release)
    entries = {}
    for record in index.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        relative = name.decode("utf-8")
        if any(relative.startswith(f"benchmarks/results/{key}/") for key in manifest_hashes):
            assert stage == b"0" and mode == b"100644"
            entries[relative] = oid
    assert set(entries) == set(expected), (set(entries) - set(expected), set(expected) - set(entries))
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=args.release,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    total = 0
    for name in sorted(expected):
        process.stdin.write(entries[name] + b"\n")
        process.stdin.flush()
        oid, kind, size = process.stdout.readline().split()
        assert oid == entries[name] and kind == b"blob"
        data = process.stdout.read(int(size))
        assert process.stdout.read(1) == b"\n"
        assert hashlib.sha256(data).hexdigest() == expected[name], name
        total += len(data)
    process.stdin.close()
    assert process.wait(timeout=10) == 0
    receipt = {"recorded_at": datetime.now(timezone.utc).isoformat(),
        "release": str(args.release), "public_archive_files": len(expected),
        "public_archive_bytes": total, "manifest_sha256": manifest_hashes,
        "all_indexed_bytes_match_frozen_manifests": True,
        "no_local_npz_staged": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
