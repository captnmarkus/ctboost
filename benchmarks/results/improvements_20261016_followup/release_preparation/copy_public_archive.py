"""Copy a completed study archive using only its explicit public manifest.

Run after the original study archive is frozen. This never runs archive.py,
deletes files, overwrites differing destination content, or copies local NPZs.
The optional copy receipt must live outside both source and destination archives.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checked_path(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError(f"Invalid manifest-relative path: {relative}")
    path = root.joinpath(*name.parts)
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError(f"Manifest path escapes the archive: {relative}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--destination", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--receipt", type=lambda value: Path(value).resolve())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (args.source == args.destination or args.source.is_relative_to(args.destination)
            or args.destination.is_relative_to(args.source)):
        raise ValueError("Use separate source and destination archives")
    if args.receipt and (args.receipt.exists() or args.receipt.is_relative_to(args.source)
                         or args.receipt.is_relative_to(args.destination)):
        raise ValueError("Use a fresh receipt path outside both archives")
    manifest_path = args.source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "files" in manifest:
        records = manifest["files"]
    elif "files_sha256" in manifest:
        records = [{"path": path, "sha256": value}
                   for path, value in manifest["files_sha256"].items()]
    else:
        raise ValueError("Unrecognized completed public manifest")
    records = records + [{"path": "manifest.json", "sha256": digest(manifest_path)}]
    if len({row["path"] for row in records}) != len(records):
        raise ValueError("Duplicate public manifest path")
    local_only = {row["path"] for row in manifest.get("local_only_artifacts", [])}
    verified = []
    for row in records:
        relative = row["path"]
        if relative in local_only or Path(relative).suffix.lower() == ".npz":
            raise ValueError("Local prediction archives must not be published")
        source = checked_path(args.source, relative)
        destination = checked_path(args.destination, relative)
        if not source.is_file() or digest(source) != row["sha256"]:
            raise ValueError(f"Source does not match frozen manifest: {relative}")
        size = source.stat().st_size
        if "bytes" in row and size != row["bytes"]:
            raise ValueError(f"Frozen source size changed: {relative}")
        if destination.exists() and (not destination.is_file() or digest(destination) != row["sha256"]):
            raise ValueError(f"Preserve differing destination content: {relative}")
        verified.append({"path": relative, "sha256": row["sha256"], "bytes": size})
    if not args.verify_only:
        for row in verified:
            destination = checked_path(args.destination, row["path"])
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checked_path(args.source, row["path"]), destination)
            if digest(destination) != row["sha256"]:
                raise ValueError(f"Copy changed bytes: {row['path']}")
        expected = {row["path"] for row in verified}
        actual = {path.relative_to(args.destination).as_posix()
                  for path in args.destination.rglob("*") if path.is_file()}
        if actual != expected:
            raise ValueError(f"Destination membership differs: extra={sorted(actual - expected)} missing={sorted(expected - actual)}")
    receipt = {"recorded_at": datetime.now(timezone.utc).isoformat(),
               "source": str(args.source), "destination": str(args.destination),
               "mode": "verify_only" if args.verify_only else "copied_and_verified",
               "source_manifest_sha256": digest(manifest_path),
               "public_file_count": len(verified), "public_bytes": sum(row["bytes"] for row in verified),
               "local_only_manifest_entries": len(local_only), "files": verified}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "files"}))


if __name__ == "__main__":
    main()
