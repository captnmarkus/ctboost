"""Standalone, hash-bound remote transport for the registered 0.1.60 worker.

The controller embeds sources and the public plan receipt. This module never
selects configurations, changes parent ownership, or retries a started parent.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

SHARD_INDEX = 0
PAYLOAD_BASE64 = ""
CTBOOST_VERSION = "0.1.60"
BENCHMARK_NAME = "ctboost_0160_lite_hpo200"
TABARENA_COMMIT = "31026f7d758390994353eba79fbfa6747616f365"
PORTFOLIO_200_SHA256 = (
    "bd1b81b98a89ab33ac4cea35cb4b7dd7727b3bcfa3bee1b044fd3fb44f965c72"
)
GIB = 1024**3
MAX_PARENTS_PER_SHARD = 12


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_relative(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
    ):
        raise ValueError("Unsafe packaged path")
    return path


def shard_specs(plan):
    parents = plan["parents"]
    identities = [row["parent_id"] for row in parents]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate parent identity")
    for ordinal, row in enumerate(parents):
        owner = "local" if ordinal % 9 < 4 else "kaggle"
        if row["ordinal"] != ordinal or row["owner"] != owner:
            raise ValueError("Parent assignment differs from the frozen partition")
    remote = [row for row in parents if row["owner"] == "kaggle"]
    # The first singleton is a transport check before the other slots activate.
    groups = ([remote[:1]] if remote else []) + [
        remote[start : start + MAX_PARENTS_PER_SHARD]
        for start in range(1, len(remote), MAX_PARENTS_PER_SHARD)
    ]
    count = len(groups)
    return [
        {
            "shard_index": index,
            "shard_count": count,
            "parent_ids": [row["parent_id"] for row in group],
            "datasets": sorted({row["dataset"] for row in group}),
            "expected_parent_results_in_shard": len(group),
            "expected_child_fits_in_shard": len(group) * 8,
        }
        for index, group in enumerate(groups)
    ]


def unpack_payload(destination, *, existing=False):
    if destination.exists() and not existing:
        raise ValueError("Use a fresh remote workspace; preserve earlier attempts")
    destination.mkdir(parents=True, exist_ok=existing)
    data = base64.b64decode(PAYLOAD_BASE64, validate=True)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            relative = safe_relative(member.name)
            if not member.isfile() or member.name in files:
                raise ValueError("Unexpected or duplicated packaged member")
            files[member.name] = archive.extractfile(member).read()
            target = destination.joinpath(*relative.parts)
            if existing:
                if (
                    target.is_symlink()
                    or not target.is_file()
                    or target.read_bytes() != files[member.name]
                ):
                    raise ValueError("Previously unpacked source changed")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(files[member.name])
    bundle = json.loads(files.pop("bundle.json"))
    if set(files) != set(bundle["files_sha256"]):
        raise ValueError("Packaged file inventory mismatch")
    if any(
        hashlib.sha256(value).hexdigest() != bundle["files_sha256"][name]
        for name, value in files.items()
    ):
        raise ValueError("Packaged source hash mismatch")
    plan = json.loads(files["plan.json"])
    if (
        json_hash(plan) != bundle["plan_sha256"]
        or shard_specs(plan) != bundle["shards"]
    ):
        raise ValueError("Packaged plan or assignment mismatch")
    return bundle, plan


def run_command(command, *, timeout=1800):
    # Commands contain public package/source identifiers only.
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True, timeout=timeout)


def install_runtime(package, workspace, artifacts, plan):
    if (
        sys.version_info[:2] != (3, 12)
        or sys.platform != "linux"
        or platform.machine() not in {"x86_64", "AMD64"}
    ):
        raise RuntimeError(
            "This registered remote run requires CPython3.12 on Linux x86_64"
        )
    candidates = [
        row
        for row in plan["public_wheels"]
        if "-cp312-cp312-manylinux" in row["filename"]
        and row["filename"].endswith("_x86_64.whl")
    ]
    if len(candidates) != 1:
        raise ValueError("Plan must identify exactly one approved Linux wheel")
    approved = candidates[0]
    wheel = artifacts / "wheels" / approved["filename"]
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(approved["url"], timeout=120) as response, wheel.open(
        "wb"
    ) as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    if file_hash(wheel) != approved["sha256"]:
        raise ValueError("Public wheel hash differs from the registered release")
    source = workspace / "tabarena-source"
    run_command(["git", "init", str(source)])
    run_command(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--depth",
            "1",
            "https://github.com/captnmarkus/tabarena.git",
            TABARENA_COMMIT,
        ]
    )
    run_command(["git", "-C", str(source), "checkout", "--detach", TABARENA_COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != TABARENA_COMMIT:
        raise ValueError("Downloaded TabArena source commit differs")
    environment = workspace / "venv"
    # Kaggle's Debian Python cannot seed venv pip through ensurepip. Its host
    # pip can manage an unseeded venv without installing into the host itself.
    run_command([sys.executable, "-m", "venv", "--without-pip", str(environment)])
    python = environment / "bin/python"
    pins = [
        f"{name}=={version}"
        for name, version in sorted(plan["package_pins"].items())
        if name != "ctboost"
    ]
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--pre",
            "--index-url",
            "https://pypi.org/simple",
            str(wheel),
            *pins,
            "-e",
            str(source / "packages/bencheval"),
            "-e",
            str(source / "packages/tabarena"),
            "-e",
            str(source / "packages/tabflow_slurm"),
        ],
        timeout=5400,
    )
    write_json(
        artifacts / "runtime-install.json",
        {
            "wheel": approved,
            "tabarena_commit": head,
            "python": sys.version,
            "pins": plan["package_pins"],
        },
    )
    return python, wheel


def checkpoint(workspace, artifacts, manifest):
    """Archive receipts and terminal results, omitting datasets and live models."""
    paths = set()
    output = workspace / "output"
    for record in manifest["parents"]:
        for key in ("manifest_path", "raw_path", "log_path"):
            if record.get(key):
                path = output / safe_relative(record[key])
                if path.is_file():
                    paths.add(path)
        for relative in record.get("evidence_paths", []):
            path = output / safe_relative(relative)
            if path.is_file():
                paths.add(path)
    paths.update(
        path for path in (output / "scheduler").glob("*.json") if path.is_file()
    )
    paths.update(
        output / name
        for name in ("preflight-kaggle.json", "registration_verified.json")
        if (output / name).is_file()
    )
    archive = artifacts / f"ctboost_0160_s{manifest['shard_index']:03d}_raw.tar.gz"
    temporary = archive.with_suffix(".tmp")
    files = []
    with tarfile.open(temporary, "w:gz") as stream:
        for path in sorted(paths):
            relative = "workspace/output/" + path.relative_to(output).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mtime = 0
            stream.addfile(info, io.BytesIO(data))
            files.append(
                {
                    "path": relative,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    temporary.replace(archive)
    manifest["workspace_files"] = files
    manifest["workspace_archive"] = {
        "path": archive.relative_to(artifacts.parent).as_posix(),
        "size_bytes": archive.stat().st_size,
        "sha256": file_hash(archive),
    }
    manifest["result_file_count"] = sum(
        row["status"] == "complete" for row in manifest["parents"]
    )
    manifest["failed_parent_count"] = sum(
        row["status"] == "failed" for row in manifest["parents"]
    )
    manifest["terminal_parent_count"] = (
        manifest["result_file_count"] + manifest["failed_parent_count"]
    )
    manifest["last_checkpoint_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(artifacts / "manifest.json", manifest)


def parent_paths(output, parent):
    relative = Path(parent["config_name"]) / str(parent["task_id"]) / "0_0"
    return (
        output / "artifacts" / relative / "manifest.json",
        output / "data" / relative / "results.pkl",
    )


def kill_owned_process_tree(process):
    """Called only with the Popen object launched by this live worker."""
    import psutil

    if process.poll() is not None:
        return
    try:
        owned = psutil.Process(process.pid)
        children = owned.children(recursive=True)
        owned.suspend()
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        owned.kill()
        process.wait(timeout=30)
    except psutil.NoSuchProcess:
        process.wait(timeout=30)


def run_parents(package, workspace, artifacts, plan, spec, manifest, python, wheel):
    import psutil

    available_cpus = psutil.Process().cpu_affinity()
    if len(available_cpus) < 4:
        raise RuntimeError(
            "Registered two-parent remote allocation requires four logical CPUs"
        )
    affinities = [available_cpus[:2], available_cpus[2:4]]
    resources = plan["resources"]
    if (
        resources["num_cpus"] != 2
        or resources["memory_limit_gb"] != 8
        or resources["parent_wall_limit_seconds"] != 4500
    ):
        raise ValueError("Unexpected parent resource contract")
    output = workspace / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    parents = {row["parent_id"]: row for row in plan["parents"]}
    pending = list(spec["parent_ids"])
    active = {}
    environment = os.environ.copy()
    environment.update(
        {
            name: "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        }
    )
    environment.update(
        CTBOOST_HIST_THREADS="2", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="160"
    )
    try:
        while pending or active:
            for slot in range(2):
                if slot in active or not pending:
                    continue
                # Reserve each active parent's remaining eight-GiB allowance.
                growth = 0
                for job in active.values():
                    try:
                        rss = psutil.Process(job["process"].pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        rss = 0
                    growth += max(0, 8 * GIB - rss)
                if psutil.virtual_memory().available - growth - 8 * GIB < 4 * GIB:
                    if not active:
                        raise RuntimeError(
                            "Insufficient memory for the registered parent allocation"
                        )
                    continue
                parent_id = pending.pop(0)
                parent = parents[parent_id]
                if parent["owner"] != "kaggle":
                    raise ValueError("Refusing a locally owned parent")
                log_path = output / "logs" / f"{parent_id}.log"
                command = [
                    str(python),
                    "-I",
                    "-B",
                    str(package / "benchmarks/tabarena/hpo_0160.py"),
                    "run-parent",
                    "--plan",
                    str(package / "plan.json"),
                    "--output",
                    str(output),
                    "--parent-id",
                    parent_id,
                    "--host",
                    "kaggle",
                    "--wheel",
                    str(wheel),
                    "--registration",
                    str(package / "registration.json"),
                    "--affinity",
                    ",".join(map(str, affinities[slot])),
                ]
                record = {
                    "parent_id": parent_id,
                    "status": "starting",
                    "affinity": affinities[slot],
                    "log_path": log_path.relative_to(output).as_posix(),
                }
                manifest["parents"].append(record)
                write_json(output / "scheduler" / f"{parent_id}.json", record)
                with log_path.open("wb") as log:
                    process = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        start_new_session=True,
                    )
                record.update(status="running", pid=process.pid)
                active[slot] = {
                    "process": process,
                    "started": time.monotonic(),
                    "record": record,
                    "parent": parent,
                }
                write_json(output / "scheduler" / f"{parent_id}.json", record)
            finished = []
            for slot, job in active.items():
                process = job["process"]
                record = job["record"]
                if process.poll() is None and time.monotonic() - job["started"] > 4500:
                    kill_owned_process_tree(process)
                    record["failure"] = "parent_wall_limit_exceeded"
                if process.poll() is None:
                    continue
                record["exit_code"] = process.returncode
                record["wall_seconds"] = time.monotonic() - job["started"]
                child_manifest, raw = parent_paths(output, job["parent"])
                child = {}
                if child_manifest.is_file():
                    child = json.loads(child_manifest.read_text())
                    record.update(
                        manifest_path=child_manifest.relative_to(output).as_posix(),
                        manifest_sha256=file_hash(child_manifest),
                    )
                    record["evidence_paths"] = [
                        path.relative_to(output).as_posix()
                        for path in sorted(child_manifest.parent.glob("*.json"))
                    ]
                if raw.is_file():
                    record.update(
                        raw_path=raw.relative_to(output).as_posix(),
                        raw_sha256=file_hash(raw),
                    )
                record["worker_status"] = child.get("status")
                record["status"] = (
                    "complete"
                    if process.returncode == 0
                    and child.get("status") == "complete"
                    and raw.is_file()
                    else "failed"
                )
                write_json(output / "scheduler" / f"{record['parent_id']}.json", record)
                finished.append(slot)
                print(
                    json.dumps(
                        {"parent_id": record["parent_id"], "status": record["status"]}
                    ),
                    flush=True,
                )
            for slot in finished:
                active.pop(slot)
            if finished:
                checkpoint(workspace, artifacts, manifest)
            if active:
                time.sleep(0.5)
    finally:
        for job in active.values():
            kill_owned_process_tree(job["process"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/kaggle/working/ctboost-0160")
    )
    parser.add_argument(
        "--installed-runtime", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    # Keep the environment, caches, and datasets outside persistent kernel output.
    # Explicit checkpoint archives below preserve the required execution evidence.
    workspace = Path(tempfile.gettempdir()) / (
        "ctboost-0160-" + file_hash(__file__)[:16]
    )
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = {
        "shard_index": SHARD_INDEX,
        "status": "initializing",
        "parents": [],
        "worker_sha256": file_hash(__file__),
        "ctboost_version": CTBOOST_VERSION,
        "benchmark_name": BENCHMARK_NAME,
        "tabarena_commit": TABARENA_COMMIT,
        "portfolio_200_sha256": PORTFOLIO_200_SHA256,
    }
    try:
        package = workspace / "package"
        bundle, plan = unpack_payload(package, existing=args.installed_runtime)
        spec = bundle["shards"][SHARD_INDEX]
        manifest.update(
            spec,
            plan_sha256=bundle["plan_sha256"],
            package_sha256=hashlib.sha256(base64.b64decode(PAYLOAD_BASE64)).hexdigest(),
            resources=plan["resources"],
            allocation={
                "parents_per_kernel": 2,
                "physical_free_reserve_bytes": 4 * GIB,
            },
        )
        checkpoint(workspace, artifacts, manifest)
        if not args.installed_runtime:
            python, wheel = install_runtime(package, workspace, artifacts, plan)
            os.execv(
                str(python),
                [
                    str(python),
                    "-I",
                    "-B",
                    str(Path(__file__).resolve()),
                    "--output-root",
                    str(root),
                    "--installed-runtime",
                ],
            )
        python = workspace / "venv/bin/python"
        if Path(sys.prefix).resolve() != (workspace / "venv").resolve():
            raise ValueError("Scheduler must use the freshly installed pinned runtime")
        wheels = list((artifacts / "wheels").glob("ctboost-0.1.60-*.whl"))
        if len(wheels) != 1:
            raise ValueError("Approved runtime wheel is missing")
        wheel = wheels[0]
        run_command(
            [
                str(python),
                "-I",
                "-B",
                str(package / "benchmarks/tabarena/hpo_0160.py"),
                "preflight",
                "--plan",
                str(package / "plan.json"),
                "--output",
                str(workspace / "output"),
                "--host",
                "kaggle",
                "--wheel",
                str(wheel),
                "--registration",
                str(package / "registration.json"),
            ]
        )
        manifest["status"] = "running"
        run_parents(package, workspace, artifacts, plan, spec, manifest, python, wheel)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="incomplete", fatal_error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        checkpoint(workspace, artifacts, manifest)


if __name__ == "__main__":
    main()
