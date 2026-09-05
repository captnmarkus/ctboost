"""Submit and collect the frozen CTBoost Lite HPO run using private Kaggle CPU jobs.

The controller keeps its queue on disk and downloads each completed version before
reusing a worker slot. Authentication is supplied through Kaggle's environment or
config file; credentials are never included in uploaded scripts or the run state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SHARD_COUNT = 156
ACTIVE_STATUSES = {"RUNNING", "QUEUED"}


class KernelIdentityUnavailable(RuntimeError):
    """The push response did not include a recognized canonical kernel URL."""


@contextmanager
def controller_lock(root: Path):
    """Release the exclusive queue lock automatically even after a process crash."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def log(root: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (root / "controller.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def redact(value: str) -> str:
    value = re.sub(r"(?:KGAT_|hf_)[A-Za-z0-9_-]+", "[redacted]", value)
    return re.sub(r"(?i)(bearer\s+)[^\s'\"]+", r"\1[redacted]", value)


def kaggle_command(executable: str, arguments: list[str], *, timeout: int = 300) -> str:
    process = subprocess.run(
        [executable, "kernels", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    output = redact(process.stdout + process.stderr).strip()
    if process.returncode:
        raise RuntimeError(
            f"Kaggle {arguments[0]} failed ({process.returncode}): {output[-1500:]}"
        )
    return output


def pushed_version(output: str) -> int:
    match = re.search(r"Kernel version (\d+) successfully pushed", output)
    if not match:
        raise RuntimeError(
            f"Kaggle did not confirm submission: {redact(output)[-1500:]}"
        )
    return int(match.group(1))


def pushed_kernel(output: str, *, owner: str) -> str:
    """Use Kaggle's returned identity: new kernels may be named from their title."""
    matches = re.findall(
        r"https://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)",
        output,
    )
    if not matches:
        raise KernelIdentityUnavailable(
            "Kaggle did not confirm one kernel URL owned by the configured account"
        )
    identities = {f"{username}/{slug}" for username, slug in matches}
    if len(identities) != 1 or any(username != owner for username, _ in matches):
        raise RuntimeError(
            "Kaggle did not confirm one kernel URL owned by the configured account"
        )
    return identities.pop()


def verify_uploaded_worker(
    executable: str,
    kernel: str,
    version: int,
    worker_hash: str,
    destination: Path,
) -> None:
    """Confirm an existing kernel by its latest uploaded source and metadata.

    Each shard has a unique rendered source hash. The exclusive controller does
    not reuse a slot until its results are collected, so matching that hash and
    owned identity binds this worker to the numerically confirmed push receipt.
    This does not assume version isolation from Kaggle's read APIs.
    """
    for attempt in range(3):
        try:
            kaggle_command(
                executable,
                ["pull", kernel, "-p", str(destination), "--metadata"],
            )
            break
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise
            time.sleep(2)
    metadata = json.loads(
        (destination / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("id") != kernel:
        raise ValueError("Source pull returned another kernel identity")
    if (
        metadata.get("is_private") is not True
        or metadata.get("enable_gpu") is not False
    ):
        raise ValueError("Source pull does not match the private CPU worker")
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not code_file:
        raise ValueError("Source pull did not identify its code file")
    source_path = (destination / code_file).resolve()
    try:
        source_path.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError("Source pull identified an unsafe code path") from exc
    # CLI pull writes text using the submitting OS's newline convention. Kaggle
    # executes the LF source that prepare_package uploads.
    source = source_path.read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != worker_hash:
        raise ValueError("Source pull does not match the submitted worker SHA-256")
    write_json(
        destination / "verified-submission.json",
        {
            "kernel": kernel,
            "push_confirmed_version": version,
            "worker_sha256": worker_hash,
            "identity_source": "latest_source_pull",
        },
    )


def record_submission(
    root: Path,
    state: dict,
    slot: dict,
    output: str,
    *,
    owner: str,
    executable: str,
    existing_kernel: str | None,
) -> None:
    """Durably record a push response before interpreting its version or URL."""
    response = redact(output)
    receipt = (
        root / "submissions" / f"shard-{slot['shard']:03d}-slot-{slot['slot']}.json"
    )
    write_json(
        receipt,
        {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "requested_kernel": slot["kernel"],
            "shard": slot["shard"],
            "worker_sha256": slot["worker_sha256"],
            "output": response,
        },
    )
    slot.update(
        submission=response,
        submission_receipt=receipt.relative_to(root).as_posix(),
        kernel_version=None,
    )
    write_json(root / "state.json", state)
    try:
        slot["kernel_version"] = pushed_version(response)
        write_json(root / "state.json", state)
        try:
            kernel = pushed_kernel(response, owner=owner)
            confirmation = "push_response_url"
        except RuntimeError:
            if existing_kernel is None:
                raise
            if not re.fullmatch(re.escape(owner) + r"/[A-Za-z0-9_-]+", existing_kernel):
                raise ValueError(
                    "Existing worker does not belong to the configured account"
                )
            # Source identity is stronger than the presentation of the CLI URL.
            verify_uploaded_worker(
                executable,
                existing_kernel,
                slot["kernel_version"],
                slot["worker_sha256"],
                receipt.with_suffix(".source"),
            )
            kernel = existing_kernel
            confirmation = "latest_source_pull"
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        slot["submission_parse_error"] = redact(str(exc))
        write_json(root / "state.json", state)
        log(
            root,
            f"Submission for shard {slot['shard']} needs reconciliation; response saved to {receipt.name}",
        )
        raise
    slot.update(
        kernel=kernel,
        phase="submitted",
        identity_confirmation=confirmation,
        submission_parse_error=None,
    )
    write_json(root / "state.json", state)
    log(root, f"Submitted shard {slot['shard']} on slot {slot['slot']}: {response}")


def load_worker(path: Path):
    spec = importlib.util.spec_from_file_location("ctboost_hpo_worker_plan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_package(
    worker: Path,
    destination: Path,
    *,
    owner: str,
    slot: int,
    shard: int,
    run_id: str,
    existing_kernel: str | None = None,
) -> str:
    source = worker.read_text(encoding="utf-8")
    source, replaced = re.subn(
        r"(?m)^SHARD_INDEX = 0$", f"SHARD_INDEX = {shard}", source
    )
    if replaced != 1:
        raise ValueError(
            "Worker must have exactly one standalone SHARD_INDEX = 0 constant"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "worker.py").write_text(source, encoding="utf-8", newline="\n")
    kernel = existing_kernel or f"{owner}/ctboost-0158-lite-hpo25-{run_id}-w{slot}"
    if not re.fullmatch(re.escape(owner) + r"/[A-Za-z0-9_-]+", kernel):
        raise ValueError("Worker kernel must belong to the configured account")
    slug = kernel.split("/", 1)[1]
    # Kaggle derives a newly created kernel's slug from the title even when id
    # supplies another slug. Match both, and retain the returned id on slot reuse.
    title = slug.replace("-", " ")
    write_json(
        destination / "kernel-metadata.json",
        {
            "id": kernel,
            "title": title,
            "code_file": "worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    return kernel


def read_download_manifest(
    destination: Path, shard: int, *, worker=None, worker_hash: str | None = None
) -> tuple[Path, dict]:
    if worker is None:
        worker = load_worker(Path(__file__).with_name("kaggle_hpo_worker.py"))
    identity = worker.shard_spec(shard)
    identity.update(
        ctboost_version=worker.CTBOOST_VERSION,
        benchmark_name=worker.BENCHMARK_NAME,
        tabarena_commit=worker.TABARENA_COMMIT,
        portfolio_200_sha256=worker.PORTFOLIO_200_SHA256,
    )
    if worker_hash is not None:
        identity["worker_sha256"] = worker_hash
    manifests = list(destination.rglob("artifacts/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"Shard {shard}: expected one manifest, found {len(manifests)}"
        )
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise ValueError(f"Shard {shard}: output identity mismatch for {key}")
    return manifests[0], manifest


def validate_archive(manifest_path: Path, manifest: dict, shard: int) -> None:
    record = manifest["workspace_archive"]
    relative = Path(record["path"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in record["path"]
        or ":" in record["path"]
    ):
        raise ValueError("Unsafe archive path in manifest")
    archive = manifest_path.parent.parent / relative
    if (
        archive.stat().st_size != record["size_bytes"]
        or file_hash(archive) != record["sha256"]
    ):
        raise ValueError(f"Shard {shard}: archive checksum mismatch")


def validate_download(
    destination: Path, shard: int, *, worker=None, worker_hash: str | None = None
) -> dict:
    manifest_path, manifest = read_download_manifest(
        destination, shard, worker=worker, worker_hash=worker_hash
    )
    if manifest.get("status") != "complete":
        raise ValueError(f"Shard {shard} incomplete: {manifest.get('fatal_error')}")
    expected = manifest.get("expected_parent_results_in_shard")
    if not expected or manifest.get("result_file_count") != expected:
        raise ValueError(f"Shard {shard}: result count mismatch")
    validate_archive(manifest_path, manifest, shard)
    return {
        "shard_index": shard,
        "manifest": str(manifest_path.relative_to(destination)),
        "manifest_sha256": file_hash(manifest_path),
        "result_count": expected,
    }


def retry_openml_setup_failure(
    root: Path, state: dict, slot: dict, destination: Path, *, worker
) -> bool:
    """Retry one verified pre-training 503 without changing the frozen worker.

    Evidence is archived before queue state changes. An interrupted archive or
    state write leaves the controller stopped rather than risking another push.
    """
    shard = slot["shard"]
    history = state.get("retry_history", [])
    if not isinstance(history, list) or any(
        not isinstance(entry, dict) or str(entry.get("shard")) == str(shard)
        for entry in history
    ):
        return False
    if str(shard) in state["completed"] or shard in state["pending"]:
        return False
    try:
        manifest_path, manifest = read_download_manifest(
            destination, shard, worker=worker, worker_hash=slot["worker_sha256"]
        )
        validation = manifest["validation"]
        if (
            not isinstance(validation, dict)
            or manifest.get("status") != "incomplete"
            or not re.match(
                r"^OpenMLServerError: Unexpected server error when calling "
                r"https://www\.openml\.org/api/v1/xml/data/qualities/[0-9]+\. "
                r"Please contact the developers!\nStatus code: 503\n",
                manifest.get("fatal_error") or "",
            )
            or manifest.get("result_file_count") != 0
            or manifest.get("result_files") != []
            or manifest.get("failures") != []
            or "benchmark_exit_code" not in manifest
            or manifest["benchmark_exit_code"] is not None
            or "run_commands" in manifest
            or "generated_job_json" in manifest
            or validation.get("bag_children_verified") != 0
            or validation.get("valid_result_count") != 0
            or validation.get("invalid_results") != []
            or validation.get("missing_datasets") != sorted(manifest["datasets"])
        ):
            return False
        validate_archive(manifest_path, manifest, shard)
    except (ValueError, KeyError, OSError, TypeError):
        return False

    version = slot["kernel_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("Cannot retry without a confirmed failed kernel version")
    root = root.resolve()
    receipt = (root / slot["submission_receipt"]).resolve()
    destination = destination.resolve()
    backup = root / "failed-attempts" / f"s{shard:03d}-v{version}-openml503"
    if not all(
        path.is_relative_to(root) and path != root
        for path in (receipt, destination, backup.resolve())
    ):
        raise ValueError("Retry evidence must stay inside the run directory")
    submission = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        submission.get("requested_kernel") != slot["kernel"]
        or submission.get("shard") != shard
        or submission.get("worker_sha256") != slot["worker_sha256"]
        or pushed_version(submission.get("output", "")) != version
    ):
        raise ValueError("Retry receipt does not match the failed attempt")
    record = {
        "shard": shard,
        "retried_at": datetime.now(timezone.utc).isoformat(),
        "reason": manifest["fatal_error"],
        "training_started": False,
        "previous_attempt": backup.relative_to(root).as_posix(),
        "kernel": slot["kernel"],
        "kernel_version": version,
        "worker_sha256": slot["worker_sha256"],
        "manifest_sha256": file_hash(manifest_path),
        "submission_receipt_sha256": file_hash(receipt),
    }
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(receipt, backup / "submission.json")
    write_json(backup / "retry.json", {**record, "previous_slot": dict(slot)})
    destination.rename(backup / "download")
    state["retry_history"] = [*history, record]
    state["pending"].insert(0, shard)
    slot.update(shard=None, phase="idle", collection_errors=0)
    write_json(root / "state.json", state)
    log(root, f"Requeued shard {shard} once after verified OpenML 503 before training")
    return True


def run_controller(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    worker_source = Path(__file__).with_name("kaggle_hpo_worker.py")
    worker = root / "worker_template.py"
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["worker_sha256"] != file_hash(worker) or state["owner"] != args.owner:
            raise ValueError("Run identity changed; use a new output directory")
        if len(state["slots"]) != args.slots:
            raise ValueError("Resume with the original number of slots")
    else:
        worker.write_text(
            worker_source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )
        state = {
            "version": "0.1.58",
            "owner": args.owner,
            "run_id": hashlib.sha256(str(root).encode()).hexdigest()[:8],
            "worker_sha256": file_hash(worker),
            "shard_count": SHARD_COUNT,
            "pending": list(range(SHARD_COUNT)),
            "completed": {},
            "failed": {},
            "slots": [{"slot": slot, "shard": None} for slot in range(args.slots)],
        }
        write_json(state_path, state)
    if args.prepare_only:
        for slot in state["slots"]:
            prepare_package(
                worker,
                root / "packages" / str(slot["slot"]),
                owner=args.owner,
                slot=slot["slot"],
                shard=slot["slot"],
                run_id=state["run_id"],
            )
        log(root, f"Prepared {SHARD_COUNT} shards; no jobs submitted")
        return
    help_text = kaggle_command(args.kaggle, ["output", "--help"])
    if "--file-pattern" not in help_text:
        raise RuntimeError(
            "This controller requires Kaggle CLI 2.2.0 or a compatible CLI with --file-pattern"
        )
    worker_plan = load_worker(worker)
    if state["failed"]:
        raise RuntimeError(
            "This run has failed shards; inspect state.json before resuming"
        )
    if any(slot.get("phase") == "submitting" for slot in state["slots"]):
        raise RuntimeError(
            "An interrupted submission needs reconciliation with Kaggle before resuming"
        )
    log(root, f"Controller running: {len(state['completed'])}/{SHARD_COUNT} collected")
    while len(state["completed"]) + len(state["failed"]) < SHARD_COUNT:
        for slot in state["slots"]:
            shard = slot["shard"]
            if shard is not None:
                status = None
                try:
                    status_text = kaggle_command(
                        args.kaggle, ["status", slot["kernel"]]
                    )
                    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status_text)
                    if not match:
                        raise RuntimeError(f"Unrecognized status: {status_text}")
                    status = match.group(1)
                    if status in ACTIVE_STATUSES:
                        continue
                    destination = root / "shards" / f"s{shard:03d}"
                    destination.mkdir(parents=True, exist_ok=True)
                    kaggle_command(
                        args.kaggle,
                        [
                            "output",
                            f"{slot['kernel']}/{slot['kernel_version']}",
                            "-p",
                            str(destination),
                            "-o",
                            "--file-pattern",
                            r"(artifacts/.*|\.log$)",
                        ],
                        timeout=1800,
                    )
                    if status != "COMPLETE":
                        raise ValueError(
                            f"Kaggle worker ended with {status}; diagnostics downloaded"
                        )
                    state["completed"][str(shard)] = validate_download(
                        destination,
                        shard,
                        worker=worker_plan,
                        worker_hash=slot["worker_sha256"],
                    )
                    slot["shard"] = None
                    slot["phase"] = "idle"
                    slot["collection_errors"] = 0
                    write_json(state_path, state)
                    log(
                        root,
                        f"Collected shard {shard}: {len(state['completed'])}/{SHARD_COUNT}",
                    )
                except (subprocess.TimeoutExpired, RuntimeError) as exc:
                    log(
                        root,
                        f"Will retry status/download for shard {shard}: {redact(str(exc))}",
                    )
                    continue
                except (ValueError, KeyError, OSError) as exc:
                    if status == "COMPLETE" and retry_openml_setup_failure(
                        root, state, slot, destination, worker=worker_plan
                    ):
                        pass
                    else:
                        slot["collection_errors"] = slot.get("collection_errors", 0) + 1
                        if slot["collection_errors"] < 3:
                            write_json(state_path, state)
                            log(
                                root,
                                f"Retrying artifact download for shard {shard}: {redact(str(exc))}",
                            )
                            continue
                        state["failed"][str(shard)] = redact(str(exc))
                        slot["shard"] = None
                        slot["phase"] = "failed"
                        write_json(state_path, state)
                        log(
                            root,
                            f"Shard {shard} failed: {redact(str(exc))}; stopping new submissions",
                        )
            if slot["shard"] is None and state["pending"] and not state["failed"]:
                shard = state["pending"][0]
                package = root / "packages" / str(slot["slot"])
                existing_kernel = slot.get("kernel")
                kernel = prepare_package(
                    worker,
                    package,
                    owner=args.owner,
                    slot=slot["slot"],
                    shard=shard,
                    run_id=state["run_id"],
                    existing_kernel=existing_kernel,
                )
                slot.update(
                    shard=shard,
                    kernel=kernel,
                    phase="submitting",
                    collection_errors=0,
                    worker_sha256=file_hash(package / "worker.py"),
                    previous_kernel_version=slot.get("kernel_version"),
                    kernel_version=None,
                    submission=None,
                    submission_receipt=None,
                    submission_parse_error=None,
                    identity_confirmation=None,
                )
                state["pending"].pop(0)
                write_json(state_path, state)
                # An ambiguous network failure is left as 'submitting', preventing duplicate
                # execution after restart. Check the remote version before resolving it.
                result = kaggle_command(
                    args.kaggle, ["push", "-p", str(package), "-t", "43200"]
                )
                record_submission(
                    root,
                    state,
                    slot,
                    result,
                    owner=args.owner,
                    executable=args.kaggle,
                    existing_kernel=existing_kernel,
                )
        if state["failed"] and all(slot["shard"] is None for slot in state["slots"]):
            raise RuntimeError(f"Stopped after failed shards: {state['failed']}")
        if len(state["completed"]) == SHARD_COUNT:
            break
        checked_at = datetime.now(timezone.utc)
        write_json(
            root / "progress.json",
            {
                "updated_at": checked_at.isoformat(),
                "poll_seconds": args.poll_seconds,
                "next_check_at": (
                    checked_at + timedelta(seconds=args.poll_seconds)
                ).isoformat(),
                "completed": len(state["completed"]),
                "total": SHARD_COUNT,
                "active": [
                    slot["shard"]
                    for slot in state["slots"]
                    if slot["shard"] is not None
                ],
                "failed": state["failed"],
            },
        )
        time.sleep(args.poll_seconds)
    log(root, "All 156 shards downloaded and checksummed; ready for evaluation")
    write_json(
        root / "progress.json",
        {
            "status": "complete",
            "completed": SHARD_COUNT,
            "total": SHARD_COUNT,
            "active": [],
            "failed": {},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kaggle", default=shutil.which("kaggle") or "kaggle")
    parser.add_argument("--slots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 15:
        parser.error("--poll-seconds must be at least 15")
    # Restrict identifiers so generated kernel paths cannot select another resource type.
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.owner):
        parser.error("Invalid Kaggle owner")
    with controller_lock(args.output_root.resolve()):
        run_controller(args)


if __name__ == "__main__":
    main()
