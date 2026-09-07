"""Versioned remote queue and source packaging for the public 0.1.60 HPO plan.

Submission receipts, canonical kernel confirmation, status calls, and downloads
reuse the existing transport. Parent fitting belongs to hpo_0160.py exclusively.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from benchmarks.tabarena import kaggle_hpo as transport
from benchmarks.tabarena import kaggle_hpo_worker_0160 as worker

ACTIVE_STATUSES = transport.ACTIVE_STATUSES | {"CANCEL_REQUESTED", "NEW_SCRIPT"}
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


class QuotaAdmissionRejected(RuntimeError):
    """The server explicitly rejected admission before confirming any version."""


def explicit_quota_rejection(message):
    # Legacy command errors retain at most 1,500 output characters. Only a
    # shorter, complete response can establish that no version was confirmed.
    return (
        re.fullmatch(r"Kaggle push failed \(\d+\): [\s\S]{0,1499}", message) is not None
        and not re.search(
            r"Kernel version\s+\d+|successfully pushed", message, re.IGNORECASE
        )
        and re.search(
            r"maximum number of (?:concurrent )?(?:(?:CPU|GPU|TPU) )?"
            r"(?:kernel|notebook|session)s?(?: allowed)? (?:has been )?(?:reached|exceeded)"
            r"|too many (?:concurrent |active |running )?(?:kernel|notebook|session)s",
            message,
            re.IGNORECASE,
        )
        is not None
    )


def kaggle_command(executable, arguments, *, timeout=300):
    # The legacy transport decodes UTF-8. Make the CLI emit it as well, including
    # notebook titles outside the Windows console code page.
    previous = os.environ.get("PYTHONIOENCODING")
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        return transport.kaggle_command(executable, arguments, timeout=timeout)
    finally:
        if previous is None:
            os.environ.pop("PYTHONIOENCODING", None)
        else:
            os.environ["PYTHONIOENCODING"] = previous


def shared_worker():
    from benchmarks.tabarena import hpo_0160

    return hpo_0160


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def render_worker(template, payload, shard):
    source, replaced = re.subn(
        r'(?m)^PAYLOAD_BASE64 = ""$', 'PAYLOAD_BASE64 = "' + payload + '"', template
    )
    if replaced != 1:
        raise ValueError("Remote template has no unique payload placeholder")
    source, replaced = re.subn(
        r"(?m)^SHARD_INDEX = 0$", f"SHARD_INDEX = {shard}", source
    )
    if replaced != 1:
        raise ValueError("Remote template has no unique shard placeholder")
    return source


def make_payload(plan, registration, *, source_root=ROOT):
    shared_worker().validate_plan(plan, verify_sources=True)
    plan_sha = worker.json_hash(plan)
    if (
        registration.get("plan_sha256") != plan_sha
        or registration.get("repository") != "captnmarkus/ctboost"
        or not re.fullmatch(r"[0-9a-f]{40}", registration.get("commit", ""))
    ):
        raise ValueError("Registration does not bind this public plan")
    worker.safe_relative(registration["plan_path"])
    files = {
        "plan.json": canonical_bytes(plan),
        "registration.json": canonical_bytes(registration),
    }
    for relative, expected in plan["source_sha256"].items():
        path = source_root.joinpath(*worker.safe_relative(relative).parts)
        if path.is_symlink():
            raise ValueError("Linked source is not allowed in the remote package")
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Registered source changed: {relative}")
        files[relative] = data
    bundle = {
        "schema_version": 1,
        "plan_sha256": plan_sha,
        "registration_sha256": worker.json_hash(registration),
        "shards": worker.shard_specs(plan),
        "controller_sha256": transport.file_hash(Path(__file__)),
        "legacy_transport_sha256": transport.file_hash(Path(transport.__file__)),
        "worker_template_sha256": transport.file_hash(Path(worker.__file__)),
        "files_sha256": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in sorted(files.items())
        },
        "allocation": {
            "parents_per_kernel": 2,
            "initial_transport_bundle_parents": 1,
            "parents_per_bundle": 12,
            "cpu_per_parent": 2,
            "memory_limit_gb": 8,
            "physical_free_reserve_gb": 4,
            "parent_wall_limit_seconds": 4500,
            "kernel_timeout_seconds": 43200,
            "fit_retries": 0,
        },
    }
    files["bundle.json"] = canonical_bytes(bundle)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed, tarfile.open(
        fileobj=compressed, mode="w"
    ) as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return base64.b64encode(output.getvalue()).decode(), bundle


def prepare_run(root, plan_path, registration_path, *, owner, slots=5):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner) or not 1 <= slots <= 5:
        raise ValueError("Invalid account or slot count")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    payload, bundle = make_payload(plan, registration)
    template = Path(worker.__file__).read_text(encoding="utf-8")
    template_with_payload = render_worker(template, payload, 0)
    worker_hashes = {
        str(spec["shard_index"]): hashlib.sha256(
            render_worker(template, payload, spec["shard_index"]).encode()
        ).hexdigest()
        for spec in bundle["shards"]
    }
    execution = {
        **bundle,
        "owner": owner,
        "slots": slots,
        "payload_sha256": hashlib.sha256(base64.b64decode(payload)).hexdigest(),
        "rendered_worker_sha256": worker_hashes,
    }
    root.mkdir(parents=True, exist_ok=True)
    execution_path = root / "execution.json"
    if execution_path.exists():
        if json.loads(execution_path.read_text()) != execution:
            raise ValueError(
                "Remote execution identity changed; use its original sources and inputs"
            )
        if transport.file_hash(root / "worker_template.py") != worker_hashes["0"]:
            raise ValueError("Frozen remote worker template changed")
    else:
        if (root / "state.json").exists():
            raise ValueError("Existing queue has no frozen execution manifest")
        (root / "worker_template.py").write_text(
            template_with_payload, encoding="utf-8", newline="\n"
        )
        transport.write_json(root / "plan.json", plan)
        transport.write_json(root / "registration.json", registration)
        transport.write_json(execution_path, execution)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["execution_sha256"] != transport.file_hash(execution_path):
            raise ValueError("Queue is not bound to this execution manifest")
    else:
        state = {
            "version": "0.1.60",
            "owner": owner,
            "run_id": bundle["plan_sha256"][:12],
            "execution_sha256": transport.file_hash(execution_path),
            "pending": list(range(len(bundle["shards"]))),
            "completed": {},
            "failed": {},
            "slots": [
                {"slot": index, "shard": None, "phase": "idle"}
                for index in range(slots)
            ],
        }
        transport.write_json(state_path, state)
    return plan, execution, state


def prepare_package(root, state, execution, slot, shard):
    kernel = (
        slot.get("kernel")
        or f"{state['owner']}/ctboost-0160-hpo200-{state['run_id']}-w{slot['slot']}"
    )
    destination = root / "packages" / str(slot["slot"])
    transport.prepare_package(
        root / "worker_template.py",
        destination,
        owner=state["owner"],
        slot=slot["slot"],
        shard=shard,
        run_id=state["run_id"],
        existing_kernel=kernel,
    )
    actual = transport.file_hash(destination / "worker.py")
    if actual != execution["rendered_worker_sha256"][str(shard)]:
        raise ValueError("Rendered worker differs from its frozen hash")
    return destination, kernel, actual


def extract_verified_archive(manifest_path, manifest, destination):
    transport.validate_archive(manifest_path, manifest, manifest["shard_index"])
    expected = {row["path"]: row for row in manifest["workspace_files"]}
    if len(expected) != len(manifest["workspace_files"]):
        raise ValueError("Duplicate archived file identity")
    for name in expected:
        worker.safe_relative(name)
        if not name.startswith("workspace/output/"):
            raise ValueError("Archive contains an unexpected workspace path")
    archive_path = manifest_path.parent.parent / manifest["workspace_archive"]["path"]
    observed = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name not in expected
                or member.name in observed
            ):
                raise ValueError("Unexpected, linked, or duplicate archive member")
            metadata = expected[member.name]
            if member.size != metadata["size_bytes"]:
                raise ValueError("Archived size differs from manifest")
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
                raise ValueError("Archived file checksum mismatch")
            target = destination.joinpath(*worker.safe_relative(member.name).parts)
            if target.exists():
                if (
                    target.is_symlink()
                    or transport.file_hash(target) != metadata["sha256"]
                ):
                    raise ValueError("Previously extracted evidence changed")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            observed.add(member.name)
    if observed != set(expected):
        raise ValueError("Archive does not contain the entire file inventory")


def validate_download(destination, shard, *, plan, execution, worker_hash):
    proxy = SimpleNamespace(
        shard_spec=lambda index: dict(execution["shards"][index]),
        CTBOOST_VERSION=worker.CTBOOST_VERSION,
        BENCHMARK_NAME=worker.BENCHMARK_NAME,
        TABARENA_COMMIT=worker.TABARENA_COMMIT,
        PORTFOLIO_200_SHA256=worker.PORTFOLIO_200_SHA256,
    )
    manifest_path, manifest = transport.read_download_manifest(
        destination, shard, worker=proxy, worker_hash=worker_hash
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("plan_sha256") != execution["plan_sha256"]
        or manifest.get("package_sha256") != execution["payload_sha256"]
    ):
        raise ValueError("Remote bundle did not complete with its registered identity")
    records = manifest["parents"]
    expected_ids = execution["shards"][shard]["parent_ids"]
    if (
        len(records) != len(expected_ids)
        or {row["parent_id"] for row in records} != set(expected_ids)
        or manifest.get("terminal_parent_count") != len(expected_ids)
    ):
        raise ValueError("Remote bundle is missing parent execution evidence")
    extracted = destination / "verified"
    extract_verified_archive(manifest_path, manifest, extracted)
    output = extracted / "workspace/output"
    parents = {row["parent_id"]: row for row in plan["parents"]}
    complete, failed = [], []
    for record in records:
        parent = parents[record["parent_id"]]
        expected_manifest, expected_raw = worker.parent_paths(output, parent)
        if (
            record.get("manifest_path")
            != expected_manifest.relative_to(output).as_posix()
            or not expected_manifest.is_file()
        ):
            raise ValueError(
                f"Missing shared-worker manifest for {record['parent_id']}"
            )
        child = json.loads(expected_manifest.read_text())
        if (
            transport.file_hash(expected_manifest) != record.get("manifest_sha256")
            or child.get("parent") != parent
            or child.get("plan_sha256") != execution["plan_sha256"]
            or child.get("host") != "kaggle"
            or child.get("resources") != plan["resources"]
            or child.get("registration", {}).get("registration_sha256")
            != execution["registration_sha256"]
            or child.get("registration", {}).get("plan_sha256")
            != execution["plan_sha256"]
            or worker.json_hash(child.get("registration", {}).get("receipt"))
            != execution["registration_sha256"]
        ):
            raise ValueError("Parent worker manifest identity mismatch")
        receipt = output / "scheduler" / f"{record['parent_id']}.json"
        if not receipt.is_file() or json.loads(receipt.read_text()) != record:
            raise ValueError("Parent execution receipt is missing or changed")
        if record["status"] == "complete":
            if (
                child.get("status") != "complete"
                or record["exit_code"] != 0
                or record.get("raw_path") != expected_raw.relative_to(output).as_posix()
            ):
                raise ValueError("Successful parent lacks its complete raw result")
            raw_hash = transport.file_hash(expected_raw)
            if (
                raw_hash != record.get("raw_sha256")
                or child.get("validation", {}).get("raw_sha256") != raw_hash
            ):
                raise ValueError("Successful parent result checksum differs")
            validated = shared_worker().validate_parent_result(
                expected_raw, parent, execution["plan_sha256"]
            )
            if validated["runtime_sha256"] != worker.json_hash(child["runtime"]):
                raise ValueError("Raw result and parent runtime provenance differ")
            complete.append(parent["parent_id"])
        elif record["status"] == "failed" and record["exit_code"] != 0:
            failed.append(
                {
                    "parent_id": parent["parent_id"],
                    "exit_code": record["exit_code"],
                    "worker_status": child.get("status"),
                    "failure": record.get("failure"),
                }
            )
        else:
            raise ValueError("Parent has ambiguous terminal execution status")
    if manifest.get("result_file_count") != len(complete) or manifest.get(
        "failed_parent_count"
    ) != len(failed):
        raise ValueError("Terminal parent counts differ from archived evidence")
    return {
        "shard_index": shard,
        "manifest": manifest_path.relative_to(destination).as_posix(),
        "manifest_sha256": transport.file_hash(manifest_path),
        "completed_parent_ids": complete,
        "failed_parents": failed,
        "terminal_parent_count": len(records),
    }


def collect_slot(root, state, slot, *, executable, plan, execution):
    shard = slot["shard"]
    status_text = kaggle_command(executable, ["status", slot["kernel"]])
    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status_text)
    if not match or match.group(1) not in ACTIVE_STATUSES | TERMINAL_STATUSES:
        raise RuntimeError("Unrecognized remote status")
    if match.group(1) in ACTIVE_STATUSES:
        return False
    destination = root / "shards" / f"s{shard:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    kaggle_command(
        executable,
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
    if match.group(1) != "COMPLETE":
        raise ValueError(
            f"Remote kernel ended {match.group(1)}; downloaded diagnostics must be reconciled"
        )
    result = validate_download(
        destination,
        shard,
        plan=plan,
        execution=execution,
        worker_hash=slot["worker_sha256"],
    )
    state["completed"][str(shard)] = result
    slot.update(shard=None, phase="idle", collection_errors=0)
    transport.write_json(root / "state.json", state)
    return True


def submit_slot(root, state, slot, *, executable, execution):
    shard = state["pending"][0]
    previous_slot = dict(slot)
    existing_kernel = slot.get("kernel")
    package, kernel, source_hash = prepare_package(root, state, execution, slot, shard)
    slot.update(
        shard=shard,
        kernel=kernel,
        phase="submitting",
        worker_sha256=source_hash,
        kernel_version=None,
        collection_errors=0,
    )
    state["pending"].pop(0)
    transport.write_json(root / "state.json", state)
    try:
        response = kaggle_command(
            executable, ["push", "-p", str(package), "-t", "43200"]
        )
    except RuntimeError as exc:
        if not explicit_quota_rejection(str(exc)):
            raise
        history = state.setdefault("admission_rejections", [])
        receipt = root / "admission_rejections" / f"rejected-{len(history):06d}.json"
        transport.write_json(
            receipt,
            {
                "shard": shard,
                "slot": slot["slot"],
                "kernel": kernel,
                "worker_sha256": source_hash,
                "reason": transport.redact(str(exc)),
                "no_confirmed_kernel_version": True,
                "received_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        history.append(
            {
                "receipt": receipt.relative_to(root).as_posix(),
                "sha256": transport.file_hash(receipt),
            }
        )
        slot.clear()
        slot.update(previous_slot)
        state["pending"].insert(0, shard)
        transport.write_json(root / "state.json", state)
        raise QuotaAdmissionRejected(
            "Server quota rejected admission; unstarted assignment preserved"
        ) from exc
    transport.record_submission(
        root,
        state,
        slot,
        response,
        owner=state["owner"],
        executable=executable,
        existing_kernel=existing_kernel,
    )


def occupied_kernels(executable, owner, state):
    """Read known activity; disclose unavailable status and defer to server admission."""
    recent = set()
    not_queried = {}
    inventory_count = 0
    cutoff = time.time() - 24 * 3600
    durable = {
        slot["kernel"]
        for slot in state["slots"]
        if slot.get("phase") in {"submitted", "submitting", "needs_reconciliation"}
        and slot.get("kernel")
    }
    unresolved = []
    unresolved_errors = {}
    for page in range(1, 101):
        output = kaggle_command(
            executable,
            ["list", "--mine", "--csv", "--page-size", "100", "--page", str(page)],
        )
        lines = output.replace("\r\r\n", "\n").splitlines()
        # The CLI can append its version notice after CSV on combined streams.
        version_notice = re.compile(
            r"Warning: Looks like you['’]re using an outdated `kaggle` version "
            r"\(installed: [0-9A-Za-z.+-]+\), please consider upgrading to the latest "
            r"version \([0-9A-Za-z.+-]+\)"
        )
        while lines and (not lines[-1].strip() or version_notice.fullmatch(lines[-1])):
            lines.pop()
        if lines and lines[-1].strip() == "Not found":
            break
        header = "ref,title,author,lastRunTime,totalVotes"
        headers = [index for index, line in enumerate(lines) if line == header]
        if len(headers) != 1:
            raise RuntimeError("Could not verify owned-kernel admission inventory")
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[headers[0] :]))))
        if not rows:
            raise RuntimeError("Could not verify owned-kernel admission inventory")
        inventory_count += len(rows)
        for index, row in enumerate(rows):
            if row == {
                "ref": "",
                "title": "[Private Notebook]",
                "author": "",
                "lastRunTime": "2010-04-01 00:00:00",
                "totalVotes": "0",
            }:
                # This exact API placeholder has no resolvable identity. Keep it
                # visible without inventing a permanent occupied session.
                unresolved.append(f"inaccessible-private-notebook:p{page}:r{index}")
            elif (
                set(row) != set(header.split(","))
                or any(value is None for value in row.values())
                or not re.fullmatch(
                    re.escape(owner) + r"/[A-Za-z0-9_-]+", row.get("ref", "")
                )
            ):
                raise RuntimeError("Could not verify owned-kernel admission inventory")
            else:
                try:
                    last_run = datetime.fromisoformat(
                        row["lastRunTime"].replace("Z", "+00:00")
                    )
                    if last_run.tzinfo is None:
                        last_run = last_run.replace(tzinfo=timezone.utc)
                    if last_run.timestamp() >= cutoff:
                        recent.add(row["ref"])
                    else:
                        not_queried[row["ref"]] = "last_run_before_recent_24h_window"
                except ValueError:
                    not_queried[row["ref"]] = "unrecognized_last_run_time"
        if len(rows) < 100:
            break
    else:
        raise RuntimeError("Kernel inventory exceeded its bounded pagination")
    active = set()
    query_refs = recent | durable
    not_queried = {
        ref: reason for ref, reason in not_queried.items() if ref not in query_refs
    }
    for ref in sorted(query_refs):
        try:
            status = kaggle_command(executable, ["status", ref], timeout=15)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            # Read failures never authorize retrying a submitted job. They only
            # make this account-capacity estimate incomplete; server admission
            # still governs new submissions and durable slots remain occupied.
            unresolved.append(ref)
            unresolved_errors[ref] = transport.redact(str(exc))[-500:]
            continue
        match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", status)
        if not match or match.group(1) not in ACTIVE_STATUSES | TERMINAL_STATUSES:
            raise RuntimeError("Could not verify kernel status for admission")
        if match.group(1) in ACTIVE_STATUSES:
            active.add(ref)
    active.update(durable)
    state["admission_unresolved"] = unresolved
    state["admission_status_errors"] = unresolved_errors
    state["admission_not_queried"] = not_queried
    state["admission_inventory_count"] = inventory_count
    state["admission_queried_count"] = len(query_refs)
    if unresolved or not_queried:
        print(
            json.dumps(
                {
                    "unresolved_session_status": unresolved,
                    "status_errors": unresolved_errors,
                    "inventory_count": inventory_count,
                    "queried_status_count": len(query_refs),
                    "not_queried_count": len(not_queried),
                    "capacity_estimate_requires_server_admission": True,
                }
            ),
            flush=True,
        )
    return active


def run_controller(args):
    root = args.output_root.resolve()
    with transport.controller_lock(root):
        plan, execution, state = prepare_run(
            root, args.plan, args.registration, owner=args.owner, slots=args.slots
        )
        if args.prepare_only:
            for slot, shard in zip(state["slots"], state["pending"]):
                if slot["shard"] is None:
                    prepare_package(root, state, execution, slot, shard)
            transport.log(
                root,
                f"Prepared {len(execution['shards'])} fixed bundles; no submissions",
            )
            return
        if "--file-pattern" not in kaggle_command(args.kaggle, ["output", "--help"]):
            raise RuntimeError(
                "Kaggle CLI2.2.0 or compatible --file-pattern support is required"
            )
        while state["pending"] or any(
            slot.get("phase") == "submitted" for slot in state["slots"]
        ):
            occupied = None
            if state["pending"] and not (root / "PAUSE").exists():
                try:
                    occupied = occupied_kernels(args.kaggle, state["owner"], state)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    transport.log(
                        root, f"Admission postponed: {transport.redact(str(exc))}"
                    )
            for slot in state["slots"]:
                if slot.get("phase") in {"submitting", "needs_reconciliation"}:
                    continue
                if slot["shard"] is not None:
                    try:
                        completed_kernel = slot["kernel"]
                        collected = collect_slot(
                            root,
                            state,
                            slot,
                            executable=args.kaggle,
                            plan=plan,
                            execution=execution,
                        )
                        if collected and occupied is not None:
                            occupied.discard(completed_kernel)
                    except (RuntimeError, subprocess.TimeoutExpired) as exc:
                        transport.log(
                            root,
                            f"Transport retry for shard{slot['shard']}: {transport.redact(str(exc))}",
                        )
                        continue
                    except (ValueError, KeyError, OSError) as exc:
                        slot["collection_errors"] = slot.get("collection_errors", 0) + 1
                        if slot["collection_errors"] >= 3:
                            state["failed"][str(slot["shard"])] = transport.redact(
                                str(exc)
                            )
                            slot.update(phase="needs_reconciliation")
                        transport.write_json(root / "state.json", state)
                        continue
                if (
                    slot["shard"] is None
                    and state["pending"]
                    and not (root / "PAUSE").exists()
                    and occupied is not None
                    and len(occupied) < 5
                ):
                    try:
                        submit_slot(
                            root,
                            state,
                            slot,
                            executable=args.kaggle,
                            execution=execution,
                        )
                    except QuotaAdmissionRejected as exc:
                        occupied = None
                        transport.log(root, str(exc))
                        continue
                    except (
                        RuntimeError,
                        ValueError,
                        OSError,
                        subprocess.TimeoutExpired,
                    ) as exc:
                        if slot.get("phase") != "submitting":
                            raise
                        # A lost push response may still represent a running job.
                        # Preserve that assignment and continue unrelated slots.
                        slot["phase"] = "needs_reconciliation"
                        state["failed"][str(slot["shard"])] = transport.redact(str(exc))
                        transport.write_json(root / "state.json", state)
                        transport.log(
                            root,
                            f"Slot {slot['slot']} needs reconciliation: {transport.redact(str(exc))}",
                        )
                    occupied.add(slot["kernel"])
            completed_parents = sum(
                len(row["completed_parent_ids"]) for row in state["completed"].values()
            )
            failed_parents = sum(
                len(row["failed_parents"]) for row in state["completed"].values()
            )
            active = [
                slot["shard"]
                for slot in state["slots"]
                if slot.get("phase") == "submitted"
            ]
            transport.write_json(
                root / "progress.json",
                {
                    "completed_bundles": len(state["completed"]),
                    "total_bundles": len(execution["shards"]),
                    "completed_parents": completed_parents,
                    "failed_parents": failed_parents,
                    "active": active,
                    "queued_bundles": len(state["pending"]),
                    "needs_reconciliation": state["failed"],
                    "admission_unresolved": state.get("admission_unresolved", []),
                    "admission_status_errors": state.get("admission_status_errors", {}),
                    "admission_not_queried": state.get("admission_not_queried", {}),
                    "admission_inventory_count": state.get("admission_inventory_count"),
                    "admission_queried_count": state.get("admission_queried_count"),
                    "paused": (root / "PAUSE").exists(),
                },
            )
            if state["pending"] and all(
                slot.get("phase") in {"submitting", "needs_reconciliation"}
                for slot in state["slots"]
            ):
                raise RuntimeError(
                    "All slots require reconciliation; execution evidence preserved"
                )
            if state["pending"] or active:
                time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slots", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--kaggle", default=shutil.which("kaggle") or "kaggle")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 15:
        parser.error("poll-seconds must be at least15")
    run_controller(args)


if __name__ == "__main__":
    main()
