"""Offline transport checks; no Kaggle calls, installations, or model fits."""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from benchmarks.tabarena import kaggle_hpo_0160 as controller
from benchmarks.tabarena import kaggle_hpo_worker_0160 as worker


@pytest.fixture(autouse=True)
def fixed_capacity_clock(monkeypatch):
    instant = datetime(2026, 9, 7, 12, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr(controller.time, "time", lambda: instant)


def fake_plan(count=27):
    return {
        "parents": [
            {
                "parent_id": f"openml-{index}-r0-f0-c000",
                "ordinal": index,
                "owner": "local" if index % 9 < 4 else "kaggle",
                "dataset": f"dataset-{index}",
                "task_id": index,
                "config_index": 0,
                "config_name": "CTBoost_c000",
                "repeat": 0,
                "fold": 0,
                "child_seeds": list(range(8)),
            }
            for index in range(count)
        ],
        "source_sha256": {},
        "resources": {
            "num_cpus": 2,
            "memory_limit_gb": 8,
            "parent_wall_limit_seconds": 4500,
        },
    }


def fake_registration(plan):
    return {
        "repository": "captnmarkus/ctboost",
        "commit": "a" * 40,
        "plan_path": "benchmarks/tabarena/registered-plan.json",
        "plan_sha256": worker.json_hash(plan),
        "registered_at_utc": "2026-09-07T00:00:00Z",
    }


@pytest.fixture
def stub_shared(monkeypatch):
    checks = []
    stub = SimpleNamespace(
        validate_plan=lambda plan, **kwargs: checks.append((plan, kwargs)),
        validate_parent_result=lambda *args: {
            "runtime_sha256": worker.json_hash({"synthetic": True})
        },
    )
    monkeypatch.setattr(controller, "shared_worker", lambda: stub)
    return stub


def test_fixed_partition_covers_every_remote_parent_once():
    plan = fake_plan(10251)
    specs = worker.shard_specs(plan)
    ids = [parent for spec in specs for parent in spec["parent_ids"]]
    assert len(specs) == 476
    assert len(ids) == len(set(ids)) == 5695
    assert sum(row["owner"] == "local" for row in plan["parents"]) == 4556
    assert ids == [
        row["parent_id"] for row in plan["parents"] if row["owner"] == "kaggle"
    ]
    assert len(specs[0]["parent_ids"]) == 1
    assert specs[0]["parent_ids"] == [plan["parents"][4]["parent_id"]]
    assert len(specs[-1]["parent_ids"]) == 6
    assert all(1 <= len(spec["parent_ids"]) <= 12 for spec in specs)
    assert all(spec["shard_count"] == 476 for spec in specs)
    assert sum(spec["expected_child_fits_in_shard"] for spec in specs) == 45560


@pytest.mark.parametrize("mutation", ["owner", "ordinal", "duplicate"])
def test_partition_rejects_assignment_changes(mutation):
    plan = fake_plan()
    if mutation == "duplicate":
        plan["parents"][1]["parent_id"] = plan["parents"][0]["parent_id"]
    else:
        plan["parents"][4][mutation] = "local" if mutation == "owner" else 5
    with pytest.raises(ValueError):
        worker.shard_specs(plan)


def test_payload_is_deterministic_and_checks_sources(
    tmp_path, stub_shared, monkeypatch
):
    plan = fake_plan()
    source = tmp_path / "benchmarks" / "example.py"
    source.parent.mkdir()
    source.write_bytes(b"example = 1\r\n")
    plan["source_sha256"] = {
        "benchmarks/example.py": hashlib.sha256(b"example = 1\n").hexdigest()
    }
    registration = fake_registration(plan)
    first = controller.make_payload(plan, registration, source_root=tmp_path)
    second = controller.make_payload(plan, registration, source_root=tmp_path)
    assert first == second
    payload, bundle = first
    assert bundle["allocation"]["initial_transport_bundle_parents"] == 1
    monkeypatch.setattr(worker, "PAYLOAD_BASE64", payload)
    package = tmp_path / "unpacked"
    actual_bundle, actual_plan = worker.unpack_payload(package)
    assert actual_bundle == bundle
    assert actual_plan == plan
    assert (package / "benchmarks/example.py").read_bytes() == b"example = 1\n"
    assert worker.unpack_payload(package, existing=True) == (bundle, plan)
    (package / "benchmarks/example.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Previously unpacked source changed"):
        worker.unpack_payload(package, existing=True)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Registered source changed"):
        controller.make_payload(plan, registration, source_root=tmp_path)


@pytest.mark.parametrize("path", ["../secret", "/secret", "C:/secret", "x\\y"])
def test_packaged_paths_cannot_escape(path):
    with pytest.raises(ValueError, match="Unsafe packaged path"):
        worker.safe_relative(path)


def prepared_run(tmp_path):
    plan = fake_plan()
    plan_path = tmp_path / "input-plan.json"
    registration_path = tmp_path / "input-registration.json"
    worker.write_json(plan_path, plan)
    worker.write_json(registration_path, fake_registration(plan))
    root = tmp_path / "queue"
    actual_plan, execution, state = controller.prepare_run(
        root, plan_path, registration_path, owner="maiernator", slots=5
    )
    return root, plan_path, registration_path, actual_plan, execution, state


def test_prepare_resume_and_rendered_worker_hashes_are_frozen(tmp_path, stub_shared):
    root, plan_path, registration_path, plan, execution, state = prepared_run(tmp_path)
    assert controller.prepare_run(
        root, plan_path, registration_path, owner="maiernator", slots=5
    ) == (plan, execution, state)
    assert execution["slots"] == 5
    assert execution["shards"][0]["expected_parent_results_in_shard"] == 1
    package, kernel, source_hash = controller.prepare_package(
        root, state, execution, state["slots"][0], 1
    )
    metadata = json.loads((package / "kernel-metadata.json").read_text())
    assert kernel.startswith("maiernator/ctboost-0160-hpo200-")
    assert metadata["id"] == kernel
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert source_hash == execution["rendered_worker_sha256"]["1"]
    assert "SHARD_INDEX = 1\n" in (package / "worker.py").read_text()
    (root / "worker_template.py").write_text("changed")
    with pytest.raises(ValueError, match="Frozen remote worker template changed"):
        controller.prepare_run(
            root, plan_path, registration_path, owner="maiernator", slots=5
        )


def completed_bundle(tmp_path, *, failed=False, child_change=None):
    plan = fake_plan()
    spec = worker.shard_specs(plan)[0]
    parent = plan["parents"][4]
    destination = tmp_path / "download"
    workspace = tmp_path / "remote-workspace"
    artifacts = destination / "artifacts"
    artifacts.mkdir(parents=True)
    output = workspace / "output"
    child_manifest, raw = worker.parent_paths(output, parent)
    child = {
        "status": "failed" if failed else "complete",
        "parent": parent,
        "plan_sha256": worker.json_hash(plan),
        "runtime": {"synthetic": True},
        "host": "kaggle",
        "resources": plan["resources"],
        "registration": {
            "registration_sha256": worker.json_hash(fake_registration(plan)),
            "plan_sha256": worker.json_hash(plan),
            "receipt": fake_registration(plan),
        },
    }
    if child_change:
        child.update(child_change)
    if not failed:
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"synthetic raw result, checked by mocked shared validator")
        child["validation"] = {"raw_sha256": worker.file_hash(raw)}
    worker.write_json(child_manifest, child)
    record = {
        "parent_id": parent["parent_id"],
        "status": child["status"],
        "exit_code": 3 if failed else 0,
        "worker_status": child["status"],
        "manifest_path": child_manifest.relative_to(output).as_posix(),
        "manifest_sha256": worker.file_hash(child_manifest),
    }
    if not failed:
        record.update(
            raw_path=raw.relative_to(output).as_posix(),
            raw_sha256=worker.file_hash(raw),
        )
    else:
        record["failure"] = "resource_limit_exceeded"
    worker.write_json(output / "scheduler" / f"{parent['parent_id']}.json", record)
    manifest = {
        **spec,
        "status": "complete",
        "parents": [record],
        "worker_sha256": "b" * 64,
        "ctboost_version": worker.CTBOOST_VERSION,
        "benchmark_name": worker.BENCHMARK_NAME,
        "tabarena_commit": worker.TABARENA_COMMIT,
        "portfolio_200_sha256": worker.PORTFOLIO_200_SHA256,
        "plan_sha256": worker.json_hash(plan),
        "package_sha256": "c" * 64,
    }
    worker.checkpoint(workspace, artifacts, manifest)
    execution = {
        "shards": worker.shard_specs(plan),
        "plan_sha256": worker.json_hash(plan),
        "payload_sha256": "c" * 64,
        "registration_sha256": worker.json_hash(fake_registration(plan)),
    }
    return destination, plan, execution, manifest


@pytest.mark.parametrize(
    "change",
    [
        {"host": "local"},
        {"resources": {"num_cpus": 4}},
        {"registration": {"registration_sha256": "changed"}},
    ],
)
def test_download_checks_host_resources_and_registration(tmp_path, stub_shared, change):
    destination, plan, execution, _manifest = completed_bundle(
        tmp_path, child_change=change
    )
    with pytest.raises(ValueError, match="Parent worker manifest identity mismatch"):
        controller.validate_download(
            destination, 0, plan=plan, execution=execution, worker_hash="b" * 64
        )


@pytest.mark.parametrize("failed", [False, True])
def test_archive_round_trip_preserves_terminal_parent_evidence(
    tmp_path, stub_shared, failed
):
    destination, plan, execution, _manifest = completed_bundle(tmp_path, failed=failed)
    result = controller.validate_download(
        destination, 0, plan=plan, execution=execution, worker_hash="b" * 64
    )
    assert result["terminal_parent_count"] == 1
    assert len(result["completed_parent_ids"]) == (0 if failed else 1)
    assert len(result["failed_parents"]) == (1 if failed else 0)
    # A repeated download/check validates existing evidence instead of overwriting it.
    assert (
        controller.validate_download(
            destination, 0, plan=plan, execution=execution, worker_hash="b" * 64
        )
        == result
    )


@pytest.mark.parametrize(
    "mutation", ["archive", "worker_identity", "parent_count", "runtime"]
)
def test_download_rejects_changed_or_ambiguous_evidence(
    tmp_path, stub_shared, mutation
):
    destination, plan, execution, manifest = completed_bundle(tmp_path)
    if mutation == "archive":
        archive = destination / manifest["workspace_archive"]["path"]
        archive.write_bytes(archive.read_bytes() + b"corruption")
    elif mutation == "worker_identity":
        manifest["worker_sha256"] = "d" * 64
    elif mutation == "parent_count":
        manifest["terminal_parent_count"] = 0
    else:
        stub_shared.validate_parent_result = lambda *args: {"runtime_sha256": "bad"}
    worker.write_json(destination / "artifacts/manifest.json", manifest)
    with pytest.raises(ValueError):
        controller.validate_download(
            destination, 0, plan=plan, execution=execution, worker_hash="b" * 64
        )


def test_ambiguous_submission_is_durable_and_never_requeued(
    tmp_path, stub_shared, monkeypatch
):
    root, _, _, _, execution, state = prepared_run(tmp_path)

    def lost_response(*args, **kwargs):
        raise RuntimeError("connection lost after push")

    monkeypatch.setattr(controller.transport, "kaggle_command", lost_response)
    with pytest.raises(RuntimeError, match="connection lost"):
        controller.submit_slot(
            root, state, state["slots"][0], executable="unused", execution=execution
        )
    saved = json.loads((root / "state.json").read_text())
    assert saved["slots"][0]["phase"] == "submitting"
    assert saved["slots"][0]["shard"] == 0
    assert 0 not in saved["pending"]
    assert saved["slots"][0]["kernel_version"] is None


def test_collect_downloads_before_releasing_slot(tmp_path, stub_shared, monkeypatch):
    root, _, _, plan, execution, state = prepared_run(tmp_path)
    slot = state["slots"][0]
    slot.update(
        phase="submitted",
        shard=0,
        kernel="maiernator/worker",
        kernel_version=3,
        worker_sha256="b" * 64,
    )
    events = []

    def remote(_executable, command, **kwargs):
        events.append(command)
        assert slot["phase"] == "submitted"
        return "KernelWorkerStatus.COMPLETE" if command[0] == "status" else "downloaded"

    def validate(*args, **kwargs):
        assert events[-1][0:2] == ["output", "maiernator/worker/3"]
        assert slot["phase"] == "submitted"
        events.append(["validated"])
        return {"completed_parent_ids": ["synthetic"], "failed_parents": []}

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    monkeypatch.setattr(controller, "validate_download", validate)
    assert controller.collect_slot(
        root, state, slot, executable="unused", plan=plan, execution=execution
    )
    assert [event[0] for event in events] == ["status", "output", "validated"]
    assert slot["phase"] == "idle"
    assert slot["shard"] is None
    assert "0" in state["completed"]


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r\r\n"])
def test_admission_reads_warning_prefixed_csv_and_other_user_work(
    monkeypatch, line_ending
):
    header = "ref,title,author,lastRunTime,totalVotes"
    csv_output = line_ending.join(
        [
            "Warning: Looks like you’re using an outdated kaggle version",
            header,
            "maiernator/existing,Existing,maiernator,2026-09-07,0",
            "maiernator/completed,Completed,maiernator,2026-09-07,0",
        ]
    )
    calls = []

    def remote(_executable, args, **kwargs):
        calls.append(args)
        if args[0] == "list":
            return csv_output
        return "KernelWorkerStatus." + (
            "COMPLETE" if args[1].endswith("completed") else "RUNNING"
        )

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    active = controller.occupied_kernels(
        "unused",
        "maiernator",
        {"slots": [{"phase": "submitting", "kernel": "maiernator/new"}]},
    )
    assert active == {"maiernator/existing", "maiernator/new"}
    assert all(call[0] in {"list", "status"} for call in calls)


@pytest.mark.parametrize(
    "response",
    [
        "warning only",
        "ref,title\nmaiernator/work,Work",
        "ref,title,author,lastRunTime,totalVotes\nsomeone/work,Work,someone,date,0",
        "ref,title,author,lastRunTime,totalVotes\nmalformed row",
    ],
)
def test_admission_rejects_unknown_inventory(monkeypatch, response):
    monkeypatch.setattr(
        controller.transport, "kaggle_command", lambda *args, **kwargs: response
    )
    with pytest.raises(RuntimeError, match="admission inventory"):
        controller.occupied_kernels("unused", "maiernator", {"slots": []})


def test_admission_no_kernels_with_cli_warning(monkeypatch):
    monkeypatch.setattr(
        controller.transport,
        "kaggle_command",
        lambda *args, **kwargs: "Warning: update\nNot found\n",
    )
    assert controller.occupied_kernels("unused", "maiernator", {"slots": []}) == set()


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r\r\n"])
def test_admission_discloses_inaccessible_placeholder_and_handles_trailing_notice(
    monkeypatch, line_ending
):
    notice = "Warning: Looks like you're using an outdated `kaggle` version (installed: 2.2.0), please consider upgrading to the latest version (2.2.2)"
    output = line_ending.join(
        [
            notice,
            "ref,title,author,lastRunTime,totalVotes",
            "",
            "maiernator/completed,Completed,Markus.JM,2026-09-07,0",
            "",
            ",[Private Notebook],,2010-04-01 00:00:00,0",
            "",
            notice,
            "",
        ]
    )
    calls = []

    def remote(_executable, args, **kwargs):
        calls.append(args)
        return output if args[0] == "list" else "KernelWorkerStatus.COMPLETE"

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    state = {"slots": []}
    occupied = controller.occupied_kernels("unused", "maiernator", state)
    assert occupied == set()
    assert state["admission_unresolved"] == ["inaccessible-private-notebook:p1:r1"]
    assert [args for args in calls if args[0] == "status"] == [
        ["status", "maiernator/completed"]
    ]


@pytest.mark.parametrize("trailer", ["Warning: unknown failure", "malformed row"])
def test_admission_rejects_unknown_trailing_output(monkeypatch, trailer):
    response = (
        "ref,title,author,lastRunTime,totalVotes\nmaiernator/work,Work,author,date,0\n"
        + trailer
    )
    monkeypatch.setattr(
        controller.transport, "kaggle_command", lambda *args, **kwargs: response
    )
    with pytest.raises(RuntimeError, match="admission inventory"):
        controller.occupied_kernels("unused", "maiernator", {"slots": []})


@pytest.mark.parametrize("status", ["unknown status", "KernelWorkerStatus.UNKNOWN"])
def test_admission_rejects_unknown_kernel_status(monkeypatch, status):
    def remote(_executable, args, **kwargs):
        if args[0] == "list":
            return "ref,title,author,lastRunTime,totalVotes\nmaiernator/work,Work,maiernator,2026-09-07,0"
        return status

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    with pytest.raises(RuntimeError, match="kernel status"):
        controller.occupied_kernels("unused", "maiernator", {"slots": []})


@pytest.mark.parametrize("previous", [None, "cp1252"])
@pytest.mark.parametrize("failed", [False, True])
def test_cli_uses_utf8_and_restores_process_environment(monkeypatch, previous, failed):
    if previous is None:
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    else:
        monkeypatch.setenv("PYTHONIOENCODING", previous)

    def remote(*args, **kwargs):
        assert os.environ["PYTHONIOENCODING"] == "utf-8"
        if failed:
            raise RuntimeError("synthetic transport failure")
        return "Unicode title: 🏠"

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    if failed:
        with pytest.raises(RuntimeError, match="synthetic transport failure"):
            controller.kaggle_command("unused", ["list"])
    else:
        assert controller.kaggle_command("unused", ["list"]) == "Unicode title: 🏠"
    assert os.environ.get("PYTHONIOENCODING") == previous


@pytest.mark.parametrize(
    "failure",
    [
        "404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus",
        "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))",
    ],
)
def test_missing_session_is_disclosed_without_hiding_known_activity(
    monkeypatch, failure
):
    def remote(_executable, args, **kwargs):
        if args[0] == "list":
            return "ref,title,author,lastRunTime,totalVotes\nmaiernator/old,Old,author,2026-09-07,0\nmaiernator/active,Active,author,2026-09-07,0"
        if args[1].endswith("/old"):
            raise RuntimeError("Kaggle status failed (1): " + failure)
        return "KernelWorkerStatus.RUNNING"

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    state = {"slots": []}
    assert controller.occupied_kernels("unused", "maiernator", state) == {
        "maiernator/active"
    }
    assert state["admission_unresolved"] == ["maiernator/old"]
    assert failure in state["admission_status_errors"]["maiernator/old"]


def test_explicit_quota_rejection_preserves_receipt_and_unstarted_assignment(
    tmp_path, stub_shared, monkeypatch
):
    root, _, _, _, execution, state = prepared_run(tmp_path)
    initial_pending = list(state["pending"])
    initial_slot = dict(state["slots"][0])

    def rejected(*args, **kwargs):
        raise RuntimeError(
            "Kaggle push failed (1): Maximum number of concurrent CPU sessions reached"
        )

    monkeypatch.setattr(controller.transport, "kaggle_command", rejected)
    with pytest.raises(controller.QuotaAdmissionRejected):
        controller.submit_slot(
            root, state, state["slots"][0], executable="unused", execution=execution
        )
    assert state["pending"] == initial_pending
    assert state["slots"][0] == initial_slot
    record = state["admission_rejections"][0]
    receipt = root / record["receipt"]
    assert record["sha256"] == worker.file_hash(receipt)
    assert json.loads(receipt.read_text())["no_confirmed_kernel_version"] is True
    assert json.loads((root / "state.json").read_text()) == state


@pytest.mark.parametrize(
    "message",
    [
        "Kaggle push failed (1): Kernel version 8 successfully pushed; too many active sessions",
        "Kaggle push failed (1): Kernel version 8: Maximum number of concurrent CPU sessions reached",
        "Kaggle push failed (1): Connection timed out after push",
        "quota status: too many active sessions",
        "Kaggle push failed (1): " + "x" * 1500 + " too many active sessions",
    ],
)
def test_ambiguous_or_unrelated_output_cannot_authorize_readmission(message):
    assert not controller.explicit_quota_rejection(message)


def test_capacity_queries_only_recent_and_durable_slots_with_bounded_timeout(
    monkeypatch,
):
    calls = []

    def remote(_executable, args, **kwargs):
        if args[0] == "list":
            return (
                "ref,title,author,lastRunTime,totalVotes\n"
                "maiernator/recent,Recent,author,2026-09-07 08:00:00,0\n"
                "maiernator/historical,Old,author,2020-01-01,0\n"
                "maiernator/owned,Owned,author,2020-01-01,0\n"
                "maiernator/unknown,Unknown,author,unavailable,0"
            )
        calls.append((args[1], kwargs["timeout"]))
        return "KernelWorkerStatus.COMPLETE"

    monkeypatch.setattr(controller.transport, "kaggle_command", remote)
    state = {"slots": [{"kernel": "maiernator/owned", "phase": "submitted"}]}
    assert controller.occupied_kernels("unused", "maiernator", state) == {
        "maiernator/owned"
    }
    assert calls == [("maiernator/owned", 15), ("maiernator/recent", 15)]
    assert state["admission_inventory_count"] == 4
    assert state["admission_queried_count"] == 2
    assert state["admission_not_queried"] == {
        "maiernator/historical": "last_run_before_recent_24h_window",
        "maiernator/unknown": "unrecognized_last_run_time",
    }


def test_controller_refills_verified_completed_slot_in_same_cycle(
    tmp_path, stub_shared, monkeypatch
):
    root, plan_path, registration_path, plan, execution, state = prepared_run(tmp_path)
    state["pending"] = [5]
    for index, slot in enumerate(state["slots"]):
        slot.update(phase="submitted", shard=index, kernel=f"maiernator/worker-{index}")
    occupied = {slot["kernel"] for slot in state["slots"]}
    monkeypatch.setattr(
        controller, "prepare_run", lambda *args, **kwargs: (plan, execution, state)
    )
    monkeypatch.setattr(controller, "occupied_kernels", lambda *args: occupied.copy())
    monkeypatch.setattr(
        controller, "kaggle_command", lambda *args, **kwargs: "--file-pattern"
    )

    class ObservedRefill(Exception):
        pass

    events = []

    def collect(_root, _state, slot, **kwargs):
        if slot["slot"] != 0:
            return False
        events.append("verified_collection")
        slot.update(phase="idle", shard=None)
        return True

    def submit(_root, _state, slot, **kwargs):
        assert slot["slot"] == 0
        assert events == ["verified_collection"]
        events.append("same_cycle_refill")
        raise ObservedRefill

    monkeypatch.setattr(controller, "collect_slot", collect)
    monkeypatch.setattr(controller, "submit_slot", submit)
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda *args: pytest.fail("Refill waited until another poll"),
    )
    with pytest.raises(ObservedRefill):
        controller.run_controller(
            SimpleNamespace(
                output_root=root,
                plan=plan_path,
                registration=registration_path,
                owner="maiernator",
                slots=5,
                prepare_only=False,
                kaggle="unused",
                poll_seconds=1200,
            )
        )
    assert events == ["verified_collection", "same_cycle_refill"]


def test_remote_scheduler_preserves_failures_and_finishes_independent_parents(
    tmp_path, monkeypatch
):
    plan = fake_plan()
    spec = worker.shard_specs(plan)[1]
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = workspace / "output"
    parents = {row["parent_id"]: row for row in plan["parents"]}
    launched = []
    manifest = {**spec, "parents": []}
    fake_psutil = SimpleNamespace(
        Process=lambda *args: SimpleNamespace(
            cpu_affinity=lambda: [0, 1, 2, 3],
            memory_info=lambda: SimpleNamespace(rss=worker.GIB),
        ),
        virtual_memory=lambda: SimpleNamespace(available=32 * worker.GIB),
        NoSuchProcess=ProcessLookupError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    def fake_popen(command, **kwargs):
        parent_id = command[command.index("--parent-id") + 1]
        affinity = command[command.index("--affinity") + 1]
        launched.append((parent_id, affinity))
        parent = parents[parent_id]
        child_manifest, raw = worker.parent_paths(output, parent)
        failed = len(launched) == 1
        worker.write_json(
            child_manifest, {"status": "failed" if failed else "complete"}
        )
        if not failed:
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"synthetic")
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["CTBOOST_HIST_THREADS"] == "2"
        assert kwargs["start_new_session"] is True
        code = 3 if failed else 0
        return SimpleNamespace(
            pid=100 + len(launched), returncode=code, poll=lambda: code
        )

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    worker.run_parents(
        tmp_path / "package",
        workspace,
        artifacts,
        plan,
        spec,
        manifest,
        tmp_path / "python",
        tmp_path / "wheel.whl",
    )
    assert [row[0] for row in launched] == spec["parent_ids"]
    assert [row[1] for row in launched] == ["0,1", "2,3"] * 6
    assert manifest["failed_parent_count"] == 1
    assert manifest["result_file_count"] == 11
    assert manifest["terminal_parent_count"] == 12
    assert manifest["parents"][0]["exit_code"] == 3
    assert len({row[0] for row in launched}) == len(launched)


def test_remote_scheduler_refuses_unavailable_reserved_memory(tmp_path, monkeypatch):
    plan = fake_plan()
    fake_psutil = SimpleNamespace(
        Process=lambda: SimpleNamespace(cpu_affinity=lambda: [0, 1, 2, 3]),
        virtual_memory=lambda: SimpleNamespace(available=11 * worker.GIB),
        NoSuchProcess=ProcessLookupError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Must not start a parent"),
    )
    with pytest.raises(RuntimeError, match="Insufficient memory"):
        worker.run_parents(
            tmp_path / "package",
            tmp_path / "workspace",
            tmp_path / "artifacts",
            plan,
            worker.shard_specs(plan)[0],
            {"parents": []},
            tmp_path / "python",
            tmp_path / "wheel.whl",
        )
