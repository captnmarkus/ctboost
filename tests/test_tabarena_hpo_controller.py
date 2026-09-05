"""Queue and artifact integrity checks without a Kaggle account or model fitting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "kaggle_hpo_controller",
    Path(__file__).parents[1] / "benchmarks/tabarena/kaggle_hpo.py",
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def artifact(tmp_path, **overrides):
    root = tmp_path / "run"
    archive = root / "artifacts" / "raw.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"test archive bytes")
    manifest = {
        **controller.load_worker(
            Path(__file__).parents[1] / "benchmarks/tabarena/kaggle_hpo_worker.py"
        ).shard_spec(3),
        "benchmark_name": "ctboost_0158_lite_hpo25_20260904",
        "tabarena_commit": "31026f7d758390994353eba79fbfa6747616f365",
        "portfolio_200_sha256": "bd1b81b98a89ab33ac4cea35cb4b7dd7727b3bcfa3bee1b044fd3fb44f965c72",
        "shard_index": 3,
        "ctboost_version": "0.1.58",
        "status": "complete",
        "expected_parent_results_in_shard": 9,
        "result_file_count": 9,
        "workspace_archive": {
            "path": "artifacts/raw.tar.gz",
            "size_bytes": archive.stat().st_size,
            "sha256": controller.file_hash(archive),
        },
    }
    manifest.update(overrides)
    controller.write_json(archive.parent / "manifest.json", manifest)
    return archive


def test_valid_output_and_corrupt_archive(tmp_path):
    archive = artifact(tmp_path)
    assert controller.validate_download(tmp_path, 3)["result_count"] == 9
    archive.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="checksum"):
        controller.validate_download(tmp_path, 3)


@pytest.mark.parametrize(
    "overrides",
    [
        {"shard_index": 2},
        {"ctboost_version": "0.1.56"},
        {"status": "incomplete"},
        {"result_file_count": 8},
        {"workspace_archive": {"path": "../raw.tar.gz"}},
    ],
)
def test_reject_stale_or_incomplete_output(tmp_path, overrides):
    artifact(tmp_path, **overrides)
    with pytest.raises(ValueError):
        controller.validate_download(tmp_path, 3)


def setup_failure(tmp_path, *, endpoint="data/qualities/46907"):
    worker = controller.load_worker(
        Path(__file__).parents[1] / "benchmarks/tabarena/kaggle_hpo_worker.py"
    )
    destination = tmp_path / "shards" / "s003"
    source_hash = "frozen-shard-source-hash"
    archive = artifact(
        destination,
        worker_sha256=source_hash,
        status="incomplete",
        fatal_error=(
            "OpenMLServerError: Unexpected server error when calling "
            f"https://www.openml.org/api/v1/xml/{endpoint}. "
            "Please contact the developers!\nStatus code: 503\n<html>503</html>"
        ),
        result_file_count=0,
        result_files=[],
        failures=[],
        benchmark_exit_code=None,
        validation={
            "bag_children_verified": 0,
            "valid_result_count": 0,
            "invalid_results": [],
            "missing_datasets": sorted(worker.shard_spec(3)["datasets"]),
        },
    )
    slot = {
        "slot": 0,
        "shard": 3,
        "phase": "submitted",
        "kernel": "example/worker-0",
        "kernel_version": 2,
        "worker_sha256": source_hash,
        "submission_receipt": "submissions/shard-003-slot-0.json",
    }
    controller.write_json(
        tmp_path / slot["submission_receipt"],
        {
            "requested_kernel": slot["kernel"],
            "shard": 3,
            "worker_sha256": source_hash,
            "output": "Kernel version 2 successfully pushed.",
        },
    )
    state = {"completed": {}, "pending": [4], "failed": {}, "slots": [slot]}
    return state, slot, destination, worker, archive


@pytest.mark.parametrize(
    "endpoint", ["data/qualities/46907", "data/46907", "task/363681"]
)
def test_pretraining_openml_503_preserves_attempt_before_same_shard_retry(
    tmp_path, endpoint
):
    state, slot, destination, worker, archive = setup_failure(
        tmp_path, endpoint=endpoint
    )
    manifest_hash = controller.file_hash(archive.parent / "manifest.json")
    assert controller.retry_openml_setup_failure(
        tmp_path, state, slot, destination, worker=worker
    )
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["pending"] == [3, 4]
    assert saved["completed"] == saved["failed"] == {}
    assert saved["slots"][0]["phase"] == "idle"
    assert saved["slots"][0]["shard"] is None
    record = saved["retry_history"][0]
    assert record["worker_sha256"] == slot["worker_sha256"]
    assert record["kernel_version"] == 2
    assert record["manifest_sha256"] == manifest_hash
    backup = tmp_path / record["previous_attempt"]
    assert (
        controller.file_hash(backup / "submission.json")
        == record["submission_receipt_sha256"]
    )
    prior_slot = json.loads((backup / "retry.json").read_text())["previous_slot"]
    assert prior_slot["shard"] == 3
    assert prior_slot["kernel_version"] == 2
    assert (backup / "download" / "run" / "artifacts" / "raw.tar.gz").is_file()
    assert not destination.exists()


@pytest.mark.parametrize(
    "endpoint",
    [
        "data/list",
        "task/list",
        "data/features/46907",
        "data/46907/extra",
        "task/363681?retry=true",
        "task/not-an-id",
    ],
)
def test_openml_retry_rejects_other_endpoints(tmp_path, endpoint):
    state, slot, destination, worker, _ = setup_failure(tmp_path, endpoint=endpoint)
    original = json.dumps(state, sort_keys=True)
    assert not controller.retry_openml_setup_failure(
        tmp_path, state, slot, destination, worker=worker
    )
    assert json.dumps(state, sort_keys=True) == original
    assert destination.exists()
    assert not (tmp_path / "failed-attempts").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"fatal_error": "OpenMLServerError: Status code: 503"},
        {"fatal_error": "RuntimeError: model failed"},
        {"result_file_count": 1},
        {"result_files": [{"path": "results.pkl"}]},
        {"failures": [{"exit_code": 1}]},
        {"benchmark_exit_code": 0},
        {"run_commands": []},
        {"generated_job_json": "job.json"},
        {"worker_sha256": "different"},
        {"config_index": 1},
        {"status": "complete"},
        {"validation": {"bag_children_verified": 1}},
        {"validation": {"invalid_results": ["bad metrics"]}},
        {"validation": []},
    ],
)
@pytest.mark.parametrize(
    "endpoint", ["data/qualities/46907", "data/46907", "task/363681"]
)
def test_only_verified_openml_failure_before_any_training_is_retried(
    tmp_path, change, endpoint
):
    state, slot, destination, worker, archive = setup_failure(
        tmp_path, endpoint=endpoint
    )
    path = archive.parent / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(change)
    controller.write_json(path, manifest)
    original = json.dumps(state, sort_keys=True)
    assert not controller.retry_openml_setup_failure(
        tmp_path, state, slot, destination, worker=worker
    )
    assert json.dumps(state, sort_keys=True) == original
    assert destination.exists()
    assert not (tmp_path / "failed-attempts").exists()


@pytest.mark.parametrize("prior_shard", [3, "3"])
@pytest.mark.parametrize(
    "endpoint", ["data/qualities/46907", "data/46907", "task/363681"]
)
def test_manual_retry_history_already_exhausts_single_retry(
    tmp_path, prior_shard, endpoint
):
    state, slot, destination, worker, _ = setup_failure(tmp_path, endpoint=endpoint)
    state["retry_history"] = [
        {"shard": prior_shard, "previous_attempt": "failed-attempts/manual"}
    ]
    assert not controller.retry_openml_setup_failure(
        tmp_path, state, slot, destination, worker=worker
    )
    assert destination.exists()
    assert state["pending"] == [4]


def test_corrupt_failed_archive_cannot_trigger_retry(tmp_path):
    state, slot, destination, worker, archive = setup_failure(tmp_path)
    archive.write_bytes(b"corrupt")
    assert not controller.retry_openml_setup_failure(
        tmp_path, state, slot, destination, worker=worker
    )
    assert state["pending"] == [4]


@pytest.mark.parametrize("failure", ["receipt", "existing_backup", "move"])
def test_retry_archival_failure_does_not_change_queue(tmp_path, monkeypatch, failure):
    state, slot, destination, worker, _ = setup_failure(tmp_path)
    if failure == "receipt":
        controller.write_json(tmp_path / slot["submission_receipt"], {})
    elif failure == "existing_backup":
        (tmp_path / "failed-attempts" / "s003-v2-openml503").mkdir(parents=True)
    else:

        def denied(*args):
            raise OSError("Archive move denied")

        monkeypatch.setattr(Path, "rename", denied)
    original = json.dumps(state, sort_keys=True)
    with pytest.raises((ValueError, OSError)):
        controller.retry_openml_setup_failure(
            tmp_path, state, slot, destination, worker=worker
        )
    assert json.dumps(state, sort_keys=True) == original
    assert destination.exists()


@pytest.mark.parametrize("status_error", [False, True])
def test_controller_retries_same_frozen_shard_and_records_next_check(
    tmp_path, monkeypatch, status_error
):
    state, slot, _, _, archive = setup_failure(tmp_path)
    source = Path(controller.__file__).with_name("kaggle_hpo_worker.py")
    template = tmp_path / "worker_template.py"
    template.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    state.update(
        owner="example", worker_sha256=controller.file_hash(template), run_id="test"
    )
    controller.prepare_package(
        template,
        tmp_path / "packages" / "0",
        owner="example",
        slot=0,
        shard=3,
        run_id="test",
        existing_kernel=slot["kernel"],
    )
    source_hash = controller.file_hash(tmp_path / "packages" / "0" / "worker.py")
    slot["worker_sha256"] = source_hash
    for path in (
        archive.parent / "manifest.json",
        tmp_path / slot["submission_receipt"],
    ):
        record = json.loads(path.read_text())
        record["worker_sha256"] = source_hash
        controller.write_json(path, record)
    controller.write_json(tmp_path / "state.json", state)
    calls = []

    def command(executable, arguments, **kwargs):
        calls.append(arguments)
        if arguments == ["output", "--help"]:
            return "--file-pattern"
        if arguments[0] == "status":
            if status_error:
                raise OSError("Cannot start status command")
            return "KernelWorkerStatus.COMPLETE"
        if arguments[0] == "output":
            return "Downloaded"
        assert arguments[0] == "push"
        assert controller.file_hash(Path(arguments[2]) / "worker.py") == source_hash
        return (
            "Kernel version 3 successfully pushed. Please check progress at "
            "https://www.kaggle.com/code/example/worker-0"
        )

    class CheckedOnce(Exception):
        pass

    def sleep(seconds):
        assert seconds == 1800
        progress = json.loads((tmp_path / "progress.json").read_text())
        assert progress["poll_seconds"] == 1800
        elapsed = datetime.fromisoformat(
            progress["next_check_at"]
        ) - datetime.fromisoformat(progress["updated_at"])
        assert elapsed.total_seconds() == 1800
        raise CheckedOnce

    monkeypatch.setattr(controller, "kaggle_command", command)
    monkeypatch.setattr(controller.time, "sleep", sleep)
    with pytest.raises(CheckedOnce):
        controller.run_controller(
            SimpleNamespace(
                output_root=tmp_path,
                owner="example",
                slots=1,
                prepare_only=False,
                kaggle="kaggle",
                poll_seconds=1800,
            )
        )
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["slots"][0]["shard"] == 3
    assert saved["slots"][0]["kernel_version"] == (2 if status_error else 3)
    assert saved["pending"] == [4]
    assert len([call for call in calls if call[0] == "push"]) == (
        0 if status_error else 1
    )


def active_queue(tmp_path):
    template = tmp_path / "worker_template.py"
    source = Path(controller.__file__).with_name("kaggle_hpo_worker.py")
    template.write_bytes(source.read_bytes())
    state = {
        "owner": "example",
        "worker_sha256": controller.file_hash(template),
        "pending": [4],
        "completed": {},
        "failed": {},
        "slots": [
            {
                "slot": 0,
                "shard": 3,
                "kernel": "example/worker-0",
                "kernel_version": 2,
                "phase": "submitted",
                "collection_errors": 2,
            }
        ],
    }
    controller.write_json(tmp_path / "state.json", state)
    args = SimpleNamespace(
        output_root=tmp_path,
        owner="example",
        slots=1,
        prepare_only=False,
        kaggle=str(tmp_path / "missing-kaggle.exe"),
        poll_seconds=1800,
    )
    return state, args


def test_missing_kaggle_cli_preflight_preserves_queue(tmp_path):
    state, args = active_queue(tmp_path)
    with pytest.raises(RuntimeError, match="Cannot launch Kaggle CLI") as error:
        controller.run_controller(args)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert json.loads((tmp_path / "state.json").read_text()) == state


@pytest.mark.parametrize("missing_during", ["status", "output"])
def test_disappearing_cli_does_not_consume_artifact_retries(
    tmp_path, monkeypatch, missing_during
):
    state, args = active_queue(tmp_path)
    failed_launches = []

    def run(command, **kwargs):
        arguments = command[2:]
        if arguments == ["output", "--help"]:
            return subprocess.CompletedProcess(command, 0, "--file-pattern", "")
        if arguments[0] == missing_during:
            failed_launches.append(arguments)
            raise FileNotFoundError("Kaggle CLI disappeared after preflight")
        assert arguments[0] == "status"
        return subprocess.CompletedProcess(
            command, 0, "KernelWorkerStatus.COMPLETE", ""
        )

    class CheckedThreeTimes(Exception):
        pass

    def sleep(seconds):
        assert seconds == 1800
        assert json.loads((tmp_path / "state.json").read_text()) == state
        if len(failed_launches) == 3:
            raise CheckedThreeTimes

    monkeypatch.setattr(controller.subprocess, "run", run)
    monkeypatch.setattr(controller.time, "sleep", sleep)
    with pytest.raises(CheckedThreeTimes):
        controller.run_controller(args)
    assert len(failed_launches) == 3
    assert json.loads((tmp_path / "state.json").read_text()) == state


def test_generated_kernel_is_private_cpu_and_selects_shard(tmp_path):
    worker = tmp_path / "worker_template.py"
    worker.write_bytes(b"SHARD_INDEX = 0\r\n")
    destination = tmp_path / "package"
    controller.prepare_package(
        worker, destination, owner="example", slot=1, shard=155, run_id="12345678"
    )
    assert (destination / "worker.py").read_bytes() == b"SHARD_INDEX = 155\n"
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert metadata["id"] == "example/ctboost-0158-lite-hpo25-12345678-w1"
    assert (
        re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
        == metadata["id"].split("/")[1]
    )


def test_slot_reuses_confirmed_returned_kernel_and_matching_title(tmp_path):
    worker = tmp_path / "worker_template.py"
    worker.write_text("SHARD_INDEX = 0\n", encoding="utf-8")
    kernel = "example/ctboost-0-1-58-lite-hpo25-worker-0"
    destination = tmp_path / "package"
    assert (
        controller.prepare_package(
            worker,
            destination,
            owner="example",
            slot=0,
            shard=5,
            run_id="12345678",
            existing_kernel=kernel,
        )
        == kernel
    )
    metadata = json.loads((destination / "kernel-metadata.json").read_text())
    assert metadata["id"] == kernel
    assert (
        re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
        == kernel.split("/")[1]
    )


def test_submission_adopts_returned_title_slug():
    output = (
        "Your kernel title does not resolve to the specified id. "
        "Kernel version 1 successfully pushed. Please check progress at "
        "https://www.kaggle.com/code/example/ctboost-0-1-58-lite-hpo25-worker-0"
    )
    assert (
        controller.pushed_kernel(output, owner="example")
        == "example/ctboost-0-1-58-lite-hpo25-worker-0"
    )


@pytest.mark.parametrize(
    "output",
    [
        "Kernel version 1 successfully pushed.",
        "https://www.kaggle.com/code/other/worker-0",
        "https://www.kaggle.com/code/example/worker-0 https://www.kaggle.com/code/example/worker-1",
    ],
)
def test_submission_without_one_confirmed_owned_url_needs_reconciliation(output):
    with pytest.raises(RuntimeError, match="confirm one kernel URL"):
        controller.pushed_kernel(output, owner="example")


def pending_submission():
    source = "SHARD_INDEX = 5\n"
    slot = {
        "slot": 1,
        "shard": 5,
        "kernel": "example/worker-1",
        "phase": "submitting",
        "kernel_version": 1,
        "worker_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    return {"slots": [slot]}, slot, source


def pull_response(destination, source, *, kernel="example/worker-1"):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "worker.py").write_bytes(source.replace("\n", "\r\n").encode())
    controller.write_json(
        destination / "kernel-metadata.json",
        {
            "id": kernel,
            "code_file": "worker.py",
            "is_private": True,
            "enable_gpu": False,
        },
    )


@pytest.mark.parametrize(
    "kernel, progress_url",
    [
        ("example/worker-1", ""),
        (
            "maiernator/ctboost-0-1-58-lite-hpo25-worker-0",
            "https://www.kaggle.com/maiernator/ctboost-0-1-58-lite-hpo25-worker-0",
        ),
    ],
)
def test_existing_push_without_recognized_url_requires_exact_source_proof(
    tmp_path, monkeypatch, kernel, progress_url
):
    state, slot, source = pending_submission()
    slot["kernel"] = kernel
    calls = []

    def command(executable, arguments):
        calls.append(arguments)
        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["slots"][0]["kernel_version"] == 2
        receipt = tmp_path / saved["slots"][0]["submission_receipt"]
        assert json.loads(receipt.read_text())["output"].startswith(
            "Kernel version 2 successfully pushed."
        )
        assert arguments[:2] == ["pull", kernel]
        pull_response(Path(arguments[3]), source, kernel=kernel)
        return "Pulled successfully"

    monkeypatch.setattr(controller, "kaggle_command", command)
    controller.record_submission(
        tmp_path,
        state,
        slot,
        "Kernel version 2 successfully pushed. Please check progress at "
        + progress_url,
        owner=kernel.split("/")[0],
        executable="kaggle",
        existing_kernel=kernel,
    )
    assert len(calls) == 1
    assert slot["phase"] == "submitted"
    assert slot["kernel"] == kernel
    assert slot["kernel_version"] == 2
    assert slot["identity_confirmation"] == "latest_source_pull"


@pytest.mark.parametrize("corruption", ["source", "identity"])
def test_existing_push_source_or_identity_mismatch_remains_pending(
    tmp_path, monkeypatch, corruption
):
    state, slot, source = pending_submission()

    def command(executable, arguments):
        pull_response(
            Path(arguments[3]),
            source + "# wrong" if corruption == "source" else source,
            kernel="other/worker" if corruption == "identity" else "example/worker-1",
        )
        return "Pulled successfully"

    monkeypatch.setattr(controller, "kaggle_command", command)
    with pytest.raises(ValueError, match="Source pull"):
        controller.record_submission(
            tmp_path,
            state,
            slot,
            "Kernel version 2 successfully pushed.",
            owner="example",
            executable="kaggle",
            existing_kernel="example/worker-1",
        )
    saved = json.loads((tmp_path / "state.json").read_text())["slots"][0]
    assert saved["phase"] == "submitting"
    assert saved["kernel_version"] == 2
    assert saved["submission_parse_error"]
    assert (tmp_path / saved["submission_receipt"]).is_file()


def test_read_only_source_confirmation_retries_without_pushing(tmp_path, monkeypatch):
    state, slot, source = pending_submission()
    calls = []

    def command(executable, arguments):
        calls.append(arguments)
        assert arguments[0] == "pull"
        if len(calls) == 1:
            raise RuntimeError("Temporary read failure")
        pull_response(Path(arguments[3]), source)
        return "Pulled successfully"

    monkeypatch.setattr(controller, "kaggle_command", command)
    monkeypatch.setattr(controller.time, "sleep", lambda seconds: None)
    controller.record_submission(
        tmp_path,
        state,
        slot,
        "Kernel version 2 successfully pushed.",
        owner="example",
        executable="kaggle",
        existing_kernel="example/worker-1",
    )
    assert len(calls) == 2
    assert slot["phase"] == "submitted"


def test_unrecognized_new_kernel_preserves_receipt_and_confirmed_version(tmp_path):
    state, slot, _ = pending_submission()
    output = "Kernel version 2 successfully pushed. KGAT_test_secret"
    with pytest.raises(controller.KernelIdentityUnavailable):
        controller.record_submission(
            tmp_path,
            state,
            slot,
            output,
            owner="example",
            executable="kaggle",
            existing_kernel=None,
        )
    saved = json.loads((tmp_path / "state.json").read_text())["slots"][0]
    assert saved["phase"] == "submitting"
    assert saved["kernel_version"] == 2
    assert "KGAT_test_secret" not in json.dumps(saved)
    assert "[redacted]" in (tmp_path / saved["submission_receipt"]).read_text()


def test_unconfirmed_push_preserves_receipt_without_stale_version(tmp_path):
    state, slot, _ = pending_submission()
    with pytest.raises(RuntimeError, match="did not confirm submission"):
        controller.record_submission(
            tmp_path,
            state,
            slot,
            "Kernel push error: Busy",
            owner="example",
            executable="kaggle",
            existing_kernel="example/worker-1",
        )
    saved = json.loads((tmp_path / "state.json").read_text())["slots"][0]
    assert saved["kernel_version"] is None
    assert saved["phase"] == "submitting"
    assert (tmp_path / saved["submission_receipt"]).is_file()


@pytest.mark.parametrize(
    "url_text",
    [
        "https://www.kaggle.com/code/Example/worker-1",
        "https://www.kaggle.com/code/other/worker-1",
        "https://www.kaggle.com/code/example/worker-1 https://www.kaggle.com/code/example/worker-2",
    ],
)
def test_mismatched_url_requires_exact_existing_source_proof(
    tmp_path, monkeypatch, url_text
):
    state, slot, source = pending_submission()
    calls = []

    def pull(executable, arguments):
        calls.append(arguments)
        assert arguments[:2] == ["pull", "example/worker-1"]
        pull_response(Path(arguments[3]), source)
        return "Pulled successfully"

    monkeypatch.setattr(controller, "kaggle_command", pull)
    controller.record_submission(
        tmp_path,
        state,
        slot,
        "Kernel version 2 successfully pushed. " + url_text,
        owner="example",
        executable="kaggle",
        existing_kernel="example/worker-1",
    )
    assert len(calls) == 1
    assert slot["phase"] == "submitted"
    assert slot["kernel"] == "example/worker-1"
    assert slot["identity_confirmation"] == "latest_source_pull"


def test_push_requires_explicit_success_even_with_zero_cli_exit():
    assert (
        controller.pushed_version(
            "Kernel version 2 successfully pushed. See https://www.kaggle.com/"
        )
        == 2
    )
    with pytest.raises(RuntimeError, match="did not confirm"):
        controller.pushed_version("Kernel push error: Too many active kernels")


def test_error_output_redacts_tokens():
    value = "error Bearer example-token and KGAT_example_secret and hf_example_secret"
    sanitized = controller.redact(value)
    assert "example-token" not in sanitized
    assert "example_secret" not in sanitized


def test_queue_lock_prevents_two_controllers(tmp_path):
    with controller.controller_lock(tmp_path), pytest.raises(
        OSError
    ), controller.controller_lock(tmp_path):
        pytest.fail("second queue owner obtained the lock")
    with controller.controller_lock(tmp_path):
        pass
