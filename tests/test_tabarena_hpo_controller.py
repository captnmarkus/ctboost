"""Queue and artifact integrity checks without a Kaggle account or model fitting."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

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


def test_existing_push_without_recognized_url_requires_exact_source_proof(
    tmp_path, monkeypatch
):
    state, slot, source = pending_submission()
    calls = []

    def command(executable, arguments):
        calls.append(arguments)
        saved = json.loads((tmp_path / "state.json").read_text())
        assert saved["slots"][0]["kernel_version"] == 2
        receipt = tmp_path / saved["slots"][0]["submission_receipt"]
        assert json.loads(receipt.read_text())["output"].startswith(
            "Kernel version 2 successfully pushed."
        )
        assert arguments[:2] == ["pull", "example/worker-1"]
        pull_response(Path(arguments[3]), source)
        return "Pulled successfully"

    monkeypatch.setattr(controller, "kaggle_command", command)
    controller.record_submission(
        tmp_path,
        state,
        slot,
        "Kernel version 2 successfully pushed. Please check progress at ",
        owner="example",
        executable="kaggle",
        existing_kernel="example/worker-1",
    )
    assert len(calls) == 1
    assert slot["phase"] == "submitted"
    assert slot["kernel"] == "example/worker-1"
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
