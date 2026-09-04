"""Queue and artifact integrity checks without a Kaggle account or model fitting."""

from __future__ import annotations

import importlib.util
import json
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
