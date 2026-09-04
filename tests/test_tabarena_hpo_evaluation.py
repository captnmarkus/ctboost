"""Fast validation tests; no benchmark fits, downloads, or foreign pickles."""

import gzip
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from benchmarks.tabarena import evaluate_kaggle_hpo as evaluation
from benchmarks.tabarena import kaggle_hpo as controller
from benchmarks.tabarena import kaggle_hpo_worker as protocol


def _task_ids():
    return {
        name: str(index + 1000)
        for index, name in enumerate(
            name for group in protocol.DATASET_GROUPS for name in group
        )
    }


def _record(path, data=b"trusted fixture"):
    return {
        "path": path,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _manifest(index=0):
    spec = protocol.shard_spec(index)
    ids = _task_ids()
    return {
        **spec,
        "schema_version": 1,
        "status": "complete",
        "benchmark_exit_code": 0,
        "fatal_error": None,
        "benchmark_name": protocol.BENCHMARK_NAME,
        "ctboost_version": protocol.CTBOOST_VERSION,
        "tabarena_commit": protocol.TABARENA_COMMIT,
        "configuration_count": 26,
        "hpo_configs_run": 25,
        "expected_parent_results_total": 1326,
        "expected_child_fits_total": 10608,
        "official_tabarena_lite_dataset_count": 51,
        "portfolio_200_sha256": protocol.PORTFOLIO_200_SHA256,
        "portfolio_25_sha256": evaluation.PORTFOLIO_25_SHA256,
        "worker_sha256": next(iter(evaluation.expected_worker_hashes(index))),
        "installed_packages": {
            "ctboost": "0.1.58",
            "autogluon.tabular": protocol.AUTOGLUON_VERSION,
        },
        "ctboost_build_info": {"package_version": "0.1.58"},
        "ctboost_wheel": {"filename": "ctboost-0.1.58-example.whl", "sha256": "a" * 64},
        "requested_resources": {
            "num_cpus": 4,
            "memory_limit_gb": 28,
            "time_limit_seconds": 3600,
        },
        "experiments_sha256": "b" * 64,
        "result_file_count": len(spec["datasets"]),
        "result_files": [
            _record(
                f"output/{protocol.BENCHMARK_NAME}/data/{spec['config_name']}/{ids[name]}/0_0/results.pkl"
            )
            for name in spec["datasets"]
        ],
    }


def _archive(path, entries):
    with tarfile.open(path, "w:gz") as stream:
        for name, data, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(data) if kind == tarfile.REGTYPE else 0
            member.linkname = "../../escaped"
            stream.addfile(member, io.BytesIO(data) if member.isfile() else None)


@pytest.mark.parametrize(
    "path",
    ["../escape", "/absolute", "C:/outside", "a\\b", "a/../../b", "a:stream", ""],
)
def test_rejects_unsafe_relative_paths(path):
    with pytest.raises(ValueError, match="relative path"):
        evaluation.safe_relative(path)


def test_manifest_requires_exact_frozen_identity_and_config_task_grid():
    manifest = _manifest()
    keys = evaluation.validate_manifest(manifest, _task_ids())
    assert len(keys) == 6
    assert set(keys.values()) == {
        (manifest["config_name"], name) for name in protocol.DATASET_GROUPS[0]
    }


def test_manifest_uses_frozen_template_instead_of_current_worker(tmp_path):
    template = tmp_path / "worker_template.py"
    template.write_text(
        Path(protocol.__file__).read_text() + "\n# frozen worker fixture\n",
        encoding="utf-8",
    )
    manifest = _manifest()
    manifest["worker_sha256"] = next(
        iter(evaluation.expected_worker_hashes(0, template))
    )
    evaluation.validate_manifest(manifest, _task_ids(), template)
    with pytest.raises(ValueError, match="worker source hash mismatch"):
        evaluation.validate_manifest(manifest, _task_ids())


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_evaluator_hash_matches_controller_lf_worker(tmp_path, newline):
    template = tmp_path / "template.py"
    template.write_bytes(f"# frozen template{newline}SHARD_INDEX = 0{newline}".encode())
    package = tmp_path / "package"
    controller.prepare_package(
        template, package, owner="example", slot=0, shard=155, run_id="test"
    )
    rendered = (package / "worker.py").read_bytes()
    assert b"\r\n" not in rendered
    assert evaluation.expected_worker_hashes(155, template) == {
        hashlib.sha256(rendered).hexdigest()
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "incomplete"),
        ("hpo_configs_run", 200),
        ("ctboost_version", "0.1.56"),
        ("worker_sha256", "f" * 64),
        ("portfolio_25_sha256", "f" * 64),
        ("expected_child_fits_total", 1326),
        ("fatal_error", "failed fit"),
    ],
)
def test_manifest_rejects_incomplete_or_changed_protocol(key, value):
    manifest = _manifest()
    manifest[key] = value
    with pytest.raises(ValueError):
        evaluation.validate_manifest(manifest, _task_ids())


def test_manifest_rejects_duplicate_result_even_if_count_is_right():
    manifest = _manifest()
    manifest["result_files"][-1] = manifest["result_files"][0]
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.validate_manifest(manifest, _task_ids())


def test_incomplete_shard_collection_never_starts_evaluation(tmp_path):
    with pytest.raises(ValueError, match="no scores produced"):
        evaluation.discover_shards(tmp_path, _task_ids())
    assert list(tmp_path.iterdir()) == []


def test_complete_preflight_requires_all_1326_config_task_pairs(tmp_path):
    for index in range(156):
        artifacts = tmp_path / f"s{index:03d}" / "artifacts"
        artifacts.mkdir(parents=True)
        archive = artifacts / "raw.tar.gz"
        archive.write_bytes(b"preflight checksum fixture")
        manifest = _manifest(index)
        manifest["workspace_archive"] = _record(
            "artifacts/raw.tar.gz", archive.read_bytes()
        )
        (artifacts / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shards = evaluation.discover_shards(tmp_path, _task_ids())
    assert len(shards) == 156
    assert sum(len(manifest["result_files"]) for _, manifest in shards) == 1326
    assert len({manifest["config_name"] for _, manifest in shards}) == 26
    (tmp_path / "s155/artifacts/raw.tar.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Size mismatch"):
        evaluation.discover_shards(tmp_path, _task_ids())


def test_stream_extracts_only_declared_results(tmp_path):
    manifest = _manifest()
    record = manifest["result_files"][0]
    archive = tmp_path / "worker.tar.gz"
    _archive(
        archive,
        [
            ("workspace/notes.txt", b"not copied", tarfile.REGTYPE),
            ("workspace/" + record["path"], b"trusted fixture", tarfile.REGTYPE),
        ],
    )
    paths = evaluation.extract_results(archive, [record], tmp_path / "raw")
    assert len(paths) == 1
    assert paths[0].read_bytes() == b"trusted fixture"
    assert list((tmp_path / "raw").rglob("notes.txt")) == []


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("workspace/link", tarfile.SYMTYPE),
        ("workspace/link", tarfile.LNKTYPE),
        ("workspace/extra/results.pkl", tarfile.REGTYPE),
    ],
)
def test_archive_rejects_traversal_links_and_undeclared_results(tmp_path, name, kind):
    archive = tmp_path / "worker.tar.gz"
    _archive(archive, [(name, b"payload", kind)])
    with pytest.raises(ValueError):
        evaluation.extract_results(archive, [], tmp_path / "raw")
    assert not (tmp_path / "escaped").exists()


def test_archive_rejects_duplicate_or_corrupt_result(tmp_path):
    record = _manifest()["result_files"][0]
    entry = ("workspace/" + record["path"], b"trusted fixture", tarfile.REGTYPE)
    archive = tmp_path / "worker.tar.gz"
    _archive(archive, [entry, entry])
    with pytest.raises(ValueError, match="Duplicate archive"):
        evaluation.extract_results(archive, [record], tmp_path / "duplicate")
    _archive(archive, [entry])
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluation.extract_results(archive, [record], tmp_path / "corrupt")


def _raw_fixture(path):
    config = "CTBoost_c1_default_BAG_L1"
    data = {
        "task_metadata": {"name": "example", "tid": 1000, "fold": 0, "repeat": 0},
        "framework": config,
        "problem_type": "binary",
        "metric_error": 0.25,
        "metric_error_val": 0.5,
        "time_train_s": 1,
        "time_infer_s": 0.1,
        "method_metadata": {
            "info": {
                "bagged_info": {"num_child_models": 8},
                "is_fit": True,
                "is_valid": True,
                "can_infer": True,
                "children_info": {
                    f"S1F{i + 1}": {
                        "is_fit": True,
                        "is_valid": True,
                        "can_infer": True,
                        "hyperparameters": {"random_seed": i},
                    }
                    for i in range(8)
                },
            }
        },
        "simulation_artifacts": {
            "pred_proba_dict_val": {config: np.full(8, 0.5)},
            "y_val": np.arange(8) % 2,
            "pred_proba_dict_test": {config: np.array([0.3, 0.7])},
            "y_test": np.array([0, 1]),
            "bag_info": {
                "val_idx_per_child": [np.array([i]) for i in range(8)],
                "pred_proba_test_per_child": [np.array([0.3, 0.7])] * 8,
            },
        },
    }
    path.parent.mkdir(parents=True)
    return data


@pytest.mark.parametrize(
    "corruption", [None, "duplicate_fold", "nan", "child_count", "identity"]
)
def test_raw_result_validates_actual_bag_children_and_predictions(tmp_path, corruption):
    path = tmp_path / "CTBoost_c1_default_BAG_L1/1000/0_0/results.pkl"
    data = _raw_fixture(path)
    if corruption == "duplicate_fold":
        data["simulation_artifacts"]["bag_info"]["val_idx_per_child"][-1] = np.array(
            [0]
        )
    elif corruption == "nan":
        data["simulation_artifacts"]["pred_proba_dict_val"][data["framework"]][0] = (
            np.nan
        )
    elif corruption == "child_count":
        data["method_metadata"]["info"]["bagged_info"]["num_child_models"] = 7
    elif corruption == "identity":
        data["task_metadata"]["fold"] = 1
    with gzip.open(path, "wb") as stream:
        pickle.dump(data, stream)
    if corruption:
        with pytest.raises(ValueError):
            evaluation.validate_raw_result(path, {"example": "1000"})
    else:
        assert (
            evaluation.validate_raw_result(path, {"example": "1000"})["child_fits"] == 8
        )


def test_public_bundle_excludes_raw_arrays_and_checksums_every_file(tmp_path):
    for relative in (
        "canonical-results/metadata.yaml",
        "canonical-results/results/hpo_results.parquet",
        "canonical-results/results/model_results.parquet",
        "reports/leaderboard_lite.csv",
        "reports/ctboost_lite_hpo_summary.json",
        "private-raw/results.pkl",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    bundle = evaluation.public_bundle(tmp_path, {"scope": "test fixture"})
    assert not list(bundle.rglob("*.pkl"))
    assert not (bundle / "private-raw").exists()
    lines = (bundle / "SHA256SUMS").read_text().splitlines()
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        assert digest == hashlib.sha256((bundle / relative).read_bytes()).hexdigest()
    assert len(lines) == 7
