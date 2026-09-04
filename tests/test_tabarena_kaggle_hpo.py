"""Coverage and provenance guards for the resumable external HPO benchmark."""

from __future__ import annotations

import gzip
import json
import pickle
import tarfile
from collections import Counter

import pytest

from benchmarks.tabarena import kaggle_hpo_worker as worker


def test_shards_cover_each_config_dataset_exactly_once():
    specs = [worker.shard_spec(index) for index in range(worker.SHARD_COUNT)]
    pairs = Counter(
        (spec["config_name"], dataset) for spec in specs for dataset in spec["datasets"]
    )
    assert len(specs) == 156
    assert len(pairs) == 1326
    assert set(pairs.values()) == {1}
    assert len({dataset for _, dataset in pairs}) == 51
    counts = Counter(config for config, _ in pairs)
    assert len(counts) == 26
    assert set(counts.values()) == {51}
    assert sum(spec["expected_child_fits_in_shard"] for spec in specs) == 10608
    assert specs[0]["config_name"] == "CTBoost_c1_default_BAG_L1"
    assert specs[-1]["config_name"] == "CTBoost_r25_default_BAG_L1"


@pytest.mark.parametrize("index", [-1, 156, 1.0, True])
def test_invalid_shard_rejected(index):
    with pytest.raises(ValueError):
        worker.shard_spec(index)


def test_checkpoint_hashes_results_and_excludes_downloaded_datasets(tmp_path):
    workspace = tmp_path / "workspace"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    raw = (
        workspace
        / "output"
        / worker.BENCHMARK_NAME
        / "data"
        / "CTBoost_r25_default_BAG_L1"
        / "363612"
        / "0_0"
        / "results.pkl"
    )
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"partial result record")
    cache = workspace / "openml_cache"
    cache.mkdir()
    (cache / "private-original-dataset.arff").write_text(
        "do not archive", encoding="utf-8"
    )
    manifest = {**worker.shard_spec(155), "status": "running"}
    worker.checkpoint(workspace, artifacts, manifest)
    saved = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert saved["result_file_count"] == 1
    assert saved["result_files"] == [
        {
            "path": raw.relative_to(workspace).as_posix(),
            "size_bytes": raw.stat().st_size,
            "sha256": worker.sha256_file(raw),
        }
    ]
    archive = tmp_path / saved["workspace_archive"]["path"]
    assert worker.sha256_file(archive) == saved["workspace_archive"]["sha256"]
    with tarfile.open(archive) as stream:
        names = stream.getnames()
    assert "workspace/" + raw.relative_to(workspace).as_posix() in names
    assert not any("openml_cache" in name for name in names)
    raw.write_bytes(b"updated record")
    worker.checkpoint(workspace, artifacts, manifest)
    assert manifest["result_files"][0]["sha256"] == worker.sha256_file(raw)


def test_empty_checkpoint_preserves_incomplete_diagnostics(tmp_path):
    manifest = {
        **worker.shard_spec(0),
        "status": "incomplete",
        "fatal_error": "dependency unavailable",
    }
    worker.checkpoint(tmp_path / "workspace", tmp_path / "artifacts", manifest)
    assert manifest["result_files"] == []
    assert manifest["result_file_count"] == 0
    assert "workspace_archive" not in manifest
    saved = json.loads(
        (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert saved["fatal_error"] == "dependency unavailable"


@pytest.fixture
def result_record(tmp_path):
    import numpy as np

    spec = worker.shard_spec(0)
    config = spec["config_name"]
    model_state = {"is_fit": True, "is_valid": True, "can_infer": True}
    children = {
        f"S1F{fold + 1}": {**model_state, "hyperparameters": {"random_seed": fold}}
        for fold in range(8)
    }
    result = {
        "framework": config,
        "metric_error": 0.3,
        "metric_error_val": 0.4,
        "problem_type": "binary",
        "task_metadata": {
            "name": spec["datasets"][0],
            "tid": 999,
            "repeat": 0,
            "fold": 0,
        },
        "method_metadata": {
            "info": {
                **model_state,
                "children_info": children,
                "bagged_info": {
                    "num_child_models": 8,
                    "child_model_names": list(children),
                },
            }
        },
        "simulation_artifacts": {
            "pred_proba_dict_val": {config: np.full(8, 0.5)},
            "y_val": np.arange(8) % 2,
            "pred_proba_dict_test": {config: np.array([0.4, 0.7])},
            "y_test": np.array([0, 1]),
            "bag_info": {
                "val_idx_per_child": [np.array([fold]) for fold in range(8)],
                "pred_proba_test_per_child": [np.array([0.4, 0.7]) for _ in range(8)],
            },
        },
    }
    path = tmp_path / config / "999" / "0_0" / "results.pkl"
    path.parent.mkdir(parents=True)
    return path, spec, result


def test_valid_generated_result_proves_eight_children(result_record):
    path, spec, result = result_record
    with gzip.open(path, "wb") as stream:
        pickle.dump(result, stream)
    assert worker.validate_result_file(path, spec) == {
        "dataset": spec["datasets"][0],
        "task_id": "999",
        "bag_children": 8,
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "nan_metric",
        "failure",
        "missing_child",
        "bad_seed",
        "nan_prediction",
        "bad_folds",
    ],
)
def test_failed_or_partial_fit_cannot_validate_as_complete(result_record, corruption):
    path, spec, result = result_record
    if corruption == "nan_metric":
        result["metric_error"] = float("nan")
    elif corruption == "failure":
        result["success"] = False
    elif corruption == "missing_child":
        del result["method_metadata"]["info"]["children_info"]["S1F8"]
    elif corruption == "bad_seed":
        result["method_metadata"]["info"]["children_info"]["S1F8"]["hyperparameters"][
            "random_seed"
        ] = 0
    elif corruption == "nan_prediction":
        result["simulation_artifacts"]["pred_proba_dict_test"][spec["config_name"]][
            0
        ] = float("nan")
    elif corruption == "bad_folds":
        result["simulation_artifacts"]["bag_info"]["val_idx_per_child"][-1][0] = 0
    with gzip.open(path, "wb") as stream:
        pickle.dump(result, stream)
    with pytest.raises(ValueError):
        worker.validate_result_file(path, spec)


def test_real_tabarena_preserves_full_portfolio_config_names_and_seeds():
    pytest.importorskip("tabarena")
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle

    try:
        from tabarena.models.ctboost.hpo import generate_configs_ctboost
    except ImportError:
        pytest.skip("Requires the pinned CTBoost TabArena integration environment")
    experiments, provenance = worker.build_experiments(4)
    complete = TabArenaV0pt1ExperimentBundle(
        models=[("CTBoost", 200)], model_verbosity=2
    ).build_experiments(
        time_limit=3600,
        num_cpus=4,
        num_gpus=0,
        memory_limit=28,
    )
    assert [item.to_yaml_dict() for item in experiments] == [
        item.to_yaml_dict() for item in complete[:26]
    ]
    selected = TabArenaV0pt1ExperimentBundle(
        models=[experiments[-1]], model_verbosity=2
    ).build_experiments(
        time_limit=3600,
        num_cpus=4,
        num_gpus=0,
        memory_limit=28,
    )
    assert [item.to_yaml_dict() for item in selected] == [complete[25].to_yaml_dict()]
    assert provenance["portfolio_25_sha256"] == worker.json_hash(
        generate_configs_ctboost(200)[:25]
    )
    seeds = []
    for index, experiment in enumerate(experiments):
        record = experiment.to_yaml_dict()
        assert record["num_bag_folds"] == 8
        assert record["time_limit"] == 3600
        ensemble = record["model_hyperparameters"]["ag_args_ensemble"]
        assert ensemble["vary_seed_across_folds"] is True
        assert ensemble["model_random_seed"] == index * 8
        seeds.extend(
            range(ensemble["model_random_seed"], ensemble["model_random_seed"] + 8)
        )
    assert len(set(seeds)) == 208
