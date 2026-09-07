"""Full outer-split protocol tests use synthetic artifacts, never official fits."""

import copy
import gzip
import pickle
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks.tabarena import hpo_full_0160 as worker
from benchmarks.tabarena.ctboost_model import generate_configs_ctboost


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    inventory = [
        {
            "dataset": f"fixture-{i:02d}",
            "task_id": 99000 + i,
            "problem_type": "binary",
            "num_instances_train": 8,
        }
        for i in range(51)
    ]
    splits = [
        {**p, "repeat": repeat, "fold": fold, "num_instances_test": 2}
        for index, p in enumerate(inventory)
        for repeat in range(3 if index < 34 else 10)
        for fold in range(3)
    ]
    names = [f"CTBoost_{'c1' if i == 0 else f'r{i}'}_default_BAG_L1" for i in range(26)]
    experiments = [
        SimpleNamespace(name=name, to_yaml_dict=lambda name=name: {"name": name})
        for name in names
    ]
    release = {
        "info": {"version": "0.1.60"},
        "urls": [
            {
                "filename": f"ctboost-0.1.60-cp312-cp312-{platform}.whl",
                "url": f"https://files.pythonhosted.org/{platform}.whl",
                "digests": {"sha256": digest * 64},
                "packagetype": "bdist_wheel",
                "upload_time_iso_8601": "2026-09-07T00:00:00Z",
                "yanked": False,
            }
            for platform, digest in [("win_amd64", "a"), ("manylinux_2_28_x86_64", "b")]
        ],
    }
    monkeypatch.setattr(worker._base(), "_fetch_json", lambda _: release)
    monkeypatch.setattr(worker._base(), "load_population", lambda _: inventory)
    monkeypatch.setattr(worker, "load_population", lambda _: splits)
    monkeypatch.setattr(worker, "FULL_SPLITS_SHA256", worker.plan_hash(splits))
    monkeypatch.setattr(
        worker,
        "build_experiments",
        lambda: (experiments, [{}] + generate_configs_ctboost(25)),
    )
    output = tmp_path / "synthetic-plan"
    plan = worker.build_plan(output=output, metadata_path="synthetic")
    return plan, output, experiments


def test_full_membership_order_and_seeds(plan_fixture):
    plan, _, _ = plan_fixture
    parents = plan["parents"]
    assert len(parents) == len({p["parent_id"] for p in parents}) == 21216
    assert sum(p["owner"] == "local" for p in parents) == 9431
    assert sum(p["owner"] == "kaggle" for p in parents) == 11785
    assert all(p["repeat"] == p["fold"] == p["config_index"] == 0 for p in parents[:51])
    assert parents[51]["repeat"] == 0 and parents[51]["fold"] == 1
    assert parents[816]["config_index"] == 1 and parents[816]["child_seeds"] == list(
        range(8, 16)
    )
    assert parents[-1]["repeat"] == 9 and parents[-1]["fold"] == 2
    assert parents[-1]["child_seeds"] == list(range(200, 208))
    assert sum(len(p["child_seeds"]) for p in parents) == 169728


@pytest.mark.parametrize(
    "change",
    ["repeat", "fold", "owner", "seed", "config", "scope", "source", "resource"],
)
def test_full_plan_rejects_protocol_drift(plan_fixture, change):
    plan = copy.deepcopy(plan_fixture[0])
    if change in {"repeat", "fold"}:
        plan["parents"][0][change] = 1
    elif change == "owner":
        plan["parents"][0]["owner"] = "kaggle"
    elif change == "seed":
        plan["parents"][0]["child_seeds"][0] = 999
    elif change == "config":
        plan["configurations"][1]["max_depth"] = 100
    elif change == "scope":
        plan["outer_splits"][0]["repeat"] = 19
        plan["outer_splits_sha256"] = worker.plan_hash(plan["outer_splits"])
    elif change == "source":
        plan["source_sha256"][worker.SOURCE_FILES[0]] = "0" * 64
    else:
        plan["resources"]["num_cpus"] = 8
    with pytest.raises(ValueError):
        worker.validate_plan(plan)


def result_fixture(parent):
    flags = {"is_fit": True, "is_valid": True, "can_infer": True}
    start = parent["config_index"] * 8
    children = {
        f"S1F{i + 1}": {**flags, "hyperparameters": {"random_seed": start + i}}
        for i in range(8)
    }
    name = parent["config_name"]
    return {
        "framework": name,
        "problem_type": "binary",
        "metric_error": 0.3,
        "metric_error_val": 0.4,
        "time_train_s": 1.0,
        "time_infer_s": 0.01,
        "task_metadata": {
            "name": parent["dataset"],
            "tid": parent["task_id"],
            "repeat": parent["repeat"],
            "fold": parent["fold"],
        },
        "method_metadata": {
            "info": {
                **flags,
                "children_info": children,
                "bagged_info": {
                    "num_child_models": 8,
                    "child_model_names": list(children),
                },
            }
        },
        "simulation_artifacts": {
            "pred_proba_dict_val": {name: np.full(8, 0.5)},
            "y_val": np.arange(8) % 2,
            "pred_proba_dict_test": {name: np.array([0.4, 0.7])},
            "y_test": np.array([0, 1]),
            "bag_info": {
                "val_idx_per_child": [np.array([i]) for i in range(8)],
                "pred_proba_test_per_child": [np.array([0.4, 0.7]) for _ in range(8)],
            },
        },
        "hpo_full_0160": {
            "ctboost_version": "0.1.60",
            "portfolio_id": worker.PORTFOLIO_ID,
            "plan_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
            "parent_id": parent["parent_id"],
            "owner": parent["owner"],
            "repeat": parent["repeat"],
            "fold": parent["fold"],
            "sample": 0,
            "outer_split": worker._outer_split_receipt(
                SimpleNamespace(
                    get_split_indices=lambda **_: (np.arange(8), np.arange(8, 10))
                ),
                parent,
            ),
            "children": {
                f"S1F{i + 1}": {
                    "random_seed": start + i,
                    "native_controls": dict(worker._base().BASELINE_CONTROLS),
                    "task_type": "CPU",
                    "feature_test": "quadratic",
                }
                for i in range(8)
            },
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        None,
        "outer_repeat",
        "outer_fold",
        "path",
        "audit",
        "seed",
        "rows",
        "oof",
        "child_mean",
        "controls",
        "time",
        "receipt",
    ],
)
def test_raw_validation_requires_exact_nonzero_outer_split(tmp_path, change):
    parent = {
        "parent_id": "openml-999-r2-f1-c025",
        "dataset": "synthetic",
        "task_id": 999,
        "repeat": 2,
        "fold": 1,
        "config_index": 25,
        "config_name": "CTBoost_r25_default_BAG_L1",
        "owner": "local",
        "problem_type": "binary",
        "num_instances_train": 8.333333333333,
        "num_instances_test": 1.666666666667,
    }
    result = result_fixture(parent)
    subpath = "0_0" if change == "path" else "2_1"
    path = tmp_path / parent["config_name"] / "999" / subpath / "results.pkl"
    path.parent.mkdir(parents=True)
    if change in {"outer_repeat", "outer_fold"}:
        result["task_metadata"][change.removeprefix("outer_")] = 0
    elif change == "audit":
        result["hpo_full_0160"]["repeat"] = 0
    elif change == "seed":
        result["method_metadata"]["info"]["children_info"]["S1F1"]["hyperparameters"][
            "random_seed"
        ] = 216
    elif change == "rows":
        result["simulation_artifacts"]["y_test"] = np.array([0])
    elif change == "oof":
        result["simulation_artifacts"]["bag_info"]["val_idx_per_child"][0] = np.array(
            [1]
        )
    elif change == "child_mean":
        result["simulation_artifacts"]["bag_info"]["pred_proba_test_per_child"][0] = (
            np.array([0.8, 0.7])
        )
    elif change == "controls":
        result["hpo_full_0160"]["children"]["S1F1"]["native_controls"][
            "leaf_estimation_iterations"
        ] = 3
    elif change == "time":
        result["time_train_s"] = float("nan")
    elif change == "receipt":
        result["hpo_full_0160"]["outer_split"]["test_rows"] = 1
    with gzip.open(path, "wb") as stream:
        pickle.dump(result, stream)
    if change:
        with pytest.raises(ValueError):
            worker.validate_parent_result(path, parent, "a" * 64)
    else:
        result = worker.validate_parent_result(path, parent, "a" * 64)
        assert (
            result["repeat"] == 2
            and result["fold"] == 1
            and result["bag_children"] == 8
        )


def test_worker_passes_true_outer_split_and_revalidates_only_matching_cache(
    plan_fixture, tmp_path, monkeypatch
):
    plan, plan_output, experiments = plan_fixture
    parent = next(
        p
        for p in plan["parents"]
        if p["repeat"] == 2 and p["fold"] == 1 and p["owner"] == "local"
    )
    output = tmp_path / "execution"
    raw = (
        output
        / "data"
        / parent["config_name"]
        / str(parent["task_id"])
        / "2_1/results.pkl"
    )
    result = result_fixture(parent)
    result["hpo_full_0160"].update(
        plan_sha256=worker.plan_hash(plan), runtime_sha256=worker.plan_hash({})
    )
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        raw.parent.mkdir(parents=True)
        with gzip.open(raw, "wb") as stream:
            pickle.dump(result, stream)

    experiments[parent["config_index"]].run = execute
    monkeypatch.setattr(worker._shared(), "_configure_process", lambda *_: [0, 1])
    monkeypatch.setattr(worker, "runtime_provenance", lambda *_: {})
    monkeypatch.setattr(worker, "verify_registration", lambda *_: {})
    monkeypatch.setattr(worker, "_audited_runner", lambda *_: object)
    spec = SimpleNamespace(
        cache_key=parent["task_id"], resolve_task_name=lambda _: parent["dataset"]
    )
    monkeypatch.setattr(
        worker,
        "_load_task",
        lambda *_: (
            spec,
            SimpleNamespace(
                eval_metric="roc_auc",
                get_split_indices=lambda **_: (np.arange(8), np.arange(8, 10)),
            ),
        ),
    )
    cache = ModuleType("tabarena.utils.cache")
    cache.CacheFunctionPickle = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "tabarena.utils.cache", cache)

    @contextmanager
    def limits(_):
        yield {"peak_rss_bytes": 1024}

    monkeypatch.setattr(worker._base(), "_limits", limits)
    kwargs = {
        "plan_path": plan_output / "plan.json",
        "output": output,
        "parent_id": parent["parent_id"],
        "host": "local",
        "affinity": [0, 1],
        "wheel_path": "synthetic",
        "registration_path": "synthetic",
    }
    manifest = worker.run_parent(**kwargs)
    assert manifest["status"] == "complete"
    assert calls[0]["repeat"] == 2 and calls[0]["fold"] == 1 and calls[0]["sample"] == 0
    assert worker.run_parent(**kwargs) == manifest and len(calls) == 1
    path = (
        output
        / "artifacts"
        / parent["config_name"]
        / str(parent["task_id"])
        / "2_1/manifest.json"
    )
    manifest["status"] = "running"
    worker._shared().write_json(path, manifest)
    with pytest.raises(ValueError, match="retries are disabled"):
        worker.run_parent(**kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize("repeat,fold", [(3, 0), (0, 3), (-1, 0)])
def test_task_loader_refuses_nonexistent_official_split(monkeypatch, repeat, fold):
    monkeypatch.setattr(
        worker._shared(),
        "_load_task",
        lambda *_: (None, SimpleNamespace(get_split_dimensions=lambda: (3, 3, 1))),
    )
    with pytest.raises(ValueError, match="does not provide"):
        worker._load_task({"repeat": repeat, "fold": fold}, None)


@pytest.mark.parametrize("problem", ["binary", "multiclass", "regression"])
def test_explicit_synthetic_nonzero_outer_split_with_actual_tabarena(tmp_path, problem):
    """Two rounds on96 synthetic rows: no official task data or production plan."""
    pytest.importorskip("tabarena", reason="Optional pinned TabArena environment")
    import pandas as pd
    from tabarena.benchmark.task.wrapper import TaskWrapper
    from tabarena.utils.cache import CacheFunctionPickle

    rng = np.random.default_rng(160)
    features = pd.DataFrame(rng.normal(size=(96, 3)), columns=["x", "z", "w"])
    labels = pd.Series(
        np.arange(96) % (3 if problem == "multiclass" else 2), name="label"
    )
    if problem == "regression":
        labels = pd.Series(features["x"] * 2 + features["z"], name="label")

    class SyntheticTask(TaskWrapper):
        problem_type = problem
        label = "label"

        def _load_data(self):
            return features, labels

        @property
        def task_id(self):
            return 999

        def get_split_dimensions(self):
            return 3, 3, 1

        def get_split_indices(self, fold=0, repeat=0, sample=0):
            indices = np.roll(np.arange(96), (repeat * 3 + fold) * 7)
            return indices[:64], indices[64:]

    experiments, _ = worker.build_experiments()
    experiment = experiments[25]
    # Only this explicit test reduces the training budget; production stays frozen.
    experiment.method_kwargs["model_hyperparameters"].update(
        iterations=2, max_depth=1, max_bins=8, min_data_in_leaf=2
    )
    experiment.method_kwargs["init_kwargs"].update(
        path=str(tmp_path / "models"), verbosity=0
    )
    parent = {
        "parent_id": "openml-999-r2-f1-c025",
        "dataset": "explicit_synthetic_full",
        "task_id": 999,
        "problem_type": problem,
        "config_index": 25,
        "config_name": experiment.name,
        "repeat": 2,
        "fold": 1,
        "owner": "local",
        "num_instances_train": 64.33333333333,
        "num_instances_test": 31.66666666667,
    }
    task = SyntheticTask()
    receipt = worker._outer_split_receipt(task, parent)
    assert receipt["train_rows"] == 64 and receipt["test_rows"] == 32
    assert receipt != worker._outer_split_receipt(
        task, {**parent, "repeat": 0, "fold": 0}
    )
    experiment.experiment_cls = worker._audited_runner(
        parent, "a" * 64, "b" * 64, receipt
    )
    directory = tmp_path / "data" / experiment.name / "999/2_1"
    cacher = CacheFunctionPickle(
        cache_path=directory, cache_name="results", include_self_in_call=True
    )
    experiment.run(
        task=task,
        fold=1,
        repeat=2,
        sample=0,
        task_name=parent["dataset"],
        cache_task_key=999,
        cacher=cacher,
        raise_on_failure=True,
        debug_mode=False,
        eval_metric_name={
            "binary": "roc_auc",
            "multiclass": "log_loss",
            "regression": "rmse",
        }[problem],
    )
    validation = worker.validate_parent_result(
        directory / "results.pkl", parent, "a" * 64, outer_split=receipt
    )
    assert (
        validation["repeat"] == 2
        and validation["fold"] == 1
        and validation["bag_children"] == 8
    )
    assert validation["outer_split"] == receipt
    assert [c["random_seed"] for c in validation["native_children"].values()] == list(
        range(200, 208)
    )
    wrong_receipt = worker._outer_split_receipt(
        task, {**parent, "repeat": 0, "fold": 0}
    )
    with pytest.raises(ValueError, match="pre-fit manifest"):
        worker.validate_parent_result(
            directory / "results.pkl", parent, "a" * 64, outer_split=wrong_receipt
        )
