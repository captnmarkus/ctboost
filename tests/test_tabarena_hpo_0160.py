"""Synthetic evidence only: plan/provenance tests never launch official fits."""

import copy
import gzip
import hashlib
import json
import pickle
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.tabarena import hpo_0160 as worker
from benchmarks.tabarena.ctboost_model import generate_configs_ctboost


def release_fixture():
    return {
        "info": {"version": "0.1.60"},
        "urls": [
            {
                "filename": f"ctboost-0.1.60-cp312-cp312-{platform}.whl",
                "url": f"https://files.pythonhosted.org/fixture-{platform}.whl",
                "digests": {"sha256": digest * 64},
                "packagetype": "bdist_wheel",
                "upload_time_iso_8601": "2026-09-07T12:00:00Z",
                "yanked": False,
            }
            for platform, digest in [("win_amd64", "a"), ("manylinux_2_28_x86_64", "b")]
        ],
    }


@pytest.fixture
def plan_fixture(tmp_path, monkeypatch):
    population = [
        {"dataset": f"fixture-{i:02d}", "task_id": 99000 + i, "problem_type": "binary"}
        for i in range(51)
    ]
    names = [
        f"CTBoost_{'c1' if i == 0 else f'r{i}'}_default_BAG_L1" for i in range(201)
    ]
    experiments = [
        SimpleNamespace(
            name=name, to_yaml_dict=lambda name=name: {"name": name, "synthetic": True}
        )
        for name in names
    ]
    monkeypatch.setattr(worker, "_fetch_json", lambda _: release_fixture())
    monkeypatch.setattr(worker, "load_population", lambda _: population)
    monkeypatch.setattr(
        worker,
        "build_experiments",
        lambda: (experiments, [{}] + generate_configs_ctboost(200)),
    )
    output = tmp_path / "synthetic-plan"
    plan = worker.build_plan(
        output=output, metadata_path=tmp_path / "synthetic-metadata"
    )
    return plan, output


def test_plan_has_exact_baseline_portfolio_and_disjoint_predeclared_ownership(
    plan_fixture,
):
    plan, _ = plan_fixture
    assert len(plan["parents"]) == 10251
    assert len({p["parent_id"] for p in plan["parents"]}) == 10251
    assert sum(len(p["child_seeds"]) for p in plan["parents"]) == 82008
    assert sum(p["owner"] == "local" for p in plan["parents"]) == 4556
    assert sum(p["owner"] == "kaggle" for p in plan["parents"]) == 5695
    assert plan["configurations"][1:26] == generate_configs_ctboost(25)
    assert plan["parents"][51]["child_seeds"] == list(range(8, 16))
    assert plan["parents"][-1]["child_seeds"] == list(range(1600, 1608))
    assert all(
        not {
            "tabarena_learning_variant",
            "leaf_estimation_backtracking",
            "multiclass_leaf_solver",
            "multiclass_feature_test",
        }.intersection(c)
        for c in plan["configurations"]
    )
    assert worker.validate_plan(plan) is plan


@pytest.mark.parametrize(
    "mutation", ["owner", "seed", "config", "resource", "population", "source", "wheel"]
)
def test_plan_rejects_drift(plan_fixture, mutation):
    plan = copy.deepcopy(plan_fixture[0])
    if mutation == "owner":
        plan["parents"][0]["owner"] = "kaggle"
    elif mutation == "seed":
        plan["parents"][0]["child_seeds"][0] = 7
    elif mutation == "config":
        plan["configurations"][1]["max_depth"] = 100
    elif mutation == "resource":
        plan["resources"]["memory_limit_gb"] = 16
    elif mutation == "population":
        plan["population"].pop()
    elif mutation == "source":
        plan["source_sha256"][worker.SOURCE_FILES[0]] = "0" * 64
    else:
        plan["public_wheels"][0]["url"] = "https://example.invalid/not-pypi.whl"
    with pytest.raises(ValueError):
        worker.validate_plan(plan)


def test_plan_refuses_overwrite_and_unpublished_release(plan_fixture, monkeypatch):
    plan, output = plan_fixture
    assert worker.build_plan(output=output, metadata_path="synthetic") == plan
    monkeypatch.setattr(
        worker, "_fetch_json", lambda _: {"info": {"version": "0.1.59"}, "urls": []}
    )
    with pytest.raises(ValueError, match="0.1.60"):
        worker.build_plan(output=output, metadata_path="synthetic")
    assert worker.read_json(output / "plan.json") == plan


def test_source_and_plan_hashes_ignore_platform_newline_spelling(tmp_path):
    lf, crlf = tmp_path / "lf", tmp_path / "crlf"
    lf.write_bytes(b"one\ntwo\n")
    crlf.write_bytes(b"one\r\ntwo\r\n")
    assert worker.source_hash(lf) == worker.source_hash(crlf)
    assert worker.plan_hash(json.loads('{"a":1,\n"b":2}')) == worker.plan_hash(
        {"b": 2, "a": 1}
    )


def test_registration_requires_exact_public_plan_and_caches_verified_identity(
    plan_fixture, tmp_path, monkeypatch
):
    plan, _ = plan_fixture
    receipt = {
        "repository": "captnmarkus/ctboost",
        "commit": "a" * 40,
        "plan_path": "benchmarks/tabarena/hpo_0160_plan.json",
        "plan_sha256": worker.plan_hash(plan),
        "registered_at_utc": "2026-09-07T12:01:00Z",
    }
    path = tmp_path / "registration.json"
    worker._shared().write_json(path, receipt)
    calls = []

    def fetch(url):
        calls.append(url)
        return plan

    monkeypatch.setattr(worker, "_fetch_json", fetch)
    output = tmp_path / "execution"
    verified = worker.verify_registration(plan, path, output)
    assert worker.verify_registration(plan, path, output) == verified
    assert len(calls) == 1
    assert f"/{'a' * 40}/" in calls[0]
    receipt["commit"] = "b" * 40
    worker._shared().write_json(path, receipt)
    with pytest.raises(ValueError, match="different public registration"):
        worker.verify_registration(plan, path, output)


def test_registration_rejects_unpublished_or_changed_plan(
    plan_fixture, tmp_path, monkeypatch
):
    plan, _ = plan_fixture
    path = tmp_path / "registration.json"
    worker._shared().write_json(
        path,
        {
            "repository": "captnmarkus/ctboost",
            "commit": "a" * 40,
            "plan_path": "benchmarks/tabarena/hpo_0160_plan.json",
            "plan_sha256": worker.plan_hash(plan),
        },
    )
    monkeypatch.setattr(worker, "_fetch_json", lambda _: {"unrelated": True})
    with pytest.raises(ValueError, match="preregistered plan"):
        worker.verify_registration(plan, path, tmp_path / "execution")
    assert not (tmp_path / "execution/registration_verified.json").exists()


@pytest.mark.parametrize("mutation", [None, "python", "native", "wheel", "outside"])
def test_wheel_verification_binds_python_and_native_bytes(tmp_path, mutation):
    package = tmp_path / "site-packages/ctboost"
    package.mkdir(parents=True)
    members = {
        "ctboost/__init__.py": b"VERSION='0.1.60'\n",
        "ctboost/_core.pyd": b"synthetic-native",
    }
    for name, data in members.items():
        (package.parent / name).write_bytes(data)
    wheel = tmp_path / "synthetic.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    plan = {
        "public_wheels": [
            {
                "filename": wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            }
        ]
    }
    native = package / "_core.pyd"
    if mutation == "python":
        (package / "__init__.py").write_bytes(b"changed")
    elif mutation == "native":
        native.write_bytes(b"changed")
    elif mutation == "wheel":
        wheel.write_bytes(b"changed")
    elif mutation == "outside":
        native = tmp_path / "unapproved/_core.pyd"
    call = lambda: worker.verify_installed_wheel(
        plan, wheel, package_root=package, native_file=native
    )
    if mutation:
        with pytest.raises(ValueError):
            call()
    else:
        _, hashes = call()
        assert set(hashes) == set(members)


@pytest.fixture
def raw_fixture(tmp_path):
    parent = {
        "dataset": "synthetic",
        "task_id": 999,
        "problem_type": "binary",
        "config_index": 200,
        "config_name": "CTBoost_r200_default_BAG_L1",
        "repeat": 0,
        "fold": 0,
        "parent_id": "openml-999-r0-f0-c200",
        "owner": "local",
    }
    config = parent["config_name"]
    flags = {"is_fit": True, "is_valid": True, "can_infer": True}
    children = {
        f"S1F{i + 1}": {**flags, "hyperparameters": {"random_seed": 1600 + i}}
        for i in range(8)
    }
    result = {
        "framework": config,
        "metric_error": 0.3,
        "metric_error_val": 0.4,
        "problem_type": "binary",
        "task_metadata": {"name": "synthetic", "tid": 999, "repeat": 0, "fold": 0},
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
            "pred_proba_dict_val": {config: np.full(8, 0.5)},
            "y_val": np.arange(8) % 2,
            "pred_proba_dict_test": {config: np.array([0.4, 0.7])},
            "y_test": np.array([0, 1]),
            "bag_info": {
                "val_idx_per_child": [np.array([i]) for i in range(8)],
                "pred_proba_test_per_child": [np.array([0.4, 0.7]) for _ in range(8)],
            },
        },
        "hpo_0160": {
            "ctboost_version": worker.VERSION,
            "portfolio_id": worker.PORTFOLIO_ID,
            "plan_sha256": "a" * 64,
            "parent_id": parent["parent_id"],
            "owner": "local",
            "runtime_sha256": "b" * 64,
            "children": {
                f"S1F{i + 1}": {
                    "random_seed": 1600 + i,
                    "task_type": "CPU",
                    "feature_test": "quadratic",
                    "native_controls": dict(worker.BASELINE_CONTROLS),
                }
                for i in range(8)
            },
        },
    }
    path = tmp_path / config / "999/0_0/results.pkl"
    path.parent.mkdir(parents=True)
    return parent, result, path


@pytest.mark.parametrize(
    "mutation", [None, "owner", "version", "seed", "controls", "oof", "average"]
)
def test_result_audit_requires_new_version_all_fold_seeds_and_baseline(
    raw_fixture, mutation
):
    parent, result, path = raw_fixture
    if mutation == "owner":
        result["hpo_0160"]["owner"] = "kaggle"
    elif mutation == "version":
        result["hpo_0160"]["ctboost_version"] = "0.1.59"
    elif mutation == "seed":
        result["method_metadata"]["info"]["children_info"]["S1F1"]["hyperparameters"][
            "random_seed"
        ] = 0
    elif mutation == "controls":
        result["hpo_0160"]["children"]["S1F1"]["native_controls"][
            "multiclass_feature_test"
        ] = "joint"
    elif mutation == "oof":
        result["simulation_artifacts"]["bag_info"]["val_idx_per_child"][0] = np.array(
            [1]
        )
    elif mutation == "average":
        result["simulation_artifacts"]["bag_info"]["pred_proba_test_per_child"][0] = (
            np.array([0.2, 0.7])
        )
    with gzip.open(path, "wb") as stream:
        pickle.dump(result, stream)
    if mutation:
        with pytest.raises(ValueError):
            worker.validate_parent_result(path, parent, "a" * 64)
    else:
        assert (
            worker.validate_parent_result(path, parent, "a" * 64)["bag_children"] == 8
        )


@pytest.mark.parametrize("status", ["preparing", "running", "failed"])
def test_started_parent_never_retrains_after_restart(
    plan_fixture, tmp_path, monkeypatch, status
):
    monkeypatch.setitem(sys.modules, "psutil", None)
    plan, plan_output = plan_fixture
    parent = plan["parents"][0]
    output = tmp_path / "execution"
    directory = (
        output / "artifacts" / parent["config_name"] / str(parent["task_id"]) / "0_0"
    )
    worker._shared().write_json(
        directory / "manifest.json",
        {"status": status, "parent": parent, "plan_sha256": worker.plan_hash(plan)},
    )
    monkeypatch.setattr(worker._shared(), "_configure_process", lambda *_: [0, 1])
    monkeypatch.setattr(worker, "verify_registration", lambda *_: {})
    monkeypatch.setattr(worker, "runtime_provenance", lambda *_: {})
    monkeypatch.setattr(
        worker._shared(),
        "_load_task",
        lambda *_: pytest.fail("started parent was retrained"),
    )
    with pytest.raises(ValueError, match="no retries"):
        worker.run_parent(
            plan_path=plan_output / "plan.json",
            output=output,
            parent_id=parent["parent_id"],
            host="local",
            wheel_path="synthetic",
            registration_path="synthetic",
            affinity=[0, 1],
        )


def test_wrong_owner_is_rejected_before_runtime(plan_fixture, monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    plan, output = plan_fixture
    parent = plan["parents"][0]
    monkeypatch.setattr(
        worker,
        "runtime_provenance",
        lambda *_: pytest.fail("wrong owner reached runtime"),
    )
    with pytest.raises(ValueError, match="does not belong"):
        worker.run_parent(
            plan_path=output / "plan.json",
            output=output,
            parent_id=parent["parent_id"],
            host="kaggle",
            wheel_path="synthetic",
            registration_path="synthetic",
        )


@pytest.mark.parametrize("drift", [False, True])
def test_preflight_attests_actual_experiments_without_loading_tasks(
    plan_fixture, tmp_path, monkeypatch, drift
):
    plan, source = plan_fixture
    output = tmp_path / "execution"
    monkeypatch.setattr(worker, "verify_registration", lambda *_: {"verified": True})
    monkeypatch.setattr(worker, "runtime_provenance", lambda *_: {"synthetic": True})
    monkeypatch.setattr(
        worker._shared(),
        "_load_task",
        lambda *_: pytest.fail("preflight loaded a task"),
    )
    if drift:
        experiments, configs = worker.build_experiments()
        experiments.pop()
        monkeypatch.setattr(worker, "build_experiments", lambda: (experiments, configs))
    call = lambda: worker.preflight(
        plan_path=source / "plan.json",
        output=output,
        host="kaggle",
        wheel_path="synthetic",
        registration_path="synthetic",
    )
    if drift:
        with pytest.raises(ValueError, match="definitions changed"):
            call()
        assert not (output / "preflight-kaggle.json").exists()
    else:
        evidence = call()
        assert evidence == worker.read_json(output / "preflight-kaggle.json")
        assert evidence["plan_sha256"] == worker.plan_hash(plan)
        assert evidence["runtime_sha256"] == worker.plan_hash({"synthetic": True})
        assert evidence["experiment_count"] == 201
        assert evidence["fit_started"] is False


@pytest.mark.parametrize("problem", ["binary", "multiclass", "regression"])
def test_explicit_synthetic_eight_child_parent_and_native_audit(tmp_path, problem):
    """Run only 96 synthetic rows and two rounds through the actual pinned API."""
    pytest.importorskip(
        "tabarena", reason="Optional pinned TabArena benchmark environment"
    )
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
            return 1, 1, 1

        def get_split_indices(self, fold=0, repeat=0, sample=0):
            return np.arange(64), np.arange(64, 96)

    experiments, _ = worker.build_experiments()
    experiment = experiments[200]
    # Explicit synthetic preflight only: never alter a production definition.
    experiment.method_kwargs["model_hyperparameters"].update(
        iterations=2, max_depth=1, max_bins=8, min_data_in_leaf=2
    )
    experiment.method_kwargs["init_kwargs"].update(
        path=str(tmp_path / "models"), verbosity=0
    )
    parent = {
        "dataset": "explicit_synthetic_fixture",
        "task_id": 999,
        "problem_type": problem,
        "config_index": 200,
        "config_name": experiment.name,
        "repeat": 0,
        "fold": 0,
        "parent_id": "openml-999-r0-f0-c200",
        "owner": "local",
    }
    experiment.experiment_cls = worker._audited_runner(parent, "a" * 64, "b" * 64)
    directory = tmp_path / "data" / experiment.name / "999" / "0_0"
    cacher = CacheFunctionPickle(
        cache_path=directory, cache_name="results", include_self_in_call=True
    )
    experiment.run(
        task=SyntheticTask(),
        fold=0,
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
        directory / "results.pkl", parent, "a" * 64
    )
    assert validation["bag_children"] == 8
    assert [
        item["random_seed"] for item in validation["native_children"].values()
    ] == list(range(1600, 1608))
    assert validation["runtime_sha256"] == "b" * 64
    experiment.run(
        task=None,
        fold=0,
        task_name=parent["dataset"],
        cache_task_key=999,
        cacher=cacher,
    )
    assert (
        worker._shared().file_hash(directory / "results.pkl")
        == validation["raw_sha256"]
    )
