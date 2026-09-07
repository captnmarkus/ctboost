"""No official data or fits: all positive gates below use explicit synthetic fixtures."""

from __future__ import annotations

import copy
import gzip
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from benchmarks.tabarena import local_hpo as worker
from benchmarks.tabarena.ctboost_model import generate_configs_ctboost_learning_options
from benchmarks.tabarena.pilot_evaluation import evaluate_pilot, expected_pilot_jobs


def _fixture_metric(labels, predictions, dataset):
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

    if dataset["problem_type"] == "binary":
        return float(1 - roc_auc_score(labels, predictions))
    if dataset["problem_type"] == "multiclass":
        return float(
            log_loss(labels, predictions, labels=np.arange(dataset["num_classes"]))
        )
    return float(np.sqrt(mean_squared_error(labels, predictions)))


@pytest.fixture
def synthetic_gate_fixture(tmp_path, monkeypatch):
    """A fake local workspace, fake pilot artifacts, and fake runtime, never official evidence."""
    from benchmarks.tabarena.local_pilot import indices_hash

    real_root = worker.ROOT
    root = tmp_path / "EXPLICIT_SYNTHETIC_GATE_FIXTURE"
    protocol = copy.deepcopy(worker.read_json(real_root / worker.PROTOCOL_FILE))
    # The canonical bytes exist only inside this monkeypatched unit-test workspace.
    protocol_path = root / worker.PROTOCOL_FILE
    worker.write_json(protocol_path, protocol)
    for name in worker.PILOT_SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((real_root / name).read_bytes())
    monkeypatch.setattr(worker, "ROOT", root)
    monkeypatch.setattr(worker, "_pilot_metric_error", _fixture_metric)
    protocol_hash = worker.file_hash(protocol_path)
    pilot = root / "fake_pilot_records"
    runtime = {
        "packages": {},
        "environment_sha256": worker.json_hash({}),
        "source_sha256": {
            name: worker.file_hash(root / name) for name in worker.PILOT_SOURCE_FILES
        },
        "native_extension_sha256": "a" * 64,
    }
    resources = {"synthetic_fixture_only": True}
    resource_hash = worker.json_hash(resources)
    prepared = []
    for dataset in protocol["datasets"]:
        name = dataset["dataset_name"]
        validation_rows = max(2, dataset["num_classes"]) * 3
        outer_rows = validation_rows * 8
        directory = pilot / "data" / name
        directory.mkdir(parents=True)
        (directory / "outer_train.pkl").write_bytes(
            b"synthetic placeholder; never unpickle"
        )
        record = {
            "dataset": name,
            "task_id": dataset["task_id"],
            "problem_type": dataset["problem_type"],
            "protocol_sha256": protocol_hash,
            "contains_outer_test_rows": False,
            "rows_outer_train": outer_rows,
            "data_sha256": worker.file_hash(directory / "outer_train.pkl"),
            "splits": {
                str(fold): {
                    "rows_train": validation_rows * 7,
                    "rows_validation": validation_rows,
                    "sha256": indices_hash(
                        np.setdiff1d(
                            np.arange(outer_rows),
                            np.arange(
                                fold * validation_rows, (fold + 1) * validation_rows
                            ),
                        ),
                        np.arange(fold * validation_rows, (fold + 1) * validation_rows),
                    ),
                }
                for fold in protocol["splits"]["inner_fold_indices"]
            },
        }
        worker.write_json(directory / "prepared.json", record)
        prepared.append(record)
    execution = {
        "protocol_sha256": protocol_hash,
        "runtime": runtime,
        "resources": resources,
        "resource_contract_sha256": resource_hash,
        "prepared": {
            "complete": True,
            "protocol_sha256": protocol_hash,
            "datasets": prepared,
        },
    }
    worker.write_json(pilot / "execution.json", execution)
    datasets = {row["dataset_name"]: row for row in protocol["datasets"]}
    prepared_by_name = {row["dataset"]: row for row in prepared}
    records = []
    paths = []
    for job in expected_pilot_jobs(protocol):
        dataset = datasets[job["dataset"]]
        count = max(2, dataset["num_classes"])
        labels = np.arange(count * 3) % count
        baseline = job["variant"] == "baseline"
        if dataset["problem_type"] == "multiclass":
            predictions = np.full(
                (len(labels), count), 1 / count if baseline else 0.1 / (count - 1)
            )
            if not baseline:
                predictions[np.arange(len(labels)), labels] = 0.9
        elif dataset["problem_type"] == "binary":
            predictions = (
                np.full(len(labels), 0.5)
                if baseline
                else np.where(labels == 1, 0.9, 0.1)
            )
        else:
            predictions = labels + (2.0 if baseline else 1.0)
        path = (
            pilot
            / "fits"
            / job["dataset"]
            / f"inner{job['inner_fold']}"
            / f"{job['variant']}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            predictions=predictions,
            labels=labels,
            validation_indices=np.arange(
                job["inner_fold"] * len(labels), (job["inner_fold"] + 1) * len(labels)
            ),
        )
        item = prepared_by_name[job["dataset"]]
        split_hash = item["splits"][str(job["inner_fold"])]["sha256"]
        record = {
            **job,
            "status": "ok",
            "protocol_sha256": protocol_hash,
            "split_sha256": split_hash,
            "resource_contract_sha256": resource_hash,
            "preprocessing_sha256": worker.json_hash(
                {
                    "data_sha256": item["data_sha256"],
                    "split_sha256": split_hash,
                    "adapter_sha256": runtime["source_sha256"][
                        "benchmarks/tabarena/ctboost_model.py"
                    ],
                }
            ),
            "execution_runtime_sha256": worker.json_hash(runtime),
            "prediction_sha256": worker.file_hash(path.with_suffix(".npz")),
            "metric_error": _fixture_metric(labels, predictions, dataset),
            "fit_seconds": 1.0,
            "peak_rss_bytes": 1024,
            "deadline_stopped": False,
        }
        worker.write_json(path, record)
        records.append(record)
        paths.append(path)
    decision = evaluate_pilot(protocol, records, protocol_sha256=protocol_hash)
    worker.write_json(pilot / "decision.json", decision)
    return SimpleNamespace(
        root=root,
        protocol=protocol,
        protocol_path=protocol_path,
        decision_path=pilot / "decision.json",
        decision=decision,
        records=records,
        record_paths=paths,
        pilot=pilot,
    )


def test_gate_recomputes_complete_decision_and_binds_every_artifact(
    synthetic_gate_fixture,
):
    fixture = synthetic_gate_fixture
    gate = worker.verify_pilot_gate(fixture.decision_path, fixture.protocol_path)
    assert gate["decision"]["full_hpo_justified"]
    assert len(gate["binding"]["pilot_evidence"]) == 14 * 2 + 88 * 2
    assert gate["binding"]["decision_sha256"] == worker.file_hash(fixture.decision_path)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "stale", "false_boolean"])
def test_plan_refuses_incomplete_forged_or_stale_decision_before_runtime(
    synthetic_gate_fixture,
    monkeypatch,
    mutation,
):
    fixture = synthetic_gate_fixture
    if mutation == "missing":
        fixture.record_paths[0].unlink()
    elif mutation == "duplicate":
        path = fixture.record_paths[0]
        (path.parent / "duplicate.json").write_bytes(path.read_bytes())
    else:
        decision = copy.deepcopy(fixture.decision)
        if mutation == "stale":
            decision["approved_modes_by_problem_type"]["binary"] = "baseline"
        else:
            decision["full_hpo_justified"] = False
        worker.write_json(fixture.decision_path, decision)
    monkeypatch.setattr(
        worker,
        "runtime_provenance",
        lambda *_: pytest.fail("runtime reached before gate"),
    )
    output = fixture.root / "must_not_be_created"
    with pytest.raises(ValueError, match="decision"):
        worker.build_plan(
            decision_path=fixture.decision_path,
            protocol_path=fixture.protocol_path,
            output=output,
        )
    assert not output.exists()


def test_complete_no_gain_pilot_cannot_launch_duplicate_hpo(synthetic_gate_fixture):
    fixture = synthetic_gate_fixture
    for record in fixture.records:
        record["metric_error"] = 0.4
    for path, record in zip(fixture.record_paths, fixture.records):
        worker.write_json(path, record)
    worker.write_json(
        fixture.decision_path,
        evaluate_pilot(
            fixture.protocol,
            fixture.records,
            protocol_sha256=worker.file_hash(fixture.protocol_path),
        ),
    )
    with pytest.raises(ValueError, match="complete, globally valid, passing"):
        worker.verify_pilot_gate(fixture.decision_path, fixture.protocol_path)


@pytest.mark.parametrize(
    "kind", ["prediction", "source", "prepared", "record_runtime", "metric"]
)
def test_passing_boolean_cannot_hide_changed_evidence(synthetic_gate_fixture, kind):
    fixture = synthetic_gate_fixture
    path = fixture.record_paths[0]
    if kind == "prediction":
        path.with_suffix(".npz").write_bytes(b"changed")
    elif kind == "source":
        (fixture.root / worker.PILOT_SOURCE_FILES[1]).write_bytes(b"changed")
    elif kind == "prepared":
        next((fixture.pilot / "data").glob("*/outer_train.pkl")).write_bytes(b"changed")
    else:
        record = fixture.records[0]
        if kind == "record_runtime":
            record["execution_runtime_sha256"] = "b" * 64
        else:
            record["metric_error"] += 0.1
        worker.write_json(path, record)
        worker.write_json(
            fixture.decision_path,
            evaluate_pilot(
                fixture.protocol,
                fixture.records,
                protocol_sha256=worker.file_hash(fixture.protocol_path),
            ),
        )
    with pytest.raises(ValueError):
        worker.verify_pilot_gate(fixture.decision_path, fixture.protocol_path)


@pytest.mark.parametrize("mutation", ["wrong_order", "outside_outer_training"])
def test_metric_preserving_index_tampering_cannot_pass_gate(
    synthetic_gate_fixture, mutation
):
    fixture = synthetic_gate_fixture
    path = fixture.record_paths[0]
    prediction_path = path.with_suffix(".npz")
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if mutation == "wrong_order":
        arrays["validation_indices"] = arrays["validation_indices"][::-1]
    else:
        arrays["validation_indices"][0] = 1000000
    np.savez_compressed(prediction_path, **arrays)
    fixture.records[0]["prediction_sha256"] = worker.file_hash(prediction_path)
    worker.write_json(path, fixture.records[0])
    # The decision and quality values are unchanged; only split binding detects this.
    with pytest.raises(ValueError, match="validation indices"):
        worker.verify_pilot_gate(fixture.decision_path, fixture.protocol_path)


class FixtureExperiment:
    def __init__(self, index):
        self.name = f"CTBoost_{'c1' if index == 0 else f'r{index}'}_default_BAG_L1"

    def to_yaml_dict(self):
        return {"name": self.name, "synthetic_fixture_only": True}


def _fixture_experiments(modes, _resources):
    return [FixtureExperiment(i) for i in range(26)], [
        {}
    ] + generate_configs_ctboost_learning_options(
        approved_modes_by_problem_type=modes,
    )


def test_immutable_plan_has_exact_1326_parents_and_10608_seeds(
    synthetic_gate_fixture, monkeypatch
):
    fixture = synthetic_gate_fixture
    monkeypatch.setattr(
        worker, "runtime_provenance", lambda *_: {"synthetic_fixture_only": True}
    )
    monkeypatch.setattr(worker, "build_experiments", _fixture_experiments)
    kwargs = dict(
        decision_path=fixture.decision_path,
        protocol_path=fixture.protocol_path,
        output=fixture.root / "fake_full_hpo_plan",
    )
    plan = worker.build_plan(**kwargs)
    assert plan["expected_parent_count"] == len(plan["parents"]) == 1326
    assert (
        len({(item["dataset"], item["config_index"]) for item in plan["parents"]})
        == 1326
    )
    assert sum(len(item["child_seeds"]) for item in plan["parents"]) == 10608
    assert plan["configurations"][0] == {}
    assert plan["parents"][51]["child_seeds"] == list(range(8, 16))
    assert worker.build_plan(**kwargs) == plan
    with pytest.raises(ValueError, match="Existing HPO plan differs"):
        worker.build_plan(**kwargs, memory_limit_gb=4)


@pytest.mark.parametrize(
    "cpus,memory", [(0, 8), (17, 8), (True, 8), (2, 0.5), (2, 9), (2, float("nan"))]
)
def test_invalid_resources_are_rejected(cpus, memory):
    with pytest.raises(ValueError):
        worker.resource_contract(cpus, memory)


def test_atomic_checkpoint_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    worker.write_json(path, {"status": "running"})
    replace = Path.replace
    attempts = []

    def locked_once(temporary, destination):
        attempts.append(destination)
        if len(attempts) == 1:
            raise PermissionError("synthetic Windows reader lock")
        return replace(temporary, destination)

    monkeypatch.setattr(Path, "replace", locked_once)
    worker.write_json(path, {"status": "complete"})
    assert worker.read_json(path) == {"status": "complete"}
    assert len(attempts) == 2
    assert list(tmp_path.iterdir()) == [path]


def test_memory_overflow_terminates_owned_worker_even_if_checkpoint_fails(
    tmp_path, monkeypatch
):
    def failed_checkpoint(*_):
        raise PermissionError("synthetic unavailable output")

    def fake_exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(worker, "write_json", failed_checkpoint)
    monkeypatch.setattr(worker.os, "_exit", fake_exit)
    with pytest.raises(SystemExit) as exit_info:
        worker._terminate_memory_overflow(tmp_path / "resource_failure.json", 9, 8)
    assert exit_info.value.code == 3


def _fake_native_child(index, *, wrong_control=False):
    from benchmarks.tabarena.learning_options import resolve_cpu_learning_options

    params = {"random_seed": index, "feature_test": "quadratic"}
    resolved, _ = resolve_cpu_learning_options(
        params, problem_type="binary", num_classes=2, variant="baseline"
    )
    controls = {name: resolved[name] for name in worker.CONTROL_NAMES}
    if wrong_control:
        controls["multiclass_feature_test"] = "joint"
    handle = SimpleNamespace(
        **{name: (lambda value=value: value) for name, value in controls.items()}
    )
    model = SimpleNamespace(
        n_classes_=2,
        get_params=lambda **_: {**resolved, "task_type": "CPU"},
        get_booster=lambda: SimpleNamespace(_handle=handle),
    )
    return SimpleNamespace(
        name=f"S1F{index + 1}", params=params, model=model, problem_type="binary"
    )


def test_audit_reads_actual_native_controls_and_rejects_mismatch():
    children = [_fake_native_child(index) for index in range(8)]
    bag = SimpleNamespace(
        models=[child.name for child in children],
        load_child=lambda name: children[int(name[3:]) - 1],
    )
    wrapper = SimpleNamespace(_load_model=lambda: bag)
    audit = worker.audit_children(wrapper, {"config_index": 0})
    assert len(audit) == 8
    children[3] = _fake_native_child(3, wrong_control=True)
    with pytest.raises(ValueError, match="Native learning controls"):
        worker.audit_children(wrapper, {"config_index": 0})


@pytest.fixture
def synthetic_raw_result(tmp_path):
    parent = {
        "dataset": "explicit_synthetic_fixture",
        "task_id": 999,
        "problem_type": "binary",
        "config_index": 1,
        "config_name": "CTBoost_r1_default_BAG_L1",
        "repeat": 0,
        "fold": 0,
    }
    config = parent["config_name"]
    flags = {"is_fit": True, "is_valid": True, "can_infer": True}
    children = {
        f"S1F{i + 1}": {**flags, "hyperparameters": {"random_seed": 8 + i}}
        for i in range(8)
    }
    controls = {
        "leaf_estimation_iterations": 3,
        "leaf_estimation_backtracking": True,
        "multiclass_leaf_solver": "diagonal",
        "multiclass_feature_test": "single",
    }
    result = {
        "framework": config,
        "metric_error": 0.3,
        "metric_error_val": 0.4,
        "problem_type": "binary",
        "task_metadata": {
            "name": parent["dataset"],
            "tid": 999,
            "repeat": 0,
            "fold": 0,
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
            "pred_proba_dict_val": {config: np.full(8, 0.5)},
            "y_val": np.arange(8) % 2,
            "pred_proba_dict_test": {config: np.array([0.4, 0.7])},
            "y_test": np.array([0, 1]),
            "bag_info": {
                "val_idx_per_child": [np.array([i]) for i in range(8)],
                "pred_proba_test_per_child": [np.array([0.4, 0.7]) for _ in range(8)],
            },
        },
        "local_hpo": {
            "ctboost_version": "0.1.59",
            "portfolio_id": worker.PORTFOLIO_ID,
            "plan_sha256": "a" * 64,
            "children": {
                f"S1F{i + 1}": {
                    "task_type": "CPU",
                    "random_seed": 8 + i,
                    "native_controls": controls,
                    "applicability": {"resolved_controls": controls},
                }
                for i in range(8)
            },
        },
    }
    path = tmp_path / config / "999" / "0_0" / "results.pkl"
    path.parent.mkdir(parents=True)
    return parent, result, path


def _save_raw(result, path):
    with gzip.open(path, "wb") as stream:
        pickle.dump(result, stream)


def test_result_requires_version_plan_children_predictions_and_exact_seeds(
    synthetic_raw_result,
):
    parent, result, path = synthetic_raw_result
    _save_raw(result, path)
    assert worker.validate_result(path, parent, "a" * 64)["bag_children"] == 8
    result["method_metadata"]["info"]["children_info"]["S1F4"]["hyperparameters"][
        "random_seed"
    ] = 0
    _save_raw(result, path)
    with pytest.raises(ValueError, match="seeds"):
        worker.validate_result(path, parent, "a" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        "old_version",
        "old_plan",
        "wrong_task",
        "child_average",
        "nonfinite",
        "missing_child",
    ],
)
def test_invalid_raw_artifacts_never_become_complete(synthetic_raw_result, mutation):
    parent, result, path = synthetic_raw_result
    if mutation == "old_version":
        result["local_hpo"]["ctboost_version"] = "0.1.58"
    elif mutation == "old_plan":
        result["local_hpo"]["plan_sha256"] = "b" * 64
    elif mutation == "wrong_task":
        parent["task_id"] = 123
    elif mutation == "child_average":
        result["simulation_artifacts"]["bag_info"]["pred_proba_test_per_child"][0][
            0
        ] = 0.8
    elif mutation == "nonfinite":
        result["simulation_artifacts"]["bag_info"]["pred_proba_test_per_child"][0][
            0
        ] = np.nan
    else:
        result["local_hpo"]["children"].pop("S1F8")
    _save_raw(result, path)
    with pytest.raises(ValueError):
        worker.validate_result(path, parent, "a" * 64)


@pytest.fixture
def synthetic_parent_worker(synthetic_gate_fixture, synthetic_raw_result, monkeypatch):
    """Exercise the worker lifecycle with a fake executor, never a benchmark fit."""
    fixture = synthetic_gate_fixture
    monkeypatch.setattr(
        worker, "runtime_provenance", lambda *_: {"synthetic_fixture_only": True}
    )
    monkeypatch.setattr(worker, "build_experiments", _fixture_experiments)
    monkeypatch.setattr(worker, "bootstrap_imports", lambda: (None, None))
    monkeypatch.setattr(worker, "_configure_process", lambda *_: [0, 1])
    monkeypatch.setattr(worker, "_audited_runner", lambda *_: None)
    output = fixture.root / "fake_worker_output"
    common = dict(
        decision_path=fixture.decision_path,
        protocol_path=fixture.protocol_path,
        output=output,
        num_cpus=2,
        memory_limit_gb=8,
    )
    plan = worker.build_plan(**common)
    parent = plan["parents"][51]
    raw = (
        output
        / "data"
        / parent["config_name"]
        / str(parent["task_id"])
        / "0_0"
        / "results.pkl"
    )
    manifest = (
        output
        / "artifacts"
        / parent["config_name"]
        / str(parent["task_id"])
        / "0_0"
        / "manifest.json"
    )
    result = copy.deepcopy(synthetic_raw_result[1])
    result["task_metadata"].update(name=parent["dataset"], tid=parent["task_id"])
    result["local_hpo"]["plan_sha256"] = worker.file_hash(output / "plan.json")
    calls = []

    def fake_execute(self, **kwargs):
        calls.append(kwargs)
        raw.parent.mkdir(parents=True, exist_ok=True)
        _save_raw(result, raw)

    monkeypatch.setattr(FixtureExperiment, "run", fake_execute, raising=False)
    spec = SimpleNamespace(
        cache_key=parent["task_id"], resolve_task_name=lambda _: parent["dataset"]
    )
    monkeypatch.setattr(
        worker, "_load_task", lambda *_: (spec, SimpleNamespace(eval_metric="roc_auc"))
    )
    cache_module = ModuleType("tabarena.utils.cache")
    cache_module.CacheFunctionPickle = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "tabarena.utils.cache", cache_module)

    @contextmanager
    def fake_memory_guard(*_):
        yield {"peak_rss_bytes": 1024}

    monkeypatch.setattr(worker, "_memory_guard", fake_memory_guard)
    return SimpleNamespace(
        common=common,
        parent=parent,
        raw=raw,
        manifest=manifest,
        output=output,
        calls=calls,
    )


def _run_synthetic_parent(fixture):
    return worker.run_parent(
        **fixture.common, dataset=fixture.parent["dataset"], config_index=1
    )


def test_worker_completes_once_and_only_revalidates_its_matching_cache(
    synthetic_parent_worker,
):
    fixture = synthetic_parent_worker
    result = _run_synthetic_parent(fixture)
    assert result["status"] == "complete"
    assert len(fixture.calls) == 1
    assert fixture.calls[0]["eval_metric_name"] == "roc_auc"
    assert fixture.calls[0]["debug_mode"] is False
    assert _run_synthetic_parent(fixture)["status"] == "complete"
    assert len(fixture.calls) == 1
    fixture.raw.write_bytes(b"modified after checkpoint")
    with pytest.raises(ValueError, match="raw result changed"):
        _run_synthetic_parent(fixture)
    assert len(fixture.calls) == 1


def test_worker_refuses_raw_artifacts_without_its_own_journal(synthetic_parent_worker):
    fixture = synthetic_parent_worker
    fixture.raw.parent.mkdir(parents=True)
    fixture.raw.write_bytes(b"foreign old artifact: must not be loaded")
    with pytest.raises(ValueError, match="without matching worker provenance"):
        _run_synthetic_parent(fixture)
    assert not fixture.calls


def test_worker_does_not_automatically_retry_incomplete_training(
    synthetic_parent_worker,
):
    fixture = synthetic_parent_worker
    worker.write_json(
        fixture.manifest,
        {
            "status": "running",
            "parent": fixture.parent,
            "plan_sha256": worker.file_hash(fixture.output / "plan.json"),
        },
    )
    with pytest.raises(ValueError, match="automatic fit retry is disabled"):
        _run_synthetic_parent(fixture)
    assert not fixture.calls


def test_worker_rejects_resource_drift_before_loading_any_task(synthetic_parent_worker):
    fixture = synthetic_parent_worker
    fixture.common["memory_limit_gb"] = 4
    with pytest.raises(ValueError, match="immutable HPO plan"):
        _run_synthetic_parent(fixture)
    assert not fixture.calls


@pytest.mark.parametrize("problem", ["binary", "multiclass", "regression"])
def test_explicit_synthetic_tabarena_parent_and_native_audit(tmp_path, problem):
    """Optional integration preflight: 96 synthetic rows, two rounds, no real gate or datasets."""
    pytest.importorskip(
        "tabarena", reason="Optional pinned TabArena benchmark environment"
    )
    import pandas as pd
    from tabarena.benchmark.task.wrapper import TaskWrapper
    from tabarena.utils.cache import CacheFunctionPickle

    rng = np.random.default_rng(159)
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

    modes = {
        "binary": "backtracking_3",
        "regression": "backtracking_3",
        "multiclass": "full_3_joint",
    }
    experiments, _ = worker.build_experiments(modes, worker.resource_contract(1, 8))
    experiment = experiments[1]
    # This reduction belongs only to the explicit synthetic integration test.
    experiment.method_kwargs["model_hyperparameters"].update(
        iterations=2,
        max_depth=1,
        max_bins=8,
        min_data_in_leaf=2,
    )
    experiment.method_kwargs["init_kwargs"].update(
        path=str(tmp_path / "models"), verbosity=0
    )
    parent = {
        "dataset": "explicit_synthetic_fixture",
        "task_id": 999,
        "problem_type": problem,
        "config_index": 1,
        "config_name": experiment.name,
        "repeat": 0,
        "fold": 0,
    }
    experiment.experiment_cls = worker._audited_runner(parent, "a" * 64)
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
    validation = worker.validate_result(directory / "results.pkl", parent, "a" * 64)
    assert validation["bag_children"] == 8
    assert [
        item["random_seed"] for item in validation["native_children"].values()
    ] == list(range(8, 16))
    experiment.run(
        task=None,
        fold=0,
        task_name=parent["dataset"],
        cache_task_key=999,
        cacher=cacher,
    )
    assert worker.file_hash(directory / "results.pkl") == validation["raw_sha256"]
