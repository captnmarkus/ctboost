import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.split_research import external_panel
from benchmarks.split_research._external_panel_data import (
    atomic_write_json,
    dataset_identity,
    load_json,
    load_openml_payload,
    preserve_openml_frame,
    sha256_file,
    validate_outer_split_indices,
)
from benchmarks.split_research._external_panel_protocol import (
    DATASETS,
    FOLDS,
    HISTOGRAM_THREADS,
    PROFILES,
    RESULT_SCHEMA_VERSION,
    TREATMENTS,
    _job_config,
    array_digest,
    assert_no_absolute_paths,
    identity_digest,
    seal_record,
    sha256_json,
)
from benchmarks.split_research._external_panel_results import (
    exact_control_checks,
    load_ledger,
    store_result,
    summarize_results,
    validate_ledger_entries,
)
from benchmarks.split_research._external_panel_worker import (
    _load_worker_payload,
    execute_worker_job,
    failure_result,
    stable_raw_prediction_digest,
)

EXPECTED_TASK_IDS = [
    15,
    29,
    9952,
    53,
    2074,
    2079,
    361234,
    361236,
    361243,
    361249,
    361258,
    361617,
    7592,
    361255,
]


def _sealed_identity(body, digest_key):
    return seal_record(body, digest_key)


def test_frozen_protocol_has_exact_tasks_profiles_treatments_and_counts():
    manifest = external_panel.protocol_manifest()

    assert [item["task_id"] for item in DATASETS] == EXPECTED_TASK_IDS
    assert [item["name"] for item in PROFILES] == [
        "depthwise-default",
        "depthwise-regularized",
        "leafwise",
    ]
    assert [item["name"] for item in TREATMENTS] == ["control", "candidate"]
    assert TREATMENTS[0]["params"] == {
        "feature_test": "quadratic",
        "feature_test_bins": 8,
        "feature_test_adjustment": "none",
    }
    assert TREATMENTS[1]["params"] == {
        "feature_test": "grouped",
        "feature_test_bins": 8,
        "feature_test_adjustment": "none",
    }
    assert manifest["expected_counts"] == {
        "decision_datasets": 12,
        "stress_datasets": 2,
        "paired_treatment_fits": 252,
        "implicit_control_check_fits": 42,
        "total_subprocess_fits": 294,
    }


def test_treatment_order_alternates_deterministically():
    observed = [external_panel.treatment_order(0, 0, index) for index in range(3)]
    assert observed == [
        ("control", "candidate"),
        ("candidate", "control"),
        ("control", "candidate"),
    ]
    assert external_panel.treatment_order(0, 1, 0) == ("candidate", "control")
    assert external_panel.treatment_order(0, 1, 0) == external_panel.treatment_order(
        0, 1, 0
    )


@pytest.mark.parametrize("problem", ["binary", "regression"])
def test_inner_validation_is_deterministic_and_never_uses_outer_test(problem):
    outer_train = np.arange(80, dtype=np.int64)
    outer_test = np.arange(80, 100, dtype=np.int64)
    target = np.asarray([0, 1] * 50) if problem == "binary" else np.arange(100)

    first = external_panel.make_inner_validation_split(outer_train, target, problem, 91)
    second = external_panel.make_inner_validation_split(
        outer_train, target, problem, 91
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[1]) == 16
    assert not np.intersect1d(first[0], outer_test).size
    assert not np.intersect1d(first[1], outer_test).size
    if problem == "binary":
        assert np.bincount(target[first[1]], minlength=2).tolist() == [8, 8]


def test_metadata_mode_does_not_import_openml_or_ctboost(tmp_path, monkeypatch):
    def fail_import(name):
        raise AssertionError("metadata attempted an import: {}".format(name))

    monkeypatch.setattr(external_panel.importlib, "import_module", fail_import)
    output = tmp_path / "metadata.json"
    assert external_panel.main(["metadata", "--output", str(output)]) == 0
    value = load_json(output)
    assert value["expected_counts"]["total_subprocess_fits"] == 294


def test_preflight_is_metadata_only_with_mocked_probes():
    source = {"source_identity_sha256": "source"}
    native = {"candidate": {"feature_test": "grouped"}}
    report = external_panel.build_preflight_report(
        source_collector=lambda: source,
        feature_probe=lambda: native,
        find_spec=lambda name: object(),
    )

    assert report["ready"]
    assert report["network_accessed"] is False
    assert report["model_fit_started"] is False
    assert report["source_identity"] == source
    assert report["native_feature_test_api"] == native


class _FakeDataset:
    dataset_id = 501
    version = 3
    md5_checksum = "dataset-md5"

    def __init__(self, frame, target):
        self.frame = frame
        self.target = target
        self.calls = []

    def get_data(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame, self.target, [True, False], list(self.frame.columns)


class _FakeTask:
    task_id = 15
    target_name = "label"

    def __init__(self, dataset):
        self.dataset = dataset
        self.split_calls = []

    def get_dataset(self):
        return self.dataset

    def get_train_test_split_indices(self, **kwargs):
        self.split_calls.append(kwargs)
        return np.arange(24), np.arange(24, 30)


def test_openml_loader_uses_task_id_published_splits_and_preserves_types():
    frame = pd.DataFrame(
        {
            "category": ["a", "b", None] * 10,
            "number": [1.0, np.nan, 3.0] * 10,
        }
    )
    target = pd.Series([0, 1] * 15, name="label")
    dataset = _FakeDataset(frame, target)
    task = _FakeTask(dataset)
    calls = []
    openml = SimpleNamespace(
        tasks=SimpleNamespace(
            get_task=lambda task_id, download_data: (
                calls.append((task_id, download_data)) or task
            )
        )
    )

    payload = load_openml_payload(DATASETS[0], openml)

    assert calls == [(15, True)]
    assert dataset.calls == [{"target": "label", "dataset_format": "dataframe"}]
    assert task.split_calls == [
        {"repeat": 0, "fold": fold, "sample": 0} for fold in FOLDS
    ]
    assert isinstance(payload["X"]["category"].dtype, pd.CategoricalDtype)
    assert payload["X"]["category"].isna().sum() == 10
    assert payload["X"]["number"].isna().sum() == 10
    assert payload["data_identity"]["openml"]["task_id"] == 15


def test_openml_staging_rejects_noninteger_published_split_indices():
    frame = pd.DataFrame({"category": ["a", "b"] * 15, "number": range(30)})
    target = pd.Series([0, 1] * 15, name="label")
    dataset = _FakeDataset(frame, target)
    task = _FakeTask(dataset)
    task.get_train_test_split_indices = lambda **kwargs: (
        np.arange(24, dtype=np.float64),
        np.arange(24, 30, dtype=np.float64),
    )
    openml = SimpleNamespace(
        tasks=SimpleNamespace(get_task=lambda task_id, download_data: task)
    )

    with pytest.raises(TypeError, match="integer dtype"):
        load_openml_payload(DATASETS[0], openml)


def test_data_identity_includes_categorical_level_order():
    frame = pd.DataFrame(
        {"category": pd.Series(pd.Categorical(["a", "b"] * 10, categories=["a", "b"]))}
    )
    target = pd.Series([0, 1] * 10)
    outer = {"train": np.arange(16), "test": np.arange(16, 20)}
    payload = {
        "metadata": {"task_id": 15, "target_name": "label"},
        "categorical_indicator": [True],
        "X": frame,
        "y": target,
        "outer_splits": {fold: outer for fold in FOLDS},
    }
    reordered = dict(payload)
    reordered_frame = frame.copy()
    reordered_frame["category"] = reordered_frame["category"].cat.reorder_categories(
        ["b", "a"]
    )
    reordered["X"] = reordered_frame

    assert (
        dataset_identity(payload)["data_identity_sha256"]
        != dataset_identity(reordered)["data_identity_sha256"]
    )


def test_data_identity_includes_ordered_categorical_indicator():
    frame = pd.DataFrame(
        {
            "left": pd.Series(pd.Categorical(["a", "b"] * 10)),
            "right": pd.Series(pd.Categorical(["x", "y"] * 10)),
        }
    )
    target = pd.Series([0, 1] * 10)
    outer = {"train": np.arange(16), "test": np.arange(16, 20)}
    base = {
        "metadata": {"task_id": 15, "target_name": "label"},
        "categorical_indicator": [True, False],
        "X": frame,
        "y": target,
        "outer_splits": {fold: outer for fold in FOLDS},
    }
    changed = {**base, "categorical_indicator": [False, True]}

    assert (
        dataset_identity(base)["data_identity_sha256"]
        != dataset_identity(changed)["data_identity_sha256"]
    )


@pytest.mark.parametrize(
    ("train", "test", "error"),
    [
        (np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0]), "integer dtype"),
        (np.asarray([0, 0, 1]), np.asarray([2, 3]), "unique"),
        (np.asarray([-1, 0]), np.asarray([1, 2, 3]), "row range"),
        (np.asarray([0, 1]), np.asarray([2, 4]), "row range"),
        (np.asarray([0, 1, 2]), np.asarray([2, 3]), "overlap"),
        (np.asarray([0, 1]), np.asarray([3]), "partition every row"),
    ],
)
def test_outer_split_validation_rejects_invalid_published_splits(train, test, error):
    with pytest.raises((TypeError, ValueError), match=error):
        validate_outer_split_indices(
            train,
            test,
            row_count=4,
            context="test split",
        )


def test_raw_prediction_digest_preserves_width_endian_and_shape():
    base64 = np.asarray([1.0], dtype="<f8")
    close64 = np.asarray([1.0 + 2**-30], dtype="<f8")
    base32 = np.asarray([1.0], dtype="<f4")
    big64 = np.asarray([1.0], dtype=">f8")

    digest64, shape64, dtype64 = stable_raw_prediction_digest(base64)
    assert digest64 != stable_raw_prediction_digest(close64)[0]
    assert digest64 != stable_raw_prediction_digest(base32)[0]
    assert digest64 == stable_raw_prediction_digest(big64)[0]
    assert digest64 != stable_raw_prediction_digest(base64.reshape(1, 1))[0]
    assert shape64 == [1]
    assert dtype64 == "<f8"


def _fake_staged_payloads(tmp_path):
    payloads = []
    for dataset in DATASETS:
        rows = 60
        if dataset["problem"] == "binary":
            target = pd.Series([0, 1] * 30)
        elif dataset["problem"] == "multiclass":
            target = pd.Series([0, 1, 2] * 20)
        else:
            target = pd.Series(np.linspace(0.0, 1.0, rows))
        payloads.append(
            {
                "metadata": {"task_id": dataset["task_id"]},
                "y": target,
                "outer_splits": {
                    fold: {"train": np.arange(48), "test": np.arange(48, 60)}
                    for fold in FOLDS
                },
                "data_identity": {
                    "data_identity_sha256": "data-{}".format(dataset["task_id"])
                },
                "_stage_path": tmp_path / "task-{}.pickle".format(dataset["task_id"]),
                "_stage_file_sha256": "stage-{}".format(dataset["task_id"]),
            }
        )
    return payloads


def test_job_keys_cover_full_schedule_and_change_with_source_identity(tmp_path):
    payloads = _fake_staged_payloads(tmp_path)
    first = external_panel.build_jobs(payloads, {"source_identity_sha256": "source-a"})
    second = external_panel.build_jobs(payloads, {"source_identity_sha256": "source-b"})

    assert len(first) == 294
    assert len({job["job_key"] for job in first}) == 294
    assert {job["job_key"] for job in first}.isdisjoint(
        {job["job_key"] for job in second}
    )
    implicit = next(
        job for job in first if job["config"]["role"] == "implicit_control_check"
    )
    assert "feature_test" not in implicit["config"]["params"]
    assert implicit["config"]["outer_test_used_for_early_stopping"] is False
    assert implicit["config"]["histogram_threads"] == 8


def _valid_public_manifest():
    source = _sealed_identity(
        {"kind": "manifest-test-source"}, "source_identity_sha256"
    )
    fold_indices = {}
    all_indices = np.arange(30, dtype=np.int64)
    for fold in FOLDS:
        outer_test = np.arange(fold * 10, (fold + 1) * 10, dtype=np.int64)
        outer_train = np.setdiff1d(all_indices, outer_test)
        fold_indices[fold] = {
            "outer": {"train": outer_train, "test": outer_test},
            "inner_train": outer_train[:-4],
            "inner_valid": outer_train[-4:],
        }
    staged_payloads = [
        {
            "data_identity": _sealed_identity(
                {
                    "openml": {"task_id": int(dataset["task_id"])},
                    "published_splits": [
                        {
                            "repeat": 0,
                            "fold": fold,
                            "sample": 0,
                            "train_count": len(fold_indices[fold]["outer"]["train"]),
                            "test_count": len(fold_indices[fold]["outer"]["test"]),
                            "train_indices_sha256": array_digest(
                                fold_indices[fold]["outer"]["train"].astype(
                                    "<i8", copy=False
                                )
                            ),
                            "test_indices_sha256": array_digest(
                                fold_indices[fold]["outer"]["test"].astype(
                                    "<i8", copy=False
                                )
                            ),
                        }
                        for fold in FOLDS
                    ],
                },
                "data_identity_sha256",
            )
        }
        for dataset in DATASETS
    ]
    data_by_task = {
        int(value["data_identity"]["openml"]["task_id"]): value["data_identity"]
        for value in staged_payloads
    }
    dataset_by_task = {int(item["task_id"]): item for item in DATASETS}
    profile_by_name = {str(item["name"]): item for item in PROFILES}
    jobs = []
    for descriptor in external_panel.planned_fit_descriptors():
        dataset = dict(dataset_by_task[int(descriptor["task_id"])])
        dataset["class_count"] = (
            None
            if dataset["problem"] == "regression"
            else 2
            if dataset["problem"] == "binary"
            else 3
        )
        fold = int(descriptor["fold"])
        config = _job_config(
            dataset,
            fold,
            profile_by_name[str(descriptor["profile"])],
            str(descriptor["treatment"]),
            str(descriptor["role"]),
            int(descriptor["order_in_pair"]),
            fold_indices[fold]["outer"],
            fold_indices[fold]["inner_train"],
            fold_indices[fold]["inner_valid"],
        )
        data_identity = data_by_task[int(descriptor["task_id"])]
        jobs.append(
            {
                "job_key": identity_digest(source, data_identity, config),
                "source_identity": source,
                "data_identity": data_identity,
                "config": config,
            }
        )
    return external_panel._public_run_manifest(jobs, source, staged_payloads)


def test_public_manifest_recomputes_config_data_source_and_full_job_hashes():
    manifest = _valid_public_manifest()
    external_panel._validate_public_run_manifest(manifest)

    config_tamper = pickle.loads(pickle.dumps(manifest))
    config_tamper["jobs"][0]["config"]["unexpected"] = True
    config_tamper = seal_record(config_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="config identity mismatch"):
        external_panel._validate_public_run_manifest(config_tamper)

    job_tamper = pickle.loads(pickle.dumps(manifest))
    job_tamper["jobs"][0]["job_key"] = "0" * 64
    job_tamper = seal_record(job_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="full job identity mismatch"):
        external_panel._validate_public_run_manifest(job_tamper)

    data_tamper = pickle.loads(pickle.dumps(manifest))
    data_tamper["data_identities"][0]["categorical_indicator"] = [True]
    data_tamper = seal_record(data_tamper, "manifest_sha256")
    with pytest.raises(ValueError, match="data_identity_sha256 mismatch"):
        external_panel._validate_public_run_manifest(data_tamper)


def test_public_manifest_rejects_fully_resealed_candidate_param_tamper():
    manifest = _valid_public_manifest()
    tampered = pickle.loads(pickle.dumps(manifest))
    entry = next(
        value
        for value in tampered["jobs"]
        if value["config"]["treatment"] == "candidate"
    )
    entry["config"]["params"]["feature_test"] = "quadratic"
    entry["config_identity_sha256"] = sha256_json(entry["config"])
    data_identity = next(
        value
        for value in tampered["data_identities"]
        if value["openml"]["task_id"] == entry["config"]["task_id"]
    )
    entry["job_key"] = identity_digest(
        tampered["source_identity"], data_identity, entry["config"]
    )
    tampered = seal_record(tampered, "manifest_sha256")

    with pytest.raises(ValueError, match="exact frozen semantics"):
        external_panel._validate_public_run_manifest(tampered)


def _valid_success_result(job):
    return seal_record(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "job_key": job["job_key"],
            "status": "success",
            "config": job["config"],
            "source_identity_sha256": job["source_identity"]["source_identity_sha256"],
            "data_identity_sha256": job["data_identity"]["data_identity_sha256"],
            "config_identity_sha256": sha256_json(job["config"]),
            "metrics": {"primary_loss": 1.0},
            "fit_seconds": 1.0,
            "peak_process_rss_bytes": 100,
            "peak_process_rss_source": "worker_process",
            "serialized_model_bytes": 10,
            "raw_prediction_sha256": "prediction",
            "canonical_tree_sha256": "tree",
            "best_iteration": 1,
        },
        "result_sha256",
    )


def test_ledger_rejects_tampered_success_and_resealed_wrong_resume_identity(tmp_path):
    manifest = _valid_public_manifest()
    job = external_panel._jobs_from_public_manifest(manifest)[0]
    path = tmp_path / "results.json"
    ledger = load_ledger(path, manifest["protocol"]["protocol_sha256"])
    store_result(path, ledger, _valid_success_result(job), expected_job=job)

    tampered = load_json(path)
    tampered["jobs"][job["job_key"]]["metrics"]["primary_loss"] = 2.0
    atomic_write_json(path, seal_record(tampered, "ledger_sha256"))
    with pytest.raises(ValueError, match="result_sha256 mismatch"):
        load_ledger(path, manifest["protocol"]["protocol_sha256"])

    wrong_config = {**job["config"], "unexpected": True}
    wrong_result = _valid_success_result({**job, "config": wrong_config})
    wrong_ledger = seal_record(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol_sha256": manifest["protocol"]["protocol_sha256"],
            "jobs": {job["job_key"]: {**wrong_result, "job_key": job["job_key"]}},
        },
        "ledger_sha256",
    )
    wrong_ledger["jobs"][job["job_key"]] = seal_record(
        wrong_ledger["jobs"][job["job_key"]], "result_sha256"
    )
    wrong_ledger = seal_record(wrong_ledger, "ledger_sha256")
    atomic_write_json(path, wrong_ledger)
    loaded = load_ledger(path, manifest["protocol"]["protocol_sha256"])
    with pytest.raises(ValueError, match="result config differs"):
        validate_ledger_entries(loaded, [job])


def test_atomic_ledger_resumes_by_exact_job_key(tmp_path):
    path = tmp_path / "results.json"
    ledger = load_ledger(path, "protocol")
    result = seal_record(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "job_key": "identity-a",
            "status": "success",
            "config": {},
            "config_identity_sha256": sha256_json({}),
        },
        "result_sha256",
    )
    store_result(path, ledger, result)

    reloaded = load_ledger(path, "protocol")
    assert reloaded["jobs"] == {"identity-a": result}
    with pytest.raises(ValueError, match="different frozen protocol"):
        load_ledger(path, "other")


class _FakeHandle:
    def export_state(self):
        return {"trees": [{"feature": 0, "value": 0.0}]}


class _FakeBooster:
    best_iteration = 2
    num_iterations_trained = 3
    _handle = _FakeHandle()

    def predict(self, pool):
        return np.zeros(len(pool.frame), dtype=np.float32)

    def save_model(self, path):
        Path(path).write_bytes(b"mock-model")


class _FakePool:
    def __init__(self, frame, label, cat_features):
        self.frame = frame
        self.label = np.asarray(label)
        self.cat_features = list(cat_features)


def test_worker_uses_only_inner_validation_and_preserves_dataframe(
    tmp_path, monkeypatch
):
    frame = preserve_openml_frame(
        pd.DataFrame(
            {
                "category": ["a", "b", None, "a"] * 5,
                "number": [1.0, np.nan, 2.0, 3.0] * 5,
            }
        ),
        [True, False],
    )
    target = pd.Series([0, 1] * 10, name="label")
    outer = {"train": np.arange(16), "test": np.arange(16, 20)}
    payload = {
        "metadata": {
            "task_id": 15,
            "dataset_id": 1,
            "dataset_version": 1,
            "dataset_md5_checksum": "md5",
            "target_name": "label",
            "problem": "binary",
            "stress_only": False,
            "frozen_display_name": "breast-w",
            "identity_basis": "OpenML task ID and published repeat/fold/sample split",
            "repeat": 0,
            "sample": 0,
        },
        "categorical_indicator": [True, False],
        "X": frame,
        "y": target,
        "outer_splits": {fold: outer for fold in FOLDS},
    }
    payload["data_identity"] = dataset_identity(payload)
    stage = tmp_path / "staged.pickle"
    stage.write_bytes(pickle.dumps(payload, protocol=5))
    inner_train, inner_valid = external_panel.make_inner_validation_split(
        outer["train"], target, "binary", external_panel.fit_seed(15, 0)
    )
    dataset = {**DATASETS[0], "class_count": 2}
    config = _job_config(
        dataset,
        0,
        {**PROFILES[0], "iterations": 3},
        "control",
        "treatment",
        0,
        outer,
        inner_train,
        inner_valid,
    )
    source = _sealed_identity({"kind": "mock-source"}, "source_identity_sha256")
    job = {
        "source_identity": source,
        "data_identity": payload["data_identity"],
        "config": config,
        "stage_file": str(stage),
        "stage_file_sha256": sha256_file(stage),
        "inner_train_indices": inner_train.tolist(),
        "inner_validation_indices": inner_valid.tolist(),
        "class_labels": ["0", "1"],
    }
    job["job_key"] = identity_digest(source, job["data_identity"], config)
    observed = {}

    def fake_train(train_pool, params, **kwargs):
        observed["train_index"] = train_pool.frame.index.to_numpy()
        observed["valid_index"] = kwargs["eval_set"].frame.index.to_numpy()
        observed["params"] = params
        observed["early_stopping_rounds"] = kwargs["early_stopping_rounds"]
        return _FakeBooster()

    fake_ctboost = SimpleNamespace(Pool=_FakePool, train=fake_train)
    monkeypatch.setenv("CTBOOST_HIST_THREADS", str(HISTOGRAM_THREADS))
    result = execute_worker_job(
        job,
        tmp_path,
        source_collector=lambda: source,
        ctboost_module=fake_ctboost,
    )

    np.testing.assert_array_equal(
        np.sort(observed["train_index"]), np.sort(inner_train)
    )
    np.testing.assert_array_equal(
        np.sort(observed["valid_index"]), np.sort(inner_valid)
    )
    assert not np.intersect1d(observed["valid_index"], outer["test"]).size
    assert observed["early_stopping_rounds"] == 50
    assert observed["params"]["feature_test"] == "quadratic"
    assert result["fit_timer_excludes_openml_cache_and_pool_construction"]
    assert result["categorical_feature_count"] == 1
    assert result["finite_predictions"]


def test_worker_payload_load_defensively_revalidates_outer_partition(tmp_path):
    frame = pd.DataFrame({"value": np.arange(20, dtype=np.float64)})
    target = pd.Series([0, 1] * 10)
    valid_outer = {"train": np.arange(16), "test": np.arange(16, 20)}
    payload = {
        "metadata": {"task_id": 15, "target_name": "label"},
        "categorical_indicator": [False],
        "X": frame,
        "y": target,
        "outer_splits": {fold: valid_outer for fold in FOLDS},
    }
    identity = dataset_identity(payload)
    payload["outer_splits"] = {
        **payload["outer_splits"],
        0: {"train": np.asarray([0, 0, *range(1, 16)]), "test": np.arange(16, 20)},
    }
    stage = tmp_path / "tampered-stage.pickle"
    stage.write_bytes(pickle.dumps(payload, protocol=5))
    job = {
        "stage_file": str(stage),
        "stage_file_sha256": sha256_file(stage),
        "data_identity": identity,
    }

    with pytest.raises(ValueError, match="unique"):
        _load_worker_payload(job)


def _synthetic_complete_ledger():
    jobs = []
    results = {}
    source = _sealed_identity({"kind": "synthetic-source"}, "source_identity_sha256")
    dataset_by_id = {item["task_id"]: item for item in DATASETS}
    data_by_id = {
        int(item["task_id"]): _sealed_identity(
            {"openml": {"task_id": int(item["task_id"])}},
            "data_identity_sha256",
        )
        for item in DATASETS
    }
    for descriptor in external_panel.planned_fit_descriptors():
        dataset = dataset_by_id[descriptor["task_id"]]
        data_identity = data_by_id[int(descriptor["task_id"])]
        config = {
            **descriptor,
            "stress_only": dataset["stress_only"],
        }
        key = identity_digest(source, data_identity, config)
        jobs.append(
            {
                "job_key": key,
                "source_identity": source,
                "data_identity": data_identity,
                "config": config,
            }
        )
        treatment = descriptor["treatment"]
        check_hash = "{}-{}".format(descriptor["task_id"], descriptor["profile"])
        is_candidate = treatment == "candidate"
        results[key] = seal_record(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "job_key": key,
                "status": "success",
                "config": config,
                "source_identity_sha256": source["source_identity_sha256"],
                "data_identity_sha256": data_identity["data_identity_sha256"],
                "config_identity_sha256": sha256_json(config),
                "metrics": {"primary_loss": 0.99 if is_candidate else 1.0},
                "fit_seconds": 1.1 if is_candidate else 1.0,
                "peak_process_rss_bytes": 100,
                "peak_process_rss_source": "worker_process",
                "serialized_model_bytes": 10,
                "finite_predictions": True,
                "correct_prediction_shape": True,
                "raw_prediction_sha256": check_hash,
                "canonical_tree_sha256": check_hash,
                "best_iteration": 7,
            },
            "result_sha256",
        )
    stress_candidate = next(
        value
        for value in results.values()
        if value["config"]["task_id"] == 7592
        and value["config"]["treatment"] == "candidate"
        and value["config"]["fold"] == 2
    )
    failed_stress = seal_record(
        {
            **stress_candidate,
            "status": "failure",
            "error_type": "MockFailure",
            "error_message": "stress-only",
            "peak_process_rss_bytes": None,
            "peak_process_rss_source": "unavailable",
        },
        "result_sha256",
    )
    results[stress_candidate["job_key"]] = failed_stress
    return jobs, {"jobs": results}, source, list(data_by_id.values())


def test_aggregation_applies_frozen_gates_and_excludes_stress_failures():
    jobs, ledger, source, data_identities = _synthetic_complete_ledger()
    summary = summarize_results(
        jobs,
        ledger,
        source,
        data_identities,
    )

    assert summary["coverage"]["failed_jobs"] == 1
    assert summary["decision_aggregation"]["win_tie_loss"] == {
        "win": 12,
        "tie": 0,
        "loss": 0,
    }
    assert summary["decision_aggregation"][
        "median_relative_primary_loss_improvement"
    ] == pytest.approx(0.01)
    assert summary["decision_aggregation"][
        "median_paired_fit_time_ratio"
    ] == pytest.approx(1.1)
    assert summary["protocol_valid"]
    assert summary["grouped_8_advances"]
    assert summary["resources"]["peak_process_rss_bytes_max"] == 100
    assert all(summary["frozen_promotion_gates"].values())


def test_control_check_detects_any_nonidentical_default_path():
    jobs, ledger, _, _ = _synthetic_complete_ledger()
    implicit = next(
        value
        for value in ledger["jobs"].values()
        if value["config"]["task_id"] == 15
        and value["config"]["role"] == "implicit_control_check"
    )
    implicit["canonical_tree_sha256"] = "different"
    checks = exact_control_checks(jobs, ledger)
    affected = [item for item in checks if item["task_id"] == 15]
    assert any(not item["exact"] for item in affected)


@pytest.mark.parametrize("writes_worker_output", [True, False])
def test_run_orchestrator_sets_eight_threads_and_uses_worker_subprocess(
    tmp_path, monkeypatch, writes_worker_output
):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    source = _sealed_identity({"kind": "mock-source"}, "source_identity_sha256")
    data_identity = _sealed_identity(
        {"openml": {"task_id": 15}}, "data_identity_sha256"
    )
    fake_payload = {
        "data_identity": data_identity,
        "_stage_path": caller / "relative-cache" / "stage.pickle",
        "_stage_file_sha256": "stage",
    }
    config = {
        "task_id": 15,
        "fold": 0,
        "profile": "depthwise-default",
        "treatment": "control",
        "role": "treatment",
        "stress_only": False,
    }
    job = {
        "source_identity": source,
        "data_identity": data_identity,
        "config": config,
        "stage_file": str(fake_payload["_stage_path"]),
        "stage_file_sha256": "stage",
        "inner_train_indices": [],
        "inner_validation_indices": [],
        "class_labels": ["0", "1"],
    }
    job["job_key"] = identity_digest(source, data_identity, config)
    staged_cache_dirs = []
    monkeypatch.setattr(
        external_panel,
        "stage_openml_payload",
        lambda cache_dir, *args: staged_cache_dirs.append(cache_dir) or fake_payload,
    )
    monkeypatch.setattr(external_panel, "build_jobs", lambda *args: [job])
    monkeypatch.setattr(
        external_panel, "_validate_public_run_manifest", lambda *args: None
    )
    monkeypatch.setattr(
        external_panel,
        "summarize_results",
        lambda *args: {"coverage": {"failed_jobs": 0}},
    )
    observed = {}

    def fake_subprocess(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        output = Path(command[command.index("--output") + 1])
        if writes_worker_output:
            atomic_write_json(
                output,
                seal_record(
                    {
                        "schema_version": RESULT_SCHEMA_VERSION,
                        "job_key": job["job_key"],
                        "status": "success",
                        "config": config,
                        "source_identity_sha256": source["source_identity_sha256"],
                        "data_identity_sha256": data_identity["data_identity_sha256"],
                        "config_identity_sha256": sha256_json(config),
                        "metrics": {"primary_loss": 1.0},
                        "fit_seconds": 0.1,
                        "peak_process_rss_bytes": 100,
                        "peak_process_rss_source": "worker_process",
                        "serialized_model_bytes": 10,
                        "raw_prediction_sha256": "predictions",
                        "canonical_tree_sha256": "trees",
                        "best_iteration": 1,
                    },
                    "result_sha256",
                ),
            )
        return SimpleNamespace(returncode=0)

    summary = external_panel.run_panel(
        Path("relative-results"),
        Path("relative-cache"),
        openml_module=SimpleNamespace(config=SimpleNamespace()),
        subprocess_runner=fake_subprocess,
        source_collector=lambda: source,
        feature_probe=lambda: {},
    )

    assert summary["coverage"]["failed_jobs"] == 0
    assert observed["env"]["CTBOOST_HIST_THREADS"] == "8"
    assert "_worker" in observed["command"]
    assert observed["command"][:3] == [
        external_panel.sys.executable,
        "-m",
        "benchmarks.split_research.external_panel",
    ]
    assert len(staged_cache_dirs) == len(DATASETS)
    assert set(staged_cache_dirs) == {(caller / "relative-cache").resolve()}
    assert Path(
        observed["command"][observed["command"].index("--job") + 1]
    ).is_absolute()
    assert Path(
        observed["command"][observed["command"].index("--output") + 1]
    ).is_absolute()
    assert (caller / "relative-results" / "results.json").is_file()
    persisted = load_json(caller / "relative-results" / "results.json")["jobs"][
        job["job_key"]
    ]
    if writes_worker_output:
        assert persisted["peak_process_rss_bytes"] == 100
        assert persisted["peak_process_rss_source"] == "worker_process"
    else:
        assert persisted["peak_process_rss_bytes"] is None
        assert persisted["peak_process_rss_source"] == "unavailable"


@pytest.mark.parametrize(
    "message",
    [
        r"failure while opening C:\private\result.json at worker",
        "failure while opening /private/result.json at worker",
        "failure while opening file:///private/result.json at worker",
    ],
)
def test_public_output_rejects_embedded_absolute_paths(message):
    with pytest.raises(ValueError, match="absolute path leaked"):
        assert_no_absolute_paths({"error": message})


def test_failure_result_redacts_embedded_paths_and_can_mark_rss_unavailable():
    job = {
        "job_key": "mock-job",
        "config": {},
        "source_identity": {"source_identity_sha256": "source"},
        "data_identity": {"data_identity_sha256": "data"},
    }
    result = failure_result(
        job,
        RuntimeError(r"failed at C:\private\input.bin and /private/output.bin"),
        record_process_rss=False,
    )

    assert "<path>" in result["error_message"]
    assert result["peak_process_rss_bytes"] is None
    assert result["peak_process_rss_source"] == "unavailable"
    assert_no_absolute_paths(result)
