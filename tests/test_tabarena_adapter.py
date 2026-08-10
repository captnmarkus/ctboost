import io
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import benchmarks.tabarena.ctboost_model as ctboost_adapter
import benchmarks.tabarena.run as tabarena_run
from benchmarks.tabarena.ctboost_model import (
    CTBoostTabArenaGPUModel,
    CTBoostTabArenaModel,
    _adaptive_training_budget,
    _bounded_categorical_pairs,
    _ctboost_eval_metric,
    _ctboost_histogram_threads,
    _finalize_search_config,
    _resolve_categorical_pair_budget,
    _resolve_time_limit,
    generate_configs_ctboost,
    normalize_tabarena_frame,
)
from benchmarks.tabarena.run import (
    _build_experiments,
    _expected_raw_keys,
    _format_table,
    _git_source_identity,
    _local_distribution_checkout,
    _resolve_effective_time_limit,
    _run_job_shard,
    _select_expected_raw_paths,
    _shard_manifest_path,
    _validate_args,
    _write_console_table,
    _write_manifest,
    _write_resource_summary,
)


def test_tabarena_frame_preserves_categorical_values_and_numeric_missing_values():
    frame = pd.DataFrame(
        {
            "city": pd.Series(["Berlin", None, "Paris"], dtype="category"),
            "flag": pd.Series([True, False, True], dtype="bool"),
            "value": np.array([1.0, np.nan, 3.0], dtype=np.float32),
        }
    )

    normalized, categorical = normalize_tabarena_frame(frame)

    assert categorical == ["city"]
    assert normalized["city"].dtype == frame["city"].dtype
    assert normalized.loc[0, "city"] == "Berlin"
    assert pd.isna(normalized.loc[1, "city"])
    assert normalized.loc[2, "city"] == "Paris"
    assert normalized["flag"].dtype == frame["flag"].dtype
    assert np.isnan(normalized.loc[1, "value"])


def test_tabarena_frame_does_not_alias_literal_category_and_missing_value():
    frame = pd.DataFrame(
        {
            "category": pd.Series(
                ["__CTBOOST_MISSING__", None, "ordinary"], dtype="category"
            )
        }
    )

    normalized, categorical = normalize_tabarena_frame(frame)

    assert categorical == ["category"]
    assert normalized.loc[0, "category"] == "__CTBOOST_MISSING__"
    assert pd.isna(normalized.loc[1, "category"])
    assert normalized.loc[2, "category"] == "ordinary"


def test_tabarena_native_categorical_missing_values_remain_separable_after_pickle():
    from ctboost import CTBoostClassifier

    literal = "__CTBOOST_MISSING__"
    frame = pd.DataFrame(
        {"category": pd.Series([literal] * 30 + [None] * 30, dtype="category")}
    )
    target = np.array([0] * 30 + [1] * 30)
    normalized, categorical = normalize_tabarena_frame(frame)
    model = CTBoostClassifier(
        iterations=40,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        random_seed=7,
        cat_features=categorical,
        verbose=False,
    )
    model.fit(normalized, target)
    probe, _ = normalize_tabarena_frame(
        pd.DataFrame({"category": pd.Series([literal, None], dtype="category")}),
        categorical_columns=categorical,
    )

    probabilities = model.predict_proba(probe)[:, 1]
    restored_probabilities = pickle.loads(pickle.dumps(model)).predict_proba(probe)[
        :, 1
    ]

    assert abs(float(probabilities[0] - probabilities[1])) > 0.2
    np.testing.assert_array_equal(restored_probabilities, probabilities)


def test_tabarena_native_mixed_frame_predicts_unseen_values_and_missing_data():
    from ctboost import CTBoostClassifier

    frame = pd.DataFrame(
        {
            "category": pd.Series(["red", "blue", None] * 20, dtype="category"),
            "object": pd.Series(["left", "right", None] * 20, dtype=object),
            "string": pd.Series(["low", "high", pd.NA] * 20, dtype="string"),
            "boolean": pd.Series([True, False, True] * 20, dtype=bool),
            "numeric": pd.Series([1.0, np.nan, 3.0] * 20, dtype=np.float64),
        }
    )
    target = np.tile([0, 1, 0], 20)
    normalized, categorical = normalize_tabarena_frame(frame)
    model = CTBoostClassifier(
        iterations=8,
        max_depth=2,
        alpha=1.0,
        random_seed=11,
        cat_features=categorical,
        verbose=False,
    )
    model.fit(normalized, target)
    probe = pd.DataFrame(
        {
            "category": pd.Series(["green", None], dtype="category"),
            "object": pd.Series(["centre", None], dtype=object),
            "string": pd.Series(["medium", pd.NA], dtype="string"),
            "boolean": pd.Series([False, True], dtype=bool),
            "numeric": pd.Series([2.0, np.nan], dtype=np.float64),
        }
    )
    probe, _ = normalize_tabarena_frame(
        probe,
        categorical_columns=categorical,
    )

    probabilities = model.predict_proba(probe)

    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)


def test_tabarena_frame_reuses_training_schema_and_rejects_missing_columns():
    frame = pd.DataFrame({"city": ["Berlin"], "value": [1.0]})
    normalized, categorical = normalize_tabarena_frame(
        frame,
        categorical_columns=["city"],
    )

    assert categorical == ["city"]
    assert normalized["city"].tolist() == ["Berlin"]

    with pytest.raises(ValueError, match="missing categorical columns"):
        normalize_tabarena_frame(
            pd.DataFrame({"value": [1.0]}), categorical_columns=["city"]
        )


def test_tabarena_output_falls_back_when_markdown_extra_is_missing():
    class FrameWithoutTabulate:
        def to_markdown(self, *, index):
            assert index is False
            raise ImportError("tabulate is not installed")

        def to_string(self, *, index):
            assert index is False
            return "plain leaderboard"

    assert _format_table(FrameWithoutTabulate()) == "plain leaderboard"


def test_tabarena_manifest_records_checkout_identity_when_available():
    identity = _git_source_identity()
    assert identity is not None
    assert len(identity["commit"]) == 40
    assert isinstance(identity["dirty"], bool)
    assert isinstance(identity["status"], list)
    if identity["dirty"]:
        assert len(identity["dirty_fingerprint_sha256"]) == 64


def test_tabarena_cpu_and_gpu_models_have_distinct_resource_contracts():
    assert CTBoostTabArenaModel.ag_name == "CTBoost"
    assert CTBoostTabArenaModel.ag_priority == 65
    assert CTBoostTabArenaModel._supported_problem_types == [
        "binary",
        "multiclass",
        "regression",
    ]
    assert CTBoostTabArenaModel.default_resources_physical_cores_only is True
    assert "supported_problem_types" not in CTBoostTabArenaModel.__dict__
    assert "_get_default_auxiliary_params" not in CTBoostTabArenaModel.__dict__
    assert (
        "object"
        in CTBoostTabArenaModel._default_auxiliary_params_extra["valid_raw_types"]
    )
    assert (
        "datetime_as_object"
        in (
            CTBoostTabArenaModel._default_auxiliary_params_extra[
                "ignored_type_group_special"
            ]
        )
    )
    assert "ignored_type_group_raw" not in (
        CTBoostTabArenaModel._default_auxiliary_params_extra
    )
    assert CTBoostTabArenaModel.default_num_gpus == 0
    assert CTBoostTabArenaModel._ctboost_task_type == "CPU"
    assert CTBoostTabArenaGPUModel.ag_name == "CTBoostGPU"
    assert CTBoostTabArenaGPUModel.default_num_gpus == 1
    assert CTBoostTabArenaGPUModel.minimum_num_gpus == 1
    assert CTBoostTabArenaGPUModel.gpu_required is True
    assert CTBoostTabArenaGPUModel._ctboost_task_type == "GPU"


def test_tabarena_memory_estimate_accounts_for_multiclass_tree_capacity():
    frame = pd.DataFrame(np.zeros((100, 4), dtype=np.float32))

    binary = CTBoostTabArenaModel._estimate_memory_usage_static(
        X=frame,
        num_classes=2,
        hyperparameters={"iterations": 100, "max_depth": 4},
    )
    multiclass = CTBoostTabArenaModel._estimate_memory_usage_static(
        X=frame,
        num_classes=20,
        hyperparameters={"iterations": 100, "max_depth": 4},
    )

    assert binary >= 512 * 1024 * 1024
    assert multiclass > binary


def test_tabarena_search_finalizes_only_meaningful_leaf_caps():
    depthwise = _finalize_search_config(
        {"grow_policy": "DepthWise", "max_depth": np.int64(6)}
    )
    leafwise = _finalize_search_config(
        {
            "grow_policy": "LeafWise",
            "max_depth": np.int64(6),
            "__leaf_fraction": np.float64(0.5),
        }
    )

    assert depthwise["max_leaves"] == 0
    assert leafwise["max_depth"] == 6
    assert leafwise["max_leaves"] == 32
    assert "__leaf_fraction" not in leafwise


def test_tabarena_search_is_deterministic_unique_and_task_safe():
    first = generate_configs_ctboost(200)
    second = generate_configs_ctboost(200)

    assert first == second
    assert len(first) == 200
    assert len({tuple(sorted(config.items())) for config in first}) == 200
    assert all(0.02 <= config["learning_rate"] <= 0.2 for config in first)
    assert all(3 <= config["max_depth"] <= 8 for config in first)
    assert all(
        config["max_leaves"] == 0
        if config["grow_policy"] == "DepthWise"
        else 4 <= config["max_leaves"] < 2 ** config["max_depth"]
        for config in first
    )
    assert all(
        config["ordered_ctr"] or "ctr_prior_strength" not in config for config in first
    )
    assert sum(config["grow_policy"] == "DepthWise" for config in first) == 100
    assert sum(config["ordered_ctr"] for config in first) == 150
    assert any(config.get("tabarena_categorical_pair_budget", 0) for config in first)
    assert all(
        not config.get("tabarena_categorical_pair_budget", 0) or config["ordered_ctr"]
        for config in first
    )
    assert all(
        (config["iterations"], config["early_stopping_rounds"])
        == _adaptive_training_budget(config["learning_rate"])
        for config in first
    )
    assert all("random_seed" not in config for config in first)
    for count in (1, 8, 200):
        selected = generate_configs_ctboost(count)
        assert selected == first[:count]
        assert len(selected) == count
        assert len({tuple(sorted(config.items())) for config in selected}) == count


def test_tabarena_categorical_pair_selection_is_bounded_and_deterministic():
    frame = pd.DataFrame(
        {
            "a": ["x", "x", "y", "y", "x", "y"],
            "b": ["m", "n", "m", "n", "m", "n"],
            "c": ["u", "v", "w", "u", "v", "w"],
            "constant": ["same"] * 6,
        }
    )

    pairs = _bounded_categorical_pairs(
        frame,
        ["a", "b", "c", "constant"],
        max_pairs=2,
        max_joint_cardinality=12,
    )

    assert pairs == [["a", "b"], ["a", "c"]]


def test_tabarena_fit_resolves_pair_budget_without_leaking_wrapper_parameter(
    monkeypatch,
):
    constructed = []

    class FakeClassifier:
        def __init__(self, **params):
            self.params = params
            constructed.append(self)

        def fit(self, _X, _y, **_kwargs):
            pass

    monkeypatch.setattr("ctboost.CTBoostClassifier", FakeClassifier)
    adapter = _model_adapter(
        problem_type="binary",
        stopping_metric=None,
        params={"ordered_ctr": True, "tabarena_categorical_pair_budget": 2},
    )

    def preprocess(frame, *, is_train=False, **_kwargs):
        if is_train:
            adapter._ctboost_categorical_columns = ["a", "b", "c"]
        return frame

    adapter.preprocess = preprocess
    frame = pd.DataFrame(
        {
            "a": ["x", "x", "y", "y"],
            "b": ["m", "n", "m", "n"],
            "c": ["u", "v", "u", "v"],
        }
    )

    adapter._fit(frame, np.array([0, 1, 0, 1]))

    assert "tabarena_categorical_pair_budget" not in constructed[0].params
    assert constructed[0].params["categorical_combinations"] == [
        ["a", "b"],
        ["a", "c"],
    ]


def test_tabarena_fit_rejects_pair_budget_above_supported_limit(monkeypatch):
    class FakeClassifier:
        def __init__(self, **_params):
            pytest.fail("invalid pair budget must fail before model construction")

    monkeypatch.setattr("ctboost.CTBoostClassifier", FakeClassifier)
    adapter = _model_adapter(
        problem_type="binary",
        stopping_metric=None,
        params={"tabarena_categorical_pair_budget": 5},
    )

    with pytest.raises(ValueError, match="between 0 and 4"):
        adapter._fit(
            pd.DataFrame({"value": [0.0, 1.0]}),
            np.array([0, 1]),
        )


@pytest.mark.parametrize("value", [True, False, 4.9, -0.5, "2", "", None])
def test_tabarena_pair_budget_rejects_noninteger_values(value):
    with pytest.raises(TypeError, match="must be an integer"):
        _resolve_categorical_pair_budget(value)


@pytest.mark.parametrize("value", [False, 0.0, "", None])
def test_tabarena_memory_estimate_uses_strict_pair_budget_validation(value):
    with pytest.raises(TypeError, match="must be an integer"):
        CTBoostTabArenaModel._estimate_memory_usage_static(
            X=pd.DataFrame({"value": [0.0, 1.0]}),
            hyperparameters={"tabarena_categorical_pair_budget": value},
        )


def test_tabarena_search_rejects_negative_counts_without_optional_dependencies():
    with pytest.raises(ValueError, match="non-negative"):
        generate_configs_ctboost(-1)
    assert generate_configs_ctboost(0) == []
    with pytest.raises(ValueError, match="cannot exceed"):
        generate_configs_ctboost(201)


def _model_adapter(
    *,
    problem_type,
    stopping_metric,
    params,
    model_cls=CTBoostTabArenaModel,
):
    adapter = object.__new__(model_cls)
    adapter.problem_type = problem_type
    adapter.stopping_metric = stopping_metric
    adapter._ctboost_categorical_columns = []
    adapter.preprocess = lambda frame, **_kwargs: frame
    adapter._get_model_params = lambda: dict(params)
    return adapter


def test_tabarena_uses_autogluon_seed_contract_without_hardcoded_default():
    assert CTBoostTabArenaModel.seed_name == "random_seed"
    defaults = {}
    adapter = object.__new__(CTBoostTabArenaModel)
    adapter._set_default_param_value = defaults.setdefault

    adapter._set_default_params()

    assert "random_seed" not in defaults


@pytest.mark.parametrize(
    ("problem_type", "metric", "expected"),
    [
        ("binary", SimpleNamespace(name="roc_auc"), "AUC"),
        ("binary", "log-loss", "Logloss"),
        ("multiclass", SimpleNamespace(name="log_loss"), "MultiClass"),
        ("regression", SimpleNamespace(name="rmse"), "RMSE"),
        ("regression", SimpleNamespace(name="root mean squared error"), "RMSE"),
        ("regression", SimpleNamespace(name="mae"), None),
    ],
)
def test_tabarena_maps_supported_stopping_metrics(problem_type, metric, expected):
    assert _ctboost_eval_metric(problem_type, metric) == expected


def test_tabarena_fit_maps_metric_but_preserves_explicit_metric_and_seed(monkeypatch):
    constructed = []

    class FakeClassifier:
        def __init__(self, **params):
            self.params = params
            constructed.append(self)

        def fit(self, _X, _y, **kwargs):
            self.fit_kwargs = kwargs

    monkeypatch.setattr("ctboost.CTBoostClassifier", FakeClassifier)
    X = pd.DataFrame({"value": [0.0, 1.0]})
    y = np.array([0, 1])

    mapped = _model_adapter(
        problem_type="binary",
        stopping_metric=SimpleNamespace(name="roc_auc"),
        params={"random_seed": 41},
    )
    mapped._fit(X, y)
    explicit = _model_adapter(
        problem_type="binary",
        stopping_metric=SimpleNamespace(name="roc_auc"),
        params={"eval_metric": "BalancedAccuracy", "random_seed": 73},
    )
    explicit._fit(X, y)

    assert constructed[0].params["eval_metric"] == "AUC"
    assert constructed[0].params["random_seed"] == 41
    assert constructed[1].params["eval_metric"] == "BalancedAccuracy"
    assert constructed[1].params["random_seed"] == 73


@pytest.mark.parametrize(
    ("model_cls", "allocated_gpus", "configured_task_type", "expected_task_type"),
    [
        (CTBoostTabArenaModel, 1, "GPU", "CPU"),
        (CTBoostTabArenaGPUModel, 0, "CPU", "GPU"),
    ],
)
def test_tabarena_adapter_pins_task_type_to_model_identity(
    monkeypatch,
    model_cls,
    allocated_gpus,
    configured_task_type,
    expected_task_type,
):
    class FakeClassifier:
        def __init__(self, **params):
            self.params = params

        def fit(self, _X, _y, **_kwargs):
            pass

    monkeypatch.setattr("ctboost.CTBoostClassifier", FakeClassifier)
    adapter = _model_adapter(
        problem_type="binary",
        stopping_metric=None,
        params={"task_type": configured_task_type},
        model_cls=model_cls,
    )

    adapter._fit(
        pd.DataFrame({"value": [0.0, 1.0]}),
        np.array([0, 1]),
        num_gpus=allocated_gpus,
    )

    assert adapter.model.params["task_type"] == expected_task_type


def test_tabarena_fit_consumes_setup_budget_and_preserves_callbacks(monkeypatch):
    constructed = []
    setup_finished = False

    class FakeRegressor:
        def __init__(self, **params):
            nonlocal setup_finished
            self.params = params
            constructed.append(self)
            setup_finished = True

        def fit(self, _X, _y, **kwargs):
            self.fit_kwargs = kwargs

    clock_values = iter(
        [
            (100.0, False),
            (101.0, True),
            (102.0, True),
            (103.2, True),
        ]
    )

    def monotonic():
        value, expected_setup_state = next(clock_values)
        assert setup_finished is expected_setup_state
        return value

    user_calls = []

    def user_callback(env):
        user_calls.append(env)
        return False

    monkeypatch.setattr("ctboost.CTBoostRegressor", FakeRegressor)
    monkeypatch.setattr(ctboost_adapter.time, "monotonic", monotonic)
    adapter = _model_adapter(
        problem_type="regression",
        stopping_metric=SimpleNamespace(name="rmse"),
        params={"callbacks": [user_callback]},
    )

    adapter._fit(
        pd.DataFrame({"value": [0.0, 1.0]}), np.array([0.0, 1.0]), time_limit=5
    )

    callbacks = constructed[0].fit_kwargs["callbacks"]
    assert callbacks[0] is user_callback
    assert callbacks[1](SimpleNamespace(iteration=0, begin_iteration=0)) is False
    assert callbacks[1](SimpleNamespace(iteration=1, begin_iteration=0)) is True
    assert user_calls == []


def test_tabarena_fit_raises_before_training_when_setup_consumes_budget(monkeypatch):
    fit_called = False

    class FakeRegressor:
        def __init__(self, **_params):
            pass

        def fit(self, _X, _y, **_kwargs):
            nonlocal fit_called
            fit_called = True

    class FakeTimeLimitExceeded(Exception):
        pass

    times = iter([100.0, 103.0])
    monkeypatch.setattr("ctboost.CTBoostRegressor", FakeRegressor)
    monkeypatch.setattr(ctboost_adapter.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        ctboost_adapter,
        "_raise_time_limit_exceeded",
        lambda: (_ for _ in ()).throw(FakeTimeLimitExceeded()),
    )
    adapter = _model_adapter(
        problem_type="regression",
        stopping_metric=SimpleNamespace(name="rmse"),
        params={},
    )

    with pytest.raises(FakeTimeLimitExceeded):
        adapter._fit(
            pd.DataFrame({"value": [0.0, 1.0]}),
            np.array([0.0, 1.0]),
            time_limit=5,
        )

    assert fit_called is False


def test_tabarena_none_time_limit_does_not_install_or_read_deadline(monkeypatch):
    class FakeRegressor:
        def __init__(self, **_params):
            pass

        def fit(self, _X, _y, **kwargs):
            self.fit_kwargs = kwargs

    monkeypatch.setattr("ctboost.CTBoostRegressor", FakeRegressor)
    monkeypatch.setattr(
        ctboost_adapter.time,
        "monotonic",
        lambda: pytest.fail("None time_limit must not read the clock"),
    )
    adapter = _model_adapter(
        problem_type="regression",
        stopping_metric=SimpleNamespace(name="rmse"),
        params={},
    )

    adapter._fit(pd.DataFrame({"value": [0.0, 1.0]}), np.array([0.0, 1.0]))

    assert "callbacks" not in adapter.model.fit_kwargs


@pytest.mark.parametrize("time_limit", [0, -1, float("inf"), float("nan")])
def test_tabarena_rejects_non_positive_or_non_finite_time_limits(time_limit):
    with pytest.raises(ValueError, match="finite and positive"):
        _resolve_time_limit(time_limit)


def test_resource_summary_exports_memory_and_timing(tmp_path):
    class Artifact:
        result = {
            "framework": "CTBoost",
            "problem_type": "regression",
            "metric": "rmse",
            "metric_error": np.float64(0.25),
            "time_train_s": 1.5,
            "time_infer_s": 0.1,
            "task_metadata": {
                "name": "public-data",
                "tid": 7,
                "fold": 1,
                "repeat": 0,
                "sample": 0,
            },
            "memory_usage": {
                "peak_mem_cpu": 150,
                "min_mem_cpu": 100,
                "peak_mem_gpu": 80,
                "min_mem_gpu": 20,
                "gpu_tracking_enabled": True,
            },
            "method_metadata": {"num_cpus": 4, "num_gpus": 1, "disk_usage": 1234},
        }

    artifact_path = tmp_path / "raw" / "data" / "CTBoost" / "7" / "0_1" / "results.pkl"
    raw_loading = SimpleNamespace(
        fetch_raw_result_paths=lambda _path: [artifact_path],
        load_and_align=lambda _path: Artifact(),
    )
    output = tmp_path / "report"

    rows = _write_resource_summary(tmp_path / "raw", output, _raw_loading=raw_loading)

    assert rows[0]["incremental_peak_mem_cpu_bytes"] == 50
    assert rows[0]["incremental_peak_mem_gpu_bytes"] == 60
    assert (output / "resources_per_split.csv").is_file()
    assert (
        json.loads((output / "resources_per_split.json").read_text())[0]["metric_error"]
        == 0.25
    )


def test_resource_summary_overwrites_stale_csv_when_no_results_exist(tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    csv_path = output / "resources_per_split.csv"
    csv_path.write_text("stale benchmark data\n", encoding="utf-8")
    raw_loading = SimpleNamespace(fetch_raw_result_paths=lambda _path: [])

    assert (
        _write_resource_summary(tmp_path / "raw", output, _raw_loading=raw_loading)
        == []
    )
    assert "stale benchmark data" not in csv_path.read_text(encoding="utf-8")
    assert json.loads((output / "resources_per_split.json").read_text()) == []


def test_resource_summary_can_stream_without_collecting_rows(tmp_path):
    class Artifact:
        result = {
            "framework": "CTBoost",
            "problem_type": "binary",
            "metric": "roc_auc",
            "metric_error": 0.2,
            "time_train_s": 1.0,
            "time_infer_s": 0.1,
            "task_metadata": {"name": "data", "tid": 1, "fold": 0, "repeat": 0},
            "memory_usage": {},
            "method_metadata": {},
        }

    path = tmp_path / "raw" / "data" / "CTBoost" / "1" / "0_0" / "results.pkl"
    raw_loading = SimpleNamespace(load_and_align=lambda _path: Artifact())

    count = _write_resource_summary(
        tmp_path / "raw",
        tmp_path / "report",
        _raw_loading=raw_loading,
        file_paths=[path],
        collect_rows=False,
    )

    assert count == 1
    assert (
        len(json.loads((tmp_path / "report" / "resources_per_split.json").read_text()))
        == 1
    )


def test_resource_summary_rejects_artifact_identity_that_disagrees_with_path(tmp_path):
    class Artifact:
        result = {
            "framework": "CTBoost_other",
            "problem_type": "binary",
            "metric": "roc_auc",
            "metric_error": 0.2,
            "time_train_s": 1.0,
            "time_infer_s": 0.1,
            "task_metadata": {"name": "data", "tid": 1, "fold": 0, "repeat": 0},
            "memory_usage": {},
            "method_metadata": {},
        }

    path = tmp_path / "raw" / "data" / "CTBoost" / "1" / "0_0" / "results.pkl"
    raw_loading = SimpleNamespace(load_and_align=lambda _path: Artifact())

    with pytest.raises(RuntimeError, match="identity does not match"):
        _write_resource_summary(
            tmp_path / "raw",
            tmp_path / "report",
            _raw_loading=raw_loading,
            file_paths=[path],
            collect_rows=False,
        )


def test_tabarena_job_shards_are_disjoint_and_bounded(tmp_path):
    class FakeContext:
        def __init__(self):
            self.calls = []

        def run_jobs(self, jobs, **kwargs):
            assert kwargs["register"] is False
            assert kwargs["debug_mode"] is True
            assert kwargs["cache_mode"] == "default"
            self.calls.append(list(jobs))
            return [{} for _ in jobs]

    context = FakeContext()
    stats = _run_job_shard(
        context,
        [list(range(4)), list(range(4, 10))],
        results_dir=tmp_path,
        shard_count=3,
        shard_index=1,
        job_batch_size=2,
        use_ray=False,
    )

    assert context.calls == [[1, 4], [7]]
    assert stats == {
        "total_jobs": 10,
        "selected_jobs": 3,
        "completed_results": 3,
        "batches": 2,
    }


def test_tabarena_raw_coverage_filters_stale_configs_and_rejects_missing(tmp_path):
    jobs = [
        SimpleNamespace(
            experiment=SimpleNamespace(name="CTBoost_c1"),
            task=SimpleNamespace(dataset="first", repeat=0, fold=0),
        ),
        SimpleNamespace(
            experiment=SimpleNamespace(name="CTBoost_c1"),
            task=SimpleNamespace(dataset="second", repeat=0, fold=0),
        ),
    ]
    context = SimpleNamespace(
        task_metadata_collection=SimpleNamespace(
            dataset_to_tid=lambda: {"first": 1, "second": 2}
        )
    )
    expected = _expected_raw_keys(context, [jobs])

    def raw_path(method, task, split):
        return tmp_path / "raw" / "data" / method / str(task) / split / "results.pkl"

    expected_paths = [
        raw_path("CTBoost_c1", 1, "0_0"),
        raw_path("CTBoost_c1", 2, "0_0"),
    ]
    stale_path = raw_path("CTBoost_r99", 1, "0_0")

    selected, coverage = _select_expected_raw_paths(
        expected,
        [*expected_paths, stale_path],
        allow_incomplete=False,
    )

    assert selected == sorted(expected_paths, key=str)
    assert coverage["complete"] is True
    assert coverage["ignored_stale_or_other_results"] == 1

    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        _select_expected_raw_paths(
            expected,
            expected_paths[:1],
            allow_incomplete=False,
        )

    wrong_task_paths = [
        expected_paths[0],
        raw_path("CTBoost_c1", 99, "0_0"),
    ]
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        _select_expected_raw_paths(
            expected,
            wrong_task_paths,
            allow_incomplete=False,
        )


def test_tabarena_cli_validation_and_shard_manifest_names(tmp_path):
    valid = SimpleNamespace(
        n_configs=200,
        shard_count=4,
        shard_index=2,
        job_batch_size=8,
        stage="run",
        num_cpus=8,
        num_gpus=0,
        memory_limit_gb=32,
        time_limit=3600,
    )
    _validate_args(valid)
    assert _shard_manifest_path(tmp_path, shard_count=4, shard_index=2).name == (
        "run_manifest.shard-00002-of-00004.json"
    )

    invalid = SimpleNamespace(**vars(valid))
    invalid.shard_index = 4
    with pytest.raises(SystemExit, match="shard-index"):
        _validate_args(invalid)

    too_many_configs = SimpleNamespace(**vars(valid))
    too_many_configs.n_configs = 201
    with pytest.raises(SystemExit, match="frozen 200-configuration"):
        _validate_args(too_many_configs)


def test_committed_tabarena_smoke_summary_is_sanitized_and_explicitly_provisional():
    path = Path(__file__).parents[1] / "benchmarks" / "tabarena" / "smoke_fd187da.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(summary)

    assert summary["status"] == "provisional_not_official"
    assert summary["leaderboard"]["elo"] == 1058.7
    assert summary["provenance"]["ctboost_commit"].startswith("fd187da")
    assert len(summary["per_split"]) == 3
    assert "C:\\" not in serialized
    assert "/home/" not in serialized
    assert "artifact" not in serialized.lower()


def test_console_table_replaces_characters_unsupported_by_windows_code_pages():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", newline="")

    _write_console_table("method ↑ ± score", stream=stream)
    stream.flush()

    assert raw.getvalue().decode("cp1252") == "method ? ± score\n"


def test_local_distribution_checkout_resolves_pep610_file_url(monkeypatch, tmp_path):
    checkout = tmp_path / "source checkout"
    checkout.mkdir()
    monkeypatch.setattr(
        "benchmarks.tabarena.run._distribution_source",
        lambda name: (
            {"url": checkout.resolve().as_uri()} if name == "ctboost" else None
        ),
    )

    assert _local_distribution_checkout("ctboost") == checkout.resolve()


def test_git_identity_finds_repository_above_editable_package(monkeypatch, tmp_path):
    repository = tmp_path / "tabarena"
    package = repository / "packages" / "tabarena"
    package.mkdir(parents=True)
    (repository / ".git").mkdir()
    revision = "a" * 40

    monkeypatch.setattr(
        "benchmarks.tabarena.run._local_distribution_checkout",
        lambda name: package if name == "tabarena" else None,
    )

    def fake_run(command, *, cwd, **_kwargs):
        assert cwd == repository
        output = revision + "\n" if command[1:] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr("benchmarks.tabarena.run.subprocess.run", fake_run)

    assert _git_source_identity("tabarena") == {
        "commit": revision,
        "dirty": False,
        "dirty_fingerprint_sha256": None,
        "status": [],
    }


def test_ctboost_git_identity_prefers_runtime_distribution_checkout(
    monkeypatch, tmp_path
):
    adapter_repository = tmp_path / "adapter-a"
    runtime_repository = tmp_path / "runtime-b"
    adapter_file = adapter_repository / "benchmarks" / "tabarena" / "run.py"
    adapter_file.parent.mkdir(parents=True)
    runtime_repository.mkdir()
    (adapter_repository / ".git").mkdir()
    (runtime_repository / ".git").mkdir()
    runtime_revision = "b" * 40

    monkeypatch.setattr(tabarena_run, "__file__", str(adapter_file))
    monkeypatch.setattr(
        tabarena_run,
        "_local_distribution_checkout",
        lambda name: runtime_repository if name == "ctboost" else None,
    )

    def fake_run(command, *, cwd, **_kwargs):
        assert cwd == runtime_repository
        output = runtime_revision + "\n" if command[1:] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(tabarena_run.subprocess, "run", fake_run)

    assert _git_source_identity()["commit"] == runtime_revision


def test_ctboost_git_identity_does_not_ascend_from_site_packages(monkeypatch, tmp_path):
    enclosing_repository = tmp_path / "unrelated-repository"
    adapter_file = (
        enclosing_repository
        / ".venv"
        / "Lib"
        / "site-packages"
        / "benchmarks"
        / "tabarena"
        / "run.py"
    )
    adapter_file.parent.mkdir(parents=True)
    (enclosing_repository / ".git").mkdir()
    monkeypatch.setattr(tabarena_run, "__file__", str(adapter_file))
    monkeypatch.setattr(
        tabarena_run, "_local_distribution_checkout", lambda _name: None
    )
    monkeypatch.setattr(
        tabarena_run.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "unrelated repository must not be queried"
        ),
    )

    assert _git_source_identity() is None


def test_effective_time_limit_uses_bundle_default_or_explicit_value():
    bundle = SimpleNamespace(DEFAULT_TIME_LIMIT=3_600)

    assert _resolve_effective_time_limit(None, bundle) == 3_600
    assert _resolve_effective_time_limit(90, bundle) == 90


@pytest.mark.parametrize(
    ("requested_num_gpus", "expected_gpu_count"),
    [(None, 1), (2, 2)],
)
def test_tabarena_both_device_builds_separate_cpu_and_gpu_resource_groups(
    monkeypatch,
    requested_num_gpus,
    expected_gpu_count,
):
    cpu_generator = object()
    gpu_generator = object()
    monkeypatch.setattr(tabarena_run, "gen_ctboost_cpu", cpu_generator)
    monkeypatch.setattr(tabarena_run, "gen_ctboost_gpu", gpu_generator)
    calls = []

    class FakeBundle:
        DEFAULT_TIME_LIMIT = 3_600

        def __init__(self, *, models):
            self.models = models

        def build_experiments(self, **resources):
            calls.append((self.models, resources))
            return list(self.models)

    args = SimpleNamespace(
        device="both",
        n_configs=7,
        rerun_competitors=False,
        num_cpus=4,
        num_gpus=requested_num_gpus,
        memory_limit_gb=16,
        time_limit=None,
    )

    experiments, effective_time_limit = _build_experiments(args, FakeBundle)

    assert experiments == [(cpu_generator, 7), (gpu_generator, 7)]
    assert effective_time_limit == 3_600
    assert calls == [
        (
            [(cpu_generator, 7)],
            {
                "time_limit": 3_600,
                "num_cpus": 4,
                "num_gpus": 0,
                "memory_limit": 16,
            },
        ),
        (
            [(gpu_generator, 7)],
            {
                "time_limit": 3_600,
                "num_cpus": 4,
                "num_gpus": expected_gpu_count,
                "memory_limit": 16,
            },
        ),
    ]


def test_manifest_records_ctboost_and_tabarena_git_identities(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "benchmarks.tabarena.run._git_source_identity",
        lambda name="ctboost": {"distribution": name},
    )
    monkeypatch.setattr(
        "benchmarks.tabarena.run._distribution_source", lambda _name: None
    )
    monkeypatch.setattr(
        "benchmarks.tabarena.run._ctboost_install_fingerprint", lambda: "f" * 64
    )
    args = SimpleNamespace(
        subset="lite",
        datasets=None,
        n_configs=0,
        ray=False,
        device="cpu",
        rerun_competitors=False,
        num_cpus=2,
        num_gpus=0,
        memory_limit_gb=4,
        time_limit=None,
        results_dir=tmp_path / "raw",
        output_dir=tmp_path / "report",
    )
    args.effective_time_limit = 3_600
    path = tmp_path / "manifest.json"

    _write_manifest(path, args, status="started")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["ctboost_git"] == {"distribution": "ctboost"}
    assert manifest["tabarena_git"] == {"distribution": "tabarena"}
    assert manifest["resources"]["requested_time_limit_seconds"] is None
    assert manifest["resources"]["time_limit_seconds"] == 3_600


def test_tabarena_cpu_budget_temporarily_limits_native_histogram_threads(monkeypatch):
    monkeypatch.setenv("CTBOOST_HIST_THREADS", "17")
    with _ctboost_histogram_threads(3):
        assert os.environ["CTBOOST_HIST_THREADS"] == "3"
    assert os.environ["CTBOOST_HIST_THREADS"] == "17"
