import json
import io
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import benchmarks.tabarena.ctboost_model as ctboost_adapter
import benchmarks.tabarena.run as tabarena_run
from benchmarks.tabarena.ctboost_model import (
    CTBoostTabArenaGPUModel,
    CTBoostTabArenaModel,
    _ctboost_eval_metric,
    _ctboost_histogram_threads,
    _resolve_time_limit,
    normalize_tabarena_frame,
)
from benchmarks.tabarena.run import (
    _format_table,
    _git_source_identity,
    _local_distribution_checkout,
    _resolve_effective_time_limit,
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
    assert normalized["city"].tolist() == ["Berlin", "__CTBOOST_MISSING__", "Paris"]
    assert normalized["flag"].dtype == np.int8
    assert np.isnan(normalized.loc[1, "value"])


def test_tabarena_frame_reuses_training_schema_and_rejects_missing_columns():
    frame = pd.DataFrame({"city": ["Berlin"], "value": [1.0]})
    normalized, categorical = normalize_tabarena_frame(
        frame,
        categorical_columns=["city"],
    )

    assert categorical == ["city"]
    assert normalized["city"].tolist() == ["Berlin"]

    with pytest.raises(ValueError, match="missing categorical columns"):
        normalize_tabarena_frame(pd.DataFrame({"value": [1.0]}), categorical_columns=["city"])


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
    assert CTBoostTabArenaModel.default_num_gpus == 0
    assert CTBoostTabArenaGPUModel.ag_name == "CTBoostGPU"
    assert CTBoostTabArenaGPUModel.default_num_gpus == 1
    assert CTBoostTabArenaGPUModel.minimum_num_gpus == 1
    assert CTBoostTabArenaGPUModel.gpu_required is True


def _model_adapter(*, problem_type, stopping_metric, params):
    adapter = object.__new__(CTBoostTabArenaModel)
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

    adapter._fit(pd.DataFrame({"value": [0.0, 1.0]}), np.array([0.0, 1.0]), time_limit=5)

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
            "task_metadata": {"name": "public-data", "tid": 7, "fold": 1, "repeat": 0, "sample": 0},
            "memory_usage": {
                "peak_mem_cpu": 150,
                "min_mem_cpu": 100,
                "peak_mem_gpu": 80,
                "min_mem_gpu": 20,
                "gpu_tracking_enabled": True,
            },
            "method_metadata": {"num_cpus": 4, "num_gpus": 1, "disk_usage": 1234},
        }

    artifact_path = tmp_path / "raw" / "results.pkl"
    raw_loading = SimpleNamespace(
        fetch_raw_result_paths=lambda _path: [artifact_path],
        load_and_align=lambda _path: Artifact(),
    )
    output = tmp_path / "report"

    rows = _write_resource_summary(tmp_path / "raw", output, _raw_loading=raw_loading)

    assert rows[0]["incremental_peak_mem_cpu_bytes"] == 50
    assert rows[0]["incremental_peak_mem_gpu_bytes"] == 60
    assert (output / "resources_per_split.csv").is_file()
    assert json.loads((output / "resources_per_split.json").read_text())[0]["metric_error"] == 0.25


def test_resource_summary_overwrites_stale_csv_when_no_results_exist(tmp_path):
    output = tmp_path / "report"
    output.mkdir()
    csv_path = output / "resources_per_split.csv"
    csv_path.write_text("stale benchmark data\n", encoding="utf-8")
    raw_loading = SimpleNamespace(fetch_raw_result_paths=lambda _path: [])

    assert _write_resource_summary(tmp_path / "raw", output, _raw_loading=raw_loading) == []
    assert "stale benchmark data" not in csv_path.read_text(encoding="utf-8")
    assert json.loads((output / "resources_per_split.json").read_text()) == []


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
        lambda name: {"url": checkout.resolve().as_uri()} if name == "ctboost" else None,
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


def test_ctboost_git_identity_prefers_runtime_distribution_checkout(monkeypatch, tmp_path):
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
    monkeypatch.setattr(tabarena_run, "_local_distribution_checkout", lambda _name: None)
    monkeypatch.setattr(
        tabarena_run.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("unrelated repository must not be queried"),
    )

    assert _git_source_identity() is None


def test_effective_time_limit_uses_bundle_default_or_explicit_value():
    bundle = SimpleNamespace(DEFAULT_TIME_LIMIT=3_600)

    assert _resolve_effective_time_limit(None, bundle) == 3_600
    assert _resolve_effective_time_limit(90, bundle) == 90


def test_manifest_records_ctboost_and_tabarena_git_identities(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "benchmarks.tabarena.run._git_source_identity",
        lambda name="ctboost": {"distribution": name},
    )
    monkeypatch.setattr("benchmarks.tabarena.run._distribution_source", lambda _name: None)
    monkeypatch.setattr("benchmarks.tabarena.run._ctboost_install_fingerprint", lambda: "f" * 64)
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
