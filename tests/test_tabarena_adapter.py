import json
import io
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.tabarena.ctboost_model import (
    CTBoostTabArenaGPUModel,
    CTBoostTabArenaModel,
    _ctboost_histogram_threads,
    normalize_tabarena_frame,
)
from benchmarks.tabarena.run import (
    _format_table,
    _git_source_identity,
    _local_distribution_checkout,
    _write_console_table,
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


def test_tabarena_cpu_budget_temporarily_limits_native_histogram_threads(monkeypatch):
    monkeypatch.setenv("CTBOOST_HIST_THREADS", "17")
    with _ctboost_histogram_threads(3):
        assert os.environ["CTBOOST_HIST_THREADS"] == "3"
    assert os.environ["CTBOOST_HIST_THREADS"] == "17"
