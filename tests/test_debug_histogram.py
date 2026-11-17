import ctboost
import numpy as np
import pytest

def test_debug_build_histogram_reports_matrix_shape():
    pool = ctboost.Pool(
        data=[[0.0, 1.0], [1.0, 0.0], [0.5, np.nan]],
        label=[0, 1, 0],
    )
    summary = ctboost._core._debug_build_histogram(pool._handle, max_bins=16, nan_mode="Min")

    assert summary["num_rows"] == 3
    assert summary["num_cols"] == 2
    assert len(summary["num_bins_per_feature"]) == 2
    assert len(summary["cut_offsets"]) == 3
    assert summary["elapsed_ms"] >= 0.0

def test_exact_quantile_builder_matches_sorted_cut_positions(monkeypatch):
    monkeypatch.setenv("CTBOOST_HIST_APPROX_THRESHOLD_ROWS", "0")

    values = np.array([7.0, 1.0, 3.0, 9.0, 5.0, 2.0, 8.0, 4.0, 6.0], dtype=np.float32)
    pool = ctboost.Pool(values.reshape(-1, 1), np.zeros(values.shape[0], dtype=np.float32))
    summary = ctboost._core._debug_build_histogram(pool._handle, max_bins=4, nan_mode="Min")

    expected = np.sort(values)[[2, 4, 6]].astype(np.float32)
    np.testing.assert_allclose(np.asarray(summary["cut_values"], dtype=np.float32), expected)

def test_selection_quantile_builder_matches_sorted_cut_positions(monkeypatch):
    monkeypatch.setenv("CTBOOST_HIST_APPROX_THRESHOLD_ROWS", "0")
    monkeypatch.setenv("CTBOOST_HIST_EXACT_SELECT_THRESHOLD_ROWS", "1")

    values = np.linspace(1.0, 4096.0, 4096, dtype=np.float32)
    rng = np.random.default_rng(7)
    rng.shuffle(values)
    pool = ctboost.Pool(values.reshape(-1, 1), np.zeros(values.shape[0], dtype=np.float32))
    summary = ctboost._core._debug_build_histogram(pool._handle, max_bins=8, nan_mode="Min")

    expected = np.sort(values)[[512, 1024, 1536, 2048, 2560, 3072, 3584]].astype(np.float32)
    np.testing.assert_allclose(np.asarray(summary["cut_values"], dtype=np.float32), expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("sampled", [False, True])
def test_automatic_quantile_keeps_a_cut_for_a_rare_numeric_tail(monkeypatch, sparse, sampled):
    monkeypatch.setenv("CTBOOST_HIST_APPROX_THRESHOLD_ROWS", "1" if sampled else "0")
    monkeypatch.setenv("CTBOOST_HIST_APPROX_SAMPLE_SIZE", "4096")
    data = np.zeros((8192, 6), dtype=np.float32)
    # The deterministic approximation samples even positions only. The lower
    # of the two rare tail values must still come from the full training pool.
    data[1, 0], data[3, 0], data[5, 0] = 7.0, 3.0, np.nan
    data[1, 1] = -2.0  # A cut at zero already separates the observations.
    data[1, 3:] = 3.0
    data[3, 5] = -2.0
    if sparse:
        data = pytest.importorskip("scipy.sparse").csc_matrix(data)
    pool = ctboost.Pool(data, np.zeros(8192, dtype=np.float32))
    result = ctboost._core._debug_build_histogram(
        pool._handle, max_bins=2, nan_mode="Min", max_bin_by_feature=[2, 2, 2, 1, 2, 2],
        feature_borders=[[], [], [], [], [0.0], []],
    )
    cuts = np.asarray(result["cut_values"])
    borders = np.split(cuts, result["cut_offsets"][1:-1])
    assert [values.tolist() for values in borders] == [[3.0], [0.0], [0.0], [], [0.0], [0.0]]
    assert result["num_bins_per_feature"] == [3, 2, 2, 1, 2, 2]


def test_rare_indicator_predictions_and_saved_quantile_borders(tmp_path):
    values = np.concatenate([np.zeros(5000), np.ones(10)]).astype(np.float32)
    pool = ctboost.Pool(values[:, None], values, weight=1.0 + values)
    params = {"objective": "Logloss", "max_depth": 1}
    automatic = ctboost.train(pool, params, num_boost_round=8)
    assert automatic.get_quantization_schema()["cut_values"] == [1.0]
    assert automatic.predict(pool)[-1] > automatic.predict(pool)[0]

    # Earlier automatic schemas could contain this empty cut. Loading and
    # continuing an existing ensemble must keep its fitted feature mapping.
    old = ctboost.train(pool, {**params, "feature_borders": [[0.0]]}, num_boost_round=8)
    path = tmp_path / "old-borders.json"
    old.save_model(path)
    restored = ctboost.Booster.load_model(path)
    np.testing.assert_array_equal(old.predict(pool), restored.predict(pool))
    continued = ctboost.train(pool, params, init_model=restored, num_boost_round=1)
    assert continued.get_quantization_schema()["cut_values"] == [0.0]
    assert continued.predict(pool)[0] == continued.predict(pool)[-1]


@pytest.mark.parametrize("common_value", [0.0, np.finfo(np.float32).max])
def test_rare_infinite_tail_keeps_quantile_border_finite(common_value, tmp_path):
    values = np.full(1000, common_value, dtype=np.float32)
    values[-2:] = np.inf
    pool = ctboost.Pool(values[:, None], np.isinf(values).astype(np.float32))
    model = ctboost.train(pool, {"objective": "Logloss", "max_depth": 1}, num_boost_round=3)
    assert model.get_quantization_schema()["cut_values"] == [np.finfo(np.float32).max]
    path = tmp_path / "infinite-tail.json"
    model.save_model(path)
    assert "Infinity" not in path.read_text(encoding="utf-8")
    predictions = model.predict(pool)
    np.testing.assert_array_equal(ctboost.Booster.load_model(path).predict(pool), predictions)
    assert (predictions[-1] > predictions[0]) == (common_value < np.finfo(np.float32).max)

def test_debug_build_histogram_supports_per_feature_controls_and_custom_borders():
    data = np.array(
        [
            [0.0, -1.0, np.nan],
            [0.2, 0.0, 0.5],
            [0.4, 1.0, 1.5],
            [0.6, 2.0, 2.5],
            [0.8, 3.0, np.nan],
        ],
        dtype=np.float32,
    )
    pool = ctboost.Pool(data, np.zeros(data.shape[0], dtype=np.float32))
    summary = ctboost._core._debug_build_histogram(
        pool._handle,
        max_bins=16,
        nan_mode="Min",
        max_bin_by_feature=[3, 5, 0],
        border_selection_method="Uniform",
        nan_mode_by_feature=["", "Max", "Min"],
        feature_borders=[[], [], [0.75, 1.25]],
    )

    assert summary["num_bins_per_feature"][0] == 3
    np.testing.assert_allclose(
        np.asarray(summary["cut_values"], dtype=np.float32)[
            summary["cut_offsets"][2] : summary["cut_offsets"][3]
        ],
        np.array([0.75, 1.25], dtype=np.float32),
    )
    assert summary["nan_modes"] == [1, 2, 1]

def test_debug_build_histogram_supports_native_external_memory(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(256, 5)).astype(np.float32)
    y = rng.normal(size=256).astype(np.float32)
    pool = ctboost.Pool(X, y)

    summary = ctboost._core._debug_build_histogram(
        pool._handle,
        max_bins=64,
        nan_mode="Min",
        external_memory=True,
        external_memory_dir=str(tmp_path / "hist_spill"),
    )

    assert summary["uses_external_bin_storage"] is True
    assert summary["num_rows"] == 256
    assert summary["num_cols"] == 5
    assert summary["storage_bytes"] < X.nbytes

def test_large_histogram_build_paths_are_deterministic(monkeypatch):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(4096, 4)).astype(np.float32)
    y = (0.7 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2]).astype(np.float32)

    monkeypatch.setenv("CTBOOST_HIST_THREADS", "2")
    monkeypatch.setenv("CTBOOST_HIST_APPROX_THRESHOLD_ROWS", "256")
    monkeypatch.setenv("CTBOOST_HIST_APPROX_SAMPLE_SIZE", "128")

    model_a = ctboost.CTBoostRegressor(iterations=6, max_depth=3)
    model_b = ctboost.CTBoostRegressor(iterations=6, max_depth=3)
    model_a.fit(X, y)
    model_b.fit(X, y)

    preds_a = model_a.predict(X[:128])
    preds_b = model_b.predict(X[:128])
    assert preds_a.shape == (128,)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)


def test_node_histogram_thread_count_preserves_exact_tree_state(monkeypatch):
    rng = np.random.default_rng(20260814)
    X = rng.normal(size=(2048, 32)).astype(np.float32)
    y = (1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] * X[:, 3]).astype(
        np.float32
    )
    pool = ctboost.Pool(X, y)
    params = {
        "objective": "RMSE",
        "iterations": 8,
        "max_depth": 4,
        "max_bins": 64,
        "alpha": 1.0,
        "random_seed": 37,
    }
    monkeypatch.setenv("CTBOOST_HIST_THREADS", "1")
    monkeypatch.setenv("CTBOOST_NODE_HIST_MIN_PARALLEL_VALUES", "0")
    monkeypatch.setenv("CTBOOST_NODE_HIST_MIN_VALUES_PER_WORKER", "1")

    monkeypatch.setenv("CTBOOST_NODE_HIST_THREADS", "1")
    serial = ctboost.train(pool, params, num_boost_round=params["iterations"])
    monkeypatch.setenv("CTBOOST_NODE_HIST_THREADS", "4")
    parallel = ctboost.train(pool, params, num_boost_round=params["iterations"])

    assert parallel._handle.export_state()["trees"] == serial._handle.export_state()["trees"]
    np.testing.assert_array_equal(parallel.predict(pool), serial.predict(pool))
