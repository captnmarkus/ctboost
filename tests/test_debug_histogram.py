import ctboost
import numpy as np

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
