import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification, make_regression

import ctboost


def _assert_histogram_summaries_equal(left, right, *, compare_storage_bytes=True):
    for key in (
        "num_rows",
        "num_cols",
        "num_bins_per_feature",
        "cut_offsets",
        "categorical_mask",
        "missing_value_mask",
        "nan_mode",
        "nan_modes",
        "cut_values_count",
        "uses_external_bin_storage",
    ):
        assert left[key] == right[key]
    if compare_storage_bytes:
        assert left["storage_bytes"] == right["storage_bytes"]
    np.testing.assert_array_equal(left["cut_values"], right["cut_values"])


@pytest.mark.parametrize("matrix_builder", [sparse.csr_matrix, sparse.csc_matrix])
def test_pool_accepts_scipy_sparse_input(matrix_builder):
    X, y = make_regression(
        n_samples=64,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=17,
    )
    X = matrix_builder(X.astype(np.float32))
    y = y.astype(np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=6,
    )

    prediction = booster.predict(pool)
    assert prediction.shape == (X.shape[0],)
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize("matrix_builder", [sparse.csr_matrix, sparse.csc_matrix])
def test_sparse_input_does_not_call_toarray(monkeypatch, matrix_builder):
    X, y = make_regression(
        n_samples=48,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=19,
    )
    X_sparse = matrix_builder(X.astype(np.float32))
    y = y.astype(np.float32)

    def fail_toarray(self, order=None, out=None):
        raise AssertionError("dense conversion should not be used for sparse input")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", fail_toarray)

    pool = ctboost.Pool(X_sparse, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=4,
    )
    prediction = booster.predict(pool)
    assert prediction.shape == (X_sparse.shape[0],)
    assert np.all(np.isfinite(prediction))


def test_classifier_accepts_sparse_input():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    X_sparse = sparse.csr_matrix(X.astype(np.float32))
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X_sparse, y)

    probabilities = clf.predict_proba(X_sparse)
    assert probabilities.shape == (X_sparse.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(X_sparse.shape[0]), atol=1e-6)


def test_sparse_histogram_and_model_match_dense_with_implicit_zero_and_missing_values(monkeypatch):
    monkeypatch.setenv("CTBOOST_HIST_APPROX_THRESHOLD_ROWS", "0")
    monkeypatch.setenv("CTBOOST_NODE_HIST_MIN_PARALLEL_VALUES", "0")
    monkeypatch.setenv("CTBOOST_NODE_HIST_MIN_VALUES_PER_WORKER", "1")
    rng = np.random.default_rng(20260814)
    dense = np.zeros((512, 7), dtype=np.float32)
    present = rng.random(size=dense.shape) < 0.08
    dense[present] = rng.normal(size=int(present.sum())).astype(np.float32)
    dense[::53, 2] = np.nan
    dense[:, 5] = rng.integers(0, 4, size=dense.shape[0]).astype(np.float32)
    labels = (
        1.3 * np.nan_to_num(dense[:, 2])
        - 0.4 * dense[:, 3]
        + 0.2 * (dense[:, 5] == 3.0)
    ).astype(np.float32)

    dense_pool = ctboost.Pool(dense, labels, cat_features=[5])
    sparse_pool = ctboost.Pool(sparse.csc_matrix(dense), labels, cat_features=[5])
    dense_hist = ctboost._core._debug_build_histogram(
        dense_pool._handle,
        max_bins=32,
        nan_mode="Max",
    )
    sparse_hist = ctboost._core._debug_build_histogram(
        sparse_pool._handle,
        max_bins=32,
        nan_mode="Max",
    )

    _assert_histogram_summaries_equal(sparse_hist, dense_hist)

    params = {
        "objective": "RMSE",
        "iterations": 8,
        "learning_rate": 0.15,
        "max_depth": 3,
        "max_bins": 32,
        "nan_mode": "Max",
        "alpha": 1.0,
        "random_seed": 17,
    }
    monkeypatch.setenv("CTBOOST_HIST_THREADS", "1")
    dense_model = ctboost.train(dense_pool, params, num_boost_round=params["iterations"])
    monkeypatch.setenv("CTBOOST_HIST_THREADS", "4")
    sparse_model = ctboost.train(sparse_pool, params, num_boost_round=params["iterations"])

    assert sparse_model._handle.export_state()["trees"] == dense_model._handle.export_state()["trees"]
    np.testing.assert_array_equal(sparse_model.predict(sparse_pool), dense_model.predict(dense_pool))


def test_sparse_forbidden_nan_validation_scans_only_stored_values():
    matrix = sparse.csc_matrix(
        (
            np.asarray([1.0, np.nan], dtype=np.float32),
            np.asarray([2, 5], dtype=np.int32),
            np.asarray([0, 2], dtype=np.int32),
        ),
        shape=(8, 1),
    )
    pool = ctboost.Pool(matrix, np.zeros(8, dtype=np.float32))

    with pytest.raises(ValueError, match="NaN values are not allowed"):
        ctboost._core._debug_build_histogram(
            pool._handle,
            max_bins=8,
            nan_mode="Forbidden",
        )


def test_empty_sparse_histogram_avoids_null_bin_buffer_arithmetic():
    matrix = sparse.csc_matrix((0, 3), dtype=np.float32)
    pool = ctboost.Pool(matrix, np.empty(0, dtype=np.float32), cat_features=[2])

    summary = ctboost._core._debug_build_histogram(
        pool._handle,
        max_bins=300,
        nan_mode="Min",
    )

    assert summary["num_rows"] == 0
    assert summary["num_cols"] == 3
    assert summary["uses_external_bin_storage"] is False


@pytest.mark.parametrize(
    ("environment", "histogram_options"),
    [
        (
            {
                "CTBOOST_HIST_APPROX_THRESHOLD_ROWS": "1",
                "CTBOOST_HIST_APPROX_SAMPLE_SIZE": "3072",
            },
            {"max_bins": 32, "nan_mode": "Max"},
        ),
        (
            {"CTBOOST_HIST_APPROX_THRESHOLD_ROWS": "0"},
            {
                "max_bins": 64,
                "nan_mode": "Min",
                "border_selection_method": "Uniform",
                "nan_mode_by_feature": ["Max", "", "Min", ""],
            },
        ),
        (
            {"CTBOOST_HIST_APPROX_THRESHOLD_ROWS": "0"},
            {
                "max_bins": 64,
                "nan_mode": "Min",
                "feature_borders": [[-0.75, 0.0, 0.5], [], [], []],
            },
        ),
        (
            {"CTBOOST_HIST_APPROX_THRESHOLD_ROWS": "0"},
            {"max_bins": 300, "nan_mode": "Min"},
        ),
    ],
)
def test_sparse_histogram_matches_dense_across_quantization_paths(
    monkeypatch,
    environment,
    histogram_options,
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    rng = np.random.default_rng(9107)
    rows = 4096 if "CTBOOST_HIST_APPROX_SAMPLE_SIZE" in environment else 1024
    dense = rng.normal(size=(rows, 4)).astype(np.float32)
    dense[rng.random(size=dense.shape) < 0.82] = 0.0
    dense[:, 1] = np.linspace(-10.0, 10.0, rows, dtype=np.float32)
    dense[::71, 0] = np.nan
    dense[:, 2] = rng.integers(0, 5, size=dense.shape[0]).astype(np.float32)

    sparse_matrix = sparse.csc_matrix(dense)
    # Retain an explicit stored zero to cover both canonical CSC zero forms.
    sparse_matrix.data[0] = 0.0
    dense = sparse_matrix.toarray().astype(np.float32, copy=False)
    labels = np.zeros(dense.shape[0], dtype=np.float32)
    dense_pool = ctboost.Pool(dense, labels, cat_features=[2])
    sparse_pool = ctboost.Pool(sparse_matrix, labels, cat_features=[2])

    dense_hist = ctboost._core._debug_build_histogram(
        dense_pool._handle,
        **histogram_options,
    )
    sparse_hist = ctboost._core._debug_build_histogram(
        sparse_pool._handle,
        **histogram_options,
    )

    _assert_histogram_summaries_equal(sparse_hist, dense_hist)
    if "CTBOOST_HIST_APPROX_SAMPLE_SIZE" in environment:
        monkeypatch.setenv("CTBOOST_HIST_APPROX_THRESHOLD_ROWS", "0")
        exact_hist = ctboost._core._debug_build_histogram(
            dense_pool._handle,
            **histogram_options,
        )
        assert not np.array_equal(dense_hist["cut_values"], exact_hist["cut_values"])
    if histogram_options["max_bins"] > 256:
        assert dense_hist["num_bins_per_feature"][1] > 256


def test_sparse_external_histogram_matches_dense(tmp_path):
    rng = np.random.default_rng(812)
    dense = rng.normal(size=(512, 6)).astype(np.float32)
    dense[rng.random(size=dense.shape) < 0.9] = 0.0
    dense[::43, 1] = np.nan
    labels = np.zeros(dense.shape[0], dtype=np.float32)
    dense_pool = ctboost.Pool(dense, labels)
    sparse_pool = ctboost.Pool(sparse.csc_matrix(dense), labels)

    dense_hist = ctboost._core._debug_build_histogram(
        dense_pool._handle,
        max_bins=300,
        nan_mode="Max",
        external_memory=True,
        external_memory_dir=str(tmp_path / "dense"),
    )
    sparse_hist = ctboost._core._debug_build_histogram(
        sparse_pool._handle,
        max_bins=300,
        nan_mode="Max",
        external_memory=True,
        external_memory_dir=str(tmp_path / "sparse"),
    )

    _assert_histogram_summaries_equal(
        sparse_hist,
        dense_hist,
        compare_storage_bytes=False,
    )
