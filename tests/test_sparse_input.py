import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification, make_regression

import ctboost


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
