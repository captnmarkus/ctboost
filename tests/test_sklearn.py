import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import ctboost


def _make_classification_data():
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=13,
    )
    return X.astype(np.float32), y.astype(np.float32)


def test_classifier_predict_proba_and_feature_importances():
    X, y = _make_classification_data()

    clf = ctboost.CTBoostClassifier(
        iterations=18,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    probabilities = clf.predict_proba(X)
    predictions = clf.predict(X)
    importances = clf.feature_importances_

    assert probabilities.shape == (X.shape[0], 2)
    assert predictions.shape == (X.shape[0],)
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(importances.sum(), 100.0, rtol=1e-5, atol=1e-5)


def test_regressor_feature_importances_match_feature_count():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=21,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    reg = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    reg.fit(X, y)

    assert reg.predict(X).shape == (X.shape[0],)
    assert reg.feature_importances_.shape == (X.shape[1],)


def test_classifier_cpu_gpu_parity_or_graceful_cuda_error():
    X, y = _make_classification_data()

    cpu = ctboost.CTBoostClassifier(
        iterations=16,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        task_type="CPU",
    )
    cpu.fit(X, y)

    if not ctboost.build_info()["cuda_enabled"]:
        gpu = ctboost.CTBoostClassifier(
            iterations=16,
            learning_rate=0.15,
            max_depth=2,
            alpha=1.0,
            lambda_l2=1.0,
            task_type="GPU",
        )
        with pytest.raises(RuntimeError, match="compiled without CUDA"):
            gpu.fit(X, y)
        return

    gpu = ctboost.CTBoostClassifier(
        iterations=16,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        task_type="GPU",
    )
    try:
        gpu.fit(X, y)
    except RuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable for parity test: {exc}")

    np.testing.assert_allclose(
        gpu.predict_proba(X),
        cpu.predict_proba(X),
        rtol=1e-4,
        atol=1e-4,
    )
