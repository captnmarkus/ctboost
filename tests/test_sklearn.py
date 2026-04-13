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


def test_regressor_early_stopping_uses_eval_set():
    X, y = make_regression(
        n_samples=256,
        n_features=8,
        n_informative=5,
        noise=20.0,
        random_state=29,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_val = X[:96], X[96:]
    y_train = y[:96]
    y_val = -y[96:]

    reg = ctboost.CTBoostRegressor(
        iterations=500,
        learning_rate=0.3,
        max_depth=4,
        alpha=1.0,
        lambda_l2=0.5,
    )
    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=15,
    )

    predictions = reg.predict(X_val)

    assert reg.best_iteration_ + 1 < 500
    assert len(reg._booster.loss_history) == reg.best_iteration_ + 1
    assert predictions.shape == (X_val.shape[0],)


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


def test_feature_pipeline_handles_ctr_text_and_embedding_columns(tmp_path):
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(41)
    category = np.where(rng.random(96) > 0.5, "alpha", "beta")
    text = np.where(category == "alpha", "red quick fox", "blue slow fox")
    embedding = [np.asarray([float(index % 3), float(index % 5), float(index % 7)], dtype=np.float32) for index in range(96)]
    numeric = rng.normal(size=96).astype(np.float32)
    target = (
        (category == "alpha").astype(np.float32) * 1.5
        + 0.25 * numeric
        + np.asarray([values[0] - 0.1 * values[1] for values in embedding], dtype=np.float32)
    )

    frame = pd.DataFrame(
        {
            "category": category,
            "text": text,
            "embedding": embedding,
            "numeric": numeric,
        }
    )

    reg = ctboost.CTBoostRegressor(
        iterations=18,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        ordered_ctr=True,
        cat_features=["category"],
        categorical_combinations=[["category", "text"]],
        text_features=["text"],
        text_hash_dim=16,
        embedding_features=["embedding"],
        embedding_stats=("mean", "std", "l2"),
    )
    reg.fit(frame, target)

    prediction = reg.predict(frame)
    assert prediction.shape == (frame.shape[0],)
    assert np.all(np.isfinite(prediction))
    assert reg.n_features_in_ > frame.shape[1]

    model_path = tmp_path / "pipeline_regressor.ctb"
    reg.save_model(model_path)
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    restored_prediction = restored.predict(frame)
    np.testing.assert_allclose(prediction, restored_prediction, rtol=1e-6, atol=1e-6)
