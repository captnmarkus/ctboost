import numpy as np
from sklearn.datasets import make_classification, make_regression

import ctboost


def test_regressor_sample_weight_matches_weighted_pool():
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    sample_weight = np.linspace(0.5, 2.0, num=X.shape[0], dtype=np.float32)

    weighted_pool = ctboost.Pool(X, y, weight=sample_weight)
    direct = ctboost.train(
        weighted_pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=12,
    )

    estimator = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    estimator.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(estimator.predict(X), direct.predict(weighted_pool), rtol=1e-6, atol=1e-6)


def test_scale_pos_weight_matches_explicit_binary_sample_weights():
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        weights=[0.8, 0.2],
        random_state=37,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    sample_weight = np.where(y == 1.0, 4.0, 1.0).astype(np.float32)

    weighted = ctboost.train(
        ctboost.Pool(X, y, weight=sample_weight),
        {
            "objective": "Logloss",
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=14,
    )
    scaled = ctboost.train(
        ctboost.Pool(X, y),
        {
            "objective": "Logloss",
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "scale_pos_weight": 4.0,
        },
        num_boost_round=14,
    )

    pool = ctboost.Pool(X, y)
    np.testing.assert_allclose(weighted.predict(pool), scaled.predict(pool), rtol=1e-6, atol=1e-6)


def test_classifier_class_weight_matches_explicit_sample_weights():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=43,
    )
    X = X.astype(np.float32)
    sample_weight = np.where(y == 1, 3.0, 1.0).astype(np.float32)

    weighted = ctboost.CTBoostClassifier(
        iterations=16,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    weighted.fit(X, y, sample_weight=sample_weight)

    balanced = ctboost.CTBoostClassifier(
        iterations=16,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        class_weight={0: 1.0, 1: 3.0},
    )
    balanced.fit(X, y)

    np.testing.assert_allclose(weighted.predict_proba(X), balanced.predict_proba(X), rtol=1e-6, atol=1e-6)
