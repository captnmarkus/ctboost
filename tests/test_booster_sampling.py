import numpy as np
import pytest
from sklearn.datasets import make_regression
import ctboost

def test_feature_subsampling_is_seeded_and_deterministic():
    X, y = make_regression(
        n_samples=160,
        n_features=10,
        n_informative=6,
        noise=0.2,
        random_state=11,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "colsample_bytree": 0.5,
        "random_seed": 23,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=8)
    booster_b = ctboost.train(pool, params, num_boost_round=8)
    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)

    booster_c = ctboost.train(
        pool,
        {**params, "random_seed": 99},
        num_boost_round=8,
    )
    preds_c = booster_c.predict(pool)
    assert not np.allclose(preds_a, preds_c)

def test_row_subsampling_is_seeded_and_random_forest_mode_trains():
    X, y = make_regression(
        n_samples=180,
        n_features=8,
        n_informative=5,
        noise=0.3,
        random_state=19,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "subsample": 0.7,
        "bootstrap_type": "Bernoulli",
        "boosting_type": "RandomForest",
        "colsample_bytree": 0.75,
        "random_seed": 29,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=10)
    booster_b = ctboost.train(pool, params, num_boost_round=10)

    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert np.mean((preds_a - y) ** 2) < np.mean(y**2)

def test_bayesian_bootstrap_with_bagging_temperature_is_seeded_and_trains():
    X, y = make_regression(
        n_samples=180,
        n_features=8,
        n_informative=5,
        noise=0.3,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.5,
        "random_seed": 41,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=10)
    booster_b = ctboost.train(pool, params, num_boost_round=10)

    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)
    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert np.mean((preds_a - y) ** 2) < np.mean(y**2)

def test_dart_boosting_is_seeded_and_trains():
    X, y = make_regression(
        n_samples=180,
        n_features=8,
        n_informative=5,
        noise=0.25,
        random_state=23,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda": 1.0,
        "boosting_type": "DART",
        "drop_rate": 0.2,
        "skip_drop": 0.0,
        "max_drop": 2,
        "random_seed": 37,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=12)
    booster_b = ctboost.train(pool, params, num_boost_round=12)
    preds_a = booster_a.predict(pool)
    preds_b = booster_b.predict(pool)

    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert np.mean((preds_a - y) ** 2) < np.mean(y**2)

def test_random_strength_is_seeded():
    rng = np.random.default_rng(83)
    base = rng.normal(size=160).astype(np.float32)
    X = np.column_stack(
        [
            base + 0.01 * rng.normal(size=160),
            base + 0.02 * rng.normal(size=160),
            base + 0.03 * rng.normal(size=160),
        ]
    ).astype(np.float32)
    y = base.astype(np.float32)
    pool = ctboost.Pool(X, y)

    params = {
        "objective": "RMSE",
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_strength": 50.0,
        "random_seed": 53,
    }
    booster_a = ctboost.train(pool, params, num_boost_round=4)
    booster_b = ctboost.train(pool, params, num_boost_round=4)
    booster_c = ctboost.train(pool, {**params, "random_seed": 61}, num_boost_round=4)
    state_a = booster_a._handle.export_state()
    state_b = booster_b._handle.export_state()
    state_c = booster_c._handle.export_state()

    assert state_a["trees"][0]["nodes"][0]["split_feature_id"] == state_b["trees"][0]["nodes"][0]["split_feature_id"]
    assert state_a["trees"][0]["nodes"][0]["split_feature_id"] != state_c["trees"][0]["nodes"][0]["split_feature_id"]
