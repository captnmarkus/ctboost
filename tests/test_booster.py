import numpy as np
from sklearn.datasets import make_regression

import ctboost


def test_booster_reduces_training_loss_on_regression():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=7,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda": 1.0,
            "max_bins": 64,
        },
        num_boost_round=10,
    )

    predictions = booster.predict(pool)
    baseline_loss = np.mean(y**2)
    trained_loss = np.mean((predictions - y) ** 2)

    assert booster.loss_history
    assert booster.loss_history[-1] < booster.loss_history[0]
    assert trained_loss < baseline_loss


def test_tree_controls_limit_leaf_growth_and_split_gain():
    X = np.linspace(-2.0, 2.0, 64, dtype=np.float32).reshape(-1, 1)
    y = np.where(X[:, 0] > 0.0, 2.0, -2.0).astype(np.float32)
    pool = ctboost.Pool(X, y)

    limited = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda": 1.0,
            "max_leaves": 2,
        },
        num_boost_round=1,
    )
    limited_state = limited._handle.export_state()
    limited_leaves = sum(1 for node in limited_state["trees"][0]["nodes"] if node["is_leaf"])
    assert limited_leaves <= 2

    blocked = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.3,
            "max_depth": 4,
            "alpha": 1.0,
            "lambda": 1.0,
            "gamma": 1e12,
            "min_data_in_leaf": 40,
        },
        num_boost_round=1,
    )
    blocked_state = blocked._handle.export_state()
    blocked_leaves = sum(1 for node in blocked_state["trees"][0]["nodes"] if node["is_leaf"])
    assert blocked_leaves == 1


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
