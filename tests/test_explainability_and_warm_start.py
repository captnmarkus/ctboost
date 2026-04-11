import numpy as np
from sklearn.datasets import make_regression

import ctboost


def test_low_level_init_model_matches_longer_single_run():
    X, y = make_regression(
        n_samples=140,
        n_features=6,
        n_informative=5,
        noise=0.2,
        random_state=37,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)
    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
    }

    initial = ctboost.train(pool, params, num_boost_round=5)
    continued = ctboost.train(pool, params, num_boost_round=4, init_model=initial)
    full = ctboost.train(pool, params, num_boost_round=9)

    np.testing.assert_allclose(continued.predict(pool), full.predict(pool), rtol=1e-6, atol=1e-6)
    assert continued.num_iterations_trained == 9


def test_predict_leaf_index_and_contrib_sum_to_prediction():
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=43,
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
            "lambda_l2": 1.0,
        },
        num_boost_round=6,
    )

    leaf_indices = booster.predict_leaf_index(pool)
    contributions = booster.predict_contrib(pool)
    predictions = booster.predict(pool)

    assert leaf_indices.shape == (X.shape[0], booster.num_iterations_trained)
    assert contributions.shape == (X.shape[0], X.shape[1] + 1)
    np.testing.assert_allclose(contributions.sum(axis=1), predictions, rtol=1e-6, atol=1e-6)


def test_regressor_warm_start_adds_iterations():
    X, y = make_regression(
        n_samples=120,
        n_features=6,
        n_informative=5,
        noise=0.1,
        random_state=59,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = ctboost.CTBoostRegressor(
        iterations=4,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        warm_start=True,
    )
    model.fit(X, y)
    model.iterations = 3
    model.fit(X, y)

    assert model._booster.num_iterations_trained == 7
    contrib = model.predict_contrib(X)
    np.testing.assert_allclose(contrib.sum(axis=1), model.predict(X), rtol=1e-6, atol=1e-6)
