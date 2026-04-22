import numpy as np
import pytest
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

def test_poisson_and_tweedie_regression_train_with_finite_predictions():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(128, 4)).astype(np.float32)
    linear = 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.1 * X[:, 2]
    y_count = np.exp(linear).astype(np.float32)
    y_tweedie = (np.exp(linear) + 0.1).astype(np.float32)

    poisson = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        loss_function="Poisson",
    )
    poisson.fit(X, y_count)
    poisson_pred = poisson.predict(X)
    assert np.all(np.isfinite(poisson_pred))

    tweedie = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.15,
        max_depth=2,
        loss_function="Tweedie",
        tweedie_variance_power=1.4,
    )
    tweedie.fit(X, y_tweedie)
    tweedie_pred = tweedie.predict(X)
    assert np.all(np.isfinite(tweedie_pred))

def test_survival_regression_improves_concordance():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(220, 4)).astype(np.float32)
    risk = 0.8 * X[:, 0] - 0.5 * X[:, 1]
    event_time = np.exp(1.1 - risk) + 0.05 * rng.random(X.shape[0])
    censor_time = np.quantile(event_time, 0.7)
    observed = event_time <= censor_time
    signed_time = np.where(observed, event_time, -np.minimum(event_time, censor_time)).astype(np.float32)

    booster = ctboost.train(
        ctboost.Pool(X, signed_time),
        {
            "objective": "Cox",
            "eval_metric": "CIndex",
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=18,
    )
    predictions = booster.predict(X)
    assert np.all(np.isfinite(predictions))

    comparable = 0.0
    concordant = 0.0
    observed_time = np.abs(signed_time)
    for i in range(X.shape[0]):
        if signed_time[i] <= 0.0:
            continue
        for j in range(X.shape[0]):
            if observed_time[j] <= observed_time[i]:
                continue
            comparable += 1.0
            if predictions[i] > predictions[j]:
                concordant += 1.0
            elif predictions[i] == predictions[j]:
                concordant += 0.5
    assert comparable > 0.0
    assert concordant / comparable > 0.6
