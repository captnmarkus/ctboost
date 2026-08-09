import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import balanced_accuracy_score
import ctboost
import ctboost._core as _core


def _signed_accuracy_metric(predictions, label, **kwargs):
    del kwargs
    resolved_predictions = np.asarray(predictions, dtype=np.float32)
    resolved_label = np.asarray(label, dtype=np.float32)
    return float(np.mean((resolved_predictions >= 0.0).astype(np.float32) == resolved_label))


@pytest.mark.parametrize(
    ("objective", "extra_params"),
    [
        ("MAE", {}),
        ("Huber", {"huber_delta": 1.5}),
        ("Quantile", {"quantile_alpha": 0.8}),
    ],
)
def test_additional_regression_objectives_train_and_predict(objective, extra_params):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=67,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    booster = ctboost.train(
        ctboost.Pool(X, y),
        {
            "objective": objective,
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            **extra_params,
        },
        num_boost_round=14,
    )

    predictions = booster.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert np.all(np.isfinite(predictions))
    assert len(booster.loss_history) == 14

def test_gamma_regressor_honors_loss_function_and_trains_with_positive_targets():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(220, 5)).astype(np.float32)
    expected_mean = np.exp(0.5 + 0.7 * X[:, 0] - 0.4 * X[:, 1])
    y = rng.gamma(shape=5.0, scale=expected_mean / 5.0).astype(np.float32)

    regressor = ctboost.CTBoostRegressor(
        iterations=24,
        learning_rate=0.15,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        loss_function="Gamma",
        random_seed=13,
    )
    regressor.fit(X, y)

    predictions = regressor.predict(X)
    assert regressor._booster.objective_name == "Gamma"
    assert predictions.shape == y.shape
    assert np.all(np.isfinite(predictions))
    assert len(regressor._booster.loss_history) == regressor.iterations
    assert regressor._booster.loss_history[-1] < regressor._booster.loss_history[0]
    np.testing.assert_array_equal(regressor.predict_raw(X), predictions)
    mean_prediction = regressor.predict_mean(X)
    np.testing.assert_allclose(mean_prediction, np.exp(predictions), rtol=1e-6)
    np.testing.assert_allclose(regressor.predict_response(X), mean_prediction)
    assert np.all(mean_prediction > 0.0)

    booster = regressor.get_booster()
    np.testing.assert_array_equal(booster.predict_raw(X), predictions)
    np.testing.assert_allclose(booster.predict_mean(X), mean_prediction)
    np.testing.assert_allclose(booster.predict_response(X), mean_prediction)

    with pytest.raises(ValueError, match="Gamma, Poisson, and Tweedie"):
        ctboost.CTBoostRegressor(iterations=1).fit(X, y).predict_mean(X)

def test_gamma_objective_is_validated_before_native_training():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([1.0, 2.0, 0.0, 3.0, 4.0, 5.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Gamma objective requires finite positive labels"):
        ctboost.train(
            ctboost.Pool(X, y),
            {"objective": "reg:gamma", "max_depth": 1, "alpha": 1.0},
            num_boost_round=2,
        )

def test_callable_metric_requires_explicit_direction_for_early_stopping():
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=81,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    with pytest.raises(ValueError, match="primary eval metric must declare higher_is_better"):
        ctboost.train(
            ctboost.Pool(X[:100], y[:100]),
            {
                "objective": "Logloss",
                "learning_rate": 0.2,
                "max_depth": 2,
                "alpha": 1.0,
                "lambda_l2": 1.0,
                "eval_metric": [_signed_accuracy_metric],
            },
            num_boost_round=20,
            eval_set=[(X[100:], y[100:])],
            early_stopping_rounds=5,
        )


@pytest.mark.parametrize("objective", ["Cox", "SurvivalExponential"])
def test_survival_objectives_train_and_report_metrics(objective):
    rng = np.random.default_rng(73)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    linear = 0.6 * X[:, 0] - 0.35 * X[:, 1]
    base_time = np.exp(1.5 - linear)
    censor_threshold = np.quantile(base_time, 0.65)
    observed = base_time <= censor_threshold
    signed_time = np.where(observed, base_time, -np.minimum(base_time, censor_threshold)).astype(np.float32)

    booster = ctboost.train(
        ctboost.Pool(X, signed_time),
        {
            "objective": objective,
            "eval_metric": "CIndex",
            "learning_rate": 0.15,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=16,
    )

    predictions = booster.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert np.all(np.isfinite(predictions))
    assert len(booster.loss_history) == 16
