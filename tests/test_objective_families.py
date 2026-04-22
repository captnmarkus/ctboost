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
