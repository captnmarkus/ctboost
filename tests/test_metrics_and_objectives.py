import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import balanced_accuracy_score

import ctboost


def test_eval_metric_auc_is_used_for_validation_history_and_early_stopping():
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=61,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=200,
        learning_rate=0.25,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        eval_metric="AUC",
    )
    clf.fit(
        X[:120],
        y[:120],
        eval_set=[(X[120:], 1.0 - y[120:])],
        early_stopping_rounds=12,
    )

    assert clf.best_iteration_ + 1 < 200
    assert "AUC" in clf.evals_result_["validation"]
    auc_history = np.asarray(clf.evals_result_["validation"]["AUC"], dtype=np.float32)
    assert auc_history.shape[0] == clf.best_iteration_ + 1
    assert np.all(np.isfinite(auc_history))


def test_balanced_accuracy_is_available_for_early_stopping_and_matches_predictions():
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=63,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y_eval = 1.0 - y[120:]

    clf = ctboost.CTBoostClassifier(
        iterations=200,
        learning_rate=0.25,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        eval_metric="BalancedAccuracy",
    )
    clf.fit(
        X[:120],
        y[:120],
        eval_set=[(X[120:], y_eval)],
        early_stopping_rounds=12,
    )

    assert clf.best_iteration_ + 1 < 200
    history = np.asarray(clf.evals_result_["validation"]["BalancedAccuracy"], dtype=np.float32)
    assert history.shape[0] == clf.best_iteration_ + 1
    prediction_score = balanced_accuracy_score(y_eval, clf.predict(X[120:]))
    np.testing.assert_allclose(history[-1], prediction_score, rtol=1e-6, atol=1e-6)


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


@pytest.mark.parametrize(
    "eval_metric",
    ["Accuracy", "BalancedAccuracy", "Precision", "Recall", "F1", "AUC"],
)
def test_binary_eval_metrics_are_exposed(eval_metric):
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=71,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=18,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        eval_metric=eval_metric,
    )
    clf.fit(X[:140], y[:140], eval_set=[(X[140:], y[140:])])

    metric_history = np.asarray(clf.evals_result_["validation"][eval_metric], dtype=np.float32)
    assert metric_history.shape[0] == len(clf._booster.eval_loss_history)
    assert np.all(np.isfinite(metric_history))


def test_train_supports_multi_eval_sets_metrics_callbacks_and_custom_eval_names(tmp_path):
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=75,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    checkpoint_path = tmp_path / "multi_eval_checkpoint.json"
    callback_iterations = []

    def record_iteration(env):
        callback_iterations.append(env.iteration)
        return False

    booster = ctboost.train(
        ctboost.Pool(X[:120], y[:120]),
        {
            "objective": "Logloss",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "eval_metric": ["BalancedAccuracy", "AUC"],
        },
        num_boost_round=20,
        eval_set=[(X[120:180], y[120:180]), (X[180:], y[180:])],
        eval_names=["public", "private"],
        early_stopping_rounds=5,
        early_stopping_metric="BalancedAccuracy",
        early_stopping_name="public",
        callbacks=[
            record_iteration,
            ctboost.checkpoint_callback(checkpoint_path, interval=2),
        ],
    )

    assert checkpoint_path.exists()
    assert callback_iterations
    assert sorted(booster.evals_result_.keys()) == ["learn", "private", "public"]
    assert sorted(booster.evals_result_["public"].keys()) == ["AUC", "BalancedAccuracy"]
    np.testing.assert_allclose(
        np.asarray(booster.eval_loss_history, dtype=np.float32),
        np.asarray(booster.evals_result_["public"]["BalancedAccuracy"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert booster.eval_metric_name == "BalancedAccuracy"

    model_path = tmp_path / "multi_eval_model.json"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)
    assert sorted(restored.evals_result_.keys()) == ["learn", "private", "public"]
    np.testing.assert_allclose(
        np.asarray(restored.evals_result_["public"]["BalancedAccuracy"], dtype=np.float32),
        np.asarray(booster.evals_result_["public"]["BalancedAccuracy"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
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
