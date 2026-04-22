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

def test_train_supports_callable_eval_metrics_and_custom_early_stopping():
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=79,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    y_eval = 1.0 - y[120:]

    metric = ctboost.make_eval_metric(
        _signed_accuracy_metric,
        name="SignedAccuracy",
        higher_is_better=True,
        allow_early_stopping=True,
    )

    booster = ctboost.train(
        ctboost.Pool(X[:120], y[:120]),
        {
            "objective": "Logloss",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "eval_metric": [metric, "AUC"],
        },
        num_boost_round=60,
        eval_set=[(X[120:], y_eval)],
        early_stopping_rounds=6,
        early_stopping_metric=metric,
    )

    assert "SignedAccuracy" in booster.evals_result_["validation"]
    history = np.asarray(booster.evals_result_["validation"]["SignedAccuracy"], dtype=np.float32)
    assert history.shape[0] == booster.best_iteration + 1
    expected_metric = _signed_accuracy_metric(booster.predict(X[120:]), y_eval)
    np.testing.assert_allclose(history[-1], expected_metric, rtol=1e-6, atol=1e-6)

def test_classifier_accepts_callable_eval_metric():
    X, y = make_classification(
        n_samples=220,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=83,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    metric = ctboost.make_eval_metric(
        _signed_accuracy_metric,
        name="SignedAccuracy",
        higher_is_better=True,
        allow_early_stopping=True,
    )
    clf = ctboost.CTBoostClassifier(
        iterations=40,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        eval_metric=metric,
    )
    clf.fit(
        X[:120],
        y[:120],
        eval_set=[(X[120:], y[120:])],
        early_stopping_rounds=6,
    )

    assert "SignedAccuracy" in clf.evals_result_["validation"]
