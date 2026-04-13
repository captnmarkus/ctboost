from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification, make_regression

import ctboost
import ctboost._core as _core


def test_booster_save_load_and_staged_predict_round_trip(tmp_path: Path):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=41,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    train_pool = ctboost.Pool(X[:120], y[:120])
    valid_pool = ctboost.Pool(X[120:], y[120:])
    full_pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        train_pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.15,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=12,
        eval_set=valid_pool,
    )

    staged_predictions = list(booster.staged_predict(full_pool))

    assert len(staged_predictions) == booster.num_iterations_trained
    np.testing.assert_allclose(staged_predictions[-1], booster.predict(full_pool), rtol=1e-6, atol=1e-6)
    assert booster.eval_loss_history

    model_path = tmp_path / "booster.pkl"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)

    np.testing.assert_allclose(restored.predict(full_pool), booster.predict(full_pool), rtol=1e-6, atol=1e-6)
    assert restored.loss_history == booster.loss_history
    assert restored.eval_loss_history == booster.eval_loss_history


def test_booster_json_save_load_round_trip(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=29,
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
        num_boost_round=8,
    )

    model_path = tmp_path / "booster.json"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)

    np.testing.assert_allclose(restored.predict(pool), booster.predict(pool), rtol=1e-6, atol=1e-6)
    assert restored.objective_name == booster.objective_name
    assert restored.eval_metric_name == booster.eval_metric_name


def test_booster_state_stores_quantization_schema_once_and_loads_legacy_tree_schema():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=13,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=4,
    )

    state = dict(booster._handle.export_state())
    assert "quantization_schema" in state
    for tree_state in state["trees"]:
        assert "num_bins_per_feature" not in tree_state
        assert "cut_offsets" not in tree_state
        assert "cut_values" not in tree_state
        assert "categorical_mask" not in tree_state
        assert "missing_value_mask" not in tree_state
        assert "nan_mode" not in tree_state

    legacy_state = dict(state)
    quantization_schema = dict(legacy_state.pop("quantization_schema"))
    legacy_tree_states = []
    for tree_state in legacy_state["trees"]:
        upgraded_tree_state = dict(tree_state)
        upgraded_tree_state.update(quantization_schema)
        legacy_tree_states.append(upgraded_tree_state)
    legacy_state["trees"] = legacy_tree_states

    restored = ctboost.Booster(_core.GradientBooster.from_state(legacy_state))
    np.testing.assert_allclose(restored.predict(pool), booster.predict(pool), rtol=1e-6, atol=1e-6)


def test_cv_returns_fold_aggregates():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=19,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    results = ctboost.cv(
        ctboost.Pool(X, y),
        {
            "objective": "Logloss",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=10,
        nfold=3,
        random_state=7,
    )

    expected_keys = {
        "iterations",
        "train_loss_mean",
        "train_loss_std",
        "valid_loss_mean",
        "valid_loss_std",
        "best_iteration_mean",
        "best_iteration_std",
    }
    assert expected_keys == set(results)
    assert results["iterations"].shape == results["train_loss_mean"].shape
    assert results["iterations"].shape == results["valid_loss_mean"].shape
    assert np.all(np.isfinite(results["train_loss_mean"]))
    assert np.all(np.isfinite(results["valid_loss_mean"]))
    assert np.isfinite(results["best_iteration_mean"])


def test_classifier_save_load_and_staged_predict_proba(tmp_path: Path):
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=23,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=14,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    staged_probabilities = list(clf.staged_predict_proba(X))

    assert len(staged_probabilities) == clf._booster.num_iterations_trained
    np.testing.assert_allclose(staged_probabilities[-1], clf.predict_proba(X), rtol=1e-6, atol=1e-6)

    model_path = tmp_path / "classifier.pkl"
    clf.save_model(model_path)
    restored = ctboost.CTBoostClassifier.load_model(model_path)

    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))


def test_classifier_json_save_load_round_trip(tmp_path: Path):
    X, y = make_classification(
        n_samples=128,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    model_path = tmp_path / "classifier.json"
    clf.save_model(model_path)
    restored = ctboost.CTBoostClassifier.load_model(model_path)

    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X), rtol=1e-6, atol=1e-6)
