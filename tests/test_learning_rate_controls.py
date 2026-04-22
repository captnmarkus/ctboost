import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import balanced_accuracy_score
import ctboost
import ctboost._core as _core

def _rebase_learning_rate(booster, learning_rate):
    state = dict(booster._handle.export_state())
    current_learning_rate = float(booster.learning_rate)
    resolved_learning_rate = float(learning_rate)
    scale = current_learning_rate / resolved_learning_rate
    for tree_state in state.get("trees", []):
        for node in tree_state.get("nodes", []):
            if bool(node.get("is_leaf", False)):
                node["leaf_weight"] = float(node["leaf_weight"]) * scale
    state["learning_rate"] = resolved_learning_rate
    return ctboost.Booster(_core.GradientBooster.from_state(state))

def test_learning_rate_schedule_matches_manual_warm_start_and_export(tmp_path):
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=91,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    schedule = [0.25, 0.25, 0.25, 0.1, 0.1, 0.05, 0.05, 0.05]
    pool = ctboost.Pool(X, y)

    base_params = {
        "objective": "RMSE",
        "learning_rate": schedule[0],
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 31,
    }

    stage_one = ctboost.train(pool, base_params, num_boost_round=3)
    stage_two = ctboost.train(
        pool,
        {**base_params, "learning_rate": 0.1},
        num_boost_round=2,
        init_model=_rebase_learning_rate(stage_one, 0.1),
    )
    manual = ctboost.train(
        pool,
        {**base_params, "learning_rate": 0.05},
        num_boost_round=3,
        init_model=_rebase_learning_rate(stage_two, 0.05),
    )
    scheduled = ctboost.train(
        pool,
        base_params,
        num_boost_round=8,
        learning_rate_schedule=schedule,
    )

    np.testing.assert_allclose(scheduled.predict(X), manual.predict(X), rtol=1e-6, atol=1e-6)
    assert scheduled.learning_rate_history == pytest.approx(schedule)

    export_path = tmp_path / "scheduled_predictor.json"
    scheduled.export_model(export_path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(export_path)
    np.testing.assert_allclose(
        np.asarray(predictor.predict(X[:24]), dtype=np.float32),
        scheduled.predict(X[:24]),
        rtol=1e-6,
        atol=1e-6,
    )

def test_callback_can_change_learning_rate_for_subsequent_iterations():
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=93,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    pool = ctboost.Pool(X, y)
    callback_rates = []

    def decay_after_first_round(env):
        callback_rates.append(env.learning_rate)
        if env.iteration == 0:
            env.model.set_learning_rate(0.05)
        return False

    params = {
        "objective": "RMSE",
        "learning_rate": 0.25,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 37,
    }
    callback_booster = ctboost.train(
        pool,
        params,
        num_boost_round=6,
        callbacks=[decay_after_first_round],
    )
    manual_first = ctboost.train(pool, params, num_boost_round=1)
    manual = ctboost.train(
        pool,
        {**params, "learning_rate": 0.05},
        num_boost_round=5,
        init_model=_rebase_learning_rate(manual_first, 0.05),
    )

    np.testing.assert_allclose(
        callback_booster.predict(X),
        manual.predict(X),
        rtol=1e-6,
        atol=1e-6,
    )
    assert callback_rates == pytest.approx([0.25, 0.05, 0.05, 0.05, 0.05, 0.05])

def test_set_learning_rate_preserves_existing_predictions():
    X, y = make_regression(
        n_samples=160,
        n_features=6,
        n_informative=4,
        noise=0.2,
        random_state=95,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    booster = ctboost.train(
        ctboost.Pool(X, y),
        {
            "objective": "RMSE",
            "learning_rate": 0.25,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 41,
        },
        num_boost_round=4,
    )

    baseline_prediction = booster.predict(X)
    booster.set_learning_rate(0.05)

    np.testing.assert_allclose(
        booster.predict(X),
        baseline_prediction,
        rtol=1e-6,
        atol=1e-6,
    )
