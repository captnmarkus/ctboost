import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import balanced_accuracy_score
import ctboost


@pytest.mark.parametrize("stop_mode", ["callback", "early_stopping"])
def test_stopped_training_does_not_request_unused_learning_rate(stop_mode):
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    pool = ctboost.Pool(X, X[:, 0])
    requested_iterations = []
    stopping_iteration = 0 if stop_mode == "callback" else 1

    def schedule(iteration):
        requested_iterations.append(iteration)
        if iteration > stopping_iteration:
            raise AssertionError("schedule called after training stopped")
        return 0.1

    kwargs = (
        {"callbacks": [lambda env: True]}
        if stop_mode == "callback"
        else {"eval_set": pool, "early_stopping_rounds": 1}
    )
    params = {"objective": "RMSE", "alpha": 1.0, "max_depth": 2}
    if stop_mode == "early_stopping":
        params["eval_metric"] = ctboost.make_eval_metric(
            lambda prediction, label: 1.0,
            name="constant",
            higher_is_better=False,
            allow_early_stopping=True,
        )

    model = ctboost.train(
        pool, params, num_boost_round=5, learning_rate_schedule=schedule, **kwargs
    )

    assert requested_iterations == list(range(stopping_iteration + 1))
    assert model.learning_rate == pytest.approx(0.1)
    assert model.num_iterations_trained == 1


@pytest.mark.parametrize("multi_strategy", ["one_output_per_tree", "multi_output_tree"])
def test_python_dart_early_stopping_restores_weights_from_best_round(multi_strategy):
    rng = np.random.default_rng(173)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    y = np.argmax(X[:, :3], axis=1).astype(np.float32)
    pool = ctboost.Pool(X, y)
    predictions_by_round = []

    def record_predictions(env):
        predictions_by_round.append(env.model.predict(pool).copy())

    constant_metric = ctboost.make_eval_metric(
        lambda prediction, label: 1.0,
        name="constant",
        higher_is_better=False,
        allow_early_stopping=True,
    )
    model = ctboost.train(
        pool,
        {
            "objective": "MultiClass",
            "num_classes": 3,
            "multi_strategy": multi_strategy,
            "boosting_type": "DART",
            "drop_rate": 1.0,
            "skip_drop": 0.0,
            "learning_rate": 0.2,
            "alpha": 1.0,
            "max_depth": 2,
            "eval_metric": constant_metric,
        },
        num_boost_round=8,
        eval_set=pool,
        early_stopping_rounds=2,
        callbacks=[record_predictions],
    )

    assert len(predictions_by_round) == 3
    assert model.best_iteration == 0
    assert model.num_iterations_trained == 1
    np.testing.assert_array_equal(model.predict(pool), predictions_by_round[0])

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
        init_model=stage_one,
    )
    manual = ctboost.train(
        pool,
        {**base_params, "learning_rate": 0.05},
        num_boost_round=3,
        init_model=stage_two,
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
        init_model=manual_first,
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
