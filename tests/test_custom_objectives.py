from pathlib import Path
import pickle

import numpy as np
import pytest

import ctboost


def _squared_error_objective(predictions, label, **_kwargs):
    return predictions - label, np.ones_like(predictions, dtype=np.float32)


def _binary_logloss_objective(predictions, label, **_kwargs):
    probabilities = np.empty_like(predictions, dtype=np.float32)
    nonnegative = predictions >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-predictions[nonnegative]))
    exp_predictions = np.exp(predictions[~nonnegative])
    probabilities[~nonnegative] = exp_predictions / (1.0 + exp_predictions)
    return probabilities - label, probabilities * (1.0 - probabilities)


def _multiclass_logloss_objective(predictions, label, **_kwargs):
    shifted = predictions - np.max(predictions, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    gradients = probabilities.copy()
    gradients[np.arange(label.shape[0]), label.astype(np.int64)] -= 1.0
    hessians = np.maximum(probabilities * (1.0 - probabilities), 1e-6)
    return gradients, hessians


def _regression_data():
    rng = np.random.default_rng(919)
    X = rng.normal(size=(160, 6)).astype(np.float32)
    y = (1.7 * X[:, 0] - 0.8 * X[:, 1] + 0.25 * X[:, 2]).astype(np.float32)
    return X, y


def test_callable_objective_matches_native_derivatives_and_receives_metadata():
    X, y = _regression_data()
    pool = ctboost.Pool(X, y, weight=np.linspace(0.5, 1.5, X.shape[0], dtype=np.float32))
    observed = []

    def objective(predictions, label, *, weight, num_classes, params):
        observed.append((predictions.shape, label.shape, weight.copy(), num_classes, params["max_depth"]))
        return predictions - label, np.ones_like(predictions)

    common_params = {
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "subsample": 1.0,
        "random_seed": 41,
    }
    expected = ctboost.train(
        pool,
        {**common_params, "objective": "RMSE"},
        num_boost_round=8,
    )
    actual = ctboost.train(
        pool,
        {
            **common_params,
            "objective": ctboost.make_objective(
                objective,
                name="WeightedSquaredError",
                native_objective="RMSE",
            ),
        },
        num_boost_round=8,
    )

    np.testing.assert_allclose(actual.predict(X), expected.predict(X), rtol=1e-6, atol=1e-6)
    assert len(observed) == 8
    assert observed[0][0] == (X.shape[0],)
    assert observed[0][1] == (X.shape[0],)
    np.testing.assert_array_equal(observed[0][2], pool.weight)
    assert observed[0][3:] == (1, 3)
    assert actual.objective_name == "WeightedSquaredError"
    assert actual.native_objective_name == "RMSE"


def test_train_obj_keyword_uses_configured_native_objective():
    X, y = _regression_data()
    booster = ctboost.train(
        X,
        {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
        label=y,
        num_boost_round=4,
        obj=_squared_error_objective,
    )

    assert booster.num_iterations_trained == 4
    assert booster.objective_name == "_squared_error_objective"
    assert booster.native_objective_name == "RMSE"


def test_custom_objective_and_metric_support_early_stopping_together():
    X, y = _regression_data()
    metric = ctboost.make_eval_metric(
        lambda predictions, label: float(np.mean(np.abs(predictions - label))),
        name="CustomMAE",
        higher_is_better=False,
        allow_early_stopping=True,
    )
    booster = ctboost.train(
        X[:100],
        {
            "objective": _squared_error_objective,
            "eval_metric": metric,
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
        },
        label=y[:100],
        num_boost_round=30,
        eval_set=(X[100:], y[100:]),
        early_stopping_rounds=5,
    )

    history = booster.evals_result_["validation"]["CustomMAE"]
    assert len(history) == booster.num_iterations_trained
    assert np.all(np.isfinite(history))
    assert booster.eval_metric_name == "CustomMAE"


@pytest.mark.parametrize(
    ("objective", "message"),
    [
        (lambda predictions, label: predictions - label, r"must return a \(gradients, hessians\) tuple"),
        (
            lambda predictions, label: (predictions[:-1], np.ones_like(predictions)),
            "gradients must have exactly the same shape",
        ),
        (
            lambda predictions, label: (np.full_like(predictions, np.nan), np.ones_like(predictions)),
            "gradients must contain only finite values",
        ),
        (
            lambda predictions, label: (predictions - label, -np.ones_like(predictions)),
            "hessians must be non-negative",
        ),
    ],
)
def test_custom_objective_outputs_are_strictly_validated(objective, message):
    X, y = _regression_data()
    with pytest.raises((TypeError, ValueError), match=message):
        ctboost.train(
            X,
            {"objective": objective, "max_depth": 1, "alpha": 1.0},
            label=y,
            num_boost_round=1,
        )


def test_custom_multiclass_objective_receives_matrix_predictions():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    scores = np.column_stack(
        [X[:, 0] - X[:, 1], X[:, 1] + X[:, 2], -X[:, 0] - X[:, 2]]
    )
    y = np.argmax(scores, axis=1)
    seen_shapes = []

    def objective(predictions, label, **kwargs):
        seen_shapes.append((predictions.shape, label.shape, kwargs["num_classes"]))
        return _multiclass_logloss_objective(predictions, label)

    model = ctboost.CTBoostClassifier(
        iterations=5,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        loss_function=objective,
    )
    model.fit(X, y)

    assert seen_shapes == [((180, 3), (180,), 3)] * 5
    assert model.get_booster().native_objective_name == "MultiClass"
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (180, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)


def test_custom_binary_classifier_uses_logloss_inference_semantics():
    rng = np.random.default_rng(111)
    X = rng.normal(size=(140, 4)).astype(np.float32)
    y = (X[:, 0] - 0.4 * X[:, 1] > 0.0).astype(np.int64)
    model = ctboost.CTBoostClassifier(
        iterations=6,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        loss_function=_binary_logloss_objective,
    ).fit(X, y)

    assert model.get_booster().native_objective_name == "Logloss"
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (140, 2)
    assert np.all(np.isfinite(probabilities))


def test_custom_objective_booster_persists_for_inference_but_requires_callable_to_continue(
    tmp_path: Path,
):
    X, y = _regression_data()
    objective = ctboost.make_objective(
        _squared_error_objective,
        name="PersistentSquaredError",
    )
    booster = ctboost.train(
        X,
        {"objective": objective, "max_depth": 2, "alpha": 1.0},
        label=y,
        num_boost_round=4,
    )
    path = tmp_path / "custom-objective.ctboost"
    booster.save_model(path)
    restored = ctboost.load_model(path)

    np.testing.assert_allclose(restored.predict(X), booster.predict(X), rtol=0.0, atol=0.0)
    assert restored.objective_name == "PersistentSquaredError"
    assert restored.native_objective_name == "RMSE"
    with pytest.raises(ValueError, match="pass that callable objective again"):
        ctboost.train(
            X,
            {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
            label=y,
            num_boost_round=1,
            init_model=restored,
        )


def test_custom_objective_estimator_json_is_inference_only(tmp_path: Path):
    X, y = _regression_data()
    model = ctboost.CTBoostRegressor(
        iterations=4,
        max_depth=2,
        alpha=1.0,
        loss_function=ctboost.make_objective(
            _squared_error_objective,
            name="EstimatorSquaredError",
        ),
    ).fit(X, y)
    path = tmp_path / "custom-estimator.ctboost"
    model.save_model(path)
    restored = ctboost.CTBoostRegressor.load_model(path)

    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=0.0, atol=0.0)
    assert restored.loss_function == "RMSE"
    assert restored.get_booster().objective_name == "EstimatorSquaredError"


def test_custom_objective_estimator_pickle_does_not_require_picklable_callables():
    X, y = _regression_data()
    model = ctboost.CTBoostRegressor(
        iterations=3,
        max_depth=2,
        alpha=1.0,
        loss_function=lambda predictions, label: (
            predictions - label,
            np.ones_like(predictions),
        ),
        eval_metric=ctboost.make_eval_metric(
            lambda predictions, label: float(np.mean(np.abs(predictions - label))),
            name="LambdaMAE",
            higher_is_better=False,
        ),
    ).fit(X, y)
    restored = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=0.0, atol=0.0)
    assert restored.loss_function == model.get_booster().native_objective_name
    assert restored.eval_metric is None
    assert restored.get_booster().objective_name == "<lambda>"


def test_custom_metric_rejects_non_finite_results():
    X, y = _regression_data()
    metric = ctboost.make_eval_metric(
        lambda _predictions, _label: np.nan,
        name="BrokenMetric",
        higher_is_better=False,
    )
    with pytest.raises(ValueError, match="finite scalar value"):
        ctboost.train(
            X[:100],
            {"objective": _squared_error_objective, "eval_metric": metric, "alpha": 1.0},
            label=y[:100],
            num_boost_round=1,
            eval_set=(X[100:], y[100:]),
        )
