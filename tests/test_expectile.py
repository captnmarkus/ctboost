import numpy as np
import pytest

import ctboost


@pytest.mark.parametrize("tau", [0, 1, -0.1, float("nan"), float("inf")])
def test_requires_interior_expectile_level(tau):
    with pytest.raises(ValueError, match="tau"):
        ctboost.make_expectile_loss(tau)


def test_expectile_loss_derivatives_weights_and_direction():
    objective, metric = ctboost.make_expectile_loss(0.75)
    label = np.array([1.0, 1.0, 1.0])
    predictions = np.array([-1.0, 2.0, 1.0])
    gradient, hessian = objective.func(predictions, label)
    np.testing.assert_allclose(gradient, [-3.0, 0.5, 0.0])
    np.testing.assert_allclose(hessian, [1.5, 0.5, 0.5])
    assert metric.func(predictions, label, weight=[1, 2, 1]) == pytest.approx(0.875)
    assert metric.func(predictions, label, weight=[0, 0, 0]) == 0
    assert metric.higher_is_better is False and metric.allow_early_stopping
    with pytest.raises(ValueError, match="scalar prediction per label"):
        objective.func(predictions[:, None], label)


def test_central_expectile_matches_native_squared_error_and_persists(tmp_path):
    rng = np.random.default_rng(22)
    X = rng.normal(size=(160, 3)).astype("float32")
    y = np.asarray(2 * X[:, 0] + X[:, 1] ** 2, dtype="float32")
    weights = rng.choice([0.0, 0.5, 2.0], size=len(X))
    pool = ctboost.Pool(X, y, weight=weights)
    objective, metric = ctboost.make_expectile_loss(0.5)
    common = {"max_depth": 2, "alpha": 1.0, "base_score": 0.7}
    model = ctboost.train(
        pool,
        {**common, "objective": objective, "eval_metric": metric},
        num_boost_round=8,
        eval_set=pool,
    )
    baseline = ctboost.train(pool, {**common, "objective": "RMSE"}, num_boost_round=8)
    expected = baseline.predict(X)
    np.testing.assert_allclose(model.predict(X), expected, atol=1e-6)
    assert model.evals_result_["validation"][metric.name][-1] == pytest.approx(
        0.5 * np.average((expected - y) ** 2, weights=weights), rel=1e-6
    )
    path = tmp_path / "expectile.json"
    model.save_model(path)
    restored = ctboost.load_model(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    other, _ = ctboost.make_expectile_loss(0.8)
    with pytest.raises(ValueError, match="objective name must match"):
        ctboost.train(
            pool, {**common, "objective": other}, init_model=restored, num_boost_round=1
        )
    continued = ctboost.train(
        pool,
        {**common, "objective": objective, "eval_metric": metric},
        init_model=restored,
        num_boost_round=1,
    )
    assert continued.num_iterations_trained == 9


def test_asymmetric_expectile_moves_toward_costlier_tail():
    X = np.zeros((100, 1), dtype=np.float32)
    y = np.tile([0.0, 0.0, 0.0, 1.0], 25)
    objective, metric = ctboost.make_expectile_loss(0.8)
    model = ctboost.CTBoostRegressor(
        loss_function=objective,
        eval_metric=metric,
        iterations=30,
        learning_rate=0.4,
        lambda_l2=0.0,
    )
    model.fit(X, y, eval_set=(X, y))
    # Three observations at zero and one at one give a population expectile of 4/7.
    np.testing.assert_allclose(model.predict(X), 4 / 7, atol=1e-5)
