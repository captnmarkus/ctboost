import numpy as np
import pytest

import ctboost


def _data():
    X = np.arange(16, dtype=np.float32).reshape(8, 2)
    return X, ctboost.Pool(X)


def test_native_weighted_rmse_base_score_and_manual_override():
    X, prediction_pool = _data()
    labels = np.asarray([1, 2, 3, 4, 10, 11, 12, 13], dtype=np.float32)
    weights = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)
    training_pool = ctboost.Pool(X, labels, weight=weights)

    native = ctboost._core.GradientBooster(iterations=1, alpha=1.0)
    native.fit(training_pool._handle)
    expected = float(np.average(labels, weights=weights))
    assert native.base_score() == pytest.approx([expected])
    np.testing.assert_allclose(
        native.predict(prediction_pool._handle, 0), expected, rtol=0.0, atol=1e-6
    )

    legacy = ctboost._core.GradientBooster(
        iterations=1, alpha=1.0, boost_from_average=False
    )
    legacy.fit(training_pool._handle)
    assert legacy.base_score() == [0.0]

    manual = ctboost._core.GradientBooster(
        iterations=1, alpha=1.0, base_score=[7.25]
    )
    manual.fit(training_pool._handle)
    assert manual.configured_base_score() == [7.25]
    assert manual.base_score() == [7.25]


def test_objective_specific_weighted_initializers():
    X, _ = _data()
    weights = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float32)
    regression_labels = np.asarray([1, 2, 3, 4, 10, 11, 12, 13], dtype=np.float32)

    mae = ctboost.train(
        X, {"objective": "MAE"}, label=regression_labels, weight=weights, num_boost_round=1
    )
    assert mae.base_score == pytest.approx([10.0])

    quantile = ctboost.train(
        X,
        {"objective": "Quantile", "quantile_alpha": 0.75},
        label=regression_labels,
        weight=weights,
        num_boost_round=1,
    )
    assert quantile.base_score == pytest.approx([12.0])

    huber_labels = np.asarray([0, 0, 0, 0, 0, 0, 0, 10], dtype=np.float32)
    huber = ctboost.train(
        X,
        {"objective": "Huber", "huber_delta": 1.0},
        label=huber_labels,
        num_boost_round=1,
    )
    assert huber.base_score == pytest.approx([1.0 / 7.0])

    count_labels = np.asarray([1, 1, 2, 2, 4, 4, 8, 8], dtype=np.float32)
    for objective in ("Poisson", "Tweedie", "Gamma"):
        model = ctboost.train(
            X,
            {"objective": objective},
            label=count_labels,
            weight=weights,
            num_boost_round=1,
        )
        assert model.base_score == pytest.approx(
            [np.log(np.average(count_labels, weights=weights))]
        )

    for objective in ("Poisson", "Tweedie"):
        all_zero = ctboost.train(
            X,
            {"objective": objective},
            label=np.zeros(X.shape[0], dtype=np.float32),
            num_boost_round=1,
        )
        assert all_zero.base_score == pytest.approx([np.log(1.0e-12)])
        base_only_raw = np.asarray(
            all_zero._handle.predict(ctboost.Pool(X)._handle, 0), dtype=np.float64
        )
        np.testing.assert_allclose(
            np.exp(base_only_raw), 1.0e-12, rtol=1e-5
        )


def test_binary_and_multiclass_prior_logits_honor_weights():
    X, _ = _data()
    binary_labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)
    binary_weights = np.asarray([1, 1, 1, 1, 1, 1, 3, 3], dtype=np.float32)
    binary = ctboost.train(
        X,
        {"objective": "Logloss"},
        label=binary_labels,
        weight=binary_weights,
        num_boost_round=1,
    )
    assert binary.base_score == pytest.approx([0.0])

    labels = np.asarray([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.float32)
    weights = np.asarray([1, 1, 1, 1, 1, 1, 3, 3], dtype=np.float32)
    multiclass = ctboost.train(
        X,
        {"objective": "MultiClass", "num_classes": 3},
        label=labels,
        weight=weights,
        num_boost_round=1,
    )
    probabilities = np.asarray([4.0, 2.0, 6.0]) / 12.0
    expected = np.log(probabilities)
    expected -= expected.mean()
    np.testing.assert_allclose(multiclass.base_score, expected, rtol=1e-7, atol=1e-7)


def test_multiclass_base_score_dimension_is_validated():
    with pytest.raises(ValueError, match="one raw margin or one margin per"):
        ctboost._core.GradientBooster(
            objective="MultiClass", num_classes=3, base_score=[0.0, 1.0]
        )


def test_base_score_round_trip_contributions_and_legacy_state():
    X, prediction_pool = _data()
    labels = np.linspace(2.0, 9.0, X.shape[0], dtype=np.float32)
    model = ctboost.train(X, {"objective": "RMSE"}, label=labels, num_boost_round=3)
    state = dict(model._handle.export_state())
    assert state["boost_from_average"] is True
    assert state["base_score"] == pytest.approx([float(labels.mean())])

    restored = ctboost._core.GradientBooster.from_state(state)
    np.testing.assert_allclose(
        restored.predict(prediction_pool._handle), model.predict(X), rtol=0.0, atol=1e-6
    )
    contributions = model.predict_contrib(X)
    np.testing.assert_allclose(contributions.sum(axis=1), model.predict(X), atol=1e-5)

    for key in ("boost_from_average", "configured_base_score", "base_score"):
        state.pop(key)
    legacy = ctboost._core.GradientBooster.from_state(state)
    assert legacy.boost_from_average() is False
    assert legacy.base_score() == [0.0]
    np.testing.assert_allclose(legacy.predict(prediction_pool._handle, 0), 0.0)


def test_nonzero_base_score_eval_history_matches_returned_early_stopped_model():
    rng = np.random.default_rng(1927)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    labels = (7.0 + 1.4 * X[:, 0] - 0.6 * X[:, 1]).astype(np.float32)
    model = ctboost.train(
        X[:80],
        {
            "objective": "RMSE",
            "learning_rate": 0.17,
            "max_depth": 2,
            "alpha": 1.0,
            "random_seed": 31,
        },
        label=labels[:80],
        num_boost_round=20,
        eval_set=(X[80:], labels[80:]),
        early_stopping_rounds=3,
    )

    assert model.base_score[0] == pytest.approx(float(labels[:80].mean()))
    prediction = model.predict(X[80:]).astype(np.float64)
    returned_rmse = float(
        np.sqrt(np.mean(np.square(prediction - labels[80:].astype(np.float64))))
    )
    assert model.eval_loss_history[-1] == pytest.approx(returned_rmse, abs=1e-7)

    restored = ctboost._core.GradientBooster.from_state(model._handle.export_state())
    np.testing.assert_array_equal(
        restored.predict(ctboost.Pool(X[80:])._handle), model.predict(X[80:])
    )


def test_automatic_initialization_is_conservative_for_baselines_and_custom_objectives():
    X, _ = _data()
    labels = np.linspace(1.0, 8.0, X.shape[0], dtype=np.float32)
    with_baseline = ctboost.train(
        ctboost.Pool(X, labels, baseline=np.full(X.shape[0], 2.0, dtype=np.float32)),
        {"objective": "RMSE"},
        num_boost_round=1,
    )
    assert with_baseline.base_score == [0.0]

    def squared_error(prediction, label):
        return prediction - label, np.ones_like(prediction)

    custom = ctboost.train(
        X,
        {"objective": ctboost.make_objective(squared_error, native_objective="RMSE")},
        label=labels,
        num_boost_round=1,
    )
    assert custom.base_score == [0.0]


def test_sklearn_surface_exposes_base_score_controls():
    model = ctboost.CTBoostRegressor(
        iterations=2, boost_from_average=False, base_score=3.5
    )
    params = model.get_params(deep=False)
    assert params["boost_from_average"] is False
    assert params["base_score"] == 3.5
