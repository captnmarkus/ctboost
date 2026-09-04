import numpy as np
import pytest

import ctboost


def _multiclass_logloss(raw_predictions, labels):
    shifted = raw_predictions - np.max(raw_predictions, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    target_probabilities = probabilities[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(target_probabilities, 1e-12, 1.0))))


@pytest.mark.parametrize("objective,strategy", [
    ("RMSE", "one_output_per_tree"),
    ("MultiClass", "one_output_per_tree"),
    ("MultiClass", "multi_output_tree"),
])
def test_dart_dropping_all_trees_uses_exact_base_score_and_baseline(objective, strategy):
    rng = np.random.default_rng(418)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    multiclass = objective == "MultiClass"
    dimension = 3 if multiclass else 1
    labels = (
        np.argmax(X[:, :3], axis=1) if multiclass else 2 * X[:, 0] - X[:, 1]
    ).astype(np.float32)
    base_score = [0.22589735785534915, 0.12509265873338382, -0.35099001658873275] if multiclass else [0.22589735785534915]
    baseline = rng.normal(scale=0.05, size=(len(X), dimension)).astype(np.float32)
    if not multiclass:
        baseline = baseline[:, 0]
    pool = ctboost.Pool(X, labels, baseline=baseline)
    captured_predictions = []

    def objective_callback(prediction, label):
        captured_predictions.append(prediction.copy())
        gradient, hessian = ctboost._core._debug_compute_objective(
            objective, prediction.ravel(), label, num_classes=dimension
        )
        return gradient.reshape(prediction.shape), hessian.reshape(prediction.shape)

    native = ctboost._core.GradientBooster(
        objective=objective, num_classes=dimension, multi_strategy=strategy,
        iterations=4, alpha=1.0, max_depth=3, learning_rate=0.15,
        boosting_type="DART", drop_rate=1.0, skip_drop=0.0,
        base_score=base_score,
    )
    native.fit_custom_objective(pool._handle, objective_callback)

    expected = baseline + np.asarray(base_score, dtype=np.float32)
    assert len(captured_predictions) == 4
    for prediction in captured_predictions:
        np.testing.assert_array_equal(prediction, expected)


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_dart_warm_start_keeps_topology_when_all_prior_trees_are_dropped(strategy):
    rng = np.random.default_rng(418)
    X = rng.normal(size=(180, 5)).astype(np.float32)
    labels = np.argmax(
        np.column_stack([X[:, 0] - X[:, 1], X[:, 1], -X[:, 0]]), axis=1
    ).astype(np.float32)
    pool = ctboost.Pool(X, labels)
    params = {
        "objective": "MultiClass", "num_classes": 3, "multi_strategy": strategy,
        "alpha": 1.0, "max_depth": 3, "learning_rate": 0.15, "random_seed": 11,
        "boosting_type": "DART", "drop_rate": 1.0, "skip_drop": 0.0,
    }
    full = ctboost.train(pool, params, num_boost_round=4)
    partial = ctboost.train(pool, params, num_boost_round=2)
    resumed = ctboost.train(pool, {}, num_boost_round=2, init_model=partial)

    assert full._handle.export_state()["trees"] == resumed._handle.export_state()["trees"]
    np.testing.assert_array_equal(full.predict(pool), resumed.predict(pool))


def test_dart_training_history_matches_stored_single_output_model():
    rng = np.random.default_rng(123)
    data = rng.normal(size=(200, 6)).astype(np.float32)
    labels = (3.0 * data[:, 0] - 2.0 * data[:, 1]).astype(np.float32)
    booster = ctboost.train(
        ctboost.Pool(data, labels),
        {
            "objective": "RMSE",
            "boosting_type": "DART",
            "drop_rate": 1.0,
            "skip_drop": 0.0,
            "max_drop": 1,
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "random_seed": 9,
        },
        num_boost_round=5,
    )

    predictions = np.asarray(booster.predict(data), dtype=np.float64)
    actual_rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
    recorded_rmse = float(booster._handle.export_state()["loss_history"][-1])
    assert recorded_rmse == pytest.approx(actual_rmse, rel=1e-6, abs=1e-6)


def test_dart_training_history_matches_stored_multiclass_model():
    rng = np.random.default_rng(321)
    data = rng.normal(size=(240, 5)).astype(np.float32)
    labels = np.argmax(
        np.column_stack(
            [data[:, 0] - data[:, 1], data[:, 1] + data[:, 2], -data[:, 0] - data[:, 2]]
        ),
        axis=1,
    ).astype(np.float32)
    booster = ctboost.train(
        ctboost.Pool(data, labels),
        {
            "objective": "MultiClass",
            "num_classes": 3,
            "boosting_type": "DART",
            "drop_rate": 1.0,
            "skip_drop": 0.0,
            "max_drop": 1,
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "random_seed": 11,
        },
        num_boost_round=5,
    )

    raw_predictions = np.asarray(booster.predict(data), dtype=np.float64)
    actual_logloss = _multiclass_logloss(raw_predictions, labels.astype(np.int64))
    recorded_logloss = float(booster._handle.export_state()["loss_history"][-1])
    assert recorded_logloss == pytest.approx(actual_logloss, rel=1e-6, abs=1e-6)


def test_dart_early_stopping_restores_exact_best_ensemble():
    rng = np.random.default_rng(81)
    train_data = rng.normal(size=(160, 5)).astype(np.float32)
    train_labels = (2.0 * train_data[:, 0] - train_data[:, 1]).astype(np.float32)
    eval_data = rng.normal(size=(120, 5)).astype(np.float32)
    eval_labels = (-2.0 * eval_data[:, 0] + eval_data[:, 1]).astype(np.float32)
    params = {
        "objective": "RMSE",
        "boosting_type": "DART",
        "drop_rate": 1.0,
        "skip_drop": 0.0,
        "max_drop": 1,
        "learning_rate": 0.3,
        "max_depth": 2,
        "alpha": 1.0,
        "random_seed": 17,
    }
    stopped = ctboost.train(
        ctboost.Pool(train_data, train_labels),
        params,
        num_boost_round=12,
        eval_set=ctboost.Pool(eval_data, eval_labels),
        early_stopping_rounds=2,
    )
    stopped_state = stopped._handle.export_state()
    best_round_count = int(stopped_state["best_iteration"]) + 1
    assert 0 < best_round_count < 12

    reference = ctboost.train(
        ctboost.Pool(train_data, train_labels),
        params,
        num_boost_round=best_round_count,
        eval_set=ctboost.Pool(eval_data, eval_labels),
    )

    np.testing.assert_allclose(
        stopped.predict(eval_data), reference.predict(eval_data), rtol=1e-6, atol=1e-6
    )


def test_zero_regularized_huber_keeps_leaf_values_finite():
    feature = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    data = feature.reshape(-1, 1)
    labels = np.where(feature > 0.0, 10.0, -10.0).astype(np.float32)
    booster = ctboost.train(
        ctboost.Pool(data, labels),
        {
            "objective": "Huber",
            "huber_delta": 1.0,
            "lambda_l2": 0.0,
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
        },
        num_boost_round=2,
    )

    predictions = np.asarray(booster.predict(data))
    assert np.all(np.isfinite(predictions))
    assert np.all(np.isfinite(booster._handle.export_state()["loss_history"]))


def test_maximum_numeric_bin_count_reserves_missing_bin():
    values = np.concatenate(
        [np.arange(65535, dtype=np.float32), np.asarray([np.nan], dtype=np.float32)]
    )
    pool = ctboost.Pool(values.reshape(-1, 1), np.zeros(values.size, dtype=np.float32))

    summary = ctboost._core._debug_build_histogram(
        pool._handle, max_bins=65535, nan_mode="Min"
    )

    assert summary["num_bins_per_feature"] == [65535]


def test_categorical_missing_bin_capacity_fails_during_quantization():
    values = np.concatenate(
        [np.arange(256, dtype=np.float32), np.asarray([np.nan], dtype=np.float32)]
    )
    pool = ctboost.Pool(
        values.reshape(-1, 1), np.zeros(values.size, dtype=np.float32), cat_features=[0]
    )

    with pytest.raises(ValueError, match="including the missing-value bin"):
        ctboost._core._debug_build_histogram(pool._handle, max_bins=256, nan_mode="Min")


def test_custom_borders_cannot_overflow_with_missing_bin():
    data = np.asarray([[0.0], [np.nan]], dtype=np.float32)
    pool = ctboost.Pool(data, np.zeros(data.shape[0], dtype=np.float32))
    borders = np.arange(65534, dtype=np.float32).tolist()

    with pytest.raises(ValueError, match="more than 65535 total bins"):
        ctboost._core._debug_build_histogram(
            pool._handle,
            max_bins=256,
            nan_mode="Min",
            feature_borders=[borders],
        )


def test_low_level_prediction_rejects_wrong_feature_count_safely():
    data = np.arange(24, dtype=np.float32).reshape(8, 3)
    labels = data[:, 0] - data[:, 1]
    booster = ctboost.train(
        ctboost.Pool(data, labels),
        {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
        num_boost_round=3,
    )
    wrong_width = np.zeros((4, 2), dtype=np.float32)

    for predict in (
        booster.predict,
        booster.predict_leaf_index,
        booster.predict_contrib,
    ):
        with pytest.raises(ValueError, match="same number of columns"):
            predict(wrong_width)
