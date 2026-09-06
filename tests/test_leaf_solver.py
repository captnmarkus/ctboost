"""Safeguarded leaf fitting must preserve conditional tree structure."""

import numpy as np
import pytest

import ctboost


def _params(**overrides):
    return {
        "iterations": 1,
        "learning_rate": 1.0,
        "max_depth": 0,
        "alpha": 1.0,
        "boost_from_average": False,
        "lambda_l2": 0.0,
        **overrides,
    }


def _multiclass_loss(margins, labels, weights=None):
    margins = np.asarray(margins, dtype=np.float64)
    shifted = margins - margins.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[
        np.arange(len(labels)), np.asarray(labels, dtype=int)
    ]
    return np.average(losses, weights=weights)


def _binary_loss(margins, labels, weights=None):
    margins = np.asarray(margins, dtype=np.float64)
    return np.average(np.logaddexp(0.0, margins) - labels * margins, weights=weights)


def _leaf_vector(model):
    trees = model._handle.export_state()["trees"]
    if len(trees) == 1:
        return np.asarray(trees[0]["nodes"][0]["leaf_weights"])
    return np.asarray([tree["nodes"][0]["leaf_weight"] for tree in trees])


def _topology(tree):
    return [
        {key: value for key, value in node.items() if key not in {"leaf_weight", "leaf_weights"}}
        for node in tree["nodes"]
    ]


@pytest.mark.parametrize("steps", [1, 5])
def test_binary_backtracking_prevents_initial_newton_overshoot(steps):
    X = np.zeros((20, 1), dtype=np.float32)
    labels = np.tile([0.0, 1.0], 10).astype(np.float32)
    baseline = np.full(len(labels), 10.0, dtype=np.float32)
    pool = ctboost.Pool(X, labels, baseline=baseline)
    params = _params(objective="Logloss", leaf_estimation_iterations=steps)
    unsafe = ctboost.train(pool, params)
    safe = ctboost.train(pool, {**params, "leaf_estimation_backtracking": True})

    initial_loss = _binary_loss(baseline, labels)
    assert _binary_loss(unsafe.predict(pool), labels) > initial_loss
    assert _binary_loss(safe.predict(pool), labels) < initial_loss
    assert _topology(safe._handle.export_state()["trees"][0]) == _topology(
        unsafe._handle.export_state()["trees"][0]
    )


def test_squared_error_backtracking_preserves_weighted_regularized_solution():
    X = np.zeros((3, 1), dtype=np.float32)
    labels = np.asarray([1.0, 3.0, 7.0], dtype=np.float32)
    weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    model = ctboost.train(
        ctboost.Pool(X, labels, weight=weights),
        _params(
            objective="RMSE", lambda_l2=2.0, leaf_estimation_iterations=5,
            leaf_estimation_backtracking=True,
        ),
    )
    expected = np.dot(weights, labels) / (weights.sum() + 2.0)
    np.testing.assert_allclose(model.predict(X), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("objective,magnitude", [("Logloss", 1000.0), ("RMSE", 1e30)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_scalar_backtracking_keeps_extreme_finite_margins_safe(objective, magnitude, sign):
    X = np.zeros((12, 1), dtype=np.float32)
    labels = np.tile([0.0, 1.0], 6).astype(np.float32)
    baseline = np.full(len(X), sign * magnitude, dtype=np.float32)
    pool = ctboost.Pool(X, labels, baseline=baseline)
    model = ctboost.train(
        pool,
        _params(objective=objective, leaf_estimation_backtracking=True, leaf_estimation_iterations=5),
    )
    prediction = model.predict(pool)
    assert np.isfinite(prediction).all()
    if objective == "Logloss":
        assert _binary_loss(prediction, labels) <= _binary_loss(baseline, labels)
    else:
        assert np.square(prediction.astype(np.float64) - labels).sum() < np.square(
            baseline.astype(np.float64) - labels
        ).sum()


@pytest.mark.parametrize("objective", ["Logloss", "MultiClass"])
def test_explicit_default_solver_controls_preserve_predictions_and_state(objective):
    rng = np.random.default_rng(620)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    labels = (
        np.argmax(X, axis=1) if objective == "MultiClass" else X[:, 0] > X[:, 1]
    ).astype(np.float32)
    params = _params(
        objective=objective, num_classes=3 if objective == "MultiClass" else 2,
        iterations=3, max_depth=2, learning_rate=0.2, lambda_l2=1.0,
    )
    implicit = ctboost.train(X, params, label=labels)
    explicit = ctboost.train(
        X,
        {**params, "leaf_estimation_backtracking": False, "multiclass_leaf_solver": "diagonal"},
        label=labels,
    )
    np.testing.assert_array_equal(explicit.predict(X), implicit.predict(X))
    assert explicit._handle.export_state() == implicit._handle.export_state()


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_full_softmax_matches_coupled_regularized_newton_solution(strategy):
    labels = np.repeat(np.arange(3), [1, 3, 8]).astype(np.float32)
    X = np.zeros((len(labels), 1), dtype=np.float32)
    model = ctboost.train(
        X,
        _params(
            objective="MultiClass", num_classes=3, lambda_l2=2.0,
            multi_strategy=strategy, multiclass_leaf_solver="full",
        ),
        label=labels,
    )
    # At uniform probabilities H = N/K * (I - 11^T/K). In the sum-zero
    # subspace the L2-regularized Newton step is -G / (N/K + lambda).
    expected = (np.asarray([1.0, 3.0, 8.0]) - 4.0) / 6.0
    np.testing.assert_allclose(_leaf_vector(model), expected, rtol=2e-6, atol=2e-6)
    assert abs(_leaf_vector(model).sum()) < 1e-6


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_full_softmax_backtracks_extreme_logits_and_preserves_first_tree(strategy):
    labels = np.repeat(np.arange(3), [2, 5, 13]).astype(np.float32)
    X = np.arange(len(labels), dtype=np.float32).reshape(-1, 1)
    baseline = np.tile(np.asarray([20.0, -20.0, -20.0], dtype=np.float32), (len(X), 1))
    pool = ctboost.Pool(X, labels, baseline=baseline)
    common = _params(objective="MultiClass", num_classes=3, multi_strategy=strategy)
    diagonal = ctboost.train(pool, common)
    full = ctboost.train(pool, {**common, "multiclass_leaf_solver": "full"})

    prediction = full.predict(pool)
    assert np.isfinite(prediction).all()
    assert _multiclass_loss(prediction, labels) < _multiclass_loss(baseline, labels)
    assert _topology(diagonal._handle.export_state()["trees"][0]) == _topology(
        full._handle.export_state()["trees"][0]
    )
    assert abs(_leaf_vector(full).sum()) < 1e-5


def test_full_softmax_is_class_permutation_equivariant_with_weights():
    rng = np.random.default_rng(719)
    X = np.zeros((48, 2), dtype=np.float32)
    labels = np.tile(np.arange(4), 12).astype(np.float32)
    baseline = rng.normal(size=(len(X), 4)).astype(np.float32)
    weights = np.linspace(0.0, 2.0, len(X), dtype=np.float32)
    permutation = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    common = _params(
        objective="MultiClass", num_classes=4, multi_strategy="multi_output_tree",
        multiclass_leaf_solver="full", leaf_estimation_iterations=5, lambda_l2=0.7,
    )
    original_pool = ctboost.Pool(X, labels, weight=weights, baseline=baseline)
    permuted_pool = ctboost.Pool(
        X, inverse[labels.astype(int)].astype(np.float32), weight=weights,
        baseline=baseline[:, permutation],
    )
    original = ctboost.train(original_pool, common)
    permuted = ctboost.train(permuted_pool, common)
    np.testing.assert_allclose(
        permuted.predict(permuted_pool), original.predict(original_pool)[:, permutation],
        rtol=2e-6, atol=2e-6,
    )


@pytest.mark.parametrize("classes", [3, 4, 10, 32])
@pytest.mark.parametrize("magnitude", [20.0, 1000.0])
def test_full_softmax_saturated_zero_l2_solver_is_class_permutation_equivariant(classes, magnitude):
    rng = np.random.default_rng(719)
    X = np.zeros((classes * 4, 1), dtype=np.float32)
    labels = np.tile(np.arange(classes), 4).astype(np.float32)
    weights = rng.uniform(0.1, 2.0, len(X)).astype(np.float32)
    baseline = np.full((len(X), classes), -magnitude, dtype=np.float32)
    baseline[:, 0] = magnitude
    permutation = rng.permutation(classes)
    common = _params(
        objective="MultiClass", num_classes=classes, multi_strategy="multi_output_tree",
        multiclass_leaf_solver="full", leaf_estimation_iterations=5,
    )
    original_pool = ctboost.Pool(X, labels, weight=weights, baseline=baseline)
    permuted_pool = ctboost.Pool(
        X, np.argsort(permutation)[labels.astype(int)], weight=weights,
        baseline=baseline[:, permutation],
    )
    original = ctboost.train(original_pool, common)
    permuted = ctboost.train(permuted_pool, common)
    values = _leaf_vector(original)
    np.testing.assert_allclose(_leaf_vector(permuted), values[permutation], rtol=2e-6, atol=1e-5)
    assert abs(values.sum()) <= 2 * np.finfo(np.float32).eps * np.abs(values).sum()
    assert _multiclass_loss(original.predict(original_pool), labels, weights) < _multiclass_loss(
        baseline, labels, weights,
    )


@pytest.mark.parametrize("cap", [0.0, 0.3, 1e-20])
def test_full_softmax_zero_l2_keeps_sum_zero_and_box_after_every_iteration(cap):
    labels = np.repeat(np.arange(3), [1, 3, 8]).astype(np.float32)
    X = np.zeros((len(labels), 1), dtype=np.float32)
    losses = []
    for steps in range(1, 6):
        model = ctboost.train(
            X,
            _params(
                objective="MultiClass", num_classes=3, multi_strategy="multi_output_tree",
                multiclass_leaf_solver="full", leaf_estimation_iterations=steps,
                max_leaf_weight=cap,
            ),
            label=labels,
        )
        values = _leaf_vector(model)
        assert np.isfinite(values).all()
        assert abs(values.sum()) <= 2e-7 * max(np.abs(values).max(), 1e-30)
        if cap:
            assert np.abs(values).max() <= cap
        losses.append(_multiclass_loss(model.predict(X), labels))
    assert np.all(np.diff(losses) <= 1e-7)
    if not cap:
        optimum = np.log([1.0, 3.0, 8.0])
        optimum -= optimum.mean()
        np.testing.assert_allclose(values, optimum, rtol=2e-5, atol=2e-5)


def test_full_softmax_storage_layouts_and_persistence_agree(tmp_path):
    rng = np.random.default_rng(810)
    X = rng.normal(size=(90, 3)).astype(np.float32)
    labels = np.argmax(X, axis=1).astype(np.float32)
    common = _params(
        objective="MultiClass", num_classes=3, max_depth=2, learning_rate=0.2,
        multiclass_leaf_solver="full", leaf_estimation_iterations=3, lambda_l2=0.3,
    )
    scalar = ctboost.train(X, {**common, "iterations": 4}, label=labels)
    vector = ctboost.train(
        X, {**common, "iterations": 4, "multi_strategy": "multi_output_tree"}, label=labels,
    )
    np.testing.assert_array_equal(vector.predict(X), scalar.predict(X))
    path = tmp_path / "coupled.json"
    vector.save_model(path)
    loaded = ctboost.load_model(path)
    assert loaded.multiclass_leaf_solver == "full"
    assert loaded.leaf_estimation_iterations == 3
    np.testing.assert_array_equal(loaded.predict(X), vector.predict(X))
    continued = ctboost.train(X, {}, label=labels, num_boost_round=1, init_model=loaded)
    reference = ctboost.train(
        X, {**common, "iterations": 5, "multi_strategy": "multi_output_tree"}, label=labels,
    )
    np.testing.assert_array_equal(continued.predict(X), reference.predict(X))


def test_full_softmax_keeps_conditional_first_tree_routing():
    rng = np.random.default_rng(908)
    X = rng.normal(size=(150, 4)).astype(np.float32)
    labels = np.argmax(X[:, :3], axis=1).astype(np.float32)
    common = _params(
        objective="MultiClass", num_classes=3, max_depth=3, lambda_l2=1.0,
        multi_strategy="multi_output_tree",
    )
    diagonal = ctboost.train(X, common, label=labels)
    full = ctboost.train(X, {**common, "multiclass_leaf_solver": "full"}, label=labels)
    original_tree = diagonal._handle.export_state()["trees"][0]
    assert len(original_tree["nodes"]) > 1
    assert _topology(original_tree) == _topology(full._handle.export_state()["trees"][0])
    np.testing.assert_array_equal(full.predict_leaf_index(X), diagonal.predict_leaf_index(X))


def test_scalar_backtracking_preserves_monotone_constraints_and_leaf_cap():
    X = np.linspace(-2, 2, 100, dtype=np.float32).reshape(-1, 1)
    labels = (X[:, 0] + 0.4 * np.sin(10 * X[:, 0]) > 0).astype(np.float32)
    model = ctboost.train(
        X,
        _params(
            objective="Logloss", iterations=4, max_depth=3,
            leaf_estimation_iterations=5, leaf_estimation_backtracking=True,
            monotone_constraints=[1], max_leaf_weight=0.4,
        ),
        label=labels,
    )
    assert np.all(np.diff(model.predict(X)) >= -1e-6)
    for tree in model._handle.export_state()["trees"]:
        for node in tree["nodes"]:
            if node["is_leaf"]:
                assert abs(node["leaf_weight"]) <= 0.4 + 1e-7
