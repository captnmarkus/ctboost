from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.base import clone

import ctboost
import ctboost._core as _core


def _tree_topology(tree_state):
    return [
        (
            node["is_leaf"],
            node["is_categorical_split"],
            node["split_feature_id"],
            node["split_bin_index"],
            node["left_child"],
            node["right_child"],
            tuple(node["left_categories"]),
        )
        for node in tree_state["nodes"]
    ]


def test_one_leaf_estimation_iteration_is_exactly_the_legacy_path(tmp_path):
    rng = np.random.default_rng(310)
    X = rng.normal(size=(96, 5)).astype(np.float32)
    y = (X[:, 0] > 0.25 * X[:, 1]).astype(np.float32)
    params = {
        "objective": "Logloss",
        "iterations": 7,
        "learning_rate": 0.13,
        "max_depth": 3,
        "alpha": 1.0,
        "lambda_l2": 0.7,
        "subsample": 0.8,
        "bootstrap_type": "Bernoulli",
        "random_seed": 29,
        "boost_from_average": False,
    }

    implicit = ctboost.train(X, params, label=y)
    explicit = ctboost.train(
        X,
        {**params, "leaf_estimation_iterations": 1},
        label=y,
    )

    np.testing.assert_array_equal(implicit.predict(X), explicit.predict(X))
    assert implicit.loss_history == explicit.loss_history
    assert implicit._handle.export_state() == explicit._handle.export_state()
    implicit_path = tmp_path / "implicit.ctb"
    explicit_path = tmp_path / "explicit.ctb"
    implicit.save_model(implicit_path)
    explicit.save_model(explicit_path)
    assert implicit_path.read_bytes() == explicit_path.read_bytes()


def test_native_constructor_preserves_legacy_positional_argument_order():
    native = _core.GradientBooster(
        "Logloss",
        1,
        0.1,
        1,
        0.05,
        1.0,
        1.0,
        "No",
        0.0,
        "GradientBoosting",
        0.1,
        0.5,
        0,
        [],
        [],
        1.0,
        [],
        [],
        0.0,
        "DepthWise",
        0,
        2,
        0,
        0.0,
        0.0,
        0.0,
        2,
        64,
        "Max",
    )
    assert native.num_classes() == 2
    assert native.max_bins() == 64
    assert native.nan_mode_name() == "Max"
    assert native.leaf_estimation_iterations() == 1


def test_second_newton_step_uses_unshrunk_fixed_tree_margin():
    X = np.zeros((4, 1), dtype=np.float32)
    y = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    learning_rate = 0.2
    booster = ctboost.train(
        X,
        {
            "objective": "Logloss",
            "iterations": 1,
            "learning_rate": learning_rate,
            "max_depth": 0,
            "lambda_l2": 0.0,
            "boost_from_average": False,
            "leaf_estimation_iterations": 2,
        },
        label=y,
    )

    first_leaf_value = 1.0
    probability = 1.0 / (1.0 + math.exp(-first_leaf_value))
    expected_leaf_value = first_leaf_value - (4.0 * probability - 3.0) / (
        4.0 * probability * (1.0 - probability)
    )
    state = booster._handle.export_state()
    actual_leaf_value = state["trees"][0]["nodes"][0]["leaf_weight"]
    assert actual_leaf_value == pytest.approx(expected_leaf_value, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(
        booster.predict(X),
        np.full(4, learning_rate * expected_leaf_value, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_leaf_refinement_preserves_sample_weighted_newton_statistics():
    X = np.zeros((4, 1), dtype=np.float32)
    y = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    weights = np.asarray([1.0, 1.0, 1.0, 10.0], dtype=np.float32)
    booster = ctboost.train(
        X,
        {
            "objective": "Logloss",
            "iterations": 1,
            "learning_rate": 1.0,
            "max_depth": 0,
            "lambda_l2": 0.0,
            "boost_from_average": False,
            "leaf_estimation_iterations": 2,
        },
        label=y,
        weight=weights,
    )

    first_leaf_value = 3.5 / 3.25
    probability = 1.0 / (1.0 + math.exp(-first_leaf_value))
    expected_leaf_value = first_leaf_value - (13.0 * probability - 10.0) / (
        13.0 * probability * (1.0 - probability)
    )
    actual_leaf_value = booster._handle.export_state()["trees"][0]["nodes"][0][
        "leaf_weight"
    ]
    assert actual_leaf_value == pytest.approx(expected_leaf_value, rel=1e-6, abs=1e-6)


def test_more_leaf_steps_improve_logloss_without_changing_split_topology():
    X = np.arange(8, dtype=np.float32).reshape(-1, 1)
    y = np.asarray([0.0] * 4 + [1.0] * 4, dtype=np.float32)
    common = {
        "objective": "Logloss",
        "iterations": 1,
        "learning_rate": 1.0,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "boost_from_average": False,
    }
    one_step = ctboost.train(X, {**common, "leaf_estimation_iterations": 1}, label=y)
    five_steps = ctboost.train(X, {**common, "leaf_estimation_iterations": 5}, label=y)

    one_tree = one_step._handle.export_state()["trees"][0]
    five_tree = five_steps._handle.export_state()["trees"][0]
    assert _tree_topology(one_tree) == _tree_topology(five_tree)
    assert five_steps.loss_history[-1] < one_step.loss_history[-1]


def test_leaf_refinement_honors_leaf_cap_and_monotone_constraint():
    X = np.linspace(-2.0, 2.0, 128, dtype=np.float32).reshape(-1, 1)
    y = (2.0 * X[:, 0] + 0.8 * np.sin(9.0 * X[:, 0])).astype(np.float32)
    booster = ctboost.train(
        X,
        {
            "objective": "Huber",
            "iterations": 4,
            "learning_rate": 0.3,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 0.2,
            "monotone_constraints": [1],
            "max_leaf_weight": 0.4,
            "leaf_estimation_iterations": 5,
            "boost_from_average": False,
        },
        label=y,
    )

    state = booster._handle.export_state()
    leaf_weights = [
        node["leaf_weight"]
        for tree in state["trees"]
        for node in tree["nodes"]
        if node["is_leaf"]
    ]
    assert max(abs(value) for value in leaf_weights) <= 0.4 + 1e-7
    predictions = booster.predict(X)
    assert np.all(np.diff(predictions) >= -1e-6)


def test_leaf_estimation_parameter_round_trips_and_legacy_state_defaults(tmp_path):
    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = (X[:, 0] > 12.0).astype(np.float32)
    booster = ctboost.train(
        X,
        {
            "objective": "Logloss",
            "iterations": 2,
            "max_depth": 1,
            "alpha": 1.0,
            "leaf_estimation_iterations": 3,
        },
        label=y,
    )
    assert booster.leaf_estimation_iterations == 3
    state = dict(booster._handle.export_state())
    assert state["leaf_estimation_iterations"] == 3

    model_path = tmp_path / "leaf-steps.ctb"
    booster.save_model(model_path)
    restored = ctboost.load_model(model_path)
    assert restored.leaf_estimation_iterations == 3
    np.testing.assert_array_equal(restored.predict(X), booster.predict(X))

    predictor_path = tmp_path / "leaf-steps-predictor.json"
    booster.export_model(predictor_path, export_format="json_predictor")
    exported = ctboost.load_exported_predictor(predictor_path)
    np.testing.assert_allclose(
        exported.predict(X),
        booster.predict(X),
        rtol=1e-7,
        atol=1e-7,
    )

    state.pop("leaf_estimation_iterations")
    legacy = ctboost.Booster(_core.GradientBooster.from_state(state))
    assert legacy.leaf_estimation_iterations == 1
    np.testing.assert_array_equal(legacy.predict(X), booster.predict(X))


@pytest.mark.parametrize("invalid_value", [0, 6])
def test_leaf_estimation_iteration_count_is_bounded(invalid_value):
    with pytest.raises(ValueError, match=r"leaf_estimation_iterations must be in \[1, 5\]"):
        _core.GradientBooster(leaf_estimation_iterations=invalid_value)


def test_sklearn_parameter_is_cloneable_and_reaches_native_training():
    estimator = ctboost.CTBoostClassifier(
        iterations=2,
        max_depth=1,
        alpha=1.0,
        leaf_estimation_iterations=4,
    )
    cloned = clone(estimator)
    assert cloned.get_params()["leaf_estimation_iterations"] == 4
    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    cloned.fit(X, y)
    assert cloned.get_booster().leaf_estimation_iterations == 4


def test_snapshot_resume_preserves_leaf_estimation_contract(tmp_path):
    rng = np.random.default_rng(117)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    y = (1.2 * X[:, 0] - 0.5 * X[:, 1] > 0.0).astype(np.float32)
    params = {
        "objective": "Logloss",
        "learning_rate": 0.15,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 37,
        "leaf_estimation_iterations": 3,
    }
    reference = ctboost.train(X, params, label=y, num_boost_round=8)
    snapshot_path = tmp_path / "leaf-resume.ctb"
    ctboost.train(
        X,
        params,
        label=y,
        num_boost_round=3,
        snapshot_path=snapshot_path,
    )
    resumed = ctboost.train(
        X,
        params,
        label=y,
        num_boost_round=8,
        snapshot_path=snapshot_path,
        resume_from_snapshot=True,
    )
    np.testing.assert_array_equal(reference.predict(X), resumed.predict(X))
    assert reference._handle.export_state()["trees"] == resumed._handle.export_state()["trees"]

    with pytest.raises(ValueError, match="Use init_model"):
        ctboost.train(
            X,
            {**params, "leaf_estimation_iterations": 2},
            label=y,
            num_boost_round=10,
            snapshot_path=snapshot_path,
            resume_from_snapshot=True,
        )


def test_init_model_inherits_or_intentionally_changes_leaf_estimation_steps():
    rng = np.random.default_rng(118)
    X = rng.normal(size=(80, 3)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0.0).astype(np.float32)
    common = {
        "objective": "Logloss",
        "learning_rate": 0.1,
        "max_depth": 2,
        "alpha": 1.0,
        "random_seed": 41,
    }
    seed = ctboost.train(
        X,
        {**common, "leaf_estimation_iterations": 2},
        label=y,
        num_boost_round=3,
    )
    seed_trees = seed._handle.export_state()["trees"]

    inherited = ctboost.train(
        X,
        common,
        label=y,
        num_boost_round=2,
        init_model=seed,
    )
    assert inherited.leaf_estimation_iterations == 2
    assert inherited._handle.export_state()["trees"][: len(seed_trees)] == seed_trees

    changed = ctboost.train(
        X,
        {**common, "leaf_estimation_iterations": 4},
        label=y,
        num_boost_round=2,
        init_model=seed,
    )
    assert changed.leaf_estimation_iterations == 4
    assert changed._handle.export_state()["trees"][: len(seed_trees)] == seed_trees


def test_refinement_supports_custom_ranking_and_survival_objectives():
    custom_calls = []

    def squared_error(predictions, label, **_kwargs):
        custom_calls.append(np.asarray(predictions).copy())
        return predictions - label, np.ones_like(predictions, dtype=np.float32)

    X = np.arange(24, dtype=np.float32).reshape(12, 2)
    regression_target = (0.4 * X[:, 0] - X[:, 1]).astype(np.float32)
    custom = ctboost.train(
        X,
        {
            "objective": squared_error,
            "iterations": 1,
            "max_depth": 1,
            "alpha": 1.0,
            "leaf_estimation_iterations": 3,
            "boost_from_average": False,
        },
        label=regression_target,
    )
    assert len(custom_calls) == 3
    assert np.all(np.isfinite(custom.predict(X)))

    group_id = np.repeat(np.arange(3, dtype=np.int64), 4)
    group_weight = np.repeat(
        np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        4,
    )
    ranking_target = np.tile(np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float32), 3)
    ranker = ctboost.train(
        X,
        {
            "objective": "PairLogit",
            "iterations": 2,
            "max_depth": 1,
            "alpha": 1.0,
            "leaf_estimation_iterations": 3,
        },
        label=ranking_target,
        group_id=group_id,
        group_weight=group_weight,
    )
    assert np.all(np.isfinite(ranker.predict(X)))

    signed_time = np.asarray(
        [1.0, -1.5, 2.0, 2.5, -3.0, 3.5, 4.0, -4.5, 5.0, 5.5, -6.0, 6.5],
        dtype=np.float32,
    )
    survival = ctboost.train(
        X,
        {
            "objective": "SurvivalExponential",
            "iterations": 2,
            "max_depth": 1,
            "alpha": 1.0,
            "leaf_estimation_iterations": 3,
        },
        label=signed_time,
    )
    assert np.all(np.isfinite(survival.predict(X)))



def test_multiclass_refinement_fails_closed():
    with pytest.raises(
        ValueError,
        match="leaf_estimation_iterations greater than 1 is not supported for multiclass",
    ):
        _core.GradientBooster(
            objective="MultiClass",
            num_classes=3,
            leaf_estimation_iterations=2,
        )


def test_gpu_leaf_refinement_keeps_topology_and_improves_logloss_when_available():
    if not ctboost.build_info()["cuda_enabled"]:
        pytest.skip("CUDA support is not compiled into this build")

    axis = np.linspace(-2.0, 2.0, 8, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)
    X = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    y = ((X[:, 0] > 0.0) | (X[:, 1] > 0.0)).astype(np.float32)
    params = {
        "objective": "Logloss",
        "iterations": 1,
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 0.3,
        "boost_from_average": False,
        "leaf_estimation_iterations": 3,
    }
    try:
        gpu_one_step = ctboost.train(
            X,
            {
                **params,
                "task_type": "GPU",
                "devices": "0",
                "leaf_estimation_iterations": 1,
            },
            label=y,
        )
        gpu_refined = ctboost.train(
            X,
            {**params, "task_type": "GPU", "devices": "0"},
            label=y,
        )
    except RuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable for leaf-refinement test: {exc}")

    one_tree = gpu_one_step._handle.export_state()["trees"][0]
    refined_tree = gpu_refined._handle.export_state()["trees"][0]
    assert _tree_topology(refined_tree) == _tree_topology(one_tree)
    one_leaves = [node["leaf_weight"] for node in one_tree["nodes"] if node["is_leaf"]]
    refined_leaves = [
        node["leaf_weight"] for node in refined_tree["nodes"] if node["is_leaf"]
    ]
    assert len(refined_leaves) > 1
    assert refined_leaves != one_leaves
    assert np.all(np.isfinite(gpu_refined.predict(X)))
    assert gpu_refined.loss_history[-1] <= gpu_one_step.loss_history[-1] + 1e-9
