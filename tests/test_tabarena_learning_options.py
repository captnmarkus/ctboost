"""Eligibility and frozen-portfolio contracts; these tests launch no benchmark fits."""

import copy

import numpy as np
import pytest

from benchmarks.tabarena.ctboost_model import (
    CTBoostTabArenaGPUModel,
    CTBoostTabArenaModel,
    generate_configs_ctboost,
    generate_configs_ctboost_learning_options,
)
from benchmarks.tabarena.learning_options import (
    LEARNING_INAPPLICABLE_PARAM,
    LEARNING_VARIANT_PARAM,
    resolve_cpu_learning_options,
)

MODES = {
    "binary": "backtracking_3",
    "regression": "baseline",
    "multiclass": "full_3_joint",
}
BASELINE = {
    "leaf_estimation_iterations": 1,
    "leaf_estimation_backtracking": False,
    "multiclass_leaf_solver": "diagonal",
    "multiclass_feature_test": "single",
}


@pytest.mark.parametrize("task_type", ["CPU", "GPU"])
def test_unmarked_legacy_parameters_are_unchanged(task_type):
    params = {"task_type": task_type, "alpha": 0.031, "feature_test": "quadratic"}
    resolved, metadata = resolve_cpu_learning_options(params, problem_type="multiclass")
    assert resolved == params
    assert resolved is not params
    assert metadata["status"] == "unchanged"
    assert not set(BASELINE).intersection(resolved)


@pytest.mark.parametrize(
    ("problem", "variant", "overrides"),
    [
        ("binary", "baseline", {}),
        ("regression", "baseline", {}),
        ("multiclass", "baseline", {}),
        (
            "binary",
            "backtracking_3",
            {"leaf_estimation_iterations": 3, "leaf_estimation_backtracking": True},
        ),
        (
            "regression",
            "backtracking_3",
            {"leaf_estimation_iterations": 3, "leaf_estimation_backtracking": True},
        ),
        (
            "multiclass",
            "full_3",
            {"leaf_estimation_iterations": 3, "multiclass_leaf_solver": "full"},
        ),
        ("multiclass", "joint", {"multiclass_feature_test": "joint"}),
        (
            "multiclass",
            "full_3_joint",
            {
                "leaf_estimation_iterations": 3,
                "multiclass_leaf_solver": "full",
                "multiclass_feature_test": "joint",
            },
        ),
    ],
)
def test_arms_change_only_declared_learning_controls(problem, variant, overrides):
    params = {
        "alpha": 0.05,
        "feature_test": "quadratic",
        "subsample": 0.8,
        "bootstrap_type": "Bernoulli",
        "random_seed": 159,
        "multiclass_tree_storage": "scalar",
    }
    original = copy.deepcopy(params)
    resolved, metadata = resolve_cpu_learning_options(
        params,
        problem_type=problem,
        num_classes=3 if problem == "multiclass" else 2,
        variant=variant,
    )
    expected = dict(BASELINE, **overrides)
    assert params == original
    assert resolved == dict(params, **expected)
    assert metadata["resolved_controls"] == expected
    assert metadata["applied_variant"] == variant
    assert metadata["applicable"]


@pytest.mark.parametrize(
    ("problem", "variant"),
    [("binary", "joint"), ("regression", "full_3"), ("multiclass", "backtracking_3")],
)
def test_wrong_task_arm_is_explicitly_skipped_or_rejected(problem, variant):
    with pytest.raises(ValueError, match="not applicable"):
        resolve_cpu_learning_options({}, problem_type=problem, variant=variant)
    resolved, metadata = resolve_cpu_learning_options(
        {},
        problem_type=problem,
        variant=variant,
        on_inapplicable="skip",
    )
    assert resolved is None
    assert metadata["status"] == "not_applicable"
    assert metadata["applied_variant"] is None
    assert not metadata["applicable"]


@pytest.mark.parametrize("variant", ["full_3", "joint", "full_3_joint"])
@pytest.mark.parametrize("classes", [2, 3, 32, 33])
def test_multiclass_bound_uses_training_class_count(variant, classes):
    resolved, metadata = resolve_cpu_learning_options(
        {},
        problem_type="multiclass",
        labels=np.arange(classes),
        variant=variant,
        on_inapplicable="skip",
    )
    assert metadata["num_classes"] == classes
    assert metadata["applicable"] == (3 <= classes <= 32)
    assert (resolved is not None) == metadata["applicable"]


def test_explicit_class_count_must_describe_training_labels():
    with pytest.raises(ValueError, match="classes present in training labels"):
        resolve_cpu_learning_options(
            {},
            problem_type="multiclass",
            labels=[0, 1, 2],
            num_classes=4,
            variant="full_3",
        )


@pytest.mark.parametrize("labels", [[], [[0], [1], [2]]])
def test_malformed_training_labels_fail_before_eligibility_fallback(labels):
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        resolve_cpu_learning_options(
            {},
            problem_type="multiclass",
            labels=labels,
            variant="full_3",
            on_inapplicable="baseline",
        )


def test_binary_backtracking_cannot_reach_native_with_multiclass_training_labels():
    resolved, metadata = resolve_cpu_learning_options(
        {},
        problem_type="binary",
        labels=[0, 1, 2],
        variant="backtracking_3",
        on_inapplicable="skip",
    )
    assert resolved is None
    assert "exactly 2" in metadata["reason"]


@pytest.mark.parametrize("policy", ["", "silent", "round_weights"])
def test_unknown_policy_is_not_silently_replaced_with_a_default(policy):
    with pytest.raises(ValueError, match="on_inapplicable"):
        resolve_cpu_learning_options(
            {},
            problem_type="binary",
            variant="baseline",
            on_inapplicable=policy,
        )


def test_conflicting_explicit_and_marked_variants_fail():
    with pytest.raises(ValueError, match="conflicting"):
        resolve_cpu_learning_options(
            {LEARNING_VARIANT_PARAM: "baseline"},
            problem_type="binary",
            variant="backtracking_3",
        )


@pytest.mark.parametrize("problem", ["binary", "regression", "multiclass"])
def test_family_mapping_is_resolved_without_changing_the_marker(problem):
    modes = dict(MODES)
    _, metadata = resolve_cpu_learning_options(
        {LEARNING_VARIANT_PARAM: modes},
        problem_type=problem,
        num_classes=3 if problem == "multiclass" else 2,
    )
    assert metadata["requested_variant"] == MODES[problem]
    assert metadata["applied_variant"] == MODES[problem]
    assert modes == MODES


@pytest.mark.parametrize("policy", ["skip", "baseline"])
def test_gpu_markers_always_fail_instead_of_becoming_cpu_or_baseline(policy):
    with pytest.raises(ValueError, match="CPU-only"):
        resolve_cpu_learning_options(
            {"task_type": "GPU"},
            problem_type="binary",
            variant="baseline",
            on_inapplicable=policy,
        )


def test_distributed_markers_fail_before_model_construction():
    with pytest.raises(ValueError, match="distributed"):
        resolve_cpu_learning_options(
            {"distributed_world_size": 2},
            problem_type="binary",
            variant="backtracking_3",
        )


@pytest.mark.parametrize("objective", ["MAE", "Huber", object()])
def test_backtracking_rejects_other_or_custom_objectives(objective):
    resolved, metadata = resolve_cpu_learning_options(
        {"objective": objective},
        problem_type="regression",
        variant="backtracking_3",
        on_inapplicable="skip",
    )
    assert resolved is None
    assert "built-in objective" in metadata["reason"]


@pytest.mark.parametrize("objective", ["RMSE", "SquaredError", "squared_error"])
def test_backtracking_accepts_builtin_squared_error_aliases(objective):
    resolved, _ = resolve_cpu_learning_options(
        {"loss_function": objective},
        problem_type="regression",
        variant="backtracking_3",
    )
    assert resolved["leaf_estimation_backtracking"]


@pytest.mark.parametrize(
    ("params", "weights", "labels", "eligible"),
    [
        ({}, [0, 1, 2], [0, 1, 2], True),
        ({}, [0.5, 1, 2], [0, 1, 2], False),
        ({"class_weight": {"a": 1, "b": 2, "c": 3}}, None, ["a", "b", "c"], True),
        ({"class_weight": [1, 0.5, 2]}, None, [0, 1, 2], False),
        ({"class_weight": [2, 2, 2]}, [0.5, 1, 1.5], [0, 1, 2], True),
        ({"class_weight": "balanced"}, None, [0, 1, 2], True),
        ({"auto_class_weights": "balanced"}, None, [0, 0, 1, 2], False),
        (
            {"bootstrap_type": "Bayesian", "bagging_temperature": 1},
            None,
            [0, 1, 2],
            False,
        ),
        (
            {"bootstrap_type": "Bayesian", "bagging_temperature": float("nan")},
            None,
            [0, 1, 2],
            False,
        ),
        (
            {"bootstrap_type": "Bayesian", "bagging_temperature": 0},
            None,
            [0, 1, 2],
            True,
        ),
    ],
)
def test_joint_eligibility_uses_actual_effective_weights(
    params, weights, labels, eligible
):
    original_weights = copy.deepcopy(weights)
    resolved, metadata = resolve_cpu_learning_options(
        params,
        problem_type="multiclass",
        labels=labels,
        sample_weight=weights,
        variant="joint",
        on_inapplicable="skip",
    )
    assert weights == original_weights
    assert metadata["applicable"] == eligible
    assert (resolved is not None) == eligible


@pytest.mark.parametrize(
    "weights", [[-1, 1, 1], [float("nan"), 1, 1], [1, 1], [[1, 1, 1]]]
)
def test_malformed_weights_cannot_be_hidden_by_baseline_fallback(weights):
    with pytest.raises(ValueError, match=r"sample.weight"):
        resolve_cpu_learning_options(
            {},
            problem_type="multiclass",
            labels=[0, 1, 2],
            sample_weight=weights,
            variant="joint",
            on_inapplicable="baseline",
        )


def test_full_solver_retains_fractional_weights_and_bayesian_bootstrap():
    params = {"bootstrap_type": "Bayesian", "bagging_temperature": 1}
    resolved, metadata = resolve_cpu_learning_options(
        params,
        problem_type="multiclass",
        labels=[0, 1, 2],
        sample_weight=[0.5, 1, 1.5],
        variant="full_3",
    )
    assert metadata["applicable"]
    assert resolved["bootstrap_type"] == "Bayesian"
    assert resolved["bagging_temperature"] == 1


def test_explicit_portfolio_fallback_removes_all_new_controls_and_records_reason():
    params = {
        LEARNING_VARIANT_PARAM: MODES,
        LEARNING_INAPPLICABLE_PARAM: "baseline",
        "feature_test": "quadratic",
        "leaf_estimation_iterations": 3,
        "multiclass_leaf_solver": "full",
        "multiclass_feature_test": "joint",
    }
    resolved, metadata = resolve_cpu_learning_options(
        params,
        problem_type="multiclass",
        num_classes=40,
    )
    assert resolved == dict(BASELINE, feature_test="quadratic")
    assert metadata["status"] == "baseline_fallback"
    assert metadata["requested_variant"] == "full_3_joint"
    assert metadata["applied_variant"] == "baseline"
    assert "3-32" in metadata["reason"]
    assert not metadata["applicable"]


def test_new_portfolio_preserves_old_numeric_prefix_and_predeclared_odd_schedule():
    frozen = generate_configs_ctboost(200)
    modes = dict(MODES)
    configs = generate_configs_ctboost_learning_options(
        approved_modes_by_problem_type=modes
    )
    assert len(configs) == 25
    assert sum(LEARNING_VARIANT_PARAM in config for config in configs) == 13
    for index, config in enumerate(configs):
        numeric = dict(config)
        if index % 2 == 0:
            assert numeric.pop(LEARNING_VARIANT_PARAM) == MODES
            assert numeric.pop(LEARNING_INAPPLICABLE_PARAM) == "baseline"
        assert numeric == frozen[index]
    assert generate_configs_ctboost(200) == frozen
    assert (
        generate_configs_ctboost_learning_options(
            7,
            approved_modes_by_problem_type=modes,
        )
        == configs[:7]
    )
    modes["binary"] = "baseline"
    configs[0][LEARNING_VARIANT_PARAM]["regression"] = "backtracking_3"
    assert configs[2][LEARNING_VARIANT_PARAM] == MODES


def test_no_pilot_family_pass_leaves_every_config_unchanged():
    assert generate_configs_ctboost_learning_options(
        approved_modes_by_problem_type=dict.fromkeys(MODES, "baseline"),
    ) == generate_configs_ctboost(25)


@pytest.mark.parametrize("count", [-1, 26, 25.5, True])
def test_new_portfolio_rejects_unapproved_sizes(count):
    with pytest.raises(ValueError, match="0 to 25"):
        generate_configs_ctboost_learning_options(
            count, approved_modes_by_problem_type=MODES
        )


@pytest.mark.parametrize(
    "modes",
    [
        {"binary": "backtracking_3"},
        dict(MODES, regression="joint"),
        dict(MODES, multiclass="grouped8"),
        dict(MODES, unknown="baseline"),
    ],
)
def test_incomplete_or_invalid_family_mapping_is_rejected(modes):
    with pytest.raises(ValueError):
        generate_configs_ctboost_learning_options(approved_modes_by_problem_type=modes)


def _adapter(params, problem, model_cls=CTBoostTabArenaModel):
    adapter = object.__new__(model_cls)
    adapter.problem_type = problem
    adapter.stopping_metric = None
    adapter._ctboost_categorical_columns = []
    adapter.preprocess = lambda frame, **_kwargs: frame
    adapter._get_model_params = lambda: dict(params)
    return adapter


def test_adapter_consumes_marker_and_preserves_weight_array(monkeypatch):
    class FakeClassifier:
        def __init__(self, **params):
            self.params = params

        def fit(self, _X, _y, **kwargs):
            self.fit_kwargs = kwargs

    monkeypatch.setattr("ctboost.CTBoostClassifier", FakeClassifier)
    params = {LEARNING_VARIANT_PARAM: MODES, LEARNING_INAPPLICABLE_PARAM: "baseline"}
    adapter = _adapter(params, "multiclass")
    # Use fit labels, not a larger class count from an outer split.
    adapter.num_classes = 40
    weights = np.array([1, 2, 3], dtype=np.float32)
    adapter._fit(np.ones((3, 1)), np.arange(3), sample_weight=weights)
    assert LEARNING_VARIANT_PARAM not in adapter.model.params
    assert LEARNING_INAPPLICABLE_PARAM not in adapter.model.params
    assert adapter.model.params["multiclass_leaf_solver"] == "full"
    assert adapter.model.params["multiclass_feature_test"] == "joint"
    assert adapter.model.fit_kwargs["sample_weight"] is weights
    assert adapter._ctboost_learning_options["num_classes"] == 3


@pytest.mark.parametrize("marker", ["baseline", MODES])
def test_gpu_adapter_rejects_cpu_markers_even_if_config_claims_cpu(marker):
    adapter = _adapter(
        {"task_type": "CPU", LEARNING_VARIANT_PARAM: marker},
        "multiclass",
        CTBoostTabArenaGPUModel,
    )
    with pytest.raises(ValueError, match="CPU-only"):
        adapter._fit(np.ones((3, 1)), np.arange(3))


def test_adapter_cannot_silently_skip_an_ineligible_config():
    adapter = _adapter(
        {LEARNING_VARIANT_PARAM: "joint", LEARNING_INAPPLICABLE_PARAM: "skip"},
        "binary",
    )
    with pytest.raises(ValueError, match="TabArena fit cannot skip"):
        adapter._fit(np.ones((2, 1)), np.arange(2))
