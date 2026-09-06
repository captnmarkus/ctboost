import itertools

import numpy as np
import pytest
from sklearn.base import clone

import ctboost
from benchmarks.multivariate_statistics_calibration import (
    native_statistic,
    reference_statistic,
    small_permutation_oracle,
)


@pytest.mark.parametrize("dimension", [1, 2, 3, 6, 16, 32])
@pytest.mark.parametrize("fractional", [False, True])
def test_joint_matches_full_conditional_covariance_reference(dimension, fractional):
    rng = np.random.default_rng(159 + dimension)
    gradients = rng.normal(size=(120, dimension)).astype(np.float32)
    if dimension > 1:
        gradients -= gradients.mean(axis=1, keepdims=True)
    bins = rng.integers(0, 4, size=len(gradients))
    weights = rng.uniform(0.1, 2, len(gradients)) if fractional else rng.integers(0, 4, len(gradients))
    native = native_statistic(gradients, bins, weights)
    reference = reference_statistic(gradients, bins, weights)
    assert native["response_rank"] == reference["response_rank"]
    assert native["degrees_of_freedom"] == reference["degrees_of_freedom"]
    assert bool(native["frequency_weights"]) is (not fractional)
    np.testing.assert_allclose(native["chi_square"], reference["chi_square"], rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(native["p_value"], reference["p_value"], rtol=2e-8, atol=1e-12)


def test_joint_uses_all_coordinates_and_removes_softmax_null_direction():
    rng = np.random.default_rng(159)
    bins = np.repeat([0, 1], 200)
    nuisance = rng.normal(size=400) * 1000
    for level in (0, 1):
        nuisance[bins == level] -= nuisance[bins == level].mean()
    signal = bins * 2 - 1
    gradients = np.column_stack([nuisance, signal, -nuisance - signal]).astype(np.float32)
    result = native_statistic(gradients, bins)
    assert result["response_rank"] == 2
    assert result["degrees_of_freedom"] == 2
    assert result["p_value"] < 1e-50
    chosen = np.argmax(np.var(gradients.astype(float), axis=0))
    single = reference_statistic(gradients[:, [chosen]], bins)
    assert single["p_value"] > 0.95


def test_joint_is_invariant_to_every_class_permutation_and_bin_relabeling():
    rng = np.random.default_rng(758)
    logits = rng.normal(size=(160, 4))
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    gradients = (probability - np.eye(4)[rng.integers(4, size=160)]).astype(np.float32)
    bins = rng.integers(4, size=160)
    baseline = native_statistic(gradients, bins)
    for permutation in itertools.permutations(range(4)):
        result = native_statistic(gradients[:, permutation], np.array(permutation)[bins])
        assert result["response_rank"] == 3
        assert result["degrees_of_freedom"] == baseline["degrees_of_freedom"]
        np.testing.assert_allclose(result["chi_square"], baseline["chi_square"], rtol=1e-11, atol=1e-11)


def test_integer_frequency_weights_equal_literal_expansion():
    rng = np.random.default_rng(72)
    gradients = rng.normal(size=(40, 4)).astype(np.float32)
    bins = rng.integers(5, size=40)
    counts = rng.integers(0, 5, size=40)
    weighted = native_statistic(gradients, bins, counts)
    expanded = native_statistic(np.repeat(gradients, counts, axis=0), np.repeat(bins, counts))
    assert weighted["frequency_weights"]
    for field in ("chi_square", "p_value", "degrees_of_freedom"):
        np.testing.assert_allclose(weighted[field], expanded[field], rtol=1e-11, atol=1e-12)


def test_fractional_weights_are_explicit_and_scaling_changes_frequency_mass():
    rng = np.random.default_rng(15)
    gradients = rng.normal(size=(90, 3)).astype(np.float32)
    bins = np.repeat([0, 1, 2], 30)
    weights = np.full(90, 0.5, dtype=np.float32)
    fractional = native_statistic(gradients, bins, weights)
    rescaled = native_statistic(gradients, bins, weights * 4)
    assert not fractional["frequency_weights"]
    assert rescaled["frequency_weights"]
    np.testing.assert_allclose(rescaled["chi_square"] / fractional["chi_square"], (180 - 1) / (45 - 1), rtol=1e-11)
    assert rescaled["p_value"] < fractional["p_value"]


@pytest.mark.parametrize("missing_bin", [0, 15])
def test_grouped_joint_preserves_missing_bin_and_matches_reference(missing_bin):
    rng = np.random.default_rng(142)
    bins = rng.integers(16, size=250)
    gradients = rng.normal(size=(250, 3)).astype(np.float32)
    gradients[:, 2] = -gradients[:, 0] - gradients[:, 1]
    weights = rng.integers(1, 5, size=250)
    weights[bins == 7] *= 20
    native = native_statistic(gradients, bins, weights, groups=4, missing_bin=missing_bin)
    reference = reference_statistic(gradients, bins, weights, groups=4, missing_bin=missing_bin)
    assert native["degrees_of_freedom"] == reference["degrees_of_freedom"]
    np.testing.assert_allclose(native["chi_square"], reference["chi_square"], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("case", ["constant", "zero_mass", "one_mass", "one_bin"])
def test_degenerate_joint_statistic_does_not_reject(case):
    gradients = np.full((40, 3), [0.1, 0.3, -0.4], dtype=np.float32)
    bins = np.tile([0, 1], 20)
    weights = np.full(40, 0.1, dtype=np.float32)
    if case != "constant":
        gradients[::2] *= -1
    if case == "zero_mass":
        weights[:] = 0
    elif case == "one_mass":
        weights[:] = 0
        weights[0] = 1
    elif case == "one_bin":
        bins[:] = 0
    result = native_statistic(gradients, bins, weights)
    assert result["degrees_of_freedom"] == 0
    assert result["chi_square"] == 0
    assert result["p_value"] == 1


def test_response_scale_does_not_act_like_a_covariance_ridge():
    rng = np.random.default_rng(99)
    gradients = rng.normal(size=(100, 3)).astype(np.float32)
    bins = rng.integers(3, size=100)
    baseline = native_statistic(gradients, bins)
    tiny = native_statistic(gradients * np.float32(2.0 ** -60), bins)
    assert tiny["response_rank"] == baseline["response_rank"]
    np.testing.assert_allclose(tiny["chi_square"], baseline["chi_square"], rtol=1e-11)


@pytest.mark.parametrize("case", ["nan_gradient", "inf_weight", "negative_weight", "too_many_classes"])
def test_joint_rejects_invalid_inputs(case):
    gradients = np.ones((10, 3), dtype=np.float32)
    weights = np.ones(10, dtype=np.float32)
    bins = np.arange(10) % 2
    if case == "nan_gradient":
        gradients[0, 0] = np.nan
    elif case == "inf_weight":
        weights[0] = np.inf
    elif case == "negative_weight":
        weights[0] = -1
    else:
        gradients = np.ones((10, 33), dtype=np.float32)
    with pytest.raises(ValueError):
        native_statistic(gradients, bins, weights)


def test_small_enumeration_matches_covariance_but_is_not_an_exact_chi_square_tail():
    result = small_permutation_oracle()
    assert result["permutations"] == 24
    assert result["covariance_max_error"] < 1e-14
    np.testing.assert_allclose(result["exact_permutation_tail"], 2 / 3)
    np.testing.assert_allclose(result["chi_square_tail"], np.exp(-1.5), rtol=1e-12)


def _training_table():
    # Class 0 has the largest gradient variance and favors feature 1. Feature 0
    # carries much more information about classes 1/2, plus enough class-0
    # association to pass the unchanged, single-class cut-gain calculation.
    counts = np.array([[[338, 182], [312, 168]],
                       [[157, 293], [53, 97]],
                       [[10, 20], [130, 240]]])
    rows, labels = [], []
    for label, first, second in itertools.product(range(3), range(2), range(2)):
        count = counts[label, first, second]
        rows.extend([(first, second)] * count)
        labels.extend([label] * count)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def _training_params(**overrides):
    return {"objective": "MultiClass", "num_classes": 3, "iterations": 1,
            "max_depth": 1, "alpha": 0.05, "lambda_l2": 0.0,
            "learning_rate": 0.1, "boost_from_average": False,
            "multiclass_feature_test": "joint", **overrides}


@pytest.mark.parametrize("feature_test", ["quadratic", "grouped"])
def test_joint_training_selects_feature_using_all_class_gradients(feature_test):
    X, y = _training_table()
    gradients = (1 / 3 - np.eye(3)[y.astype(int)]).astype(np.float32)
    assert np.argmax(gradients.astype(float).var(axis=0)) == 0
    single_scores = [reference_statistic(gradients[:, [0]], X[:, feature])["chi_square"]
                     for feature in range(2)]
    joint_scores = [reference_statistic(gradients, X[:, feature])["chi_square"]
                    for feature in range(2)]
    assert single_scores[1] > single_scores[0]
    assert joint_scores[0] > joint_scores[1]
    common = _training_params(feature_test=feature_test, feature_test_bins=4)
    single = ctboost.train(X, {**common, "multiclass_feature_test": "single"}, label=y)
    joint = ctboost.train(X, common, label=y)
    single_root = single._handle.export_state()["trees"][0]["nodes"][0]
    joint_root = joint._handle.export_state()["trees"][0]["nodes"][0]
    assert not single_root["is_leaf"] and single_root["split_feature_id"] == 1
    assert not joint_root["is_leaf"] and joint_root["split_feature_id"] == 0


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
@pytest.mark.parametrize("bootstrap", ["No", "Bernoulli", "Poisson", "Bayesian"])
@pytest.mark.parametrize("grow_policy", ["DepthWise", "LeafWise"])
def test_grouped_joint_training_handles_missing_and_weighted_node_subsets(nan_mode, bootstrap, grow_policy):
    rng = np.random.default_rng(915)
    X = rng.normal(size=(180, 4)).astype(np.float32)
    y = np.argmax(np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]]), axis=1)
    X[::5, 0] = np.nan
    X[::7, 1] = np.nan
    weights = rng.integers(1, 4, len(y)).astype(np.float32)
    weights[::11] = 0.0
    params = _training_params(iterations=4, max_depth=3, feature_test="grouped",
                              feature_test_bins=4, feature_test_adjustment="bonferroni",
                              nan_mode=nan_mode, bootstrap_type=bootstrap,
                              subsample=0.8 if bootstrap in {"Bernoulli", "Poisson"} else 1.0,
                              grow_policy=grow_policy, max_leaves=6,
                              random_seed=91)
    scalar = ctboost.train(X, params, label=y, weight=weights)
    vector = ctboost.train(X, {**params, "multi_strategy": "multi_output_tree"}, label=y, weight=weights)
    assert scalar.multiclass_feature_test == "joint"
    assert np.isfinite(scalar.predict(X)).all()
    assert np.isfinite(scalar.loss_history).all()
    assert any(not tree["nodes"][0]["is_leaf"] for tree in scalar._handle.export_state()["trees"])
    np.testing.assert_array_equal(scalar.predict(X), vector.predict(X))
    np.testing.assert_array_equal(scalar.predict_leaf_index(X)[:, ::3], vector.predict_leaf_index(X))


@pytest.mark.parametrize("bootstrap", ["No", "Bernoulli", "Poisson"])
def test_joint_rejects_fractional_input_before_sampling_or_model_mutation(bootstrap):
    X, y = _training_table()
    params = _training_params(bootstrap_type=bootstrap,
                              subsample=1e-8 if bootstrap != "No" else 1.0)
    model = ctboost.train(X, params, label=y)
    before = model._handle.export_state()
    weights = np.ones(len(y), dtype=np.float32)
    weights[0] = 0.5
    pool = ctboost.Pool(X, y, weight=weights)
    with pytest.raises(ValueError, match="joint feature test requires integer frequency weights"):
        model._handle.fit(pool._handle, continue_training=True)
    assert model._handle.export_state() == before


@pytest.mark.parametrize("temperature", [1e-20, 0.1, 1.0, np.nan, np.inf])
def test_joint_rejects_nonzero_or_nonfinite_bayesian_temperature_before_any_draw(temperature):
    with pytest.raises(ValueError, match="fractional sample/class/bootstrap weights are unsupported"):
        ctboost._core.GradientBooster(**_training_params(
            bootstrap_type="Bayesian", bagging_temperature=temperature))


def test_joint_rejects_fractional_effective_class_weights():
    X, y = _training_table()
    estimator = ctboost.CTBoostClassifier(iterations=1, multiclass_feature_test="joint",
                                        class_weight={0: 0.5, 1: 1, 2: 2})
    with pytest.raises(ValueError, match="joint feature test requires integer frequency weights"):
        estimator.fit(X, y)


def test_full_leaf_solver_still_accepts_fractional_weights_without_joint_test():
    X, y = _training_table()
    weights = np.linspace(0.2, 2.0, len(y), dtype=np.float32)
    model = ctboost.train(X, _training_params(multiclass_feature_test="single",
                                              multiclass_leaf_solver="full",
                                              leaf_estimation_iterations=3,
                                              bootstrap_type="Bayesian",
                                              bagging_temperature=1.0),
                          label=y, weight=weights)
    assert np.isfinite(model.predict(X)).all()
    assert np.isfinite(model.loss_history).all()


@pytest.mark.skipif(not ctboost.build_info()["cuda_enabled"], reason="requires a CUDA-enabled build")
@pytest.mark.parametrize("control", ["leaf_estimation_backtracking", "multiclass_leaf_solver",
                                    "multiclass_feature_test"])
def test_advanced_fit_rechecks_gpu_flag_after_native_load_state(control):
    X, y = _training_table()
    params = _training_params(multiclass_feature_test="single")
    if control == "leaf_estimation_backtracking":
        y = (y == 0).astype(np.float32)
        params.update(objective="Logloss", num_classes=2, leaf_estimation_backtracking=True)
    else:
        params[control] = "full" if control == "multiclass_leaf_solver" else "joint"
    model = ctboost.train(X, params, label=y)
    state = model._handle.export_state()
    state["task_type"] = "GPU"
    model._handle.load_state(state)
    before = model._handle.export_state()
    assert before["task_type"] == "GPU"
    pool = ctboost.Pool(X, y)
    with pytest.raises(ValueError, match="require non-distributed CPU training"):
        model._handle.fit(pool._handle, continue_training=True)
    assert model._handle.export_state() == before


def test_single_class_feature_test_default_preserves_existing_training_exactly():
    X, y = _training_table()
    params = _training_params(iterations=3, max_depth=2)
    del params["multiclass_feature_test"]
    implicit = ctboost.train(X, params, label=y)
    explicit = ctboost.train(X, {**params, "multiclass_feature_test": "single"}, label=y)
    assert implicit._handle.export_state() == explicit._handle.export_state()
    np.testing.assert_array_equal(implicit.predict(X), explicit.predict(X))


def test_joint_training_constant_features_produce_finite_stumps():
    X = np.zeros((60, 3), dtype=np.float32)
    X[:, 1] = np.nan
    y = np.tile([0, 1, 2], 20).astype(np.float32)
    model = ctboost.train(X, _training_params(iterations=3, max_depth=3), label=y)
    assert all(tree["nodes"][0]["is_leaf"] for tree in model._handle.export_state()["trees"])
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize("overrides, message", [
    ({"multiclass_feature_test": "unknown"}, "multiclass_feature_test"),
    ({"task_type": "GPU"}, "CPU"),
    ({"distributed_world_size": 2}, "CPU"),
    ({"objective": "Logloss", "num_classes": 2}, "multiclass objective"),
    ({"num_classes": 33}, "3-32"),
])
def test_joint_training_rejects_unsupported_modes(overrides, message):
    with pytest.raises(ValueError, match=message):
        ctboost._core.GradientBooster(**_training_params(**overrides))


def test_joint_training_rejects_custom_objective_before_callback():
    X, y = _training_table()
    calls = []

    def custom(predictions, label):
        calls.append(True)
        return np.zeros_like(predictions), np.ones_like(predictions)

    estimator = ctboost.CTBoostClassifier(iterations=1, loss_function=custom,
                                        multiclass_feature_test="joint")
    with pytest.raises(ValueError, match="built-in objective"):
        estimator.fit(X, y)
    assert calls == []


def test_joint_training_parameter_survives_clone_persistence_and_resume(tmp_path):
    X, y = _training_table()
    estimator = clone(ctboost.CTBoostClassifier(iterations=2, max_depth=1,
                                                multiclass_feature_test="joint"))
    assert estimator.get_params()["multiclass_feature_test"] == "joint"
    estimator.fit(X, y)
    model = estimator.get_booster()
    path = tmp_path / "joint.ctb"
    model.save_model(path)
    restored = ctboost.load_model(path)
    assert restored.multiclass_feature_test == "joint"
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))

    params = _training_params(iterations=4, random_seed=19)
    reference = ctboost.train(X, params, label=y)
    snapshot = tmp_path / "joint-snapshot.ctb"
    ctboost.train(X, {**params, "iterations": 2}, label=y, snapshot_path=snapshot)
    resumed = ctboost.train(X, params, label=y, snapshot_path=snapshot, resume_from_snapshot=True)
    assert resumed.multiclass_feature_test == "joint"
    np.testing.assert_array_equal(reference.predict(X), resumed.predict(X))
    with pytest.raises(ValueError, match="Use init_model"):
        ctboost.train(X, {**params, "iterations": 5, "multiclass_feature_test": "single"},
                      label=y, snapshot_path=snapshot, resume_from_snapshot=True)
