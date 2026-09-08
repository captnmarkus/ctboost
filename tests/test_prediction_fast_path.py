import copy

import numpy as np
import pytest

import ctboost


def _reference_predict_and_leaves(predictor, data, num_iteration):
    rows = np.asarray(data, dtype=np.float32)
    prediction_dimension = predictor.prediction_dimension
    vector_leaves = predictor.multi_strategy == "multi_output_tree"
    trees_per_iteration = 1 if vector_leaves else prediction_dimension
    tree_limit = min(
        len(predictor.trees),
        num_iteration * trees_per_iteration,
    )
    predictions = np.broadcast_to(
        np.asarray(predictor.base_score, dtype=np.float32),
        (rows.shape[0], prediction_dimension),
    ).copy()
    leaf_indices = np.full((rows.shape[0], tree_limit), -1, dtype=np.int32)

    for row_index, row in enumerate(rows):
        bins = [predictor._bin_value(index, value) for index, value in enumerate(row)]
        for tree_index, tree in enumerate(predictor.trees[:tree_limit]):
            nodes = tree["nodes"]
            node_index = 0
            while not nodes[node_index]["is_leaf"]:
                node = nodes[node_index]
                split_bin = bins[int(node["split_feature_id"])]
                if node["is_categorical_split"]:
                    go_left = node["left_categories"][split_bin] != 0
                else:
                    go_left = split_bin <= int(node["split_bin_index"])
                node_index = int(node["left_child"] if go_left else node["right_child"])

            leaf_indices[row_index, tree_index] = node_index
            iteration_index = tree_index // trees_per_iteration
            learning_rate = (
                predictor.tree_learning_rates[iteration_index]
                if iteration_index < len(predictor.tree_learning_rates)
                else predictor.learning_rate
            )
            outputs = range(prediction_dimension) if vector_leaves else [tree_index % prediction_dimension]
            for output in outputs:
                weight = (
                    nodes[node_index]["leaf_weights"][output]
                    if vector_leaves else nodes[node_index]["leaf_weight"]
                )
                predictions[row_index, output] = np.float32(
                    predictions[row_index, output]
                    + np.float32(learning_rate) * np.float32(weight)
                )

    if prediction_dimension == 1:
        predictions = predictions[:, 0]
    return predictions, leaf_indices


def _assert_native_prediction_matches_exported_reference(
    booster, predictor, data, iteration_limits
):
    for num_iteration in iteration_limits:
        reference_prediction, reference_leaves = _reference_predict_and_leaves(
            predictor, data, num_iteration
        )
        actual = np.asarray(booster.predict(data, num_iteration=num_iteration))
        native_reference = np.asarray(booster._handle._predict_uncached(
            ctboost.Pool(data)._handle, num_iteration
        )).reshape(actual.shape)
        np.testing.assert_array_equal(actual.view(np.uint32), native_reference.view(np.uint32))
        np.testing.assert_allclose(actual, reference_prediction, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            booster.predict_leaf_index(data, num_iteration=num_iteration),
            reference_leaves,
        )


def test_legacy_zero_intercept_state_keeps_sub_micro_float_prediction_compatibility(
    tmp_path,
):
    rng = np.random.default_rng(812)
    data = rng.normal(size=(220, 6)).astype(np.float32)
    labels = (2.0 * data[:, 0] - 1.3 * data[:, 1] + 0.4 * data[:, 4]).astype(
        np.float32
    )
    booster = ctboost.train(
        data,
        {
            "objective": "RMSE",
            "boost_from_average": False,
            "learning_rate": 0.123456789,
            "max_depth": 3,
            "alpha": 1.0,
            "random_seed": 37,
        },
        label=labels,
        num_boost_round=80,
    )
    export_path = tmp_path / "legacy_float_reference.json"
    booster.export_model(export_path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(export_path)
    reference, _ = _reference_predict_and_leaves(predictor, data, 80)

    legacy_state = dict(booster._handle.export_state())
    for key in ("boost_from_average", "configured_base_score", "base_score"):
        legacy_state.pop(key)
    restored = ctboost._core.GradientBooster.from_state(legacy_state)
    actual = np.asarray(restored.predict(ctboost.Pool(data)._handle))

    assert restored.base_score() == [0.0]
    assert float(np.max(np.abs(actual - reference))) <= 1.0e-6


def test_compact_prediction_fast_path_matches_exported_multiclass_reference(tmp_path):
    rng = np.random.default_rng(741)
    data = rng.normal(size=(240, 4)).astype(np.float32)
    categories = rng.integers(0, 4, size=data.shape[0]).astype(np.float32)
    categories[::19] = np.nan
    data[:, 1] = categories
    data[::23, 2] = np.nan
    labels = np.mod(
        np.nan_to_num(categories, nan=0.0).astype(np.int64)
        + (data[:, 0] > 0.0).astype(np.int64),
        3,
    ).astype(np.float32)
    booster = ctboost.train(
        ctboost.Pool(data, labels, cat_features=[1]),
        {
            "objective": "MultiClass",
            "num_classes": 3,
            "max_bins": 64,
            "max_depth": 3,
            "alpha": 1.0,
            "random_seed": 12,
        },
        num_boost_round=6,
    )
    export_path = tmp_path / "compact_multiclass.json"
    booster.export_model(export_path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(export_path)

    prediction_data = data[:41].copy()
    prediction_data[0, 1] = 9.0
    prediction_data[1, 1] = np.nan
    prediction_data[2, 2] = np.nan
    _assert_native_prediction_matches_exported_reference(
        booster, predictor, prediction_data, (1, 3, 6)
    )


def test_wide_prediction_fast_path_matches_exported_reference(tmp_path):
    rng = np.random.default_rng(913)
    data = rng.normal(size=(420, 3)).astype(np.float32)
    data[:, 0] = np.linspace(-5.0, 5.0, data.shape[0], dtype=np.float32)
    data[::37, 0] = np.nan
    data[:, 2] = rng.integers(0, 5, size=data.shape[0]).astype(np.float32)
    labels = (
        2.0 * np.nan_to_num(data[:, 0], nan=-5.0)
        - 0.4 * data[:, 1]
        + 0.2 * data[:, 2]
    ).astype(np.float32)
    booster = ctboost.train(
        ctboost.Pool(data, labels, cat_features=[2]),
        {
            "objective": "RMSE",
            "max_bins": 300,
            "max_depth": 3,
            "alpha": 1.0,
            "random_seed": 21,
        },
        num_boost_round=6,
    )
    schema = booster._handle.export_state()["quantization_schema"]
    assert max(schema["num_bins_per_feature"]) > 256
    export_path = tmp_path / "wide_regression.json"
    booster.export_model(export_path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(export_path)

    prediction_data = data[100:149].copy()
    prediction_data[0, 0] = np.nan
    prediction_data[1, 2] = 99.0
    _assert_native_prediction_matches_exported_reference(
        booster, predictor, prediction_data, (1, 4, 6)
    )


def test_external_histogram_prediction_update_keeps_checked_fallback(tmp_path):
    rng = np.random.default_rng(119)
    data = rng.normal(size=(96, 5)).astype(np.float32)
    labels = (1.4 * data[:, 0] - 0.8 * data[:, 3]).astype(np.float32)
    params = {
        "objective": "RMSE",
        "max_bins": 32,
        "max_depth": 3,
        "alpha": 1.0,
        "random_seed": 31,
    }
    in_memory = ctboost.train(
        ctboost.Pool(data, labels),
        params,
        num_boost_round=4,
    )
    external = ctboost.train(
        ctboost.Pool(data, labels),
        {
            **params,
            "external_memory": True,
            "external_memory_dir": str(tmp_path / "native_hist"),
        },
        num_boost_round=4,
    )

    np.testing.assert_allclose(
        external.predict(data), in_memory.predict(data), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        external._handle.export_state()["loss_history"],
        in_memory._handle.export_state()["loss_history"],
        rtol=0.0,
        atol=0.0,
    )


def _routing_state(dimension=1, strategy="one_output_per_tree", wide=False, nan_mode=1):
    """Fixed trees exercise every categorical bitset word without relying on fitting."""
    native = ctboost._core.GradientBooster(
        objective="RMSE" if dimension == 1 else "MultiClass",
        num_classes=dimension,
        multi_strategy=strategy,
    )
    state = native.export_state()
    numeric_cuts = list(np.linspace(-2.0, 2.0, 299 if wide else 3).astype(float))
    category_cuts = list(range(255))
    state["quantization_schema"] = {
        "num_bins_per_feature": [len(numeric_cuts) + 2, 256, 2],
        "cut_offsets": [0, len(numeric_cuts), len(numeric_cuts) + 255, len(numeric_cuts) + 256],
        "cut_values": numeric_cuts + category_cuts + [0.0],
        "categorical_mask": [0, 1, 0],
        "missing_value_mask": [1, 1, 0],
        "nan_mode": nan_mode,
        "nan_modes": [nan_mode, nan_mode, 0],
    }
    vector = strategy == "multi_output_tree"

    def node(leaf, weight, feature=-1, categorical=False):
        result = {
            "is_leaf": leaf,
            "is_categorical_split": categorical,
            "split_feature_id": feature,
            "split_bin_index": len(numeric_cuts) // 2,
            "left_child": -1 if leaf else 1,
            "right_child": -1 if leaf else 2,
            "leaf_weight": weight,
            "left_categories": [
                1 if index in {0, 1, 63, 64, 127, 128, 191, 192, 254, 255} else 0
                for index in range(256)
            ],
        }
        if vector:
            result["leaf_weights"] = [weight * (output + 1.3) for output in range(dimension)]
        return result

    state["trees"] = [
        {
            "nodes": [
                node(False, 0.0, feature=iteration % 2, categorical=iteration % 2 == 1),
                node(True, (output + 0.125) * (iteration + 1)),
                node(True, -(output + 0.375) / (iteration + 1)),
            ],
            "feature_importances": [1.0, 1.0, 0.0],
        }
        for iteration in range(3)
        for output in range(1 if vector else dimension)
    ]
    state["base_score"] = [0.123456789 * (output + 1) for output in range(dimension)]
    state["tree_learning_rates"] = [0.123456789, 0.333333333, 0.071428571]
    state["loss_history"] = [1.0, 0.8, 0.7]
    state["best_iteration"] = 2
    state["best_score"] = 0.7
    return state


def _assert_cached_native_reference(native, data, tmp_path, baseline=None):
    booster = ctboost.Booster(native)
    path = tmp_path / "cache_reference.json"
    booster.export_model(path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(path)
    pool = ctboost.Pool(data, cat_features=[1], baseline=baseline)
    before = native.export_state()
    # Non-monotonic limits also exercise a cache first built for a short prefix.
    for limit in (1, -1, 0, 2, 99, 1):
        reference, leaves = _reference_predict_and_leaves(
            predictor, data, native.num_iterations_trained() if limit < 0 else limit
        )
        if baseline is not None:
            reference = reference + baseline
        actual = np.asarray(native.predict(pool._handle, limit)).reshape(reference.shape)
        # Match the actual legacy helper's compiler arithmetic exactly. GCC
        # FMA builds can contract the legacy product/add across statements.
        native_reference = np.asarray(native._predict_uncached(pool._handle, limit)).reshape(reference.shape)
        np.testing.assert_array_equal(actual.view(np.uint32), native_reference.view(np.uint32))
        np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(
            np.asarray(native.predict_leaf_indices(pool._handle, limit)).reshape(leaves.shape),
            leaves,
        )
    assert native.export_state() == before


@pytest.mark.parametrize("dimension,strategy", [
    (1, "one_output_per_tree"),
    (3, "one_output_per_tree"),
    (3, "multi_output_tree"),
])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("nan_mode", [1, 2])
def test_cached_predictions_preserve_routing_rounding_and_prefixes(
    dimension, strategy, wide, nan_mode, tmp_path
):
    state = _routing_state(dimension, strategy, wide, nan_mode)
    native = ctboost._core.GradientBooster.from_state(state)
    data = np.column_stack([
        np.linspace(-4, 4, 259),
        np.r_[np.arange(255), np.nan, -1, 900, np.nan],
        np.zeros(259),
    ]).astype(np.float32)
    data[::17, 0] = np.nan
    baseline = np.linspace(-0.2, 0.3, len(data) * dimension, dtype=np.float32)
    if dimension > 1:
        baseline = baseline.reshape(len(data), dimension)
    _assert_cached_native_reference(native, data, tmp_path, baseline)

    # An unused forbidden-NaN feature must remain unquantized on warm cache hits.
    with_unused_nan = data.copy()
    with_unused_nan[:, 2] = np.nan
    np.testing.assert_array_equal(
        native.predict(ctboost.Pool(with_unused_nan)._handle),
        native.predict(ctboost.Pool(data)._handle),
    )


@pytest.mark.parametrize("strategy", ["one_output_per_tree", "multi_output_tree"])
def test_cached_state_replacement_truncation_and_schema_reload(strategy, tmp_path):
    state = _routing_state(3, strategy)
    native = ctboost._core.GradientBooster.from_state(state)
    data = np.asarray([[-3, 0, 0], [0, 64, 0], [3, 128, 0]], dtype=np.float32)
    _assert_cached_native_reference(native, data, tmp_path)

    replacement = copy.deepcopy(state)
    replacement["tree_learning_rates"] = [0.9, 0.2, 0.3]
    replacement["base_score"] = [-0.15, 0.04, 0.7]
    for tree in replacement["trees"]:
        tree["nodes"][0]["left_categories"] = [1] * 256
        for node in tree["nodes"][1:]:
            node["leaf_weight"] *= -2
            if "leaf_weights" in node:
                node["leaf_weights"] = [value * -2 for value in node["leaf_weights"]]
    native.load_state(replacement)
    _assert_cached_native_reference(native, data, tmp_path)

    trees_per_round = 1 if strategy == "multi_output_tree" else 3
    replacement["trees"] = replacement["trees"][:trees_per_round]
    replacement["tree_learning_rates"] = replacement["tree_learning_rates"][:1]
    replacement["loss_history"] = replacement["loss_history"][:1]
    replacement["best_iteration"] = 0
    native.load_state(replacement)
    _assert_cached_native_reference(native, data, tmp_path)

    schema = copy.deepcopy(replacement["quantization_schema"])
    schema["cut_values"][:3] = [-20, -10, -5]
    native.load_quantization_schema(schema)
    _assert_cached_native_reference(native, data, tmp_path)

    # A larger replacement schema must refresh the cached feature mask too.
    expected = native.predict(ctboost.Pool(data)._handle)
    schema["num_bins_per_feature"].append(2)
    schema["cut_offsets"].append(len(schema["cut_values"]) + 1)
    schema["cut_values"].append(0.0)
    schema["categorical_mask"].append(0)
    schema["missing_value_mask"].append(0)
    schema["nan_modes"].append(0)
    native.load_quantization_schema(schema)
    extended = np.column_stack([data, np.full(len(data), np.nan)]).astype(np.float32)
    np.testing.assert_array_equal(native.predict(ctboost.Pool(extended)._handle), expected)


def test_cached_prefix_does_not_quantize_features_first_used_later():
    state = _routing_state()
    state["quantization_schema"]["nan_modes"][1] = 0
    native = ctboost._core.GradientBooster.from_state(state)
    data = np.asarray([[0.5, np.nan, 0.0]], dtype=np.float32)
    pool = ctboost.Pool(data)._handle
    for _ in range(2):
        assert np.isfinite(native.predict(pool, 1)).all()
        with pytest.raises(ValueError, match="NaN values are not allowed"):
            native.predict(pool, 2)


@pytest.mark.parametrize("difference", [
    "numeric_threshold", "numeric_feature", "numeric_children",
    "categorical_routes", "categorical_children", "constant_tree",
])
def test_scalar_multiclass_cache_keeps_distinct_class_topologies(difference, tmp_path):
    state = _routing_state(3)
    native = ctboost._core.GradientBooster.from_state(state)
    data = np.asarray([
        [-3, 0, 1], [-1, 63, -1], [0, 64, 0], [1, 191, 2], [3, 254, -2],
    ], dtype=np.float32)
    native.predict(ctboost.Pool(data)._handle)  # Prime a previously fuseable model.
    index = 4 if difference.startswith("categorical") else 1
    tree = state["trees"][index]
    root = tree["nodes"][0]
    if difference == "numeric_threshold":
        root["split_bin_index"] = 2
    elif difference == "numeric_feature":
        root["split_feature_id"] = 2
        root["split_bin_index"] = 0
    elif difference.endswith("children"):
        root["left_child"], root["right_child"] = root["right_child"], root["left_child"]
    elif difference == "categorical_routes":
        root["left_categories"][64] ^= 1
    else:
        tree["nodes"] = [tree["nodes"][1]]
    native.load_state(state)
    _assert_cached_native_reference(native, data, tmp_path)


@pytest.mark.parametrize("dimension", [4, 5])
def test_scalar_multiclass_cache_preserves_arbitrary_class_group_widths(dimension, tmp_path):
    native = ctboost._core.GradientBooster.from_state(_routing_state(dimension))
    data = np.asarray([[-3, 0, 0], [0, 63, 0], [3, 254, 0]], dtype=np.float32)
    _assert_cached_native_reference(native, data, tmp_path)


@pytest.mark.parametrize("num_rows", [0, 1, 3, 7, 8, 9, 63, 64, 65, 1000])
@pytest.mark.parametrize("dimension,strategy", [
    (1, "one_output_per_tree"),
    (3, "one_output_per_tree"),
    (3, "multi_output_tree"),
])
def test_prediction_preserves_mixed_depth_routes_at_batch_boundaries(
    num_rows, dimension, strategy, tmp_path
):
    state = _routing_state(dimension, strategy, wide=num_rows % 2 == 1)
    for tree in state["trees"]:
        root, left, right = tree["nodes"]
        branch = copy.deepcopy(root)
        branch["split_feature_id"] = 1 - root["split_feature_id"]
        branch["is_categorical_split"] = branch["split_feature_id"] == 1
        branch["split_bin_index"] = 1
        branch["left_child"], branch["right_child"] = 3, 4
        deeper = copy.deepcopy(root)
        deeper["left_child"], deeper["right_child"] = 5, 6
        other_leaf = copy.deepcopy(right)
        other_leaf["leaf_weight"] *= 0.37
        if "leaf_weights" in other_leaf:
            other_leaf["leaf_weights"] = [value * 0.37 for value in other_leaf["leaf_weights"]]
        tree["nodes"] = [root, branch, right, left, deeper, other_leaf, copy.deepcopy(left)]
    native = ctboost._core.GradientBooster.from_state(state)
    rng = np.random.default_rng(237)
    data = np.column_stack([
        rng.uniform(-3, 3, num_rows),
        rng.choice([0, 1, 63, 64, 127, 191, 254, np.nan], num_rows),
        np.zeros(num_rows),
    ]).astype(np.float32)
    data[::11, 0] = np.nan
    _assert_cached_native_reference(native, data, tmp_path)


@pytest.mark.parametrize("num_rows", [1, 65, 1000])
@pytest.mark.parametrize("dimension,strategy", [
    (1, "one_output_per_tree"),
    (3, "one_output_per_tree"),
    (3, "multi_output_tree"),
])
def test_root_only_updates_preserve_zero_baseline_and_tree_rounding_order(
    num_rows, dimension, strategy, tmp_path
):
    state = _routing_state(dimension, strategy)
    vector = strategy == "multi_output_tree"
    width = 1 if vector else dimension
    zero_trees = []
    for iteration in (0, 2):
        for output in range(width):
            tree = state["trees"][iteration * width + output]
            leaf = copy.deepcopy(tree["nodes"][1])
            sign = 1 if iteration == 0 else -1
            leaf["leaf_weight"] = float(sign * 2**24 * (output + 1))
            if vector:
                leaf["leaf_weights"] = [float(sign * 2**24 * (index + 1)) for index in range(dimension)]
            tree["nodes"] = [leaf]
            if iteration == 2:
                zero_tree = copy.deepcopy(tree)
                zero_leaf = zero_tree["nodes"][0]
                zero_leaf["leaf_weight"] = -0.0 if output % 2 == 0 else 0.0
                if vector:
                    zero_leaf["leaf_weights"] = [-0.0 if index % 2 == 0 else 0.0 for index in range(dimension)]
                zero_trees.append(zero_tree)
    if not vector and dimension > 1:
        # Keep an unfused iteration with one constant class and other split trees.
        state["trees"][width + 1]["nodes"] = [state["trees"][width + 1]["nodes"][1]]
    state["trees"].extend(zero_trees)
    state["base_score"] = [-0.17 * (index + 1) for index in range(dimension)]
    state["tree_learning_rates"] = [1.0] * 4
    state["loss_history"] = [1.0] * 4
    state["best_iteration"] = 3
    native = ctboost._core.GradientBooster.from_state(state)
    data = np.column_stack([
        np.linspace(-3, 3, num_rows), np.arange(num_rows) % 255, np.zeros(num_rows),
    ]).astype(np.float32)
    baseline = np.linspace(-0.7, -0.1, num_rows * dimension, dtype=np.float32)
    if dimension > 1:
        baseline = baseline.reshape(num_rows, dimension)
    _assert_cached_native_reference(native, data, tmp_path, baseline)


@pytest.mark.parametrize("boosting_type", ["Plain", "DART"])
@pytest.mark.parametrize("dimension,strategy", [
    (1, "one_output_per_tree"),
    (3, "one_output_per_tree"),
    (3, "multi_output_tree"),
])
def test_cached_model_warm_start_and_refit_match_uncached_training(
    boosting_type, dimension, strategy
):
    rng = np.random.default_rng(781)
    data = rng.normal(size=(96, 4)).astype(np.float32)
    label = (data[:, 0] - data[:, 1] if dimension == 1 else np.argmax(data[:, :3], axis=1)).astype(np.float32)
    native = ctboost._core.GradientBooster(
        objective="RMSE" if dimension == 1 else "MultiClass",
        num_classes=dimension, multi_strategy=strategy, iterations=3,
        max_depth=2, max_bins=16, alpha=1.0, random_seed=42,
        boosting_type=boosting_type, skip_drop=0.0, drop_rate=0.7,
    )
    native.fit(ctboost.Pool(data, label)._handle)
    query = ctboost.Pool(data)._handle
    for continuation in (True, False):
        native.set_learning_rate(0.213456789)
        native.predict(query)  # Populate the cache before the model is changed.
        control = ctboost._core.GradientBooster.from_state(native.export_state())
        for model in (native, control):
            model.fit(ctboost.Pool(data, label)._handle, continue_training=continuation)
        np.testing.assert_array_equal(native.predict(query), control.predict(query))
        assert native.export_state() == control.export_state()


@pytest.mark.parametrize("interrupt", [False, True])
def test_custom_objective_predictions_never_reuse_a_partial_fit_cache(interrupt):
    rng = np.random.default_rng(74)
    data = rng.normal(size=(64, 3)).astype(np.float32)
    label = data[:, 0] - data[:, 1]
    native = ctboost._core.GradientBooster(
        iterations=4, max_depth=2, alpha=1.0, boosting_type="DART",
        drop_rate=0.7, skip_drop=0.0,
    )
    query = ctboost.Pool(data)._handle
    calls = []

    def objective(prediction, target):
        control = ctboost._core.GradientBooster.from_state(native.export_state())
        np.testing.assert_array_equal(native.predict(query), control.predict(query))
        calls.append(native.num_trees())
        if interrupt and len(calls) == 4:
            raise RuntimeError("intentional objective interruption")
        return prediction - target, np.ones_like(prediction)

    if interrupt:
        with pytest.raises(RuntimeError, match="intentional objective interruption"):
            native.fit_custom_objective(ctboost.Pool(data, label)._handle, objective)
    else:
        native.fit_custom_objective(ctboost.Pool(data, label)._handle, objective)
    assert calls == [0, 1, 2, 3]
    control = ctboost._core.GradientBooster.from_state(native.export_state())
    np.testing.assert_array_equal(native.predict(query), control.predict(query))


@pytest.mark.parametrize("dimension,strategy", [
    (1, "one_output_per_tree"),
    (3, "one_output_per_tree"),
    (3, "multi_output_tree"),
])
def test_prediction_matches_legacy_compiler_math_on_fma_sensitive_values(dimension, strategy):
    state = _routing_state(dimension, strategy)
    # This finite triple differs by two float ULPs when multiply/add contracts.
    initial = np.asarray([0xbfd9d812], dtype=np.uint32).view(np.float32)[0]
    weight = np.asarray([0x412277bf], dtype=np.uint32).view(np.float32)[0]
    state["base_score"] = [float(initial)] * dimension
    state["tree_learning_rates"] = [0.123456789] * 3
    for tree in state["trees"]:
        for node in tree["nodes"]:
            if node["is_leaf"]:
                node["leaf_weight"] = float(weight)
                if "leaf_weights" in node:
                    node["leaf_weights"] = [float(weight)] * dimension
    native = ctboost._core.GradientBooster.from_state(state)
    pool = ctboost.Pool(np.asarray([[-3, 0, 0], [3, 128, 0]], dtype=np.float32))._handle
    before = native.export_state()
    for limit in (1, -1, 2, 0, 1):
        expected = np.asarray(native._predict_uncached(pool, limit))
        actual = np.asarray(native.predict(pool, limit))
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert native.export_state() == before
