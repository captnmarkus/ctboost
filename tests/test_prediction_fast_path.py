import numpy as np

import ctboost


def _reference_predict_and_leaves(predictor, data, num_iteration):
    rows = np.asarray(data, dtype=np.float32)
    prediction_dimension = predictor.prediction_dimension
    tree_limit = min(
        len(predictor.trees),
        num_iteration * prediction_dimension,
    )
    predictions = np.zeros((rows.shape[0], prediction_dimension), dtype=np.float32)
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
            iteration_index = tree_index // prediction_dimension
            learning_rate = (
                predictor.tree_learning_rates[iteration_index]
                if iteration_index < len(predictor.tree_learning_rates)
                else predictor.learning_rate
            )
            class_index = tree_index % prediction_dimension
            predictions[row_index, class_index] = np.float32(
                predictions[row_index, class_index]
                + learning_rate * float(nodes[node_index]["leaf_weight"])
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
        np.testing.assert_allclose(
            booster.predict(data, num_iteration=num_iteration),
            reference_prediction,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            booster.predict_leaf_index(data, num_iteration=num_iteration),
            reference_leaves,
        )


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
