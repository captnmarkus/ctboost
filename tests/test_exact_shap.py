import math

import numpy as np

import ctboost


def _fit_three_feature_booster():
    rng = np.random.default_rng(20260809)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    y = (
        1.7 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.6 * X[:, 2]
        + X[:, 0] * X[:, 1]
    ).astype(np.float32)
    booster = ctboost.train(
        ctboost.Pool(X, y),
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "lambda_l2": 0.5,
        },
        num_boost_round=6,
    )
    return booster, X


def _coalition_values(booster, row, background):
    feature_count = row.shape[0]
    values = {}
    for mask in range(1 << feature_count):
        hybrid = background.copy()
        for feature in range(feature_count):
            if mask & (1 << feature):
                hybrid[:, feature] = row[feature]
        values[mask] = float(np.mean(booster.predict(hybrid)))
    return values


def _brute_force_shap(values, feature_count):
    result = np.zeros(feature_count + 1, dtype=np.float64)
    result[-1] = values[0]
    for feature in range(feature_count):
        feature_bit = 1 << feature
        for mask in range(1 << feature_count):
            if mask & feature_bit:
                continue
            size = bin(mask).count("1")
            weight = (
                math.factorial(size)
                * math.factorial(feature_count - size - 1)
                / math.factorial(feature_count)
            )
            result[feature] += weight * (
                values[mask | feature_bit] - values[mask]
            )
    return result


def _brute_force_interactions(values, shap_values, feature_count):
    result = np.zeros((feature_count + 1, feature_count + 1), dtype=np.float64)
    result[-1, -1] = values[0]
    for first in range(feature_count):
        for second in range(first + 1, feature_count):
            first_bit = 1 << first
            second_bit = 1 << second
            interaction = 0.0
            for mask in range(1 << feature_count):
                if mask & first_bit or mask & second_bit:
                    continue
                size = bin(mask).count("1")
                weight = (
                    math.factorial(size)
                    * math.factorial(feature_count - size - 2)
                    / (2.0 * math.factorial(feature_count - 1))
                )
                interaction += weight * (
                    values[mask | first_bit | second_bit]
                    - values[mask | first_bit]
                    - values[mask | second_bit]
                    + values[mask]
                )
            result[first, second] = interaction
            result[second, first] = interaction
    for feature in range(feature_count):
        result[feature, feature] = (
            shap_values[feature] - result[feature, :feature_count].sum()
        )
    return result


def test_exact_tree_shap_and_interactions_match_coalition_enumeration():
    booster, X = _fit_three_feature_booster()
    foreground = X[:2]
    background = X[20:24]

    shap_values = booster.predict_shap(foreground, background)
    interaction_values = booster.predict_shap_interactions(foreground, background)

    assert shap_values.shape == (2, X.shape[1] + 1)
    assert interaction_values.shape == (2, X.shape[1] + 1, X.shape[1] + 1)
    for row_index, row in enumerate(foreground):
        coalition_values = _coalition_values(booster, row, background)
        expected_shap = _brute_force_shap(coalition_values, X.shape[1])
        expected_interactions = _brute_force_interactions(
            coalition_values, expected_shap, X.shape[1]
        )
        np.testing.assert_allclose(
            shap_values[row_index], expected_shap, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            interaction_values[row_index],
            expected_interactions,
            rtol=1e-6,
            atol=1e-6,
        )


def test_exact_tree_shap_is_additive_and_honors_background_weights_and_baseline():
    booster, X = _fit_three_feature_booster()
    background_weights = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    background = ctboost.Pool(X[12:16], weight=background_weights)
    prediction_baseline = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    foreground = ctboost.Pool(X[:3], baseline=prediction_baseline)

    shap_values = booster.predict_shap(foreground, background, num_iteration=3)
    interaction_values = booster.predict_shap_interactions(
        foreground, background, num_iteration=3
    )
    predictions = booster.predict(foreground, num_iteration=3)
    expected_value = np.average(
        booster.predict(background, num_iteration=3), weights=background_weights
    )

    np.testing.assert_allclose(shap_values.sum(axis=1), predictions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        interaction_values.sum(axis=(1, 2)), predictions, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        shap_values[:, -1], expected_value + prediction_baseline, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        interaction_values[:, :-1, :-1].sum(axis=2),
        shap_values[:, :-1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_exact_tree_shap_supports_categorical_and_missing_routes():
    X = np.asarray(
        [[0.0, np.nan], [1.0, -1.0], [2.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, np.nan]]
        * 8,
        dtype=np.float32,
    )
    y = (X[:, 0] + np.nan_to_num(X[:, 1])).astype(np.float32)
    train_pool = ctboost.Pool(X, y, cat_features=[0])
    booster = ctboost.train(
        train_pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 3,
            "alpha": 1.0,
            "nan_mode": "Max",
        },
        num_boost_round=5,
    )
    foreground = ctboost.Pool(X[:6], cat_features=[0])
    background = ctboost.Pool(X[7:18], cat_features=[0])

    shap_values = booster.predict_shap(foreground, background)

    np.testing.assert_allclose(
        shap_values.sum(axis=1), booster.predict(foreground), rtol=1e-6, atol=1e-6
    )


def test_multiclass_tree_shap_shapes_and_sklearn_aliases():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(72, 4)).astype(np.float32)
    y = np.argmax(
        np.column_stack([X[:, 0], X[:, 1], -X[:, 0] - X[:, 1]]), axis=1
    )
    model = ctboost.CTBoostClassifier(
        iterations=4,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        random_seed=7,
    ).fit(X, y)

    shap_values = model.predict_shap_values(X[:3], X[20:25])
    interactions = model.predict_shap_interaction_values(X[:3], X[20:25])
    raw_predictions = model.get_booster().predict(X[:3])

    assert shap_values.shape == (3, 3, X.shape[1] + 1)
    assert interactions.shape == (3, 3, X.shape[1] + 1, X.shape[1] + 1)
    np.testing.assert_allclose(
        shap_values.sum(axis=2), raw_predictions, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        interactions.sum(axis=(2, 3)), raw_predictions, rtol=1e-6, atol=1e-6
    )


def test_exact_tree_shap_rejects_empty_background():
    booster, X = _fit_three_feature_booster()
    with np.testing.assert_raises_regex(ValueError, "at least one row"):
        booster.predict_shap(X[:1], np.empty((0, X.shape[1]), dtype=np.float32))


def test_tree_visualization_and_feature_statistics_are_exposed():
    booster, X = _fit_three_feature_booster()
    target = (1.5 * X[:, 0] - X[:, 1]).astype(np.float32)

    dot = booster.tree_to_dot(0)
    statistics = booster.calc_feature_statistics(
        X,
        target,
        feature=[0, 1],
    )

    assert dot.startswith("digraph CTBoostTree")
    assert "n0" in dot
    assert set(statistics) == {"f0", "f1"}
    for item in statistics.values():
        assert item["object_count"].sum() == X.shape[0]
        assert item["mean_prediction"].shape == item["object_count"].shape
        assert item["mean_target"].shape == item["object_count"].shape


def test_sklearn_tree_and_feature_statistics_aliases_use_raw_feature_names():
    rng = np.random.default_rng(53)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 2]).astype(np.float32)
    model = ctboost.CTBoostRegressor(
        iterations=4,
        max_depth=2,
        alpha=1.0,
        random_seed=11,
    ).fit(X, y)

    statistics = model.calc_feature_statistics(X, y, feature=2)

    assert list(statistics) == ["f2"]
    assert "digraph CTBoostTree" in model.tree_to_dot()
