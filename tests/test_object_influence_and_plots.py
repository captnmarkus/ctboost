import numpy as np
import pytest

import ctboost


def _regression_booster(iterations=6):
    rng = np.random.default_rng(441)
    X = rng.normal(size=(72, 4)).astype(np.float32)
    y = (1.7 * X[:, 0] - 0.8 * X[:, 1] + 0.2 * X[:, 2]).astype(np.float32)
    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "iterations": iterations,
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
            "random_seed": 17,
        },
    )
    return booster, pool, X, y


def test_leaf_influence_is_a_signed_tree_output_decomposition():
    booster, train_pool, X, _ = _regression_booster()

    influence, coverage = booster.calc_leaf_influence(
        X[:7], train_pool, return_coverage=True
    )

    assert influence.shape == (7, X.shape[0])
    np.testing.assert_allclose(coverage, 1.0)
    np.testing.assert_allclose(
        influence.sum(axis=1),
        booster.predict(X[:7]) - booster.base_score[0],
        rtol=1e-6,
        atol=1e-6,
    )

    partial = booster.calc_leaf_influence(X[:7], train_pool, num_iteration=3)
    np.testing.assert_allclose(
        partial.sum(axis=1),
        booster.predict(X[:7], num_iteration=3) - booster.base_score[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_leaf_influence_distributes_by_reference_pool_weight():
    booster, _, X, y = _regression_booster(iterations=1)
    weights = np.linspace(0.5, 2.0, X.shape[0], dtype=np.float32)
    weighted_reference = ctboost.Pool(X, y, weight=weights)
    influence = booster.calc_leaf_influence(X[:1], weighted_reference)[0]
    leaf_indices = booster.predict_leaf_index(X)
    same_leaf = np.flatnonzero(leaf_indices[:, 0] == leaf_indices[0, 0])

    assert np.count_nonzero(influence) == same_leaf.size
    ratios = influence[same_leaf] / weights[same_leaf]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        influence.sum(), booster.predict(X[:1])[0] - booster.base_score[0], rtol=1e-6
    )


def test_ranked_object_importance_matches_leaf_scores_and_is_deterministic():
    booster, train_pool, X, _ = _regression_booster()
    influence = booster.calc_leaf_influence(X[:4], train_pool)

    per_indices, per_scores = booster.get_object_importance(
        X[:4], train_pool, top_size=5, importance_type="PerObject"
    )
    assert per_indices.shape == per_scores.shape == (4, 5)
    np.testing.assert_allclose(
        per_scores, np.take_along_axis(influence, per_indices, axis=1)
    )
    assert np.all(np.abs(per_scores[:, :-1]) >= np.abs(per_scores[:, 1:]))

    average_indices, average_scores = booster.get_object_importance(
        X[:4], train_pool, top_size=6, importance_type="Average"
    )
    expected_average = influence.mean(axis=0)
    np.testing.assert_allclose(average_scores, expected_average[average_indices])
    assert np.all(np.abs(average_scores[:-1]) >= np.abs(average_scores[1:]))


def test_multiclass_leaf_influence_preserves_output_dimension():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(75, 5)).astype(np.float32)
    y = np.argmax(X[:, :3], axis=1).astype(np.float32)
    pool = ctboost.Pool(X, y)
    booster = ctboost.train(
        pool,
        {
            "objective": "MultiClass",
            "num_classes": 3,
            "iterations": 4,
            "max_depth": 2,
            "alpha": 1.0,
            "random_seed": 9,
        },
    )

    influence, coverage = booster.calc_leaf_influence(
        X[:3], pool, return_coverage=True
    )
    assert influence.shape == (3, 3, X.shape[0])
    assert coverage.shape == (3, 3)
    np.testing.assert_allclose(
        influence.sum(axis=2), booster.predict(X[:3]) - booster.base_score, rtol=1e-6
    )
    with pytest.raises(ValueError, match="prediction_dimension is required"):
        booster.get_object_importance(X[:2], pool, top_size=3)
    indices, scores = booster.get_object_importance(
        X[:2], pool, top_size=3, prediction_dimension=1
    )
    assert indices.shape == scores.shape == (3,)


def test_leaf_influence_reports_reference_coverage_and_validates_inputs():
    booster, train_pool, X, _ = _regression_booster()
    _, coverage = booster.calc_leaf_influence(
        X, X[:1], return_coverage=True
    )
    assert np.all((coverage >= 0.0) & (coverage <= 1.0))
    assert np.any(coverage < 1.0)

    with pytest.raises(ValueError, match="at least one row"):
        booster.calc_leaf_influence(X[:1], np.empty((0, X.shape[1]), dtype=np.float32))
    with pytest.raises(ValueError, match="same transformed feature count"):
        booster.calc_leaf_influence(X[:1], X[:, :-1])
    with pytest.raises(ValueError, match="top_size"):
        booster.get_object_importance(X[:1], train_pool, top_size=0)
    with pytest.raises(ValueError, match="importance_type"):
        booster.get_object_importance(
            X[:1], train_pool, importance_type="ExactRefit"
        )


def test_prediction_and_feature_statistics_plotting_conveniences():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    booster, _, X, y = _regression_booster()
    prediction_axis = booster.plot_predictions(X, y)
    residual_axis = booster.plot_predictions(X, y, kind="residual")
    sequence_axis = booster.plot_predictions(X, kind="prediction")
    statistics_axis = booster.plot_feature_statistics(X, y, feature=0)

    assert prediction_axis.get_title() == "CTBoost actual vs predicted"
    assert residual_axis.get_title() == "CTBoost residuals"
    assert sequence_axis.get_title() == "CTBoost predictions"
    assert statistics_axis.get_title() == "CTBoost feature statistics: f0"
    assert len(statistics_axis.figure.axes) == 2  # primary curve plus object-count axis
    with pytest.raises(ValueError, match="requires target"):
        booster.plot_predictions(X, kind="residual")
    with pytest.raises(TypeError, match="numeric target"):
        booster.plot_predictions(X, np.asarray(["label"] * X.shape[0]))
    plt.close("all")


def test_sklearn_object_influence_and_plot_aliases():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _, _, X, y = _regression_booster()
    model = ctboost.CTBoostRegressor(
        iterations=4,
        max_depth=2,
        alpha=1.0,
        random_seed=3,
    ).fit(X, y)

    influence = model.calc_leaf_influence(X[:2], X)
    indices, scores = model.get_object_importance(
        X[:2], X, top_size=4, importance_type="PerObject"
    )
    assert influence.shape == (2, X.shape[0])
    assert indices.shape == scores.shape == (2, 4)
    assert model.plot_predictions(X, y).get_title() == "CTBoost actual vs predicted"
    assert (
        model.plot_feature_statistics(X, y, feature=1).get_title()
        == "CTBoost feature statistics: f1"
    )
    plt.close("all")
