import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from ctboost import CTBoostClassifier, CTBoostRegressor
from ctboost.sklearn import model_selection
from ctboost.sklearn.model_selection import compare_estimators


def _regression_data():
    rng = np.random.default_rng(71)
    X = rng.normal(size=(90, 4)).astype(np.float32)
    y = (2.5 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(scale=0.05, size=90)).astype(np.float32)
    return X, y


def _comparison_regression_data():
    """Return three identical, deterministic folds with a strong signal."""

    row = np.arange(30)
    signal = (row % 10).astype(np.float32) - 4.5
    block = np.column_stack(
        [
            signal,
            (row * 7 % 11).astype(np.float32),
            (row * 3 % 5).astype(np.float32),
            np.ones(30, dtype=np.float32),
        ]
    )
    X = np.tile(block, (3, 1))
    y = np.tile(3.0 * signal, 3).astype(np.float32)
    return X, y


def test_pyplot_loader_is_lazy_and_cached(monkeypatch):
    sentinel = object()
    calls = []

    def fake_import_module(name):
        calls.append(name)
        return sentinel

    model_selection._load_pyplot.cache_clear()
    monkeypatch.setattr(model_selection, "import_module", fake_import_module)

    assert model_selection._require_pyplot("plot=True") is sentinel
    assert model_selection._require_pyplot("plot_metrics") is sentinel
    assert calls == ["matplotlib.pyplot"]
    model_selection._load_pyplot.cache_clear()


def test_pyplot_loader_preserves_actionable_optional_dependency_error(monkeypatch):
    def missing_matplotlib(name):
        raise ImportError(name)

    model_selection._load_pyplot.cache_clear()
    monkeypatch.setattr(model_selection, "import_module", missing_matplotlib)

    with pytest.raises(ImportError, match="plot_metrics requires matplotlib"):
        model_selection._require_pyplot("plot_metrics")
    model_selection._load_pyplot.cache_clear()


def test_grid_search_refits_estimator_and_returns_cv_details():
    X, y = _regression_data()
    model = CTBoostRegressor(iterations=4, max_depth=1, alpha=1.0, random_seed=3)

    result = model.grid_search(
        {"learning_rate": [0.05, 0.2], "lambda_l2": [0.0, 1.0]},
        X,
        y,
        cv=2,
    )

    assert set(result) >= {"params", "best_score", "best_index", "cv_results"}
    assert len(result["cv_results"]["params"]) == 4
    assert model.predict(X).shape == (90,)
    assert model.get_params()["learning_rate"] == result["params"]["learning_rate"]


def test_randomized_search_is_reproducible_and_refits():
    X, y = _regression_data()
    model = CTBoostRegressor(iterations=3, max_depth=1, alpha=1.0, random_seed=5)

    result = model.randomized_search(
        {"learning_rate": [0.03, 0.1, 0.3], "max_depth": [1, 2]},
        X,
        y,
        n_iter=3,
        cv=2,
        random_state=11,
    )

    assert len(result["cv_results"]["params"]) == 3
    assert model.predict(X).shape == (90,)


def test_select_features_ranks_raw_features_and_can_train_selected_model():
    X, y = _regression_data()
    model = CTBoostRegressor(iterations=60, max_depth=3, alpha=1.0, random_seed=7)

    result = model.select_features(
        X,
        y,
        num_features_to_select=2,
        n_repeats=2,
        random_state=13,
        train_final_model=True,
    )

    assert len(result["selected_features"]) == 2
    assert set(result["selected_features"]).isdisjoint(result["eliminated_features"])
    assert model.n_features_in_ == 2
    assert model.predict(X[:, result["selected_features"]]).shape == (90,)


def test_select_features_operates_in_raw_categorical_input_space():
    X = np.asarray(
        [["a", 0.0], ["a", 0.2], ["b", 0.8], ["b", 1.0]] * 8,
        dtype=object,
    )
    y = np.asarray([0, 0, 1, 1] * 8, dtype=np.int64)
    model = CTBoostClassifier(
        iterations=8,
        max_depth=1,
        alpha=1.0,
        cat_features=[0],
        ordered_ctr=True,
        random_seed=17,
    )

    result = model.select_features(
        X,
        y,
        num_features_to_select=1,
        n_repeats=2,
        random_state=19,
    )

    assert len(result["feature_importances"]) == 2
    assert model.predict(X).shape == (32,)


def test_compare_estimators_uses_identical_folds_and_ranks_scores():
    X, y = _comparison_regression_data()
    ctboost_model = CTBoostRegressor(
        iterations=20, max_depth=2, alpha=1.0, random_seed=23
    )
    result = compare_estimators(
        {"constant": DummyRegressor(), "ctboost": ctboost_model},
        X,
        y,
        cv=3,
        scoring="neg_mean_squared_error",
    )

    assert result["primary_metric"] == "score"
    assert result["best_model"] == "ctboost"
    assert [row["name"] for row in result["results"]] == ["ctboost", "constant"]
    assert all(row["metrics"]["score"]["values"].shape == (3,) for row in result["results"])

    via_model = ctboost_model.compare(
        {"constant": DummyRegressor()},
        X,
        y,
        cv=2,
        scoring="r2",
    )
    assert set(row["name"] for row in via_model["results"]) == {"CTBoost", "constant"}


def test_compare_estimators_materializes_one_shot_cv_splits_once():
    X, y = _regression_data()
    indices = np.arange(len(y))
    folds = (
        (indices[indices % 2 != fold], indices[indices % 2 == fold])
        for fold in range(2)
    )

    result = compare_estimators(
        [DummyRegressor(strategy="mean"), DummyRegressor(strategy="median")],
        X,
        y,
        cv=folds,
        scoring="neg_mean_squared_error",
    )

    assert len(result["results"]) == 2
    assert all(row["metrics"]["score"]["values"].shape == (2,) for row in result["results"])


def test_compare_estimator_names_cannot_silently_collide_after_string_conversion():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="unique after string conversion"):
        compare_estimators(
            {1: DummyRegressor(), "1": DummyRegressor()},
            X,
            y,
            cv=2,
        )
