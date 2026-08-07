import json
import pickle
import typing

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError
from sklearn.base import clone, is_classifier, is_regressor

import ctboost
from ctboost.sklearn._fit import _BaseFitMixin


def _classifier_params():
    return {
        "iterations": 8,
        "learning_rate": 0.2,
        "max_depth": 2,
        "alpha": 1.0,
        "random_seed": 17,
    }


def test_estimator_mixins_report_sklearn_estimator_types():
    classifier = ctboost.CTBoostClassifier()
    regressor = ctboost.CTBoostRegressor()
    assert is_classifier(classifier)
    assert not is_regressor(classifier)
    assert is_regressor(regressor)
    assert not is_classifier(regressor)
    assert classifier._estimator_type == "classifier"
    assert regressor._estimator_type == "regressor"


def test_classifier_pool_labels_and_explicit_y_are_encoded():
    X, y = make_classification(
        n_samples=128,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=23,
    )
    X = X.astype(np.float32)
    arbitrary_labels = np.where(y == 0, 2.0, 4.0).astype(np.float32)

    reference = ctboost.CTBoostClassifier(**_classifier_params()).fit(X, arbitrary_labels)
    pool = ctboost.Pool(
        X,
        arbitrary_labels,
        feature_names=[f"f{index}" for index in range(X.shape[1])],
    )
    from_pool = ctboost.CTBoostClassifier(**_classifier_params()).fit(pool)

    np.testing.assert_allclose(
        from_pool.predict_proba(X),
        reference.predict_proba(X),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(from_pool.classes_, np.asarray([2.0, 4.0], dtype=np.float32))
    assert from_pool.data_schema_["feature_names"] == [f"f{index}" for index in range(X.shape[1])]

    string_labels = np.where(y == 0, "negative", "positive")
    misleading_pool = ctboost.Pool(X, 1 - y)
    explicit_y = ctboost.CTBoostClassifier(**_classifier_params()).fit(
        misleading_pool,
        string_labels,
    )
    explicit_reference = ctboost.CTBoostClassifier(**_classifier_params()).fit(X, string_labels)
    np.testing.assert_array_equal(explicit_y.predict(X), explicit_reference.predict(X))

    multiclass_X = np.arange(180, dtype=np.float32).reshape(60, 3)
    multiclass_y = np.repeat(np.asarray([10.0, 20.0, 30.0], dtype=np.float32), 20)
    multiclass = ctboost.CTBoostClassifier(
        iterations=2,
        alpha=1.0,
        random_seed=3,
    ).fit(ctboost.Pool(multiclass_X, multiclass_y))
    np.testing.assert_array_equal(multiclass.classes_, np.asarray([10.0, 20.0, 30.0]))
    assert multiclass.predict_proba(multiclass_X).shape == (60, 3)


def test_classifier_binary_sigmoid_is_stable_for_extreme_scores():
    scores = np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32)
    with np.errstate(over="raise", invalid="raise"):
        probabilities = ctboost.CTBoostClassifier._sigmoid(scores)
    np.testing.assert_allclose(probabilities, np.asarray([0.0, 0.5, 1.0], dtype=np.float32))


def test_ranker_feature_pipeline_accepts_raw_dataframe_and_eval_set():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "a", "b"] * 3,
            "value": np.linspace(-1.0, 1.0, 12, dtype=np.float32),
        }
    )
    target = np.asarray([3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0] + [2.0, 3.0, 0.0, 1.0], dtype=np.float32)
    group_id = np.repeat(np.arange(3), 4)

    ranker = ctboost.CTBoostRanker(
        iterations=3,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
        ordered_ctr=True,
        random_seed=11,
    ).fit(
        frame,
        target,
        group_id=group_id,
        eval_set=(frame, target, group_id),
    )

    assert ranker.predict(frame).shape == (len(frame),)
    assert ranker.n_features_in_ == frame.shape[1]
    assert ranker.n_transformed_features_ >= frame.shape[1]
    np.testing.assert_array_equal(ranker.feature_names_in_, np.asarray(frame.columns, dtype=object))


def test_prediction_validates_fitted_state_shape_count_and_feature_names():
    pd = pytest.importorskip("pandas")
    X, y = make_regression(n_samples=64, n_features=3, random_state=29)
    frame = pd.DataFrame(X.astype(np.float32), columns=["a", "b", "c"])
    regressor = ctboost.CTBoostRegressor(
        iterations=3,
        alpha=1.0,
        random_seed=7,
    ).fit(frame, y.astype(np.float32))

    assert regressor.n_features_in_ == 3
    assert regressor.n_transformed_features_ == 3
    np.testing.assert_array_equal(
        regressor.feature_names_in_,
        np.asarray(["a", "b", "c"], dtype=object),
    )

    with pytest.raises(ValueError, match="feature names and order"):
        regressor.predict(frame[["b", "a", "c"]])
    mismatched_pool = ctboost.Pool(
        frame.to_numpy(dtype=np.float32),
        np.zeros(len(frame), dtype=np.float32),
        feature_names=["b", "a", "c"],
    )
    with pytest.raises(ValueError, match="feature names and order"):
        regressor.predict(mismatched_pool)
    with pytest.raises(ValueError, match="has 4 features"):
        regressor.predict(np.column_stack([frame.to_numpy(), np.ones(len(frame))]))
    with pytest.raises(ValueError, match="2D feature matrix"):
        regressor.predict(frame.to_numpy()[0])
    with pytest.raises(NotFittedError):
        ctboost.CTBoostRegressor().predict(frame)


def test_raw_and_transformed_feature_metadata_round_trip_and_load_legacy_json(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "text": ["red quick fox", "blue slow fox"] * 24,
            "numeric": np.linspace(-1.0, 1.0, 48, dtype=np.float32),
        }
    )
    target = np.linspace(0.0, 4.0, 48, dtype=np.float32)
    regressor = ctboost.CTBoostRegressor(
        iterations=3,
        alpha=1.0,
        text_features=["text"],
        text_hash_dim=8,
        random_seed=13,
    ).fit(frame, target)

    assert regressor.n_features_in_ == 2
    assert regressor.n_transformed_features_ == 9
    np.testing.assert_array_equal(
        regressor.feature_names_in_,
        np.asarray(["text", "numeric"], dtype=object),
    )

    model_path = tmp_path / "metadata.json"
    regressor.save_model(model_path)
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    assert restored.n_features_in_ == regressor.n_features_in_
    assert restored.n_transformed_features_ == regressor.n_transformed_features_
    np.testing.assert_array_equal(restored.feature_names_in_, regressor.feature_names_in_)
    np.testing.assert_allclose(restored.predict(frame), regressor.predict(frame), rtol=1e-6, atol=1e-6)

    legacy_path = tmp_path / "legacy_metadata.json"
    document = json.loads(model_path.read_text(encoding="utf-8"))
    fitted_state = document["fitted_state"]
    fitted_state["n_features_in_"] = regressor.n_transformed_features_
    fitted_state.pop("n_transformed_features_")
    fitted_state.pop("feature_names_in_")
    fitted_state["best_score_"] = {
        "__ctboost_type__": "dict",
        "items": [["learn", 123.0]],
    }
    legacy_path.write_text(json.dumps(document), encoding="utf-8")

    legacy = ctboost.CTBoostRegressor.load_model(legacy_path)
    assert legacy.n_features_in_ == regressor.n_features_in_
    assert legacy.n_transformed_features_ == regressor.n_transformed_features_
    np.testing.assert_array_equal(legacy.feature_names_in_, regressor.feature_names_in_)
    assert all(isinstance(scores, dict) for scores in legacy.best_score_.values())
    np.testing.assert_allclose(legacy.predict(frame), regressor.predict(frame), rtol=1e-6, atol=1e-6)


def test_estimator_compatibility_getters_and_leaf_aliases_return_copies():
    X, y = make_regression(n_samples=72, n_features=4, random_state=31)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    regressor = ctboost.CTBoostRegressor(iterations=4, alpha=1.0, random_seed=5)

    assert not regressor.is_fitted()
    with pytest.raises(NotFittedError):
        regressor.get_booster()

    regressor.fit(X[:48], y[:48], eval_set=(X[48:], y[48:]))
    assert regressor.is_fitted()
    assert regressor.get_booster() is regressor._booster
    assert regressor.get_best_iteration() == regressor.best_iteration_
    assert regressor.get_best_score() == regressor.best_score_
    assert regressor.get_evals_result() == regressor.evals_result_
    assert regressor.evals_result() == regressor.evals_result_

    copied_history = regressor.get_evals_result()
    copied_history.clear()
    assert regressor.evals_result_

    leaf_indices = regressor.predict_leaf_index(X)
    np.testing.assert_array_equal(regressor.apply(X), leaf_indices)
    np.testing.assert_array_equal(regressor.calc_leaf_indexes(X), leaf_indices)


def test_best_score_reports_each_metrics_actual_best_value():
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=37,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    classifier = ctboost.CTBoostClassifier(
        iterations=20,
        alpha=1.0,
        eval_metric=["Logloss", "AUC"],
        random_seed=9,
    ).fit(X[:100], y[:100], eval_set=(X[100:], y[100:]))

    history = classifier.evals_result_["validation"]
    best_scores = classifier.best_score_["validation"]
    assert best_scores["Logloss"] == pytest.approx(min(history["Logloss"]))
    assert best_scores["AUC"] == pytest.approx(max(history["AUC"]))


def test_xgboost_and_catboost_constructor_aliases_clone_and_set_params():
    regressor = ctboost.CTBoostRegressor(
        n_estimators=7,
        depth=3,
        reg_lambda=2.5,
        random_state=19,
    )
    assert regressor.iterations == 7
    assert regressor.max_depth == 3
    assert regressor.lambda_l2 == 2.5
    assert regressor.random_seed == 19

    cloned = clone(regressor)
    assert cloned.get_params()["n_estimators"] == 7
    assert cloned.iterations == 7
    cloned.set_params(n_estimators=5, depth=2, l2_leaf_reg=1.5, random_state=23)
    assert cloned.iterations == 5
    assert cloned.max_depth == 2
    assert cloned.lambda_l2 == 1.5
    assert cloned.random_seed == 23

    with pytest.raises(ValueError, match="conflicts"):
        ctboost.CTBoostRegressor(iterations=8, n_estimators=7)
    with pytest.raises(ValueError, match="conflicting aliases"):
        ctboost.CTBoostRegressor(reg_lambda=2.0, l2_leaf_reg=3.0)


def test_constructor_aliases_remain_cloneable_after_pickle_round_trip(tmp_path):
    X, y = make_regression(n_samples=40, n_features=3, random_state=41)
    regressor = ctboost.CTBoostRegressor(
        n_estimators=3,
        depth=2,
        reg_lambda=2.5,
        random_state=1003,
        alpha=1.0,
    ).fit(X.astype(np.float32), y.astype(np.float32))

    model_path = tmp_path / "aliases.pkl"
    regressor.save_model(model_path)
    restored_models = [
        pickle.loads(pickle.dumps(regressor)),
        ctboost.CTBoostRegressor.load_model(model_path),
    ]
    for restored in restored_models:
        assert restored.lambda_l2 is restored.reg_lambda
        assert restored.random_seed is restored.random_state
        cloned = clone(restored)
        assert cloned.lambda_l2 == 2.5
        assert cloned.random_seed == 1003


def test_legacy_pickle_without_alias_attributes_migrates_and_clones():
    X, y = make_regression(n_samples=32, n_features=2, random_state=47)
    regressor = ctboost.CTBoostRegressor(
        iterations=3,
        max_depth=2,
        lambda_l2=2.0,
        random_seed=1007,
        alpha=1.0,
    ).fit(X.astype(np.float32), y.astype(np.float32))
    for alias_name in ("n_estimators", "depth", "reg_lambda", "l2_leaf_reg", "random_state"):
        delattr(regressor, alias_name)

    restored = pickle.loads(pickle.dumps(regressor))
    params = restored.get_params(deep=False)
    assert all(params[name] is None for name in (
        "n_estimators", "depth", "reg_lambda", "l2_leaf_reg", "random_state"
    ))
    cloned = clone(restored)
    assert cloned.iterations == 3
    assert cloned.max_depth == 2
    assert cloned.lambda_l2 == 2.0
    assert cloned.random_seed == 1007


def test_pipeline_estimator_rejects_unprepared_pool_and_accepts_tagged_pool():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "category": ["z", "a", "m", "z", "m", "a"] * 8,
            "value": np.linspace(-1.0, 1.0, 48, dtype=np.float32),
        }
    )
    target = (
        frame["value"].to_numpy(dtype=np.float32)
        + (frame["category"] == "z").to_numpy(dtype=np.float32)
    )
    regressor = ctboost.CTBoostRegressor(
        iterations=5,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
        random_seed=17,
    ).fit(frame, target)

    with pytest.raises(ValueError, match="fitted feature pipeline"):
        regressor.predict(ctboost.Pool(frame))

    prepared_pool = regressor._feature_pipeline.transform_pool(frame)
    np.testing.assert_allclose(
        regressor.predict(prepared_pool),
        regressor.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )
    equivalent_pipeline = ctboost.FeaturePipeline.from_state(
        regressor._feature_pipeline.to_state()
    )
    equivalent_pool = equivalent_pipeline.transform_pool(frame)
    np.testing.assert_allclose(
        regressor.predict(equivalent_pool),
        regressor.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        regressor.get_booster().predict(frame),
        regressor.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )


def test_loaded_pipeline_estimator_booster_predicts_raw_data(tmp_path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "category": ["left", "right"] * 24,
            "value": np.linspace(-2.0, 2.0, 48, dtype=np.float32),
        }
    )
    target = frame["value"].to_numpy(dtype=np.float32)
    regressor = ctboost.CTBoostRegressor(
        iterations=4,
        alpha=1.0,
        cat_features=["category"],
        ordered_ctr=True,
        random_seed=23,
    ).fit(frame, target)

    model_path = tmp_path / "pipeline_estimator.json"
    regressor.save_model(model_path)
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    np.testing.assert_allclose(
        restored.get_booster().predict(frame),
        restored.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )


def test_legacy_flat_best_score_is_migrated_on_pickle_load(tmp_path):
    X, y = make_regression(n_samples=48, n_features=3, random_state=43)
    regressor = ctboost.CTBoostRegressor(
        iterations=4,
        alpha=1.0,
        random_seed=19,
    ).fit(X[:32].astype(np.float32), y[:32].astype(np.float32), eval_set=(X[32:], y[32:]))
    regressor.best_score_ = {"learn": 123.0, "validation": 456.0}

    model_path = tmp_path / "legacy_best_score.pkl"
    regressor.save_model(model_path)
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    assert all(isinstance(scores, dict) for scores in restored.best_score_.values())
    assert restored.best_score_ == restored._compute_best_score()


def test_fit_type_hints_resolve_pathlike_annotations():
    methods = [
        ctboost.CTBoostClassifier.fit,
        ctboost.CTBoostRegressor.fit,
        ctboost.CTBoostRanker.fit,
        _BaseFitMixin._fit_impl,
    ]
    for method in methods:
        hints = typing.get_type_hints(method)
        assert "snapshot_path" in hints
