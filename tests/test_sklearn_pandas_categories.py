import numpy as np
import pytest

import ctboost


pd = pytest.importorskip("pandas")


@pytest.mark.parametrize("estimator", [ctboost.CTBoostClassifier, ctboost.CTBoostRegressor])
@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_default_pandas_categories_reuse_training_codes(estimator, dtype, tmp_path):
    colors = ["blue", "red"] * 48
    y = np.tile([0, 1], 48)
    X = pd.DataFrame({"color": pd.Series(colors, dtype=dtype), "constant": 1.0})
    red = X.iloc[[1]].copy()
    if dtype == "category":
        red["color"] = red["color"].cat.reorder_categories(["red", "blue"])
    model = estimator(iterations=6).fit(X, y, eval_set=(red, [1]))
    predict = "predict_proba" if estimator is ctboost.CTBoostClassifier else "predict"
    batch = getattr(model, predict)(X)
    assert not np.array_equal(batch[0], batch[1])
    expected = batch[[1]]
    np.testing.assert_array_equal(getattr(model, predict)(red), expected)
    np.testing.assert_array_equal(getattr(model, predict)(ctboost.Pool([[1, 1]])), expected)
    np.testing.assert_array_equal(getattr(model, predict)(np.array([[1, 1]], dtype="f")), expected)
    prepared_validation = ctboost.Pool([[1, 1]], [1], cat_features=[0], feature_names=list(X.columns))
    reference = estimator(iterations=6).fit(X, y, eval_set=prepared_validation)
    assert model.evals_result_ == reference.evals_result_
    for suffix in ["json", "pkl"]:
        path = tmp_path / f"model.{suffix}"
        model.save_model(path)
        restored = estimator.load_model(path)
        np.testing.assert_array_equal(getattr(restored, predict)(red), expected)
    contract = model.get_inference_manifest()["input"]
    assert contract["representation"] == "numeric_or_preencoded_categorical_features"
    assert contract["features"][0]["categorical_schema"]["pandas_category_labels"] == ["blue", "red"]
    portable_path = tmp_path / "predictor.json"
    model.export_model(portable_path)
    portable = ctboost.load_exported_predictor(portable_path)
    np.testing.assert_allclose(getattr(portable, predict)([[1, 1]]), expected, rtol=1e-6)


def test_pandas_warm_start_freezes_category_order_and_unknown_route():
    train = pd.DataFrame({"x": pd.Categorical(["a", "b", None] * 40, categories=["b", "a"])})
    y = np.tile([8.0, -4.0, 1.0], 40)
    model = ctboost.CTBoostRegressor(iterations=5, warm_start=True).fit(train, y)
    continuation = pd.DataFrame({"x": pd.Categorical(["b", "c", None] * 20, categories=["c", "b"])})
    target = np.tile([-4.0, 1.0, 1.0], 20)
    explicit = ctboost.CTBoostRegressor(iterations=5).fit(
        ctboost.Pool(np.tile([0, -1, -1], 20).reshape(-1, 1), target, cat_features=[0], feature_names=["x"]),
        init_model=model,
    )
    model.fit(continuation, target)
    assert model.data_schema_["categorical_schema"]["x"]["pandas_category_labels"] == ["b", "a"]
    probes = pd.DataFrame({"x": ["b", "a", "new", None]})
    expected = explicit.predict(np.array([[0], [1], [-1], [-1]]))
    np.testing.assert_array_equal(model.predict(probes), expected)


def test_mixed_category_types_remain_distinct_after_json(tmp_path):
    labels = [1, "1", b"1", pd.Timestamp("2026-01-01")] * 32
    dtype = pd.CategoricalDtype(categories=pd.Index(labels[:4], dtype=object))
    X = pd.DataFrame({"x": pd.Series(labels, dtype=dtype)})
    y = np.tile([0.0, 3.0, 7.0, 15.0], 32)
    model = ctboost.CTBoostRegressor(iterations=12).fit(X, y)
    path = tmp_path / "typed.json"
    model.save_model(path)
    restored = ctboost.CTBoostRegressor.load_model(path)
    expected = model.predict(np.arange(4).reshape(-1, 1))
    assert len(np.unique(expected)) == 4
    for index, value in enumerate(labels[:4]):
        row = pd.DataFrame({"x": pd.Series([value], dtype=object)})
        np.testing.assert_array_equal(restored.predict(row), expected[index:index + 1])


def test_default_pandas_ranker_keeps_grouped_validation_codes():
    frame = pd.DataFrame({"x": ["a", "b"] * 24})
    groups = np.repeat(np.arange(24), 2)
    relevance = np.tile([0, 1], 24)
    model = ctboost.CTBoostRanker(iterations=6).fit(
        frame, relevance, group_id=groups, eval_set=(frame.iloc[:2], [0, 1], [0, 0])
    )
    assert model.data_schema_["categorical_schema"]["x"]["pandas_category_labels"] == ["a", "b"]
    np.testing.assert_array_equal(model.predict(frame.iloc[[1]]), model.predict(frame)[[1]])


def test_snapshot_and_legacy_warm_start_category_contracts(tmp_path):
    frame = pd.DataFrame({"x": ["a", "b"] * 32})
    y = np.tile([0.0, 1.0], 32)
    for name, data, expected_schema in [
        ("current", frame, {"x": {"pandas_category_labels": ["a", "b"]}}),
        ("legacy", ctboost.Pool(frame, y), None),
    ]:
        # A prepared Pool has the same persisted schema as older estimators.
        snapshot = tmp_path / f"{name}.json"
        initial = ctboost.CTBoostRegressor(iterations=2).fit(data, y, snapshot_path=snapshot)
        for kwargs in [{"init_model": initial}, {"resume_from_snapshot": snapshot}]:
            resumed = ctboost.CTBoostRegressor(iterations=4).fit(frame, y, **kwargs)
            assert resumed.data_schema_["categorical_schema"] == expected_schema
