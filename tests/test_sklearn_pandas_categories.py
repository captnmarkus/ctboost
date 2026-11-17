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


@pytest.mark.parametrize("columns", [[1, "1"], ["same", "same"]])
def test_colliding_category_names_use_positional_vocabularies(columns, tmp_path):
    codes = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (32, 1))
    X = pd.DataFrame(np.column_stack([np.where(codes[:, 0], "b", "a"),
                                      np.where(codes[:, 1], "d", "c")]), columns=columns)
    y = codes @ [5.0, 3.0]
    query = X.iloc[[1, 2]].copy()
    model = ctboost.CTBoostRegressor(iterations=6).fit(X, y, eval_set=(query, y[[1, 2]]))
    key = str(columns[0])
    expected = {key: {"pandas_category_labels_by_position": {"0": ["a", "b"], "1": ["c", "d"]}}}
    assert model.data_schema_["categorical_schema"] == expected
    assert len(np.unique(model.predict(codes))) == 4
    np.testing.assert_array_equal(model.predict(X), model.predict(codes))
    np.testing.assert_array_equal(model.predict(query.iloc[[1]]), model.predict(codes[[2]]))
    validation = ctboost.Pool(codes[[1, 2]], y[[1, 2]], cat_features=[0, 1], feature_names=[key, key])
    reference = ctboost.CTBoostRegressor(iterations=6).fit(X, y, eval_set=validation)
    assert model.evals_result_ == reference.evals_result_
    path = tmp_path / "model.json"
    model.save_model(path)
    restored = ctboost.CTBoostRegressor.load_model(path)
    np.testing.assert_array_equal(restored.predict(query), restored.predict(codes[[1, 2]]))
    continued = ctboost.CTBoostRegressor(iterations=8).fit(query, y[[1, 2]], init_model=restored)
    assert continued.data_schema_["categorical_schema"] == expected
    np.testing.assert_array_equal(continued.predict(X), continued.predict(codes))


def test_duplicate_names_mixed_numeric_and_categorical_snapshot_resume(tmp_path):
    codes = np.tile([[0, 0, 1], [0, 1, 2], [1, 0, 4], [1, 1, 8]], (24, 1))
    category = pd.Series(pd.Categorical(np.where(codes[:, 0], "b", "a"), categories=["a", "b"]))
    other = pd.Series(np.where(codes[:, 1], "y", "x"))
    frame = pd.concat([category, other, pd.Series(codes[:, 2])], axis=1)
    frame.columns = [1, "unique", "1"]
    target = codes @ [5.0, 3.0, 1.0]
    snapshot = tmp_path / "snapshot.json"
    first = ctboost.CTBoostRegressor(iterations=3).fit(frame, target, snapshot_path=snapshot)
    assert first.data_schema_["categorical_schema"] == {
        "1": {"pandas_category_labels_by_position": {"0": ["a", "b"]}},
        "unique": {"pandas_category_labels": ["x", "y"]},
    }
    frame.isetitem(0, category.cat.reorder_categories(["b", "a"]).cat.add_categories(["new"]))
    frame.iloc[0, 0] = "new"
    resumed = ctboost.CTBoostRegressor(iterations=6).fit(frame, target, resume_from_snapshot=snapshot)
    codes[0, 0] = -1
    assert resumed.data_schema_["categorical_schema"] == first.data_schema_["categorical_schema"]
    np.testing.assert_array_equal(resumed.predict(frame), resumed.predict(codes))


def test_legacy_colliding_names_report_only_the_unrecoverable_column(tmp_path):
    codes = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (32, 1))
    y = codes @ [5.0, 3.0]
    old_schema = {"x": {"pandas_category_labels": ["c", "d"]}}
    pool = ctboost.Pool(codes, y, cat_features=[0, 1], feature_names=["x", "x"], categorical_schema=old_schema)
    snapshot = tmp_path / "legacy-snapshot.json"
    model = ctboost.CTBoostRegressor(iterations=6).fit(pool, snapshot_path=snapshot)
    path = tmp_path / "legacy.json"
    model.save_model(path)
    model = ctboost.CTBoostRegressor.load_model(path)
    raw = pd.DataFrame([["a", "d"]], columns=["x", "x"])
    with pytest.raises(ValueError, match="training codes.*position 0"):
        model.predict(raw)
    with pytest.raises(ValueError, match="training codes.*position 0"):
        ctboost.CTBoostRegressor(iterations=8).fit(raw, [3.0], init_model=model)
    raw.isetitem(0, [0])
    np.testing.assert_array_equal(model.predict(raw), model.predict(codes[[1]]))
    prepared_first = pd.concat([pd.Series(codes[:, 0]), pd.Series(np.where(codes[:, 1], "d", "c"))], axis=1)
    prepared_first.columns = ["x", "x"]
    for seed in [{"init_model": model}, {"resume_from_snapshot": snapshot}]:
        continued = ctboost.CTBoostRegressor(iterations=8).fit(prepared_first, y, **seed)
        assert continued.data_schema_["cat_features"] == [0, 1]
        np.testing.assert_array_equal(continued.predict(prepared_first), continued.predict(codes))
    with pytest.raises(ValueError, match="categorical indices of a prepared Pool"):
        ctboost.CTBoostRegressor(iterations=8).fit(ctboost.Pool(codes, y), init_model=model)
