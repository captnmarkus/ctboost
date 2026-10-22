"""Single-row string categorical extraction preserves scalar and pipeline semantics."""

import numpy as np
import pytest

from ctboost.feature_pipeline import FeaturePipeline


pd = pytest.importorskip("pandas")


def _assert_cells_equal(expected, actual):
    assert expected.shape == actual.shape
    assert expected.dtype == actual.dtype == np.dtype(object)
    for left, right in zip(expected.flat, actual.flat):
        assert type(left) is type(right)
        if left is right:
            continue
        if bool(pd.isna(left)):
            assert bool(pd.isna(right))
        else:
            assert left == right


def _frame_case(case):
    if case == "empty_rows":
        return pd.DataFrame({"a": pd.Categorical([], categories=["a", "b"])})
    if case == "empty_columns":
        return pd.DataFrame(index=[0])
    if case == "one_column":
        return pd.DataFrame({"a": pd.Categorical(["a"], categories=["b", "a"])})
    if case == "all_missing":
        return pd.DataFrame({"a": pd.Categorical([None], categories=["a", "b"]),
                             "b": pd.Categorical([None], categories=["b", "a"])})
    if case == "different_order":
        return pd.DataFrame({"a": pd.Categorical(["a"], categories=["a", "b"], ordered=True),
                             "b": pd.Categorical(["a"], categories=["b", "a"], ordered=True)})
    if case == "unicode_escaped":
        return pd.DataFrame({"a": pd.Categorical(["λ|é\\n"], categories=["λ|é\\n", "東京"]),
                             "b": pd.Categorical(["\x00|\\"], categories=["\x00|\\", "😀"])})
    if case == "integer_float":
        return pd.DataFrame({"a": pd.Categorical([2**62 + 1]), "b": pd.Categorical([1.5])})
    if case == "integer_missing":
        return pd.DataFrame({"a": pd.Categorical([2**62 + 1]),
                             "b": pd.Categorical([None], categories=[1, 2])})
    if case == "mixed_numeric":
        return pd.DataFrame({"a": [2**62 + 1], "b": pd.Categorical(["a"])})
    if case == "numpy_string":
        return pd.DataFrame({"a": pd.Categorical([np.str_("a")],
                             categories=pd.Index([np.str_("a"), np.str_("b")], dtype=object))})
    if case == "bytes":
        return pd.DataFrame({"a": pd.Categorical([b"a"]), "b": pd.Categorical([b"b"])})
    if case == "object_vocabulary":
        return pd.DataFrame({"a": pd.Categorical([2**62 + 1], categories=pd.Index([2**62 + 1, "a"], dtype=object)),
                             "b": pd.Categorical([1.5], categories=pd.Index([1.5, "b"], dtype=object))})
    if case == "nullable_text":
        return pd.DataFrame({"a": pd.Series([pd.NA], dtype="string"),
                             "b": pd.Series(["a"], dtype="string")})
    if case == "nullable_text_categories":
        return pd.DataFrame({"a": pd.Categorical(["a"], categories=pd.Index(["a", "b"], dtype="string"))})
    if case == "large_vocabulary":
        return pd.DataFrame({"a": pd.Categorical(["a0"], categories=[f"a{i}" for i in range(33)])})
    if case == "empty_vocabulary":
        return pd.DataFrame({"a": pd.Categorical([None])})
    if case == "multiple_rows":
        return pd.DataFrame({"a": pd.Categorical(["a", None], categories=["a", "b"])})
    raise AssertionError(case)


@pytest.mark.parametrize("case", [
    "empty_rows", "empty_columns", "one_column", "all_missing", "different_order",
    "unicode_escaped", "integer_float", "integer_missing", "mixed_numeric", "numpy_string",
    "bytes", "object_vocabulary", "nullable_text", "nullable_text_categories",
    "large_vocabulary", "empty_vocabulary", "multiple_rows",
])
def test_categorical_row_preserves_exact_cells(case):
    frame = _frame_case(case)
    expected = frame.to_numpy(dtype=object, copy=False)
    actual, names = FeaturePipeline._extract_frame(frame, allow_categorical_row=True)
    _assert_cells_equal(expected, actual)
    assert names == [str(name) for name in frame.columns]


def test_small_string_categorical_row_avoids_dataframe_block_conversion(monkeypatch):
    frame = _frame_case("different_order")
    expected = frame.to_numpy(dtype=object, copy=False)

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("should extract this safe string row without DataFrame.to_numpy")

    monkeypatch.setattr(pd.DataFrame, "to_numpy", unexpected_conversion)
    actual, _ = FeaturePipeline._extract_frame(frame, allow_categorical_row=True)
    _assert_cells_equal(expected, actual)


def test_categorical_string_subclass_keeps_conversion_hooks_unchanged():
    class CustomString(str):
        armed = False
        comparisons = 0
        conversions = 0
        __hash__ = str.__hash__

        def __eq__(self, other):
            type(self).comparisons += 1
            if type(self).armed:
                raise RuntimeError("custom string equality")
            return str.__eq__(self, other)

        def __str__(self):
            type(self).conversions += 1
            return "custom:" + str.__str__(self)

    left = CustomString("a")
    right = CustomString("a")
    frame = pd.DataFrame({
        "a": pd.Categorical([left], categories=pd.Index([left, "b"], dtype=object)),
        "b": pd.Categorical([right], categories=pd.Index([right, "c"], dtype=object)),
    })
    CustomString.armed = True
    CustomString.comparisons = CustomString.conversions = 0
    expected = frame.to_numpy(dtype=object, copy=False)
    baseline_hooks = (CustomString.comparisons, CustomString.conversions)
    CustomString.comparisons = CustomString.conversions = 0
    actual, _ = FeaturePipeline._extract_frame(frame, allow_categorical_row=True)
    assert (CustomString.comparisons, CustomString.conversions) == baseline_hooks
    _assert_cells_equal(expected, actual)


def test_dataframe_subclass_keeps_custom_numpy_contract():
    class CustomFrame(pd.DataFrame):
        def to_numpy(self, *args, **kwargs):
            return np.array([["custom-value"]], dtype=object)

    frame = CustomFrame({"a": pd.Categorical(["a"])})
    actual, names = FeaturePipeline._extract_frame(frame, allow_categorical_row=True)
    assert actual[0, 0] == "custom-value"
    assert names == ["a"]


@pytest.mark.parametrize("first,second", [("new", "λ|é\\n"), (None, None), ("a", "東京")])
def test_categorical_row_native_transform_matches_existing_conversion(first, second, monkeypatch):
    train = pd.DataFrame({"a": pd.Categorical(["a", "b", "a", "b"]),
                          "b": pd.Categorical(["λ|é\\n", "東京", "東京", "λ|é\\n"])})
    pipeline = FeaturePipeline(cat_features=["a", "b"], ordered_ctr=True)
    pipeline.fit(train, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    frame = pd.DataFrame({"a": pd.Categorical([first], categories=["new", "b", "a"]),
                          "b": pd.Categorical([second], categories=["東京", "λ|é\\n"])})
    expected, expected_cats, expected_names = pipeline._native.transform_array(
        frame.to_numpy(dtype=object, copy=False), ["a", "b"])
    state_before = pipeline.to_state()

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("fitted all-categorical pipeline should use the safe row extraction")

    monkeypatch.setattr(pd.DataFrame, "to_numpy", unexpected_conversion)
    actual, actual_cats, actual_names = pipeline.transform_array(frame)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert actual_cats == list(expected_cats)
    assert actual_names == list(expected_names)
    assert pipeline.to_state() == state_before
    restored = FeaturePipeline.from_state(state_before)
    np.testing.assert_array_equal(restored.transform_array(frame)[0].view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("kind", ["numeric", "mixed", "one_hot", "combination"])
def test_ineligible_fitted_pipeline_skips_row_dtype_inspection(kind, monkeypatch):
    if kind == "numeric":
        train = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.5, 2.5, 3.5, 4.5]})
        pipeline = FeaturePipeline()
    elif kind == "mixed":
        train = pd.DataFrame({"a": [1, 2, 3, 4], "b": pd.Categorical(["a", "b", "a", "b"])})
        pipeline = FeaturePipeline(cat_features=["b"])
    else:
        train = pd.DataFrame({"a": pd.Categorical(["a", "b", "a", "b"]),
                              "b": pd.Categorical(["b", "a", "b", "a"])})
        options = {"one_hot_max_size": 2} if kind == "one_hot" else {"categorical_combinations": [["a", "b"]]}
        pipeline = FeaturePipeline(cat_features=["a", "b"], **options)
    pipeline.fit(train, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    frame = train.iloc[:1]
    expected = pipeline._native.transform_array(frame.to_numpy(dtype=object, copy=False), ["a", "b"])[0]

    def unexpected_dtypes(self):
        raise AssertionError("ineligible fitted schema must not inspect row dtypes")

    monkeypatch.setattr(pd.DataFrame, "dtypes", property(unexpected_dtypes))
    actual = pipeline.transform_array(frame)[0]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_unfitted_extractor_does_not_inspect_categorical_vocabulary(monkeypatch):
    frame = _frame_case("one_column")
    expected = frame.to_numpy(dtype=object, copy=False)

    def unexpected_dtypes(self):
        raise AssertionError("row shortcut is opt-in for fitted transforms")

    monkeypatch.setattr(pd.DataFrame, "dtypes", property(unexpected_dtypes))
    actual, _ = FeaturePipeline._extract_frame(frame)
    _assert_cells_equal(expected, actual)
