"""Focused tests for the isolated categorical-code experiment, not a public API."""
import copy
import pickle

import numpy as np
import pytest

import ctboost
from benchmarks.native_categorical_codes_prototype import build_code_plan, try_code_transform

pd = pytest.importorskip("pandas")


def expected(pipeline, frame):
    return pipeline._native.transform_array(frame.to_numpy(dtype=object, copy=False),
                                            [str(name) for name in frame.columns])


def assert_exact(actual, reference):
    assert actual is not None
    assert actual[0].flags.f_contiguous
    np.testing.assert_array_equal(actual[0].view(np.uint32), reference[0].view(np.uint32))
    assert list(actual[1]) == list(reference[1])
    assert list(actual[2]) == list(reference[2])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("prior", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("classes", [False, True])
def test_codes_exact_current_legacy_multiclass_and_smoothing(legacy, prior, classes):
    values = [None, "__ctboost_missing__", "__ctboost_other__", "\\m", "a||b", "λ東京", "", "red"]
    raw = np.array([[values[i % len(values)], values[(i * 3) % len(values)]] for i in range(80)], dtype=object)
    labels = (np.arange(len(raw)) % 3 if classes else np.arange(len(raw)) / 7).astype(np.float32)
    pipeline = ctboost.FeaturePipeline(cat_features=[0, 1], simple_ctr=["Mean", "Frequency"],
                                      ctr_prior_strength=prior, max_cat_threshold=4)
    if legacy:
        state = pipeline.to_state()
        state["feature_pipeline_format_version"] = 2
        state.pop("categorical_key_encoding_version")
        pipeline = ctboost.FeaturePipeline.from_state(state)
    pipeline.fit(raw, labels)
    category_order = [value for value in values if value is not None][::-1] + ["unseen", "unused"]
    frame = pd.DataFrame({"0": pd.Categorical([None, "red", "unseen", "λ東京"] * 16, categories=category_order),
                          "1": pd.Categorical(["a||b", "", None, "\\m"] * 16, categories=category_order[::-1])})
    # Fit array used no feature names; the DataFrame's string names remain valid.
    before = copy.deepcopy(pipeline.to_state())
    plan = build_code_plan(before)
    assert_exact(try_code_transform(pipeline, frame, plan), expected(pipeline, frame))
    assert pipeline.to_state() == before
    restored = pickle.loads(pickle.dumps(pipeline))
    assert_exact(try_code_transform(restored, frame, build_code_plan(restored.to_state())), expected(pipeline, frame))


@pytest.mark.parametrize("mode", ["all_missing", "one_category", "sliced", "unused_surrogate", "empty_vocabulary"])
def test_codes_used_values_and_missing(mode):
    pipeline = ctboost.FeaturePipeline(cat_features=[0], simple_ctr=["Mean", "Frequency"]).fit(
        np.array([["a"], ["b"], [None]] * 4, dtype=object), np.arange(12, dtype=np.float32))
    categories = ["b", "a", "never_seen"]
    values = ["a", None, "b", "a"] * 4
    if mode == "all_missing":
        values = [None] * 16
    elif mode == "one_category":
        values = ["a"] * 16
    elif mode == "unused_surrogate":
        categories += ["\ud800"]
    elif mode == "empty_vocabulary":
        values, categories = [None] * 16, []
    frame = pd.DataFrame({"0": pd.Categorical(values, categories=categories)})
    if mode == "sliced":
        frame = frame.iloc[::-2]
    assert_exact(try_code_transform(pipeline, frame, build_code_plan(pipeline.to_state())), expected(pipeline, frame))


def test_codes_used_invalid_unicode_has_same_exception():
    pipeline = ctboost.FeaturePipeline(cat_features=[0]).fit(np.array([["a"], ["b"]], dtype=object), [0, 1])
    frame = pd.DataFrame({"0": pd.Categorical(["a", "\ud800", "\ud801"])})
    with pytest.raises(Exception) as slow:
        expected(pipeline, frame)
    with pytest.raises(type(slow.value)) as fast:
        try_code_transform(pipeline, frame, build_code_plan(pipeline.to_state()))
    assert str(fast.value) == str(slow.value)


@pytest.mark.parametrize("case", ["one_row", "empty", "numeric", "mixed", "large", "custom", "npstr", "nullable"])
def test_codes_conservative_fallback(case):
    pipeline = ctboost.FeaturePipeline(cat_features=[0]).fit(np.array([["a"], ["b"]], dtype=object), [0, 1])
    if case == "one_row":
        frame = pd.DataFrame({"0": pd.Categorical(["a"])})
    elif case == "empty":
        frame = pd.DataFrame({"0": pd.Categorical([], categories=["a"])})
    elif case == "numeric":
        frame = pd.DataFrame({"0": pd.Categorical([2**62 + 1, 2**62 + 2])})
    elif case == "mixed":
        frame = pd.DataFrame({"0": ["a", "b"]})
    elif case == "large":
        frame = pd.DataFrame({"0": pd.Categorical(["a", "b"], categories=["a", "b"] + [str(i) for i in range(31)])})
    elif case == "custom":
        class Text(str):
            def __str__(self):
                raise AssertionError("custom hook must not be invoked by eligibility")
        frame = pd.DataFrame({"0": pd.Categorical(["a", "a"], categories=pd.Index(["a", Text("b")], dtype=object))})
    elif case == "npstr":
        frame = pd.DataFrame({"0": pd.Categorical([np.str_("a"), np.str_("b")], categories=pd.Index([np.str_("a"), np.str_("b")], dtype=object))})
    else:
        frame = pd.DataFrame({"0": pd.Categorical(["a", "b"], categories=pd.Index(["a", "b"], dtype="string"))})
    assert try_code_transform(pipeline, frame, build_code_plan(pipeline.to_state())) is None


@pytest.mark.parametrize("change", ["onehot", "combination", "multisource_ctr", "duplicate", "gap", "output_gap"])
def test_codes_plan_rejects_unsupported_state(change):
    pipeline = ctboost.FeaturePipeline(cat_features=[0, 1], simple_ctr=["Mean"]).fit(
        np.array([["a", "x"], ["b", "y"]] * 4, dtype=object), np.arange(8, dtype=np.float32))
    state = pipeline.to_state()
    if change == "onehot":
        state["one_hot_states"] = [{}]
    elif change == "combination":
        state["combination_states"] = [{}]
    elif change == "multisource_ctr":
        state["ctr_states"][0]["source_indices"] = [0, 1]
    elif change == "duplicate":
        state["categorical_states"][1]["source_index"] = 0
    elif change == "gap":
        state["numeric_indices"] = [0]
    else:
        state["output_feature_names_"].pop()
    assert build_code_plan(state) is None


def test_codes_feature_name_validation_preserved():
    frame = pd.DataFrame({"a": pd.Categorical(["red", "blue"] * 8)})
    pipeline = ctboost.FeaturePipeline(cat_features=["a"]).fit(frame, np.arange(16, dtype=np.float32))
    wrong = frame.rename(columns={"a": "wrong"})
    with pytest.raises(ValueError, match="feature names"):
        try_code_transform(pipeline, wrong, build_code_plan(pipeline.to_state()))


def test_codes_prediction_bits_after_preparation():
    frame = pd.DataFrame({"a": pd.Categorical(["red", "blue", None, "green"] * 24),
                          "b": pd.Categorical(["y", "x"] * 48)})
    y = (np.arange(len(frame)) % 3).astype(np.float32)
    model = ctboost.CTBoostClassifier(iterations=5, max_depth=2, alpha=.2,
                                     cat_features=["a", "b"], ordered_ctr=True, verbose=False).fit(frame, y)
    pipeline = model._feature_pipeline
    output, cats, names = try_code_transform(pipeline, frame, build_code_plan(pipeline.to_state()))
    candidate = ctboost.Pool(output, cat_features=cats, feature_names=names)
    candidate._feature_pipeline = pipeline
    baseline = model._transform_prediction_pool(frame)
    np.testing.assert_array_equal(np.ascontiguousarray(model.predict_proba(candidate)).view(np.uint8),
                                  np.ascontiguousarray(model.predict_proba(baseline)).view(np.uint8))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
@pytest.mark.parametrize("view", ["plain", "reversed", "empty"])
def test_native_codes_signed_widths_and_strided_views(dtype, view):
    pipeline = ctboost.FeaturePipeline(cat_features=[0], simple_ctr=["Mean", "Frequency"]).fit(
        np.array([["a"], ["b"], [None]] * 4, dtype=object), np.arange(12, dtype=np.float32))
    dictionary = np.array(["unused", "b", "a"], dtype=object)[::-1]
    codes = np.array([-1, 0, 2, 1, 0, -1], dtype=dtype)
    if view == "reversed":
        codes = codes[::-1]
    elif view == "empty":
        codes = codes[:0]
    raw = np.array([[np.nan if code == -1 else dictionary[int(code)]] for code in codes], dtype=object).reshape(-1, 1)
    actual = pipeline._native._transform_categorical_codes([codes], [dictionary])
    reference = pipeline._native.transform_array(raw)
    assert_exact(actual, reference)


@pytest.mark.parametrize("kind", ["float", "unsigned", "byteswapped", "rank2", "range_negative", "range_positive", "dictionary_type", "dictionary_large", "width"])
def test_native_codes_rejects_invalid_buffers(kind):
    pipeline = ctboost.FeaturePipeline(cat_features=[0]).fit(
        np.array([["a"], ["b"]], dtype=object), [0, 1])
    codes = np.array([0, 1], dtype=np.int8)
    dictionary = np.array(["a", "b"], dtype=object)
    if kind == "float":
        codes = codes.astype(np.float32)
    elif kind == "unsigned":
        codes = codes.astype(np.uint8)
    elif kind == "byteswapped":
        codes = codes.astype('>i4')
    elif kind == "rank2":
        codes = codes.reshape(-1, 1)
    elif kind == "range_negative":
        codes[0] = -2
    elif kind == "range_positive":
        codes[0] = 2
    elif kind == "dictionary_type":
        dictionary = np.array(["a", 2], dtype=object)
    elif kind == "dictionary_large":
        dictionary = np.array([str(i) for i in range(33)], dtype=object)
    args = ([codes], [dictionary]) if kind != "width" else ([], [])
    with pytest.raises(ValueError):
        pipeline._native._transform_categorical_codes(*args)


def test_native_codes_validates_row_count_before_gather():
    pipeline = ctboost.FeaturePipeline(cat_features=[0, 1]).fit(
        np.array([["a", "x"], ["b", "y"]], dtype=object), [0, 1])
    with pytest.raises(ValueError, match="row count"):
        pipeline._native._transform_categorical_codes(
            [np.array([0, 1], dtype=np.int8), np.array([0], dtype=np.int8)],
            [np.array(["a", "b"], dtype=object), np.array(["x", "y"], dtype=object)])


def test_native_codes_maximum_dictionary_plus_missing_slot():
    dictionary = np.array([str(i) for i in range(32)], dtype=object)
    raw = np.array([[value] for value in dictionary] + [[np.nan]], dtype=object)
    pipeline = ctboost.FeaturePipeline(cat_features=[0], simple_ctr=["Mean", "Frequency"]).fit(
        raw, np.arange(len(raw), dtype=np.float32))
    actual = pipeline._native._transform_categorical_codes(
        [np.array(list(range(32)) + [-1], dtype=np.int8)], [dictionary])
    assert_exact(actual, pipeline._native.transform_array(raw))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_native_codes_unaligned_integer_storage(dtype):
    codes = np.ndarray((4,), dtype=dtype, buffer=bytearray(4 * np.dtype(dtype).itemsize + 1), offset=1)
    codes[:] = [0, -1, 1, 0]
    dictionary = np.array(["a", "b"], dtype=object)
    raw = np.array([["a"], [np.nan], ["b"], ["a"]], dtype=object)
    pipeline = ctboost.FeaturePipeline(cat_features=[0], simple_ctr=["Mean", "Frequency"]).fit(
        raw, np.arange(4, dtype=np.float32))
    assert not codes.flags.aligned
    assert_exact(pipeline._native._transform_categorical_codes([codes], [dictionary]),
                 pipeline._native.transform_array(raw))
