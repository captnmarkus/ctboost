"""Exactness and dispatch coverage for native categorical-code transforms."""
import copy
import pickle

import numpy as np
import pytest

import ctboost
from ctboost._categorical_codes import build_code_plan, try_code_transform

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


def pipeline(width=32):
    return ctboost.FeaturePipeline(cat_features=list(range(width)), ordered_ctr=False).fit(
        np.array([["a"] * width, ["b"] * width] * 4, dtype=object), np.arange(8, dtype=np.float32))


def object_categorical_frame(values, width=32, categories=("a", "b")):
    """Keep positive-path fixtures eligible under pandas 2 and 3 inference."""
    names = pd.Index([str(i) for i in range(width)], dtype=object)
    vocabulary = pd.Index(categories, dtype=object)
    frame = pd.DataFrame({name: pd.Categorical(values, categories=vocabulary)
                          for name in names}, columns=names)
    assert frame.columns.dtype == np.dtype(object)
    assert all(dtype.categories.dtype == np.dtype(object) for dtype in frame.dtypes)
    return frame


class MutatingName:
    def __init__(self):
        self.frame = None
        self.calls = 0

    def __str__(self):
        self.calls += 1
        self.frame.iat[0, 0] = "b"
        return "0"


def mutating_frame():
    name = MutatingName()
    frame = object_categorical_frame(["a"] * 4096)
    frame.columns = pd.Index([name] + [str(i) for i in range(1, 32)], dtype=object)
    name.frame = frame
    return frame, name


@pytest.mark.parametrize("kind", ["range", "numeric_index", "multi", "str_subclass", "nullable_string", "index_subclass"])
def test_custom_or_nonplain_names_fall_back_without_hooks(kind):
    class Text(str):
        def __str__(self):
            raise AssertionError("Name hook must not run during eligibility")

    class HookIndex(pd.Index):
        def __iter__(self):
            raise AssertionError("Index hook must not run during eligibility")

    frame = object_categorical_frame(["a"] * 4096)
    if kind == "range":
        names = pd.RangeIndex(32)
    elif kind == "numeric_index":
        names = pd.Index(np.arange(32) + 42)
    elif kind == "multi":
        names = pd.MultiIndex.from_tuples([("a", str(i)) for i in range(32)])
    elif kind == "str_subclass":
        names = pd.Index([Text("0")] + [str(i) for i in range(1, 32)], dtype=object)
    elif kind == "nullable_string":
        names = pd.Index([str(i) for i in range(32)], dtype="string")
    else:
        names = HookIndex._simple_new(np.array([str(i) for i in range(32)], dtype=object))
    frame.columns = names
    model = pipeline()
    assert try_code_transform(model, frame, build_code_plan(model.to_state())) is None


@pytest.mark.parametrize("width", [31, 32])
@pytest.mark.parametrize("rows", [999, 1000])
def test_public_dispatch_width_and_row_boundary(width, rows):
    model = pipeline(width)
    frame = object_categorical_frame((["a", "b"] * ((rows + 1) // 2))[:rows], width)
    plan = build_code_plan(model.to_state())
    result = try_code_transform(model, frame, plan)
    reference = model._native.transform_array(frame.to_numpy(dtype=object, copy=False), list(frame.columns))
    if width < 32:
        assert plan is None
    if width < 32 or rows < 1000:
        assert result is None
    else:
        assert result is not None
        np.testing.assert_array_equal(result[0].view(np.uint32), reference[0].view(np.uint32))
    public = model.transform_array(frame)
    np.testing.assert_array_equal(public[0].view(np.uint32), reference[0].view(np.uint32))


@pytest.mark.parametrize("kind", ["late_33", "objects"])
def test_unsupported_wide_frame_rejects_before_series_iteration(monkeypatch, kind):
    model = pipeline()
    frame = object_categorical_frame(["a"] * 1000)
    if kind == "late_33":
        frame["31"] = pd.Categorical(["a"] * len(frame),
                                      categories=pd.Index(["a", "b"] + [str(i) for i in range(31)], dtype=object))
        assert len(frame["31"].cat.categories) == 33
        assert frame["31"].cat.categories.dtype == np.dtype(object)
    else:
        frame = frame.astype(object)
    def forbidden(*args, **kwargs):
        raise AssertionError("Series iteration must not occur before fallback")
    monkeypatch.setattr(pd.DataFrame, "items", forbidden)
    assert try_code_transform(model, frame, build_code_plan(model.to_state())) is None


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("prior", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("classes", [False, True])
def test_public_native_codes_preserves_codec_ctrs_and_persistence(legacy, prior, classes):
    values = [None, "__ctboost_missing__", "__ctboost_other__", "\\m", "a||b", "λ東京", "", "red"]
    raw = np.array([[values[(row + col) % len(values)] for col in range(32)]
                    for row in range(80)], dtype=object)
    labels = (np.arange(len(raw)) % 3 if classes else np.arange(len(raw)) / 7).astype(np.float32)
    model = ctboost.FeaturePipeline(cat_features=list(range(32)), simple_ctr=["Mean", "Frequency"],
                                    ctr_prior_strength=prior, max_cat_threshold=4)
    if legacy:
        state = model.to_state()
        state["feature_pipeline_format_version"] = 2
        state.pop("categorical_key_encoding_version")
        model = ctboost.FeaturePipeline.from_state(state)
    model.fit(raw, labels)
    vocabulary = [value for value in values if value is not None][::-1] + ["unseen", "unused"]
    frame = object_categorical_frame([None, "red", "unseen", "λ東京"] * 256,
                                     categories=vocabulary)
    state = copy.deepcopy(model.to_state())
    assert model._categorical_code_plan is not None
    reference = expected(model, frame)
    assert_exact(try_code_transform(model, frame, model._categorical_code_plan), reference)
    assert_exact(model.transform_array(frame), reference)
    assert model.to_state() == state
    restored = pickle.loads(pickle.dumps(model))
    assert restored._categorical_code_plan is not None
    assert_exact(restored.transform_array(frame), reference)


def test_public_native_codes_fallback_preserves_mutating_name_snapshot():
    model = pipeline()
    reference_frame, reference_name = mutating_frame()
    actual_frame, actual_name = mutating_frame()
    with pd.option_context("mode.copy_on_write", False):
        matrix = reference_frame.to_numpy(dtype=object, copy=False)
        names = [str(name) for name in reference_frame.columns]
        reference = model._native.transform_array(matrix, names)
        assert_exact(model.transform_array(actual_frame), reference)
    assert actual_name.calls == reference_name.calls == 1


def test_public_native_codes_feature_names_still_validated():
    frame = object_categorical_frame(["a", "b"] * 500)
    model = ctboost.FeaturePipeline(cat_features=list(frame.columns)).fit(frame, np.arange(1000, dtype=np.float32))
    assert_exact(try_code_transform(model, frame, model._categorical_code_plan), expected(model, frame))
    wrong = frame.rename(columns={"31": "wrong"})
    wrong.columns = pd.Index(list(wrong.columns), dtype=object)
    with pytest.raises(ValueError, match="feature names"):
        model.transform_array(wrong)


@pytest.mark.parametrize("extension_index", ["categories", "columns"])
def test_string_extension_indexes_fall_back_with_public_bit_parity(extension_index):
    model = pipeline()
    frame = object_categorical_frame(["a", "b", None, "unseen"] * 256,
                                     categories=["a", "b", "unseen"])
    plan = model._categorical_code_plan
    assert_exact(try_code_transform(model, frame, plan), expected(model, frame))
    string_dtype = pd.StringDtype(storage="python")
    if extension_index == "categories":
        frame["31"] = pd.Categorical(["a", "b", None, "unseen"] * 256,
                                      categories=pd.Index(["a", "b", "unseen"], dtype=string_dtype))
        assert isinstance(frame["31"].cat.categories.dtype, pd.StringDtype)
        assert frame.columns.dtype == np.dtype(object)
    else:
        frame.columns = pd.Index(list(frame.columns), dtype=string_dtype)
        assert isinstance(frame.columns.dtype, pd.StringDtype)
        assert all(dtype.categories.dtype == np.dtype(object) for dtype in frame.dtypes)
    reference = expected(model, frame)
    before = copy.deepcopy(model.to_state())
    assert try_code_transform(model, frame, plan) is None
    assert_exact(model.transform_array(frame), reference)
    assert model.to_state() == before


def test_categorical_codes_new_python_falls_back_with_legacy_native():
    """An older extension without the optional method keeps existing transforms."""
    frame = object_categorical_frame(["a", "b", None, "c"] * 256, categories=["a", "b", "c"])
    labels = (np.arange(len(frame)) % 3).astype(np.float32)
    pipeline = ctboost.FeaturePipeline(cat_features=list(frame.columns)).fit(frame, labels)
    assert pipeline._categorical_code_plan is not None
    assert_exact(try_code_transform(pipeline, frame, pipeline._categorical_code_plan),
                 expected(pipeline, frame))
    reference = pipeline.transform_array(frame)
    native = pipeline._native

    class LegacyNative:
        def __getattr__(self, name):
            if name == "_transform_categorical_codes":
                raise AttributeError(name)
            return getattr(native, name)

    pipeline._native = LegacyNative()
    pipeline._refresh_metadata()
    assert pipeline._categorical_code_plan is None
    actual = pipeline.transform_array(frame)
    np.testing.assert_array_equal(actual[0].view(np.uint32), reference[0].view(np.uint32))
    assert actual[1:] == reference[1:]
