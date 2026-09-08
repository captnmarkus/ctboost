"""Numeric inference must retain the existing object-path conversion contract."""

import copy

import numpy as np
import pytest

from ctboost import FeaturePipeline


def _pipeline(columns=3, names=None):
    data = np.arange(6 * columns, dtype=np.float64).reshape(6, columns)
    return FeaturePipeline().fit(data, np.arange(6), feature_names=names)


def _assert_object_equivalent(pipeline, data, names=None):
    reference_data = (
        data.to_numpy(dtype=object)
        if hasattr(data, "to_numpy")
        else np.asarray(data, dtype=object)
    )
    expected, expected_cat, expected_names = pipeline.transform_array(
        reference_data, feature_names=names
    )
    actual, actual_cat, actual_names = pipeline.transform_array(
        data, feature_names=names
    )
    assert actual.dtype == np.float32
    assert actual.flags.f_contiguous
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert actual_cat == expected_cat
    assert actual_names == expected_names


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        ">i8",
        ">u8",
        ">f4",
        ">f8",
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed", "unaligned"])
def test_numeric_dtypes_and_layouts_match_object_conversion(dtype, layout):
    dtype = np.dtype(dtype)
    data = np.arange(36).reshape(12, 3).astype(dtype)
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "strided":
        data = data[::2]
    elif layout == "reversed":
        data = data[::-1, ::-1]
    elif layout == "unaligned":
        data = np.ndarray(
            data.shape, dtype=dtype, buffer=bytearray(data.nbytes + 1), offset=1
        )
        data[:] = np.arange(36).reshape(12, 3)
    data.flags.writeable = False
    _assert_object_equivalent(_pipeline(), data)


@pytest.mark.parametrize(
    "dtype,pivot", [(np.int64, 2**63 - 2**38), (np.uint64, 2**64 - 2**39)]
)
def test_large_integer_rounding_matches_python_double_intermediate(dtype, pivot):
    limits = np.iinfo(dtype)
    values = [limits.min, limits.max, pivot - 1, pivot, pivot + 1, 2**53 + 1]
    _assert_object_equivalent(_pipeline(), np.array(values, dtype=dtype).reshape(2, 3))


def test_float_extremes_and_missing_values_are_bitwise_preserved():
    values = np.array(
        [
            0.0,
            -0.0,
            np.nan,
            np.inf,
            -np.inf,
            np.finfo(np.float64).max,
            np.finfo(np.float32).smallest_subnormal,
            -np.finfo(np.float32).smallest_subnormal,
            np.nextafter(0.0, 1.0),
            1 + 2**-24,
            np.nextafter(1 + 2**-24, 2.0),
            -1 - 2**-24,
        ]
    ).reshape(4, 3)
    _assert_object_equivalent(_pipeline(), values)


def test_mixed_numeric_dataframe_uses_python_scalar_rounding_and_preserves_names():
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(
        {
            "signed": np.array([2**63 - 2**38 - 1, -(2**63), 0], dtype=np.int64),
            "unsigned": np.array([2**64 - 2**39 + 1, 2**64 - 1, 0], dtype=np.uint64),
            "float": np.array([np.nan, -0.0, 1.25], dtype=np.float32),
            "boolean": [True, False, True],
        }
    )
    data = pd.concat([data] * 6, ignore_index=True)
    names = list(data.columns)
    _assert_object_equivalent(_pipeline(4, names), data, names)


@pytest.mark.parametrize("rows", [1, 15, 16])
def test_numeric_dataframe_batch_boundary_keeps_exact_values(rows):
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(
        {
            "integer": np.full(rows, 2**63 - 2**38 - 1, dtype=np.int64),
            "float": np.full(rows, -0.0, dtype=np.float32),
        }
    )
    names = list(data.columns)
    _assert_object_equivalent(_pipeline(2, names), data, names)


@pytest.mark.parametrize("rows", [1, 15, 16])
@pytest.mark.parametrize("kind", ["Int64", "Float64", "timestamp"])
def test_dataframe_nullable_and_timestamp_errors_keep_object_semantics(rows, kind):
    pd = pytest.importorskip("pandas")
    values = (
        pd.Series([pd.Timestamp("2020-01-01")] * rows)
        if kind == "timestamp"
        else pd.Series([1] * (rows - 1) + [pd.NA], dtype=kind)
    )
    data = pd.DataFrame({"value": values})
    pipeline = _pipeline(1, ["value"])
    # In particular, default DataFrame.to_numpy() would convert nullable pd.NA
    # into float64 NaN; the established object input deliberately remains pd.NA.
    extracted, _ = pipeline._extract_frame(data, allow_numeric=True)
    assert extracted.dtype == object
    if kind != "timestamp":
        assert extracted[-1, 0] is pd.NA
    with pytest.raises((RuntimeError, TypeError, ValueError)) as reference:
        pipeline.transform_array(data.to_numpy(dtype=object), feature_names=["value"])
    with pytest.raises(type(reference.value)):
        pipeline.transform_array(data)


@pytest.mark.parametrize(
    "ordered_ctr,one_hot_max_size", [(False, 0), (True, 0), (False, 4)]
)
def test_numeric_category_keys_keep_the_object_fallback(ordered_ctr, one_hot_max_size):
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame({"category": [1, 2, 1, 2, 1, 2], "value": np.arange(6) * 0.25})
    pipeline = FeaturePipeline(
        cat_features=["category"],
        ordered_ctr=ordered_ctr,
        one_hot_max_size=one_hot_max_size,
    ).fit(train, [0, 1, 0, 1, 0, 1])
    query = pd.DataFrame({"category": [1, 3], "value": [0.5, np.nan]})
    _assert_object_equivalent(pipeline, query, list(query.columns))
    # The native API must also reject numeric dispatch for categorical state.
    integer_query = np.array([[1, 0], [3, 1]], dtype=np.int64)
    expected = pipeline._native.transform_array(
        integer_query.astype(object), list(train.columns)
    )
    actual = pipeline._native.transform_array(integer_query, list(train.columns))
    np.testing.assert_array_equal(actual[0], expected[0])


def test_transform_does_not_reserialize_immutable_fitted_metadata(monkeypatch):
    pipeline = _pipeline(names=["a", "b", "c"])
    before = copy.deepcopy(pipeline.to_state())
    original = pipeline._native

    class InferenceOnlyNative:
        transform_array = original.transform_array

        def to_state(self):
            pytest.fail("inference must not serialize the fitted dictionaries")

    monkeypatch.setattr(pipeline, "_native", InferenceOnlyNative())
    for _ in range(2):
        _assert_object_equivalent(pipeline, np.arange(9).reshape(3, 3), ["a", "b", "c"])
    assert pipeline.n_features_in_ == before["n_features_in_"]
    assert pipeline.feature_names_in_ == before["feature_names_in_"]
    assert pipeline.cat_feature_indices_ == before["cat_feature_indices_"]
    assert pipeline.output_feature_names_ == before["output_feature_names_"]


@pytest.mark.parametrize("format_version", [3, 4])
def test_loaded_numeric_pipeline_keeps_schema_and_conversion(format_version):
    state = _pipeline(names=["a", "b", "c"]).to_state()
    state["feature_pipeline_format_version"] = format_version
    pipeline = FeaturePipeline.from_state(state)
    _assert_object_equivalent(pipeline, np.arange(9).reshape(3, 3), ["a", "b", "c"])
    assert pipeline.to_state() == state
    with pytest.raises(ValueError, match="feature names"):
        pipeline.transform_array(np.zeros((2, 3)), feature_names=["c", "b", "a"])
    with pytest.raises(ValueError, match="feature count"):
        pipeline.transform_array(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="2D"):
        pipeline.transform_array(np.zeros(3))
    transformed, _, names = pipeline.transform_array(np.empty((0, 3)))
    assert transformed.shape == (0, 3)
    assert names == ["a", "b", "c"]


def test_numeric_frame_extraction_preserves_typed_arrays_but_fit_keeps_object_path():
    data = np.arange(12, dtype=np.int64).reshape(4, 3)
    pipeline = _pipeline()
    typed, _ = pipeline._extract_frame(data, allow_numeric=True)
    training, _ = pipeline._extract_frame(data)
    assert typed is data
    assert training.dtype == object
    # Lists remain object arrays: inferring a common dtype can change scalars.
    mixed, _ = pipeline._extract_frame([[2**63 - 1, 1.0, True]], allow_numeric=True)
    assert mixed.dtype == object
