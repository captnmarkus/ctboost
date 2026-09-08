"""Boundary contracts for vectorized numeric inference and optional frames."""

import warnings

import numpy as np
import pytest

from ctboost import FeaturePipeline


@pytest.mark.parametrize("dtype", ["f2", "f4", ">f4", "f8", ">f8"])
@pytest.mark.parametrize("input_kind", ["numpy", "native", "pandas"])
@pytest.mark.parametrize("policy", ["raise", "warning_error"])
def test_signaling_nan_preserves_scalar_conversion_under_numpy_error_policy(
    dtype, input_kind, policy
):
    dtype = np.dtype(dtype)
    uint = np.dtype(dtype.str.replace("f", "u"))
    signaling = {2: 0x7C01, 4: 0x7F800001, 8: 0x7FF0000000000001}[dtype.itemsize]
    sign_bit = 1 << (dtype.itemsize * 8 - 1)
    values = np.array([signaling, signaling | sign_bit] * 8, dtype=uint).view(dtype).reshape(16, 1)
    pipeline = FeaturePipeline().fit(np.zeros((3, 1)), [0, 1, 2], feature_names=["value"])
    if input_kind == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(values, columns=["value"])
    else:
        data = values

    def transform():
        return (
            pipeline._native.transform_array(values, ["value"])[0]
            if input_kind == "native"
            else pipeline.transform_array(data, feature_names=["value"])[0]
        )

    previous = np.geterr()
    with warnings.catch_warnings(), np.errstate(all="raise" if policy == "raise" else "warn"):
        warnings.simplefilter("error", RuntimeWarning)
        active = np.geterr()
        try:
            boxed = data.to_numpy(dtype=object) if input_kind == "pandas" else np.asarray(values, dtype=object)
            expected = pipeline.transform_array(boxed, feature_names=["value"])[0]
        except (FloatingPointError, RuntimeWarning) as reference:
            # NumPy 1.24 also rejects scalar boxing; later NumPy permits it.
            with pytest.raises(type(reference)):
                transform()
        else:
            actual = transform()
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            assert np.isnan(actual).all()
        assert np.geterr() == active
    assert np.geterr() == previous


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean", "int64[pyarrow]"])
@pytest.mark.parametrize("has_missing", [False, True])
def test_pandas_extension_arrays_keep_object_conversion_contract(dtype, has_missing):
    pd = pytest.importorskip("pandas")
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    data = pd.DataFrame({"value": pd.Series([1] * 15 + [pd.NA if has_missing else 0], dtype=dtype)})
    pipeline = FeaturePipeline().fit(np.zeros((3, 1)), [0, 1, 2], feature_names=["value"])
    extracted, names = pipeline._extract_frame(data, allow_numeric=True)
    assert extracted.dtype == object and names == ["value"]
    if has_missing:
        assert extracted[-1, 0] is pd.NA
        with pytest.raises((RuntimeError, TypeError, ValueError)) as reference:
            pipeline.transform_array(data.to_numpy(dtype=object), feature_names=["value"])
        with pytest.raises(type(reference.value)):
            pipeline.transform_array(data)
    else:
        np.testing.assert_array_equal(
            pipeline.transform_array(data)[0][:, 0], np.asarray([1] * 15 + [0], dtype=np.float32)
        )


@pytest.mark.parametrize("kind", ["arrow_table", "arrow_batch", "polars"])
def test_columnar_numeric_nulls_and_names_keep_object_fallback(kind):
    integer = 2**63 - 2**38 - 1
    columns = {"integer": [integer, None, 0], "value": [1.25, None, -0.0]}
    if kind == "polars":
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame(columns)
    else:
        pa = pytest.importorskip("pyarrow")
        frame = pa.table(columns)
        if kind == "arrow_batch":
            frame = frame.to_batches()[0]
    pipeline = FeaturePipeline().fit(np.zeros((3, 2)), [0, 1, 2], feature_names=list(columns))
    reference, reference_names = pipeline._extract_frame(frame, allow_numeric=False)
    extracted, names = pipeline._extract_frame(frame, allow_numeric=True)
    assert extracted.dtype == object and names == reference_names
    np.testing.assert_array_equal(extracted, reference)
    try:
        expected, expected_cat, expected_names = pipeline._native.transform_array(reference, reference_names)
    except (RuntimeError, TypeError, ValueError) as error:
        # Older optional libraries can reject nulls or an unrecognized frame.
        with pytest.raises(type(error)):
            pipeline.transform_array(frame)
    else:
        actual, cat_features, output_names = pipeline.transform_array(frame)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert cat_features == expected_cat and output_names == expected_names


@pytest.mark.parametrize("dtype", ["complex128", "U3", "datetime64[D]", "timedelta64[D]"])
def test_unsupported_numeric_inputs_retain_object_path_errors(dtype):
    values = np.array([["bad"], ["no"]] if dtype == "U3" else [[1], [2]], dtype=dtype)
    pipeline = FeaturePipeline().fit(np.zeros((3, 1)), [0, 1, 2])
    with pytest.raises((RuntimeError, TypeError, ValueError)) as reference:
        pipeline.transform_array(values.astype(object))
    with pytest.raises(type(reference.value)):
        pipeline.transform_array(values)
