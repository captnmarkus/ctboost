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


@pytest.mark.parametrize("dtype", ["f2", "f4", ">f4", "f8", ">f8"])
@pytest.mark.parametrize("input_kind", ["numpy", "native", "pandas", "pandas_mixed"])
@pytest.mark.parametrize("policy", ["warn", "call", "log"])
def test_signaling_nan_preserves_numpy_warning_and_callback_policy(dtype, input_kind, policy):
    dtype = np.dtype(dtype)
    uint = np.dtype(dtype.str.replace("f", "u"))
    signaling = {2: 0x7C01, 4: 0x7F800001, 8: 0x7FF0000000000001}[dtype.itemsize]
    values = np.ones((16, 2), dtype=dtype)
    values.view(uint)[-1, -1] = signaling
    if input_kind.startswith("pandas"):
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(values, columns=["a", "b"])
        if input_kind == "pandas_mixed":
            data["a"] = np.arange(16, dtype=np.int64)
    else:
        data = values
    pipeline = FeaturePipeline().fit(np.zeros((3, 2)), [0, 1, 2], feature_names=["a", "b"])

    class Handler:
        def __init__(self):
            self.events = []

        def __call__(self, error, flag):
            self.events.append((error, flag))

        def write(self, message):
            self.events.append(message)

    previous = np.geterr()
    previous_handler = np.geterrcall()
    outcomes = []
    try:
        for reference in [True, False]:
            handler = Handler()
            np.seterrcall(handler)
            with warnings.catch_warnings(record=True) as caught, np.errstate(invalid=policy):
                warnings.simplefilter("always", RuntimeWarning)
                if reference:
                    boxed = data.to_numpy(dtype=object) if input_kind.startswith("pandas") else values.astype(object)
                    result = pipeline._native.transform_array(boxed, ["a", "b"])[0]
                elif input_kind == "native":
                    result = pipeline._native.transform_array(values, ["a", "b"])[0]
                else:
                    result = pipeline.transform_array(data, feature_names=["a", "b"])[0]
                outcomes.append((result.view(np.uint32), [(item.category, str(item.message)) for item in caught], handler.events))
        np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
        assert outcomes[0][1:] == outcomes[1][1:]
        assert np.geterr() == previous
    finally:
        np.seterrcall(previous_handler)


@pytest.mark.parametrize("dtype", ["f4", ">f4"])
@pytest.mark.parametrize("layout", ["F", "reversed", "unaligned"])
def test_float32_late_signaling_nan_restores_entire_object_transform(dtype, layout):
    values = np.arange(32, dtype=np.dtype(dtype)).reshape(16, 2)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "reversed":
        values = values[::-1, ::-1]
    else:
        values = np.ndarray(values.shape, dtype=dtype, buffer=bytearray(values.nbytes + 1), offset=1)
        values[:] = np.arange(32).reshape(16, 2)
    values.view(np.dtype(dtype).str.replace("f", "u"))[-1, -1] = 0xFF800001
    pipeline = FeaturePipeline().fit(np.zeros((3, 2)), [0, 1, 2])
    with np.errstate(invalid="ignore"):
        expected = pipeline._native.transform_array(values.astype(object), None)[0]
        actual = pipeline._native.transform_array(values, None)[0]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("dtype", ["float32", ">f4"])
@pytest.mark.parametrize("mixed", [False, True])
def test_float32_dataframe_does_not_widen_before_native_conversion(dtype, mixed):
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame(np.ones((16, 2), dtype=dtype), columns=["a", "b"])
    if mixed:
        data["a"] = np.arange(16, dtype=np.int64)
    extracted, names = FeaturePipeline._extract_frame(data, allow_numeric=True)
    assert extracted.dtype == (object if mixed else np.dtype(dtype))
    assert names == ["a", "b"]


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
    assert extracted.shape == reference.shape
    for actual_value, reference_value in zip(extracted.flat, reference.flat):
        if isinstance(reference_value, (float, np.floating)) and np.isnan(reference_value):
            assert isinstance(actual_value, (float, np.floating)) and np.isnan(actual_value)
        else:
            assert actual_value == reference_value
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
