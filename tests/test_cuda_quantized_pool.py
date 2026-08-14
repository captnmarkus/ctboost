import gc
import weakref

import numpy as np
import pytest

import ctboost


def _schema_two_numeric_features():
    return {
        "num_bins_per_feature": [2, 2],
        "cut_offsets": [0, 1, 2],
        "cut_values": [0.5, 1.5],
        "categorical_mask": [0, 0],
        "missing_value_mask": [0, 0],
        "nan_mode": 1,
        "nan_modes": [1, 1],
    }


def _cuda_interface(*, shape=(3, 2), typestr="|u1", pointer=4096, **updates):
    interface = {
        "shape": shape,
        "typestr": typestr,
        "data": (pointer, False),
        "version": 3,
        "strides": None,
        "stream": None,
    }
    interface.update(updates)
    return interface


class _FakeCudaArray:
    def __init__(self, interface=None, host_value=None):
        self.__cuda_array_interface__ = (
            _cuda_interface() if interface is None else interface
        )
        self.host_value = host_value
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        if self.host_value is None:
            raise AssertionError("device-only construction must not call get()")
        return self.host_value


def _fake_pool(*, schema=None, interface=None, label=None):
    owner = _FakeCudaArray(interface=interface)
    resolved_label = (
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if label is None
        else np.asarray(label, dtype=np.float32)
    )
    pool = ctboost.Pool.from_cuda_quantized(
        owner,
        _schema_two_numeric_features() if schema is None else schema,
        resolved_label,
    )
    return pool, owner


def test_cuda_quantized_pool_parses_without_host_transfer_and_retains_owner():
    pool, owner = _fake_pool()

    assert pool.num_rows == 3
    assert pool.num_cols == 2
    assert pool.has_cuda_quantized_features is True
    assert pool._handle.has_cuda_quantized_features() is True
    assert owner.get_calls == 0
    assert pool._cuda_quantized_ref.source is owner
    assert pool._cuda_quantized_ref.__cuda_array_interface__["stream"] is None
    owner.__cuda_array_interface__["stream"] = 12345
    assert pool._cuda_quantized_ref.__cuda_array_interface__["stream"] is None
    np.testing.assert_array_equal(pool.label, np.asarray([0.0, 1.0, 0.0]))
    with pytest.raises(RuntimeError, match="device-only"):
        _ = pool.data

    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is not None
    del pool
    gc.collect()
    assert owner_ref() is None


def test_ordinary_pool_cuda_array_path_still_performs_explicit_host_copy():
    host_value = np.asarray(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32
    )
    owner = _FakeCudaArray(host_value=host_value)

    pool = ctboost.Pool(owner, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    assert owner.get_calls == 1
    assert pool.has_cuda_quantized_features is False
    np.testing.assert_array_equal(pool.data, host_value)


@pytest.mark.parametrize(
    ("interface", "message"),
    [
        (_cuda_interface(version=2), "version 3"),
        (_cuda_interface(typestr="<f4"), "uint8 or uint16"),
        (_cuda_interface(typestr=">u2"), "uint8 or uint16"),
        (_cuda_interface(shape=(3, 2, 1)), "2D shape"),
        (_cuda_interface(pointer=0), "non-null pointer"),
        (_cuda_interface(strides=(5, 1)), "C-contiguous or Fortran-contiguous"),
        (_cuda_interface(strides=(-2, 1)), "non-negative"),
        (_cuda_interface(stream=0), "stream must be"),
        (_cuda_interface(stream=2), "per-thread default stream marker 2"),
        (_cuda_interface(mask=object()), "masked"),
        (_cuda_interface(typestr="<u2", pointer=4097), "not aligned"),
    ],
)
def test_cuda_quantized_pool_rejects_invalid_cuda_array_interfaces(
    interface, message
):
    owner = _FakeCudaArray(interface=interface)
    with pytest.raises((TypeError, ValueError, RuntimeError), match=message):
        ctboost.Pool.from_cuda_quantized(
            owner,
            _schema_two_numeric_features(),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )
    assert owner.get_calls == 0


def test_cuda_quantized_pool_requires_all_cuda_array_interface_fields():
    interface = _cuda_interface()
    del interface["data"]
    owner = _FakeCudaArray(interface=interface)

    with pytest.raises(ValueError, match="missing required field 'data'"):
        ctboost.Pool.from_cuda_quantized(
            owner,
            _schema_two_numeric_features(),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            {
                **_schema_two_numeric_features(),
                "num_bins_per_feature": [2],
                "cut_offsets": [0, 1],
                "cut_values": [0.5],
                "categorical_mask": [0],
                "missing_value_mask": [0],
                "nan_modes": [1],
            },
            "feature count",
        ),
        (
            {**_schema_two_numeric_features(), "cut_offsets": [0, 2, 2]},
            "cut count is inconsistent",
        ),
        (
            {
                **_schema_two_numeric_features(),
                "cut_values": [1.0, 0.5],
                "cut_offsets": [0, 2, 2],
                "num_bins_per_feature": [3, 1],
            },
            "strictly increasing",
        ),
        (
            {**_schema_two_numeric_features(), "nan_mode": 9},
            "invalid nan_mode",
        ),
        (
            {
                **_schema_two_numeric_features(),
                "missing_value_mask": [1, 0],
                "nan_modes": [0, 1],
            },
            "nan_mode='Forbidden'",
        ),
    ],
)
def test_cuda_quantized_pool_rejects_inconsistent_schema(schema, message):
    owner = _FakeCudaArray()
    with pytest.raises(ValueError, match=message):
        ctboost.Pool.from_cuda_quantized(
            owner,
            schema,
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )


def test_uint8_cuda_quantized_pool_rejects_schema_wider_than_256_bins():
    schema = {
        "num_bins_per_feature": [257],
        "cut_offsets": [0, 256],
        "cut_values": [float(value) for value in range(256)],
        "categorical_mask": [0],
        "missing_value_mask": [0],
        "nan_mode": 1,
        "nan_modes": [1],
    }
    owner = _FakeCudaArray(interface=_cuda_interface(shape=(3, 1)))
    with pytest.raises(ValueError, match="more than 256 bins"):
        ctboost.Pool.from_cuda_quantized(
            owner, schema, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        )


def test_cuda_quantized_pool_requires_exact_categorical_schema_match():
    schema = {
        "num_bins_per_feature": [2, 2],
        "cut_offsets": [0, 2, 3],
        "cut_values": [0.0, 1.0, 1.5],
        "categorical_mask": [1, 0],
        "missing_value_mask": [0, 0],
        "nan_mode": 1,
        "nan_modes": [1, 1],
    }
    owner = _FakeCudaArray()

    with pytest.raises(ValueError, match="exactly match"):
        ctboost.Pool.from_cuda_quantized(
            owner,
            schema,
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            cat_features=[],
        )


def test_cuda_quantized_pool_training_fails_closed_on_cpu_task_type():
    pool, _ = _fake_pool()

    with pytest.raises(ValueError, match="requires task_type='GPU'"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "CPU"},
            num_boost_round=1,
        )


@pytest.mark.parametrize(
    ("extra_params", "message"),
    [
        ({"external_memory": True}, "cannot be combined with external_memory"),
        ({"distributed_world_size": 2}, "does not yet support distributed"),
    ],
)
def test_cuda_quantized_pool_training_rejects_unsupported_modes(
    tmp_path, extra_params, message
):
    pool, _ = _fake_pool()
    params = {
        "objective": "RMSE",
        "task_type": "GPU",
        **extra_params,
    }
    if params.get("distributed_world_size", 1) > 1:
        params["distributed_rank"] = 0
        params["distributed_root"] = str(tmp_path / "distributed")

    with pytest.raises(ValueError, match=message):
        ctboost.train(pool, params, num_boost_round=1)


def test_cuda_quantized_pool_training_rejects_eval_set_before_cuda_dispatch():
    pool, _ = _fake_pool()
    eval_pool = ctboost.Pool(
        np.zeros((3, 2), dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="does not yet support eval_set"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU"},
            num_boost_round=1,
            eval_set=eval_pool,
        )


def test_cuda_quantized_pool_cannot_be_used_as_an_eval_set():
    device_eval_pool, _ = _fake_pool()
    train_pool = ctboost.Pool(
        np.zeros((3, 2), dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="not yet supported in eval_set"):
        ctboost.train(
            train_pool,
            {"objective": "RMSE", "task_type": "GPU"},
            num_boost_round=1,
            eval_set=device_eval_pool,
        )


def test_cuda_quantized_pool_training_rejects_warm_start_before_cuda_dispatch():
    X = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    y = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    initial_model = ctboost.train(
        ctboost.Pool(X, y), {"objective": "RMSE"}, num_boost_round=1
    )
    pool, _ = _fake_pool(label=y)

    with pytest.raises(ValueError, match="does not yet support init_model"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU"},
            num_boost_round=1,
            init_model=initial_model,
        )


def test_cuda_quantized_pool_training_rejects_python_callback_surface():
    pool, _ = _fake_pool()

    with pytest.raises(ValueError, match="requires the native training loop"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU"},
            num_boost_round=1,
            callbacks=[lambda environment: None],
        )


def test_cpu_wheel_fails_closed_when_cuda_quantized_training_is_requested():
    if ctboost.build_info()["cuda_enabled"]:
        pytest.skip("test applies to CPU-only wheels")
    pool, _ = _fake_pool()

    with pytest.raises(RuntimeError, match="compiled without CUDA support"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU"},
            num_boost_round=1,
        )


def _cupy_with_device():
    if not ctboost.build_info()["cuda_enabled"]:
        pytest.skip("CUDA support is not compiled into this build")
    cupy = pytest.importorskip("cupy")
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device discovery failed: {exc}")
    if device_count < 1:
        pytest.skip("no CUDA device is visible")
    return cupy, device_count


def _quantize_numeric_matrix(data, schema):
    data = np.asarray(data, dtype=np.float32)
    dtype = (
        np.uint8
        if max(int(value) for value in schema["num_bins_per_feature"]) <= 256
        else np.uint16
    )
    result = np.empty(data.shape, dtype=dtype)
    for feature in range(data.shape[1]):
        if int(schema["categorical_mask"][feature]) != 0:
            raise AssertionError("GPU parity fixture expects numerical features")
        begin = int(schema["cut_offsets"][feature])
        end = int(schema["cut_offsets"][feature + 1])
        cuts = np.asarray(schema["cut_values"][begin:end], dtype=np.float32)
        result[:, feature] = np.searchsorted(
            cuts, data[:, feature], side="right"
        ).astype(dtype)
    return result


@pytest.mark.parametrize("order", ["C", "F"])
def test_cuda_quantized_pool_matches_host_quantized_gpu_training(order):
    cupy, _ = _cupy_with_device()
    rng = np.random.default_rng(15511)
    X = rng.normal(size=(384, 5)).astype(np.float32)
    y = (
        1.8 * X[:, 0]
        - 0.9 * X[:, 1]
        + 0.6 * X[:, 2] * X[:, 3]
    ).astype(np.float32)
    params = {
        "objective": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "max_depth": 3,
        "learning_rate": 0.15,
        "random_seed": 15511,
        "feature_test": "quadratic",
        "feature_test_adjustment": "bonferroni",
    }
    host_model = ctboost.train(
        ctboost.Pool(X, y), params, num_boost_round=5
    )
    schema = host_model.get_quantization_schema()
    host_bins = _quantize_numeric_matrix(X, schema)
    device_bins = cupy.array(host_bins, order=order)
    device_pool = ctboost.Pool.from_cuda_quantized(device_bins, schema, y)

    device_model = ctboost.train(device_pool, params, num_boost_round=5)

    np.testing.assert_allclose(
        device_model.predict(X),
        host_model.predict(X),
        rtol=1e-6,
        atol=1e-6,
    )
    assert device_model.get_quantization_schema() == schema


def test_cuda_quantized_pool_honors_non_default_producer_stream():
    cupy, _ = _cupy_with_device()
    schema = _schema_two_numeric_features()
    y = np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    stream = cupy.cuda.Stream(non_blocking=True)
    with stream:
        bins = cupy.asarray(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=cupy.uint8
        )
        bins += cupy.uint8(0)
        pool = ctboost.Pool.from_cuda_quantized(bins, schema, y)

    model = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "task_type": "GPU",
            "devices": "0",
            "max_depth": 2,
            "random_seed": 15512,
        },
        num_boost_round=2,
    )

    assert model.num_iterations_trained == 2


def test_cuda_quantized_pool_validates_device_bins_against_schema():
    cupy, _ = _cupy_with_device()
    bins = cupy.asarray([[0, 0], [1, 0], [2, 1]], dtype=cupy.uint8)
    pool = ctboost.Pool.from_cuda_quantized(
        bins,
        _schema_two_numeric_features(),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="outside its schema range"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU", "devices": "0"},
            num_boost_round=1,
        )


def test_cuda_quantized_pool_rejects_declared_span_past_device_allocation():
    cupy, _ = _cupy_with_device()
    allocation = int(cupy.cuda.runtime.malloc(8))
    pool = None
    try:
        owner = _FakeCudaArray(
            interface=_cuda_interface(
                shape=(2, 1),
                pointer=allocation + 7,
            )
        )
        schema = {
            "num_bins_per_feature": [2],
            "cut_offsets": [0, 1],
            "cut_values": [0.5],
            "categorical_mask": [0],
            "missing_value_mask": [0],
            "nan_mode": 1,
            "nan_modes": [1],
        }
        pool = ctboost.Pool.from_cuda_quantized(
            owner,
            schema,
            np.asarray([0.0, 1.0], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="outside its CUDA device allocation"):
            ctboost.train(
                pool,
                {"objective": "RMSE", "task_type": "GPU", "devices": "0"},
                num_boost_round=1,
            )
    finally:
        del pool
        cupy.cuda.runtime.free(allocation)


def test_cuda_quantized_pool_uint16_device_copy_trains():
    cupy, _ = _cupy_with_device()
    num_bins = 300
    host_bins = (np.arange(600, dtype=np.uint16) % num_bins).reshape(-1, 1)
    schema = {
        "num_bins_per_feature": [num_bins],
        "cut_offsets": [0, num_bins - 1],
        "cut_values": [float(value) + 0.5 for value in range(num_bins - 1)],
        "categorical_mask": [0],
        "missing_value_mask": [0],
        "nan_mode": 1,
        "nan_modes": [1],
    }
    y = (host_bins[:, 0].astype(np.float32) / num_bins).copy()
    pool = ctboost.Pool.from_cuda_quantized(cupy.asarray(host_bins), schema, y)

    model = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "task_type": "GPU",
            "devices": "0",
            "max_depth": 2,
            "random_seed": 15513,
        },
        num_boost_round=2,
    )

    assert model.num_iterations_trained == 2


def test_cuda_quantized_pool_rejects_multi_device_workspace():
    cupy, _ = _cupy_with_device()
    bins = cupy.zeros((3, 2), dtype=cupy.uint8)
    pool = ctboost.Pool.from_cuda_quantized(
        bins,
        _schema_two_numeric_features(),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="exactly one CUDA device"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU", "devices": "0,1"},
            num_boost_round=1,
        )


def test_cuda_quantized_pool_rejects_source_and_training_device_mismatch():
    cupy, device_count = _cupy_with_device()
    if device_count < 2:
        pytest.skip("two CUDA devices are required for device mismatch validation")
    with cupy.cuda.Device(1):
        bins = cupy.zeros((3, 2), dtype=cupy.uint8)
        pool = ctboost.Pool.from_cuda_quantized(
            bins,
            _schema_two_numeric_features(),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        )

    with pytest.raises(ValueError, match="same CUDA device"):
        ctboost.train(
            pool,
            {"objective": "RMSE", "task_type": "GPU", "devices": "0"},
            num_boost_round=1,
        )
