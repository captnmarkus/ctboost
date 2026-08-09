import numpy as np
import pytest

import ctboost


pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")


def _regression_data():
    x0 = np.linspace(-2.0, 2.0, 80, dtype=np.float32)
    x1 = np.sin(x0).astype(np.float32)
    y = (1.7 * x0 - 0.4 * x1).astype(np.float32)
    return x0, x1, y


def test_pool_accepts_arrow_table_and_columnar_metadata_vectors():
    x0, x1, y = _regression_data()
    table = pa.table({"signal": x0, "curve": x1})
    weights = pa.array(np.linspace(0.5, 1.5, y.size, dtype=np.float32))

    pool = ctboost.Pool(table, pa.array(y), weight=weights)

    assert pool.num_rows == y.size
    assert pool.num_cols == 2
    assert pool.feature_names == ["signal", "curve"]
    np.testing.assert_allclose(pool.data, np.column_stack([x0, x1]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pool.label, y, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pool.weight, np.asarray(weights), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "frame_factory",
    [
        lambda values: pa.table(values),
        lambda values: pl.DataFrame(values),
    ],
)
def test_sklearn_pipeline_accepts_columnar_categorical_frames(frame_factory):
    rng = np.random.default_rng(404)
    row_count = 120
    city = np.asarray(["Berlin", "Paris", "Rome"] * 40, dtype=object)
    value = rng.normal(size=row_count).astype(np.float32)
    target = (value + (city == "Paris") * 1.5 - (city == "Rome") * 0.7).astype(np.float32)
    frame = frame_factory({"city": city.tolist(), "value": value})

    model = ctboost.CTBoostRegressor(
        iterations=12,
        max_depth=2,
        alpha=1.0,
        cat_features=["city"],
        ordered_ctr=True,
        random_seed=17,
    ).fit(frame, target)

    prediction = model.predict(frame)
    assert prediction.shape == target.shape
    assert np.all(np.isfinite(prediction))
    assert model.feature_names_in_.tolist() == ["city", "value"]


def test_classifier_accepts_polars_string_labels():
    x0, x1, _ = _regression_data()
    frame = pl.DataFrame({"signal": x0, "curve": x1})
    labels = pl.Series("label", np.where(x0 > 0.0, "up", "down"))

    model = ctboost.CTBoostClassifier(
        iterations=10,
        max_depth=2,
        alpha=1.0,
        random_seed=9,
    ).fit(frame, labels)

    prediction = model.predict(frame)
    assert set(prediction.tolist()) == {"down", "up"}


def test_prepare_pool_accepts_arrow_external_memory(tmp_path):
    x0, x1, y = _regression_data()
    table = pa.table({"signal": x0, "curve": x1})

    pool = ctboost.prepare_pool(
        table,
        pa.array(y),
        external_memory=True,
        external_memory_dir=tmp_path / "arrow-pool",
    )

    assert pool.feature_names == ["signal", "curve"]
    np.testing.assert_allclose(pool.data, np.column_stack([x0, x1]), rtol=0.0, atol=0.0)


def test_prepare_pool_external_memory_allows_unlabeled_prediction_data(tmp_path):
    x0, x1, _ = _regression_data()
    matrix = np.column_stack([x0, x1]).astype(np.float32)

    pool = ctboost.prepare_pool(
        matrix,
        None,
        external_memory=True,
        external_memory_dir=tmp_path / "unlabeled-pool",
    )

    assert pool.label.size == 0
    np.testing.assert_allclose(pool.data, matrix, rtol=0.0, atol=0.0)


def test_polars_lazy_frame_requires_explicit_collection():
    lazy = pl.DataFrame({"value": [1.0, 2.0]}).lazy()
    with pytest.raises(TypeError, match=r"collect\(\)"):
        ctboost.Pool(lazy, [0.0, 1.0])


def test_pool_accepts_cpu_dlpack_matrix_and_label():
    class DLPackValue:
        def __init__(self, array):
            self.array = np.asarray(array)

        def __dlpack__(self, *args, **kwargs):
            return self.array.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self):
            return self.array.__dlpack_device__()

    x0, x1, y = _regression_data()
    matrix = np.column_stack([x0, x1]).astype(np.float32)
    pool = ctboost.Pool(DLPackValue(matrix), DLPackValue(y))

    np.testing.assert_allclose(pool.data, matrix, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pool.label, y, rtol=0.0, atol=0.0)


def test_pool_accepts_cuda_array_protocol_with_explicit_host_copy():
    class HostCopyCudaArray:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.__cuda_array_interface__ = {
                "shape": self.array.shape,
                "strides": self.array.strides,
                "typestr": self.array.dtype.str,
                "data": (1, False),
                "version": 3,
            }

        def get(self):
            return self.array.copy()

    x0, x1, y = _regression_data()
    matrix = np.column_stack([x0, x1]).astype(np.float32)
    pool = ctboost.Pool(HostCopyCudaArray(matrix), HostCopyCudaArray(y))

    np.testing.assert_allclose(pool.data, matrix, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pool.label, y, rtol=0.0, atol=0.0)
