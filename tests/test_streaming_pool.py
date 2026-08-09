from pathlib import Path

import numpy as np
import pytest

from ctboost.streaming import PoolBatch, pool_from_batches
from ctboost.training import train


def test_pool_from_batches_is_one_pass_and_disk_backed(tmp_path: Path):
    visited = []

    def batches():
        for batch_index in range(3):
            visited.append(batch_index)
            start = batch_index * 4
            data = np.arange(start * 2, (start + 4) * 2, dtype=np.float32).reshape(4, 2)
            label = data[:, 0] * 0.5 - data[:, 1]
            yield PoolBatch(data, label, weight=np.full(4, batch_index + 1, dtype=np.float32))

    pool = pool_from_batches(
        batches(), directory=tmp_path, feature_names=["left", "right"]
    )

    assert visited == [0, 1, 2]
    assert pool.num_rows == 12
    assert pool.num_cols == 2
    assert pool.feature_names == ["left", "right"]
    assert pool.streaming_batch_count == 3
    assert pool.streaming_directory.parent == tmp_path
    assert (pool.streaming_directory / "data.npy").is_file()
    assert not (pool.streaming_directory / "staging").exists()
    np.testing.assert_array_equal(pool.weight, np.repeat([1.0, 2.0, 3.0], 4))

    prediction_data = pool.data.copy()
    model = train(pool, {"iterations": 3, "depth": 2, "random_seed": 4})
    assert model.predict(prediction_data).shape == (12,)


def test_pool_from_batches_accepts_mapping_tuple_and_existing_pool(tmp_path: Path):
    from ctboost import Pool, PoolBatch as PublicPoolBatch, pool_from_batches as public_factory

    assert PublicPoolBatch is PoolBatch
    assert public_factory is pool_from_batches

    first = {"data": np.array([[0.0], [1.0]]), "label": np.array([0.0, 1.0])}
    second = (np.array([[2.0], [3.0]]), np.array([2.0, 3.0]))
    third = Pool(np.array([[4.0], [5.0]]), np.array([4.0, 5.0]))
    pool = pool_from_batches([first, second, third], directory=tmp_path)
    np.testing.assert_array_equal(pool.data[:, 0], np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(pool.label, np.arange(6, dtype=np.float32))

    via_class = Pool.from_batches(
        [(np.array([[6.0], [7.0]]), np.array([6.0, 7.0]))],
        directory=tmp_path,
    )
    np.testing.assert_array_equal(via_class.label, [6.0, 7.0])


def test_pool_from_batches_preserves_existing_pool_feature_schema(tmp_path: Path):
    from ctboost import Pool

    first = Pool(
        np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        cat_features=[0],
        feature_names=["category_code", "value"],
    )
    second = Pool(
        np.asarray([[2.0, 3.0], [0.0, 4.0]], dtype=np.float32),
        np.asarray([2.0, 3.0], dtype=np.float32),
        cat_features=[0],
        feature_names=["category_code", "value"],
    )

    combined = pool_from_batches([first, second], directory=tmp_path)

    assert combined.cat_features == [0]
    assert combined.feature_names == ["category_code", "value"]

    mismatched = Pool(
        np.asarray([[3.0, 5.0]], dtype=np.float32),
        np.asarray([4.0], dtype=np.float32),
        cat_features=[1],
        feature_names=["category_code", "value"],
    )
    with pytest.raises(ValueError, match="categorical feature indices must match"):
        pool_from_batches([first, mismatched], directory=tmp_path)


def test_pool_from_batches_validates_schema_and_metadata(tmp_path: Path):
    with pytest.raises(ValueError, match="present in every batch"):
        pool_from_batches(
            [
                PoolBatch(np.ones((2, 2)), np.ones(2)),
                PoolBatch(np.ones((2, 2))),
            ],
            directory=tmp_path,
        )

    with pytest.raises(ValueError, match="expected 2"):
        pool_from_batches(
            [
                PoolBatch(np.ones((2, 2)), np.ones(2)),
                PoolBatch(np.ones((2, 3)), np.ones(2)),
            ],
            directory=tmp_path,
        )

    with pytest.raises(ValueError, match="at least one"):
        pool_from_batches(iter(()), directory=tmp_path)
