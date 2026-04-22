import numpy as np
import pytest
import ctboost
import ctboost.core as ctcore
import ctboost._core as _core

def test_pool_reports_dimensions_and_column_major_storage():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    label = np.array([0.0, 1.0], dtype=np.float32)

    pool = ctboost.Pool(data, label, cat_features=[1])

    assert pool.num_rows == 2
    assert pool.num_cols == 3
    assert pool.cat_features == [1]
    np.testing.assert_array_equal(
        pool.feature_data,
        np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(pool.label, label)
    np.testing.assert_array_equal(pool.data, data)

def test_pool_rejects_mismatched_label_length():
    data = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    label = np.random.default_rng(1).normal(size=(3,)).astype(np.float32)

    with pytest.raises(ValueError, match="label size must match"):
        ctboost.Pool(data, label)

def test_pool_accepts_pandas_dataframe_with_categorical_columns():
    pd = pytest.importorskip("pandas")

    data = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "city": pd.Categorical(["berlin", "paris", "berlin"]),
            "segment": pd.Series(["retail", "enterprise", "retail"], dtype="string"),
        }
    )
    label = pd.Series([0.0, 1.0, 0.0], dtype="float32")

    pool = ctboost.Pool(data, label, cat_features=[0])

    assert pool.num_rows == 3
    assert pool.num_cols == 3
    assert pool.cat_features == [0, 1, 2]
    assert pool.feature_names == ["numeric", "city", "segment"]
    assert pool.data.dtype == np.float32
    np.testing.assert_array_equal(pool.label, label.to_numpy())
    assert set(np.unique(pool.data[:, 1]).tolist()) == {0.0, 1.0}
    assert set(np.unique(pool.data[:, 2]).tolist()) == {0.0, 1.0}

def test_pool_preserves_ranking_metadata_and_baseline():
    data = np.asarray(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 0.0],
            [-1.0, 1.0],
        ],
        dtype=np.float32,
    )
    label = np.zeros(4, dtype=np.float32)
    group_id = np.asarray([0, 0, 1, 1], dtype=np.int64)
    group_weight = np.asarray([1.0, 1.0, 2.5, 2.5], dtype=np.float32)
    subgroup_id = np.asarray([0, 1, 0, 1], dtype=np.int64)
    baseline = np.asarray([0.25, -0.1, 0.4, -0.3], dtype=np.float32)
    pairs = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    pairs_weight = np.asarray([1.5, 0.75], dtype=np.float32)

    pool = ctboost.Pool(
        data,
        label,
        group_id=group_id,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
    )

    np.testing.assert_array_equal(pool.group_id, group_id)
    np.testing.assert_allclose(pool.group_weight, group_weight, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(pool.subgroup_id, subgroup_id)
    np.testing.assert_allclose(pool.baseline, baseline, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(pool.pairs, pairs)
    np.testing.assert_allclose(pool.pairs_weight, pairs_weight, rtol=0.0, atol=0.0)

def test_pool_preserves_schema_metadata():
    data = np.asarray([[1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
    label = np.asarray([0.0, 1.0], dtype=np.float32)

    pool = ctboost.Pool(
        data,
        label,
        feature_names=["score", "city_code"],
        column_roles={"score": "numeric", "city_code": "categorical"},
        feature_metadata={
            "score": {"description": "normalized score"},
            "city_code": {"source": "city_lookup"},
        },
        categorical_schema={"city_code": {"categories": ["berlin", "paris"]}},
    )

    assert pool.feature_names == ["score", "city_code"]
    assert pool.column_roles == ["numeric", "categorical"]
    assert pool.feature_metadata == {
        "score": {"description": "normalized score"},
        "city_code": {"source": "city_lookup"},
    }
    assert pool.categorical_schema == {"city_code": {"categories": ["berlin", "paris"]}}

def test_pool_passes_fortran_ordered_matrix_to_native(monkeypatch):
    captured = {}
    original_pool = ctcore._core.Pool

    class FakePoolHandle:
        def __init__(self, data, label, cat_features, weight, group_id, *metadata):
            captured["data"] = data
            captured["label"] = label
            captured["cat_features"] = cat_features
            captured["weight"] = weight
            captured["group_id"] = group_id

        def num_rows(self):
            return captured["data"].shape[0]

        def num_cols(self):
            return captured["data"].shape[1]

        def cat_features(self):
            return list(captured["cat_features"])

        def feature_data(self):
            return np.array(captured["data"], dtype=np.float32, order="F").ravel(order="F")

        def label(self):
            return captured["label"]

        def set_feature_storage_releasable(self, releasable):
            captured["releasable_feature_storage"] = releasable

    monkeypatch.setattr(ctcore._core, "Pool", FakePoolHandle)

    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    label = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    pool = ctboost.Pool(data, label, cat_features=[2])

    assert pool.num_rows == 4
    assert pool.num_cols == 3
    assert pool.cat_features == [2]
    assert captured["data"].flags.f_contiguous
    assert not captured["data"].flags.c_contiguous
    np.testing.assert_array_equal(captured["label"], label)

    monkeypatch.setattr(ctcore._core, "Pool", original_pool)

def test_dataframe_pool_preserves_fortran_layout_to_native(monkeypatch):
    pd = pytest.importorskip("pandas")
    captured = {}
    original_pool = ctcore._core.Pool

    class FakePoolHandle:
        def __init__(self, data, label, cat_features, weight, group_id, *metadata):
            captured["data"] = data
            captured["cat_features"] = cat_features

        def num_rows(self):
            return captured["data"].shape[0]

        def num_cols(self):
            return captured["data"].shape[1]

        def cat_features(self):
            return list(captured["cat_features"])

        def feature_data(self):
            return np.array(captured["data"], dtype=np.float32, order="F").ravel(order="F")

        def label(self):
            return np.zeros(captured["data"].shape[0], dtype=np.float32)

        def set_feature_storage_releasable(self, releasable):
            captured["releasable_feature_storage"] = releasable

    monkeypatch.setattr(ctcore._core, "Pool", FakePoolHandle)

    data = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "city": pd.Categorical(["berlin", "paris", "berlin"]),
        }
    )
    label = pd.Series([0.0, 1.0, 0.0], dtype="float32")

    pool = ctboost.Pool(data, label)

    assert pool.cat_features == [1]
    assert captured["data"].flags.f_contiguous
    assert not captured["data"].flags.c_contiguous

    monkeypatch.setattr(ctcore._core, "Pool", original_pool)
