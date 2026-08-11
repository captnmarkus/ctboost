import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

import ctboost
from ctboost._integration_utils import concat_partitions, split_feature_frame


class _Computed:
    def __init__(self, value):
        self.value = value

    def compute(self):
        return self.value


class _RayDataset:
    def __init__(self, frame):
        self.frame = frame
        self.map_call = None

    def to_pandas(self):
        return self.frame.copy()

    def map_batches(self, function, **kwargs):
        self.map_call = (function, kwargs)
        return self


class _SparkFrame:
    def __init__(self, frame):
        self.frame = frame
        self.columns = list(frame.columns)

    def select(self, *columns):
        return _SparkFrame(self.frame[list(columns)])

    def toPandas(self):
        return self.frame.copy()


def _regression_data():
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.2,
        random_state=151,
    )
    frame = pd.DataFrame(X.astype(np.float32), columns=[f"f{i}" for i in range(5)])
    return frame, y.astype(np.float32)


def _params():
    return {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 2,
        "alpha": 1.0,
        "lambda_l2": 1.0,
        "random_seed": 17,
    }


def test_partition_concat_and_frame_split_preserve_dataframe_schema():
    city_dtype = pd.CategoricalDtype(categories=["berlin", "oslo", "rome"])
    left = pd.DataFrame(
        {
            "city": pd.Categorical(["berlin", "oslo"], dtype=city_dtype),
            "value": [1.0, 2.0],
            "y": [0.0, 1.0],
        }
    )
    right = pd.DataFrame(
        {
            "city": pd.Categorical(["rome"], dtype=city_dtype),
            "value": [3.0],
            "y": [2.0],
        }
    )
    combined = concat_partitions([left, right])
    features, label, metadata = split_feature_frame(combined, label="y")

    assert list(features.columns) == ["city", "value"]
    assert str(features["city"].dtype) == "category"
    assert features["city"].tolist() == ["berlin", "oslo", "rome"]
    np.testing.assert_array_equal(label.to_numpy(), [0.0, 1.0, 2.0])
    assert metadata == {}
    np.testing.assert_array_equal(
        concat_partitions([np.ones((2, 2)), np.zeros((1, 2))]),
        np.asarray([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="must not include the label"):
        split_feature_frame(combined, label="y", feature_columns=["city", "y"])
    with pytest.raises(ValueError, match="metadata columns are missing"):
        split_feature_frame(combined, label="y", metadata_columns=["missing_weight"])


def test_dask_materialize_mode_and_partition_prediction_are_public_conveniences():
    dask_api = importlib.import_module("ctboost.dask")
    frame, label = _regression_data()
    booster = dask_api.train(
        None,
        _Computed(frame),
        _Computed(label),
        _params(),
        num_boost_round=8,
        mode="materialize",
    )
    expected = ctboost.train(frame, _params(), label=label, num_boost_round=8)

    np.testing.assert_allclose(booster.predict(frame), expected.predict(frame), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(
        dask_api.predict(booster, _Computed(frame), mode="materialize"),
        booster.predict(frame),
    )


def test_dask_process_workers_run_native_distributed_training():
    dd = pytest.importorskip("dask.dataframe")
    distributed = pytest.importorskip("dask.distributed")
    dask_api = importlib.import_module("ctboost.dask")
    frame, label = _regression_data()
    full_frame = frame.assign(target=label)
    collection = dd.from_pandas(full_frame, npartitions=4)

    with distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    ) as cluster:
        with distributed.Client(cluster) as client:
            booster = dask_api.train(
                client,
                collection,
                "target",
                _params(),
                num_boost_round=6,
                num_workers=2,
                mode="distributed",
                timeout=60.0,
            )
            prediction = dask_api.predict(
                booster,
                collection.drop(columns=["target"]),
            ).compute()

    reference = ctboost.train(frame, _params(), label=label, num_boost_round=6)
    np.testing.assert_array_equal(np.asarray(prediction), reference.predict(frame))


def test_ray_collect_mode_and_lazy_prediction(monkeypatch):
    ray_api = importlib.import_module("ctboost.ray")
    monkeypatch.setattr(ray_api, "_require_ray", lambda: object())
    frame, label = _regression_data()
    dataset = _RayDataset(frame.assign(target=label))
    booster = ray_api.train(
        dataset,
        _params(),
        label="target",
        feature_columns=list(frame.columns),
        num_boost_round=8,
        mode="collect",
    )
    prediction_dataset = ray_api.predict(
        booster,
        dataset,
        feature_columns=list(frame.columns),
    )

    assert prediction_dataset is dataset
    function, kwargs = dataset.map_call
    batch_result = function(
        dataset.frame,
        **kwargs["fn_kwargs"],
    )
    assert list(batch_result.columns) == ["prediction"]
    np.testing.assert_array_equal(batch_result["prediction"], booster.predict(frame))
    assert kwargs["udf_modifying_row_count"] is False


def test_ray_eval_auto_features_exclude_optional_training_metadata(monkeypatch):
    ray_api = importlib.import_module("ctboost.ray")
    monkeypatch.setattr(ray_api, "_require_ray", lambda: object())
    frame, label = _regression_data()
    weighted = frame.assign(
        target=label,
        sample_weight=np.linspace(0.5, 1.5, len(frame), dtype=np.float32),
    )
    model = ray_api.train(
        _RayDataset(weighted),
        _params(),
        label="target",
        weight="sample_weight",
        eval_set=_RayDataset(weighted),
        num_boost_round=3,
        mode="collect",
    )
    assert len(model.feature_names) == frame.shape[1]


def test_spark_collect_adapter_preserves_named_features_and_is_explicit(monkeypatch):
    spark_api = importlib.import_module("ctboost.spark")
    monkeypatch.setattr(spark_api, "_require_pyspark", lambda: object())
    frame, label = _regression_data()
    spark_frame = _SparkFrame(frame.assign(target=label, sample_weight=np.linspace(0.5, 1.5, len(frame))))
    model = spark_api.train(
        spark_frame,
        _params(),
        label_col="target",
        feature_cols=list(frame.columns),
        weight_col="sample_weight",
        num_boost_round=8,
        mode="collect",
    )

    assert isinstance(model, spark_api.SparkCTBoostModel)
    assert list(model.feature_cols) == list(frame.columns)
    assert model.booster.feature_names == list(frame.columns)
    np.testing.assert_array_equal(model.booster.predict(frame), model.booster.predict(frame))

    try:
        spark_api.train(
            spark_frame,
            _params(),
            label_col="target",
            feature_cols=list(frame.columns),
            mode="distributed",
        )
    except ValueError as exc:
        assert "only mode='collect'" in str(exc)
    else:
        raise AssertionError("Spark distributed mode must not silently collect")
