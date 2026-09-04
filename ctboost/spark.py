"""PySpark DataFrame convenience adapter for CTBoost.

Distributed training uses Spark barrier tasks to bind one DataFrame partition
to each native CTBoost TCP rank.  The explicit ``collect`` mode remains
available for small datasets and local debugging.  Inference is partitioned
through a pandas UDF.
"""

from __future__ import annotations

import os
import pickle
import socket
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Mapping, Optional, Sequence
from uuid import uuid4

import numpy as np

from ._integration_utils import (
    allocate_tcp_endpoint,
    prediction_columns,
    split_feature_frame,
    train_distributed_shard,
)


def _require_pyspark() -> Any:
    try:
        import pyspark
    except ImportError as exc:
        raise ImportError(
            "CTBoost's Spark adapter requires PySpark and Arrow. Install 'ctboost[spark]'."
        ) from exc
    return pyspark


def _vector_matrix(values: Any) -> np.ndarray:
    rows = []
    for value in values:
        if value is None:
            raise ValueError("Spark vector features cannot contain null values")
        to_array = getattr(value, "toArray", None)
        rows.append(np.asarray(to_array() if callable(to_array) else value, dtype=np.float32))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    width = int(rows[0].size)
    if any(int(row.size) != width for row in rows):
        raise ValueError("Spark vector features must have a fixed width")
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def _barrier_slot_capacity(dataframe: Any, fallback: int) -> int:
    """Estimate concurrent barrier slots from Spark's configured parallelism."""
    try:
        capacity = int(dataframe.sparkSession.sparkContext.defaultParallelism)
    except (AttributeError, TypeError, ValueError):
        capacity = int(fallback)
    return max(capacity, 1)


def _encode_distributed_group_ids(dataframe: Any, group_column: str) -> Any:
    """Map keys globally without collecting group keys or training rows."""
    from pyspark.sql.types import LongType, StructField, StructType

    field = dataframe.schema[str(group_column)]
    key_name = f"__ctboost_group_key_{uuid4().hex}"
    id_name = f"__ctboost_group_id_{uuid4().hex}"
    distinct_values = dataframe.select(str(group_column)).distinct().rdd.map(
        lambda row: row[0]
    )
    mapping_rows = distinct_values.zipWithIndex().map(
        lambda item: (item[0], int(item[1]))
    )
    mapping_schema = StructType(
        [
            StructField(key_name, field.dataType, field.nullable),
            StructField(id_name, LongType(), False),
        ]
    )
    mapping = dataframe.sparkSession.createDataFrame(mapping_rows, mapping_schema)
    joined = dataframe.join(
        mapping,
        dataframe[str(group_column)].eqNullSafe(mapping[key_name]),
        "inner",
    )
    return (
        joined.drop(dataframe[str(group_column)])
        .drop(mapping[key_name])
        .withColumnRenamed(id_name, str(group_column))
    )


def _partition_distributed_group_ids(
    dataframe: Any,
    group_column: str,
    num_workers: int,
) -> Any:
    """Assign each encoded ranking group to an exact, non-relocating worker bucket."""
    group_index = list(dataframe.columns).index(str(group_column))
    schema = dataframe.schema
    partitioned_rows = (
        dataframe.rdd.map(
            lambda row: (int(row[group_index]) % int(num_workers), row)
        )
        .partitionBy(int(num_workers), lambda bucket: int(bucket))
        .values()
    )
    return dataframe.sparkSession.createDataFrame(partitioned_rows, schema)


def _collect_training_frame(
    dataframe: Any,
    *,
    label_col: str,
    feature_cols: Optional[Sequence[str]],
    features_col: Optional[str],
    metadata_columns: Mapping[str, str],
) -> tuple[Any, Any, Dict[str, Any], Optional[Sequence[str]]]:
    if feature_cols is not None and features_col is not None:
        raise ValueError("pass either feature_cols or features_col, not both")
    selected = [label_col, *metadata_columns.values()]
    if features_col is not None:
        selected.append(features_col)
    elif feature_cols is not None:
        selected.extend(feature_cols)
    else:
        selected.extend(
            column
            for column in dataframe.columns
            if column not in {label_col, *metadata_columns.values()}
        )
    selected = list(dict.fromkeys(selected))
    frame = dataframe.select(*selected).toPandas()
    if features_col is not None:
        features = _vector_matrix(frame[features_col])
        labels = frame[label_col]
        metadata = {
            argument: frame[column]
            for argument, column in metadata_columns.items()
        }
        return features, labels, metadata, None
    features, labels, by_column = split_feature_frame(
        frame,
        label=label_col,
        feature_columns=feature_cols,
        metadata_columns=list(metadata_columns.values()),
    )
    metadata = {
        argument: by_column[column]
        for argument, column in metadata_columns.items()
        if column in by_column
    }
    return features, labels, metadata, list(features.columns)


def _spark_partition_train(
    partition_index: int,
    rows: Any,
    *,
    selected_columns: Sequence[str],
    label_col: str,
    feature_cols: Optional[Sequence[str]],
    features_col: Optional[str],
    metadata_columns: Mapping[str, str],
    params: Mapping[str, Any],
    world_size: int,
    distributed_root: Optional[str],
    run_id: str,
    timeout: float,
    num_boost_round: Optional[int],
    train_kwargs: Mapping[str, Any],
) -> Any:
    import pandas as pd
    from pyspark import BarrierTaskContext

    context = BarrierTaskContext.get()
    if context is None:
        raise RuntimeError("distributed Spark training requires a barrier task context")
    rank = int(context.partitionId())
    if rank != int(partition_index):
        raise RuntimeError("Spark barrier partition and CTBoost rank identities differ")
    if len(context.getTaskInfos()) != int(world_size):
        raise RuntimeError("Spark barrier stage size does not match the CTBoost world size")

    records = [
        row.asDict(recursive=False) if hasattr(row, "asDict") else dict(row)
        for row in rows
    ]
    if not records:
        raise ValueError("distributed Spark training requires at least one row per rank")
    frame = pd.DataFrame.from_records(records, columns=list(selected_columns))
    if features_col is not None:
        features = _vector_matrix(frame[features_col])
        labels = frame[label_col]
        resolved_feature_names = None
    else:
        features, labels, _ = split_feature_frame(
            frame,
            label=label_col,
            feature_columns=feature_cols,
            metadata_columns=list(metadata_columns.values()),
        )
        resolved_feature_names = None if feature_cols is None else list(feature_cols)
    row_metadata = {
        argument: frame[column]
        for argument, column in metadata_columns.items()
    }

    identity = f"{socket.gethostname()}:{os.getpid()}"
    worker_identities = context.allGather(identity)
    if len(set(worker_identities)) != int(world_size):
        raise ValueError(
            "distributed Spark training requires one Python worker process per rank"
        )

    if distributed_root is None:
        root_candidate = allocate_tcp_endpoint(socket.gethostname())[0] if rank == 0 else ""
        root_candidates = context.allGather(root_candidate)
        concrete_root = str(root_candidates[0])
        if not concrete_root:
            raise RuntimeError("Spark rank zero did not publish a distributed endpoint")
    else:
        concrete_root = str(distributed_root)

    local_params = dict(params)
    previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    uses_gpu = str(local_params.get("task_type", "CPU")).strip().lower() == "gpu"
    try:
        if uses_gpu:
            resources = context.resources()
            gpu_resource = resources.get("gpu")
            addresses = [] if gpu_resource is None else list(gpu_resource.addresses)
            if not addresses:
                raise ValueError(
                    "distributed Spark GPU training requires a Spark GPU resource "
                    "assigned to every barrier task"
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in addresses)
            local_params["devices"] = ",".join(str(index) for index in range(len(addresses)))

        model = train_distributed_shard(
            features,
            labels,
            local_params,
            rank=rank,
            world_size=world_size,
            distributed_root=concrete_root,
            run_id=run_id,
            timeout=timeout,
            num_boost_round=num_boost_round,
            row_metadata=row_metadata,
            train_kwargs={
                **dict(train_kwargs),
                **(
                    {}
                    if resolved_feature_names is None
                    else {"feature_names": resolved_feature_names}
                ),
            },
        )
    finally:
        if previous_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible_devices

    yield rank, pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL) if rank == 0 else b""


def _train_distributed_spark_frame(
    dataframe: Any,
    *,
    selected_columns: Sequence[str],
    label_col: str,
    feature_cols: Optional[Sequence[str]],
    features_col: Optional[str],
    metadata_columns: Mapping[str, str],
    params: Mapping[str, Any],
    num_workers: int,
    distributed_root: Optional[str],
    run_id: str,
    timeout: float,
    num_boost_round: Optional[int],
    train_kwargs: Mapping[str, Any],
) -> Any:
    selected = dataframe.select(*selected_columns)
    group_id_column = metadata_columns.get("group_id")
    if group_id_column is None:
        selected = selected.repartition(int(num_workers))
    else:
        distinct_group_count = (
            selected.select(str(group_id_column))
            .distinct()
            .limit(int(num_workers))
            .count()
        )
        if int(distinct_group_count) < int(num_workers):
            raise ValueError(
                "distributed Spark ranking requires at least num_workers distinct "
                "groups so every native rank receives data; reduce num_workers or "
                "use mode='collect'"
            )
        # A ranking group must never straddle native collective ranks. Map all
        # key types (including nulls) to contiguous global IDs, assign those
        # IDs round-robin with an exact RDD partitioner, and sort each rank for
        # Pool's contiguous-group contract.
        selected = _encode_distributed_group_ids(selected, str(group_id_column))
        selected = _partition_distributed_group_ids(
            selected,
            str(group_id_column),
            int(num_workers),
        ).sortWithinPartitions(str(group_id_column))
    worker = partial(
        _spark_partition_train,
        selected_columns=list(selected_columns),
        label_col=label_col,
        feature_cols=None if feature_cols is None else list(feature_cols),
        features_col=features_col,
        metadata_columns=dict(metadata_columns),
        params=dict(params),
        world_size=int(num_workers),
        distributed_root=distributed_root,
        run_id=run_id,
        timeout=float(timeout),
        num_boost_round=num_boost_round,
        train_kwargs=dict(train_kwargs),
    )
    results = selected.rdd.barrier().mapPartitionsWithIndex(worker).collect()
    root_payloads = [payload for rank, payload in results if int(rank) == 0 and payload]
    if len(root_payloads) != 1:
        raise RuntimeError("distributed Spark training did not return exactly one root model")
    return pickle.loads(root_payloads[0])


@dataclass
class SparkCTBoostModel:
    """A fitted CTBoost booster with Spark DataFrame transform metadata."""

    booster: Any
    feature_cols: Optional[Sequence[str]] = None
    features_col: Optional[str] = None

    def transform(
        self,
        dataframe: Any,
        *,
        prediction_col: str = "prediction",
    ) -> Any:
        """Append raw CTBoost predictions using a partitioned Arrow pandas UDF."""

        _require_pyspark()
        try:
            import pandas as pd
            from pyspark.sql import functions as F
            from pyspark.sql.types import ArrayType, DoubleType
        except ImportError as exc:
            raise ImportError(
                "Spark CTBoost inference requires pandas and pyarrow on every executor"
            ) from exc
        columns = prediction_columns(self.booster.prediction_dimension, prediction_col)
        return_type = DoubleType() if len(columns) == 1 else ArrayType(DoubleType())
        payload = pickle.dumps(self.booster, protocol=pickle.HIGHEST_PROTOCOL)
        feature_cols = None if self.feature_cols is None else list(self.feature_cols)
        use_vector = self.features_col is not None

        @F.pandas_udf(return_type)
        def predict_batch(*series: Any) -> Any:
            model = pickle.loads(payload)
            if use_vector:
                features = _vector_matrix(series[0])
            else:
                features = pd.concat(
                    [column.reset_index(drop=True) for column in series],
                    axis=1,
                )
                features.columns = feature_cols
            values = np.asarray(model.predict(features), dtype=np.float64)
            if values.ndim == 1:
                return pd.Series(values)
            return pd.Series([row.tolist() for row in values])

        if use_vector:
            return dataframe.withColumn(prediction_col, predict_batch(F.col(str(self.features_col))))
        if not feature_cols:
            raise ValueError("feature_cols are required for non-vector Spark inference")
        return dataframe.withColumn(
            prediction_col,
            predict_batch(*(F.col(column) for column in feature_cols)),
        )

    def save_model(self, path: Any, *, model_format: Optional[str] = None) -> None:
        self.booster.save_model(path, model_format=model_format)


def train(
    dataframe: Any,
    params: Mapping[str, Any],
    *,
    label_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    features_col: Optional[str] = None,
    num_boost_round: Optional[int] = None,
    mode: str = "auto",
    num_workers: Optional[int] = None,
    distributed_root: Optional[str] = None,
    run_id: Optional[str] = None,
    timeout: float = 600.0,
    weight_col: Optional[str] = None,
    group_id_col: Optional[str] = None,
    group_weight_col: Optional[str] = None,
    subgroup_id_col: Optional[str] = None,
    baseline_col: Optional[str] = None,
    eval_set: Any = None,
    **train_kwargs: Any,
) -> SparkCTBoostModel:
    """Fit CTBoost from a Spark DataFrame.

    ``mode='distributed'`` runs one native CTBoost rank in each Spark barrier
    task without collecting the training rows on the driver. ``mode='collect'``
    is an explicit driver-memory fallback. The returned model performs
    partitioned Spark inference through :meth:`SparkCTBoostModel.transform`.
    """

    _require_pyspark()
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"auto", "collect", "distributed"}:
        raise ValueError("mode must be one of: auto, collect, distributed")
    if float(timeout) <= 0.0:
        raise ValueError("timeout must be positive")
    metadata_columns = {
        name: column
        for name, column in (
            ("weight", weight_col),
            ("group_id", group_id_col),
            ("group_weight", group_weight_col),
            ("subgroup_id", subgroup_id_col),
            ("baseline", baseline_col),
        )
        if column is not None
    }
    if feature_cols is not None and features_col is not None:
        raise ValueError("pass either feature_cols or features_col, not both")
    missing_columns = [
        column
        for column in [str(label_col), *metadata_columns.values()]
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Spark columns are missing: {missing_columns}")
    if features_col is not None:
        if str(features_col) not in dataframe.columns:
            raise ValueError(f"Spark features column is missing: {features_col!r}")
        resolved_feature_cols = None
        selected_columns = [str(label_col), *metadata_columns.values(), str(features_col)]
    else:
        excluded = {str(label_col), *metadata_columns.values()}
        resolved_feature_cols = (
            [column for column in dataframe.columns if column not in excluded]
            if feature_cols is None
            else [str(column) for column in feature_cols]
        )
        if not resolved_feature_cols:
            raise ValueError("at least one Spark feature column is required")
        if len(set(resolved_feature_cols)) != len(resolved_feature_cols):
            raise ValueError("Spark feature columns must not contain duplicates")
        missing_features = [
            column for column in resolved_feature_cols if column not in dataframe.columns
        ]
        if missing_features:
            raise ValueError(f"Spark feature columns are missing: {missing_features}")
        leaked = [column for column in resolved_feature_cols if column in excluded]
        if leaked:
            raise ValueError(
                "Spark feature columns must not include labels or row metadata: "
                f"{leaked}"
            )
        selected_columns = [
            str(label_col),
            *metadata_columns.values(),
            *resolved_feature_cols,
        ]
    selected_columns = list(dict.fromkeys(selected_columns))

    source_partitions = (
        1
        if normalized_mode == "collect"
        else int(dataframe.rdd.getNumPartitions())
    )
    barrier_capacity = _barrier_slot_capacity(dataframe, source_partitions)
    resolved_workers = (
        min(source_partitions, barrier_capacity)
        if num_workers is None
        else int(num_workers)
    )
    if resolved_workers <= 0:
        raise ValueError("num_workers must be positive")
    if normalized_mode == "auto":
        if resolved_workers <= 1:
            raise ValueError(
                "automatic Spark training cannot schedule at least two barrier "
                "workers; "
                "increase Spark capacity or pass mode='collect' explicitly to allow "
                "driver-memory collection"
            )
        normalized_mode = "distributed"

    if normalized_mode == "distributed":
        if resolved_workers <= 1:
            raise ValueError("distributed Spark training requires num_workers >= 2")
        if resolved_workers > barrier_capacity:
            raise ValueError(
                "distributed Spark num_workers exceeds the estimated barrier task "
                f"capacity ({barrier_capacity}); reduce num_workers or adjust "
                "Spark resources"
            )
        if eval_set is not None:
            raise ValueError(
                "distributed Spark eval_set is not supported yet; use mode='collect' "
                "or omit eval_set"
            )
        booster = _train_distributed_spark_frame(
            dataframe,
            selected_columns=selected_columns,
            label_col=str(label_col),
            feature_cols=resolved_feature_cols,
            features_col=None if features_col is None else str(features_col),
            metadata_columns=metadata_columns,
            params=dict(params),
            num_workers=resolved_workers,
            distributed_root=distributed_root,
            run_id=str(run_id or f"spark-{uuid4().hex}"),
            timeout=float(timeout),
            num_boost_round=num_boost_round,
            train_kwargs=dict(train_kwargs),
        )
        return SparkCTBoostModel(
            booster=booster,
            feature_cols=resolved_feature_cols,
            features_col=features_col,
        )

    features, labels, metadata, resolved_feature_cols = _collect_training_frame(
        dataframe,
        label_col=str(label_col),
        feature_cols=feature_cols,
        features_col=features_col,
        metadata_columns=metadata_columns,
    )
    eager_eval_set = None
    if eval_set is not None:
        eval_features, eval_labels, _, _ = _collect_training_frame(
            eval_set,
            label_col=str(label_col),
            feature_cols=feature_cols,
            features_col=features_col,
            # Exclude training metadata columns from automatically selected
            # evaluation features when they are present in both frames.
            metadata_columns={
                name: column
                for name, column in metadata_columns.items()
                if column in eval_set.columns
            },
        )
        eager_eval_set = (eval_features, eval_labels)

    from .training import train as local_train

    booster = local_train(
        features,
        params,
        label=labels,
        num_boost_round=num_boost_round,
        eval_set=eager_eval_set,
        **metadata,
        **train_kwargs,
    )
    return SparkCTBoostModel(
        booster=booster,
        feature_cols=resolved_feature_cols,
        features_col=features_col,
    )


def predict(
    model: SparkCTBoostModel,
    dataframe: Any,
    *,
    prediction_col: str = "prediction",
) -> Any:
    return model.transform(dataframe, prediction_col=prediction_col)


__all__ = ["SparkCTBoostModel", "predict", "train"]
