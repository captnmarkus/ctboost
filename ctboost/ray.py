"""Ray Dataset integration for CTBoost."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence
from uuid import uuid4

import numpy as np

from ._integration_utils import (
    allocate_tcp_endpoint,
    prediction_columns,
    split_feature_frame,
    train_distributed_shard,
)


def _require_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise ImportError(
            "CTBoost's Ray integration requires Ray Data. Install 'ctboost[ray]'."
        ) from exc
    return ray


def _ray_endpoint_task() -> tuple[str, str]:
    import ray

    try:
        host = ray.util.get_node_ip_address()
    except AttributeError:
        host = None
    root, _ = allocate_tcp_endpoint(host)
    return root, str(ray.get_runtime_context().get_node_id())


def _ray_train_task(
    dataset_shard: Any,
    eval_shard: Any,
    label_column: str,
    eval_label_column: str,
    feature_columns: Optional[Sequence[str]],
    metadata_columns: Mapping[str, str],
    params: Mapping[str, Any],
    rank: int,
    world_size: int,
    root: str,
    run_id: str,
    timeout: float,
    num_boost_round: Optional[int],
    train_kwargs: Mapping[str, Any],
) -> Any:
    frame = dataset_shard.to_pandas()
    features, labels, metadata_by_column = split_feature_frame(
        frame,
        label=label_column,
        feature_columns=feature_columns,
        metadata_columns=list(metadata_columns.values()),
    )
    row_metadata = {
        argument: metadata_by_column[column]
        for argument, column in metadata_columns.items()
        if column in metadata_by_column
    }
    eval_set = None
    if eval_shard is not None:
        eval_frame = eval_shard.to_pandas()
        eval_features, eval_labels, _ = split_feature_frame(
            eval_frame,
            label=eval_label_column,
            feature_columns=feature_columns,
            metadata_columns=[
                column for column in metadata_columns.values() if column in eval_frame.columns
            ],
        )
        eval_set = (eval_features, eval_labels)
    return train_distributed_shard(
        features,
        labels,
        params,
        rank=rank,
        world_size=world_size,
        distributed_root=root,
        run_id=run_id,
        timeout=timeout,
        num_boost_round=num_boost_round,
        eval_set=eval_set,
        row_metadata=row_metadata,
        train_kwargs=train_kwargs,
    )


def train(
    dataset: Any,
    params: Mapping[str, Any],
    *,
    label: str,
    feature_columns: Optional[Sequence[str]] = None,
    num_boost_round: Optional[int] = None,
    num_workers: int = 1,
    mode: str = "auto",
    distributed_root: Optional[str] = None,
    run_id: Optional[str] = None,
    timeout: float = 600.0,
    num_cpus_per_worker: float = 1.0,
    weight: Optional[str] = None,
    group_id: Optional[str] = None,
    group_weight: Optional[str] = None,
    subgroup_id: Optional[str] = None,
    baseline: Optional[str] = None,
    eval_set: Any = None,
    eval_label: Optional[str] = None,
    **train_kwargs: Any,
) -> Any:
    """Train CTBoost from a Ray Dataset.

    ``label`` and optional row metadata are column names.  One Ray Dataset
    shard is consumed by each native CTBoost TCP rank.  Set ``mode='collect'``
    for an explicit driver-memory fallback.
    """

    ray = _require_ray()
    normalized_mode = str(mode).lower()
    if normalized_mode not in {"auto", "distributed", "collect"}:
        raise ValueError("mode must be one of: auto, distributed, collect")
    resolved_workers = int(num_workers)
    if resolved_workers <= 0:
        raise ValueError("num_workers must be positive")
    if float(num_cpus_per_worker) <= 0.0:
        raise ValueError("num_cpus_per_worker must be positive")
    if normalized_mode == "auto":
        normalized_mode = "distributed" if resolved_workers > 1 else "collect"

    metadata_columns = {
        name: column
        for name, column in (
            ("weight", weight),
            ("group_id", group_id),
            ("group_weight", group_weight),
            ("subgroup_id", subgroup_id),
            ("baseline", baseline),
        )
        if column is not None
    }
    if normalized_mode == "collect":
        from .training import train as local_train

        frame = dataset.to_pandas()
        features, labels, metadata_by_column = split_feature_frame(
            frame,
            label=label,
            feature_columns=feature_columns,
            metadata_columns=list(metadata_columns.values()),
        )
        row_metadata = {
            argument: metadata_by_column[column]
            for argument, column in metadata_columns.items()
            if column in metadata_by_column
        }
        eager_eval_set = None
        if eval_set is not None:
            eval_frame = eval_set.to_pandas()
            eval_features, eval_labels, _ = split_feature_frame(
                eval_frame,
                label=str(eval_label or label),
                feature_columns=feature_columns,
                metadata_columns=[
                    column for column in metadata_columns.values() if column in eval_frame.columns
                ],
            )
            eager_eval_set = (eval_features, eval_labels)
        return local_train(
            features,
            params,
            label=labels,
            num_boost_round=num_boost_round,
            eval_set=eager_eval_set,
            **row_metadata,
            **train_kwargs,
        )

    if resolved_workers <= 1:
        raise ValueError("distributed Ray training requires num_workers >= 2")
    required_cpus = float(resolved_workers) * float(num_cpus_per_worker)
    cluster_cpus = float(ray.cluster_resources().get("CPU", 0.0))
    if required_cpus > cluster_cpus:
        raise ValueError(
            f"requested {required_cpus:g} Ray CPUs but the cluster exposes {cluster_cpus:g}"
        )

    shards = dataset.split(resolved_workers, equal=False)
    eval_shards = (
        [None] * resolved_workers
        if eval_set is None
        else eval_set.split(resolved_workers, equal=False)
    )
    if len(shards) != resolved_workers or len(eval_shards) != resolved_workers:
        raise RuntimeError("Ray Dataset could not create the requested number of shards")

    endpoint_remote = ray.remote(_ray_endpoint_task)
    concrete_root = distributed_root
    endpoint_node_id = None
    if concrete_root is None:
        concrete_root, endpoint_node_id = ray.get(
            endpoint_remote.options(num_cpus=float(num_cpus_per_worker)).remote()
        )
    concrete_run_id = str(run_id or f"ray-{uuid4().hex}")

    worker_remote = ray.remote(_ray_train_task)
    futures = []
    for rank, (shard, eval_shard) in enumerate(zip(shards, eval_shards)):
        options: Dict[str, Any] = {"num_cpus": float(num_cpus_per_worker)}
        if rank == 0 and endpoint_node_id:
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
                node_id=endpoint_node_id,
                soft=False,
            )
        futures.append(
            worker_remote.options(**options).remote(
                shard,
                eval_shard,
                str(label),
                str(eval_label or label),
                None if feature_columns is None else list(feature_columns),
                metadata_columns,
                dict(params),
                rank,
                resolved_workers,
                str(concrete_root),
                concrete_run_id,
                float(timeout),
                num_boost_round,
                dict(train_kwargs),
            )
        )
    models = ray.get(futures)
    return models[0]


def _predict_batch(
    batch: Any,
    *,
    model: Any,
    feature_columns: Optional[Sequence[str]],
    output_columns: Sequence[str],
    include_input: bool,
) -> Any:
    import pandas as pd

    features = batch if feature_columns is None else batch[list(feature_columns)]
    values = np.asarray(model.predict(features), dtype=np.float32)
    output = batch.copy() if include_input else pd.DataFrame(index=batch.index)
    if values.ndim == 1:
        output[output_columns[0]] = values
    else:
        for index, column in enumerate(output_columns):
            output[column] = values[:, index]
    return output


def predict(
    model: Any,
    dataset: Any,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    prediction_name: str = "prediction",
    include_input: bool = False,
    batch_size: Optional[int] = None,
    concurrency: Optional[int] = None,
) -> Any:
    """Return a lazy Ray Dataset whose batches contain CTBoost predictions."""

    _require_ray()
    columns = prediction_columns(model.prediction_dimension, prediction_name)
    kwargs: Dict[str, Any] = {
        "batch_format": "pandas",
        "zero_copy_batch": False,
        "fn_kwargs": {
            "model": model,
            "feature_columns": None if feature_columns is None else list(feature_columns),
            "output_columns": columns,
            "include_input": bool(include_input),
        },
        "udf_modifying_row_count": False,
    }
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
    if concurrency is not None:
        kwargs["concurrency"] = int(concurrency)
    return dataset.map_batches(_predict_batch, **kwargs)


__all__ = ["predict", "train"]
