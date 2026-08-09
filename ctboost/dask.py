"""Dask collection integration for CTBoost.

Training uses CTBoost's native TCP collective across pinned Dask workers when
``mode='distributed'``.  ``mode='materialize'`` is an explicit driver-memory
fallback for small datasets and local debugging.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np

from ._integration_utils import (
    allocate_tcp_endpoint,
    concat_partitions,
    materialize,
    prediction_columns,
    train_distributed_shard,
)


def _require_distributed() -> Any:
    try:
        from dask.distributed import Client
    except ImportError as exc:
        raise ImportError(
            "CTBoost's Dask integration requires dask and distributed. "
            "Install 'ctboost[dask]'."
        ) from exc
    return Client


def _row_partitions(collection: Any) -> List[Any]:
    if collection is None:
        return []
    delayed = getattr(collection, "to_delayed", None)
    if not callable(delayed):
        return [collection]
    value = collection
    ndim = int(getattr(value, "ndim", 1))
    if ndim == 2 and hasattr(value, "rechunk"):
        value = value.rechunk({1: -1})
    partitions = value.to_delayed()
    if isinstance(partitions, np.ndarray):
        partitions = partitions.reshape(-1).tolist()
    else:
        partitions = list(partitions)
    return partitions


def _partition_groups(partitions: Sequence[Any], world_size: int) -> List[List[Any]]:
    if len(partitions) < world_size:
        raise ValueError(
            f"collection has {len(partitions)} row partitions, fewer than num_workers={world_size}"
        )
    indices = np.array_split(np.arange(len(partitions)), world_size)
    return [[partitions[int(index)] for index in group] for group in indices]


def _worker_host(worker_address: str) -> Optional[str]:
    parsed = urlparse(str(worker_address))
    if parsed.hostname:
        return parsed.hostname
    return None


def _worker_process_id() -> int:
    return int(os.getpid())


def _dask_train_task(
    data_parts: Sequence[Any],
    label_parts: Sequence[Any],
    metadata_parts: Mapping[str, Sequence[Any]],
    eval_data_parts: Optional[Sequence[Any]],
    eval_label_parts: Optional[Sequence[Any]],
    params: Mapping[str, Any],
    rank: int,
    world_size: int,
    root: str,
    run_id: str,
    timeout: float,
    num_boost_round: Optional[int],
    train_kwargs: Mapping[str, Any],
) -> Any:
    data = concat_partitions(data_parts)
    label = concat_partitions(label_parts)
    metadata = {
        name: concat_partitions(parts)
        for name, parts in metadata_parts.items()
        if parts
    }
    eval_set = None
    if eval_data_parts is not None and eval_label_parts is not None:
        eval_set = (
            concat_partitions(eval_data_parts),
            concat_partitions(eval_label_parts),
        )
    return train_distributed_shard(
        data,
        label,
        params,
        rank=rank,
        world_size=world_size,
        distributed_root=root,
        run_id=run_id,
        timeout=timeout,
        num_boost_round=num_boost_round,
        eval_set=eval_set,
        row_metadata=metadata,
        train_kwargs=train_kwargs,
    )


def _resolve_column(data: Any, value: Any, *, drop: bool) -> tuple[Any, Any]:
    if not isinstance(value, str):
        return data, value
    try:
        resolved = data[value]
    except Exception as exc:
        raise ValueError(f"column {value!r} is not present in the Dask frame") from exc
    if drop:
        data = data.drop(columns=[value])
    return data, resolved


def _compute_with_client(client: Any, value: Any) -> Any:
    if value is None:
        return None
    if client is None:
        return materialize(value)
    future = client.compute(value)
    result = getattr(future, "result", None)
    return result() if callable(result) else future


def _partition_futures(client: Any, partitions: Sequence[Any]) -> List[Any]:
    if not partitions:
        return []
    if all(hasattr(partition, "__dask_graph__") for partition in partitions):
        return list(client.compute(list(partitions)))
    scattered = client.scatter(list(partitions), broadcast=False)
    return list(scattered) if isinstance(scattered, (tuple, list)) else [scattered]


def train(
    client: Any,
    data: Any,
    label: Any,
    params: Mapping[str, Any],
    num_boost_round: Optional[int] = None,
    *,
    num_workers: Optional[int] = None,
    mode: str = "auto",
    distributed_root: Optional[str] = None,
    run_id: Optional[str] = None,
    timeout: float = 600.0,
    weight: Any = None,
    group_id: Any = None,
    group_weight: Any = None,
    subgroup_id: Any = None,
    baseline: Any = None,
    eval_set: Any = None,
    **train_kwargs: Any,
) -> Any:
    """Train CTBoost from Dask DataFrame/Series or row-chunked Array inputs.

    String targets and metadata values are interpreted as columns and removed
    from the feature frame.  Distributed mode pins one collective rank to each
    selected Dask worker; materialize mode computes collections on the driver.
    """

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"auto", "distributed", "materialize"}:
        raise ValueError("mode must be one of: auto, distributed, materialize")
    data, label = _resolve_column(data, label, drop=True)
    metadata_values: Dict[str, Any] = {}
    for name, value in (
        ("weight", weight),
        ("group_id", group_id),
        ("group_weight", group_weight),
        ("subgroup_id", subgroup_id),
        ("baseline", baseline),
    ):
        data, resolved = _resolve_column(data, value, drop=isinstance(value, str))
        if resolved is not None:
            metadata_values[name] = resolved

    data_parts = _row_partitions(data)
    if normalized_mode == "auto":
        worker_count = 0 if client is None else len(client.scheduler_info().get("workers", {}))
        normalized_mode = (
            "distributed" if worker_count > 1 and len(data_parts) > 1 else "materialize"
        )

    if normalized_mode == "materialize":
        from .training import train as local_train

        eager_eval_set = None
        if eval_set is not None:
            if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
                raise TypeError("Dask eval_set must be a (data, label) pair")
            eager_eval_set = (
                _compute_with_client(client, eval_set[0]),
                _compute_with_client(client, eval_set[1]),
            )
        eager_metadata = {
            name: _compute_with_client(client, value)
            for name, value in metadata_values.items()
        }
        return local_train(
            _compute_with_client(client, data),
            params,
            label=_compute_with_client(client, label),
            num_boost_round=num_boost_round,
            eval_set=eager_eval_set,
            **eager_metadata,
            **train_kwargs,
        )

    _require_distributed()
    if client is None:
        raise ValueError("distributed Dask training requires a dask.distributed.Client")
    worker_info = client.scheduler_info().get("workers", {})
    workers = sorted(worker_info)
    if not workers:
        raise RuntimeError("the Dask client has no connected workers")
    resolved_world_size = min(len(workers), len(data_parts)) if num_workers is None else int(num_workers)
    if resolved_world_size <= 1:
        raise ValueError("distributed Dask training requires at least two workers")
    if resolved_world_size > len(workers):
        raise ValueError("num_workers exceeds the number of connected Dask workers")
    selected_workers = workers[:resolved_world_size]
    worker_pids = client.gather(
        [
            client.submit(
                _worker_process_id,
                workers=[worker],
                allow_other_workers=False,
                pure=False,
            )
            for worker in selected_workers
        ]
    )
    worker_processes = {
        (_worker_host(worker) or str(worker), int(pid))
        for worker, pid in zip(selected_workers, worker_pids)
    }
    if len(worker_processes) != resolved_world_size:
        raise ValueError(
            "distributed CTBoost requires process-based Dask workers; multiple TCP ranks "
            "cannot run in threads of the same Python process"
        )

    label_parts = _row_partitions(label)
    if len(label_parts) != len(data_parts):
        raise ValueError("Dask feature and label collections must have matching row partitions")
    metadata_partition_lists = {
        name: _row_partitions(value) for name, value in metadata_values.items()
    }
    for name, partitions in metadata_partition_lists.items():
        if len(partitions) != len(data_parts):
            raise ValueError(f"Dask {name} must have the same row partitions as data")

    eval_data_parts: Optional[List[Any]] = None
    eval_label_parts: Optional[List[Any]] = None
    if eval_set is not None:
        if not isinstance(eval_set, (tuple, list)) or len(eval_set) != 2:
            raise TypeError("Dask eval_set must be a (data, label) pair")
        eval_data_parts = _row_partitions(eval_set[0])
        eval_label_parts = _row_partitions(eval_set[1])
        if len(eval_data_parts) != len(eval_label_parts):
            raise ValueError("Dask eval data and labels must have matching row partitions")

    data_parts = _partition_futures(client, data_parts)
    label_parts = _partition_futures(client, label_parts)
    metadata_partition_lists = {
        name: _partition_futures(client, partitions)
        for name, partitions in metadata_partition_lists.items()
    }
    if eval_data_parts is not None:
        eval_data_parts = _partition_futures(client, eval_data_parts)
    if eval_label_parts is not None:
        eval_label_parts = _partition_futures(client, eval_label_parts)

    concrete_root = distributed_root
    if concrete_root is None:
        endpoint_future = client.submit(
            allocate_tcp_endpoint,
            _worker_host(selected_workers[0]),
            workers=[selected_workers[0]],
            allow_other_workers=False,
            pure=False,
        )
        concrete_root = endpoint_future.result()[0]
    concrete_run_id = str(run_id or f"dask-{uuid4().hex}")

    data_groups = _partition_groups(data_parts, resolved_world_size)
    label_groups = _partition_groups(label_parts, resolved_world_size)
    metadata_groups = {
        name: _partition_groups(partitions, resolved_world_size)
        for name, partitions in metadata_partition_lists.items()
    }
    eval_data_groups = (
        None if eval_data_parts is None else _partition_groups(eval_data_parts, resolved_world_size)
    )
    eval_label_groups = (
        None if eval_label_parts is None else _partition_groups(eval_label_parts, resolved_world_size)
    )
    futures = []
    for rank, worker in enumerate(selected_workers):
        futures.append(
            client.submit(
                _dask_train_task,
                data_groups[rank],
                label_groups[rank],
                {name: groups[rank] for name, groups in metadata_groups.items()},
                None if eval_data_groups is None else eval_data_groups[rank],
                None if eval_label_groups is None else eval_label_groups[rank],
                dict(params),
                rank,
                resolved_world_size,
                str(concrete_root),
                concrete_run_id,
                float(timeout),
                num_boost_round,
                dict(train_kwargs),
                workers=[worker],
                allow_other_workers=False,
                pure=False,
            )
        )
    models = client.gather(futures)
    return models[0]


def _predict_dataframe_partition(partition: Any, model: Any, columns: Sequence[str]) -> Any:
    import pandas as pd

    values = np.asarray(model.predict(partition), dtype=np.float32)
    if values.ndim == 1:
        return pd.Series(values, index=partition.index, name=columns[0], dtype=np.float32)
    return pd.DataFrame(values, index=partition.index, columns=list(columns), dtype=np.float32)


def predict(
    model: Any,
    data: Any,
    *,
    mode: str = "partitioned",
    prediction_name: str = "prediction",
) -> Any:
    """Predict lazily on each Dask partition, or eagerly with ``mode='materialize'``."""

    normalized_mode = str(mode).lower()
    if normalized_mode == "materialize":
        return model.predict(materialize(data))
    if normalized_mode != "partitioned":
        raise ValueError("mode must be one of: partitioned, materialize")
    columns = prediction_columns(model.prediction_dimension, prediction_name)
    if hasattr(data, "map_partitions") and hasattr(data, "columns"):
        import pandas as pd

        meta = (
            pd.Series([], name=columns[0], dtype=np.float32)
            if len(columns) == 1
            else pd.DataFrame({column: pd.Series(dtype=np.float32) for column in columns})
        )
        return data.map_partitions(
            _predict_dataframe_partition,
            model,
            columns,
            meta=meta,
        )
    if hasattr(data, "map_blocks"):
        array = data.rechunk({1: -1}) if int(getattr(data, "ndim", 1)) == 2 else data
        if len(columns) == 1:
            return array.map_blocks(
                lambda block: np.asarray(model.predict(block), dtype=np.float32),
                dtype=np.float32,
                chunks=(array.chunks[0],),
                drop_axis=1,
            )
        return array.map_blocks(
            lambda block: np.asarray(model.predict(block), dtype=np.float32),
            dtype=np.float32,
            chunks=(array.chunks[0], (len(columns),)),
        )
    raise TypeError("data must be a Dask DataFrame or Dask Array for partitioned prediction")


__all__ = ["predict", "train"]
