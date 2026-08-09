"""Shared helpers for optional distributed-data integrations."""

from __future__ import annotations

import socket
from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def concat_partitions(partitions: Iterable[Any]) -> Any:
    """Concatenate row partitions without erasing dataframe schema information."""

    resolved = [partition for partition in partitions if partition is not None]
    if not resolved:
        raise ValueError("at least one non-empty partition is required")
    if len(resolved) == 1:
        return resolved[0]

    first = resolved[0]
    module = type(first).__module__.split(".", 1)[0]
    if module == "pandas":
        import pandas as pd

        return pd.concat(resolved, axis=0, ignore_index=True)
    if module == "polars":
        import polars as pl

        return pl.concat(resolved, how="vertical")
    if module == "pyarrow":
        import pyarrow as pa

        if isinstance(first, pa.Table):
            return pa.concat_tables(resolved)
        return pa.concat_arrays(resolved)
    if module == "scipy":
        try:
            from scipy import sparse
        except ImportError:
            pass
        else:
            if sparse.issparse(first):
                return sparse.vstack(resolved, format="csr")

    arrays = [np.asarray(partition) for partition in resolved]
    return np.concatenate(arrays, axis=0)


def materialize(value: Any) -> Any:
    """Compute a lazy collection while leaving eager inputs untouched."""

    if value is None or isinstance(value, (str, bytes)):
        return value
    compute = getattr(value, "compute", None)
    if callable(compute):
        return compute()
    return value


def split_feature_frame(
    frame: Any,
    *,
    label: str,
    feature_columns: Optional[Sequence[str]] = None,
    metadata_columns: Sequence[Optional[str]] = (),
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Split a pandas-like frame into features, target, and row metadata."""

    if label not in frame.columns:
        raise ValueError(f"label column {label!r} is not present")
    resolved_metadata = [name for name in metadata_columns if name is not None]
    missing_metadata = [name for name in resolved_metadata if name not in frame.columns]
    if missing_metadata:
        raise ValueError(f"metadata columns are missing: {missing_metadata}")
    excluded = {label, *resolved_metadata}
    resolved_features = (
        [column for column in frame.columns if column not in excluded]
        if feature_columns is None
        else list(feature_columns)
    )
    if len(set(resolved_features)) != len(resolved_features):
        raise ValueError("feature columns must not contain duplicates")
    missing = [column for column in resolved_features if column not in frame.columns]
    if missing:
        raise ValueError(f"feature columns are missing: {missing}")
    leaked = [column for column in resolved_features if column in excluded]
    if leaked:
        raise ValueError(
            "feature columns must not include the label or row metadata columns: "
            f"{leaked}"
        )
    if not resolved_features:
        raise ValueError("at least one feature column is required")
    metadata = {
        name: frame[name]
        for name in resolved_metadata
    }
    return frame[resolved_features], frame[label], metadata


def allocate_tcp_endpoint(host_hint: Optional[str] = None) -> tuple[str, str]:
    """Return a worker-reachable TCP root and the current runtime node id placeholder."""

    from .distributed.tcp import authenticated_tcp_root

    host = str(host_hint or "").strip()
    if not host or host in {"0.0.0.0", "::", "localhost"}:
        host = socket.gethostname()
    host = socket.gethostbyname(host)
    if host in {"0.0.0.0", "::"}:
        raise ValueError("distributed endpoint allocation requires a concrete host")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        port = int(listener.getsockname()[1])
    return authenticated_tcp_root(host, port), ""


def train_distributed_shard(
    data: Any,
    label: Any,
    params: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    distributed_root: str,
    run_id: str,
    timeout: float,
    num_boost_round: Optional[int],
    eval_set: Any = None,
    row_metadata: Optional[Mapping[str, Any]] = None,
    train_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Train one CTBoost collective rank from a concrete row shard."""

    from .distributed import wait_for_distributed_tcp_coordinator
    from .training import train

    if rank != 0:
        wait_for_distributed_tcp_coordinator(
            distributed_root,
            float(timeout),
            run_id=str(run_id),
            world_size=int(world_size),
        )
    distributed_params = dict(params)
    distributed_params.update(
        distributed_world_size=int(world_size),
        distributed_rank=int(rank),
        distributed_root=str(distributed_root),
        distributed_run_id=str(run_id),
        distributed_timeout=float(timeout),
    )
    kwargs = dict(train_kwargs or {})
    kwargs.update(dict(row_metadata or {}))
    return train(
        data,
        distributed_params,
        label=label,
        num_boost_round=num_boost_round,
        eval_set=eval_set,
        **kwargs,
    )


def prediction_columns(prediction_dimension: int, base_name: str) -> List[str]:
    if int(prediction_dimension) <= 1:
        return [str(base_name)]
    return [f"{base_name}_{index}" for index in range(int(prediction_dimension))]
