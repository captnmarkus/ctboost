"""Distributed payload helpers for ctboost.training."""

from __future__ import annotations

import json
import pickle
import struct
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..distributed import distributed_tcp_request, pickle_payload
from ..core import Pool

def _unpack_gathered_payloads(payload: bytes) -> List[bytes]:
    view = memoryview(payload)
    if len(view) < 8:
        raise RuntimeError("distributed allgather payload is truncated")
    (count,) = struct.unpack_from("<Q", view, 0)
    offset = 8
    payloads: List[bytes] = []
    for _ in range(int(count)):
        if offset + 8 > len(view):
            raise RuntimeError("distributed allgather payload is truncated")
        (payload_size,) = struct.unpack_from("<Q", view, offset)
        offset += 8
        next_offset = offset + int(payload_size)
        if next_offset > len(view):
            raise RuntimeError("distributed allgather payload is truncated")
        payloads.append(bytes(view[offset:next_offset]))
        offset = next_offset
    return payloads

def _distributed_allgather_value(distributed: Dict[str, Any], key: str, value: Any) -> List[Any]:
    response = distributed_tcp_request(
        distributed["root"],
        distributed["timeout"],
        "allgather",
        key,
        distributed["rank"],
        distributed["world_size"],
        pickle_payload(value),
    )
    return [pickle.loads(item) for item in _unpack_gathered_payloads(response)]

def _distributed_broadcast_value(
    distributed: Dict[str, Any],
    key: str,
    *,
    value: Any = None,
) -> Any:
    payload = pickle_payload(value) if distributed["rank"] == 0 else b""
    response = distributed_tcp_request(
        distributed["root"],
        distributed["timeout"],
        "broadcast",
        key,
        distributed["rank"],
        distributed["world_size"],
        payload,
    )
    return pickle.loads(response) if response else None

def _quantization_schema_from_debug_histogram(hist_state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "num_bins_per_feature": list(hist_state["num_bins_per_feature"]),
        "cut_offsets": list(hist_state["cut_offsets"]),
        "cut_values": list(hist_state["cut_values"]),
        "categorical_mask": list(hist_state["categorical_mask"]),
        "missing_value_mask": list(hist_state["missing_value_mask"]),
        "nan_mode": int(hist_state["nan_mode"]),
        "nan_modes": list(hist_state["nan_modes"]),
    }

def _validate_distributed_group_shards(group_arrays: List[np.ndarray]) -> None:
    if not group_arrays:
        return
    seen_groups = set()
    for shard_groups in group_arrays:
        unique_groups = {int(value) for value in np.unique(np.asarray(shard_groups, dtype=np.int64))}
        overlap = seen_groups.intersection(unique_groups)
        if overlap:
            raise ValueError(
                "distributed ranking/group training requires each group_id to live on exactly one rank"
            )
        seen_groups.update(unique_groups)

def _serialize_distributed_pool_shard(pool: Pool) -> Dict[str, Any]:
    sparse_components = getattr(pool, "_sparse_csc_components", None)
    payload: Dict[str, Any] = {
        "num_rows": int(pool.num_rows),
        "num_cols": int(pool.num_cols),
        "cat_features": list(pool.cat_features),
        "feature_names": None if pool.feature_names is None else list(pool.feature_names),
        "label": np.asarray(pool.label, dtype=np.float32),
        "weight": np.ones(pool.num_rows, dtype=np.float32)
        if pool.weight is None
        else np.asarray(pool.weight, dtype=np.float32),
        "group_id": None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64),
        "baseline": None if pool.baseline is None else np.asarray(pool.baseline, dtype=np.float32),
    }
    if sparse_components is None:
        dense_data = getattr(pool, "_dense_data_ref", None)
        if dense_data is None:
            dense_data = np.asfortranarray(pool.data, dtype=np.float32)
        payload["storage"] = "dense"
        payload["data"] = np.asfortranarray(dense_data, dtype=np.float32)
    else:
        sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
        payload["storage"] = "sparse"
        payload["shape"] = (int(shape[0]), int(shape[1]))
        payload["sparse_data"] = np.asarray(sparse_data, dtype=np.float32)
        payload["sparse_indices"] = np.asarray(sparse_indices, dtype=np.int64)
        payload["sparse_indptr"] = np.asarray(sparse_indptr, dtype=np.int64)
    return payload

def _merge_dense_distributed_payloads(shards: List[Dict[str, Any]]) -> Pool:
    data = np.asfortranarray(
        np.concatenate([np.asarray(shard["data"], dtype=np.float32) for shard in shards], axis=0),
        dtype=np.float32,
    )
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_baseline = any(shard["baseline"] is not None for shard in shards)
    if has_baseline and any(shard["baseline"] is None for shard in shards):
        raise ValueError("distributed shards must either all include baseline or all omit it")
    baseline = None
    if has_baseline:
        baseline = np.concatenate(
            [np.asarray(shard["baseline"], dtype=np.float32) for shard in shards],
            axis=0,
        )
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_distributed_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)
    return Pool(
        data=data,
        label=label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        baseline=baseline,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )

def _merge_sparse_distributed_payloads(shards: List[Dict[str, Any]]) -> Pool:
    total_rows = sum(int(shard["num_rows"]) for shard in shards)
    num_cols = int(shards[0]["num_cols"])
    total_nnz = sum(int(np.asarray(shard["sparse_data"]).shape[0]) for shard in shards)
    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int64)
    indptr = np.empty(num_cols + 1, dtype=np.int64)
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_baseline = any(shard["baseline"] is not None for shard in shards)
    if has_baseline and any(shard["baseline"] is None for shard in shards):
        raise ValueError("distributed shards must either all include baseline or all omit it")
    baseline = None
    if has_baseline:
        baseline = np.concatenate(
            [np.asarray(shard["baseline"], dtype=np.float32) for shard in shards],
            axis=0,
        )
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_distributed_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)

    row_bases = [0]
    for shard in shards[:-1]:
        row_bases.append(row_bases[-1] + int(shard["num_rows"]))

    nnz_offset = 0
    indptr[0] = 0
    for col in range(num_cols):
        for shard_index, shard in enumerate(shards):
            sparse_data = np.asarray(shard["sparse_data"], dtype=np.float32)
            sparse_indices = np.asarray(shard["sparse_indices"], dtype=np.int64)
            sparse_indptr = np.asarray(shard["sparse_indptr"], dtype=np.int64)
            begin = int(sparse_indptr[col])
            end = int(sparse_indptr[col + 1])
            nnz = end - begin
            if nnz <= 0:
                continue
            next_offset = nnz_offset + nnz
            data[nnz_offset:next_offset] = sparse_data[begin:end]
            indices[nnz_offset:next_offset] = sparse_indices[begin:end] + row_bases[shard_index]
            nnz_offset = next_offset
        indptr[col + 1] = nnz_offset

    return Pool.from_csc_components(
        data,
        indices,
        indptr,
        (total_rows, num_cols),
        label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        baseline=baseline,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )

def _merge_distributed_payloads(shards: List[Dict[str, Any]]) -> Pool:
    if not shards:
        raise ValueError("distributed training requires at least one shard payload")
    storage_kinds = {str(shard["storage"]) for shard in shards}
    if len(storage_kinds) != 1:
        raise ValueError("distributed shards must agree on storage kind")
    num_cols = int(shards[0]["num_cols"])
    cat_features = list(shards[0]["cat_features"])
    feature_names = shards[0]["feature_names"]
    for shard in shards[1:]:
        if int(shard["num_cols"]) != num_cols:
            raise ValueError("distributed shards must agree on num_cols")
        if list(shard["cat_features"]) != cat_features:
            raise ValueError("distributed shards must agree on cat_features")
        if shard["feature_names"] != feature_names:
            raise ValueError("distributed shards must agree on feature_names")
    if storage_kinds == {"dense"}:
        return _merge_dense_distributed_payloads(shards)
    return _merge_sparse_distributed_payloads(shards)

