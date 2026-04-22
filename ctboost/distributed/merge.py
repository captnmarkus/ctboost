"""Pool-merge helpers for ctboost.distributed."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from .. import _core
from ..core import Pool
from .payloads import unpickle_payload


def _validate_group_shards(group_arrays: List[np.ndarray]) -> None:
    seen_groups = set()
    for shard_groups in group_arrays:
        unique_groups = {int(value) for value in np.unique(np.asarray(shard_groups, dtype=np.int64))}
        overlap = seen_groups.intersection(unique_groups)
        if overlap:
            raise ValueError("distributed ranking/group training requires each group_id to live on exactly one rank")
        seen_groups.update(unique_groups)


def _merge_dense_payloads(shards: List[Dict[str, Any]]) -> Pool:
    data = np.asfortranarray(
        np.concatenate([np.asarray(shard["data"], dtype=np.float32) for shard in shards], axis=0),
        dtype=np.float32,
    )
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)
    return Pool(
        data=data,
        label=label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_sparse_payloads(shards: List[Dict[str, Any]]) -> Pool:
    total_rows = sum(int(shard["num_rows"]) for shard in shards)
    num_cols = int(shards[0]["num_cols"])
    total_nnz = sum(int(np.asarray(shard["sparse_data"]).shape[0]) for shard in shards)
    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int64)
    indptr = np.empty(num_cols + 1, dtype=np.int64)
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_group_shards(group_arrays)
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
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_payloads(shards: List[Dict[str, Any]]) -> Pool:
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
    return _merge_dense_payloads(shards) if storage_kinds == {"dense"} else _merge_sparse_payloads(shards)


def _quantization_schema_from_debug_histogram(hist_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_bins_per_feature": list(hist_state["num_bins_per_feature"]),
        "cut_offsets": list(hist_state["cut_offsets"]),
        "cut_values": list(hist_state["cut_values"]),
        "categorical_mask": list(hist_state["categorical_mask"]),
        "missing_value_mask": list(hist_state["missing_value_mask"]),
        "nan_mode": int(hist_state["nan_mode"]),
        "nan_modes": list(hist_state["nan_modes"]),
    }


def build_schema_collect_response(payloads: List[bytes]) -> bytes:
    requests = [unpickle_payload(payload) for payload in payloads]
    merged_pool = _merge_payloads([request["shard"] for request in requests])
    schema_request = requests[0]["schema_request"]
    hist_state = _core._debug_build_histogram(
        merged_pool._handle,
        max_bins=int(schema_request["max_bins"]),
        nan_mode=str(schema_request["nan_mode"]),
        max_bin_by_feature=list(schema_request["max_bin_by_feature"]),
        border_selection_method=str(schema_request["border_selection_method"]),
        nan_mode_by_feature=list(schema_request["nan_mode_by_feature"]),
        feature_borders=list(schema_request["feature_borders"]),
        external_memory=bool(schema_request["external_memory"]),
        external_memory_dir=str(schema_request["external_memory_dir"]),
    )
    return json.dumps(_quantization_schema_from_debug_histogram(hist_state), sort_keys=True).encode("utf-8")
