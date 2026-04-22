"""Filesystem shard-merge helpers for distributed training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..core import Pool
from ._distributed_payloads import _validate_distributed_group_shards
from ._distributed_storage_io import _open_memmap_array

def _merge_dense_distributed_shards(
    run_root: Path,
    manifests: List[Dict[str, Any]],
) -> Pool:
    total_rows = sum(int(manifest["num_rows"]) for manifest in manifests)
    num_cols = int(manifests[0]["num_cols"])
    merged_root = run_root / "merged_dense"
    merged_root.mkdir(parents=True, exist_ok=True)

    data_map = _open_memmap_array(
        merged_root / "data.npy",
        (total_rows, num_cols),
        dtype=np.float32,
        fortran_order=True,
    )
    label_map = _open_memmap_array(merged_root / "label.npy", (total_rows,), dtype=np.float32)
    weight_map = _open_memmap_array(merged_root / "weight.npy", (total_rows,), dtype=np.float32)

    has_group_id = any(manifest.get("group_id_path") is not None for manifest in manifests)
    has_baseline = any(manifest.get("baseline_path") is not None for manifest in manifests)
    group_map = (
        _open_memmap_array(merged_root / "group_id.npy", (total_rows,), dtype=np.int64)
        if has_group_id
        else None
    )
    baseline_map = None
    if has_baseline:
        baseline_shapes = []
        for manifest in manifests:
            baseline_path = manifest.get("baseline_path")
            if baseline_path is None:
                raise ValueError("distributed shards must either all include baseline or all omit it")
            shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
            baseline_shapes.append(np.load(shard_root / baseline_path, mmap_mode="r").shape)
        baseline_suffixes = {tuple(shape[1:]) for shape in baseline_shapes}
        if len(baseline_suffixes) != 1:
            raise ValueError("distributed baseline shards must agree on prediction dimensions")
        baseline_map = _open_memmap_array(
            merged_root / "baseline.npy",
            (total_rows, *next(iter(baseline_suffixes))),
            dtype=np.float32,
        )
    group_arrays = []

    row_offset = 0
    for manifest in manifests:
        shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
        shard_rows = int(manifest["num_rows"])
        next_offset = row_offset + shard_rows
        shard_data = np.load(shard_root / manifest["data_path"], mmap_mode="r")
        data_map[row_offset:next_offset, :] = np.asarray(shard_data, dtype=np.float32)
        label_map[row_offset:next_offset] = np.load(shard_root / manifest["label_path"], mmap_mode="r")
        weight_map[row_offset:next_offset] = np.load(shard_root / manifest["weight_path"], mmap_mode="r")
        if group_map is not None:
            group_path = manifest.get("group_id_path")
            if group_path is None:
                raise ValueError("distributed shards must either all include group_id or all omit it")
            shard_groups = np.load(shard_root / group_path, mmap_mode="r")
            group_arrays.append(np.asarray(shard_groups, dtype=np.int64))
            group_map[row_offset:next_offset] = shard_groups
        if baseline_map is not None:
            baseline_path = manifest.get("baseline_path")
            if baseline_path is None:
                raise ValueError("distributed shards must either all include baseline or all omit it")
            baseline_map[row_offset:next_offset, ...] = np.load(shard_root / baseline_path, mmap_mode="r")
        row_offset = next_offset

    data_map.flush()
    label_map.flush()
    weight_map.flush()
    if group_map is not None:
        _validate_distributed_group_shards(group_arrays)
        group_map.flush()
    if baseline_map is not None:
        baseline_map.flush()

    return Pool(
        data=np.load(merged_root / "data.npy", mmap_mode="r"),
        label=np.load(merged_root / "label.npy", mmap_mode="r"),
        cat_features=list(manifests[0]["cat_features"]),
        weight=np.load(merged_root / "weight.npy", mmap_mode="r"),
        group_id=None
        if group_map is None
        else np.load(merged_root / "group_id.npy", mmap_mode="r"),
        baseline=None
        if baseline_map is None
        else np.load(merged_root / "baseline.npy", mmap_mode="r"),
        feature_names=manifests[0]["feature_names"],
        _releasable_feature_storage=True,
    )

def _merge_sparse_distributed_shards(
    run_root: Path,
    manifests: List[Dict[str, Any]],
) -> Pool:
    total_rows = sum(int(manifest["num_rows"]) for manifest in manifests)
    num_cols = int(manifests[0]["num_cols"])
    merged_root = run_root / "merged_sparse"
    merged_root.mkdir(parents=True, exist_ok=True)

    total_nnz = 0
    shard_payloads = []
    for manifest in manifests:
        shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
        sparse_data = np.load(shard_root / manifest["sparse_data_path"], mmap_mode="r")
        sparse_indices = np.load(shard_root / manifest["sparse_indices_path"], mmap_mode="r")
        sparse_indptr = np.load(shard_root / manifest["sparse_indptr_path"], mmap_mode="r")
        total_nnz += int(sparse_data.shape[0])
        shard_payloads.append((manifest, sparse_data, sparse_indices, sparse_indptr))

    data_map = _open_memmap_array(merged_root / "sparse_data.npy", (total_nnz,), dtype=np.float32)
    indices_map = _open_memmap_array(
        merged_root / "sparse_indices.npy", (total_nnz,), dtype=np.int64
    )
    indptr_map = _open_memmap_array(
        merged_root / "sparse_indptr.npy", (num_cols + 1,), dtype=np.int64
    )
    label_map = _open_memmap_array(merged_root / "label.npy", (total_rows,), dtype=np.float32)
    weight_map = _open_memmap_array(merged_root / "weight.npy", (total_rows,), dtype=np.float32)

    has_group_id = any(manifest.get("group_id_path") is not None for manifest in manifests)
    has_baseline = any(manifest.get("baseline_path") is not None for manifest in manifests)
    group_map = (
        _open_memmap_array(merged_root / "group_id.npy", (total_rows,), dtype=np.int64)
        if has_group_id
        else None
    )
    baseline_map = None
    if has_baseline:
        baseline_shapes = []
        for manifest in manifests:
            baseline_path = manifest.get("baseline_path")
            if baseline_path is None:
                raise ValueError("distributed shards must either all include baseline or all omit it")
            shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
            baseline_shapes.append(np.load(shard_root / baseline_path, mmap_mode="r").shape)
        baseline_suffixes = {tuple(shape[1:]) for shape in baseline_shapes}
        if len(baseline_suffixes) != 1:
            raise ValueError("distributed baseline shards must agree on prediction dimensions")
        baseline_map = _open_memmap_array(
            merged_root / "baseline.npy",
            (total_rows, *next(iter(baseline_suffixes))),
            dtype=np.float32,
        )
    group_arrays = []

    row_offset = 0
    for manifest, _sparse_data, _sparse_indices, _sparse_indptr in shard_payloads:
        shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
        shard_rows = int(manifest["num_rows"])
        next_offset = row_offset + shard_rows
        label_map[row_offset:next_offset] = np.load(shard_root / manifest["label_path"], mmap_mode="r")
        weight_map[row_offset:next_offset] = np.load(shard_root / manifest["weight_path"], mmap_mode="r")
        if group_map is not None:
            group_path = manifest.get("group_id_path")
            if group_path is None:
                raise ValueError("distributed shards must either all include group_id or all omit it")
            shard_groups = np.load(shard_root / group_path, mmap_mode="r")
            group_arrays.append(np.asarray(shard_groups, dtype=np.int64))
            group_map[row_offset:next_offset] = shard_groups
        if baseline_map is not None:
            baseline_path = manifest.get("baseline_path")
            if baseline_path is None:
                raise ValueError("distributed shards must either all include baseline or all omit it")
            baseline_map[row_offset:next_offset, ...] = np.load(shard_root / baseline_path, mmap_mode="r")
        row_offset = next_offset

    global_offset = 0
    indptr_map[0] = 0
    row_base = [0]
    for manifest in manifests[:-1]:
        row_base.append(row_base[-1] + int(manifest["num_rows"]))
    for col in range(num_cols):
        for shard_index, (_manifest, sparse_data, sparse_indices, sparse_indptr) in enumerate(shard_payloads):
            begin = int(sparse_indptr[col])
            end = int(sparse_indptr[col + 1])
            nnz = end - begin
            if nnz == 0:
                continue
            next_offset = global_offset + nnz
            data_map[global_offset:next_offset] = sparse_data[begin:end]
            indices_map[global_offset:next_offset] = sparse_indices[begin:end] + row_base[shard_index]
            global_offset = next_offset
        indptr_map[col + 1] = global_offset

    data_map.flush()
    indices_map.flush()
    indptr_map.flush()
    label_map.flush()
    weight_map.flush()
    if group_map is not None:
        _validate_distributed_group_shards(group_arrays)
        group_map.flush()
    if baseline_map is not None:
        baseline_map.flush()

    return Pool.from_csc_components(
        np.load(merged_root / "sparse_data.npy", mmap_mode="r"),
        np.load(merged_root / "sparse_indices.npy", mmap_mode="r"),
        np.load(merged_root / "sparse_indptr.npy", mmap_mode="r"),
        (total_rows, num_cols),
        np.load(merged_root / "label.npy", mmap_mode="r"),
        cat_features=list(manifests[0]["cat_features"]),
        weight=np.load(merged_root / "weight.npy", mmap_mode="r"),
        group_id=None
        if group_map is None
        else np.load(merged_root / "group_id.npy", mmap_mode="r"),
        baseline=None
        if baseline_map is None
        else np.load(merged_root / "baseline.npy", mmap_mode="r"),
        feature_names=manifests[0]["feature_names"],
        _releasable_feature_storage=True,
    )

def _merge_distributed_shards(run_root: Path, manifests: List[Dict[str, Any]]) -> Pool:
    if not manifests:
        raise ValueError("distributed training requires at least one shard manifest")

    storage_kinds = {str(manifest["storage"]) for manifest in manifests}
    if len(storage_kinds) != 1:
        raise ValueError("distributed training requires all shards to use the same storage kind")

    num_cols = int(manifests[0]["num_cols"])
    cat_features = list(manifests[0]["cat_features"])
    feature_names = manifests[0]["feature_names"]
    for manifest in manifests[1:]:
        if int(manifest["num_cols"]) != num_cols:
            raise ValueError("distributed shards must agree on num_cols")
        if list(manifest["cat_features"]) != cat_features:
            raise ValueError("distributed shards must agree on cat_features")
        if manifest["feature_names"] != feature_names:
            raise ValueError("distributed shards must agree on feature_names")

    if storage_kinds == {"dense"}:
        return _merge_dense_distributed_shards(run_root, manifests)
    return _merge_sparse_distributed_shards(run_root, manifests)

