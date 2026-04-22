"""Filesystem shard-writing helpers for distributed training."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np

from ..core import Pool

def _open_memmap_array(
    path: Path,
    shape: tuple[int, ...],
    *,
    dtype: Any,
    fortran_order: bool = False,
) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=shape,
        fortran_order=fortran_order,
    )

def _wait_for_path(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for distributed artifact: {path}")
        time.sleep(0.1)

def _wait_for_manifests(run_root: Path, world_size: int, timeout: float) -> List[Path]:
    deadline = time.monotonic() + timeout
    manifest_paths = [run_root / f"rank_{rank:05d}" / "manifest.json" for rank in range(world_size)]
    while True:
        missing = [path for path in manifest_paths if not path.exists()]
        if not missing:
            return manifest_paths
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for distributed shard manifests: "
                + ", ".join(str(path) for path in missing)
            )
        time.sleep(0.1)

def _write_distributed_shard(pool: Pool, config: Dict[str, Any]) -> Path:
    run_root = Path(config["root"]) / config["run_id"]
    shard_root = run_root / f"rank_{config['rank']:05d}"
    shard_root.mkdir(parents=True, exist_ok=True)

    sparse_components = getattr(pool, "_sparse_csc_components", None)
    label = np.asarray(pool.label, dtype=np.float32)
    weight = np.ones(pool.num_rows, dtype=np.float32) if pool.weight is None else np.asarray(pool.weight, dtype=np.float32)
    group_id = None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64)
    baseline = None if pool.baseline is None else np.asarray(pool.baseline, dtype=np.float32)

    label_map = _open_memmap_array(shard_root / "label.npy", label.shape, dtype=np.float32)
    label_map[...] = label
    label_map.flush()
    weight_map = _open_memmap_array(shard_root / "weight.npy", weight.shape, dtype=np.float32)
    weight_map[...] = weight
    weight_map.flush()

    group_path = None
    if group_id is not None:
        group_path = shard_root / "group_id.npy"
        group_map = _open_memmap_array(group_path, group_id.shape, dtype=np.int64)
        group_map[...] = group_id
        group_map.flush()

    baseline_path = None
    if baseline is not None:
        baseline_path = shard_root / "baseline.npy"
        baseline_map = _open_memmap_array(baseline_path, baseline.shape, dtype=np.float32)
        baseline_map[...] = baseline
        baseline_map.flush()

    manifest: Dict[str, Any] = {
        "rank": int(config["rank"]),
        "num_rows": int(pool.num_rows),
        "num_cols": int(pool.num_cols),
        "cat_features": list(pool.cat_features),
        "feature_names": None if pool.feature_names is None else list(pool.feature_names),
        "label_path": "label.npy",
        "weight_path": "weight.npy",
        "group_id_path": None if group_path is None else group_path.name,
        "baseline_path": None if baseline_path is None else baseline_path.name,
    }

    if sparse_components is None:
        dense_data = getattr(pool, "_dense_data_ref", None)
        if dense_data is None:
            dense_data = np.asfortranarray(pool.data, dtype=np.float32)
        else:
            dense_data = np.asfortranarray(dense_data, dtype=np.float32)
        data_map = _open_memmap_array(
            shard_root / "data.npy",
            dense_data.shape,
            dtype=np.float32,
            fortran_order=True,
        )
        data_map[...] = dense_data
        data_map.flush()
        manifest["storage"] = "dense"
        manifest["data_path"] = "data.npy"
    else:
        sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
        data_map = _open_memmap_array(
            shard_root / "sparse_data.npy", sparse_data.shape, dtype=np.float32
        )
        data_map[...] = np.asarray(sparse_data, dtype=np.float32)
        data_map.flush()
        indices_map = _open_memmap_array(
            shard_root / "sparse_indices.npy", sparse_indices.shape, dtype=np.int64
        )
        indices_map[...] = np.asarray(sparse_indices, dtype=np.int64)
        indices_map.flush()
        indptr_map = _open_memmap_array(
            shard_root / "sparse_indptr.npy", sparse_indptr.shape, dtype=np.int64
        )
        indptr_map[...] = np.asarray(sparse_indptr, dtype=np.int64)
        indptr_map.flush()
        manifest["storage"] = "sparse"
        manifest["shape"] = [int(shape[0]), int(shape[1])]
        manifest["sparse_data_path"] = "sparse_data.npy"
        manifest["sparse_indices_path"] = "sparse_indices.npy"
        manifest["sparse_indptr_path"] = "sparse_indptr.npy"

    manifest_path = shard_root / "manifest.json"
    manifest_temp_path = manifest_path.with_suffix(".tmp")
    with manifest_temp_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    manifest_temp_path.replace(manifest_path)
    return manifest_path

def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

