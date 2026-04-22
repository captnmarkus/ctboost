"""Private helpers for pool preparation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import Pool, _is_scipy_sparse_matrix, _normalize_categorical_features, _scipy_sparse_to_csc_components
from .feature_pipeline import FeaturePipeline


class ExternalMemoryBacking:
    """Owns temporary disk-backed arrays for a Pool."""

    def __init__(self, root: Path, arrays: Sequence[np.ndarray]) -> None:
        self.root = Path(root)
        self.arrays = list(arrays)

    def cleanup(self) -> None:
        for array in self.arrays:
            flush = getattr(array, "flush", None)
            if callable(flush):
                flush()


_FEATURE_PIPELINE_KEYS = {
    "cat_features",
    "ordered_ctr",
    "one_hot_max_size",
    "max_cat_to_onehot",
    "max_cat_threshold",
    "categorical_combinations",
    "pairwise_categorical_combinations",
    "simple_ctr",
    "combinations_ctr",
    "per_feature_ctr",
    "text_features",
    "text_hash_dim",
    "embedding_features",
    "embedding_stats",
    "ctr_prior_strength",
    "random_seed",
}

_EXTERNAL_MEMORY_KEYS = {
    "external_memory",
    "external_memory_dir",
    "eval_external_memory",
    "eval_external_memory_dir",
}


def _normalize_optional_array(values: Any, dtype: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=dtype)


def _open_memmap(path: Path, shape: Tuple[int, ...], *, dtype: Any, fortran_order: bool = False) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=shape,
        fortran_order=fortran_order,
    )


def _prepare_feature_pipeline(
    data: Any,
    label: Any,
    *,
    feature_names: Optional[Sequence[str]],
    params: Mapping[str, Any],
    feature_pipeline: Optional[FeaturePipeline] = None,
    fit_feature_pipeline: bool = True,
) -> Tuple[Any, Any, Sequence[int], Optional[Sequence[str]], Optional[FeaturePipeline]]:
    uses_pipeline = feature_pipeline is not None or _uses_feature_pipeline_params(params)
    if not uses_pipeline:
        return data, label, _normalize_categorical_features(params.get("cat_features")), feature_names, None

    pipeline = feature_pipeline
    if pipeline is None:
        pipeline = FeaturePipeline(
            cat_features=params.get("cat_features"),
            ordered_ctr=bool(params.get("ordered_ctr", False)),
            one_hot_max_size=int(params.get("one_hot_max_size", params.get("max_cat_to_onehot", 0))),
            max_cat_threshold=int(params.get("max_cat_threshold", 0)),
            categorical_combinations=params.get("categorical_combinations"),
            pairwise_categorical_combinations=bool(params.get("pairwise_categorical_combinations", False)),
            simple_ctr=params.get("simple_ctr"),
            combinations_ctr=params.get("combinations_ctr"),
            per_feature_ctr=params.get("per_feature_ctr"),
            text_features=params.get("text_features"),
            text_hash_dim=int(params.get("text_hash_dim", 64)),
            embedding_features=params.get("embedding_features"),
            embedding_stats=params.get("embedding_stats", ("mean", "std", "min", "max", "l2")),
            ctr_prior_strength=float(params.get("ctr_prior_strength", 1.0)),
            random_seed=int(params.get("random_seed", 0)),
        )

    if fit_feature_pipeline:
        transformed_data, transformed_cat_features, transformed_feature_names = (
            pipeline.fit_transform_array(data, label, feature_names=feature_names)
        )
    else:
        transformed_data, transformed_cat_features, transformed_feature_names = (
            pipeline.transform_array(data, feature_names=feature_names)
        )
    return transformed_data, label, transformed_cat_features, transformed_feature_names, pipeline


def _uses_feature_pipeline_params(params: Mapping[str, Any]) -> bool:
    return bool(
        params.get("ordered_ctr")
        or params.get("cat_features")
        or params.get("one_hot_max_size")
        or params.get("max_cat_to_onehot")
        or params.get("max_cat_threshold")
        or params.get("categorical_combinations")
        or params.get("pairwise_categorical_combinations")
        or params.get("simple_ctr")
        or params.get("combinations_ctr")
        or params.get("per_feature_ctr")
        or params.get("text_features")
        or params.get("embedding_features")
    )


def _dense_external_memory_pool(
    data: Any,
    label: Any,
    *,
    cat_features: Sequence[int],
    weight: Any,
    group_id: Any,
    group_weight: Any,
    subgroup_id: Any,
    baseline: Any,
    pairs: Any,
    pairs_weight: Any,
    feature_names: Optional[Sequence[str]],
    column_roles: Any,
    feature_metadata: Optional[Mapping[str, Any]],
    categorical_schema: Optional[Mapping[str, Any]],
    directory: Optional[Any],
) -> Pool:
    root = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="ctboost-ext-"))
    root.mkdir(parents=True, exist_ok=True)
    data_array = np.asarray(data, dtype=np.float32, order="F")
    data_map = _open_memmap(root / "data.npy", data_array.shape, dtype=np.float32, fortran_order=True)
    data_map[...] = data_array
    data_map.flush()
    weight_array = _normalize_optional_array(weight, np.float32)
    group_array = _normalize_optional_array(group_id, np.int64)
    pool = Pool(
        data=np.load(root / "data.npy", mmap_mode="r"),
        label=np.asarray(label, dtype=np.float32),
        cat_features=list(cat_features),
        weight=weight_array,
        group_id=group_array,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=None if feature_names is None else list(feature_names),
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        _releasable_feature_storage=True,
    )
    pool._external_memory_backing = ExternalMemoryBacking(root, [data_map])
    return pool


def _sparse_external_memory_pool(
    data: Any,
    label: Any,
    *,
    cat_features: Sequence[int],
    weight: Any,
    group_id: Any,
    group_weight: Any,
    subgroup_id: Any,
    baseline: Any,
    pairs: Any,
    pairs_weight: Any,
    feature_names: Optional[Sequence[str]],
    column_roles: Any,
    feature_metadata: Optional[Mapping[str, Any]],
    categorical_schema: Optional[Mapping[str, Any]],
    directory: Optional[Any],
) -> Pool:
    root = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="ctboost-ext-"))
    root.mkdir(parents=True, exist_ok=True)
    sparse_data, sparse_indices, sparse_indptr, shape = _scipy_sparse_to_csc_components(data)
    data_map = _open_memmap(root / "sparse_data.npy", sparse_data.shape, dtype=np.float32)
    data_map[...] = sparse_data
    data_map.flush()
    indices_map = _open_memmap(root / "sparse_indices.npy", sparse_indices.shape, dtype=np.int64)
    indices_map[...] = sparse_indices
    indices_map.flush()
    indptr_map = _open_memmap(root / "sparse_indptr.npy", sparse_indptr.shape, dtype=np.int64)
    indptr_map[...] = sparse_indptr
    indptr_map.flush()
    weight_array = _normalize_optional_array(weight, np.float32)
    group_array = _normalize_optional_array(group_id, np.int64)
    pool = Pool.from_csc_components(
        np.load(root / "sparse_data.npy", mmap_mode="r"),
        np.load(root / "sparse_indices.npy", mmap_mode="r"),
        np.load(root / "sparse_indptr.npy", mmap_mode="r"),
        shape,
        np.asarray(label, dtype=np.float32),
        cat_features=list(cat_features),
        weight=weight_array,
        group_id=group_array,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=None if feature_names is None else list(feature_names),
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        _releasable_feature_storage=True,
    )
    pool._external_memory_backing = ExternalMemoryBacking(root, [data_map, indices_map, indptr_map])
    return pool


def _external_memory_pool(
    data: Any,
    label: Any,
    *,
    cat_features: Sequence[int],
    weight: Any,
    group_id: Any,
    group_weight: Any,
    subgroup_id: Any,
    baseline: Any,
    pairs: Any,
    pairs_weight: Any,
    feature_names: Optional[Sequence[str]],
    column_roles: Any,
    feature_metadata: Optional[Mapping[str, Any]],
    categorical_schema: Optional[Mapping[str, Any]],
    directory: Optional[Any],
) -> Pool:
    if _is_scipy_sparse_matrix(data):
        return _sparse_external_memory_pool(
            data,
            label,
            cat_features=cat_features,
            weight=weight,
            group_id=group_id,
            group_weight=group_weight,
            subgroup_id=subgroup_id,
            baseline=baseline,
            pairs=pairs,
            pairs_weight=pairs_weight,
            feature_names=feature_names,
            column_roles=column_roles,
            feature_metadata=feature_metadata,
            categorical_schema=categorical_schema,
            directory=directory,
        )
    return _dense_external_memory_pool(
        data,
        label,
        cat_features=cat_features,
        weight=weight,
        group_id=group_id,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=feature_names,
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        directory=directory,
    )
