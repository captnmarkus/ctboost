"""Ranking metadata helpers for ctboost.core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from .normalization import _is_pandas_series
if TYPE_CHECKING:
    from .pool import Pool


def _normalize_group_id(group_id: Any) -> Optional[np.ndarray]:
    return _normalize_identifier_array(group_id, name="group_id")


def _normalize_identifier_array(values: Any, *, name: str) -> Optional[np.ndarray]:
    if values is None:
        return None
    if _is_pandas_series(values):
        values = values.to_numpy(copy=False)
    group_array = np.asarray(values)
    if group_array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.issubdtype(group_array.dtype, np.integer):
        return np.ascontiguousarray(group_array, dtype=np.int64)
    _, inverse = np.unique(group_array, return_inverse=True)
    return np.ascontiguousarray(inverse, dtype=np.int64)


def _normalize_group_weight(
    group_weight: Any,
    *,
    group_id: Optional[np.ndarray],
    num_rows: int,
) -> Optional[np.ndarray]:
    if group_weight is None:
        return None
    if group_id is None:
        raise ValueError("group_weight requires group_id")
    if _is_pandas_series(group_weight):
        group_weight = group_weight.to_numpy(copy=False)
    weight_array = np.asarray(group_weight, dtype=np.float32)
    if weight_array.ndim != 1 or weight_array.shape[0] != num_rows:
        raise ValueError("group_weight size must match the number of data rows")
    return np.ascontiguousarray(weight_array, dtype=np.float32)


def _normalize_baseline(baseline: Any, *, num_rows: int) -> Optional[np.ndarray]:
    if baseline is None:
        return None
    if _is_pandas_series(baseline):
        baseline = baseline.to_numpy(copy=False)
    baseline_array = np.asarray(baseline, dtype=np.float32)
    if baseline_array.ndim == 1:
        if baseline_array.shape[0] != num_rows:
            raise ValueError("baseline size must match the number of data rows")
        return np.ascontiguousarray(baseline_array, dtype=np.float32)
    if baseline_array.ndim == 2:
        if baseline_array.shape[0] != num_rows:
            raise ValueError("baseline row count must match the number of data rows")
        if baseline_array.shape[1] <= 0:
            raise ValueError("baseline must have at least one prediction column")
        return np.ascontiguousarray(baseline_array, dtype=np.float32)
    raise ValueError("baseline must be a 1D or 2D array")


def _normalize_pairs(
    pairs: Any,
    *,
    pairs_weight: Any = None,
    group_id: Optional[np.ndarray],
    num_rows: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if pairs is None:
        if pairs_weight is not None:
            raise ValueError("pairs_weight requires pairs")
        return None, None
    if group_id is None:
        raise ValueError("pairs requires group_id")
    pairs_array = np.asarray(pairs, dtype=np.int64)
    if pairs_array.ndim != 2 or pairs_array.shape[1] != 2:
        raise ValueError("pairs must be a 2D array with shape (n_pairs, 2)")
    if pairs_array.size and (np.any(pairs_array < 0) or np.any(pairs_array >= num_rows)):
        raise ValueError("pairs must reference valid row indices")
    resolved_pairs = np.ascontiguousarray(pairs_array, dtype=np.int64)
    if pairs_weight is None:
        return resolved_pairs, None
    weight_array = np.asarray(pairs_weight, dtype=np.float32)
    if weight_array.ndim != 1 or weight_array.shape[0] != resolved_pairs.shape[0]:
        raise ValueError("pairs_weight size must match the number of pairs")
    return resolved_pairs, np.ascontiguousarray(weight_array, dtype=np.float32)


def _clone_pool(
    pool: "Pool",
    *,
    label: Any = None,
    weight: Any = None,
    group_id: Any = None,
    group_weight: Any = None,
    subgroup_id: Any = None,
    baseline: Any = None,
    pairs: Any = None,
    pairs_weight: Any = None,
    feature_names: Optional[List[str]] = None,
    column_roles: Any = None,
    feature_metadata: Any = None,
    categorical_schema: Any = None,
    releasable_feature_storage: bool = True,
) -> "Pool":
    resolved_label = pool.label if label is None else label
    resolved_weight = pool.weight if weight is None else weight
    resolved_group_id = pool.group_id if group_id is None else group_id
    resolved_group_weight = pool.group_weight if group_weight is None else group_weight
    resolved_subgroup_id = pool.subgroup_id if subgroup_id is None else subgroup_id
    resolved_baseline = pool.baseline if baseline is None else baseline
    resolved_pairs = pool.pairs if pairs is None else pairs
    resolved_pairs_weight = pool.pairs_weight if pairs_weight is None else pairs_weight
    resolved_feature_names = pool.feature_names if feature_names is None else feature_names
    resolved_column_roles = pool.column_roles if column_roles is None else column_roles
    resolved_feature_metadata = pool.feature_metadata if feature_metadata is None else feature_metadata
    resolved_categorical_schema = pool.categorical_schema if categorical_schema is None else categorical_schema
    sparse_components = getattr(pool, "_sparse_csc_components", None)
    if sparse_components is not None:
        sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
        from .pool import Pool

        cloned = Pool.from_csc_components(
            sparse_data,
            sparse_indices,
            sparse_indptr,
            shape,
            resolved_label,
            cat_features=pool.cat_features,
            weight=resolved_weight,
            group_id=resolved_group_id,
            group_weight=resolved_group_weight,
            subgroup_id=resolved_subgroup_id,
            baseline=resolved_baseline,
            pairs=resolved_pairs,
            pairs_weight=resolved_pairs_weight,
            feature_names=resolved_feature_names,
            column_roles=resolved_column_roles,
            feature_metadata=resolved_feature_metadata,
            categorical_schema=resolved_categorical_schema,
            _releasable_feature_storage=releasable_feature_storage,
        )
    else:
        dense_data = getattr(pool, "_dense_data_ref", None)
        if dense_data is None:
            dense_data = pool.data
        from .pool import Pool

        cloned = Pool(
            data=dense_data,
            label=resolved_label,
            cat_features=pool.cat_features,
            weight=resolved_weight,
            group_id=resolved_group_id,
            group_weight=resolved_group_weight,
            subgroup_id=resolved_subgroup_id,
            baseline=resolved_baseline,
            pairs=resolved_pairs,
            pairs_weight=resolved_pairs_weight,
            feature_names=resolved_feature_names,
            column_roles=resolved_column_roles,
            feature_metadata=resolved_feature_metadata,
            categorical_schema=resolved_categorical_schema,
            _releasable_feature_storage=releasable_feature_storage,
        )
    cloned._feature_pipeline = getattr(pool, "_feature_pipeline", None)
    cloned._external_memory_backing = getattr(pool, "_external_memory_backing", None)
    return cloned
