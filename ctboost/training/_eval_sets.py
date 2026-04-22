"""Evaluation-set helpers for ctboost.training."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

from ..core import Pool, _clone_pool

def _load_sklearn_splitters():
    try:
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
    except ModuleNotFoundError as exc:
        raise ImportError(
            "ctboost.cv requires scikit-learn>=1.3. "
            "Install 'ctboost[sklearn]' or add scikit-learn to your environment."
        ) from exc
    return GroupKFold, KFold, StratifiedKFold

def _is_eval_set_entry(value: Any) -> bool:
    return isinstance(value, Pool) or (isinstance(value, tuple) and len(value) in {2, 3})

def _normalize_eval_sets(eval_set: Any) -> List[Any]:
    if eval_set is None:
        return []
    if _is_eval_set_entry(eval_set):
        return [eval_set]
    if isinstance(eval_set, list) and all(_is_eval_set_entry(item) for item in eval_set):
        return list(eval_set)
    raise TypeError(
        "eval_set must be a Pool, a single (X, y) or (X, y, group_id) tuple, "
        "or a list of such entries"
    )

def _normalize_eval_set(eval_set: Any) -> Optional[Any]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    if len(normalized) != 1:
        raise TypeError("expected a single eval_set entry")
    return normalized[0]

def _normalize_eval_names(eval_names: Any, eval_count: int) -> List[str]:
    if eval_count == 0:
        if eval_names is not None and eval_names != [] and eval_names != ():
            raise ValueError("eval_names requires at least one eval_set")
        return []
    if eval_names is None:
        if eval_count == 1:
            return ["validation"]
        return [f"validation_{index}" for index in range(eval_count)]
    if isinstance(eval_names, str):
        if eval_count != 1:
            raise ValueError("eval_names must provide one name per eval_set")
        names = [eval_names]
    elif isinstance(eval_names, (list, tuple)):
        if len(eval_names) != eval_count:
            raise ValueError("eval_names must provide one name per eval_set")
        names = [str(name) for name in eval_names]
    else:
        raise TypeError("eval_names must be a string or a sequence of strings")
    if len(set(names)) != len(names):
        raise ValueError("eval_names entries must be unique")
    return names

def _resolve_eval_pool(eval_set: Any) -> Optional[Pool]:
    normalized = _normalize_eval_set(eval_set)
    if normalized is None or isinstance(normalized, Pool):
        return normalized

    if len(normalized) == 2:
        X_eval, y_eval = normalized
        return _pool_from_data_and_label(X_eval, y_eval)

    X_eval, y_eval, group_id = normalized
    return _pool_from_data_and_label(X_eval, y_eval, group_id=group_id)

def _slice_baseline(baseline: Optional[np.ndarray], index_array: np.ndarray) -> Optional[np.ndarray]:
    if baseline is None:
        return None
    resolved = np.asarray(baseline, dtype=np.float32)
    if resolved.ndim == 1:
        return resolved[index_array]
    return resolved[index_array, :]

def _slice_pairs(
    pairs: Optional[np.ndarray],
    pairs_weight: Optional[np.ndarray],
    index_array: np.ndarray,
    *,
    original_num_rows: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if pairs is None:
        return None, None

    resolved_pairs = np.asarray(pairs, dtype=np.int64)
    if resolved_pairs.size == 0:
        empty_pairs = np.empty((0, 2), dtype=np.int64)
        empty_pairs_weight = None
        if pairs_weight is not None:
            empty_pairs_weight = np.empty((0,), dtype=np.float32)
        return empty_pairs, empty_pairs_weight

    remap = np.full(original_num_rows, -1, dtype=np.int64)
    remap[index_array] = np.arange(index_array.shape[0], dtype=np.int64)
    remapped_pairs = remap[resolved_pairs]
    keep_mask = np.all(remapped_pairs >= 0, axis=1)
    kept_pairs = np.ascontiguousarray(remapped_pairs[keep_mask], dtype=np.int64)
    if pairs_weight is None:
        return kept_pairs, None
    resolved_pairs_weight = np.asarray(pairs_weight, dtype=np.float32)
    return kept_pairs, np.ascontiguousarray(resolved_pairs_weight[keep_mask], dtype=np.float32)

def _slice_pool(pool: Pool, indices: Iterable[int]) -> Pool:
    index_array = np.asarray(list(indices), dtype=np.int64)
    weight = None if pool.weight is None else pool.weight[index_array]
    group_weight = None if pool.group_weight is None else np.asarray(pool.group_weight, dtype=np.float32)[index_array]
    subgroup_id = None if pool.subgroup_id is None else np.asarray(pool.subgroup_id, dtype=np.int64)[index_array]
    baseline = _slice_baseline(pool.baseline, index_array)
    pairs, pairs_weight = _slice_pairs(
        pool.pairs,
        pool.pairs_weight,
        index_array,
        original_num_rows=pool.num_rows,
    )
    return Pool(
        data=pool.data[index_array],
        label=pool.label[index_array],
        cat_features=pool.cat_features,
        weight=weight,
        group_id=None if pool.group_id is None else pool.group_id[index_array],
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=pool.feature_names,
        column_roles=pool.column_roles,
        feature_metadata=pool.feature_metadata,
        categorical_schema=pool.categorical_schema,
        _releasable_feature_storage=True,
    )

def _stack_histories(histories: List[np.ndarray]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=np.float32)
    max_length = max(history.shape[0] for history in histories)
    stacked = np.full((max_length, len(histories)), np.nan, dtype=np.float32)
    for column_index, history in enumerate(histories):
        stacked[: history.shape[0], column_index] = history
    return stacked

