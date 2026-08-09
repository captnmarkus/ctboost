"""Training-schema helpers for CTBoost."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..core import Pool

def _feature_name_to_index_map(pool: Pool) -> Optional[Dict[str, int]]:
    if pool.feature_names is None:
        return None
    return {str(name): index for index, name in enumerate(pool.feature_names)}

def _resolve_feature_index(
    key: Any,
    pool: Pool,
    param_name: str,
    feature_name_map: Optional[Dict[str, int]],
) -> int:
    if isinstance(key, (np.integer, int)):
        feature_index = int(key)
    elif isinstance(key, str):
        if feature_name_map is None:
            raise ValueError(
                f"{param_name} uses feature names, but the training pool does not have feature_names"
            )
        if key not in feature_name_map:
            raise ValueError(f"{param_name} references unknown feature name {key!r}")
        feature_index = feature_name_map[key]
    else:
        raise TypeError(f"{param_name} keys must be feature indices or feature names")

    if feature_index < 0 or feature_index >= pool.num_cols:
        raise ValueError(f"{param_name} feature index {feature_index} is out of bounds")
    return feature_index

def _resolve_feature_int_config(value: Any, pool: Pool, param_name: str) -> List[int]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved = [0] * pool.num_cols
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            entry = int(raw_entry)
            if entry < 0:
                raise ValueError(f"{param_name} entries must be non-negative")
            resolved[feature_index] = entry
        return resolved

    entries = list(np.asarray(value).reshape(-1).tolist()) if isinstance(value, np.ndarray) else list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    resolved = [int(entry) for entry in entries]
    if any(entry < 0 for entry in resolved):
        raise ValueError(f"{param_name} entries must be non-negative")
    return resolved

def _resolve_feature_string_config(value: Any, pool: Pool, param_name: str) -> List[str]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved = [""] * pool.num_cols
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            resolved[feature_index] = "" if raw_entry is None else str(raw_entry)
        return resolved

    entries = list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    return ["" if entry is None else str(entry) for entry in entries]

def _resolve_feature_border_config(value: Any, pool: Pool, param_name: str) -> List[List[float]]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved: List[List[float]] = [[] for _ in range(pool.num_cols)]
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            resolved[feature_index] = [float(border) for border in raw_entry]
        return resolved

    entries = list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    return [[float(border) for border in borders] for borders in entries]

def _resolve_feature_float_config(value: Any, pool: Pool, param_name: str) -> List[float]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved = [0.0] * pool.num_cols
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            entry = float(raw_entry)
            if entry < 0.0:
                raise ValueError(f"{param_name} entries must be non-negative")
            resolved[feature_index] = entry
        return resolved

    entries = list(np.asarray(value).reshape(-1).tolist()) if isinstance(value, np.ndarray) else list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    resolved = [float(entry) for entry in entries]
    if any(entry < 0.0 for entry in resolved):
        raise ValueError(f"{param_name} entries must be non-negative")
    return resolved

def _objective_name(params: Mapping[str, Any]) -> str:
    objective = params.get("objective", params.get("loss_function", "RMSE"))
    native_objective = getattr(objective, "native_objective", None)
    if native_objective is not None:
        return str(native_objective)
    return "RMSE" if callable(objective) else str(objective)

def _is_binary_classification_objective(objective_name: str) -> bool:
    return objective_name.lower() in {"logloss", "binary_logloss", "binary:logistic"}

def _is_classification_objective(objective_name: str) -> bool:
    return objective_name.lower() in {
        "logloss",
        "binary_logloss",
        "binary:logistic",
        "multiclass",
        "softmax",
        "softmaxloss",
    }

def _is_ranking_objective(objective_name: str) -> bool:
    return objective_name.lower() in {
        "pairlogit",
        "pairwise",
        "ranknet",
        "lambdamart",
        "lambdarank",
        "rank:ndcg",
    }

def _validate_uniform_query_weights(pool: Pool, *, context: str, consumer: str) -> None:
    if pool.group_id is None or pool.weight is None:
        return
    group_ids = np.asarray(pool.group_id, dtype=np.int64)
    weights = np.asarray(pool.weight, dtype=np.float64)
    _, first_indices, inverse = np.unique(
        group_ids,
        return_index=True,
        return_inverse=True,
    )
    reference_weights = weights[first_indices[inverse]]
    if not np.all(
        np.isclose(weights, reference_weights, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError(
            f"{consumer} requires sample weights to be uniform within each "
            f"group_id in {context}; use one query weight per group (or "
            "group_weight) instead"
        )

def _validate_ndcg_weights(pool: Pool, *, context: str) -> None:
    _validate_uniform_query_weights(pool, context=context, consumer="NDCG")

def _validate_objective_labels(objective_name: str, pool: Pool, *, context: str) -> None:
    normalized = objective_name.lower()
    labels = np.asarray(pool.label, dtype=np.float32)
    if normalized in {"gamma", "gammaloss", "reg:gamma"}:
        if labels.ndim != 1 or not np.all(np.isfinite(labels) & (labels > 0.0)):
            raise ValueError(f"Gamma objective requires finite positive labels in {context}")
    if normalized in {"lambdamart", "lambdarank", "rank:ndcg"}:
        if labels.ndim != 1 or not np.all(np.isfinite(labels) & (labels >= 0.0)):
            raise ValueError(
                f"LambdaMART requires finite non-negative relevance labels in {context}"
            )
        _validate_uniform_query_weights(
            pool,
            context=context,
            consumer="LambdaMART",
        )

def _pool_has_extended_ranking_metadata(pool: Pool) -> bool:
    return pool.group_weight is not None or pool.subgroup_id is not None or pool.pairs is not None

def _validate_distributed_pool_metadata(pool: Pool, *, context: str) -> None:
    if _pool_has_extended_ranking_metadata(pool):
        raise ValueError(
            f"distributed training does not yet support subgroup_id, group_weight, or pairs metadata on {context}"
        )

def _baseline_matrix_for_prediction(pool: Pool, prediction_dimension: int) -> Optional[np.ndarray]:
    if pool.baseline is None:
        return None
    baseline = np.asarray(pool.baseline, dtype=np.float32)
    if baseline.ndim == 1:
        if prediction_dimension != 1:
            raise ValueError("pool baseline dimension does not match the model prediction dimension")
        return baseline.reshape((pool.num_rows, 1))
    if baseline.ndim != 2:
        raise ValueError("baseline must be a 1D or 2D array")
    if baseline.shape[1] != prediction_dimension:
        if prediction_dimension == 1 and baseline.shape[1] == 1:
            return baseline
        raise ValueError("pool baseline dimension does not match the model prediction dimension")
    return baseline

def _pool_schema_metadata(pool: Pool) -> Dict[str, Any]:
    return {
        "num_features": int(pool.num_cols),
        "cat_features": [int(feature_index) for feature_index in pool.cat_features],
        "feature_names": None if pool.feature_names is None else [str(name) for name in pool.feature_names],
        "column_roles": None if getattr(pool, "column_roles", None) is None else list(pool.column_roles),
        "feature_metadata": getattr(pool, "feature_metadata", None),
        "categorical_schema": getattr(pool, "categorical_schema", None),
    }

