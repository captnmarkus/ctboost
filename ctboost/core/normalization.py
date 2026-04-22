"""Input normalization helpers for ctboost.core."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional at runtime
    pd = None


def _is_pandas_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _is_pandas_series(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.Series)


def _normalize_categorical_features(cat_features: Optional[List[int]]) -> List[int]:
    if cat_features is None:
        return []
    return sorted({int(feature_index) for feature_index in cat_features})


def _normalize_json_metadata(value: Any, *, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_metadata(item, name=name)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_normalize_json_metadata(item, name=name) for item in value]
    if isinstance(value, list):
        return [_normalize_json_metadata(item, name=name) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{name} must contain JSON-serializable values")


def _normalize_column_roles(
    column_roles: Any,
    *,
    num_cols: int,
    feature_names: Optional[List[str]],
) -> Optional[List[Optional[str]]]:
    if column_roles is None:
        return None
    if isinstance(column_roles, Mapping):
        feature_name_map = None if feature_names is None else {
            str(feature_name): index for index, feature_name in enumerate(feature_names)
        }
        resolved_roles: List[Optional[str]] = [None] * num_cols
        for key, value in column_roles.items():
            if isinstance(key, (np.integer, int)):
                feature_index = int(key)
            elif isinstance(key, str):
                if feature_name_map is None:
                    raise ValueError("column_roles cannot use feature names without feature_names")
                if key not in feature_name_map:
                    raise ValueError(f"column_roles references unknown feature name {key!r}")
                feature_index = feature_name_map[key]
            else:
                raise TypeError("column_roles keys must be feature indices or feature names")
            if feature_index < 0 or feature_index >= num_cols:
                raise ValueError("column_roles feature index is out of bounds")
            resolved_roles[feature_index] = None if value is None else str(value)
        return resolved_roles
    resolved_roles = [None if value is None else str(value) for value in list(column_roles)]
    if len(resolved_roles) != num_cols:
        raise ValueError("column_roles size must match the number of data columns")
    return resolved_roles


def _is_pandas_categorical_series(series: Any) -> bool:
    if pd is None:
        return False
    dtype = series.dtype
    return bool(
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def _series_to_float32(series: Any) -> np.ndarray:
    try:
        values = series.to_numpy(dtype=np.float32, copy=False)
    except (TypeError, ValueError):
        values = np.asarray(series.to_numpy(copy=False), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)
