"""Python wrapper types for the native CTBoost core."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from . import _core

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


def _dataframe_to_numpy(
    dataframe: Any, cat_features: Optional[List[int]]
) -> Tuple[np.ndarray, List[int]]:
    resolved_cat_features = set(_normalize_categorical_features(cat_features))
    data_array = np.empty((dataframe.shape[0], dataframe.shape[1]), dtype=np.float32)

    for column_index, (_, series) in enumerate(dataframe.items()):
        if _is_pandas_categorical_series(series):
            resolved_cat_features.add(column_index)
            if isinstance(series.dtype, pd.CategoricalDtype):
                codes = series.cat.codes
            else:
                codes = series.astype("category").cat.codes
            data_array[:, column_index] = codes.to_numpy(copy=False)
            continue

        data_array[:, column_index] = _series_to_float32(series)

    return data_array, sorted(resolved_cat_features)


class Pool:
    """Python wrapper around the native `ctboost::Pool`."""

    def __init__(
        self,
        data: Any,
        label: Any,
        cat_features: Optional[List[int]] = None,
        weight: Any = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        resolved_cat_features = _normalize_categorical_features(cat_features)
        if _is_pandas_dataframe(data):
            if feature_names is None:
                feature_names = [str(column_name) for column_name in data.columns]
            data, resolved_cat_features = _dataframe_to_numpy(data, resolved_cat_features)
        if _is_pandas_series(label):
            label = label.to_numpy(copy=False)

        data_array = np.ascontiguousarray(data, dtype=np.float32)
        label_array = np.ascontiguousarray(label, dtype=np.float32)
        if weight is None:
            native_weight = np.ones(data_array.shape[0], dtype=np.float32)
            self.weight = None
        else:
            native_weight = np.ascontiguousarray(weight, dtype=np.float32)
            self.weight = native_weight

        self._handle = _core.Pool(data_array, label_array, resolved_cat_features, native_weight)
        self.num_rows = self._handle.num_rows()
        self.num_cols = self._handle.num_cols()
        self.cat_features = list(self._handle.cat_features())
        self.feature_names = None if feature_names is None else list(feature_names)

        if self.weight is not None:
            if self.weight.ndim != 1:
                raise ValueError("weight must be a 1D NumPy array")
            if self.weight.shape[0] != self.num_rows:
                raise ValueError("weight size must match the number of data rows")
        if self.feature_names is not None:
            if len(self.feature_names) != self.num_cols:
                raise ValueError("feature_names size must match the number of data columns")

    @property
    def feature_data(self) -> np.ndarray:
        return self._handle.feature_data()

    @property
    def label(self) -> np.ndarray:
        return self._handle.label()

    @property
    def data(self) -> np.ndarray:
        return self.feature_data.reshape((self.num_cols, self.num_rows)).T.copy()

    def __len__(self) -> int:
        return self.num_rows
