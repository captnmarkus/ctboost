"""Sparse and DataFrame conversion helpers for ctboost.core."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .normalization import (
    _is_pandas_categorical_series,
    _normalize_categorical_features,
    _series_to_float32,
    pd,
)

try:
    from scipy import sparse as sp
except ImportError:  # pragma: no cover - scipy is optional at runtime
    sp = None


def _is_scipy_sparse_matrix(value: Any) -> bool:
    return sp is not None and sp.issparse(value)


def _scipy_sparse_to_csc_components(matrix: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    csc_matrix = matrix.astype(np.float32, copy=False)
    if not sp.isspmatrix_csc(csc_matrix):
        csc_matrix = csc_matrix.tocsc(copy=False)
    sparse_data = np.ascontiguousarray(csc_matrix.data, dtype=np.float32)
    sparse_indices = np.ascontiguousarray(csc_matrix.indices, dtype=np.int64)
    sparse_indptr = np.ascontiguousarray(csc_matrix.indptr, dtype=np.int64)
    return sparse_data, sparse_indices, sparse_indptr, tuple(int(dimension) for dimension in csc_matrix.shape)


def _dataframe_to_numpy(
    dataframe: Any,
    cat_features: Optional[List[int]],
) -> Tuple[np.ndarray, List[int]]:
    resolved_cat_features = set(_normalize_categorical_features(cat_features))
    data_array = np.empty((dataframe.shape[0], dataframe.shape[1]), dtype=np.float32, order="F")
    for column_index, (_, series) in enumerate(dataframe.items()):
        if _is_pandas_categorical_series(series):
            resolved_cat_features.add(column_index)
            if pd is not None and isinstance(series.dtype, pd.CategoricalDtype):
                codes = series.cat.codes
            else:
                codes = series.astype("category").cat.codes
            data_array[:, column_index] = codes.to_numpy(copy=False)
            continue
        data_array[:, column_index] = _series_to_float32(series)
    return data_array, sorted(resolved_cat_features)
