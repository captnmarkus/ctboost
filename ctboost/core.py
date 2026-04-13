"""Python wrapper types for the native CTBoost core."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from . import _core

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional at runtime
    pd = None

try:
    from scipy import sparse as sp
except ImportError:  # pragma: no cover - scipy is optional at runtime
    sp = None


def _is_pandas_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _is_pandas_series(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.Series)


def _is_scipy_sparse_matrix(value: Any) -> bool:
    return sp is not None and sp.issparse(value)


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


def _normalize_group_id(group_id: Any) -> Optional[np.ndarray]:
    if group_id is None:
        return None
    if _is_pandas_series(group_id):
        group_id = group_id.to_numpy(copy=False)

    group_array = np.asarray(group_id)
    if group_array.ndim != 1:
        raise ValueError("group_id must be a 1D array")
    if np.issubdtype(group_array.dtype, np.integer):
        return np.ascontiguousarray(group_array, dtype=np.int64)

    _, inverse = np.unique(group_array, return_inverse=True)
    return np.ascontiguousarray(inverse, dtype=np.int64)


def _scipy_sparse_to_csc_components(matrix: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    csc_matrix = matrix.astype(np.float32, copy=False)
    if not sp.isspmatrix_csc(csc_matrix):
        csc_matrix = csc_matrix.tocsc(copy=False)
    sparse_data = np.ascontiguousarray(csc_matrix.data, dtype=np.float32)
    sparse_indices = np.ascontiguousarray(csc_matrix.indices, dtype=np.int64)
    sparse_indptr = np.ascontiguousarray(csc_matrix.indptr, dtype=np.int64)
    return sparse_data, sparse_indices, sparse_indptr, tuple(int(dimension) for dimension in csc_matrix.shape)


def _dataframe_to_numpy(
    dataframe: Any, cat_features: Optional[List[int]]
) -> Tuple[np.ndarray, List[int]]:
    resolved_cat_features = set(_normalize_categorical_features(cat_features))
    data_array = np.empty((dataframe.shape[0], dataframe.shape[1]), dtype=np.float32, order="F")

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

    @classmethod
    def from_csc_components(
        cls,
        sparse_data: np.ndarray,
        sparse_indices: np.ndarray,
        sparse_indptr: np.ndarray,
        shape: Tuple[int, int],
        label: Any,
        *,
        cat_features: Optional[List[int]] = None,
        weight: Any = None,
        group_id: Any = None,
        feature_names: Optional[List[str]] = None,
        _releasable_feature_storage: bool = False,
    ) -> "Pool":
        resolved_cat_features = _normalize_categorical_features(cat_features)
        if _is_pandas_series(label):
            label = label.to_numpy(copy=False)
        resolved_group_id = _normalize_group_id(group_id)
        label_array = np.ascontiguousarray(label, dtype=np.float32)
        if weight is None:
            native_weight = np.ones(int(shape[0]), dtype=np.float32)
            resolved_weight = None
        else:
            native_weight = np.ascontiguousarray(weight, dtype=np.float32)
            resolved_weight = native_weight
        native_group_id = (
            None if resolved_group_id is None else np.ascontiguousarray(resolved_group_id, dtype=np.int64)
        )

        handle = _core.Pool.from_csc(
            np.ascontiguousarray(sparse_data, dtype=np.float32),
            np.ascontiguousarray(sparse_indices, dtype=np.int64),
            np.ascontiguousarray(sparse_indptr, dtype=np.int64),
            int(shape[0]),
            int(shape[1]),
            label_array,
            resolved_cat_features,
            native_weight,
            native_group_id,
        )
        return cls._from_handle(
            handle,
            weight=resolved_weight,
            group_id=native_group_id,
            feature_names=feature_names,
            releasable_feature_storage=_releasable_feature_storage,
            dense_data=None,
            sparse_components=(
                np.ascontiguousarray(sparse_data, dtype=np.float32),
                np.ascontiguousarray(sparse_indices, dtype=np.int64),
                np.ascontiguousarray(sparse_indptr, dtype=np.int64),
                (int(shape[0]), int(shape[1])),
            ),
        )

    @classmethod
    def _from_handle(
        cls,
        handle: Any,
        *,
        weight: Optional[np.ndarray],
        group_id: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        releasable_feature_storage: bool,
        dense_data: Optional[np.ndarray],
        sparse_components: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]],
    ) -> "Pool":
        self = cls.__new__(cls)
        self._handle = handle
        if releasable_feature_storage:
            self._handle.set_feature_storage_releasable(True)
        self.num_rows = self._handle.num_rows()
        self.num_cols = self._handle.num_cols()
        self.cat_features = list(self._handle.cat_features())
        self.feature_names = None if feature_names is None else list(feature_names)
        self.group_id = None if group_id is None else np.asarray(group_id, dtype=np.int64).copy()
        self.weight = None if weight is None else np.asarray(weight, dtype=np.float32)
        self._releasable_feature_storage = bool(releasable_feature_storage)
        self._dense_data_ref = dense_data
        self._sparse_csc_components = sparse_components
        self._feature_pipeline = None
        self._external_memory_backing = None

        if self.weight is not None:
            if self.weight.ndim != 1:
                raise ValueError("weight must be a 1D NumPy array")
            if self.weight.shape[0] != self.num_rows:
                raise ValueError("weight size must match the number of data rows")
        if self.group_id is not None and self.group_id.shape[0] != self.num_rows:
            raise ValueError("group_id size must match the number of data rows")
        if self.feature_names is not None:
            if len(self.feature_names) != self.num_cols:
                raise ValueError("feature_names size must match the number of data columns")
        return self

    def __init__(
        self,
        data: Any,
        label: Any,
        cat_features: Optional[List[int]] = None,
        weight: Any = None,
        group_id: Any = None,
        feature_names: Optional[List[str]] = None,
        _releasable_feature_storage: bool = False,
    ) -> None:
        resolved_cat_features = _normalize_categorical_features(cat_features)
        sparse_components = None
        if _is_pandas_dataframe(data):
            if feature_names is None:
                feature_names = [str(column_name) for column_name in data.columns]
            data, resolved_cat_features = _dataframe_to_numpy(data, resolved_cat_features)
        elif _is_scipy_sparse_matrix(data):
            sparse_components = _scipy_sparse_to_csc_components(data)
        if _is_pandas_series(label):
            label = label.to_numpy(copy=False)
        resolved_group_id = _normalize_group_id(group_id)

        # The native pool stores features column-major and can memcpy Fortran-ordered
        # matrices directly, avoiding an extra full-table transpose on large datasets.
        label_array = np.ascontiguousarray(label, dtype=np.float32)
        if weight is None:
            num_rows = (
                sparse_components[3][0]
                if sparse_components is not None
                else int(np.asarray(data).shape[0])
            )
            native_weight = np.ones(num_rows, dtype=np.float32)
            self.weight = None
        else:
            native_weight = np.ascontiguousarray(weight, dtype=np.float32)
            self.weight = native_weight
        native_group_id = None if resolved_group_id is None else np.ascontiguousarray(resolved_group_id, dtype=np.int64)

        if sparse_components is None:
            data_array = np.asfortranarray(data, dtype=np.float32)
            handle = _core.Pool(
                data_array,
                label_array,
                resolved_cat_features,
                native_weight,
                native_group_id,
            )
        else:
            sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
            handle = _core.Pool.from_csc(
                sparse_data,
                sparse_indices,
                sparse_indptr,
                shape[0],
                shape[1],
                label_array,
                resolved_cat_features,
                native_weight,
                native_group_id,
            )
        initialized = self._from_handle(
            handle,
            weight=self.weight,
            group_id=native_group_id,
            feature_names=feature_names,
            releasable_feature_storage=_releasable_feature_storage,
            dense_data=None if sparse_components is not None else data_array,
            sparse_components=sparse_components,
        )
        self.__dict__.update(initialized.__dict__)

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


def _clone_pool(
    pool: Pool,
    *,
    label: Any = None,
    weight: Any = None,
    group_id: Any = None,
    feature_names: Optional[List[str]] = None,
    releasable_feature_storage: bool = True,
) -> Pool:
    resolved_label = pool.label if label is None else label
    resolved_weight = pool.weight if weight is None else weight
    resolved_group_id = pool.group_id if group_id is None else group_id
    resolved_feature_names = pool.feature_names if feature_names is None else feature_names

    sparse_components = getattr(pool, "_sparse_csc_components", None)
    if sparse_components is not None:
        sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
        cloned = Pool.from_csc_components(
            sparse_data,
            sparse_indices,
            sparse_indptr,
            shape,
            resolved_label,
            cat_features=pool.cat_features,
            weight=resolved_weight,
            group_id=resolved_group_id,
            feature_names=resolved_feature_names,
            _releasable_feature_storage=releasable_feature_storage,
        )
    else:
        dense_data = getattr(pool, "_dense_data_ref", None)
        if dense_data is None:
            dense_data = pool.data
        cloned = Pool(
            data=dense_data,
            label=resolved_label,
            cat_features=pool.cat_features,
            weight=resolved_weight,
            group_id=resolved_group_id,
            feature_names=resolved_feature_names,
            _releasable_feature_storage=releasable_feature_storage,
        )

    cloned._feature_pipeline = getattr(pool, "_feature_pipeline", None)
    cloned._external_memory_backing = getattr(pool, "_external_memory_backing", None)
    return cloned
