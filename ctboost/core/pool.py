"""Pool wrapper for the native CTBoost core."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from .. import _core
from ._handle import _pool_from_handle
from .normalization import (
    _is_pandas_dataframe,
    _is_pandas_series,
    _normalize_categorical_features,
    _normalize_column_roles,
    _normalize_json_metadata,
)
from .ranking_metadata import (
    _normalize_baseline,
    _normalize_group_id,
    _normalize_group_weight,
    _normalize_identifier_array,
    _normalize_pairs,
)
from .sparse import _dataframe_to_numpy, _is_scipy_sparse_matrix, _scipy_sparse_to_csc_components


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
        group_weight: Any = None,
        subgroup_id: Any = None,
        baseline: Any = None,
        pairs: Any = None,
        pairs_weight: Any = None,
        feature_names: Optional[List[str]] = None,
        column_roles: Any = None,
        feature_metadata: Optional[Mapping[str, Any]] = None,
        categorical_schema: Optional[Mapping[str, Any]] = None,
        _releasable_feature_storage: bool = False,
    ) -> "Pool":
        resolved_cat_features = _normalize_categorical_features(cat_features)
        resolved_feature_names = None if feature_names is None else [str(name) for name in feature_names]
        resolved_column_roles = _normalize_column_roles(
            column_roles,
            num_cols=int(shape[1]),
            feature_names=resolved_feature_names,
        )
        resolved_feature_metadata = _normalize_json_metadata(feature_metadata, name="feature_metadata")
        resolved_categorical_schema = _normalize_json_metadata(categorical_schema, name="categorical_schema")
        if _is_pandas_series(label):
            label = label.to_numpy(copy=False)
        resolved_group_id = _normalize_group_id(group_id)
        resolved_subgroup_id = _normalize_identifier_array(subgroup_id, name="subgroup_id")
        resolved_group_weight = _normalize_group_weight(group_weight, group_id=resolved_group_id, num_rows=int(shape[0]))
        resolved_baseline = _normalize_baseline(baseline, num_rows=int(shape[0]))
        resolved_pairs, resolved_pairs_weight = _normalize_pairs(
            pairs,
            pairs_weight=pairs_weight,
            group_id=resolved_group_id,
            num_rows=int(shape[0]),
        )
        label_array = np.ascontiguousarray(label, dtype=np.float32)
        native_weight = np.ones(int(shape[0]), dtype=np.float32) if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
        resolved_weight = None if weight is None else native_weight
        native_group_id = None if resolved_group_id is None else np.ascontiguousarray(resolved_group_id, dtype=np.int64)
        native_group_weight = None if resolved_group_weight is None else np.ascontiguousarray(resolved_group_weight, dtype=np.float32)
        native_subgroup_id = None if resolved_subgroup_id is None else np.ascontiguousarray(resolved_subgroup_id, dtype=np.int64)
        native_baseline = None if resolved_baseline is None else np.ascontiguousarray(resolved_baseline, dtype=np.float32)
        native_pairs = None if resolved_pairs is None else np.ascontiguousarray(resolved_pairs, dtype=np.int64)
        native_pairs_weight = None if resolved_pairs_weight is None else np.ascontiguousarray(resolved_pairs_weight, dtype=np.float32)
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
            native_group_weight,
            native_subgroup_id,
            native_baseline,
            native_pairs,
            native_pairs_weight,
        )
        return _pool_from_handle(
            cls,
            handle,
            weight=resolved_weight,
            group_id=native_group_id,
            group_weight=native_group_weight,
            subgroup_id=native_subgroup_id,
            baseline=native_baseline,
            pairs=native_pairs,
            pairs_weight=native_pairs_weight,
            feature_names=resolved_feature_names,
            column_roles=resolved_column_roles,
            feature_metadata=resolved_feature_metadata,
            categorical_schema=resolved_categorical_schema,
            releasable_feature_storage=_releasable_feature_storage,
            dense_data=None,
            sparse_components=(
                np.ascontiguousarray(sparse_data, dtype=np.float32),
                np.ascontiguousarray(sparse_indices, dtype=np.int64),
                np.ascontiguousarray(sparse_indptr, dtype=np.int64),
                (int(shape[0]), int(shape[1])),
            ),
        )

    def __init__(
        self,
        data: Any,
        label: Any,
        cat_features: Optional[List[int]] = None,
        weight: Any = None,
        group_id: Any = None,
        group_weight: Any = None,
        subgroup_id: Any = None,
        baseline: Any = None,
        pairs: Any = None,
        pairs_weight: Any = None,
        feature_names: Optional[List[str]] = None,
        column_roles: Any = None,
        feature_metadata: Optional[Mapping[str, Any]] = None,
        categorical_schema: Optional[Mapping[str, Any]] = None,
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
        resolved_feature_names = None if feature_names is None else [str(name) for name in feature_names]
        if _is_pandas_series(label):
            label = label.to_numpy(copy=False)
        resolved_group_id = _normalize_group_id(group_id)
        resolved_subgroup_id = _normalize_identifier_array(subgroup_id, name="subgroup_id")
        label_array = np.ascontiguousarray(label, dtype=np.float32)
        num_rows = sparse_components[3][0] if sparse_components is not None else int(np.asarray(data).shape[0])
        num_cols = sparse_components[3][1] if sparse_components is not None else int(np.asarray(data).shape[1])
        native_weight = np.ones(num_rows, dtype=np.float32) if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
        self.weight = None if weight is None else native_weight
        resolved_column_roles = _normalize_column_roles(column_roles, num_cols=num_cols, feature_names=resolved_feature_names)
        resolved_feature_metadata = _normalize_json_metadata(feature_metadata, name="feature_metadata")
        resolved_categorical_schema = _normalize_json_metadata(categorical_schema, name="categorical_schema")
        native_group_id = None if resolved_group_id is None else np.ascontiguousarray(resolved_group_id, dtype=np.int64)
        native_group_weight = _normalize_group_weight(group_weight, group_id=resolved_group_id, num_rows=num_rows)
        native_subgroup_id = None if resolved_subgroup_id is None else np.ascontiguousarray(resolved_subgroup_id, dtype=np.int64)
        native_baseline = _normalize_baseline(baseline, num_rows=num_rows)
        native_pairs, native_pairs_weight = _normalize_pairs(
            pairs,
            pairs_weight=pairs_weight,
            group_id=resolved_group_id,
            num_rows=num_rows,
        )
        if sparse_components is None:
            data_array = np.asfortranarray(data, dtype=np.float32)
            handle = _core.Pool(
                data_array,
                label_array,
                resolved_cat_features,
                native_weight,
                native_group_id,
                native_group_weight,
                native_subgroup_id,
                native_baseline,
                native_pairs,
                native_pairs_weight,
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
                native_group_weight,
                native_subgroup_id,
                native_baseline,
                native_pairs,
                native_pairs_weight,
            )
            data_array = None
        initialized = _pool_from_handle(
            type(self),
            handle,
            weight=self.weight,
            group_id=native_group_id,
            group_weight=native_group_weight,
            subgroup_id=native_subgroup_id,
            baseline=native_baseline,
            pairs=native_pairs,
            pairs_weight=native_pairs_weight,
            feature_names=resolved_feature_names,
            column_roles=resolved_column_roles,
            feature_metadata=resolved_feature_metadata,
            categorical_schema=resolved_categorical_schema,
            releasable_feature_storage=_releasable_feature_storage,
            dense_data=data_array,
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
