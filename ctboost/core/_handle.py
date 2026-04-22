"""Pool handle construction helpers."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .normalization import _normalize_json_metadata


def _pool_from_handle(
    cls: type,
    handle: Any,
    *,
    weight: Optional[np.ndarray],
    group_id: Optional[np.ndarray],
    group_weight: Optional[np.ndarray],
    subgroup_id: Optional[np.ndarray],
    baseline: Optional[np.ndarray],
    pairs: Optional[np.ndarray],
    pairs_weight: Optional[np.ndarray],
    feature_names: Optional[List[str]],
    column_roles: Optional[List[Optional[str]]],
    feature_metadata: Any,
    categorical_schema: Any,
    releasable_feature_storage: bool,
    dense_data: Optional[np.ndarray],
    sparse_components: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]],
) -> Any:
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
    self.group_weight = None if group_weight is None else np.asarray(group_weight, dtype=np.float32).copy()
    self.subgroup_id = None if subgroup_id is None else np.asarray(subgroup_id, dtype=np.int64).copy()
    self.baseline = None if baseline is None else np.asarray(baseline, dtype=np.float32).copy()
    self.pairs = None if pairs is None else np.asarray(pairs, dtype=np.int64).copy()
    self.pairs_weight = None if pairs_weight is None else np.asarray(pairs_weight, dtype=np.float32).copy()
    self.column_roles = None if column_roles is None else list(column_roles)
    self.feature_metadata = _normalize_json_metadata(feature_metadata, name="feature_metadata")
    self.categorical_schema = _normalize_json_metadata(categorical_schema, name="categorical_schema")
    self._releasable_feature_storage = bool(releasable_feature_storage)
    self._dense_data_ref = dense_data
    self._sparse_csc_components = sparse_components
    self._feature_pipeline = None
    self._external_memory_backing = None
    if self.weight is not None and (self.weight.ndim != 1 or self.weight.shape[0] != self.num_rows):
        raise ValueError("weight size must match the number of data rows")
    if self.group_id is not None and self.group_id.shape[0] != self.num_rows:
        raise ValueError("group_id size must match the number of data rows")
    if self.group_weight is not None and self.group_weight.shape[0] != self.num_rows:
        raise ValueError("group_weight size must match the number of data rows")
    if self.subgroup_id is not None and self.subgroup_id.shape[0] != self.num_rows:
        raise ValueError("subgroup_id size must match the number of data rows")
    if self.baseline is not None:
        if self.baseline.ndim == 1 and self.baseline.shape[0] != self.num_rows:
            raise ValueError("baseline size must match the number of data rows")
        if self.baseline.ndim == 2 and self.baseline.shape[0] != self.num_rows:
            raise ValueError("baseline row count must match the number of data rows")
        if self.baseline.ndim not in {1, 2}:
            raise ValueError("baseline must be a 1D or 2D NumPy array")
    if self.pairs is not None:
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("pairs must be a 2D NumPy array with shape (n_pairs, 2)")
        if self.pairs_weight is not None and self.pairs_weight.shape[0] != self.pairs.shape[0]:
            raise ValueError("pairs_weight size must match the number of pairs")
    if self.feature_names is not None and len(self.feature_names) != self.num_cols:
        raise ValueError("feature_names size must match the number of data columns")
    if self.column_roles is not None and len(self.column_roles) != self.num_cols:
        raise ValueError("column_roles size must match the number of data columns")
    return self
