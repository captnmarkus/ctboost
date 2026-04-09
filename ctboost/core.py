"""Python wrapper types for the native CTBoost core."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core


class Pool:
    """Python wrapper around the native `ctboost::Pool`."""

    def __init__(
        self,
        data: Any,
        label: Any,
        cat_features: list[int] | None = None,
        weight: Any = None,
        feature_names: list[str] | None = None,
    ) -> None:
        if cat_features is None:
            cat_features = []

        data_array = np.ascontiguousarray(data, dtype=np.float32)
        label_array = np.ascontiguousarray(label, dtype=np.float32)

        self._handle = _core.Pool(data_array, label_array, list(cat_features))
        self.num_rows = self._handle.num_rows()
        self.num_cols = self._handle.num_cols()
        self.cat_features = list(self._handle.cat_features())
        self.weight = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
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
