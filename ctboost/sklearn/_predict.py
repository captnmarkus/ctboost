"""Prediction-pool mixin for ctboost.sklearn."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import Pool


class _BasePredictionMixin:
        @staticmethod
        def _prediction_pool(X: Any) -> Pool:
            if isinstance(X, Pool):
                return X
            num_rows = int(X.shape[0]) if hasattr(X, "shape") else len(X)
            return Pool(data=X, label=np.zeros(num_rows, dtype=np.float32))
        def _transform_prediction_pool(self, X: Any) -> Pool:
            if isinstance(X, Pool):
                return X
            if self._feature_pipeline is None:
                return self._prediction_pool(X)
            num_rows = int(X.shape[0]) if hasattr(X, "shape") else len(X)
            transformed, cat_features, feature_names = self._feature_pipeline.transform_array(X)
            return Pool(
                data=transformed,
                label=np.zeros(num_rows, dtype=np.float32),
                cat_features=cat_features,
                feature_names=feature_names,
            )
