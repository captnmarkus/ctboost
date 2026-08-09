"""Prediction-pool mixin for ctboost.sklearn."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..core import Pool
from ..core.columnar import (
    _columnar_frame_metadata,
    _is_polars_lazy_frame,
)
from ..feature_pipeline import _feature_pipelines_equivalent


class _BasePredictionMixin:
        @staticmethod
        def _input_feature_metadata(X: Any) -> tuple[int, Any]:
            if isinstance(X, Pool):
                return int(X.num_cols), None if X.feature_names is None else list(X.feature_names)

            columnar_metadata = _columnar_frame_metadata(X)
            if columnar_metadata is not None:
                return int(columnar_metadata[1]), list(columnar_metadata[2])
            if _is_polars_lazy_frame(X):
                raise TypeError(
                    "Polars LazyFrame input is not eager; call collect() before passing it to CTBoost"
                )

            shape = getattr(X, "shape", None)
            if shape is None:
                shape = np.asarray(X, dtype=object).shape
            if len(shape) != 2:
                raise ValueError("X must be a 2D feature matrix")

            columns = getattr(X, "columns", None)
            feature_names = None if columns is None else [str(name) for name in columns]
            return int(shape[1]), feature_names
        def _validate_prediction_input(self, X: Any) -> None:
            check_is_fitted(self, attributes="_booster")
            if (
                isinstance(X, Pool)
                and self._feature_pipeline is not None
                and not _feature_pipelines_equivalent(
                    self._feature_pipeline,
                    getattr(X, "_feature_pipeline", None),
                )
            ):
                raise ValueError(
                    "Pool input must contain features transformed by this estimator's "
                    "fitted feature pipeline; pass the original raw array or DataFrame instead"
                )
            if isinstance(X, Pool) and self._feature_pipeline is not None:
                X._feature_pipeline = self._feature_pipeline
            feature_count, feature_names = self._input_feature_metadata(X)

            uses_prepared_pool = isinstance(X, Pool) and self._feature_pipeline is not None
            expected_count = (
                int(getattr(self, "n_transformed_features_", self.n_features_in_))
                if uses_prepared_pool
                else int(self.n_features_in_)
            )
            if feature_count != expected_count:
                feature_space = "transformed" if uses_prepared_pool else "raw"
                raise ValueError(
                    f"X has {feature_count} features, but {type(self).__name__} expects "
                    f"{expected_count} {feature_space} features"
                )

            if feature_names is None:
                return
            if uses_prepared_pool:
                expected_names = self._booster.feature_names
            else:
                stored_names = getattr(self, "feature_names_in_", None)
                expected_names = None if stored_names is None else [str(name) for name in stored_names]
            if expected_names is not None and list(feature_names) != list(expected_names):
                raise ValueError(
                    "X feature names and order do not match the feature names seen during fit"
                )
        @staticmethod
        def _prediction_pool(X: Any) -> Pool:
            if isinstance(X, Pool):
                return X
            return Pool(data=X)
        def _transform_prediction_pool(self, X: Any) -> Pool:
            self._validate_prediction_input(X)
            if isinstance(X, Pool):
                return X
            if self._feature_pipeline is None:
                return self._prediction_pool(X)
            transformed, cat_features, feature_names = self._feature_pipeline.transform_array(X)
            pool = Pool(
                data=transformed,
                cat_features=cat_features,
                feature_names=feature_names,
            )
            pool._feature_pipeline = self._feature_pipeline
            return pool
