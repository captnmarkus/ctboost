"""Serialization helpers for ctboost.sklearn."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .. import _core
from .._serialization import load_estimator_document, save_estimator
from ..feature_pipeline import FeaturePipeline
from ..training import Booster

PathLike = Union[str, Path]

def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {
            "__ctboost_type__": "dict",
            "items": [[_serialize_value(key), _serialize_value(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {
            "__ctboost_type__": "tuple",
            "items": [_serialize_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value

def _deserialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    marker = value.get("__ctboost_type__")
    if marker == "dict":
        return {
            _deserialize_value(key): _deserialize_value(item)
            for key, item in value["items"]
        }
    if marker == "tuple":
        return tuple(_deserialize_value(item) for item in value["items"])
    return {key: _deserialize_value(item) for key, item in value.items()}


class _SerializationMixin:
        def _extra_serialized_state(self) -> Dict[str, Any]:
            return {}
        def _load_extra_serialized_state(self, fitted_state: Dict[str, Any]) -> None:
            return None
        def save_model(self, path: PathLike, *, model_format: Optional[str] = None) -> None:
            check_is_fitted(self, attributes="_booster")
            destination = Path(path)
            fitted_state = {
                "booster_state": dict(self._booster._handle.export_state()),
                "booster_training_metadata": _serialize_value(getattr(self._booster, "_training_metadata", None)),
                "n_features_in_": int(self.n_features_in_),
                "n_transformed_features_": int(
                    getattr(self, "n_transformed_features_", self.n_features_in_)
                ),
                "best_iteration_": int(self.best_iteration_),
                "evals_result_": _serialize_value(self.evals_result_),
                "best_score_": _serialize_value(self.best_score_),
                "python_object": self,
            }
            if hasattr(self, "feature_names_in_"):
                fitted_state["feature_names_in_"] = _serialize_value(self.feature_names_in_)
            if self._feature_pipeline is not None:
                fitted_state["feature_pipeline_state"] = self._feature_pipeline.to_state()
            fitted_state.update(self._extra_serialized_state())
            save_estimator(
                destination,
                estimator_class=type(self).__name__,
                init_params={key: _serialize_value(value) for key, value in self.get_params(deep=False).items()},
                fitted_state=fitted_state,
                model_format=model_format,
            )
        def export_model(
            self,
            path: PathLike,
            *,
            export_format: Optional[str] = None,
            prepared_features: bool = False,
        ) -> None:
            self._booster.export_model(
                path,
                export_format=export_format,
                prepared_features=prepared_features,
            )
        @classmethod
        def load_model(cls, path: PathLike) -> "_BaseCTBoost":
            destination = Path(path)
            document = load_estimator_document(destination)
            if document is None:
                import pickle
    
                with destination.open("rb") as stream:
                    model = pickle.load(stream)
                if not isinstance(model, cls):
                    raise TypeError(f"serialized model does not contain a {cls.__name__} instance")
                model._ensure_input_metadata_compatibility()
                return model
    
            if document["estimator_class"] != cls.__name__:
                raise TypeError(f"serialized model does not contain a {cls.__name__} instance")
    
            model = cls(**{key: _deserialize_value(value) for key, value in document["init_params"].items()})
            fitted_state = document["fitted_state"]
            model._booster = Booster(
                _core.GradientBooster.from_state(fitted_state["booster_state"]),
                training_metadata=_deserialize_value(fitted_state.get("booster_training_metadata")),
            )
            stored_feature_count = int(fitted_state["n_features_in_"])
            model.best_iteration_ = int(fitted_state["best_iteration_"])
            model.evals_result_ = _deserialize_value(fitted_state["evals_result_"])
            model.best_score_ = _deserialize_value(fitted_state["best_score_"])
            pipeline_state = fitted_state.get("feature_pipeline_state")
            model._feature_pipeline = None if pipeline_state is None else FeaturePipeline.from_state(pipeline_state)
            if "n_transformed_features_" in fitted_state:
                model.n_features_in_ = stored_feature_count
                model.n_transformed_features_ = int(fitted_state["n_transformed_features_"])
            else:
                model.n_transformed_features_ = stored_feature_count
                pipeline_feature_count = (
                    None if model._feature_pipeline is None else model._feature_pipeline.n_features_in_
                )
                model.n_features_in_ = (
                    stored_feature_count
                    if pipeline_feature_count is None
                    else int(pipeline_feature_count)
                )
            if "feature_names_in_" in fitted_state:
                model.feature_names_in_ = np.asarray(
                    _deserialize_value(fitted_state["feature_names_in_"]),
                    dtype=object,
                )
            model._load_extra_serialized_state(fitted_state)
            model._ensure_input_metadata_compatibility()
            return model
        def _ensure_input_metadata_compatibility(self) -> None:
            self._synchronize_compatibility_aliases()
            if not hasattr(self, "_feature_pipeline"):
                self._feature_pipeline = None
            if self._feature_pipeline is not None:
                self._booster._feature_pipeline = self._feature_pipeline
            if not hasattr(self, "n_transformed_features_"):
                legacy_feature_count = int(self.n_features_in_)
                self.n_transformed_features_ = legacy_feature_count
                pipeline_feature_count = (
                    None if self._feature_pipeline is None else self._feature_pipeline.n_features_in_
                )
                if pipeline_feature_count is not None:
                    self.n_features_in_ = int(pipeline_feature_count)
            if hasattr(self, "feature_names_in_"):
                self.feature_names_in_ = np.asarray(self.feature_names_in_, dtype=object)
            else:
                feature_names = (
                    None if self._feature_pipeline is None else self._feature_pipeline.feature_names_in_
                )
                if feature_names is None:
                    feature_names = self._booster.feature_names
                if feature_names is not None and len(feature_names) == int(self.n_features_in_):
                    self.feature_names_in_ = np.asarray(feature_names, dtype=object)

            best_score = getattr(self, "best_score_", None)
            if not isinstance(best_score, dict) or any(
                not isinstance(dataset_scores, dict)
                for dataset_scores in best_score.values()
            ):
                self.best_score_ = self._compute_best_score()
        def is_fitted(self) -> bool:
            return hasattr(self, "_booster")
        def get_booster(self) -> Booster:
            check_is_fitted(self, attributes="_booster")
            return self._booster
        def get_best_iteration(self) -> int:
            check_is_fitted(self, attributes="_booster")
            return int(self.best_iteration_)
        def get_best_score(self) -> Dict[str, Any]:
            check_is_fitted(self, attributes="_booster")
            return deepcopy(self.best_score_)
        def get_evals_result(self) -> Dict[str, Any]:
            check_is_fitted(self, attributes="_booster")
            return deepcopy(self.evals_result_)
        def evals_result(self) -> Dict[str, Any]:
            return self.get_evals_result()
        @property
        def feature_importances_(self) -> np.ndarray:
            return np.asarray(self._booster.feature_importances_, dtype=np.float32)
        def predict_leaf_index(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
            pool = self._transform_prediction_pool(X)
            return self._booster.predict_leaf_index(pool, num_iteration=num_iteration)
        def apply(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
            return self.predict_leaf_index(X, num_iteration=num_iteration)
        def calc_leaf_indexes(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
            return self.predict_leaf_index(X, num_iteration=num_iteration)
        def predict_contrib(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
            pool = self._transform_prediction_pool(X)
            return self._booster.predict_contrib(pool, num_iteration=num_iteration)
