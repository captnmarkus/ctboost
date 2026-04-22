"""Serialization helpers for ctboost.sklearn."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

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
            destination = Path(path)
            fitted_state = {
                "booster_state": dict(self._booster._handle.export_state()),
                "booster_training_metadata": _serialize_value(getattr(self._booster, "_training_metadata", None)),
                "n_features_in_": int(self.n_features_in_),
                "best_iteration_": int(self.best_iteration_),
                "evals_result_": _serialize_value(self.evals_result_),
                "best_score_": _serialize_value(self.best_score_),
                "python_object": self,
            }
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
                return model
    
            if document["estimator_class"] != cls.__name__:
                raise TypeError(f"serialized model does not contain a {cls.__name__} instance")
    
            model = cls(**{key: _deserialize_value(value) for key, value in document["init_params"].items()})
            fitted_state = document["fitted_state"]
            model._booster = Booster(
                _core.GradientBooster.from_state(fitted_state["booster_state"]),
                training_metadata=_deserialize_value(fitted_state.get("booster_training_metadata")),
            )
            model.n_features_in_ = int(fitted_state["n_features_in_"])
            model.best_iteration_ = int(fitted_state["best_iteration_"])
            model.evals_result_ = _deserialize_value(fitted_state["evals_result_"])
            model.best_score_ = _deserialize_value(fitted_state["best_score_"])
            pipeline_state = fitted_state.get("feature_pipeline_state")
            model._feature_pipeline = None if pipeline_state is None else FeaturePipeline.from_state(pipeline_state)
            model._load_extra_serialized_state(fitted_state)
            return model
        @property
        def feature_importances_(self) -> np.ndarray:
            return np.asarray(self._booster.feature_importances_, dtype=np.float32)
        def predict_leaf_index(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
            pool = self._transform_prediction_pool(X)
            return self._booster.predict_leaf_index(pool, num_iteration=num_iteration)
        def predict_contrib(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
            pool = self._transform_prediction_pool(X)
            return self._booster.predict_contrib(pool, num_iteration=num_iteration)
