"""Booster wrapper for ctboost.training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np

from .. import _core
from .._export import export_model as _export_model
from .._serialization import load_booster_document, save_booster
from ..feature_pipeline import FeaturePipeline
from ..core import Pool
from ._booster_schema import _borders_from_quantization_schema
from ._pool_build import _prediction_pool, _resolve_num_iteration
from .eval_metrics import _copy_evals_result
from .schema import _baseline_matrix_for_prediction

PathLike = Union[str, Path]

class Booster:
    """Small Python wrapper around the native gradient booster."""

    def __init__(
        self,
        handle: Any,
        *,
        feature_pipeline: Optional[FeaturePipeline] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._handle = handle
        self._feature_pipeline = feature_pipeline
        self._training_metadata = None if training_metadata is None else dict(training_metadata)

    def _set_training_metadata(
        self,
        *,
        evals_result: Optional[Mapping[str, Mapping[str, Iterable[float]]]] = None,
        eval_loss_history: Optional[Iterable[float]] = None,
        best_iteration: Optional[int] = None,
        best_score: Optional[float] = None,
        eval_metric_name: Optional[str] = None,
        data_schema: Optional[Mapping[str, Any]] = None,
        learning_rate_history: Optional[Iterable[float]] = None,
    ) -> None:
        if (
            evals_result is None
            and eval_loss_history is None
            and best_iteration is None
            and best_score is None
            and eval_metric_name is None
            and data_schema is None
            and learning_rate_history is None
        ):
            self._training_metadata = None
            return
        metadata = {} if self._training_metadata is None else dict(self._training_metadata)
        if evals_result is not None:
            metadata["evals_result"] = _copy_evals_result(evals_result)
        if eval_loss_history is not None:
            metadata["eval_loss_history"] = [float(value) for value in eval_loss_history]
        if best_iteration is not None:
            metadata["best_iteration"] = int(best_iteration)
        if best_score is not None:
            metadata["best_score"] = float(best_score)
        if eval_metric_name is not None:
            metadata["eval_metric_name"] = str(eval_metric_name)
        if data_schema is not None:
            metadata["data_schema"] = dict(data_schema)
        if learning_rate_history is not None:
            metadata["learning_rate_history"] = [float(value) for value in learning_rate_history]
        self._training_metadata = metadata

    def _prediction_pool(self, data: Any) -> Pool:
        if isinstance(data, Pool):
            return data
        if self._feature_pipeline is None:
            return _prediction_pool(data)
        num_rows = int(data.shape[0]) if hasattr(data, "shape") else len(data)
        transformed, cat_features, feature_names = self._feature_pipeline.transform_array(data)
        return Pool(
            data=transformed,
            label=np.zeros(num_rows, dtype=np.float32),
            cat_features=cat_features,
            feature_names=feature_names,
        )

    def _predict_raw(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._prediction_pool(data)
        raw = np.asarray(
            self._handle.predict(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.float32,
        )
        prediction_dimension = int(self._handle.prediction_dimension())
        if prediction_dimension > 1:
            return raw.reshape((pool.num_rows, prediction_dimension))
        return raw

    def predict(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        return self._predict_raw(data, num_iteration=num_iteration)

    def staged_predict(self, data: Any) -> Iterable[np.ndarray]:
        pool = self._prediction_pool(data)
        for iteration in range(1, self.num_iterations_trained + 1):
            yield self._predict_raw(pool, num_iteration=iteration)

    def predict_leaf_index(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._prediction_pool(data)
        values = np.asarray(
            self._handle.predict_leaf_indices(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.int32,
        )
        tree_count = 0 if pool.num_rows == 0 else values.size // pool.num_rows
        return values.reshape((pool.num_rows, tree_count))

    def predict_contrib(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._prediction_pool(data)
        values = np.asarray(
            self._handle.predict_contributions(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.float32,
        )
        width = pool.num_cols + 1
        if self.prediction_dimension > 1:
            contributions = values.reshape((pool.num_rows, self.prediction_dimension, width))
            baseline = _baseline_matrix_for_prediction(pool, self.prediction_dimension)
            if baseline is not None:
                contributions[:, :, -1] += baseline
            return contributions
        contributions = values.reshape((pool.num_rows, width))
        baseline = _baseline_matrix_for_prediction(pool, 1)
        if baseline is not None:
            contributions[:, -1] += baseline[:, 0]
        return contributions

    def save_model(self, path: PathLike, *, model_format: Optional[str] = None) -> None:
        destination = Path(path)
        save_booster(
            destination,
            self._handle,
            model_format=model_format,
            feature_pipeline_state=None
            if self._feature_pipeline is None
            else self._feature_pipeline.to_state(),
            training_state=self._training_metadata,
        )

    def export_model(
        self,
        path: PathLike,
        *,
        export_format: Optional[str] = None,
        prepared_features: bool = False,
    ) -> None:
        _export_model(
            path,
            self._handle,
            export_format=export_format,
            feature_pipeline=self._feature_pipeline,
            prepared_features=prepared_features,
        )

    @classmethod
    def load_model(cls, path: PathLike) -> "Booster":
        document = load_booster_document(Path(path))
        pipeline_state = document.get("feature_pipeline_state")
        return cls(
            _core.GradientBooster.from_state(document["booster_state"]),
            feature_pipeline=None if pipeline_state is None else FeaturePipeline.from_state(pipeline_state),
            training_metadata=document.get("training_state"),
        )

    def get_quantization_schema(self) -> Optional[Dict[str, Any]]:
        state = self._handle.quantization_schema_state()
        if state is None:
            return None
        return dict(state)

    def get_borders(self) -> Optional[Dict[str, Any]]:
        return _borders_from_quantization_schema(self.get_quantization_schema())

    def set_learning_rate(self, learning_rate: float) -> None:
        resolved_learning_rate = float(learning_rate)
        if resolved_learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if hasattr(self._handle, "set_learning_rate"):
            self._handle.set_learning_rate(resolved_learning_rate)
            return
        current_learning_rate = float(self._handle.learning_rate())
        if np.isclose(current_learning_rate, resolved_learning_rate, rtol=1e-12, atol=1e-12):
            return
        state = dict(self._handle.export_state())
        scale = current_learning_rate / resolved_learning_rate
        for tree_state in state.get("trees", []):
            for node in tree_state.get("nodes", []):
                if bool(node.get("is_leaf", False)):
                    node["leaf_weight"] = float(node["leaf_weight"]) * scale
        state["learning_rate"] = resolved_learning_rate
        self._handle = _core.GradientBooster.from_state(state)

    @property
    def loss_history(self) -> List[float]:
        return list(self._handle.loss_history())

    @property
    def eval_loss_history(self) -> List[float]:
        if self._training_metadata is not None and "eval_loss_history" in self._training_metadata:
            return list(self._training_metadata["eval_loss_history"])
        return list(self._handle.eval_loss_history())

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._handle.feature_importances(), dtype=np.float32)

    @property
    def evals_result_(self) -> Dict[str, Dict[str, List[float]]]:
        if self._training_metadata is not None and "evals_result" in self._training_metadata:
            return _copy_evals_result(self._training_metadata["evals_result"])
        result = {"learn": {"loss": self.loss_history}}
        if self.eval_loss_history:
            metric_name = "loss" if self.eval_metric_name.lower() == self.objective_name.lower() else self.eval_metric_name
            result["validation"] = {metric_name: self.eval_loss_history}
        return result

    @property
    def num_classes(self) -> int:
        return int(self._handle.num_classes())

    @property
    def prediction_dimension(self) -> int:
        return int(self._handle.prediction_dimension())

    @property
    def num_iterations_trained(self) -> int:
        return int(self._handle.num_iterations_trained())

    @property
    def best_iteration(self) -> int:
        if self._training_metadata is not None and "best_iteration" in self._training_metadata:
            return int(self._training_metadata["best_iteration"])
        return int(self._handle.best_iteration())

    @property
    def objective_name(self) -> str:
        return str(self._handle.objective_name())

    @property
    def eval_metric_name(self) -> str:
        if self._training_metadata is not None and "eval_metric_name" in self._training_metadata:
            return str(self._training_metadata["eval_metric_name"])
        return str(self._handle.eval_metric_name())

    @property
    def data_schema(self) -> Dict[str, Any]:
        if self._training_metadata is None or "data_schema" not in self._training_metadata:
            return {}
        return dict(self._training_metadata["data_schema"])

    @property
    def learning_rate(self) -> float:
        return float(self._handle.learning_rate())

    @property
    def learning_rate_history(self) -> List[float]:
        if self._training_metadata is not None and "learning_rate_history" in self._training_metadata:
            return [float(value) for value in self._training_metadata["learning_rate_history"]]
        if hasattr(self._handle, "tree_learning_rates"):
            history = list(self._handle.tree_learning_rates())
            if history:
                return [float(value) for value in history]
        return [self.learning_rate] * self.num_iterations_trained

    @property
    def feature_names(self) -> Optional[List[str]]:
        schema = self.data_schema
        feature_names = schema.get("feature_names")
        return None if feature_names is None else list(feature_names)

def load_model(path: PathLike) -> Booster:
    return Booster.load_model(path)

