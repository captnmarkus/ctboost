"""Booster wrapper for ctboost.training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from .. import _core
from .._export import (
    export_inference_manifest as _export_inference_manifest,
    export_model as _export_model,
    get_inference_manifest as _get_inference_manifest,
)
from .._serialization import load_booster_document, save_booster
from ..feature_pipeline import FeaturePipeline, _feature_pipelines_equivalent
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

    def __getstate__(self) -> Dict[str, Any]:
        """Return a pickle-safe representation of the native booster."""
        return {
            "handle_state": dict(self._handle.export_state()),
            "feature_pipeline_state": (
                None if self._feature_pipeline is None else self._feature_pipeline.to_state()
            ),
            "training_metadata": self._training_metadata,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._handle = _core.GradientBooster.from_state(dict(state["handle_state"]))
        pipeline_state = state.get("feature_pipeline_state")
        self._feature_pipeline = (
            None if pipeline_state is None else FeaturePipeline.from_state(pipeline_state)
        )
        training_metadata = state.get("training_metadata")
        self._training_metadata = (
            None if training_metadata is None else dict(training_metadata)
        )

    def _prediction_pool(self, data: Any) -> Pool:
        if isinstance(data, Pool):
            if (
                self._feature_pipeline is not None
                and not _feature_pipelines_equivalent(
                    self._feature_pipeline,
                    getattr(data, "_feature_pipeline", None),
                )
            ):
                raise ValueError(
                    "Pool input must contain features transformed by this booster's fitted "
                    "feature pipeline; pass the original raw array or DataFrame instead"
                )
            if self._feature_pipeline is not None:
                data._feature_pipeline = self._feature_pipeline
            return data
        if self._feature_pipeline is None:
            return _prediction_pool(data)
        transformed, cat_features, feature_names = self._feature_pipeline.transform_array(data)
        pool = Pool(
            data=transformed,
            cat_features=cat_features,
            feature_names=feature_names,
        )
        pool._feature_pipeline = self._feature_pipeline
        return pool

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

    def predict_raw(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        """Return the additive raw score before any objective inverse link."""
        return self._predict_raw(data, num_iteration=num_iteration)

    def predict(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        """Return raw scores, preserving CTBoost's existing prediction contract."""
        return self.predict_raw(data, num_iteration=num_iteration)

    def predict_mean(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        """Return the response-scale mean for a supported log-link objective."""
        objective = self.native_objective_name.lower()
        if objective not in {
            "gamma",
            "gammaloss",
            "reg:gamma",
            "poisson",
            "poissonregression",
            "tweedie",
            "tweedieloss",
            "reg:tweedie",
        }:
            raise ValueError(
                "predict_mean is available for Gamma, Poisson, and Tweedie "
                "log-link objectives"
            )
        raw_prediction = self.predict_raw(
            data,
            num_iteration=num_iteration,
        ).astype(np.float64, copy=False)
        return np.exp(np.clip(raw_prediction, -745.0, 709.0))

    def predict_response(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        """Alias for :meth:`predict_mean` on log-link mean objectives."""
        return self.predict_mean(data, num_iteration=num_iteration)

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

    def predict_shap(
        self,
        data: Any,
        background: Any,
        *,
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """Return exact interventional TreeSHAP values for raw model output.

        ``background`` defines the empirical reference distribution.  Its
        ``Pool.weight`` values are honored when present; otherwise rows are
        weighted uniformly.  The final column contains the expected model
        output over that distribution, so every row sums to ``predict(data)``.

        This method is exact and distinct from :meth:`predict_contrib`, which
        is a faster path-based additive decomposition.
        """
        from ..explain import explain_booster

        return explain_booster(
            self,
            data,
            background,
            num_iteration=num_iteration,
            interaction_values=False,
        )

    def predict_shap_values(
        self,
        data: Any,
        background: Any,
        *,
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """Alias for :meth:`predict_shap`."""
        return self.predict_shap(data, background, num_iteration=num_iteration)

    def predict_shap_interactions(
        self,
        data: Any,
        background: Any,
        *,
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """Return exact pairwise interventional SHAP interaction values.

        For a single-output model the shape is
        ``(n_rows, n_features + 1, n_features + 1)``.  Multiclass output adds
        an output dimension after ``n_rows``.  The final row and column are
        reserved for bias and the expected value is stored at ``[-1, -1]``.
        Each feature row sums to that feature's SHAP value and the entire
        matrix sums to the raw model prediction.
        """
        from ..explain import explain_booster

        return explain_booster(
            self,
            data,
            background,
            num_iteration=num_iteration,
            interaction_values=True,
        )

    def predict_shap_interaction_values(
        self,
        data: Any,
        background: Any,
        *,
        num_iteration: Optional[int] = None,
    ) -> np.ndarray:
        """Alias for :meth:`predict_shap_interactions`."""
        return self.predict_shap_interactions(
            data, background, num_iteration=num_iteration
        )

    def calc_leaf_influence(
        self,
        data: Any,
        reference_data: Any,
        *,
        num_iteration: Optional[int] = None,
        return_coverage: bool = False,
    ) -> Any:
        """Return signed shared-leaf object attribution scores.

        This is a leaf co-membership approximation, not exact leave-one-out
        influence and not a model refit. See :func:`ctboost.explain.calc_leaf_influence`.
        """
        from ..explain import calc_leaf_influence

        return calc_leaf_influence(
            self,
            data,
            reference_data,
            num_iteration=num_iteration,
            return_coverage=return_coverage,
        )

    def get_object_importance(
        self,
        data: Any,
        reference_data: Any,
        *,
        top_size: int = -1,
        importance_type: str = "Average",
        prediction_dimension: Optional[int] = None,
        num_iteration: Optional[int] = None,
    ) -> Any:
        """Rank reference rows by approximate shared-leaf influence."""
        from ..explain import get_object_importance

        return get_object_importance(
            self,
            data,
            reference_data,
            top_size=top_size,
            importance_type=importance_type,
            prediction_dimension=prediction_dimension,
            num_iteration=num_iteration,
        )

    def tree_to_dot(
        self,
        tree_index: int = 0,
        *,
        rankdir: str = "TB",
        precision: int = 6,
    ) -> str:
        """Return one fitted tree as dependency-free Graphviz DOT source."""
        from ..explain import tree_to_dot

        return tree_to_dot(self, tree_index, rankdir=rankdir, precision=precision)

    def plot_tree(
        self,
        tree_index: int = 0,
        *,
        ax: Any = None,
        figsize: Any = (12.0, 7.0),
        precision: int = 4,
    ) -> Any:
        """Plot one fitted tree with matplotlib and return its axes."""
        from ..explain import plot_tree

        return plot_tree(
            self,
            tree_index,
            ax=ax,
            figsize=figsize,
            precision=precision,
        )

    def calc_feature_statistics(
        self,
        data: Any,
        target: Any = None,
        *,
        feature: Any = None,
        prediction_dimension: Optional[int] = None,
        plot: bool = False,
        axes: Any = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate predictions and optional targets by fitted feature bin."""
        from ..explain import calc_feature_statistics, plot_feature_statistics

        result = calc_feature_statistics(
            self,
            data,
            target,
            feature=feature,
            prediction_dimension=prediction_dimension,
        )
        if plot:
            plot_feature_statistics(result, axes=axes)
        return result

    def plot_feature_statistics(
        self,
        data: Any,
        target: Any = None,
        *,
        feature: Any = None,
        prediction_dimension: Optional[int] = None,
        axes: Any = None,
        figsize: Any = None,
        show_object_count: bool = True,
    ) -> Any:
        """Calculate and plot fitted-bin feature statistics, returning axes."""
        from ..explain import calc_feature_statistics, plot_feature_statistics

        statistics = calc_feature_statistics(
            self,
            data,
            target,
            feature=feature,
            prediction_dimension=prediction_dimension,
        )
        return plot_feature_statistics(
            statistics,
            axes=axes,
            figsize=figsize,
            show_object_count=show_object_count,
        )

    def plot_predictions(
        self,
        data: Any,
        target: Any = None,
        *,
        kind: str = "auto",
        prediction_dimension: Optional[int] = None,
        num_iteration: Optional[int] = None,
        ax: Any = None,
        figsize: Any = (7.0, 5.0),
    ) -> Any:
        """Plot raw predictions or numeric-target residual diagnostics."""
        from ..explain import plot_predictions

        return plot_predictions(
            self,
            data,
            target,
            kind=kind,
            prediction_dimension=prediction_dimension,
            num_iteration=num_iteration,
            ax=ax,
            figsize=figsize,
        )

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
        class_labels: Optional[Sequence[Any]] = None,
        estimator_name: Optional[str] = None,
    ) -> None:
        _export_model(
            path,
            self._handle,
            export_format=export_format,
            feature_pipeline=self._feature_pipeline,
            prepared_features=prepared_features,
            data_schema=self.data_schema,
            class_labels=class_labels,
            estimator_name=estimator_name,
        )

    def get_inference_manifest(
        self,
        *,
        prepared_features: bool = False,
        class_labels: Optional[Sequence[Any]] = None,
        estimator_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the model's versioned input/output deployment contract."""
        return _get_inference_manifest(
            self._handle,
            feature_pipeline=self._feature_pipeline,
            prepared_features=prepared_features,
            data_schema=self.data_schema,
            class_labels=class_labels,
            estimator_name=estimator_name,
        )

    def export_inference_manifest(
        self,
        path: PathLike,
        *,
        prepared_features: bool = False,
        class_labels: Optional[Sequence[Any]] = None,
        estimator_name: Optional[str] = None,
    ) -> None:
        """Write the model's versioned input/output deployment contract as JSON."""
        _export_inference_manifest(
            path,
            self._handle,
            feature_pipeline=self._feature_pipeline,
            prepared_features=prepared_features,
            data_schema=self.data_schema,
            class_labels=class_labels,
            estimator_name=estimator_name,
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
        if self._training_metadata is not None and "objective_name" in self._training_metadata:
            return str(self._training_metadata["objective_name"])
        return str(self._handle.objective_name())

    @property
    def native_objective_name(self) -> str:
        """Return the native objective used for model shape and inference semantics."""
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
    def boost_from_average(self) -> bool:
        """Whether fresh built-in training estimates an objective-specific intercept.

        A configured ``base_score`` takes precedence. Automatic initialization is
        deliberately skipped for ranking, survival, custom objectives, and pools
        carrying per-row baselines.
        """
        return bool(self._handle.boost_from_average())

    @property
    def configured_base_score(self) -> List[float]:
        """Return the optional user-configured raw-margin intercept."""
        return [float(value) for value in self._handle.configured_base_score()]

    @property
    def base_score(self) -> np.ndarray:
        """Return the fitted raw-margin intercept, one value per output."""
        return np.asarray(self._handle.base_score(), dtype=np.float64)

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

