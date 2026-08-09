"""Shared utilities for independent-tree multi-target estimators."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from ..core import Pool
from ..export_runtime import load_exported_predictor
from ..training import _normalize_eval_sets
from .serialization import _inference_safe_estimator_state


def _row_count(data: Any) -> int:
    if isinstance(data, Pool):
        return int(data.num_rows)
    shape = getattr(data, "shape", None)
    if shape is not None:
        try:
            if len(shape) == 2:
                return int(shape[0])
        except (TypeError, ValueError):
            pass
    try:
        return int(len(data))
    except TypeError as exc:
        raise TypeError("X must be a row-oriented array, DataFrame, sparse matrix, or Pool") from exc


def _target_matrix(y: Any, *, name: str, numeric: bool) -> np.ndarray:
    raw_values = np.asarray(y)
    if np.iscomplexobj(raw_values):
        raise ValueError(f"Complex data not supported in {name}")
    values = np.asarray(raw_values, dtype=np.float32 if numeric else None)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (n_rows, n_outputs)")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one output column")
    if numeric and not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_sample_weight(
    sample_weight: Any,
    *,
    n_rows: int,
    n_outputs: int,
) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    values = np.asarray(sample_weight, dtype=np.float32)
    if values.ndim == 1:
        if values.shape[0] != n_rows:
            raise ValueError("sample_weight size must match the number of rows")
    elif values.ndim == 2:
        if values.shape != (n_rows, n_outputs):
            raise ValueError(
                "2D sample_weight must have shape (n_rows, n_outputs)"
            )
    else:
        raise ValueError("sample_weight must be a 1D or 2D array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("sample_weight entries must be finite and non-negative")
    if values.ndim == 1 and float(values.sum()) <= 0.0:
        raise ValueError("sample_weight must have positive total weight")
    if values.ndim == 2 and np.any(values.sum(axis=0) <= 0.0):
        raise ValueError("sample_weight must have positive total weight for every output")
    return values


def _sample_weight_for_output(
    sample_weight: Optional[np.ndarray],
    output_index: int,
) -> Optional[np.ndarray]:
    if sample_weight is None or sample_weight.ndim == 1:
        return sample_weight
    return sample_weight[:, output_index]


def _slice_eval_sets(
    eval_set: Any,
    *,
    output_index: int,
    n_outputs: int,
    numeric: bool,
) -> Any:
    entries = _normalize_eval_sets(eval_set)
    if not entries:
        return None
    sliced: List[Any] = []
    for eval_index, entry in enumerate(entries):
        if isinstance(entry, Pool):
            raise TypeError(
                "multi-target eval_set entries must be (X, y) tuples because Pool labels are 1D"
            )
        X_eval, y_eval, *metadata = entry
        y_matrix = _target_matrix(
            y_eval,
            name=f"eval_set[{eval_index}] labels",
            numeric=numeric,
        )
        if y_matrix.shape[1] != n_outputs:
            raise ValueError(
                f"eval_set[{eval_index}] must contain {n_outputs} output columns"
            )
        if y_matrix.shape[0] != _row_count(X_eval):
            raise ValueError(
                f"eval_set[{eval_index}] labels must match its number of rows"
            )
        sliced.append((X_eval, y_matrix[:, output_index], *metadata))
    return sliced[0] if len(sliced) == 1 else sliced


def _target_snapshot_path(path: Any, output_index: int) -> Any:
    if path is None:
        return None
    source = Path(path)
    suffix = source.suffix or ".ctboost"
    return source.with_name(f"{source.stem}.output_{output_index}{suffix}")


def _validate_execution(estimator: Any, n_jobs: Optional[int], *, X: Any) -> int:
    resolved_n_jobs = 1 if n_jobs is None else int(n_jobs)
    if resolved_n_jobs == 0:
        raise ValueError("n_jobs must be non-zero")
    params = estimator.get_params(deep=False)
    if int(params.get("distributed_world_size", 1)) != 1:
        raise ValueError(
            "independent multi-target estimators do not orchestrate distributed child fits; "
            "distributed_world_size must be 1"
        )
    if str(params.get("task_type", "CPU")).upper() == "GPU" and resolved_n_jobs != 1:
        raise ValueError("GPU multi-target training requires n_jobs=1 to avoid device contention")
    if isinstance(X, Pool) and resolved_n_jobs != 1:
        raise ValueError("Pool input requires n_jobs=1 because native Pool handles are process-local")
    return resolved_n_jobs


def _validate_process_payload(
    estimator: Any,
    fit_params: Mapping[str, Any],
    init_model: Any,
    *,
    n_jobs: int,
) -> None:
    """Fail early when process-parallel child configuration cannot be serialized."""
    if n_jobs == 1:
        return
    try:
        from joblib.externals import cloudpickle

        cloudpickle.dumps((estimator, dict(fit_params), init_model))
    except Exception as exc:
        raise ValueError(
            "process-parallel multi-target training requires serializable estimator "
            "parameters and fit arguments; set n_jobs=1 for a non-picklable custom "
            "loss, eval_metric, learning-rate schedule, or callback"
        ) from exc


def _fit_child(
    estimator: Any,
    X: Any,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    eval_set: Any,
    fit_params: Mapping[str, Any],
    *,
    allow_shallow_parameter_clone: bool = False,
) -> Any:
    try:
        child = clone(estimator)
    except Exception:
        if not allow_shallow_parameter_clone:
            raise
        # sklearn.clone deep-copies every constructor parameter.  That is
        # unnecessary for a sequential child fit and rejects otherwise valid
        # Python objectives whose state intentionally cannot be pickled.  A
        # fresh CTBoost estimator built from the same parameter references is
        # still unfitted and preserves the callable identity without sharing
        # any learned booster state.
        child = type(estimator)(**estimator.get_params(deep=False))
    params = dict(fit_params)
    if sample_weight is not None:
        params["sample_weight"] = sample_weight
    if eval_set is not None:
        params["eval_set"] = eval_set
    return child.fit(X, y, **params)


class _IndependentPersistenceMixin:
    """Pickle persistence plus a JSON-predictor directory bundle export."""

    _bundle_kind = "independent"

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        template = state.get("estimator")
        fitted_children = state.get("estimators_")
        if template is not None and fitted_children:
            # The fitted children already omit Python objective/metric code in
            # their own pickle state.  Sanitize the unfitted constructor
            # template too; otherwise a lambda or deliberately non-picklable
            # callable retained in ``self.estimator`` makes the entire wrapper
            # impossible to persist for inference.
            safe_params = _inference_safe_estimator_state(
                fitted_children[0],
                dict(template.get_params(deep=False)),
            )
            state["estimator"] = type(template)(**safe_params)
        return state

    def save_model(self, path: Any) -> None:
        check_is_fitted(self, attributes="estimators_")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, path: Any) -> Any:
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"serialized model does not contain a {cls.__name__} instance")
        return model

    def _bundle_metadata(self) -> Dict[str, Any]:
        return {}

    def get_inference_manifest(self, *, prepared_features: bool = False) -> Dict[str, Any]:
        check_is_fitted(self, attributes="estimators_")
        return {
            "schema_version": 1,
            "artifact_type": "ctboost.independent_model_bundle",
            "estimator": type(self).__name__,
            "kind": self._bundle_kind,
            "tree_semantics": "one independent CTBoost booster per output",
            "n_outputs": len(self.estimators_),
            "prepared_features": bool(prepared_features),
            "models": [f"model_{index}.json" for index in range(len(self.estimators_))],
            **self._bundle_metadata(),
        }

    def export_inference_manifest(
        self,
        path: Any,
        *,
        prepared_features: bool = False,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.get_inference_manifest(prepared_features=prepared_features),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def export_model(
        self,
        path: Any,
        *,
        export_format: str = "json_predictor",
        prepared_features: bool = False,
    ) -> None:
        if str(export_format).lower() not in {"json", "json_predictor"}:
            raise ValueError(
                "independent model bundles currently support export_format='json_predictor'"
            )
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = self.get_inference_manifest(prepared_features=prepared_features)
        for index, estimator in enumerate(self.estimators_):
            estimator.export_model(
                destination / manifest["models"][index],
                export_format="json_predictor",
                prepared_features=prepared_features,
            )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_exported_model(cls, path: Any) -> "IndependentExportedPredictor":
        predictor = IndependentExportedPredictor(path)
        expected_kind = cls._bundle_kind
        if predictor.kind != expected_kind:
            raise TypeError(
                f"exported bundle kind is {predictor.kind!r}, expected {expected_kind!r}"
            )
        return predictor


class IndependentExportedPredictor:
    """Dependency-light predictor for an exported independent-model bundle."""

    def __init__(self, path: Any) -> None:
        source = Path(path)
        with (source / "manifest.json").open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported independent model bundle schema version")
        if self.manifest.get("artifact_type") != "ctboost.independent_model_bundle":
            raise ValueError("directory does not contain a CTBoost independent model bundle")
        self.kind = str(self.manifest["kind"])
        if self.kind not in {
            "aft_lognormal",
            "multilabel_classification",
            "multioutput_regression",
        }:
            raise ValueError(f"unsupported independent model bundle kind: {self.kind!r}")
        model_files = self.manifest.get("models")
        n_outputs = self.manifest.get("n_outputs")
        if (
            not isinstance(model_files, list)
            or not model_files
            or not isinstance(n_outputs, int)
            or n_outputs != len(model_files)
            or any(not isinstance(filename, str) or not filename for filename in model_files)
        ):
            raise ValueError("independent model bundle has an invalid model list")
        resolved_source = source.resolve()
        resolved_models = []
        for filename in model_files:
            model_path = (source / filename).resolve()
            if resolved_source not in model_path.parents:
                raise ValueError("independent model bundle paths must stay inside its directory")
            resolved_models.append(model_path)
        self.predictors = [load_exported_predictor(path) for path in resolved_models]

    @staticmethod
    def _as_vector(values: Any) -> np.ndarray:
        array = np.asarray(values)
        return array.reshape(1) if array.ndim == 0 else array.reshape(-1)

    def predict_log_time(self, X: Any) -> np.ndarray:
        if self.kind != "aft_lognormal":
            raise AttributeError("predict_log_time is only available for AFT bundles")
        return self._as_vector(self.predictors[0].predict(X)).astype(np.float64)

    def predict_time(self, X: Any, *, kind: str = "median") -> np.ndarray:
        log_time = self.predict_log_time(X)
        normalized = str(kind).lower()
        if normalized == "median":
            shift = 0.0
        elif normalized == "mean":
            scale = float(self.manifest["scale"])
            shift = 0.5 * scale * scale
        else:
            raise ValueError("kind must be 'median' or 'mean'")
        return np.exp(np.clip(log_time + shift, -745.0, 709.0))

    def predict_proba(self, X: Any) -> List[np.ndarray]:
        if self.kind != "multilabel_classification":
            raise AttributeError("predict_proba is only available for multilabel bundles")
        probabilities = []
        for predictor in self.predictors:
            values = np.asarray(predictor.predict_proba(X))
            probabilities.append(values.reshape(1, -1) if values.ndim == 1 else values)
        return probabilities

    def predict_positive_proba(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.column_stack([values[:, 1] for values in probabilities])

    def predict(self, X: Any) -> np.ndarray:
        if self.kind == "multioutput_regression":
            return np.column_stack(
                [self._as_vector(predictor.predict(X)) for predictor in self.predictors]
            )
        if self.kind == "multilabel_classification":
            return np.column_stack(
                [self._as_vector(predictor.predict_class(X)) for predictor in self.predictors]
            )
        if self.kind == "aft_lognormal":
            prediction_type = str(self.manifest.get("prediction_type", "time"))
            return (
                self.predict_log_time(X)
                if prediction_type == "log_time"
                else self.predict_time(X)
            )
        raise RuntimeError(f"unsupported independent bundle kind: {self.kind!r}")
