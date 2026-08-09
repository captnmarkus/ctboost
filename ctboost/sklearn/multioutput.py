"""Independent-tree multi-output and multilabel CTBoost estimators."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._independent import (
    _IndependentPersistenceMixin,
    _fit_child,
    _row_count,
    _sample_weight_for_output,
    _slice_eval_sets,
    _target_matrix,
    _target_snapshot_path,
    _validate_execution,
    _validate_process_payload,
    _validate_sample_weight,
)
from .classifier import CTBoostClassifier
from .regressor import CTBoostRegressor


def _fit_params_for_output(
    fit_params: Dict[str, Any],
    output_index: int,
    init_model: Any,
) -> Dict[str, Any]:
    resolved = dict(fit_params)
    for name in ("snapshot_path", "resume_from_snapshot"):
        value = resolved.get(name)
        if value is None or isinstance(value, bool):
            continue
        resolved[name] = _target_snapshot_path(value, output_index)
    if init_model is not None:
        resolved["init_model"] = init_model.estimators_[output_index]
    return resolved


class CTBoostMultiOutputRegressor(
    _IndependentPersistenceMixin,
    MultiOutputMixin,
    RegressorMixin,
    BaseEstimator,
):
    """Fit one independent :class:`CTBoostRegressor` per output column.

    Trees and conditional split tests are not shared between outputs.  CPU
    children may be fitted in parallel with ``n_jobs``.  GPU children must be
    sequential, and distributed child fitting is intentionally unsupported.
    """

    _bundle_kind = "multioutput_regression"

    def __init__(
        self,
        estimator: Optional[CTBoostRegressor] = None,
        *,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        init_model: Any = None,
        **fit_params: Any,
    ) -> "CTBoostMultiOutputRegressor":
        y_matrix = _target_matrix(y, name="y", numeric=True)
        if y_matrix.shape[0] != _row_count(X):
            raise ValueError("y must contain one row per input sample")
        n_outputs = int(y_matrix.shape[1])
        weight_matrix = _validate_sample_weight(
            sample_weight,
            n_rows=y_matrix.shape[0],
            n_outputs=n_outputs,
        )
        base_estimator = self.estimator or CTBoostRegressor()
        if not isinstance(base_estimator, CTBoostRegressor):
            raise TypeError("estimator must be a CTBoostRegressor")
        resolved_n_jobs = _validate_execution(base_estimator, self.n_jobs, X=X)
        if resolved_n_jobs != 1 and fit_params.get("callbacks"):
            raise ValueError("callbacks require n_jobs=1 for deterministic ordering")
        if init_model is not None:
            if not isinstance(init_model, CTBoostMultiOutputRegressor):
                raise TypeError("init_model must be a CTBoostMultiOutputRegressor")
            check_is_fitted(init_model, attributes="estimators_")
            if len(init_model.estimators_) != n_outputs:
                raise ValueError("init_model output count must match y")
        _validate_process_payload(
            base_estimator,
            fit_params,
            init_model,
            n_jobs=resolved_n_jobs,
        )

        jobs = []
        for output_index in range(n_outputs):
            jobs.append(
                delayed(_fit_child)(
                    base_estimator,
                    X,
                    y_matrix[:, output_index],
                    _sample_weight_for_output(weight_matrix, output_index),
                    _slice_eval_sets(
                        eval_set,
                        output_index=output_index,
                        n_outputs=n_outputs,
                        numeric=True,
                    ),
                    _fit_params_for_output(fit_params, output_index, init_model),
                    allow_shallow_parameter_clone=resolved_n_jobs == 1,
                )
            )
        self.estimators_ = Parallel(n_jobs=resolved_n_jobs, prefer="processes")(jobs)
        self.n_outputs_ = n_outputs
        self.best_iteration_ = np.asarray(
            [estimator.best_iteration_ for estimator in self.estimators_],
            dtype=np.int64,
        )
        self.evals_result_ = [estimator.evals_result_ for estimator in self.estimators_]
        self.best_score_ = [estimator.best_score_ for estimator in self.estimators_]
        self.n_features_in_ = int(self.estimators_[0].n_features_in_)
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = np.asarray(
                self.estimators_[0].feature_names_in_, dtype=object
            )
        columns = getattr(y, "columns", None)
        self.output_names_ = (
            None if columns is None else [str(column) for column in columns]
        )
        return self

    def _more_tags(self) -> Dict[str, bool]:
        return {
            "allow_nan": True,
            "multioutput": True,
            "multioutput_only": True,
        }

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.column_stack(
            [estimator.predict(X) for estimator in self.estimators_]
        ).astype(np.float64, copy=False)

    def get_boosters(self) -> List[Any]:
        check_is_fitted(self, attributes="estimators_")
        return [estimator.get_booster() for estimator in self.estimators_]

    def get_best_iteration(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return self.best_iteration_.copy()

    def get_evals_result(self) -> List[Dict[str, Any]]:
        check_is_fitted(self, attributes="estimators_")
        return deepcopy(self.evals_result_)

    def get_best_score(self) -> List[Dict[str, Any]]:
        check_is_fitted(self, attributes="estimators_")
        return deepcopy(self.best_score_)

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.mean(
            np.vstack([estimator.feature_importances_ for estimator in self.estimators_]),
            axis=0,
        )

    @property
    def feature_importances_per_output_(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.vstack([estimator.feature_importances_ for estimator in self.estimators_])

    def _bundle_metadata(self) -> Dict[str, Any]:
        return {"output_names": self.output_names_}


class CTBoostMultiLabelClassifier(
    _IndependentPersistenceMixin,
    MultiOutputMixin,
    ClassifierMixin,
    BaseEstimator,
):
    """Fit one independent binary :class:`CTBoostClassifier` per label.

    ``predict_proba`` follows scikit-learn's multi-output convention and
    returns one ``(n_rows, 2)`` probability matrix per label.
    """

    _bundle_kind = "multilabel_classification"

    def __init__(
        self,
        estimator: Optional[CTBoostClassifier] = None,
        *,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        init_model: Any = None,
        **fit_params: Any,
    ) -> "CTBoostMultiLabelClassifier":
        y_matrix = _target_matrix(y, name="y", numeric=False)
        if y_matrix.shape[0] != _row_count(X):
            raise ValueError("y must contain one row per input sample")
        n_outputs = int(y_matrix.shape[1])
        for output_index in range(n_outputs):
            classes = np.unique(y_matrix[:, output_index])
            if classes.size != 2:
                raise ValueError(
                    "each multilabel output must contain exactly two classes; "
                    f"output {output_index} contains {classes.size}"
                )
        weight_matrix = _validate_sample_weight(
            sample_weight,
            n_rows=y_matrix.shape[0],
            n_outputs=n_outputs,
        )
        base_estimator = self.estimator or CTBoostClassifier()
        if not isinstance(base_estimator, CTBoostClassifier):
            raise TypeError("estimator must be a CTBoostClassifier")
        resolved_n_jobs = _validate_execution(base_estimator, self.n_jobs, X=X)
        if resolved_n_jobs != 1 and fit_params.get("callbacks"):
            raise ValueError("callbacks require n_jobs=1 for deterministic ordering")
        if init_model is not None:
            if not isinstance(init_model, CTBoostMultiLabelClassifier):
                raise TypeError("init_model must be a CTBoostMultiLabelClassifier")
            check_is_fitted(init_model, attributes="estimators_")
            if len(init_model.estimators_) != n_outputs:
                raise ValueError("init_model label count must match y")
        _validate_process_payload(
            base_estimator,
            fit_params,
            init_model,
            n_jobs=resolved_n_jobs,
        )

        jobs = []
        for output_index in range(n_outputs):
            jobs.append(
                delayed(_fit_child)(
                    base_estimator,
                    X,
                    y_matrix[:, output_index],
                    _sample_weight_for_output(weight_matrix, output_index),
                    _slice_eval_sets(
                        eval_set,
                        output_index=output_index,
                        n_outputs=n_outputs,
                        numeric=False,
                    ),
                    _fit_params_for_output(fit_params, output_index, init_model),
                    allow_shallow_parameter_clone=resolved_n_jobs == 1,
                )
            )
        self.estimators_ = Parallel(n_jobs=resolved_n_jobs, prefer="processes")(jobs)
        self.n_outputs_ = n_outputs
        self.n_labels_ = n_outputs
        self.best_iteration_ = np.asarray(
            [estimator.best_iteration_ for estimator in self.estimators_],
            dtype=np.int64,
        )
        self.evals_result_ = [estimator.evals_result_ for estimator in self.estimators_]
        self.best_score_ = [estimator.best_score_ for estimator in self.estimators_]
        self.classes_ = [np.asarray(estimator.classes_) for estimator in self.estimators_]
        self.n_features_in_ = int(self.estimators_[0].n_features_in_)
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = np.asarray(
                self.estimators_[0].feature_names_in_, dtype=object
            )
        columns = getattr(y, "columns", None)
        self.output_names_ = (
            None if columns is None else [str(column) for column in columns]
        )
        return self

    def _more_tags(self) -> Dict[str, bool]:
        return {
            "allow_nan": True,
            "multioutput": True,
            "multioutput_only": True,
            "multilabel": True,
        }

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])

    def predict_proba(self, X: Any) -> List[np.ndarray]:
        check_is_fitted(self, attributes="estimators_")
        return [np.asarray(estimator.predict_proba(X)) for estimator in self.estimators_]

    def predict_positive_proba(self, X: Any) -> np.ndarray:
        return np.column_stack([probabilities[:, 1] for probabilities in self.predict_proba(X)])

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.column_stack(
            [estimator.get_booster().predict(X) for estimator in self.estimators_]
        )

    def get_boosters(self) -> List[Any]:
        check_is_fitted(self, attributes="estimators_")
        return [estimator.get_booster() for estimator in self.estimators_]

    def get_best_iteration(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return self.best_iteration_.copy()

    def get_evals_result(self) -> List[Dict[str, Any]]:
        check_is_fitted(self, attributes="estimators_")
        return deepcopy(self.evals_result_)

    def get_best_score(self) -> List[Dict[str, Any]]:
        check_is_fitted(self, attributes="estimators_")
        return deepcopy(self.best_score_)

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.mean(
            np.vstack([estimator.feature_importances_ for estimator in self.estimators_]),
            axis=0,
        )

    @property
    def feature_importances_per_output_(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimators_")
        return np.vstack([estimator.feature_importances_ for estimator in self.estimators_])

    def _bundle_metadata(self) -> Dict[str, Any]:
        return {
            "output_names": self.output_names_,
            "classes": [classes.tolist() for classes in self.classes_],
        }
