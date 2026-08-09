"""Log-normal accelerated-failure-time survival convenience estimator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import log_ndtr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from ..core import Pool
from ..training import _normalize_eval_sets, make_eval_metric, make_objective
from ._independent import _IndependentPersistenceMixin, _row_count
from .regressor import CTBoostRegressor


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_FLOAT32_LIMIT = np.finfo(np.float32).max / 1024.0


def _aft_bounds(y: Any, *, name: str = "y") -> Tuple[np.ndarray, np.ndarray]:
    tuple_of_bound_vectors = False
    if isinstance(y, tuple) and len(y) == 2:
        lower_candidate = np.asarray(y[0])
        upper_candidate = np.asarray(y[1])
        lower_is_vector = lower_candidate.ndim > 0
        upper_is_vector = upper_candidate.ndim > 0
        if lower_is_vector != upper_is_vector:
            raise ValueError(
                f"{name} bounds tuple must contain two array-like vectors"
            )
        tuple_of_bound_vectors = lower_is_vector and upper_is_vector

    if tuple_of_bound_vectors:
        raw_lower = np.asarray(y[0])
        raw_upper = np.asarray(y[1])
        if np.iscomplexobj(raw_lower) or np.iscomplexobj(raw_upper):
            raise ValueError(f"Complex data not supported in {name}")
        lower = np.asarray(raw_lower, dtype=np.float64)
        upper = np.asarray(raw_upper, dtype=np.float64)
    else:
        raw_values = np.asarray(y)
        if np.iscomplexobj(raw_values):
            raise ValueError(f"Complex data not supported in {name}")
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim == 1:
            lower = values
            upper = values.copy()
        elif values.ndim == 2 and values.shape[1] == 2:
            lower = values[:, 0]
            upper = values[:, 1]
        else:
            raise ValueError(
                f"{name} must be exact times with shape (n_rows,), bounds with "
                "shape (n_rows, 2), or a (lower, upper) tuple"
            )
    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
        raise ValueError(f"{name} lower and upper bounds must be same-length 1D arrays")
    if lower.size == 0:
        raise ValueError(f"{name} must contain at least one observation")
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError(f"{name} bounds must not contain NaN")
    if np.any(np.isposinf(lower)) or np.any(np.isneginf(upper)):
        raise ValueError(f"{name} has an invalid infinite bound direction")
    if np.any((np.isfinite(lower) & (lower < 0.0))) or np.any(
        np.isfinite(upper) & (upper <= 0.0)
    ):
        raise ValueError(
            f"{name} time bounds must be positive; use lower=0 or -inf for left censoring"
        )
    if np.any(np.isfinite(lower) & np.isfinite(upper) & (lower > upper)):
        raise ValueError(f"{name} lower bounds must not exceed upper bounds")
    if np.any((~np.isfinite(lower) | (lower == 0.0)) & np.isposinf(upper)):
        raise ValueError(f"{name} contains an observation censored on both sides")
    return np.ascontiguousarray(lower), np.ascontiguousarray(upper)


def _aft_sample_weight(
    sample_weight: Any,
    *,
    n_rows: int,
    name: str = "sample_weight",
) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    values = np.asarray(sample_weight, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != n_rows:
        raise ValueError(f"{name} must be a 1D array with one value per row")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} entries must be finite and non-negative")
    if float(values.sum()) <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return values


def _log_standard_normal_density(value: np.ndarray) -> np.ndarray:
    return -0.5 * value * value - _LOG_SQRT_2PI


def _logdiffexp(log_large: np.ndarray, log_small: np.ndarray) -> np.ndarray:
    difference = log_small - log_large
    return log_large + np.log(-np.expm1(difference))


def _interval_log_probability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    use_survival = a >= 0.0
    result = np.empty_like(a)
    if np.any(~use_survival):
        result[~use_survival] = _logdiffexp(
            log_ndtr(b[~use_survival]),
            log_ndtr(a[~use_survival]),
        )
    if np.any(use_survival):
        result[use_survival] = _logdiffexp(
            log_ndtr(-a[use_survival]),
            log_ndtr(-b[use_survival]),
        )
    return result


def _aft_grad_hess_nll(
    prediction: Any,
    lower: Any,
    upper: Any,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log-normal AFT NLL derivatives with respect to log-time location."""
    mu = np.asarray(prediction, dtype=np.float64).reshape(-1)
    lower_values = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_values = np.asarray(upper, dtype=np.float64).reshape(-1)
    if mu.shape != lower_values.shape or mu.shape != upper_values.shape:
        raise ValueError("prediction and AFT bounds must have matching shapes")
    sigma = float(scale)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("scale must be a finite positive value")

    gradient = np.empty_like(mu)
    hessian = np.empty_like(mu)
    nll = np.empty_like(mu)
    finite_lower = np.isfinite(lower_values) & (lower_values > 0.0)
    finite_upper = np.isfinite(upper_values)
    exact = finite_lower & finite_upper & (lower_values == upper_values)
    left = (~finite_lower) & finite_upper
    right = finite_lower & np.isposinf(upper_values)
    interval = finite_lower & finite_upper & (lower_values < upper_values)
    if not np.all(exact | left | right | interval):
        raise ValueError("AFT bounds do not describe exact, left-, right-, or interval-censoring")

    if np.any(exact):
        log_time = np.log(lower_values[exact])
        z = (log_time - mu[exact]) / sigma
        gradient[exact] = (mu[exact] - log_time) / (sigma * sigma)
        hessian[exact] = 1.0 / (sigma * sigma)
        nll[exact] = (
            math.log(sigma)
            + log_time
            + _LOG_SQRT_2PI
            + 0.5 * z * z
        )

    if np.any(left):
        b = (np.log(upper_values[left]) - mu[left]) / sigma
        log_cdf = log_ndtr(b)
        ratio = np.exp(_log_standard_normal_density(b) - log_cdf)
        gradient[left] = ratio / sigma
        hessian[left] = ratio * (b + ratio) / (sigma * sigma)
        nll[left] = -log_cdf

    if np.any(right):
        a = (np.log(lower_values[right]) - mu[right]) / sigma
        log_survival = log_ndtr(-a)
        ratio = np.exp(_log_standard_normal_density(a) - log_survival)
        gradient[right] = -ratio / sigma
        hessian[right] = ratio * (ratio - a) / (sigma * sigma)
        nll[right] = -log_survival

    if np.any(interval):
        a = (np.log(lower_values[interval]) - mu[interval]) / sigma
        b = (np.log(upper_values[interval]) - mu[interval]) / sigma
        log_probability = _interval_log_probability(a, b)
        ratio_a = np.exp(_log_standard_normal_density(a) - log_probability)
        ratio_b = np.exp(_log_standard_normal_density(b) - log_probability)
        difference = ratio_b - ratio_a
        gradient[interval] = difference / sigma
        hessian[interval] = (
            b * ratio_b - a * ratio_a + difference * difference
        ) / (sigma * sigma)
        nll[interval] = -log_probability

    if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise FloatingPointError(
            "AFT derivatives became non-finite; check censoring bounds and scale"
        )
    if not np.all(np.isfinite(nll)):
        raise FloatingPointError(
            "AFT likelihood became non-finite; check censoring bounds and scale"
        )
    gradient = np.clip(gradient, -_FLOAT32_LIMIT, _FLOAT32_LIMIT)
    hessian = np.clip(hessian, 1e-12, _FLOAT32_LIMIT)
    return (
        gradient.astype(np.float32),
        hessian.astype(np.float32),
        nll,
    )


@dataclass
class _AFTLookup:
    lower: np.ndarray
    upper: np.ndarray
    scale: float

    def bounds(self, labels: Any) -> Tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(labels, dtype=np.float64).reshape(-1)
        indices = np.rint(raw).astype(np.int64)
        if not np.array_equal(raw, indices.astype(np.float64)):
            raise ValueError("internal AFT row identifiers must be integers")
        if np.any(indices < 0) or np.any(indices >= self.lower.shape[0]):
            raise ValueError("internal AFT row identifier is out of bounds")
        return self.lower[indices], self.upper[indices]


@dataclass
class _LogNormalAFTObjective:
    lookup: _AFTLookup

    def __call__(self, predictions: Any, label: Any, **_kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        lower, upper = self.lookup.bounds(label)
        gradient, hessian, _ = _aft_grad_hess_nll(
            predictions,
            lower,
            upper,
            self.lookup.scale,
        )
        return gradient, hessian


@dataclass
class _LogNormalAFTMetric:
    lookup: _AFTLookup

    def __call__(self, predictions: Any, label: Any, *, weight: Any = None, **_kwargs: Any) -> float:
        lower, upper = self.lookup.bounds(label)
        _, _, nll = _aft_grad_hess_nll(
            predictions,
            lower,
            upper,
            self.lookup.scale,
        )
        if weight is None:
            return float(np.mean(nll))
        weights = np.asarray(weight, dtype=np.float64).reshape(-1)
        if (
            weights.shape != nll.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or float(weights.sum()) <= 0.0
        ):
            raise ValueError(
                "AFT metric weights must be finite, non-negative, match rows, "
                "and have positive total weight"
            )
        return float(np.average(nll, weights=weights))


class CTBoostAFTSurvivalRegressor(
    _IndependentPersistenceMixin,
    RegressorMixin,
    BaseEstimator,
):
    """Log-normal AFT survival model backed by one ordinary CTBoost booster.

    The booster predicts the normal location of ``log(T)``.  Exact times use
    ``lower == upper``; left-censoring uses ``lower=0`` or ``-inf``;
    right-censoring uses ``upper=inf``; finite unequal bounds are intervals.
    """

    _bundle_kind = "aft_lognormal"

    def __init__(
        self,
        estimator: Optional[CTBoostRegressor] = None,
        *,
        scale: float = 1.0,
        prediction_type: str = "time",
    ) -> None:
        self.estimator = estimator
        self.scale = scale
        self.prediction_type = prediction_type

    def _validated_configuration(self) -> CTBoostRegressor:
        scale = float(self.scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be a finite positive value")
        prediction_type = str(self.prediction_type).lower()
        if prediction_type not in {"time", "log_time"}:
            raise ValueError("prediction_type must be 'time' or 'log_time'")
        base = self.estimator or CTBoostRegressor()
        if not isinstance(base, CTBoostRegressor):
            raise TypeError("estimator must be a CTBoostRegressor")
        params = base.get_params(deep=False)
        if int(params.get("distributed_world_size", 1)) != 1:
            raise ValueError("AFT convenience training does not support distributed_world_size > 1")
        if base._uses_feature_pipeline():
            raise ValueError(
                "AFT convenience training currently requires numeric or already-prepared features; "
                "target-aware categorical/text/embedding preprocessing is not supported"
            )
        return base

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        init_model: Any = None,
        **fit_params: Any,
    ) -> "CTBoostAFTSurvivalRegressor":
        base = self._validated_configuration()
        lower, upper = _aft_bounds(y)
        if lower.shape[0] != _row_count(X):
            raise ValueError("y bounds must contain one row per input sample")
        explicit_weight = _aft_sample_weight(
            sample_weight,
            n_rows=lower.shape[0],
        )
        pool_weight = (
            _aft_sample_weight(X.weight, n_rows=lower.shape[0], name="Pool.weight")
            if isinstance(X, Pool) and X.weight is not None
            else None
        )
        if explicit_weight is None:
            effective_weight = pool_weight
        elif pool_weight is None:
            effective_weight = explicit_weight
        else:
            effective_weight = explicit_weight * pool_weight
            if float(effective_weight.sum()) <= 0.0:
                raise ValueError("combined Pool.weight and sample_weight must have positive total weight")

        eval_entries = _normalize_eval_sets(eval_set)
        all_lower: List[np.ndarray] = [lower]
        all_upper: List[np.ndarray] = [upper]
        coded_eval_entries: List[Any] = []
        offset = lower.shape[0]
        for eval_index, entry in enumerate(eval_entries):
            if isinstance(entry, Pool):
                raise TypeError(
                    "AFT eval_set entries must be (X, bounds) tuples because Pool labels are 1D"
                )
            X_eval, y_eval, *metadata = entry
            eval_lower, eval_upper = _aft_bounds(
                y_eval,
                name=f"eval_set[{eval_index}] bounds",
            )
            if eval_lower.shape[0] != _row_count(X_eval):
                raise ValueError(
                    f"eval_set[{eval_index}] bounds must match its number of rows"
                )
            codes = np.arange(offset, offset + eval_lower.shape[0], dtype=np.float32)
            offset += eval_lower.shape[0]
            all_lower.append(eval_lower)
            all_upper.append(eval_upper)
            coded_eval_entries.append((X_eval, codes, *metadata))
        if offset > 16_777_216:
            raise ValueError("AFT convenience encoding supports at most 16,777,216 total rows")

        lookup = _AFTLookup(
            np.concatenate(all_lower),
            np.concatenate(all_upper),
            float(self.scale),
        )
        objective_name = f"AFTLogNormal(scale={float(self.scale):.17g})"
        objective = make_objective(
            _LogNormalAFTObjective(lookup),
            name=objective_name,
            native_objective="RMSE",
        )
        metric = make_eval_metric(
            _LogNormalAFTMetric(lookup),
            name="AFTNLL",
            higher_is_better=False,
            allow_early_stopping=True,
        )
        try:
            child = clone(base)
        except Exception:
            # AFT always performs one sequential child fit and replaces the
            # base objective and metric below.  Reconstructing a fresh
            # estimator from shallow constructor parameters therefore keeps
            # irrelevant, deliberately non-picklable callable parameters from
            # making sklearn.clone fail without sharing fitted model state.
            child = type(base)(**base.get_params(deep=False))
        child.set_params(loss_function=objective, eval_metric=metric)
        if init_model is not None:
            if not isinstance(init_model, CTBoostAFTSurvivalRegressor):
                raise TypeError("init_model must be a CTBoostAFTSurvivalRegressor")
            check_is_fitted(init_model, attributes="estimator_")
            if not np.isclose(float(init_model.scale), float(self.scale), rtol=0.0, atol=0.0):
                raise ValueError("init_model scale must match scale")
            fit_params["init_model"] = init_model.estimator_
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        if coded_eval_entries:
            fit_params["eval_set"] = (
                coded_eval_entries[0]
                if len(coded_eval_entries) == 1
                else coded_eval_entries
            )
        train_codes = np.arange(lower.shape[0], dtype=np.float32)
        self.estimator_ = child.fit(X, train_codes, **fit_params)
        self.estimators_ = [self.estimator_]
        self.n_features_in_ = int(self.estimator_.n_features_in_)
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(
                self.estimator_.feature_names_in_, dtype=object
            )
        self.best_iteration_ = int(self.estimator_.best_iteration_)
        self.evals_result_ = self.estimator_.evals_result_
        self.best_score_ = self.estimator_.best_score_
        self.objective_name_ = objective_name
        train_prediction = self.predict_log_time(X)
        _, _, train_nll = _aft_grad_hess_nll(
            train_prediction,
            lower,
            upper,
            float(self.scale),
        )
        if effective_weight is None:
            self.training_nll_ = float(np.mean(train_nll))
        else:
            self.training_nll_ = float(
                np.average(train_nll, weights=effective_weight)
            )
        return self

    def _more_tags(self) -> Dict[str, bool]:
        return {"allow_nan": True}

    def predict_log_time(self, X: Any) -> np.ndarray:
        check_is_fitted(self, attributes="estimator_")
        return np.asarray(self.estimator_.predict(X), dtype=np.float64).reshape(-1)

    def predict_time(self, X: Any, *, kind: str = "median") -> np.ndarray:
        log_time = self.predict_log_time(X)
        normalized = str(kind).lower()
        if normalized == "median":
            shift = 0.0
        elif normalized == "mean":
            shift = 0.5 * float(self.scale) ** 2
        else:
            raise ValueError("kind must be 'median' or 'mean'")
        return np.exp(np.clip(log_time + shift, -745.0, 709.0))

    def predict(self, X: Any) -> np.ndarray:
        return (
            self.predict_log_time(X)
            if str(self.prediction_type).lower() == "log_time"
            else self.predict_time(X)
        )

    def negative_log_likelihood(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
    ) -> float:
        lower, upper = _aft_bounds(y)
        prediction = self.predict_log_time(X)
        if prediction.shape[0] != lower.shape[0]:
            raise ValueError("y bounds must match the number of prediction rows")
        _, _, nll = _aft_grad_hess_nll(
            prediction,
            lower,
            upper,
            float(self.scale),
        )
        if sample_weight is None:
            return float(np.mean(nll))
        weights = _aft_sample_weight(sample_weight, n_rows=nll.shape[0])
        assert weights is not None
        return float(np.average(nll, weights=weights))

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        return -self.negative_log_likelihood(X, y, sample_weight=sample_weight)

    def get_booster(self) -> Any:
        check_is_fitted(self, attributes="estimator_")
        return self.estimator_.get_booster()

    def get_best_iteration(self) -> int:
        check_is_fitted(self, attributes="estimator_")
        return int(self.best_iteration_)

    def get_evals_result(self) -> Dict[str, Any]:
        check_is_fitted(self, attributes="estimator_")
        return deepcopy(self.evals_result_)

    def get_best_score(self) -> Dict[str, Any]:
        check_is_fitted(self, attributes="estimator_")
        return deepcopy(self.best_score_)

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, attributes="estimator_")
        return np.asarray(self.estimator_.feature_importances_, dtype=np.float32)

    def _bundle_metadata(self) -> Dict[str, Any]:
        return {
            "distribution": "lognormal",
            "scale": float(self.scale),
            "prediction_type": str(self.prediction_type).lower(),
            "objective_semantics": "log-normal AFT negative log-likelihood",
            "native_objective_surrogate": (
                "RMSE supplies scalar output shape and model serialization only; "
                "tree gradients and Hessians come from the AFT likelihood"
            ),
            "time_prediction": "median=exp(log_time); mean=exp(log_time + scale^2 / 2)",
            "censoring": {
                "exact": "lower == upper",
                "left": "lower == 0 or -inf",
                "right": "upper == inf",
                "interval": "0 < lower < upper < inf",
            },
        }


CTBoostAFTRegressor = CTBoostAFTSurvivalRegressor
