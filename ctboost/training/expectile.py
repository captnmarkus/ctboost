"""Expectile regression through CTBoost's callable objective interface."""

from __future__ import annotations

from functools import partial

import numpy as np

from .eval_metrics import EvalMetricSpec, make_eval_metric
from .objectives import ObjectiveSpec, make_objective


def _expectile_terms(predictions, label, tau):
    predictions = np.asarray(predictions, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    if predictions.ndim != 1 or predictions.shape != label.shape:
        raise ValueError(
            "expectile regression requires one scalar prediction per label"
        )
    residual = predictions - label
    side_weight = np.where(residual < 0, tau, 1.0 - tau)
    return residual, side_weight


def _derivatives(predictions, label, *, tau):
    residual, side_weight = _expectile_terms(predictions, label, tau)
    hessian = 2.0 * side_weight
    return hessian * residual, hessian


def _loss(predictions, label, *, tau, weight=None):
    residual, side_weight = _expectile_terms(predictions, label, tau)
    if weight is not None and not np.any(weight):
        return 0.0  # Same empty-weight convention as native regression metrics.
    return float(np.average(side_weight * np.square(residual), weights=weight))


def make_expectile_loss(tau: float) -> tuple[ObjectiveSpec, EvalMetricSpec]:
    """Return an expectile training objective and matching validation metric.

    A level above 0.5 penalizes underprediction more heavily. At 0.5 the
    derivatives match squared-error training and the metric is half the
    weighted mean squared error. Reuse the same level for warm starts.
    """
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be strictly between 0 and 1")
    name = f"Expectile(tau={tau!r})"
    objective = make_objective(partial(_derivatives, tau=tau), name=name)
    metric = make_eval_metric(
        partial(_loss, tau=tau),
        name=name,
        higher_is_better=False,
        allow_early_stopping=True,
    )
    return objective, metric
