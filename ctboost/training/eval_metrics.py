"""Eval-metric specifications for ctboost.training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .. import _core

@dataclass(frozen=True)
class EvalMetricSpec:
    func: Callable[..., Any]
    name: Optional[str] = None
    higher_is_better: Optional[bool] = None
    allow_early_stopping: bool = False

def make_eval_metric(
    func: Callable[..., Any],
    *,
    name: Optional[str] = None,
    higher_is_better: Optional[bool] = None,
    allow_early_stopping: bool = False,
) -> EvalMetricSpec:
    if not callable(func):
        raise TypeError("func must be callable")
    if higher_is_better is not None and not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be a boolean or None")
    return EvalMetricSpec(
        func=func,
        name=name,
        higher_is_better=higher_is_better,
        allow_early_stopping=bool(allow_early_stopping),
    )

def _normalize_eval_metric_name(metric: Any, *, default_name: str) -> str:
    if isinstance(metric, EvalMetricSpec):
        raw_name = metric.name
    elif isinstance(metric, str):
        raw_name = metric
    else:
        raw_name = getattr(metric, "name", None) or getattr(metric, "__name__", None)
    metric_name = default_name if raw_name is None else str(raw_name).strip()
    if not metric_name:
        raise ValueError("custom eval metrics must define a non-empty name")
    return metric_name

def _normalize_single_eval_metric(metric: Any, *, default_name: str) -> Dict[str, Any]:
    if isinstance(metric, str):
        metric_name = metric.strip()
        if not metric_name:
            raise ValueError("eval_metric names must be non-empty")
        return {
            "kind": "native",
            "name": metric_name,
            "callable": None,
            "higher_is_better": None,
            "allow_early_stopping": True,
        }
    if isinstance(metric, EvalMetricSpec):
        metric_name = _normalize_eval_metric_name(metric, default_name=default_name)
        return {
            "kind": "callable",
            "name": metric_name,
            "callable": metric.func,
            "higher_is_better": metric.higher_is_better,
            "allow_early_stopping": bool(metric.allow_early_stopping),
        }
    if callable(metric):
        metric_name = _normalize_eval_metric_name(metric, default_name=default_name)
        return {
            "kind": "callable",
            "name": metric_name,
            "callable": metric,
            "higher_is_better": getattr(metric, "higher_is_better", None),
            "allow_early_stopping": bool(getattr(metric, "allow_early_stopping", False)),
        }
    raise TypeError("eval_metric must be a string, callable, EvalMetricSpec, or a sequence of them")

def _normalize_eval_metrics(eval_metric: Any) -> List[Dict[str, Any]]:
    if eval_metric is None:
        return []
    if isinstance(eval_metric, str) and not eval_metric.strip():
        return []
    raw_metrics = [eval_metric] if not isinstance(eval_metric, (list, tuple)) else list(eval_metric)
    if not raw_metrics:
        return []
    normalized = [
        _normalize_single_eval_metric(metric, default_name=f"custom_metric_{index}")
        for index, metric in enumerate(raw_metrics)
    ]
    seen_names = set()
    for metric in normalized:
        lowered = metric["name"].lower()
        if lowered in seen_names:
            raise ValueError("eval_metric entries must have unique names")
        seen_names.add(lowered)
    return normalized

def _copy_evals_result(result: Mapping[str, Mapping[str, Iterable[float]]]) -> Dict[str, Dict[str, List[float]]]:
    return {
        str(dataset_name): {
            str(metric_name): [float(value) for value in metric_history]
            for metric_name, metric_history in dataset_metrics.items()
        }
        for dataset_name, dataset_metrics in result.items()
    }

def _metric_config(params: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "quantile_alpha": float(params.get("quantile_alpha", 0.5)),
        "huber_delta": float(params.get("huber_delta", 1.0)),
        "tweedie_variance_power": float(params.get("tweedie_variance_power", 1.5)),
    }

def _metric_higher_is_better(metric_name: str, params: Mapping[str, Any]) -> bool:
    config = _metric_config(params)
    return bool(
        _core._metric_higher_is_better(
            metric_name,
            quantile_alpha=config["quantile_alpha"],
            huber_delta=config["huber_delta"],
            tweedie_variance_power=config["tweedie_variance_power"],
        )
    )

def _eval_metric_direction(metric_spec: Mapping[str, Any], params: Mapping[str, Any]) -> Optional[bool]:
    if metric_spec["kind"] == "native":
        return _metric_higher_is_better(str(metric_spec["name"]), params)
    direction = metric_spec.get("higher_is_better")
    if direction is None:
        return None
    return bool(direction)

