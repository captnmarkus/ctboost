"""Learning-rate schedule helpers for ctboost.training."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional


def _normalize_learning_rate_schedule(learning_rate_schedule: Any) -> Optional[Any]:
    if learning_rate_schedule is None:
        return None
    if callable(learning_rate_schedule):
        return learning_rate_schedule
    if isinstance(learning_rate_schedule, (str, bytes)):
        raise TypeError("learning_rate_schedule must be a callable or a sequence of numeric values")
    try:
        resolved_schedule = list(learning_rate_schedule)
    except TypeError as exc:
        raise TypeError(
            "learning_rate_schedule must be a callable or a sequence of numeric values"
        ) from exc
    if not resolved_schedule:
        raise ValueError("learning_rate_schedule must not be empty")
    return resolved_schedule


def _call_learning_rate_schedule(schedule: Callable[..., Any], iteration: int, total_iterations: int) -> Any:
    try:
        signature = inspect.signature(schedule)
    except (TypeError, ValueError):
        return schedule(iteration, total_iterations=total_iterations)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    keyword_arguments = {"total_iterations": total_iterations}
    if accepts_kwargs:
        return schedule(iteration, **keyword_arguments)
    filtered_kwargs = {
        key: value for key, value in keyword_arguments.items() if key in signature.parameters
    }
    return schedule(iteration, **filtered_kwargs)


def _resolve_learning_rate_for_iteration(
    learning_rate_schedule: Any,
    *,
    iteration: int,
    total_iterations: int,
) -> float:
    if callable(learning_rate_schedule):
        resolved_learning_rate = _call_learning_rate_schedule(
            learning_rate_schedule, iteration, total_iterations
        )
    else:
        if iteration >= len(learning_rate_schedule):
            raise ValueError(
                "learning_rate_schedule does not provide enough entries for the requested number of rounds"
            )
        resolved_learning_rate = learning_rate_schedule[iteration]
    learning_rate = float(resolved_learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate_schedule entries must be positive")
    return learning_rate
