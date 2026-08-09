"""Callable objective specifications and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from ..core import Pool


@dataclass(frozen=True)
class ObjectiveSpec:
    """Describe a Python objective and the native model semantics it extends.

    The callable receives raw predictions followed by labels and returns
    ``(gradients, hessians)``.  ``native_objective`` controls prediction shape,
    built-in metric defaults, label weighting, and inference metadata; it does
    not replace the derivatives returned by ``func`` during training.
    """

    func: Callable[..., Any]
    name: Optional[str] = None
    native_objective: str = "RMSE"

    def __post_init__(self) -> None:
        if not callable(self.func):
            raise TypeError("func must be callable")
        native_objective = str(self.native_objective).strip()
        if not native_objective:
            raise ValueError("native_objective must be a non-empty objective name")
        object.__setattr__(self, "native_objective", native_objective)
        if self.name is not None:
            name = str(self.name).strip()
            if not name:
                raise ValueError("custom objective name must be non-empty")
            object.__setattr__(self, "name", name)


def make_objective(
    func: Callable[..., Any],
    *,
    name: Optional[str] = None,
    native_objective: str = "RMSE",
) -> ObjectiveSpec:
    """Create a named custom objective specification.

    ``native_objective`` should match the estimator family: ``RMSE`` for
    regression, ``Logloss`` for binary classification, ``MultiClass`` for
    multiclass classification, or a ranking objective for ranking.
    """

    return ObjectiveSpec(
        func=func,
        name=name,
        native_objective=native_objective,
    )


def _callable_name(func: Callable[..., Any]) -> str:
    raw_name = getattr(func, "name", None) or getattr(func, "__name__", None)
    if raw_name is None:
        raw_name = type(func).__name__
    name = str(raw_name).strip()
    return name or "custom_objective"


def _with_native_objective(value: Any, native_objective: str) -> Any:
    if isinstance(value, ObjectiveSpec):
        return value
    if callable(value):
        return ObjectiveSpec(
            value,
            native_objective=native_objective,
        )
    return value


def _resolve_objective_spec(
    value: Any,
    *,
    default_native_objective: str,
) -> Optional[ObjectiveSpec]:
    if isinstance(value, ObjectiveSpec):
        return ObjectiveSpec(
            func=value.func,
            name=value.name or _callable_name(value.func),
            native_objective=value.native_objective,
        )
    if not callable(value):
        return None
    native_objective = getattr(value, "native_objective", default_native_objective)
    return ObjectiveSpec(
        func=value,
        name=_callable_name(value),
        native_objective=str(native_objective),
    )


def _objective_value(config: Mapping[str, Any]) -> Any:
    return config.get("objective", config.get("loss_function", "RMSE"))


def _resolve_objective_runtime(config: Mapping[str, Any]) -> Optional[ObjectiveSpec]:
    num_classes = int(config.get("num_classes", 1))
    default_native_objective = "MultiClass" if num_classes > 2 else "RMSE"
    return _resolve_objective_spec(
        _objective_value(config),
        default_native_objective=default_native_objective,
    )


def _native_objective_config(
    config: Mapping[str, Any],
    objective: Optional[ObjectiveSpec],
) -> Dict[str, Any]:
    native_config = dict(config)
    if objective is None:
        return native_config
    native_config.pop("loss_function", None)
    native_config["objective"] = objective.native_objective
    return native_config


def _call_with_supported_kwargs(
    func: Callable[..., Any],
    predictions: np.ndarray,
    label: np.ndarray,
    keyword_arguments: Mapping[str, Any],
) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(predictions, label, **dict(keyword_arguments))

    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return func(predictions, label, **dict(keyword_arguments))
    filtered_kwargs = {
        key: value
        for key, value in keyword_arguments.items()
        if key in signature.parameters
    }
    return func(predictions, label, **filtered_kwargs)


def _custom_objective_callback(
    objective: ObjectiveSpec,
    pool: Pool,
    params: Mapping[str, Any],
) -> Callable[[np.ndarray, np.ndarray], Any]:
    keyword_arguments = {
        "weight": (
            np.asarray(pool.weight, dtype=np.float32)
            if pool.weight is not None
            else np.ones(pool.num_rows, dtype=np.float32)
        ),
        "group_id": (
            None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64)
        ),
        "group_weight": (
            None
            if pool.group_weight is None
            else np.asarray(pool.group_weight, dtype=np.float32)
        ),
        "subgroup_id": (
            None
            if pool.subgroup_id is None
            else np.asarray(pool.subgroup_id, dtype=np.int64)
        ),
        "pairs": None if pool.pairs is None else np.asarray(pool.pairs, dtype=np.int64),
        "pairs_weight": (
            None
            if pool.pairs_weight is None
            else np.asarray(pool.pairs_weight, dtype=np.float32)
        ),
        "num_classes": int(params["num_classes"]),
        "params": dict(params),
    }

    def callback(predictions: np.ndarray, label: np.ndarray) -> Any:
        return _call_with_supported_kwargs(
            objective.func,
            predictions,
            label,
            keyword_arguments,
        )

    return callback


def _stored_custom_objective_metadata(model: Any) -> Optional[Mapping[str, Any]]:
    candidate = model
    if hasattr(candidate, "get_booster"):
        candidate = candidate.get_booster()
    elif hasattr(candidate, "_booster"):
        candidate = candidate._booster
    metadata = getattr(candidate, "_training_metadata", None)
    if not isinstance(metadata, Mapping) or not metadata.get("custom_objective", False):
        return None
    return metadata


def _validate_custom_objective_continuation(
    init_model: Any,
    objective: Optional[ObjectiveSpec],
) -> None:
    metadata = _stored_custom_objective_metadata(init_model)
    if metadata is None:
        return
    stored_name = str(metadata.get("objective_name", "custom_objective"))
    if objective is None:
        raise ValueError(
            "init_model was trained with custom objective "
            f"{stored_name!r}; pass that callable objective again to continue training"
        )
    if objective.name != stored_name:
        raise ValueError(
            "custom objective name must match init_model: "
            f"expected {stored_name!r}, got {objective.name!r}"
        )


def _custom_objective_native_name(value: Any, *, default: str) -> Optional[str]:
    spec = _resolve_objective_spec(value, default_native_objective=default)
    return None if spec is None else spec.native_objective

