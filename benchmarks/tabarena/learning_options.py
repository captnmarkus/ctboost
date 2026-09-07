"""Task-aware CPU variants for a separately versioned CTBoost experiment.

This module does not change the frozen TabArena search space. Applicability is
determined from the training task and weights, never validation/test outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

LEARNING_VARIANT_PARAM = "tabarena_learning_variant"
LEARNING_INAPPLICABLE_PARAM = "tabarena_learning_on_inapplicable"
CPU_LEARNING_VARIANTS = (
    "baseline",
    "backtracking_3",
    "full_3",
    "joint",
    "full_3_joint",
)
ELIGIBLE_LEARNING_VARIANTS = {
    "binary": ("baseline", "backtracking_3"),
    "regression": ("baseline", "backtracking_3"),
    "multiclass": ("baseline", "full_3", "joint", "full_3_joint"),
}
_BASELINE_CONTROLS = {
    "leaf_estimation_iterations": 1,
    "leaf_estimation_backtracking": False,
    "multiclass_leaf_solver": "diagonal",
    "multiclass_feature_test": "single",
}
_OBJECTIVES = {
    "binary": {"logloss", "binary_logloss", "binary:logistic"},
    "regression": {"rmse", "squarederror", "squared_error"},
    "multiclass": {"multiclass", "softmax", "softmaxloss"},
}


def validate_learning_modes_by_problem_type(modes: Mapping[str, str]) -> dict[str, str]:
    """Validate a complete, preselected family mapping without selecting an arm."""
    if not isinstance(modes, Mapping) or set(modes) != set(ELIGIBLE_LEARNING_VARIANTS):
        raise ValueError(
            "learning modes must specify binary, regression, and multiclass exactly"
        )
    resolved = dict(modes)
    for problem_type, mode in resolved.items():
        if mode not in ELIGIBLE_LEARNING_VARIANTS[problem_type]:
            raise ValueError(f"learning variant {mode!r} is invalid for {problem_type}")
    return resolved


def _effective_joint_weight_reason(
    params: Mapping[str, Any],
    sample_weight: Any,
    labels: Any,
) -> Optional[str]:
    """Check the effective float32 weights that native joint training receives."""
    y = None if labels is None else np.asarray(labels)
    if y is not None and y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if sample_weight is None:
        weights = np.ones(1 if y is None else len(y), dtype=np.float32)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            weights = np.asarray(sample_weight, dtype=np.float32)
        if weights.ndim != 1 or (y is not None and len(weights) != len(y)):
            raise ValueError("sample_weight must be one-dimensional and match labels")
        weights = weights.copy()
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample weights must be finite and nonnegative")

    supplied = [
        key
        for key in ("class_weight", "class_weights", "auto_class_weights")
        if params.get(key) is not None
    ]
    if len(supplied) > 1:
        raise ValueError("specify only one class-weight setting")
    if supplied:
        if y is None:
            return "joint class-weight eligibility requires training labels"
        setting = params[supplied[0]]
        unique, counts = np.unique(y, return_counts=True)
        if supplied[0] == "auto_class_weights" or isinstance(setting, str):
            if str(setting).lower() != "balanced":
                raise ValueError("automatic class weights must be 'balanced'")
            class_weights = dict(zip(unique.tolist(), len(y) / (len(unique) * counts)))
        elif isinstance(setting, Mapping):
            class_weights = dict(setting)
            if set(class_weights) != set(unique.tolist()):
                raise ValueError(
                    "class weights must cover every training label exactly"
                )
        else:
            values = np.asarray(setting, dtype=np.float32)
            if values.ndim != 1 or len(values) != len(unique):
                raise ValueError(
                    "class weights must match the number of training classes"
                )
            class_weights = dict(zip(unique.tolist(), values.tolist()))
        objective_weights = np.ones(len(y), dtype=np.float32)
        for label, value in class_weights.items():
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("class weights must be finite and positive")
            with np.errstate(over="ignore", invalid="ignore"):
                objective_weights[y == label] *= float(value)
        with np.errstate(over="ignore", invalid="ignore"):
            weights *= objective_weights
    if params.get("scale_pos_weight") is not None:
        return "scale_pos_weight is a binary-only control and cannot accompany joint multiclass"
    if not np.isfinite(weights).all():
        raise ValueError("effective weights must remain finite in float32")
    if np.any(weights != np.floor(weights)):
        return "joint requires integer effective sample/class weights"
    return None


def resolve_cpu_learning_options(
    params: Mapping[str, Any],
    *,
    problem_type: str,
    num_classes: Optional[int] = None,
    sample_weight: Any = None,
    labels: Any = None,
    variant: Any = None,
    on_inapplicable: Optional[str] = None,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Resolve one pilot arm or a predeclared task-mapped portfolio marker.

    ``skip`` returns ``(None, metadata)`` for an ineligible pilot arm. ``error``
    raises before model construction. ``baseline`` is an explicit portfolio
    fallback, recorded in metadata. Invalid CPU/GPU identity or malformed
    settings always raise; weights are never rounded, dropped, or rescaled.
    No marker/variant leaves legacy parameters unchanged.
    """
    resolved = dict(params)
    marker = resolved.pop(LEARNING_VARIANT_PARAM, None)
    marked_policy = resolved.pop(LEARNING_INAPPLICABLE_PARAM, None)
    if variant is not None and marker is not None and variant != marker:
        raise ValueError("conflicting explicit and marked learning variants")
    requested = marker if variant is None else variant
    if (
        on_inapplicable is not None
        and marked_policy is not None
        and on_inapplicable != marked_policy
    ):
        raise ValueError("conflicting learning applicability policies")
    policy = on_inapplicable if on_inapplicable is not None else marked_policy
    if policy is None:
        policy = "error"
    if policy not in {"error", "skip", "baseline"}:
        raise ValueError("on_inapplicable must be 'error', 'skip', or 'baseline'")
    if requested is None:
        if marked_policy is not None:
            raise ValueError("learning applicability policy requires a variant marker")
        return resolved, {
            "status": "unchanged",
            "applicable": True,
            "requested_variant": None,
            "applied_variant": None,
            "reason": None,
        }
    if str(resolved.get("task_type", "CPU")).strip().upper() != "CPU":
        raise ValueError(
            "learning-variant markers are CPU-only; GPU portfolio is unchanged"
        )
    if int(resolved.get("distributed_world_size", 1)) != 1:
        raise ValueError("CPU learning variants do not support distributed training")
    problem = str(problem_type).strip().lower()
    if problem not in ELIGIBLE_LEARNING_VARIANTS:
        raise ValueError(
            "learning variants support binary, regression, and multiclass tasks"
        )
    mapping = None
    if isinstance(requested, Mapping):
        mapping = validate_learning_modes_by_problem_type(requested)
        requested = mapping[problem]
    if not isinstance(requested, str) or requested not in CPU_LEARNING_VARIANTS:
        raise ValueError(f"unknown CPU learning variant: {requested!r}")
    if num_classes is not None and (
        isinstance(num_classes, (bool, np.bool_))
        or not isinstance(num_classes, (int, np.integer))
        or num_classes < 1
    ):
        raise ValueError("num_classes must be a positive integer")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("labels must be a nonempty one-dimensional array")
    if labels is not None and problem != "regression":
        observed_classes = len(np.unique(labels))
        if num_classes is None:
            num_classes = observed_classes
        elif int(num_classes) != observed_classes:
            # The sklearn wrapper derives output dimension from its fit labels.
            raise ValueError(
                "num_classes must match the classes present in training labels"
            )

    metadata = {
        "requested_variant": requested,
        "applied_variant": requested,
        "problem_type": problem,
        "num_classes": None if num_classes is None else int(num_classes),
        "applicable": True,
        "status": "applied",
        "reason": None,
        "on_inapplicable": policy,
    }
    if mapping is not None:
        metadata["modes_by_problem_type"] = mapping
    reason = None
    if requested not in ELIGIBLE_LEARNING_VARIANTS[problem]:
        reason = f"{requested} is not applicable to {problem}"
    elif requested != "baseline":
        objective = resolved.get("objective", resolved.get("loss_function"))
        if objective is not None and (
            not isinstance(objective, str)
            or objective.lower() not in _OBJECTIVES[problem]
        ):
            reason = f"{requested} requires the built-in objective for {problem}"
        elif problem == "binary" and num_classes is not None and num_classes != 2:
            reason = "binary backtracking requires exactly 2 training classes"
        elif problem == "multiclass" and (
            num_classes is None or not 3 <= num_classes <= 32
        ):
            reason = "full/joint variants require 3-32 training classes"
        elif requested in {"joint", "full_3_joint"}:
            bootstrap = str(resolved.get("bootstrap_type", "No")).lower()
            temperature = float(resolved.get("bagging_temperature", 0.0))
            if bootstrap == "bayesian" and (
                temperature > 0 or not np.isfinite(temperature)
            ):
                reason = "joint cannot use fractional Bayesian bootstrap weights"
            else:
                reason = _effective_joint_weight_reason(resolved, sample_weight, labels)
    if reason is not None:
        metadata.update(applicable=False, reason=reason)
        if policy == "error":
            raise ValueError(reason)
        if policy == "skip":
            metadata.update(status="not_applicable", applied_variant=None)
            return None, metadata
        metadata.update(status="baseline_fallback", applied_variant="baseline")

    applied = metadata["applied_variant"]
    controls = dict(_BASELINE_CONTROLS)
    if applied == "backtracking_3":
        controls.update(leaf_estimation_backtracking=True, leaf_estimation_iterations=3)
    elif applied in {"full_3", "full_3_joint"}:
        controls.update(multiclass_leaf_solver="full", leaf_estimation_iterations=3)
    if applied in {"joint", "full_3_joint"}:
        controls["multiclass_feature_test"] = "joint"
    resolved.update(controls)
    metadata["resolved_controls"] = controls
    return resolved, metadata


__all__ = [
    "CPU_LEARNING_VARIANTS",
    "ELIGIBLE_LEARNING_VARIANTS",
    "LEARNING_INAPPLICABLE_PARAM",
    "LEARNING_VARIANT_PARAM",
    "resolve_cpu_learning_options",
    "validate_learning_modes_by_problem_type",
]
