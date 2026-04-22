"""Snapshot and resume helpers for ctboost.training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .. import _core
from ..core import Pool
from ._learning_rate_schedule import _normalize_learning_rate_schedule, _resolve_learning_rate_for_iteration
from .booster import Booster
from .eval_metrics import _copy_evals_result
from .schema import _pool_schema_metadata

PathLike = Union[str, Path]

def _normalize_snapshot_interval(snapshot_interval: int) -> int:
    resolved_interval = int(snapshot_interval)
    if resolved_interval <= 0:
        raise ValueError("snapshot_interval must be positive")
    return resolved_interval

def _resolve_snapshot_path(snapshot_path: Optional[PathLike]) -> Optional[Path]:
    if snapshot_path is None:
        return None
    return Path(snapshot_path)

def _resolve_resume_snapshot_path(
    resume_from_snapshot: Any,
    snapshot_path: Optional[Path],
) -> Optional[Path]:
    if resume_from_snapshot in {None, False}:
        return None
    if resume_from_snapshot is True:
        if snapshot_path is None:
            raise ValueError("resume_from_snapshot=True requires snapshot_path")
        return snapshot_path
    return Path(resume_from_snapshot)

def _normalize_resume_signature_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalize_resume_signature_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_resume_signature_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return [_normalize_resume_signature_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_resume_signature_value(item) for item in value]
    return value

def _resume_signature_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_resume_signature_values_match(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_resume_signature_values_match(left_item, right_item) for left_item, right_item in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        try:
            return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))
        except (TypeError, ValueError):
            return left == right
    return left == right

def _resume_snapshot_config_signature(*, pool: Pool, **kwargs: Any) -> Dict[str, Any]:
    return {
        "objective_name": str(kwargs["objective"]),
        "learning_rate": float(kwargs["learning_rate"]),
        "max_depth": int(kwargs["max_depth"]),
        "alpha": float(kwargs["alpha"]),
        "lambda_l2": float(kwargs["lambda_l2"]),
        "subsample": float(kwargs["subsample"]),
        "bootstrap_type": str(kwargs["bootstrap_type"]),
        "bagging_temperature": float(kwargs["bagging_temperature"]),
        "boosting_type": str(kwargs["boosting_type"]),
        "drop_rate": float(kwargs["drop_rate"]),
        "skip_drop": float(kwargs["skip_drop"]),
        "max_drop": int(kwargs["max_drop"]),
        "monotone_constraints": _normalize_resume_signature_value(kwargs["monotone_constraints"]),
        "interaction_constraints": _normalize_resume_signature_value(kwargs["interaction_constraints"]),
        "colsample_bytree": float(kwargs["colsample_bytree"]),
        "feature_weights": _normalize_resume_signature_value(kwargs["feature_weights"]),
        "first_feature_use_penalties": _normalize_resume_signature_value(
            kwargs["first_feature_use_penalties"]
        ),
        "random_strength": float(kwargs["random_strength"]),
        "grow_policy": str(kwargs["grow_policy"]),
        "max_leaves": int(kwargs["max_leaves"]),
        "min_samples_split": int(kwargs["min_samples_split"]),
        "min_data_in_leaf": int(kwargs["min_data_in_leaf"]),
        "min_child_weight": float(kwargs["min_child_weight"]),
        "gamma": float(kwargs["gamma"]),
        "max_leaf_weight": float(kwargs["max_leaf_weight"]),
        "num_classes": int(kwargs["num_classes"]),
        "max_bins": int(kwargs["max_bins"]),
        "nan_mode": str(kwargs["nan_mode"]),
        "max_bin_by_feature": _normalize_resume_signature_value(kwargs["max_bin_by_feature"]),
        "border_selection_method": str(kwargs["border_selection_method"]),
        "nan_mode_by_feature": _normalize_resume_signature_value(kwargs["nan_mode_by_feature"]),
        "feature_borders": _normalize_resume_signature_value(kwargs["feature_borders"]),
        "external_memory": bool(kwargs["native_external_memory"]),
        "external_memory_dir": str(kwargs["native_external_memory_dir"]),
        "quantile_alpha": float(kwargs["quantile_alpha"]),
        "huber_delta": float(kwargs["huber_delta"]),
        "tweedie_variance_power": float(kwargs["tweedie_variance_power"]),
        "task_type": str(kwargs["task_type"]),
        "devices": str(kwargs["devices"]),
        "distributed_world_size": int(kwargs["distributed_world_size"]),
        "distributed_rank": int(kwargs["distributed_rank"]),
        "distributed_root": str(kwargs["distributed_root"]),
        "distributed_run_id": str(kwargs["distributed_run_id"]),
        "distributed_timeout": float(kwargs["distributed_timeout"]),
        "random_seed": int(kwargs["random_seed"]),
        "data_schema": _normalize_resume_signature_value(_pool_schema_metadata(pool)),
    }

def _resume_snapshot_state_signature(init_state: Mapping[str, Any], init_model: Any) -> Dict[str, Any]:
    return {
        "objective_name": str(init_state["objective_name"]),
        "learning_rate": float(init_state["learning_rate"]),
        "max_depth": int(init_state["max_depth"]),
        "alpha": float(init_state["alpha"]),
        "lambda_l2": float(init_state["lambda_l2"]),
        "subsample": float(init_state.get("subsample", 1.0)),
        "bootstrap_type": str(init_state.get("bootstrap_type", "No")),
        "bagging_temperature": float(init_state.get("bagging_temperature", 0.0)),
        "boosting_type": str(init_state.get("boosting_type", "GradientBoosting")),
        "drop_rate": float(init_state.get("drop_rate", 0.1)),
        "skip_drop": float(init_state.get("skip_drop", 0.5)),
        "max_drop": int(init_state.get("max_drop", 0)),
        "monotone_constraints": _normalize_resume_signature_value(
            init_state.get("monotone_constraints", [])
        ),
        "interaction_constraints": _normalize_resume_signature_value(
            init_state.get("interaction_constraints", [])
        ),
        "colsample_bytree": float(init_state.get("colsample_bytree", 1.0)),
        "feature_weights": _normalize_resume_signature_value(init_state.get("feature_weights", [])),
        "first_feature_use_penalties": _normalize_resume_signature_value(
            init_state.get("first_feature_use_penalties", [])
        ),
        "random_strength": float(init_state.get("random_strength", 0.0)),
        "grow_policy": str(init_state.get("grow_policy", "DepthWise")),
        "max_leaves": int(init_state.get("max_leaves", 0)),
        "min_samples_split": int(init_state.get("min_samples_split", 2)),
        "min_data_in_leaf": int(init_state.get("min_data_in_leaf", 0)),
        "min_child_weight": float(init_state.get("min_child_weight", 0.0)),
        "gamma": float(init_state.get("gamma", 0.0)),
        "max_leaf_weight": float(init_state.get("max_leaf_weight", 0.0)),
        "num_classes": int(init_state["num_classes"]),
        "max_bins": int(init_state["max_bins"]),
        "nan_mode": str(init_state["nan_mode"]),
        "max_bin_by_feature": _normalize_resume_signature_value(init_state.get("max_bin_by_feature", [])),
        "border_selection_method": str(init_state.get("border_selection_method", "Quantile")),
        "nan_mode_by_feature": _normalize_resume_signature_value(init_state.get("nan_mode_by_feature", [])),
        "feature_borders": _normalize_resume_signature_value(init_state.get("feature_borders", [])),
        "external_memory": bool(init_state.get("external_memory", False)),
        "external_memory_dir": str(init_state.get("external_memory_dir", "")),
        "quantile_alpha": float(init_state["quantile_alpha"]),
        "huber_delta": float(init_state["huber_delta"]),
        "tweedie_variance_power": float(init_state.get("tweedie_variance_power", 1.5)),
        "task_type": str(init_state["task_type"]),
        "devices": str(init_state["devices"]),
        "distributed_world_size": int(init_state.get("distributed_world_size", 1)),
        "distributed_rank": int(init_state.get("distributed_rank", 0)),
        "distributed_root": str(init_state.get("distributed_root", "")),
        "distributed_run_id": str(init_state.get("distributed_run_id", "default")),
        "distributed_timeout": float(init_state.get("distributed_timeout", 600.0)),
        "random_seed": int(init_state.get("random_seed", 0)),
        "data_schema": _normalize_resume_signature_value(getattr(init_model, "data_schema", {})),
    }

def _validate_resume_snapshot_contract(
    *,
    init_state: Mapping[str, Any],
    init_model: Any,
    requested_signature: Mapping[str, Any],
    learning_rate_schedule: Any,
    requested_iterations: int,
) -> None:
    snapshot_signature = _resume_snapshot_state_signature(init_state, init_model)
    for key, expected_value in snapshot_signature.items():
        if key == "data_schema" and not expected_value:
            continue
        actual_value = _normalize_resume_signature_value(requested_signature[key])
        if not _resume_signature_values_match(actual_value, expected_value):
            raise ValueError(
                "resume_from_snapshot requires the saved training configuration and data schema; "
                f"mismatch for {key!r}. Use init_model=... for a generic warm start instead."
            )
    if learning_rate_schedule is None:
        return
    existing_history = list(getattr(init_model, "learning_rate_history", []))
    if not existing_history:
        prediction_dimension = max(int(init_state.get("num_classes", 1)), 1)
        trained_iterations = int(len(init_state.get("trees", [])) // prediction_dimension)
        existing_history = [float(init_state["learning_rate"])] * trained_iterations
    for iteration, stored_learning_rate in enumerate(existing_history):
        expected_learning_rate = _resolve_learning_rate_for_iteration(
            learning_rate_schedule,
            iteration=iteration,
            total_iterations=requested_iterations,
        )
        if not np.isclose(
            float(stored_learning_rate),
            float(expected_learning_rate),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "resume_from_snapshot learning_rate_schedule must reproduce the saved schedule "
                f"through iteration {iteration + 1}"
            )

def _resolve_init_model_handle(init_model: Any) -> Optional[Any]:
    if init_model is None:
        return None
    if isinstance(init_model, Booster):
        return init_model._handle
    if isinstance(init_model, _core.GradientBooster):
        return init_model
    booster = getattr(init_model, "_booster", None)
    if isinstance(booster, Booster):
        return booster._handle
    raise TypeError("init_model must be a ctboost.Booster, native GradientBooster, or CTBoost estimator")

def _initial_evals_result_from_model(init_model: Any) -> Optional[Dict[str, Dict[str, List[float]]]]:
    metadata = None
    if isinstance(init_model, Booster):
        metadata = getattr(init_model, "_training_metadata", None)
        if metadata is not None and "evals_result" in metadata:
            return _copy_evals_result(metadata["evals_result"])
    booster = getattr(init_model, "_booster", None)
    if isinstance(booster, Booster):
        metadata = getattr(booster, "_training_metadata", None)
    if metadata is not None and "evals_result" in metadata:
        return _copy_evals_result(metadata["evals_result"])
    return None

def _initial_learning_rate_history_from_model(init_model: Any) -> Optional[List[float]]:
    if isinstance(init_model, Booster):
        metadata = getattr(init_model, "_training_metadata", None)
        if metadata is not None and "learning_rate_history" in metadata:
            return [float(value) for value in metadata["learning_rate_history"]]
    booster = getattr(init_model, "_booster", None)
    if isinstance(booster, Booster):
        metadata = getattr(booster, "_training_metadata", None)
        if metadata is not None and "learning_rate_history" in metadata:
            return [float(value) for value in metadata["learning_rate_history"]]
    return None

