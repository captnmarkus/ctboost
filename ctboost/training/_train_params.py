"""Training-parameter helpers for ctboost.training."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np

from ..core import Pool
from .schema import (
    _resolve_feature_border_config,
    _resolve_feature_float_config,
    _resolve_feature_int_config,
    _resolve_feature_string_config,
)


def _normalize_callback_list(callbacks: Optional[Any]) -> List[Callable[..., Any]]:
    if callbacks is None:
        return []
    try:
        callback_list = list(callbacks)
    except TypeError as exc:
        raise TypeError("callbacks must be an iterable of callables") from exc
    for callback in callback_list:
        if not callable(callback):
            raise TypeError("callbacks must contain only callables")
    return callback_list


def _resolve_native_training_params(
    config: Mapping[str, Any],
    pool: Pool,
    *,
    init_state: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    objective = str(
        config.get(
            "objective",
            config.get("loss_function", "RMSE" if init_state is None else init_state["objective_name"]),
        )
    )
    params = {
        "objective": objective,
        "learning_rate": float(config.get("learning_rate", 0.1 if init_state is None else init_state["learning_rate"])),
        "max_depth": int(config.get("max_depth", 6 if init_state is None else init_state["max_depth"])),
        "alpha": float(config.get("alpha", 0.05 if init_state is None else init_state["alpha"])),
        "lambda_l2": float(
            config.get("lambda", config.get("lambda_l2", 1.0 if init_state is None else init_state["lambda_l2"]))
        ),
        "subsample": float(config.get("subsample", 1.0 if init_state is None else init_state.get("subsample", 1.0))),
        "bootstrap_type": str(config.get("bootstrap_type", "No" if init_state is None else init_state.get("bootstrap_type", "No"))),
        "bagging_temperature": float(config.get("bagging_temperature", 0.0 if init_state is None else init_state.get("bagging_temperature", 0.0))),
        "boosting_type": str(
            config.get(
                "boosting_type",
                "GradientBoosting" if init_state is None else init_state.get("boosting_type", "GradientBoosting"),
            )
        ),
        "drop_rate": float(config.get("drop_rate", 0.1 if init_state is None else init_state.get("drop_rate", 0.1))),
        "skip_drop": float(config.get("skip_drop", 0.5 if init_state is None else init_state.get("skip_drop", 0.5))),
        "max_drop": int(config.get("max_drop", 0 if init_state is None else init_state.get("max_drop", 0))),
        "monotone_constraints": list(
            config.get("monotone_constraints", [] if init_state is None else init_state.get("monotone_constraints", []))
            or []
        ),
        "interaction_constraints": list(
            config.get(
                "interaction_constraints",
                [] if init_state is None else init_state.get("interaction_constraints", []),
            )
            or []
        ),
        "colsample_bytree": float(
            config.get("colsample_bytree", 1.0 if init_state is None else init_state.get("colsample_bytree", 1.0))
        ),
        "random_strength": float(
            config.get("random_strength", 0.0 if init_state is None else init_state.get("random_strength", 0.0))
        ),
        "grow_policy": str(
            config.get("grow_policy", "DepthWise" if init_state is None else init_state.get("grow_policy", "DepthWise"))
        ),
        "max_leaves": int(config.get("max_leaves", 0 if init_state is None else init_state.get("max_leaves", 0))),
        "min_samples_split": int(
            config.get("min_samples_split", 2 if init_state is None else init_state.get("min_samples_split", 2))
        ),
        "min_data_in_leaf": int(
            config.get("min_data_in_leaf", 0 if init_state is None else init_state.get("min_data_in_leaf", 0))
        ),
        "min_child_weight": float(
            config.get("min_child_weight", 0.0 if init_state is None else init_state.get("min_child_weight", 0.0))
        ),
        "gamma": float(config.get("gamma", 0.0 if init_state is None else init_state.get("gamma", 0.0))),
        "max_leaf_weight": float(
            config.get("max_leaf_weight", 0.0 if init_state is None else init_state.get("max_leaf_weight", 0.0))
        ),
        "quantile_alpha": float(config.get("quantile_alpha", 0.5 if init_state is None else init_state["quantile_alpha"])),
        "huber_delta": float(config.get("huber_delta", 1.0 if init_state is None else init_state["huber_delta"])),
        "tweedie_variance_power": float(
            config.get(
                "tweedie_variance_power",
                1.5 if init_state is None else init_state.get("tweedie_variance_power", 1.5),
            )
        ),
        "task_type": str(config.get("task_type", "CPU" if init_state is None else init_state["task_type"])),
        "devices": str(config.get("devices", "0" if init_state is None else init_state["devices"])),
        "distributed_world_size": int(
            config.get("distributed_world_size", 1 if init_state is None else init_state.get("distributed_world_size", 1))
        ),
        "distributed_rank": int(
            config.get("distributed_rank", 0 if init_state is None else init_state.get("distributed_rank", 0))
        ),
        "distributed_root": str(config.get("distributed_root", "" if init_state is None else init_state.get("distributed_root", "")) or ""),
        "distributed_run_id": str(
            config.get("distributed_run_id", "default" if init_state is None else init_state.get("distributed_run_id", "default"))
        ),
        "distributed_timeout": float(
            config.get("distributed_timeout", 600.0 if init_state is None else init_state.get("distributed_timeout", 600.0))
        ),
        "native_external_memory": bool(
            config.get("external_memory", False if init_state is None else init_state.get("external_memory", False))
        ),
        "native_external_memory_dir": str(
            config.get("external_memory_dir", "" if init_state is None else init_state.get("external_memory_dir", "")) or ""
        ),
        "random_seed": int(config.get("random_seed", 0 if init_state is None else init_state.get("random_seed", 0))),
        "verbose": bool(config.get("verbose", False if init_state is None else init_state.get("verbose", False))),
    }
    if "num_classes" in config:
        params["num_classes"] = int(config["num_classes"])
    elif init_state is not None:
        params["num_classes"] = int(init_state["num_classes"])
    elif objective.lower() in {"multiclass", "softmax", "softmaxloss"}:
        params["num_classes"] = int(np.unique(pool.label).size)
    else:
        params["num_classes"] = 1
    params["max_bins"] = int(config.get("max_bins", 256 if init_state is None else init_state["max_bins"]))
    params["nan_mode"] = str(config.get("nan_mode", "Min" if init_state is None else init_state["nan_mode"]))
    params["feature_weights"] = _resolve_feature_float_config(
        config.get("feature_weights", [] if init_state is None else init_state.get("feature_weights", [])),
        pool,
        "feature_weights",
    )
    params["first_feature_use_penalties"] = _resolve_feature_float_config(
        config.get(
            "first_feature_use_penalties",
            [] if init_state is None else init_state.get("first_feature_use_penalties", []),
        ),
        pool,
        "first_feature_use_penalties",
    )
    params["max_bin_by_feature"] = _resolve_feature_int_config(
        config.get("max_bin_by_feature", [] if init_state is None else init_state.get("max_bin_by_feature", [])),
        pool,
        "max_bin_by_feature",
    )
    params["border_selection_method"] = str(
        config.get(
            "border_selection_method",
            "Quantile" if init_state is None else init_state.get("border_selection_method", "Quantile"),
        )
    )
    params["nan_mode_by_feature"] = _resolve_feature_string_config(
        config.get("nan_mode_by_feature", [] if init_state is None else init_state.get("nan_mode_by_feature", [])),
        pool,
        "nan_mode_by_feature",
    )
    params["feature_borders"] = _resolve_feature_border_config(
        config.get("feature_borders", [] if init_state is None else init_state.get("feature_borders", [])),
        pool,
        "feature_borders",
    )
    return params
