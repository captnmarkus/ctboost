"""Eval-metric runtime selection for ctboost.training."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .eval_metrics import (
    _eval_metric_direction,
    _normalize_eval_metrics,
    _normalize_single_eval_metric,
)


def _resolve_metric_runtime(
    *,
    config: Mapping[str, Any],
    init_state: Optional[Mapping[str, Any]],
    eval_pools: List[Any],
    resolved_eval_names: List[str],
    default_eval_names: List[str],
    early_stopping_rounds: Optional[int],
    early_stopping_metric: Optional[Any],
    early_stopping_name: Optional[str],
    user_callback_list: List[Any],
    learning_rate_schedule: Any,
    snapshot_callback: Optional[Any],
    distributed_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    eval_metric_specs = _normalize_eval_metrics(
        config.get("eval_metric", "" if init_state is None else init_state["eval_metric_name"])
    )
    objective_name = str(
        config.get(
            "objective",
            config.get("loss_function", "RMSE" if init_state is None else init_state["objective_name"]),
        )
    )
    if not eval_metric_specs:
        eval_metric_specs = _normalize_eval_metrics(
            objective_name if init_state is None else str(init_state["eval_metric_name"])
        )
    if early_stopping_metric is not None:
        requested_metric = _normalize_single_eval_metric(
            early_stopping_metric,
            default_name="custom_early_stopping_metric",
        )
        matching_metric = next(
            (
                metric
                for metric in eval_metric_specs
                if str(metric["name"]).lower() == str(requested_metric["name"]).lower()
            ),
            None,
        )
        primary_eval_metric_spec = requested_metric if matching_metric is None else matching_metric
        if matching_metric is None:
            eval_metric_specs = [requested_metric] + eval_metric_specs
    else:
        primary_eval_metric_spec = eval_metric_specs[0]
    primary_eval_metric = str(primary_eval_metric_spec["name"])
    if early_stopping_name is not None:
        if not resolved_eval_names:
            raise ValueError("early_stopping_name requires eval_set")
        primary_eval_name = str(early_stopping_name)
        if primary_eval_name not in resolved_eval_names:
            raise ValueError("early_stopping_name must match one of the provided eval_names")
    else:
        primary_eval_name = resolved_eval_names[0] if resolved_eval_names else None
    metric_directions = {
        str(metric_spec["name"]): _eval_metric_direction(metric_spec, config)
        for metric_spec in eval_metric_specs
    }
    primary_metric_direction = metric_directions[primary_eval_metric]
    if primary_metric_direction is None:
        raise ValueError("the primary eval metric must declare higher_is_better when using a callable metric")
    if (
        early_stopping_rounds is not None
        and primary_eval_metric_spec["kind"] == "callable"
        and not primary_eval_metric_spec["allow_early_stopping"]
    ):
        raise ValueError("custom early_stopping_metric must be created with allow_early_stopping=True")
    native_eval_metric = next(
        (
            str(metric_spec["name"])
            for metric_spec in eval_metric_specs
            if metric_spec["kind"] == "native"
        ),
        objective_name if init_state is None else str(init_state["eval_metric_name"]),
    )
    use_python_eval_surface = (
        bool(user_callback_list)
        or learning_rate_schedule is not None
        or len(eval_pools) > 1
        or len(eval_metric_specs) > 1
        or resolved_eval_names != default_eval_names
        or any(metric_spec["kind"] == "callable" for metric_spec in eval_metric_specs)
        or (snapshot_callback is not None and distributed_config is not None)
    )
    callback_list = list(user_callback_list)
    if snapshot_callback is not None and use_python_eval_surface:
        callback_list.append(snapshot_callback)
    return {
        "callback_list": callback_list,
        "eval_metric_specs": eval_metric_specs,
        "eval_metric_names": [str(metric_spec["name"]) for metric_spec in eval_metric_specs],
        "metric_directions": metric_directions,
        "primary_eval_metric": primary_eval_metric,
        "primary_eval_name": primary_eval_name,
        "primary_metric_direction": bool(primary_metric_direction),
        "native_eval_metric": native_eval_metric,
        "use_python_eval_surface": use_python_eval_surface,
    }
