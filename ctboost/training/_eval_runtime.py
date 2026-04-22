"""Eval-metric runtime helpers for ctboost.training."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Mapping

import numpy as np

from .. import _core
from ..core import Pool
from ._distributed_payloads import _distributed_allgather_value
from .eval_metrics import _metric_config

def _call_metric_with_supported_kwargs(
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

def _evaluate_custom_metric_from_inputs(
    metric_spec: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    num_classes: int,
    params: Mapping[str, Any],
) -> float:
    predictions = np.asarray(inputs["predictions"], dtype=np.float32)
    label = np.asarray(inputs["label"], dtype=np.float32)
    keyword_arguments = {
        "weight": np.asarray(inputs["weight"], dtype=np.float32),
        "group_id": None if inputs.get("group_id") is None else np.asarray(inputs["group_id"], dtype=np.int64),
        "group_weight": None
        if inputs.get("group_weight") is None
        else np.asarray(inputs["group_weight"], dtype=np.float32),
        "subgroup_id": None
        if inputs.get("subgroup_id") is None
        else np.asarray(inputs["subgroup_id"], dtype=np.int64),
        "pairs": None if inputs.get("pairs") is None else np.asarray(inputs["pairs"], dtype=np.int64),
        "pairs_weight": None
        if inputs.get("pairs_weight") is None
        else np.asarray(inputs["pairs_weight"], dtype=np.float32),
        "num_classes": int(num_classes),
        "params": dict(params),
    }
    result = _call_metric_with_supported_kwargs(
        metric_spec["callable"],
        predictions,
        label,
        keyword_arguments,
    )
    result_array = np.asarray(result, dtype=np.float32)
    if result_array.ndim != 0:
        raise TypeError("custom eval metrics must return a scalar float value")
    return float(result_array)

def _evaluate_metric_on_pool(
    metric_name: str,
    predictions: np.ndarray,
    pool: Pool,
    *,
    num_classes: int,
    params: Mapping[str, Any],
) -> float:
    config = _metric_config(params)
    weights = (
        np.asarray(pool.weight, dtype=np.float32)
        if pool.weight is not None
        else np.ones(pool.num_rows, dtype=np.float32)
    )
    group_ids = None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64)
    group_weight = None if pool.group_weight is None else np.asarray(pool.group_weight, dtype=np.float32)
    subgroup_id = None if pool.subgroup_id is None else np.asarray(pool.subgroup_id, dtype=np.int64)
    pairs = None if pool.pairs is None else np.asarray(pool.pairs, dtype=np.int64)
    pairs_weight = None if pool.pairs_weight is None else np.asarray(pool.pairs_weight, dtype=np.float32)
    return float(
        _core._evaluate_metric(
            np.asarray(predictions, dtype=np.float32),
            np.asarray(pool.label, dtype=np.float32),
            weights,
            metric_name,
            num_classes,
            group_ids,
            group_weight,
            subgroup_id,
            pairs,
            pairs_weight,
            quantile_alpha=config["quantile_alpha"],
            huber_delta=config["huber_delta"],
            tweedie_variance_power=config["tweedie_variance_power"],
        )
    )

def _collect_prediction_metric_inputs(predictions: np.ndarray, pool: Pool) -> Dict[str, Any]:
    return {
        "predictions": np.asarray(predictions, dtype=np.float32),
        "label": np.asarray(pool.label, dtype=np.float32),
        "weight": (
            np.asarray(pool.weight, dtype=np.float32)
            if pool.weight is not None
            else np.ones(pool.num_rows, dtype=np.float32)
        ),
        "group_id": None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64),
        "group_weight": None
        if pool.group_weight is None
        else np.asarray(pool.group_weight, dtype=np.float32),
        "subgroup_id": None
        if pool.subgroup_id is None
        else np.asarray(pool.subgroup_id, dtype=np.int64),
        "pairs": None if pool.pairs is None else np.asarray(pool.pairs, dtype=np.int64),
        "pairs_weight": None
        if pool.pairs_weight is None
        else np.asarray(pool.pairs_weight, dtype=np.float32),
    }

def _evaluate_metric_from_inputs(
    metric_name: Any,
    inputs: Mapping[str, Any],
    *,
    num_classes: int,
    params: Mapping[str, Any],
) -> float:
    if isinstance(metric_name, Mapping):
        metric_spec = metric_name
        if metric_spec["kind"] == "callable":
            return _evaluate_custom_metric_from_inputs(
                metric_spec,
                inputs,
                num_classes=num_classes,
                params=params,
            )
        resolved_metric_name = str(metric_spec["name"])
    else:
        resolved_metric_name = str(metric_name)
    config = _metric_config(params)
    return float(
        _core._evaluate_metric(
            np.asarray(inputs["predictions"], dtype=np.float32),
            np.asarray(inputs["label"], dtype=np.float32),
            np.asarray(inputs["weight"], dtype=np.float32),
            resolved_metric_name,
            num_classes,
            None if inputs.get("group_id") is None else np.asarray(inputs["group_id"], dtype=np.int64),
            None
            if inputs.get("group_weight") is None
            else np.asarray(inputs["group_weight"], dtype=np.float32),
            None
            if inputs.get("subgroup_id") is None
            else np.asarray(inputs["subgroup_id"], dtype=np.int64),
            None if inputs.get("pairs") is None else np.asarray(inputs["pairs"], dtype=np.int64),
            None
            if inputs.get("pairs_weight") is None
            else np.asarray(inputs["pairs_weight"], dtype=np.float32),
            quantile_alpha=config["quantile_alpha"],
            huber_delta=config["huber_delta"],
            tweedie_variance_power=config["tweedie_variance_power"],
        )
    )

def _merge_prediction_metric_inputs(shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "predictions": np.concatenate([np.asarray(shard["predictions"]) for shard in shards], axis=0),
        "label": np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0),
        "weight": np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0),
    }
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id:
        if any(shard["group_id"] is None for shard in shards):
            raise ValueError("distributed metric shards must either all include group_id or all omit it")
        merged["group_id"] = np.concatenate(
            [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards],
            axis=0,
        )
    else:
        merged["group_id"] = None
    has_group_weight = any(shard.get("group_weight") is not None for shard in shards)
    if has_group_weight:
        if any(shard.get("group_weight") is None for shard in shards):
            raise ValueError("distributed metric shards must either all include group_weight or all omit it")
        merged["group_weight"] = np.concatenate(
            [np.asarray(shard["group_weight"], dtype=np.float32) for shard in shards],
            axis=0,
        )
    else:
        merged["group_weight"] = None
    has_subgroup_id = any(shard.get("subgroup_id") is not None for shard in shards)
    if has_subgroup_id:
        if any(shard.get("subgroup_id") is None for shard in shards):
            raise ValueError("distributed metric shards must either all include subgroup_id or all omit it")
        merged["subgroup_id"] = np.concatenate(
            [np.asarray(shard["subgroup_id"], dtype=np.int64) for shard in shards],
            axis=0,
        )
    else:
        merged["subgroup_id"] = None
    pair_arrays = []
    pair_weight_arrays = []
    use_pair_weight = False
    row_offset = 0
    for shard in shards:
        shard_pairs = shard.get("pairs")
        if shard_pairs is not None:
            resolved_pairs = np.asarray(shard_pairs, dtype=np.int64)
            pair_arrays.append(resolved_pairs + np.int64(row_offset))
            shard_pairs_weight = shard.get("pairs_weight")
            if shard_pairs_weight is not None:
                use_pair_weight = True
                pair_weight_arrays.append(np.asarray(shard_pairs_weight, dtype=np.float32))
            elif resolved_pairs.shape[0] > 0:
                pair_weight_arrays.append(np.ones(resolved_pairs.shape[0], dtype=np.float32))
        row_offset += int(np.asarray(shard["predictions"]).shape[0])
    if pair_arrays:
        merged["pairs"] = np.concatenate(pair_arrays, axis=0)
        merged["pairs_weight"] = (
            np.concatenate(pair_weight_arrays, axis=0) if use_pair_weight else None
        )
    else:
        merged["pairs"] = None
        merged["pairs_weight"] = None
    return merged

def _evaluate_metric_distributed(
    distributed: Optional[Dict[str, Any]],
    *,
    key: str,
    metric_name: Any,
    predictions: np.ndarray,
    pool: Pool,
    num_classes: int,
    params: Mapping[str, Any],
) -> float:
    if distributed is None:
        return _evaluate_metric_from_inputs(
            metric_name,
            _collect_prediction_metric_inputs(predictions, pool),
            num_classes=num_classes,
            params=params,
        )
    shards = _distributed_allgather_value(
        distributed,
        key,
        _collect_prediction_metric_inputs(predictions, pool),
    )
    return _evaluate_metric_from_inputs(
        metric_name,
        _merge_prediction_metric_inputs(shards),
        num_classes=num_classes,
        params=params,
    )

