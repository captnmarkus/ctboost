"""Pool construction helpers for ctboost.training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core import Pool, _clone_pool
from ..feature_pipeline import FeaturePipeline
from ..prepared_data import prepare_pool

def _pool_from_data_and_label(
    data: Any,
    label: Any,
    group_id: Any = None,
    *,
    weight: Any = None,
    group_weight: Any = None,
    subgroup_id: Any = None,
    baseline: Any = None,
    pairs: Any = None,
    pairs_weight: Any = None,
    feature_names: Optional[List[str]] = None,
    column_roles: Any = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    categorical_schema: Optional[Mapping[str, Any]] = None,
) -> Pool:
    if isinstance(data, Pool):
        if (
            label is None
            and group_id is None
            and weight is None
            and group_weight is None
            and subgroup_id is None
            and baseline is None
            and pairs is None
            and pairs_weight is None
            and feature_names is None
            and column_roles is None
            and feature_metadata is None
            and categorical_schema is None
        ):
            return data
        resolved_label = data.label if label is None else label
        resolved_weight = data.weight if weight is None else weight
        resolved_group_id = data.group_id if group_id is None else group_id
        resolved_group_weight = data.group_weight if group_weight is None else group_weight
        resolved_subgroup_id = data.subgroup_id if subgroup_id is None else subgroup_id
        resolved_baseline = data.baseline if baseline is None else baseline
        resolved_pairs = data.pairs if pairs is None else pairs
        resolved_pairs_weight = data.pairs_weight if pairs_weight is None else pairs_weight
        return _clone_pool(
            data,
            label=resolved_label,
            weight=resolved_weight,
            group_id=resolved_group_id,
            group_weight=resolved_group_weight,
            subgroup_id=resolved_subgroup_id,
            baseline=resolved_baseline,
            pairs=resolved_pairs,
            pairs_weight=resolved_pairs_weight,
            feature_names=data.feature_names if feature_names is None else feature_names,
            column_roles=data.column_roles if column_roles is None else column_roles,
            feature_metadata=data.feature_metadata if feature_metadata is None else feature_metadata,
            categorical_schema=data.categorical_schema if categorical_schema is None else categorical_schema,
            releasable_feature_storage=True,
        )

    if label is None:
        raise ValueError("y must be provided when data is not a Pool")
    return Pool(
        data=data,
        label=label,
        weight=weight,
        group_id=group_id,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=feature_names,
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        _releasable_feature_storage=True,
    )

def _feature_pipeline_config_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "cat_features": config.get("cat_features"),
        "ordered_ctr": bool(config.get("ordered_ctr", False)),
        "one_hot_max_size": int(config.get("one_hot_max_size", config.get("max_cat_to_onehot", 0))),
        "max_cat_threshold": int(config.get("max_cat_threshold", 0)),
        "categorical_combinations": config.get("categorical_combinations"),
        "pairwise_categorical_combinations": bool(config.get("pairwise_categorical_combinations", False)),
        "simple_ctr": config.get("simple_ctr"),
        "combinations_ctr": config.get("combinations_ctr"),
        "per_feature_ctr": config.get("per_feature_ctr"),
        "text_features": config.get("text_features"),
        "text_hash_dim": int(config.get("text_hash_dim", 64)),
        "embedding_features": config.get("embedding_features"),
        "embedding_stats": list(config.get("embedding_stats", ("mean", "std", "min", "max", "l2"))),
        "ctr_prior_strength": float(config.get("ctr_prior_strength", 1.0)),
        "random_seed": int(config.get("random_seed", 0)),
    }

def _normalize_pipeline_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_pipeline_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_pipeline_value(item) for item in value]
    return value

def _assert_compatible_feature_pipeline(
    feature_pipeline: FeaturePipeline,
    config: Mapping[str, Any],
) -> None:
    pipeline_state = feature_pipeline.to_state()
    current_signature = _feature_pipeline_config_signature(config)
    saved_signature = {
        "cat_features": pipeline_state.get("cat_features"),
        "ordered_ctr": bool(pipeline_state.get("ordered_ctr", False)),
        "one_hot_max_size": int(pipeline_state.get("one_hot_max_size", 0)),
        "max_cat_threshold": int(pipeline_state.get("max_cat_threshold", 0)),
        "categorical_combinations": pipeline_state.get("categorical_combinations"),
        "pairwise_categorical_combinations": bool(
            pipeline_state.get("pairwise_categorical_combinations", False)
        ),
        "simple_ctr": pipeline_state.get("simple_ctr"),
        "combinations_ctr": pipeline_state.get("combinations_ctr"),
        "per_feature_ctr": pipeline_state.get("per_feature_ctr"),
        "text_features": pipeline_state.get("text_features"),
        "text_hash_dim": int(pipeline_state.get("text_hash_dim", 64)),
        "embedding_features": pipeline_state.get("embedding_features"),
        "embedding_stats": list(pipeline_state.get("embedding_stats", ("mean", "std", "min", "max", "l2"))),
        "ctr_prior_strength": float(pipeline_state.get("ctr_prior_strength", 1.0)),
        "random_seed": int(pipeline_state.get("random_seed", 0)),
    }
    if _normalize_pipeline_value(saved_signature) != _normalize_pipeline_value(current_signature):
        raise ValueError(
            "raw-data warm start requires feature-pipeline parameters that match init_model"
        )

def _resolve_init_model_pipeline(init_model: Any) -> Optional[FeaturePipeline]:
    from .booster import Booster

    if isinstance(init_model, Booster):
        return init_model._feature_pipeline
    booster = getattr(init_model, "_booster", None)
    if isinstance(booster, Booster):
        return booster._feature_pipeline
    return None

def _prepare_pool_from_raw(
    data: Any,
    label: Any,
    *,
    group_id: Any = None,
    weight: Any = None,
    group_weight: Any = None,
    subgroup_id: Any = None,
    baseline: Any = None,
    pairs: Any = None,
    pairs_weight: Any = None,
    feature_names: Optional[List[str]] = None,
    column_roles: Any = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    categorical_schema: Optional[Mapping[str, Any]] = None,
    params: Mapping[str, Any],
    feature_pipeline: Optional[FeaturePipeline] = None,
    fit_feature_pipeline: bool = True,
    use_eval_external_memory: bool = False,
) -> Pool:
    from .distributed import _normalize_distributed_config

    external_memory_key = "eval_external_memory" if use_eval_external_memory else "external_memory"
    external_memory_dir_key = (
        "eval_external_memory_dir" if use_eval_external_memory else "external_memory_dir"
    )
    external_memory = bool(
        params.get(external_memory_key, params.get("external_memory", False) if use_eval_external_memory else False)
    )
    external_memory_dir = params.get(external_memory_dir_key)
    if use_eval_external_memory and external_memory_dir is None:
        external_memory_dir = params.get("external_memory_dir")
    distributed = _normalize_distributed_config(params)
    if external_memory and external_memory_dir is not None and distributed is not None:
        shard_suffix = (
            f"eval_rank_{distributed['rank']:05d}"
            if use_eval_external_memory
            else f"rank_{distributed['rank']:05d}"
        )
        external_memory_dir = Path(external_memory_dir) / shard_suffix

    return prepare_pool(
        data,
        label,
        cat_features=params.get("cat_features"),
        weight=weight,
        group_id=group_id,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=feature_names,
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        external_memory=external_memory,
        external_memory_dir=external_memory_dir,
        ordered_ctr=bool(params.get("ordered_ctr", False)),
        one_hot_max_size=int(params.get("one_hot_max_size", params.get("max_cat_to_onehot", 0))),
        max_cat_threshold=int(params.get("max_cat_threshold", 0)),
        categorical_combinations=params.get("categorical_combinations"),
        pairwise_categorical_combinations=bool(params.get("pairwise_categorical_combinations", False)),
        simple_ctr=params.get("simple_ctr"),
        combinations_ctr=params.get("combinations_ctr"),
        per_feature_ctr=params.get("per_feature_ctr"),
        text_features=params.get("text_features"),
        text_hash_dim=int(params.get("text_hash_dim", 64)),
        embedding_features=params.get("embedding_features"),
        embedding_stats=params.get("embedding_stats", ("mean", "std", "min", "max", "l2")),
        ctr_prior_strength=float(params.get("ctr_prior_strength", 1.0)),
        random_seed=int(params.get("random_seed", 0)),
        feature_pipeline=feature_pipeline,
        fit_feature_pipeline=fit_feature_pipeline,
    )

def _prediction_pool(data: Any) -> Pool:
    if isinstance(data, Pool):
        return data
    num_rows = int(data.shape[0]) if hasattr(data, "shape") else len(data)
    return Pool(data=data, label=np.zeros(num_rows, dtype=np.float32))

def _resolve_num_iteration(num_iteration: Optional[int]) -> int:
    if num_iteration is None:
        return -1
    if num_iteration <= 0:
        raise ValueError("num_iteration must be a positive integer")
    return int(num_iteration)

