"""Helpers for raw-data, feature-pipeline, and external-memory pool preparation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ._prepared_data_support import (
    _EXTERNAL_MEMORY_KEYS,
    _FEATURE_PIPELINE_KEYS,
    _external_memory_pool,
    _prepare_feature_pipeline,
    _uses_feature_pipeline_params,
)
from .core import Pool
from .feature_pipeline import FeaturePipeline


def preparation_param_keys() -> Sequence[str]:
    return tuple(sorted(_FEATURE_PIPELINE_KEYS | _EXTERNAL_MEMORY_KEYS))


def preparation_params_from_mapping(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: params[key] for key in preparation_param_keys() if key in params}


def uses_feature_pipeline_params(params: Mapping[str, Any]) -> bool:
    return _uses_feature_pipeline_params(params)


def prepare_pool(
    data: Any,
    label: Any,
    *,
    cat_features: Optional[Sequence[Any]] = None,
    weight: Any = None,
    group_id: Any = None,
    group_weight: Any = None,
    subgroup_id: Any = None,
    baseline: Any = None,
    pairs: Any = None,
    pairs_weight: Any = None,
    feature_names: Optional[Sequence[str]] = None,
    column_roles: Any = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    categorical_schema: Optional[Mapping[str, Any]] = None,
    external_memory: bool = False,
    external_memory_dir: Optional[Any] = None,
    ordered_ctr: bool = False,
    one_hot_max_size: int = 0,
    max_cat_to_onehot: int = 0,
    max_cat_threshold: int = 0,
    categorical_combinations: Optional[Any] = None,
    pairwise_categorical_combinations: bool = False,
    simple_ctr: Optional[Any] = None,
    combinations_ctr: Optional[Any] = None,
    per_feature_ctr: Optional[Any] = None,
    text_features: Optional[Any] = None,
    text_hash_dim: int = 64,
    embedding_features: Optional[Any] = None,
    embedding_stats: Sequence[str] = ("mean", "std", "min", "max", "l2"),
    ctr_prior_strength: float = 1.0,
    random_seed: int = 0,
    feature_pipeline: Optional[FeaturePipeline] = None,
    fit_feature_pipeline: bool = True,
) -> Pool:
    prepared_data, prepared_label, prepared_cat_features, prepared_feature_names, pipeline = (
        _prepare_feature_pipeline(
            data,
            label,
            feature_names=feature_names,
            params={
                "cat_features": cat_features,
                "ordered_ctr": ordered_ctr,
                "one_hot_max_size": one_hot_max_size if one_hot_max_size else max_cat_to_onehot,
                "max_cat_threshold": max_cat_threshold,
                "categorical_combinations": categorical_combinations,
                "pairwise_categorical_combinations": pairwise_categorical_combinations,
                "simple_ctr": simple_ctr,
                "combinations_ctr": combinations_ctr,
                "per_feature_ctr": per_feature_ctr,
                "text_features": text_features,
                "text_hash_dim": text_hash_dim,
                "embedding_features": embedding_features,
                "embedding_stats": embedding_stats,
                "ctr_prior_strength": ctr_prior_strength,
                "random_seed": random_seed,
            },
            feature_pipeline=feature_pipeline,
            fit_feature_pipeline=fit_feature_pipeline,
        )
    )
    if external_memory:
        pool = _external_memory_pool(
            prepared_data,
            prepared_label,
            cat_features=prepared_cat_features,
            weight=weight,
            group_id=group_id,
            group_weight=group_weight,
            subgroup_id=subgroup_id,
            baseline=baseline,
            pairs=pairs,
            pairs_weight=pairs_weight,
            feature_names=prepared_feature_names,
            column_roles=column_roles,
            feature_metadata=feature_metadata,
            categorical_schema=categorical_schema,
            directory=external_memory_dir,
        )
    else:
        pool = Pool(
            data=prepared_data,
            label=prepared_label,
            cat_features=list(prepared_cat_features),
            weight=weight,
            group_id=group_id,
            group_weight=group_weight,
            subgroup_id=subgroup_id,
            baseline=baseline,
            pairs=pairs,
            pairs_weight=pairs_weight,
            feature_names=None if prepared_feature_names is None else list(prepared_feature_names),
            column_roles=column_roles,
            feature_metadata=feature_metadata,
            categorical_schema=categorical_schema,
            _releasable_feature_storage=True,
        )
    pool._feature_pipeline = pipeline
    return pool
