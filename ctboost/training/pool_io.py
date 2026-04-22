"""Pool preparation helpers for ctboost.training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from ..core import Pool
from ..feature_pipeline import FeaturePipeline
from ..prepared_data import uses_feature_pipeline_params
from ._eval_sets import _normalize_eval_names, _normalize_eval_sets
from ._pool_build import (
    _assert_compatible_feature_pipeline,
    _pool_from_data_and_label,
    _prepare_pool_from_raw,
    _resolve_init_model_pipeline,
)
from .distributed import (
    _distributed_collective_context,
    _normalize_distributed_config,
    _prepare_distributed_feature_pipeline_pool,
)

@dataclass
class PreparedTrainingData:
    pool: Pool
    eval_pools: List[Pool]
    eval_names: List[str]
    feature_pipeline: Optional[FeaturePipeline] = None

def prepare_training_data(
    pool: Any,
    params: Mapping[str, Any],
    *,
    label: Any = None,
    weight: Any = None,
    group_id: Any = None,
    group_weight: Any = None,
    subgroup_id: Any = None,
    baseline: Any = None,
    pairs: Any = None,
    pairs_weight: Any = None,
    feature_names: Optional[List[str]] = None,
    column_roles: Any = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    categorical_schema: Optional[Mapping[str, Any]] = None,
    eval_set: Any = None,
    eval_names: Any = None,
    init_model: Any = None,
) -> PreparedTrainingData:
    if isinstance(pool, PreparedTrainingData):
        if (
            label is not None
            or weight is not None
            or group_id is not None
            or group_weight is not None
            or subgroup_id is not None
            or baseline is not None
            or pairs is not None
            or pairs_weight is not None
            or feature_names is not None
            or column_roles is not None
            or feature_metadata is not None
            or categorical_schema is not None
        ):
            raise ValueError(
                "label, weight, group_id, group_weight, subgroup_id, baseline, pairs, pairs_weight, "
                "feature_names, column_roles, feature_metadata, and categorical_schema cannot be passed "
                "when pool is PreparedTrainingData"
            )
        if eval_set is not None or eval_names is not None:
            raise ValueError("eval_set and eval_names cannot be passed when pool is PreparedTrainingData")
        return pool

    config = dict(params)
    distributed_config = _normalize_distributed_config(config)
    init_pipeline = _resolve_init_model_pipeline(init_model)
    shared_feature_pipeline = init_pipeline

    if isinstance(pool, Pool):
        if (
            label is not None
            or weight is not None
            or group_id is not None
            or group_weight is not None
            or subgroup_id is not None
            or baseline is not None
            or pairs is not None
            or pairs_weight is not None
            or feature_names is not None
            or column_roles is not None
            or feature_metadata is not None
            or categorical_schema is not None
        ):
            pool = _pool_from_data_and_label(
                pool,
                label,
                group_id=group_id,
                weight=weight,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                baseline=baseline,
                pairs=pairs,
                pairs_weight=pairs_weight,
                feature_names=feature_names,
                column_roles=column_roles,
                feature_metadata=feature_metadata,
                categorical_schema=categorical_schema,
            )
    else:
        if label is None:
            raise ValueError("label must be provided when training data is not a Pool")
        if init_pipeline is not None:
            if uses_feature_pipeline_params(config):
                _assert_compatible_feature_pipeline(init_pipeline, config)
            shared_feature_pipeline = init_pipeline
            pool = _prepare_pool_from_raw(
                pool,
                label,
                group_id=group_id,
                weight=weight,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                baseline=baseline,
                pairs=pairs,
                pairs_weight=pairs_weight,
                feature_names=feature_names,
                column_roles=column_roles,
                feature_metadata=feature_metadata,
                categorical_schema=categorical_schema,
                params=config,
                feature_pipeline=shared_feature_pipeline,
                fit_feature_pipeline=False,
            )
        elif distributed_config is not None and uses_feature_pipeline_params(config):
            if distributed_config["backend"] == "tcp":
                with _distributed_collective_context(distributed_config):
                    pool, shared_feature_pipeline = _prepare_distributed_feature_pipeline_pool(
                        pool,
                        label,
                        group_id=group_id,
                        weight=weight,
                        group_weight=group_weight,
                        subgroup_id=subgroup_id,
                        baseline=baseline,
                        pairs=pairs,
                        pairs_weight=pairs_weight,
                        feature_names=feature_names,
                        column_roles=column_roles,
                        feature_metadata=feature_metadata,
                        categorical_schema=categorical_schema,
                        params=config,
                        distributed=distributed_config,
                    )
            else:
                pool, shared_feature_pipeline = _prepare_distributed_feature_pipeline_pool(
                    pool,
                    label,
                    group_id=group_id,
                    weight=weight,
                    group_weight=group_weight,
                    subgroup_id=subgroup_id,
                    baseline=baseline,
                    pairs=pairs,
                    pairs_weight=pairs_weight,
                    feature_names=feature_names,
                    column_roles=column_roles,
                    feature_metadata=feature_metadata,
                    categorical_schema=categorical_schema,
                    params=config,
                    distributed=distributed_config,
                )
        else:
            pool = _prepare_pool_from_raw(
                pool,
                label,
                group_id=group_id,
                weight=weight,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                baseline=baseline,
                pairs=pairs,
                pairs_weight=pairs_weight,
                feature_names=feature_names,
                column_roles=column_roles,
                feature_metadata=feature_metadata,
                categorical_schema=categorical_schema,
                params=config,
            )
    if not isinstance(pool, Pool):
        raise TypeError("pool must be an instance of ctboost.Pool")

    feature_pipeline = getattr(pool, "_feature_pipeline", None) or shared_feature_pipeline
    normalized_eval_sets = _normalize_eval_sets(eval_set)
    resolved_eval_names = _normalize_eval_names(eval_names, len(normalized_eval_sets))
    eval_pools: List[Pool] = []
    for normalized_eval_set in normalized_eval_sets:
        if isinstance(normalized_eval_set, Pool):
            eval_pools.append(normalized_eval_set)
            continue
        if len(normalized_eval_set) == 2:
            X_eval, y_eval = normalized_eval_set
            eval_group_id = None
        else:
            X_eval, y_eval, eval_group_id = normalized_eval_set
        if feature_pipeline is not None:
            eval_pools.append(
                _prepare_pool_from_raw(
                    X_eval,
                    y_eval,
                    group_id=eval_group_id,
                    params=config,
                    feature_pipeline=feature_pipeline,
                    fit_feature_pipeline=False,
                    use_eval_external_memory=True,
                )
            )
        else:
            eval_pools.append(
                _prepare_pool_from_raw(
                    X_eval,
                    y_eval,
                    group_id=eval_group_id,
                    params=config,
                    use_eval_external_memory=True,
                )
            )

    return PreparedTrainingData(
        pool=pool,
        eval_pools=eval_pools,
        eval_names=resolved_eval_names,
        feature_pipeline=feature_pipeline,
    )

