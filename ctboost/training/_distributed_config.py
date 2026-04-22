"""Distributed-config helpers for ctboost.training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..distributed import parse_distributed_root
from ..feature_pipeline import FeaturePipeline

_DISTRIBUTED_PARAM_KEYS = {
    "distributed_world_size",
    "distributed_rank",
    "distributed_root",
    "distributed_run_id",
    "distributed_timeout",
}

def _normalize_distributed_config(params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    world_size = int(params.get("distributed_world_size", 1) or 1)
    if world_size <= 1:
        return None

    rank = int(params.get("distributed_rank", 0) or 0)
    if rank < 0 or rank >= world_size:
        raise ValueError("distributed_rank must be in [0, distributed_world_size)")

    root_value = params.get("distributed_root")
    if root_value is None:
        raise ValueError("distributed_root is required when distributed_world_size > 1")
    parsed_root = parse_distributed_root(root_value)
    run_id = str(params.get("distributed_run_id", "default"))
    timeout = float(params.get("distributed_timeout", 600.0))
    if timeout <= 0.0:
        raise ValueError("distributed_timeout must be positive")

    return {
        "world_size": world_size,
        "rank": rank,
        "root": parsed_root.root,
        "backend": parsed_root.backend,
        "host": parsed_root.host,
        "port": parsed_root.port,
        "run_id": run_id,
        "timeout": timeout,
    }

def _strip_distributed_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in params.items() if key not in _DISTRIBUTED_PARAM_KEYS}

def _feature_pipeline_from_params(params: Mapping[str, Any]) -> FeaturePipeline:
    return FeaturePipeline(
        cat_features=params.get("cat_features"),
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
    )

def _serialize_raw_feature_pipeline_shard(
    data: Any,
    label: Any,
    feature_names: Optional[List[str]],
) -> Dict[str, Any]:
    raw_matrix, resolved_feature_names = FeaturePipeline._extract_frame(data, feature_names)
    return {
        "data": np.asarray(raw_matrix, dtype=object),
        "label": np.asarray(label, dtype=np.float32).reshape(-1),
        "feature_names": None if resolved_feature_names is None else list(resolved_feature_names),
    }

def _merge_raw_feature_pipeline_shards(shards: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not shards:
        raise ValueError("distributed feature-pipeline fitting requires at least one shard")
    feature_names = shards[0]["feature_names"]
    num_cols = int(np.asarray(shards[0]["data"], dtype=object).shape[1])
    for shard in shards[1:]:
        shard_data = np.asarray(shard["data"], dtype=object)
        if int(shard_data.shape[1]) != num_cols:
            raise ValueError("distributed feature-pipeline shards must agree on column count")
        if shard["feature_names"] != feature_names:
            raise ValueError("distributed feature-pipeline shards must agree on feature_names")
    return {
        "data": np.concatenate([np.asarray(shard["data"], dtype=object) for shard in shards], axis=0),
        "label": np.concatenate(
            [np.asarray(shard["label"], dtype=np.float32).reshape(-1) for shard in shards],
            axis=0,
        ),
        "feature_names": feature_names,
    }

