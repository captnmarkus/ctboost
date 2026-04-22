"""Feature-pipeline preparation for distributed training."""

from __future__ import annotations

from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..core import Pool
from ..feature_pipeline import FeaturePipeline
from ._distributed_config import (
    _feature_pipeline_from_params,
    _merge_raw_feature_pipeline_shards,
    _serialize_raw_feature_pipeline_shard,
)
from ._distributed_payloads import _distributed_allgather_value

def _prepare_transformed_training_pool(
    data: Any,
    label: Any,
    *,
    cat_features: List[int],
    weight: Any,
    group_id: Any,
    group_weight: Any,
    subgroup_id: Any,
    baseline: Any,
    pairs: Any,
    pairs_weight: Any,
    feature_names: Optional[List[str]],
    column_roles: Any,
    feature_metadata: Optional[Mapping[str, Any]],
    categorical_schema: Optional[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> Pool:
    del params
    return Pool(
        data=np.asarray(data, dtype=np.float32),
        label=np.asarray(label, dtype=np.float32).reshape(-1),
        cat_features=list(cat_features),
        weight=weight,
        group_id=group_id,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=None if feature_names is None else list(feature_names),
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        _releasable_feature_storage=True,
    )

def _prepare_distributed_feature_pipeline_pool(
    data: Any,
    label: Any,
    *,
    group_id: Any,
    weight: Any,
    group_weight: Any,
    subgroup_id: Any,
    baseline: Any,
    pairs: Any,
    pairs_weight: Any,
    feature_names: Optional[List[str]],
    column_roles: Any,
    feature_metadata: Optional[Mapping[str, Any]],
    categorical_schema: Optional[Mapping[str, Any]],
    params: Mapping[str, Any],
    distributed: Dict[str, Any],
) -> tuple[Pool, FeaturePipeline]:
    local_shard = _serialize_raw_feature_pipeline_shard(data, label, feature_names)
    if distributed["backend"] == "tcp":
        shard_payloads = _distributed_allgather_value(
            distributed,
            f"{distributed['run_id']}/feature_pipeline/fit",
            local_shard,
        )
    else:
        run_root = Path(distributed["root"]) / distributed["run_id"]
        run_root.mkdir(parents=True, exist_ok=True)
        shard_path = run_root / f"feature_pipeline_rank_{distributed['rank']:05d}.pkl"
        with shard_path.open("wb") as stream:
            pickle.dump(local_shard, stream, protocol=pickle.HIGHEST_PROTOCOL)

        shard_paths = [
            run_root / f"feature_pipeline_rank_{rank:05d}.pkl"
            for rank in range(distributed["world_size"])
        ]
        deadline = time.monotonic() + float(distributed["timeout"])
        while any(not path.exists() for path in shard_paths):
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for distributed feature-pipeline shards")
            time.sleep(0.1)
        shard_payloads = []
        for path in shard_paths:
            with path.open("rb") as stream:
                shard_payloads.append(pickle.load(stream))

    merged = _merge_raw_feature_pipeline_shards(shard_payloads)
    row_counts = [int(np.asarray(shard["data"], dtype=object).shape[0]) for shard in shard_payloads]
    row_offsets = np.cumsum([0, *row_counts])
    pipeline = _feature_pipeline_from_params(params)
    transformed_data, transformed_cat_features, transformed_feature_names = pipeline.fit_transform_array(
        merged["data"],
        merged["label"],
        feature_names=merged["feature_names"],
    )
    row_begin = int(row_offsets[distributed["rank"]])
    row_end = int(row_offsets[distributed["rank"] + 1])
    local_pool = _prepare_transformed_training_pool(
        transformed_data[row_begin:row_end],
        np.asarray(label, dtype=np.float32).reshape(-1),
        cat_features=list(transformed_cat_features),
        weight=weight,
        group_id=group_id,
        group_weight=group_weight,
        subgroup_id=subgroup_id,
        baseline=baseline,
        pairs=pairs,
        pairs_weight=pairs_weight,
        feature_names=list(transformed_feature_names),
        column_roles=column_roles,
        feature_metadata=feature_metadata,
        categorical_schema=categorical_schema,
        params=params,
    )
    local_pool._feature_pipeline = pipeline
    return local_pool, pipeline

