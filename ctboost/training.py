"""Training entry points for the Python API."""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from . import _core
from ._export import export_model as _export_model
from ._serialization import load_booster_document, save_booster
from .core import Pool, _clone_pool
from .distributed import (
    distributed_tcp_request,
    parse_distributed_root,
    pickle_payload,
    wait_for_distributed_tcp_coordinator,
)
from .feature_pipeline import FeaturePipeline
from .prepared_data import prepare_pool, uses_feature_pipeline_params

PathLike = Union[str, Path]


@dataclass
class TrainingCallbackEnv:
    model: "Booster"
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: List[tuple[str, str, float, bool]]
    history: Dict[str, Dict[str, List[float]]]
    best_iteration: int
    best_score: Optional[float]


@dataclass
class PreparedTrainingData:
    pool: Pool
    eval_pools: List[Pool]
    eval_names: List[str]
    feature_pipeline: Optional[FeaturePipeline] = None


def log_evaluation(period: int = 1) -> Callable[[TrainingCallbackEnv], bool]:
    resolved_period = int(period)
    if resolved_period <= 0:
        raise ValueError("log_evaluation period must be positive")

    def _callback(env: TrainingCallbackEnv) -> bool:
        if (env.iteration + 1) % resolved_period != 0 and env.iteration + 1 != env.end_iteration:
            return False
        metrics = " ".join(
            f"{dataset_name}-{metric_name}:{metric_value:.6f}"
            for dataset_name, metric_name, metric_value, _ in env.evaluation_result_list
        )
        print(f"[{env.iteration + 1}] {metrics}")
        return False

    return _callback


def checkpoint_callback(
    path: PathLike,
    *,
    interval: int = 1,
    model_format: Optional[str] = None,
    save_best_only: bool = False,
) -> Callable[[TrainingCallbackEnv], bool]:
    resolved_interval = int(interval)
    if resolved_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    destination = Path(path)

    def _callback(env: TrainingCallbackEnv) -> bool:
        if save_best_only:
            if env.best_iteration != env.iteration:
                return False
        elif (env.iteration + 1) % resolved_interval != 0 and env.iteration + 1 != env.end_iteration:
            return False
        env.model.save_model(destination, model_format=model_format)
        return False

    return _callback


def _load_sklearn_splitters():
    try:
        from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
    except ModuleNotFoundError as exc:
        raise ImportError(
            "ctboost.cv requires scikit-learn>=1.3. "
            "Install 'ctboost[sklearn]' or add scikit-learn to your environment."
        ) from exc
    return GroupKFold, KFold, StratifiedKFold


def _pool_from_data_and_label(
    data: Any,
    label: Any,
    group_id: Any = None,
    *,
    weight: Any = None,
    feature_names: Optional[List[str]] = None,
) -> Pool:
    if isinstance(data, Pool):
        if label is None and group_id is None and weight is None and feature_names is None:
            return data
        resolved_label = data.label if label is None else label
        resolved_weight = data.weight if weight is None else weight
        resolved_group_id = data.group_id if group_id is None else group_id
        return _clone_pool(
            data,
            label=resolved_label,
            weight=resolved_weight,
            group_id=resolved_group_id,
            feature_names=data.feature_names if feature_names is None else feature_names,
            releasable_feature_storage=True,
        )

    if label is None:
        raise ValueError("y must be provided when data is not a Pool")
    return Pool(
        data=data,
        label=label,
        weight=weight,
        group_id=group_id,
        feature_names=feature_names,
        _releasable_feature_storage=True,
    )


def _is_eval_set_entry(value: Any) -> bool:
    return isinstance(value, Pool) or (isinstance(value, tuple) and len(value) in {2, 3})


def _normalize_eval_sets(eval_set: Any) -> List[Any]:
    if eval_set is None:
        return []
    if _is_eval_set_entry(eval_set):
        return [eval_set]
    if isinstance(eval_set, list) and all(_is_eval_set_entry(item) for item in eval_set):
        return list(eval_set)
    raise TypeError(
        "eval_set must be a Pool, a single (X, y) or (X, y, group_id) tuple, "
        "or a list of such entries"
    )


def _normalize_eval_set(eval_set: Any) -> Optional[Any]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    if len(normalized) != 1:
        raise TypeError("expected a single eval_set entry")
    return normalized[0]


def _normalize_eval_names(eval_names: Any, eval_count: int) -> List[str]:
    if eval_count == 0:
        if eval_names is not None and eval_names != [] and eval_names != ():
            raise ValueError("eval_names requires at least one eval_set")
        return []
    if eval_names is None:
        if eval_count == 1:
            return ["validation"]
        return [f"validation_{index}" for index in range(eval_count)]
    if isinstance(eval_names, str):
        if eval_count != 1:
            raise ValueError("eval_names must provide one name per eval_set")
        names = [eval_names]
    elif isinstance(eval_names, (list, tuple)):
        if len(eval_names) != eval_count:
            raise ValueError("eval_names must provide one name per eval_set")
        names = [str(name) for name in eval_names]
    else:
        raise TypeError("eval_names must be a string or a sequence of strings")
    if len(set(names)) != len(names):
        raise ValueError("eval_names entries must be unique")
    return names


def _normalize_eval_metrics(eval_metric: Any) -> List[str]:
    if eval_metric is None:
        return []
    if isinstance(eval_metric, str):
        metric_name = eval_metric.strip()
        return [] if not metric_name else [metric_name]
    if isinstance(eval_metric, (list, tuple)):
        metrics = [str(metric).strip() for metric in eval_metric]
        if not metrics or any(not metric for metric in metrics):
            raise ValueError("eval_metric sequences must contain non-empty metric names")
        return metrics
    raise TypeError("eval_metric must be a string or a sequence of strings")


def _copy_evals_result(result: Mapping[str, Mapping[str, Iterable[float]]]) -> Dict[str, Dict[str, List[float]]]:
    return {
        str(dataset_name): {
            str(metric_name): [float(value) for value in metric_history]
            for metric_name, metric_history in dataset_metrics.items()
        }
        for dataset_name, dataset_metrics in result.items()
    }


def _unpack_gathered_payloads(payload: bytes) -> List[bytes]:
    view = memoryview(payload)
    if len(view) < 8:
        raise RuntimeError("distributed allgather payload is truncated")
    (count,) = struct.unpack_from("<Q", view, 0)
    offset = 8
    payloads: List[bytes] = []
    for _ in range(int(count)):
        if offset + 8 > len(view):
            raise RuntimeError("distributed allgather payload is truncated")
        (payload_size,) = struct.unpack_from("<Q", view, offset)
        offset += 8
        next_offset = offset + int(payload_size)
        if next_offset > len(view):
            raise RuntimeError("distributed allgather payload is truncated")
        payloads.append(bytes(view[offset:next_offset]))
        offset = next_offset
    return payloads


def _distributed_allgather_value(distributed: Dict[str, Any], key: str, value: Any) -> List[Any]:
    response = distributed_tcp_request(
        distributed["root"],
        distributed["timeout"],
        "allgather",
        key,
        distributed["rank"],
        distributed["world_size"],
        pickle_payload(value),
    )
    return [pickle.loads(item) for item in _unpack_gathered_payloads(response)]


def _distributed_broadcast_value(
    distributed: Dict[str, Any],
    key: str,
    *,
    value: Any = None,
) -> Any:
    payload = pickle_payload(value) if distributed["rank"] == 0 else b""
    response = distributed_tcp_request(
        distributed["root"],
        distributed["timeout"],
        "broadcast",
        key,
        distributed["rank"],
        distributed["world_size"],
        payload,
    )
    return pickle.loads(response) if response else None


def _resolve_eval_pool(eval_set: Any) -> Optional[Pool]:
    normalized = _normalize_eval_set(eval_set)
    if normalized is None or isinstance(normalized, Pool):
        return normalized

    if len(normalized) == 2:
        X_eval, y_eval = normalized
        return _pool_from_data_and_label(X_eval, y_eval)

    X_eval, y_eval, group_id = normalized
    return _pool_from_data_and_label(X_eval, y_eval, group_id=group_id)


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
    feature_names: Optional[List[str]] = None,
    params: Mapping[str, Any],
    feature_pipeline: Optional[FeaturePipeline] = None,
    fit_feature_pipeline: bool = True,
    use_eval_external_memory: bool = False,
) -> Pool:
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
        feature_names=feature_names,
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


def _prepare_transformed_training_pool(
    data: Any,
    label: Any,
    *,
    cat_features: List[int],
    weight: Any,
    group_id: Any,
    feature_names: Optional[List[str]],
    params: Mapping[str, Any],
) -> Pool:
    del params
    return Pool(
        data=np.asarray(data, dtype=np.float32),
        label=np.asarray(label, dtype=np.float32).reshape(-1),
        cat_features=list(cat_features),
        weight=weight,
        group_id=group_id,
        feature_names=None if feature_names is None else list(feature_names),
        _releasable_feature_storage=True,
    )


def _prepare_distributed_feature_pipeline_pool(
    data: Any,
    label: Any,
    *,
    group_id: Any,
    weight: Any,
    feature_names: Optional[List[str]],
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
        feature_names=list(transformed_feature_names),
        params=params,
    )
    local_pool._feature_pipeline = pipeline
    return local_pool, pipeline


def _open_memmap_array(
    path: Path,
    shape: tuple[int, ...],
    *,
    dtype: Any,
    fortran_order: bool = False,
) -> np.memmap:
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=shape,
        fortran_order=fortran_order,
    )


def _wait_for_path(path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for distributed artifact: {path}")
        time.sleep(0.1)


def _wait_for_manifests(run_root: Path, world_size: int, timeout: float) -> List[Path]:
    deadline = time.monotonic() + timeout
    manifest_paths = [run_root / f"rank_{rank:05d}" / "manifest.json" for rank in range(world_size)]
    while True:
        missing = [path for path in manifest_paths if not path.exists()]
        if not missing:
            return manifest_paths
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for distributed shard manifests: "
                + ", ".join(str(path) for path in missing)
            )
        time.sleep(0.1)


def _write_distributed_shard(pool: Pool, config: Dict[str, Any]) -> Path:
    run_root = Path(config["root"]) / config["run_id"]
    shard_root = run_root / f"rank_{config['rank']:05d}"
    shard_root.mkdir(parents=True, exist_ok=True)

    sparse_components = getattr(pool, "_sparse_csc_components", None)
    label = np.asarray(pool.label, dtype=np.float32)
    weight = np.ones(pool.num_rows, dtype=np.float32) if pool.weight is None else np.asarray(pool.weight, dtype=np.float32)
    group_id = None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64)

    label_map = _open_memmap_array(shard_root / "label.npy", label.shape, dtype=np.float32)
    label_map[...] = label
    label_map.flush()
    weight_map = _open_memmap_array(shard_root / "weight.npy", weight.shape, dtype=np.float32)
    weight_map[...] = weight
    weight_map.flush()

    group_path = None
    if group_id is not None:
        group_path = shard_root / "group_id.npy"
        group_map = _open_memmap_array(group_path, group_id.shape, dtype=np.int64)
        group_map[...] = group_id
        group_map.flush()

    manifest: Dict[str, Any] = {
        "rank": int(config["rank"]),
        "num_rows": int(pool.num_rows),
        "num_cols": int(pool.num_cols),
        "cat_features": list(pool.cat_features),
        "feature_names": None if pool.feature_names is None else list(pool.feature_names),
        "label_path": "label.npy",
        "weight_path": "weight.npy",
        "group_id_path": None if group_path is None else group_path.name,
    }

    if sparse_components is None:
        dense_data = getattr(pool, "_dense_data_ref", None)
        if dense_data is None:
            dense_data = np.asfortranarray(pool.data, dtype=np.float32)
        else:
            dense_data = np.asfortranarray(dense_data, dtype=np.float32)
        data_map = _open_memmap_array(
            shard_root / "data.npy",
            dense_data.shape,
            dtype=np.float32,
            fortran_order=True,
        )
        data_map[...] = dense_data
        data_map.flush()
        manifest["storage"] = "dense"
        manifest["data_path"] = "data.npy"
    else:
        sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
        data_map = _open_memmap_array(
            shard_root / "sparse_data.npy", sparse_data.shape, dtype=np.float32
        )
        data_map[...] = np.asarray(sparse_data, dtype=np.float32)
        data_map.flush()
        indices_map = _open_memmap_array(
            shard_root / "sparse_indices.npy", sparse_indices.shape, dtype=np.int64
        )
        indices_map[...] = np.asarray(sparse_indices, dtype=np.int64)
        indices_map.flush()
        indptr_map = _open_memmap_array(
            shard_root / "sparse_indptr.npy", sparse_indptr.shape, dtype=np.int64
        )
        indptr_map[...] = np.asarray(sparse_indptr, dtype=np.int64)
        indptr_map.flush()
        manifest["storage"] = "sparse"
        manifest["shape"] = [int(shape[0]), int(shape[1])]
        manifest["sparse_data_path"] = "sparse_data.npy"
        manifest["sparse_indices_path"] = "sparse_indices.npy"
        manifest["sparse_indptr_path"] = "sparse_indptr.npy"

    manifest_path = shard_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return manifest_path


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _merge_dense_distributed_shards(
    run_root: Path,
    manifests: List[Dict[str, Any]],
) -> Pool:
    total_rows = sum(int(manifest["num_rows"]) for manifest in manifests)
    num_cols = int(manifests[0]["num_cols"])
    merged_root = run_root / "merged_dense"
    merged_root.mkdir(parents=True, exist_ok=True)

    data_map = _open_memmap_array(
        merged_root / "data.npy",
        (total_rows, num_cols),
        dtype=np.float32,
        fortran_order=True,
    )
    label_map = _open_memmap_array(merged_root / "label.npy", (total_rows,), dtype=np.float32)
    weight_map = _open_memmap_array(merged_root / "weight.npy", (total_rows,), dtype=np.float32)

    has_group_id = any(manifest.get("group_id_path") is not None for manifest in manifests)
    group_map = (
        _open_memmap_array(merged_root / "group_id.npy", (total_rows,), dtype=np.int64)
        if has_group_id
        else None
    )
    group_arrays = []

    row_offset = 0
    for manifest in manifests:
        shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
        shard_rows = int(manifest["num_rows"])
        next_offset = row_offset + shard_rows
        shard_data = np.load(shard_root / manifest["data_path"], mmap_mode="r")
        data_map[row_offset:next_offset, :] = np.asarray(shard_data, dtype=np.float32)
        label_map[row_offset:next_offset] = np.load(shard_root / manifest["label_path"], mmap_mode="r")
        weight_map[row_offset:next_offset] = np.load(shard_root / manifest["weight_path"], mmap_mode="r")
        if group_map is not None:
            group_path = manifest.get("group_id_path")
            if group_path is None:
                raise ValueError("distributed shards must either all include group_id or all omit it")
            shard_groups = np.load(shard_root / group_path, mmap_mode="r")
            group_arrays.append(np.asarray(shard_groups, dtype=np.int64))
            group_map[row_offset:next_offset] = shard_groups
        row_offset = next_offset

    data_map.flush()
    label_map.flush()
    weight_map.flush()
    if group_map is not None:
        _validate_distributed_group_shards(group_arrays)
        group_map.flush()

    return Pool(
        data=np.load(merged_root / "data.npy", mmap_mode="r"),
        label=np.load(merged_root / "label.npy", mmap_mode="r"),
        cat_features=list(manifests[0]["cat_features"]),
        weight=np.load(merged_root / "weight.npy", mmap_mode="r"),
        group_id=None
        if group_map is None
        else np.load(merged_root / "group_id.npy", mmap_mode="r"),
        feature_names=manifests[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_sparse_distributed_shards(
    run_root: Path,
    manifests: List[Dict[str, Any]],
) -> Pool:
    total_rows = sum(int(manifest["num_rows"]) for manifest in manifests)
    num_cols = int(manifests[0]["num_cols"])
    merged_root = run_root / "merged_sparse"
    merged_root.mkdir(parents=True, exist_ok=True)

    total_nnz = 0
    shard_payloads = []
    for manifest in manifests:
        shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
        sparse_data = np.load(shard_root / manifest["sparse_data_path"], mmap_mode="r")
        sparse_indices = np.load(shard_root / manifest["sparse_indices_path"], mmap_mode="r")
        sparse_indptr = np.load(shard_root / manifest["sparse_indptr_path"], mmap_mode="r")
        total_nnz += int(sparse_data.shape[0])
        shard_payloads.append((manifest, sparse_data, sparse_indices, sparse_indptr))

    data_map = _open_memmap_array(merged_root / "sparse_data.npy", (total_nnz,), dtype=np.float32)
    indices_map = _open_memmap_array(
        merged_root / "sparse_indices.npy", (total_nnz,), dtype=np.int64
    )
    indptr_map = _open_memmap_array(
        merged_root / "sparse_indptr.npy", (num_cols + 1,), dtype=np.int64
    )
    label_map = _open_memmap_array(merged_root / "label.npy", (total_rows,), dtype=np.float32)
    weight_map = _open_memmap_array(merged_root / "weight.npy", (total_rows,), dtype=np.float32)

    has_group_id = any(manifest.get("group_id_path") is not None for manifest in manifests)
    group_map = (
        _open_memmap_array(merged_root / "group_id.npy", (total_rows,), dtype=np.int64)
        if has_group_id
        else None
    )
    group_arrays = []

    row_offset = 0
    for manifest, _sparse_data, _sparse_indices, _sparse_indptr in shard_payloads:
        shard_root = run_root / f"rank_{int(manifest['rank']):05d}"
        shard_rows = int(manifest["num_rows"])
        next_offset = row_offset + shard_rows
        label_map[row_offset:next_offset] = np.load(shard_root / manifest["label_path"], mmap_mode="r")
        weight_map[row_offset:next_offset] = np.load(shard_root / manifest["weight_path"], mmap_mode="r")
        if group_map is not None:
            group_path = manifest.get("group_id_path")
            if group_path is None:
                raise ValueError("distributed shards must either all include group_id or all omit it")
            shard_groups = np.load(shard_root / group_path, mmap_mode="r")
            group_arrays.append(np.asarray(shard_groups, dtype=np.int64))
            group_map[row_offset:next_offset] = shard_groups
        row_offset = next_offset

    global_offset = 0
    indptr_map[0] = 0
    row_base = [0]
    for manifest in manifests[:-1]:
        row_base.append(row_base[-1] + int(manifest["num_rows"]))
    for col in range(num_cols):
        for shard_index, (_manifest, sparse_data, sparse_indices, sparse_indptr) in enumerate(shard_payloads):
            begin = int(sparse_indptr[col])
            end = int(sparse_indptr[col + 1])
            nnz = end - begin
            if nnz == 0:
                continue
            next_offset = global_offset + nnz
            data_map[global_offset:next_offset] = sparse_data[begin:end]
            indices_map[global_offset:next_offset] = sparse_indices[begin:end] + row_base[shard_index]
            global_offset = next_offset
        indptr_map[col + 1] = global_offset

    data_map.flush()
    indices_map.flush()
    indptr_map.flush()
    label_map.flush()
    weight_map.flush()
    if group_map is not None:
        _validate_distributed_group_shards(group_arrays)
        group_map.flush()

    return Pool.from_csc_components(
        np.load(merged_root / "sparse_data.npy", mmap_mode="r"),
        np.load(merged_root / "sparse_indices.npy", mmap_mode="r"),
        np.load(merged_root / "sparse_indptr.npy", mmap_mode="r"),
        (total_rows, num_cols),
        np.load(merged_root / "label.npy", mmap_mode="r"),
        cat_features=list(manifests[0]["cat_features"]),
        weight=np.load(merged_root / "weight.npy", mmap_mode="r"),
        group_id=None
        if group_map is None
        else np.load(merged_root / "group_id.npy", mmap_mode="r"),
        feature_names=manifests[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_distributed_shards(run_root: Path, manifests: List[Dict[str, Any]]) -> Pool:
    if not manifests:
        raise ValueError("distributed training requires at least one shard manifest")

    storage_kinds = {str(manifest["storage"]) for manifest in manifests}
    if len(storage_kinds) != 1:
        raise ValueError("distributed training requires all shards to use the same storage kind")

    num_cols = int(manifests[0]["num_cols"])
    cat_features = list(manifests[0]["cat_features"])
    feature_names = manifests[0]["feature_names"]
    for manifest in manifests[1:]:
        if int(manifest["num_cols"]) != num_cols:
            raise ValueError("distributed shards must agree on num_cols")
        if list(manifest["cat_features"]) != cat_features:
            raise ValueError("distributed shards must agree on cat_features")
        if manifest["feature_names"] != feature_names:
            raise ValueError("distributed shards must agree on feature_names")

    if storage_kinds == {"dense"}:
        return _merge_dense_distributed_shards(run_root, manifests)
    return _merge_sparse_distributed_shards(run_root, manifests)


def _quantization_schema_from_debug_histogram(hist_state: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "num_bins_per_feature": list(hist_state["num_bins_per_feature"]),
        "cut_offsets": list(hist_state["cut_offsets"]),
        "cut_values": list(hist_state["cut_values"]),
        "categorical_mask": list(hist_state["categorical_mask"]),
        "missing_value_mask": list(hist_state["missing_value_mask"]),
        "nan_mode": int(hist_state["nan_mode"]),
        "nan_modes": list(hist_state["nan_modes"]),
    }


def _validate_distributed_group_shards(group_arrays: List[np.ndarray]) -> None:
    if not group_arrays:
        return
    seen_groups = set()
    for shard_groups in group_arrays:
        unique_groups = {int(value) for value in np.unique(np.asarray(shard_groups, dtype=np.int64))}
        overlap = seen_groups.intersection(unique_groups)
        if overlap:
            raise ValueError(
                "distributed ranking/group training requires each group_id to live on exactly one rank"
            )
        seen_groups.update(unique_groups)


def _serialize_distributed_pool_shard(pool: Pool) -> Dict[str, Any]:
    sparse_components = getattr(pool, "_sparse_csc_components", None)
    payload: Dict[str, Any] = {
        "num_rows": int(pool.num_rows),
        "num_cols": int(pool.num_cols),
        "cat_features": list(pool.cat_features),
        "feature_names": None if pool.feature_names is None else list(pool.feature_names),
        "label": np.asarray(pool.label, dtype=np.float32),
        "weight": np.ones(pool.num_rows, dtype=np.float32)
        if pool.weight is None
        else np.asarray(pool.weight, dtype=np.float32),
        "group_id": None if pool.group_id is None else np.asarray(pool.group_id, dtype=np.int64),
    }
    if sparse_components is None:
        dense_data = getattr(pool, "_dense_data_ref", None)
        if dense_data is None:
            dense_data = np.asfortranarray(pool.data, dtype=np.float32)
        payload["storage"] = "dense"
        payload["data"] = np.asfortranarray(dense_data, dtype=np.float32)
    else:
        sparse_data, sparse_indices, sparse_indptr, shape = sparse_components
        payload["storage"] = "sparse"
        payload["shape"] = (int(shape[0]), int(shape[1]))
        payload["sparse_data"] = np.asarray(sparse_data, dtype=np.float32)
        payload["sparse_indices"] = np.asarray(sparse_indices, dtype=np.int64)
        payload["sparse_indptr"] = np.asarray(sparse_indptr, dtype=np.int64)
    return payload


def _merge_dense_distributed_payloads(shards: List[Dict[str, Any]]) -> Pool:
    data = np.asfortranarray(
        np.concatenate([np.asarray(shard["data"], dtype=np.float32) for shard in shards], axis=0),
        dtype=np.float32,
    )
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_distributed_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)
    return Pool(
        data=data,
        label=label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_sparse_distributed_payloads(shards: List[Dict[str, Any]]) -> Pool:
    total_rows = sum(int(shard["num_rows"]) for shard in shards)
    num_cols = int(shards[0]["num_cols"])
    total_nnz = sum(int(np.asarray(shard["sparse_data"]).shape[0]) for shard in shards)
    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int64)
    indptr = np.empty(num_cols + 1, dtype=np.int64)
    label = np.concatenate([np.asarray(shard["label"], dtype=np.float32) for shard in shards], axis=0)
    weight = np.concatenate([np.asarray(shard["weight"], dtype=np.float32) for shard in shards], axis=0)
    has_group_id = any(shard["group_id"] is not None for shard in shards)
    if has_group_id and any(shard["group_id"] is None for shard in shards):
        raise ValueError("distributed shards must either all include group_id or all omit it")
    group_id = None
    if has_group_id:
        group_arrays = [np.asarray(shard["group_id"], dtype=np.int64) for shard in shards]
        _validate_distributed_group_shards(group_arrays)
        group_id = np.concatenate(group_arrays, axis=0)

    row_bases = [0]
    for shard in shards[:-1]:
        row_bases.append(row_bases[-1] + int(shard["num_rows"]))

    nnz_offset = 0
    indptr[0] = 0
    for col in range(num_cols):
        for shard_index, shard in enumerate(shards):
            sparse_data = np.asarray(shard["sparse_data"], dtype=np.float32)
            sparse_indices = np.asarray(shard["sparse_indices"], dtype=np.int64)
            sparse_indptr = np.asarray(shard["sparse_indptr"], dtype=np.int64)
            begin = int(sparse_indptr[col])
            end = int(sparse_indptr[col + 1])
            nnz = end - begin
            if nnz <= 0:
                continue
            next_offset = nnz_offset + nnz
            data[nnz_offset:next_offset] = sparse_data[begin:end]
            indices[nnz_offset:next_offset] = sparse_indices[begin:end] + row_bases[shard_index]
            nnz_offset = next_offset
        indptr[col + 1] = nnz_offset

    return Pool.from_csc_components(
        data,
        indices,
        indptr,
        (total_rows, num_cols),
        label,
        cat_features=list(shards[0]["cat_features"]),
        weight=weight,
        group_id=group_id,
        feature_names=shards[0]["feature_names"],
        _releasable_feature_storage=True,
    )


def _merge_distributed_payloads(shards: List[Dict[str, Any]]) -> Pool:
    if not shards:
        raise ValueError("distributed training requires at least one shard payload")
    storage_kinds = {str(shard["storage"]) for shard in shards}
    if len(storage_kinds) != 1:
        raise ValueError("distributed shards must agree on storage kind")
    num_cols = int(shards[0]["num_cols"])
    cat_features = list(shards[0]["cat_features"])
    feature_names = shards[0]["feature_names"]
    for shard in shards[1:]:
        if int(shard["num_cols"]) != num_cols:
            raise ValueError("distributed shards must agree on num_cols")
        if list(shard["cat_features"]) != cat_features:
            raise ValueError("distributed shards must agree on cat_features")
        if shard["feature_names"] != feature_names:
            raise ValueError("distributed shards must agree on feature_names")
    if storage_kinds == {"dense"}:
        return _merge_dense_distributed_payloads(shards)
    return _merge_sparse_distributed_payloads(shards)


@contextlib.contextmanager
def _distributed_collective_context(distributed: Optional[Dict[str, Any]]):
    server_process = None
    if distributed is not None and distributed["backend"] == "tcp" and distributed["rank"] == 0:
        if distributed["host"] is None or distributed["port"] is None:
            raise ValueError("tcp distributed coordination requires a host and port")
        server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ctboost.distributed",
                str(distributed["host"]),
                str(distributed["port"]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wait_for_distributed_tcp_coordinator(
            distributed["root"],
            min(10.0, float(distributed["timeout"])),
        )
    try:
        yield
    finally:
        if distributed is not None and distributed["backend"] == "tcp":
            try:
                distributed_tcp_request(
                    distributed["root"],
                    distributed["timeout"],
                    "barrier",
                    f"{distributed['run_id']}/__shutdown__",
                    distributed["rank"],
                    distributed["world_size"],
                    b"",
                )
            except Exception:
                if server_process is None:
                    raise
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5.0)


def _resolve_distributed_quantization_schema(
    pool: Pool,
    distributed: Dict[str, Any],
    *,
    max_bins: int,
    nan_mode: str,
    max_bin_by_feature: List[int],
    border_selection_method: str,
    nan_mode_by_feature: List[str],
    feature_borders: List[List[float]],
    external_memory: bool,
) -> Dict[str, Any]:
    if distributed["backend"] == "tcp":
        payload = pickle_payload(
            {
                "shard": _serialize_distributed_pool_shard(pool),
                "schema_request": {
                    "max_bins": int(max_bins),
                    "nan_mode": str(nan_mode),
                    "max_bin_by_feature": list(max_bin_by_feature),
                    "border_selection_method": str(border_selection_method),
                    "nan_mode_by_feature": list(nan_mode_by_feature),
                    "feature_borders": list(feature_borders),
                    "external_memory": bool(external_memory),
                    "external_memory_dir": "",
                },
            }
        )
        response = distributed_tcp_request(
            distributed["root"],
            distributed["timeout"],
            "schema_collect",
            distributed["run_id"],
            distributed["rank"],
            distributed["world_size"],
            payload,
        )
        return json.loads(response.decode("utf-8"))

    run_root = Path(distributed["root"]) / distributed["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    _write_distributed_shard(pool, distributed)
    manifests = [
        _load_manifest(path)
        for path in _wait_for_manifests(
            run_root, distributed["world_size"], distributed["timeout"]
        )
    ]

    schema_path = run_root / "quantization_schema.json"
    if distributed["rank"] == 0:
        merged_pool = _merge_distributed_shards(run_root, manifests)
        schema_hist_dir = run_root / "schema_hist"
        hist_state = _core._debug_build_histogram(
            merged_pool._handle,
            max_bins=max_bins,
            nan_mode=nan_mode,
            max_bin_by_feature=max_bin_by_feature,
            border_selection_method=border_selection_method,
            nan_mode_by_feature=nan_mode_by_feature,
            feature_borders=feature_borders,
            external_memory=external_memory,
            external_memory_dir=str(schema_hist_dir),
        )
        schema_state = _quantization_schema_from_debug_histogram(hist_state)
        schema_temp_path = schema_path.with_suffix(".tmp")
        with schema_temp_path.open("w", encoding="utf-8") as stream:
            json.dump(schema_state, stream, indent=2, sort_keys=True)
        schema_temp_path.replace(schema_path)

    _wait_for_path(schema_path, distributed["timeout"])
    return _load_manifest(schema_path)


def _distributed_train(
    pool: Pool,
    config: Mapping[str, Any],
    iterations: int,
) -> "Booster":
    distributed = _normalize_distributed_config(config)
    if distributed is None:
        raise ValueError("distributed training was requested without a valid distributed config")
    if pool.group_id is not None:
        raise ValueError("distributed multi-host training does not yet support group_id/ranking data")

    run_root = distributed["root"] / distributed["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    _write_distributed_shard(pool, distributed)
    manifests = [_load_manifest(path) for path in _wait_for_manifests(run_root, distributed["world_size"], distributed["timeout"])]
    model_path = run_root / "model.ctboost"
    done_path = run_root / "done.txt"

    if distributed["rank"] == 0:
        merged_pool = _merge_distributed_shards(run_root, manifests)
        merged_params = _strip_distributed_params(config)
        booster = train(
            merged_pool,
            merged_params,
            num_boost_round=iterations,
            eval_set=None,
            early_stopping_rounds=None,
            init_model=None,
        )
        booster.save_model(model_path)
        done_path.write_text("ok\n", encoding="utf-8")
        return booster

    _wait_for_path(done_path, distributed["timeout"])
    _wait_for_path(model_path, distributed["timeout"])
    return Booster.load_model(model_path)


def _filesystem_distributed_phase_config(
    distributed: Dict[str, Any],
    phase_name: str,
) -> Dict[str, Any]:
    return {
        **distributed,
        "root": str(Path(distributed["root"]) / distributed["run_id"]),
        "run_id": phase_name,
    }


def _stage_filesystem_distributed_phase(
    pool: Pool,
    distributed: Dict[str, Any],
    phase_name: str,
) -> tuple[Path, List[Dict[str, Any]]]:
    phase_config = _filesystem_distributed_phase_config(distributed, phase_name)
    _write_distributed_shard(pool, phase_config)
    phase_root = Path(phase_config["root"]) / phase_config["run_id"]
    manifests = [
        _load_manifest(path)
        for path in _wait_for_manifests(
            phase_root,
            distributed["world_size"],
            distributed["timeout"],
        )
    ]
    return phase_root, manifests


def _distributed_train_filesystem_compat(
    pool: Pool,
    config: Mapping[str, Any],
    *,
    distributed: Dict[str, Any],
    iterations: int,
    eval_pools: List[Pool],
    eval_names: List[str],
    feature_pipeline: Optional[FeaturePipeline],
    early_stopping_rounds: Optional[int],
    early_stopping_metric: Optional[str],
    early_stopping_name: Optional[str],
    callbacks: List[Callable[[TrainingCallbackEnv], Any]],
    init_model: Any,
) -> "Booster":
    train_phase_root, train_manifests = _stage_filesystem_distributed_phase(
        pool,
        distributed,
        "filesystem_compat_train",
    )
    eval_phase_roots: List[Path] = []
    eval_phase_manifests: List[List[Dict[str, Any]]] = []
    for eval_index, eval_pool in enumerate(eval_pools):
        phase_root, manifests = _stage_filesystem_distributed_phase(
            eval_pool,
            distributed,
            f"filesystem_compat_eval_{eval_index:02d}",
        )
        eval_phase_roots.append(phase_root)
        eval_phase_manifests.append(manifests)

    compat_root = Path(distributed["root"]) / distributed["run_id"] / "filesystem_compat"
    compat_root.mkdir(parents=True, exist_ok=True)
    model_path = compat_root / "model.ctboost"
    done_path = compat_root / "done.txt"

    if distributed["rank"] == 0:
        merged_pool = _merge_distributed_shards(train_phase_root, train_manifests)
        merged_eval_pools = [
            _merge_distributed_shards(phase_root, manifests)
            for phase_root, manifests in zip(eval_phase_roots, eval_phase_manifests)
        ]
        merged_params = _strip_distributed_params(config)
        prepared = PreparedTrainingData(
            pool=merged_pool,
            eval_pools=merged_eval_pools,
            eval_names=list(eval_names),
            feature_pipeline=feature_pipeline,
        )
        booster = train(
            prepared,
            merged_params,
            num_boost_round=iterations,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callbacks,
            init_model=init_model,
        )
        booster.save_model(model_path)
        done_path.write_text("ok\n", encoding="utf-8")
        return booster

    _wait_for_path(done_path, distributed["timeout"])
    _wait_for_path(model_path, distributed["timeout"])
    return Booster.load_model(model_path)


def _feature_name_to_index_map(pool: Pool) -> Optional[Dict[str, int]]:
    if pool.feature_names is None:
        return None
    return {str(name): index for index, name in enumerate(pool.feature_names)}


def _resolve_feature_index(
    key: Any,
    pool: Pool,
    param_name: str,
    feature_name_map: Optional[Dict[str, int]],
) -> int:
    if isinstance(key, (np.integer, int)):
        feature_index = int(key)
    elif isinstance(key, str):
        if feature_name_map is None:
            raise ValueError(
                f"{param_name} uses feature names, but the training pool does not have feature_names"
            )
        if key not in feature_name_map:
            raise ValueError(f"{param_name} references unknown feature name {key!r}")
        feature_index = feature_name_map[key]
    else:
        raise TypeError(f"{param_name} keys must be feature indices or feature names")

    if feature_index < 0 or feature_index >= pool.num_cols:
        raise ValueError(f"{param_name} feature index {feature_index} is out of bounds")
    return feature_index


def _resolve_feature_int_config(value: Any, pool: Pool, param_name: str) -> List[int]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved = [0] * pool.num_cols
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            entry = int(raw_entry)
            if entry < 0:
                raise ValueError(f"{param_name} entries must be non-negative")
            resolved[feature_index] = entry
        return resolved

    entries = list(np.asarray(value).reshape(-1).tolist()) if isinstance(value, np.ndarray) else list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    resolved = [int(entry) for entry in entries]
    if any(entry < 0 for entry in resolved):
        raise ValueError(f"{param_name} entries must be non-negative")
    return resolved


def _resolve_feature_string_config(value: Any, pool: Pool, param_name: str) -> List[str]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved = [""] * pool.num_cols
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            resolved[feature_index] = "" if raw_entry is None else str(raw_entry)
        return resolved

    entries = list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    return ["" if entry is None else str(entry) for entry in entries]


def _resolve_feature_border_config(value: Any, pool: Pool, param_name: str) -> List[List[float]]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved: List[List[float]] = [[] for _ in range(pool.num_cols)]
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            resolved[feature_index] = [float(border) for border in raw_entry]
        return resolved

    entries = list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    return [[float(border) for border in borders] for borders in entries]


def _resolve_feature_float_config(value: Any, pool: Pool, param_name: str) -> List[float]:
    if value is None:
        return []

    feature_name_map = _feature_name_to_index_map(pool)
    if isinstance(value, Mapping):
        resolved = [0.0] * pool.num_cols
        for key, raw_entry in value.items():
            feature_index = _resolve_feature_index(key, pool, param_name, feature_name_map)
            entry = float(raw_entry)
            if entry < 0.0:
                raise ValueError(f"{param_name} entries must be non-negative")
            resolved[feature_index] = entry
        return resolved

    entries = list(np.asarray(value).reshape(-1).tolist()) if isinstance(value, np.ndarray) else list(value)
    if len(entries) > pool.num_cols:
        raise ValueError(f"{param_name} cannot specify more entries than the number of features")
    resolved = [float(entry) for entry in entries]
    if any(entry < 0.0 for entry in resolved):
        raise ValueError(f"{param_name} entries must be non-negative")
    return resolved


def _objective_name(params: Mapping[str, Any]) -> str:
    return str(params.get("objective", params.get("loss_function", "RMSE")))


def _is_binary_classification_objective(objective_name: str) -> bool:
    return objective_name.lower() in {"logloss", "binary_logloss", "binary:logistic"}


def _is_classification_objective(objective_name: str) -> bool:
    return objective_name.lower() in {
        "logloss",
        "binary_logloss",
        "binary:logistic",
        "multiclass",
        "softmax",
        "softmaxloss",
    }


def _is_ranking_objective(objective_name: str) -> bool:
    return objective_name.lower() in {"pairlogit", "pairwise", "ranknet"}


def _combine_weight_arrays(*arrays: Optional[np.ndarray]) -> Optional[np.ndarray]:
    combined: Optional[np.ndarray] = None
    for array in arrays:
        if array is None:
            continue
        resolved = np.asarray(array, dtype=np.float32)
        combined = resolved.copy() if combined is None else combined * resolved
    return combined


def _apply_sample_weight_to_pool(pool: Pool, sample_weight: Optional[Any]) -> Pool:
    if sample_weight is None:
        return pool

    resolved_weight = np.asarray(sample_weight, dtype=np.float32)
    if resolved_weight.ndim != 1:
        raise ValueError("sample_weight must be a 1D array")
    if resolved_weight.shape[0] != len(pool):
        raise ValueError("sample_weight size must match the number of rows")

    combined_weight = _combine_weight_arrays(pool.weight, resolved_weight)
    return _clone_pool(
        pool,
        weight=combined_weight,
        releasable_feature_storage=True,
    )


def _balanced_class_weight(labels: np.ndarray) -> Dict[Any, float]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_count = unique_labels.size
    sample_count = labels.shape[0]
    return {
        label.item() if hasattr(label, "item") else label: float(sample_count / (class_count * count))
        for label, count in zip(unique_labels, counts)
    }


def _resolve_class_weight_map(
    labels: np.ndarray,
    class_weights: Optional[Any],
    auto_class_weights: Optional[str],
) -> Optional[Dict[Any, float]]:
    if class_weights is not None and auto_class_weights is not None:
        raise ValueError("class_weights and auto_class_weights cannot be used together")

    if auto_class_weights is not None:
        normalized = str(auto_class_weights).lower()
        if normalized != "balanced":
            raise ValueError("auto_class_weights must be 'balanced'")
        return _balanced_class_weight(labels)

    if class_weights is None:
        return None

    if isinstance(class_weights, Mapping):
        return {
            key.item() if hasattr(key, "item") else key: float(value)
            for key, value in class_weights.items()
        }

    weight_values = np.asarray(class_weights, dtype=np.float32).reshape(-1)
    unique_labels = np.unique(labels)
    if weight_values.shape[0] != unique_labels.shape[0]:
        raise ValueError("class_weights must match the number of unique labels")
    return {
        label.item() if hasattr(label, "item") else label: float(weight)
        for label, weight in zip(unique_labels.tolist(), weight_values.tolist())
    }


def _classification_weight_multiplier(
    labels: np.ndarray,
    objective_name: str,
    *,
    class_weights: Optional[Any] = None,
    auto_class_weights: Optional[str] = None,
    scale_pos_weight: Optional[float] = None,
) -> Optional[np.ndarray]:
    if (
        class_weights is None
        and auto_class_weights is None
        and scale_pos_weight is None
    ):
        return None

    if not _is_classification_objective(objective_name):
        raise ValueError("class weighting parameters require a classification objective")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("classification labels must be a 1D array")

    multiplier = np.ones(labels.shape[0], dtype=np.float32)
    class_weight_map = _resolve_class_weight_map(labels, class_weights, auto_class_weights)
    if class_weight_map is not None:
        unique_labels = {
            label.item() if hasattr(label, "item") else label for label in np.unique(labels).tolist()
        }
        if set(class_weight_map) != unique_labels:
            raise ValueError("class_weights must provide a weight for every class in the data")
        for label_value, weight in class_weight_map.items():
            if weight <= 0.0:
                raise ValueError("class weights must be positive")
            multiplier[labels == label_value] *= float(weight)

    if scale_pos_weight is not None:
        if not _is_binary_classification_objective(objective_name):
            raise ValueError("scale_pos_weight is only supported for binary classification")
        scale_value = float(scale_pos_weight)
        if scale_value <= 0.0:
            raise ValueError("scale_pos_weight must be positive")
        positive_mask = labels == 1
        multiplier[positive_mask] *= scale_value

    return multiplier


def _apply_objective_weights(pool: Pool, params: Mapping[str, Any], objective_name: str) -> Pool:
    objective_weight = _classification_weight_multiplier(
        np.asarray(pool.label),
        objective_name,
        class_weights=params.get("class_weights"),
        auto_class_weights=params.get("auto_class_weights"),
        scale_pos_weight=params.get("scale_pos_weight"),
    )
    if objective_weight is None:
        return pool

    combined_weight = _combine_weight_arrays(pool.weight, objective_weight)
    return _clone_pool(
        pool,
        weight=combined_weight,
        releasable_feature_storage=True,
    )


def _slice_pool(pool: Pool, indices: Iterable[int]) -> Pool:
    index_array = np.asarray(list(indices), dtype=np.int64)
    weight = None if pool.weight is None else pool.weight[index_array]
    return Pool(
        data=pool.data[index_array],
        label=pool.label[index_array],
        cat_features=pool.cat_features,
        weight=weight,
        group_id=None if pool.group_id is None else pool.group_id[index_array],
        feature_names=pool.feature_names,
        _releasable_feature_storage=True,
    )


def _stack_histories(histories: List[np.ndarray]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=np.float32)
    max_length = max(history.shape[0] for history in histories)
    stacked = np.full((max_length, len(histories)), np.nan, dtype=np.float32)
    for column_index, history in enumerate(histories):
        stacked[: history.shape[0], column_index] = history
    return stacked


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


def _metric_config(params: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "quantile_alpha": float(params.get("quantile_alpha", 0.5)),
        "huber_delta": float(params.get("huber_delta", 1.0)),
        "tweedie_variance_power": float(params.get("tweedie_variance_power", 1.5)),
    }


def _metric_higher_is_better(metric_name: str, params: Mapping[str, Any]) -> bool:
    config = _metric_config(params)
    return bool(
        _core._metric_higher_is_better(
            metric_name,
            quantile_alpha=config["quantile_alpha"],
            huber_delta=config["huber_delta"],
            tweedie_variance_power=config["tweedie_variance_power"],
        )
    )


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
    return float(
        _core._evaluate_metric(
            np.asarray(predictions, dtype=np.float32),
            np.asarray(pool.label, dtype=np.float32),
            weights,
            metric_name,
            num_classes,
            group_ids,
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
    }


def _evaluate_metric_from_inputs(
    metric_name: str,
    inputs: Mapping[str, Any],
    *,
    num_classes: int,
    params: Mapping[str, Any],
) -> float:
    config = _metric_config(params)
    return float(
        _core._evaluate_metric(
            np.asarray(inputs["predictions"], dtype=np.float32),
            np.asarray(inputs["label"], dtype=np.float32),
            np.asarray(inputs["weight"], dtype=np.float32),
            metric_name,
            num_classes,
            None if inputs.get("group_id") is None else np.asarray(inputs["group_id"], dtype=np.int64),
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
    return merged


def _evaluate_metric_distributed(
    distributed: Optional[Dict[str, Any]],
    *,
    key: str,
    metric_name: str,
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


def _prune_booster_to_best_iteration(handle: Any, best_iteration: int) -> None:
    state = dict(handle.export_state())
    retained_iterations = int(best_iteration) + 1
    prediction_dimension = int(handle.prediction_dimension())
    state["trees"] = state["trees"][: retained_iterations * prediction_dimension]
    state["loss_history"] = state["loss_history"][:retained_iterations]
    state["eval_loss_history"] = state["eval_loss_history"][:retained_iterations]
    state["best_iteration"] = int(best_iteration)
    if retained_iterations > 0:
        eval_history = state["eval_loss_history"]
        if eval_history:
            state["best_score"] = float(eval_history[retained_iterations - 1])
        else:
            state["best_score"] = float(state["loss_history"][retained_iterations - 1])
    handle.load_state(state)


def _initial_evals_result_from_model(init_model: Any) -> Optional[Dict[str, Dict[str, List[float]]]]:
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


def _distributed_is_root(distributed: Optional[Dict[str, Any]]) -> bool:
    return distributed is None or int(distributed["rank"]) == 0


def prepare_training_data(
    pool: Any,
    params: Mapping[str, Any],
    *,
    label: Any = None,
    weight: Any = None,
    group_id: Any = None,
    feature_names: Optional[List[str]] = None,
    eval_set: Any = None,
    eval_names: Any = None,
    init_model: Any = None,
) -> PreparedTrainingData:
    if isinstance(pool, PreparedTrainingData):
        if label is not None or weight is not None or group_id is not None or feature_names is not None:
            raise ValueError(
                "label, weight, group_id, and feature_names cannot be passed when pool is PreparedTrainingData"
            )
        if eval_set is not None or eval_names is not None:
            raise ValueError("eval_set and eval_names cannot be passed when pool is PreparedTrainingData")
        return pool

    config = dict(params)
    distributed_config = _normalize_distributed_config(config)
    init_pipeline = _resolve_init_model_pipeline(init_model)
    shared_feature_pipeline = init_pipeline

    if isinstance(pool, Pool):
        if label is not None or weight is not None or group_id is not None or feature_names is not None:
            pool = _pool_from_data_and_label(
                pool,
                label,
                group_id=group_id,
                weight=weight,
                feature_names=feature_names,
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
                feature_names=feature_names,
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
                        feature_names=feature_names,
                        params=config,
                        distributed=distributed_config,
                    )
            else:
                pool, shared_feature_pipeline = _prepare_distributed_feature_pipeline_pool(
                    pool,
                    label,
                    group_id=group_id,
                    weight=weight,
                    feature_names=feature_names,
                    params=config,
                    distributed=distributed_config,
                )
        else:
            pool = _prepare_pool_from_raw(
                pool,
                label,
                group_id=group_id,
                weight=weight,
                feature_names=feature_names,
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


class Booster:
    """Small Python wrapper around the native gradient booster."""

    def __init__(
        self,
        handle: Any,
        *,
        feature_pipeline: Optional[FeaturePipeline] = None,
        training_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._handle = handle
        self._feature_pipeline = feature_pipeline
        self._training_metadata = None if training_metadata is None else dict(training_metadata)

    def _set_training_metadata(
        self,
        *,
        evals_result: Optional[Mapping[str, Mapping[str, Iterable[float]]]] = None,
        eval_loss_history: Optional[Iterable[float]] = None,
        best_iteration: Optional[int] = None,
        best_score: Optional[float] = None,
        eval_metric_name: Optional[str] = None,
    ) -> None:
        if evals_result is None and eval_loss_history is None and best_iteration is None and best_score is None and eval_metric_name is None:
            self._training_metadata = None
            return
        metadata = {} if self._training_metadata is None else dict(self._training_metadata)
        if evals_result is not None:
            metadata["evals_result"] = _copy_evals_result(evals_result)
        if eval_loss_history is not None:
            metadata["eval_loss_history"] = [float(value) for value in eval_loss_history]
        if best_iteration is not None:
            metadata["best_iteration"] = int(best_iteration)
        if best_score is not None:
            metadata["best_score"] = float(best_score)
        if eval_metric_name is not None:
            metadata["eval_metric_name"] = str(eval_metric_name)
        self._training_metadata = metadata

    def _prediction_pool(self, data: Any) -> Pool:
        if isinstance(data, Pool):
            return data
        if self._feature_pipeline is None:
            return _prediction_pool(data)
        num_rows = int(data.shape[0]) if hasattr(data, "shape") else len(data)
        transformed, cat_features, feature_names = self._feature_pipeline.transform_array(data)
        return Pool(
            data=transformed,
            label=np.zeros(num_rows, dtype=np.float32),
            cat_features=cat_features,
            feature_names=feature_names,
        )

    def _predict_raw(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._prediction_pool(data)
        raw = np.asarray(
            self._handle.predict(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.float32,
        )
        prediction_dimension = int(self._handle.prediction_dimension())
        if prediction_dimension > 1:
            return raw.reshape((pool.num_rows, prediction_dimension))
        return raw

    def predict(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        return self._predict_raw(data, num_iteration=num_iteration)

    def staged_predict(self, data: Any) -> Iterable[np.ndarray]:
        pool = self._prediction_pool(data)
        for iteration in range(1, self.num_iterations_trained + 1):
            yield self._predict_raw(pool, num_iteration=iteration)

    def predict_leaf_index(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._prediction_pool(data)
        values = np.asarray(
            self._handle.predict_leaf_indices(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.int32,
        )
        tree_count = 0 if pool.num_rows == 0 else values.size // pool.num_rows
        return values.reshape((pool.num_rows, tree_count))

    def predict_contrib(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._prediction_pool(data)
        values = np.asarray(
            self._handle.predict_contributions(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.float32,
        )
        width = pool.num_cols + 1
        if self.prediction_dimension > 1:
            return values.reshape((pool.num_rows, self.prediction_dimension, width))
        return values.reshape((pool.num_rows, width))

    def save_model(self, path: PathLike, *, model_format: Optional[str] = None) -> None:
        destination = Path(path)
        save_booster(
            destination,
            self._handle,
            model_format=model_format,
            feature_pipeline_state=None
            if self._feature_pipeline is None
            else self._feature_pipeline.to_state(),
            training_state=self._training_metadata,
        )

    def export_model(
        self,
        path: PathLike,
        *,
        export_format: Optional[str] = None,
        prepared_features: bool = False,
    ) -> None:
        _export_model(
            path,
            self._handle,
            export_format=export_format,
            feature_pipeline=self._feature_pipeline,
            prepared_features=prepared_features,
        )

    @classmethod
    def load_model(cls, path: PathLike) -> "Booster":
        document = load_booster_document(Path(path))
        pipeline_state = document.get("feature_pipeline_state")
        return cls(
            _core.GradientBooster.from_state(document["booster_state"]),
            feature_pipeline=None if pipeline_state is None else FeaturePipeline.from_state(pipeline_state),
            training_metadata=document.get("training_state"),
        )

    def get_quantization_schema(self) -> Optional[Dict[str, Any]]:
        state = self._handle.quantization_schema_state()
        if state is None:
            return None
        return dict(state)

    def get_borders(self) -> Optional[Dict[str, Any]]:
        schema = self.get_quantization_schema()
        if schema is None:
            return None

        cut_offsets = list(schema["cut_offsets"])
        cut_values = list(schema["cut_values"])
        feature_borders = [
            cut_values[cut_offsets[index] : cut_offsets[index + 1]]
            for index in range(len(cut_offsets) - 1)
        ]
        nan_mode_lookup = {0: "Forbidden", 1: "Min", 2: "Max"}
        nan_modes = schema.get("nan_modes")
        if nan_modes is None:
            default_mode = nan_mode_lookup.get(int(schema["nan_mode"]), "Min")
            nan_mode_by_feature = [default_mode] * len(feature_borders)
        else:
            nan_mode_by_feature = [
                nan_mode_lookup.get(int(mode), "Min") for mode in nan_modes
            ]
        return {
            "feature_borders": feature_borders,
            "nan_mode_by_feature": nan_mode_by_feature,
            "num_bins_per_feature": list(schema["num_bins_per_feature"]),
            "categorical_mask": list(schema["categorical_mask"]),
            "missing_value_mask": list(schema["missing_value_mask"]),
        }

    @property
    def loss_history(self) -> List[float]:
        return list(self._handle.loss_history())

    @property
    def eval_loss_history(self) -> List[float]:
        if self._training_metadata is not None and "eval_loss_history" in self._training_metadata:
            return list(self._training_metadata["eval_loss_history"])
        return list(self._handle.eval_loss_history())

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._handle.feature_importances(), dtype=np.float32)

    @property
    def evals_result_(self) -> Dict[str, Dict[str, List[float]]]:
        if self._training_metadata is not None and "evals_result" in self._training_metadata:
            return _copy_evals_result(self._training_metadata["evals_result"])
        result = {"learn": {"loss": self.loss_history}}
        if self.eval_loss_history:
            metric_name = "loss" if self.eval_metric_name.lower() == self.objective_name.lower() else self.eval_metric_name
            result["validation"] = {metric_name: self.eval_loss_history}
        return result

    @property
    def num_classes(self) -> int:
        return int(self._handle.num_classes())

    @property
    def prediction_dimension(self) -> int:
        return int(self._handle.prediction_dimension())

    @property
    def num_iterations_trained(self) -> int:
        return int(self._handle.num_iterations_trained())

    @property
    def best_iteration(self) -> int:
        if self._training_metadata is not None and "best_iteration" in self._training_metadata:
            return int(self._training_metadata["best_iteration"])
        return int(self._handle.best_iteration())

    @property
    def objective_name(self) -> str:
        return str(self._handle.objective_name())

    @property
    def eval_metric_name(self) -> str:
        if self._training_metadata is not None and "eval_metric_name" in self._training_metadata:
            return str(self._training_metadata["eval_metric_name"])
        return str(self._handle.eval_metric_name())


def load_model(path: PathLike) -> Booster:
    return Booster.load_model(path)


def cv(
    pool: Pool,
    params: Mapping[str, Any],
    num_boost_round: Optional[int] = None,
    *,
    nfold: int = 3,
    shuffle: bool = True,
    random_state: Optional[int] = 0,
    stratified: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(pool, Pool):
        raise TypeError("pool must be an instance of ctboost.Pool")
    if nfold < 2:
        raise ValueError("nfold must be at least 2")

    GroupKFold, KFold, StratifiedKFold = _load_sklearn_splitters()
    objective = _objective_name(params)
    labels = np.asarray(pool.label)
    if _is_ranking_objective(objective):
        if pool.group_id is None:
            raise ValueError("ranking cross-validation requires group_id on the input pool")
        splitter = GroupKFold(n_splits=nfold)
        split_iterator = splitter.split(
            np.zeros(labels.shape[0], dtype=np.float32),
            labels,
            groups=np.asarray(pool.group_id),
        )
    else:
        use_stratified = _is_classification_objective(objective) if stratified is None else bool(stratified)

        if use_stratified:
            splitter = StratifiedKFold(
                n_splits=nfold,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
            split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32), labels)
        else:
            splitter = KFold(
                n_splits=nfold,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
            split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32))

    train_histories: List[np.ndarray] = []
    valid_histories: List[np.ndarray] = []
    best_iterations: List[int] = []

    for train_index, valid_index in split_iterator:
        train_pool = _slice_pool(pool, train_index)
        valid_pool = _slice_pool(pool, valid_index)
        booster = train(
            train_pool,
            params,
            num_boost_round=num_boost_round,
            eval_set=valid_pool,
            early_stopping_rounds=early_stopping_rounds,
        )
        train_histories.append(np.asarray(booster.loss_history, dtype=np.float32))
        valid_histories.append(np.asarray(booster.eval_loss_history, dtype=np.float32))
        best_iterations.append(booster.best_iteration)

    train_matrix = _stack_histories(train_histories)
    valid_matrix = _stack_histories(valid_histories)
    iteration_count = train_matrix.shape[0]

    return {
        "iterations": np.arange(1, iteration_count + 1, dtype=np.int32),
        "train_loss_mean": np.nanmean(train_matrix, axis=1),
        "train_loss_std": np.nanstd(train_matrix, axis=1),
        "valid_loss_mean": np.nanmean(valid_matrix, axis=1),
        "valid_loss_std": np.nanstd(valid_matrix, axis=1),
        "best_iteration_mean": float(np.mean(best_iterations)) if best_iterations else float("nan"),
        "best_iteration_std": float(np.std(best_iterations)) if best_iterations else float("nan"),
    }


def train(
    pool: Any,
    params: Mapping[str, Any],
    num_boost_round: Optional[int] = None,
    *,
    label: Any = None,
    weight: Any = None,
    group_id: Any = None,
    feature_names: Optional[List[str]] = None,
    eval_set: Any = None,
    eval_names: Any = None,
    early_stopping_rounds: Optional[int] = None,
    early_stopping_metric: Optional[str] = None,
    early_stopping_name: Optional[str] = None,
    callbacks: Optional[Iterable[Callable[[TrainingCallbackEnv], Any]]] = None,
    init_model: Any = None,
) -> Booster:
    config = dict(params)
    prepared_inputs = pool if isinstance(pool, PreparedTrainingData) else None
    distributed_config = _normalize_distributed_config(config)
    if prepared_inputs is not None:
        if label is not None or weight is not None or group_id is not None or feature_names is not None:
            raise ValueError(
                "label, weight, group_id, and feature_names cannot be passed when pool is PreparedTrainingData"
            )
        if eval_set is not None or eval_names is not None:
            raise ValueError("eval_set and eval_names cannot be passed when pool is PreparedTrainingData")
        prepared_pool = prepared_inputs.pool
        if not isinstance(prepared_pool, Pool):
            raise TypeError("PreparedTrainingData.pool must be an instance of ctboost.Pool")
        pool = prepared_pool
        feature_pipeline = prepared_inputs.feature_pipeline or getattr(pool, "_feature_pipeline", None)
        eval_pools = list(prepared_inputs.eval_pools)
        resolved_eval_names = list(prepared_inputs.eval_names)
        default_eval_names = _normalize_eval_names(None, len(eval_pools))
    else:
        init_pipeline = _resolve_init_model_pipeline(init_model)
        shared_feature_pipeline = init_pipeline
        if isinstance(pool, Pool):
            if label is not None or weight is not None or group_id is not None or feature_names is not None:
                pool = _pool_from_data_and_label(
                    pool,
                    label,
                    group_id=group_id,
                    weight=weight,
                    feature_names=feature_names,
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
                    feature_names=feature_names,
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
                            feature_names=feature_names,
                            params=config,
                            distributed=distributed_config,
                        )
                else:
                    pool, shared_feature_pipeline = _prepare_distributed_feature_pipeline_pool(
                        pool,
                        label,
                        group_id=group_id,
                        weight=weight,
                        feature_names=feature_names,
                        params=config,
                        distributed=distributed_config,
                    )
            else:
                pool = _prepare_pool_from_raw(
                    pool,
                    label,
                    group_id=group_id,
                    weight=weight,
                    feature_names=feature_names,
                    params=config,
                )
        if not isinstance(pool, Pool):
            raise TypeError("pool must be an instance of ctboost.Pool")

        feature_pipeline = getattr(pool, "_feature_pipeline", None) or shared_feature_pipeline

        normalized_eval_sets = _normalize_eval_sets(eval_set)
        resolved_eval_names = _normalize_eval_names(eval_names, len(normalized_eval_sets))
        default_eval_names = _normalize_eval_names(None, len(normalized_eval_sets))
        eval_pools = []
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

    if callbacks is None:
        callback_list: List[Callable[[TrainingCallbackEnv], Any]] = []
    else:
        try:
            callback_list = list(callbacks)
        except TypeError as exc:
            raise TypeError("callbacks must be an iterable of callables") from exc
        for callback in callback_list:
            if not callable(callback):
                raise TypeError("callbacks must contain only callables")

    if early_stopping_rounds is not None:
        if early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be a positive integer")
        if not eval_pools:
            raise ValueError("early_stopping_rounds requires eval_set")
    early_stopping = 0 if early_stopping_rounds is None else int(early_stopping_rounds)

    init_handle = _resolve_init_model_handle(init_model)
    init_state = None if init_handle is None else dict(init_handle.export_state())
    iterations = int(num_boost_round if num_boost_round is not None else config.get("iterations", 100))
    objective = str(
        config.get(
            "objective",
            config.get(
                "loss_function",
                "RMSE" if init_state is None else init_state["objective_name"],
            ),
        )
    )
    learning_rate = float(
        config.get("learning_rate", 0.1 if init_state is None else init_state["learning_rate"])
    )
    max_depth = int(config.get("max_depth", 6 if init_state is None else init_state["max_depth"]))
    alpha = float(config.get("alpha", 0.05 if init_state is None else init_state["alpha"]))
    lambda_l2 = float(
        config.get(
            "lambda",
            config.get("lambda_l2", 1.0 if init_state is None else init_state["lambda_l2"]),
        )
    )
    subsample = float(
        config.get("subsample", 1.0 if init_state is None else init_state.get("subsample", 1.0))
    )
    bootstrap_type = str(
        config.get(
            "bootstrap_type",
            "No" if init_state is None else init_state.get("bootstrap_type", "No"),
        )
    )
    bagging_temperature = float(
        config.get(
            "bagging_temperature",
            0.0 if init_state is None else init_state.get("bagging_temperature", 0.0),
        )
    )
    boosting_type = str(
        config.get(
            "boosting_type",
            "GradientBoosting" if init_state is None else init_state.get("boosting_type", "GradientBoosting"),
        )
    )
    drop_rate = float(
        config.get("drop_rate", 0.1 if init_state is None else init_state.get("drop_rate", 0.1))
    )
    skip_drop = float(
        config.get("skip_drop", 0.5 if init_state is None else init_state.get("skip_drop", 0.5))
    )
    max_drop = int(
        config.get("max_drop", 0 if init_state is None else init_state.get("max_drop", 0))
    )
    monotone_constraints_value = config.get(
        "monotone_constraints",
        [] if init_state is None else init_state.get("monotone_constraints", []),
    )
    monotone_constraints = [] if monotone_constraints_value is None else list(monotone_constraints_value)
    interaction_constraints_value = config.get(
        "interaction_constraints",
        [] if init_state is None else init_state.get("interaction_constraints", []),
    )
    interaction_constraints = (
        [] if interaction_constraints_value is None else list(interaction_constraints_value)
    )
    colsample_bytree = float(
        config.get(
            "colsample_bytree",
            1.0 if init_state is None else init_state.get("colsample_bytree", 1.0),
        )
    )
    feature_weights = _resolve_feature_float_config(
        config.get(
            "feature_weights",
            [] if init_state is None else init_state.get("feature_weights", []),
        ),
        pool,
        "feature_weights",
    )
    first_feature_use_penalties = _resolve_feature_float_config(
        config.get(
            "first_feature_use_penalties",
            []
            if init_state is None
            else init_state.get("first_feature_use_penalties", []),
        ),
        pool,
        "first_feature_use_penalties",
    )
    random_strength = float(
        config.get(
            "random_strength",
            0.0 if init_state is None else init_state.get("random_strength", 0.0),
        )
    )
    grow_policy = str(
        config.get(
            "grow_policy",
            "DepthWise" if init_state is None else init_state.get("grow_policy", "DepthWise"),
        )
    )
    max_leaves = int(
        config.get("max_leaves", 0 if init_state is None else init_state.get("max_leaves", 0))
    )
    min_samples_split = int(
        config.get(
            "min_samples_split",
            2 if init_state is None else init_state.get("min_samples_split", 2),
        )
    )
    min_data_in_leaf = int(
        config.get(
            "min_data_in_leaf",
            0 if init_state is None else init_state.get("min_data_in_leaf", 0),
        )
    )
    min_child_weight = float(
        config.get(
            "min_child_weight",
            0.0 if init_state is None else init_state.get("min_child_weight", 0.0),
        )
    )
    gamma = float(
        config.get("gamma", 0.0 if init_state is None else init_state.get("gamma", 0.0))
    )
    max_leaf_weight = float(
        config.get(
            "max_leaf_weight",
            0.0 if init_state is None else init_state.get("max_leaf_weight", 0.0),
        )
    )
    if "num_classes" in config:
        num_classes = int(config["num_classes"])
    elif init_state is not None:
        num_classes = int(init_state["num_classes"])
    elif objective.lower() in {"multiclass", "softmax", "softmaxloss"}:
        num_classes = int(np.unique(pool.label).size)
    else:
        num_classes = 1
    max_bins = int(config.get("max_bins", 256 if init_state is None else init_state["max_bins"]))
    nan_mode = str(config.get("nan_mode", "Min" if init_state is None else init_state["nan_mode"]))
    max_bin_by_feature = _resolve_feature_int_config(
        config.get(
            "max_bin_by_feature",
            [] if init_state is None else init_state.get("max_bin_by_feature", []),
        ),
        pool,
        "max_bin_by_feature",
    )
    border_selection_method = str(
        config.get(
            "border_selection_method",
            "Quantile" if init_state is None else init_state.get("border_selection_method", "Quantile"),
        )
    )
    nan_mode_by_feature = _resolve_feature_string_config(
        config.get(
            "nan_mode_by_feature",
            [] if init_state is None else init_state.get("nan_mode_by_feature", []),
        ),
        pool,
        "nan_mode_by_feature",
    )
    feature_borders = _resolve_feature_border_config(
        config.get(
            "feature_borders",
            [] if init_state is None else init_state.get("feature_borders", []),
        ),
        pool,
        "feature_borders",
    )
    quantile_alpha = float(
        config.get("quantile_alpha", 0.5 if init_state is None else init_state["quantile_alpha"])
    )
    huber_delta = float(config.get("huber_delta", 1.0 if init_state is None else init_state["huber_delta"]))
    tweedie_variance_power = float(
        config.get(
            "tweedie_variance_power",
            1.5 if init_state is None else init_state.get("tweedie_variance_power", 1.5),
        )
    )
    task_type = str(config.get("task_type", "CPU" if init_state is None else init_state["task_type"]))
    devices = str(config.get("devices", "0" if init_state is None else init_state["devices"]))
    distributed_world_size = int(
        config.get(
            "distributed_world_size",
            1 if init_state is None else init_state.get("distributed_world_size", 1),
        )
    )
    distributed_rank = int(
        config.get(
            "distributed_rank",
            0 if init_state is None else init_state.get("distributed_rank", 0),
        )
    )
    distributed_root = str(
        config.get(
            "distributed_root",
            "" if init_state is None else init_state.get("distributed_root", ""),
        )
        or ""
    )
    distributed_run_id = str(
        config.get(
            "distributed_run_id",
            "default" if init_state is None else init_state.get("distributed_run_id", "default"),
        )
    )
    distributed_timeout = float(
        config.get(
            "distributed_timeout",
            600.0 if init_state is None else init_state.get("distributed_timeout", 600.0),
        )
    )
    native_external_memory = bool(
        config.get(
            "external_memory",
            False if init_state is None else init_state.get("external_memory", False),
        )
    )
    native_external_memory_dir = str(
        config.get(
            "external_memory_dir",
            "" if init_state is None else init_state.get("external_memory_dir", ""),
        )
        or ""
    )
    random_seed = int(
        config.get("random_seed", 0 if init_state is None else init_state.get("random_seed", 0))
    )
    verbose = bool(config.get("verbose", False if init_state is None else init_state.get("verbose", False)))

    eval_metric_names = _normalize_eval_metrics(
        config.get("eval_metric", "" if init_state is None else init_state["eval_metric_name"])
    )
    if not eval_metric_names:
        eval_metric_names = [objective if init_state is None else str(init_state["eval_metric_name"])]
    if early_stopping_metric is not None:
        requested_metric = str(early_stopping_metric)
        matching_metric = next(
            (metric for metric in eval_metric_names if metric.lower() == requested_metric.lower()),
            None,
        )
        if matching_metric is None:
            eval_metric_names = [requested_metric] + eval_metric_names
            primary_eval_metric = requested_metric
        else:
            primary_eval_metric = matching_metric
    else:
        primary_eval_metric = eval_metric_names[0]

    if early_stopping_name is not None:
        if not resolved_eval_names:
            raise ValueError("early_stopping_name requires eval_set")
        primary_eval_name = str(early_stopping_name)
        if primary_eval_name not in resolved_eval_names:
            raise ValueError("early_stopping_name must match one of the provided eval_names")
    else:
        primary_eval_name = resolved_eval_names[0] if resolved_eval_names else None

    metric_directions = {
        metric_name: _metric_higher_is_better(metric_name, config)
        for metric_name in eval_metric_names
    }
    use_python_eval_surface = (
        bool(callback_list)
        or len(eval_pools) > 1
        or len(eval_metric_names) > 1
        or resolved_eval_names != default_eval_names
    )

    filesystem_compat_required = (
        distributed_config is not None
        and distributed_config["backend"] != "tcp"
        and (
            task_type.upper() == "GPU"
            or bool(eval_pools)
            or use_python_eval_surface
            or pool.group_id is not None
        )
    )
    if filesystem_compat_required:
        return _distributed_train_filesystem_compat(
            pool,
            config,
            distributed=distributed_config,
            iterations=iterations,
            eval_pools=eval_pools,
            eval_names=resolved_eval_names,
            feature_pipeline=feature_pipeline,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callback_list,
            init_model=init_model,
        )

    if distributed_config is not None:
        if task_type.upper() == "GPU" and distributed_config["backend"] != "tcp":
            raise ValueError("distributed GPU training requires distributed_root='tcp://host:port'")
        if use_python_eval_surface and distributed_config["backend"] != "tcp":
            raise ValueError(
                "distributed multi-metric, multi-watchlist, and callback workflows require "
                "distributed_root='tcp://host:port'"
            )
        if eval_pools and not use_python_eval_surface and distributed_config["backend"] != "tcp":
            raise ValueError("distributed eval_set support requires distributed_root='tcp://host:port'")
    with _distributed_collective_context(distributed_config):
        distributed_quantization_schema = None
        if distributed_config is not None and init_state is None:
            distributed_quantization_schema = _resolve_distributed_quantization_schema(
                pool,
                distributed_config,
                max_bins=max_bins,
                nan_mode=nan_mode,
                max_bin_by_feature=max_bin_by_feature,
                border_selection_method=border_selection_method,
                nan_mode_by_feature=nan_mode_by_feature,
                feature_borders=feature_borders,
                external_memory=native_external_memory,
            )
        if init_state is not None:
            if objective.lower() != str(init_state["objective_name"]).lower():
                raise ValueError("init_model objective must match the current training objective")
            if num_classes != int(init_state["num_classes"]):
                raise ValueError("init_model num_classes must match the current training configuration")
        weighted_pool = _apply_objective_weights(pool, config, objective)
        weighted_eval_pools = [
            _apply_objective_weights(eval_pool, config, objective) for eval_pool in eval_pools
        ]

        booster = _core.GradientBooster(
            objective=objective,
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            lambda_l2=lambda_l2,
            subsample=subsample,
            bootstrap_type=bootstrap_type,
            bagging_temperature=bagging_temperature,
            boosting_type=boosting_type,
            drop_rate=drop_rate,
            skip_drop=skip_drop,
            max_drop=max_drop,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            colsample_bytree=colsample_bytree,
            feature_weights=feature_weights,
            first_feature_use_penalties=first_feature_use_penalties,
            random_strength=random_strength,
            grow_policy=grow_policy,
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            min_data_in_leaf=min_data_in_leaf,
            min_child_weight=min_child_weight,
            gamma=gamma,
            max_leaf_weight=max_leaf_weight,
            num_classes=num_classes,
            max_bins=max_bins,
            nan_mode=nan_mode,
            max_bin_by_feature=max_bin_by_feature,
            border_selection_method=border_selection_method,
            nan_mode_by_feature=nan_mode_by_feature,
            feature_borders=feature_borders,
            external_memory=native_external_memory,
            external_memory_dir=native_external_memory_dir,
            eval_metric=primary_eval_metric,
            quantile_alpha=quantile_alpha,
            huber_delta=huber_delta,
            tweedie_variance_power=tweedie_variance_power,
            task_type=task_type,
            devices=devices,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_root=distributed_root,
            distributed_run_id=distributed_run_id,
            distributed_timeout=distributed_timeout,
            random_seed=random_seed,
            verbose=verbose,
        )
        if init_state is not None:
            booster.load_state(init_state)
        elif distributed_quantization_schema is not None:
            booster.load_quantization_schema(distributed_quantization_schema)
        if not use_python_eval_surface:
            native_eval_pool = weighted_eval_pools[0] if weighted_eval_pools else None
            booster.fit(
                weighted_pool._handle,
                None if native_eval_pool is None else native_eval_pool._handle,
                early_stopping,
                init_state is not None,
            )
            return Booster(booster, feature_pipeline=feature_pipeline)

        wrapped_booster = Booster(booster, feature_pipeline=feature_pipeline)
        train_pool_template = _clone_pool(weighted_pool, releasable_feature_storage=True)
        seeded_evals_result = _initial_evals_result_from_model(init_model)
        evals_result: Dict[str, Dict[str, List[float]]] = {
            "learn": {"loss": [float(value) for value in booster.loss_history()]}
        }
        for eval_name in resolved_eval_names:
            evals_result[eval_name] = {}
            for metric_name in eval_metric_names:
                if (
                    seeded_evals_result is not None
                    and eval_name in seeded_evals_result
                    and metric_name in seeded_evals_result[eval_name]
                ):
                    history = seeded_evals_result[eval_name][metric_name]
                elif (
                    len(weighted_eval_pools) == 1
                    and eval_name == "validation"
                    and metric_name.lower() == primary_eval_metric.lower()
                    and booster.eval_loss_history()
                ):
                    history = booster.eval_loss_history()
                else:
                    history = []
                evals_result[eval_name][metric_name] = [float(value) for value in history]

        begin_iteration = int(booster.num_iterations_trained())
        end_iteration = begin_iteration + iterations
        if primary_eval_name is not None:
            monitor_history = evals_result[primary_eval_name][primary_eval_metric]
            best_iteration = -1
            best_score = float("-inf") if metric_directions[primary_eval_metric] else float("inf")
            for history_index, history_value in enumerate(monitor_history):
                improved = (
                    best_iteration < 0
                    or (
                        metric_directions[primary_eval_metric]
                        and history_value > best_score
                    )
                    or (
                        not metric_directions[primary_eval_metric]
                        and history_value < best_score
                    )
                )
                if improved:
                    best_iteration = history_index
                    best_score = float(history_value)
        else:
            monitor_history = []
            best_iteration = begin_iteration - 1 if begin_iteration > 0 else -1
            best_score = (
                float(evals_result["learn"]["loss"][best_iteration])
                if best_iteration >= 0 and evals_result["learn"]["loss"]
                else None
            )

        wrapped_booster._set_training_metadata(
            evals_result=evals_result,
            eval_loss_history=monitor_history,
            best_iteration=best_iteration,
            best_score=best_score,
            eval_metric_name=primary_eval_metric,
        )

        stopped_by_early_stopping = False
        callback_stop_requested = False
        for _ in range(iterations):
            iteration_train_pool = _clone_pool(train_pool_template, releasable_feature_storage=True)
            booster.set_iterations(1)
            booster.fit(
                iteration_train_pool._handle,
                None,
                0,
                booster.num_iterations_trained() > 0,
            )
            booster.set_iterations(iterations)
            evals_result["learn"]["loss"] = [float(value) for value in booster.loss_history()]
            current_iteration = int(booster.num_iterations_trained()) - 1
            evaluation_result_list: List[tuple[str, str, float, bool]] = [
                ("learn", "loss", float(evals_result["learn"]["loss"][-1]), False)
            ]

            for eval_name, weighted_eval_pool in zip(resolved_eval_names, weighted_eval_pools):
                prediction_inputs = _collect_prediction_metric_inputs(
                    wrapped_booster.predict(weighted_eval_pool),
                    weighted_eval_pool,
                )
                if distributed_config is not None:
                    prediction_inputs = _merge_prediction_metric_inputs(
                        _distributed_allgather_value(
                            distributed_config,
                            f"{distributed_run_id}/py_eval/{current_iteration}/{eval_name}",
                            prediction_inputs,
                        )
                    )
                for metric_name in eval_metric_names:
                    metric_value = _evaluate_metric_from_inputs(
                        metric_name,
                        prediction_inputs,
                        num_classes=num_classes,
                        params=config,
                    )
                    evals_result[eval_name][metric_name].append(metric_value)
                    evaluation_result_list.append(
                        (
                            eval_name,
                            metric_name,
                            float(metric_value),
                            bool(metric_directions[metric_name]),
                        )
                    )

            if primary_eval_name is not None:
                current_score = float(evals_result[primary_eval_name][primary_eval_metric][-1])
                improved = (
                    best_iteration < 0
                    or (
                        metric_directions[primary_eval_metric]
                        and current_score > float(best_score)
                    )
                    or (
                        not metric_directions[primary_eval_metric]
                        and current_score < float(best_score)
                    )
                )
                if improved:
                    best_iteration = current_iteration
                    best_score = current_score
                elif early_stopping > 0 and current_iteration - best_iteration >= early_stopping:
                    stopped_by_early_stopping = True
            else:
                best_iteration = current_iteration
                best_score = float(evals_result["learn"]["loss"][-1])

            wrapped_booster._set_training_metadata(
                evals_result=evals_result,
                eval_loss_history=(
                    evals_result[primary_eval_name][primary_eval_metric]
                    if primary_eval_name is not None
                    else []
                ),
                best_iteration=best_iteration,
                best_score=best_score,
                eval_metric_name=primary_eval_metric,
            )

            if callback_list:
                env = TrainingCallbackEnv(
                    model=wrapped_booster,
                    iteration=current_iteration,
                    begin_iteration=begin_iteration,
                    end_iteration=end_iteration,
                    evaluation_result_list=evaluation_result_list,
                    history=wrapped_booster.evals_result_,
                    best_iteration=best_iteration,
                    best_score=None if best_score is None else float(best_score),
                )
                if _distributed_is_root(distributed_config):
                    for callback in callback_list:
                        if callback(env):
                            callback_stop_requested = True
                            break
                if distributed_config is not None:
                    callback_stop_requested = bool(
                        _distributed_broadcast_value(
                            distributed_config,
                            f"{distributed_run_id}/py_callback_stop/{current_iteration}",
                            value=callback_stop_requested,
                        )
                    )

            if stopped_by_early_stopping or callback_stop_requested:
                break

        booster.set_iterations(iterations)
        if stopped_by_early_stopping and best_iteration >= 0 and best_iteration + 1 < booster.num_iterations_trained():
            _prune_booster_to_best_iteration(booster, best_iteration)
            retained_iterations = best_iteration + 1
            evals_result["learn"]["loss"] = evals_result["learn"]["loss"][:retained_iterations]
            for eval_name in resolved_eval_names:
                for metric_name in eval_metric_names:
                    evals_result[eval_name][metric_name] = evals_result[eval_name][metric_name][
                        :retained_iterations
                    ]

        wrapped_booster._set_training_metadata(
            evals_result=evals_result,
            eval_loss_history=(
                evals_result[primary_eval_name][primary_eval_metric]
                if primary_eval_name is not None
                else []
            ),
            best_iteration=best_iteration,
            best_score=best_score,
            eval_metric_name=primary_eval_metric,
        )
        return wrapped_booster
