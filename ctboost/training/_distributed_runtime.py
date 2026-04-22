"""Distributed runtime orchestration for ctboost.training."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

from .. import _core
from ..core import Pool
from ..distributed import distributed_tcp_request, pickle_payload, wait_for_distributed_tcp_coordinator
from ..feature_pipeline import FeaturePipeline
from ._distributed_config import _normalize_distributed_config, _strip_distributed_params
from ._distributed_payloads import (
    _distributed_allgather_value,
    _quantization_schema_from_debug_histogram,
    _serialize_distributed_pool_shard,
)
from ._distributed_storage_io import _load_manifest, _wait_for_manifests, _wait_for_path, _write_distributed_shard
from ._distributed_storage_merge import _merge_distributed_shards

@contextlib.contextmanager
def _distributed_collective_context(distributed: Optional[Dict[str, Any]]):
    server_process = None
    body_error: Optional[BaseException] = None
    shutdown_error: Optional[BaseException] = None
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
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        if distributed is not None and distributed["backend"] == "tcp" and body_error is None:
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
            except Exception as exc:
                shutdown_error = exc
        if server_process is not None:
            server_process.terminate()
            try:
                server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=5.0)
        if shutdown_error is not None:
            raise shutdown_error

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
    from .api import train
    from .booster import Booster

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
    early_stopping_metric: Optional[Any],
    early_stopping_name: Optional[str],
    callbacks: List[Callable[[TrainingCallbackEnv], Any]],
    init_model: Any,
) -> "Booster":
    from .api import train
    from .booster import Booster
    from .pool_io import PreparedTrainingData

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

def _distributed_is_root(distributed: Optional[Dict[str, Any]]) -> bool:
    return distributed is None or int(distributed["rank"]) == 0

