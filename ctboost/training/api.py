"""Training entry points for the Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np

from ..core import Pool
from ._eval_sets import _load_sklearn_splitters, _normalize_eval_names, _slice_pool, _stack_histories
from ._early_stopping import _finalize_early_stopping
from ._train_metrics import _resolve_metric_runtime
from ._train_native import _train_native_only
from ._train_params import _normalize_callback_list, _resolve_native_training_params
from ._train_python import _train_with_python_surface
from .booster import Booster
from .callbacks import TrainingCallbackEnv, checkpoint_callback
from .distributed import (
    _distributed_collective_context,
    _distributed_train_filesystem_compat,
    _normalize_distributed_config,
    _resolve_distributed_quantization_schema,
)
from .pool_io import PreparedTrainingData, prepare_training_data
from .resume import (
    _initial_learning_rate_history_from_model,
    _normalize_learning_rate_schedule,
    _resolve_init_model_handle,
    _resolve_resume_snapshot_path,
    _resolve_snapshot_path,
    _resume_snapshot_config_signature,
    _validate_resume_snapshot_contract,
)
from .schema import _is_classification_objective, _is_ranking_objective, _objective_name, _validate_distributed_pool_metadata
from .weights import _apply_objective_weights

PathLike = Union[str, Path]


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
        split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32), labels, groups=np.asarray(pool.group_id))
    else:
        use_stratified = _is_classification_objective(objective) if stratified is None else bool(stratified)
        if use_stratified:
            splitter = StratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=random_state if shuffle else None)
            split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32), labels)
        else:
            splitter = KFold(n_splits=nfold, shuffle=shuffle, random_state=random_state if shuffle else None)
            split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32))
    train_histories: List[np.ndarray] = []
    valid_histories: List[np.ndarray] = []
    best_iterations: List[int] = []
    for train_index, valid_index in split_iterator:
        booster = train(
            _slice_pool(pool, train_index),
            params,
            num_boost_round=num_boost_round,
            eval_set=_slice_pool(pool, valid_index),
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
    early_stopping_rounds: Optional[int] = None,
    early_stopping_metric: Optional[Any] = None,
    early_stopping_name: Optional[str] = None,
    callbacks: Optional[Iterable[Callable[[TrainingCallbackEnv], Any]]] = None,
    learning_rate_schedule: Any = None,
    snapshot_path: Optional[PathLike] = None,
    snapshot_interval: int = 1,
    snapshot_save_best_only: bool = False,
    snapshot_model_format: Optional[str] = None,
    resume_from_snapshot: Any = None,
    init_model: Any = None,
) -> Booster:
    config = dict(params)
    distributed_config = _normalize_distributed_config(config)
    resolved_snapshot_path = _resolve_snapshot_path(snapshot_path)
    resolved_resume_snapshot_path = _resolve_resume_snapshot_path(resume_from_snapshot, resolved_snapshot_path)
    resolved_init_model = init_model
    if resolved_resume_snapshot_path is not None:
        if resolved_init_model is not None:
            raise ValueError("resume_from_snapshot cannot be combined with init_model")
        if not resolved_resume_snapshot_path.exists():
            raise FileNotFoundError(f"snapshot file does not exist: {resolved_resume_snapshot_path}")
        resolved_init_model = Booster.load_model(resolved_resume_snapshot_path)
    prepared_inputs = prepare_training_data(
        pool,
        config,
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
        eval_set=eval_set,
        eval_names=eval_names,
        init_model=resolved_init_model,
    )
    pool = prepared_inputs.pool
    eval_pools = list(prepared_inputs.eval_pools)
    resolved_eval_names = list(prepared_inputs.eval_names)
    default_eval_names = _normalize_eval_names(None, len(eval_pools))
    feature_pipeline = prepared_inputs.feature_pipeline or getattr(pool, "_feature_pipeline", None)
    if distributed_config is not None:
        _validate_distributed_pool_metadata(pool, context="the training pool")
        for eval_index, eval_pool in enumerate(eval_pools):
            _validate_distributed_pool_metadata(eval_pool, context=f"eval_set[{eval_index}]")
    user_callback_list = _normalize_callback_list(callbacks)
    resolved_learning_rate_schedule = _normalize_learning_rate_schedule(learning_rate_schedule)
    snapshot_callback = None if resolved_snapshot_path is None else checkpoint_callback(
        resolved_snapshot_path,
        interval=int(snapshot_interval),
        model_format=snapshot_model_format,
        save_best_only=bool(snapshot_save_best_only),
    )
    if early_stopping_rounds is not None and early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be a positive integer")
    if early_stopping_rounds is not None and not eval_pools:
        raise ValueError("early_stopping_rounds requires eval_set")
    early_stopping = 0 if early_stopping_rounds is None else int(early_stopping_rounds)
    requested_iterations = int(num_boost_round if num_boost_round is not None else config.get("iterations", 100))
    init_handle = _resolve_init_model_handle(resolved_init_model)
    init_state = None if init_handle is None else dict(init_handle.export_state())
    iterations = requested_iterations
    if resolved_resume_snapshot_path is not None:
        trained_iterations = int(init_handle.num_iterations_trained())
        if requested_iterations < trained_iterations:
            raise ValueError("requested num_boost_round is smaller than the snapshot iteration count")
        iterations = requested_iterations - trained_iterations
        if iterations == 0:
            return Booster(init_handle, feature_pipeline=feature_pipeline, training_metadata=getattr(resolved_init_model, "_training_metadata", None))
    native_params = _resolve_native_training_params(config, pool, init_state=init_state)
    native_params["early_stopping"] = early_stopping
    metric_runtime = _resolve_metric_runtime(
        config=config,
        init_state=init_state,
        eval_pools=eval_pools,
        resolved_eval_names=resolved_eval_names,
        default_eval_names=default_eval_names,
        early_stopping_rounds=early_stopping_rounds,
        early_stopping_metric=early_stopping_metric,
        early_stopping_name=early_stopping_name,
        user_callback_list=user_callback_list,
        learning_rate_schedule=resolved_learning_rate_schedule,
        snapshot_callback=snapshot_callback,
        distributed_config=distributed_config,
    )
    filesystem_compat_required = distributed_config is not None and distributed_config["backend"] != "tcp" and (
        native_params["task_type"].upper() == "GPU" or bool(eval_pools) or metric_runtime["use_python_eval_surface"] or pool.group_id is not None
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
            callbacks=metric_runtime["callback_list"],
            init_model=resolved_init_model,
        )
    with _distributed_collective_context(distributed_config):
        distributed_quantization_schema = None
        if distributed_config is not None and init_state is None:
            distributed_quantization_schema = _resolve_distributed_quantization_schema(
                pool,
                distributed_config,
                max_bins=native_params["max_bins"],
                nan_mode=native_params["nan_mode"],
                max_bin_by_feature=native_params["max_bin_by_feature"],
                border_selection_method=native_params["border_selection_method"],
                nan_mode_by_feature=native_params["nan_mode_by_feature"],
                feature_borders=native_params["feature_borders"],
                external_memory=native_params["native_external_memory"],
            )
        if init_state is not None:
            if native_params["objective"].lower() != str(init_state["objective_name"]).lower():
                raise ValueError("init_model objective must match the current training objective")
            if native_params["num_classes"] != int(init_state["num_classes"]):
                raise ValueError("init_model num_classes must match the current training configuration")
        weighted_pool = _apply_objective_weights(pool, config, native_params["objective"])
        weighted_eval_pools = [_apply_objective_weights(eval_pool, config, native_params["objective"]) for eval_pool in eval_pools]
        if resolved_resume_snapshot_path is not None:
            _validate_resume_snapshot_contract(
                init_state=init_state,
                init_model=resolved_init_model,
                requested_signature=_resume_snapshot_config_signature(pool=weighted_pool, **native_params),
                learning_rate_schedule=resolved_learning_rate_schedule,
                requested_iterations=requested_iterations,
            )
        if not metric_runtime["use_python_eval_surface"]:
            return _train_native_only(
                native_params=native_params,
                iterations=iterations,
                early_stopping=early_stopping,
                init_state=init_state,
                distributed_quantization_schema=distributed_quantization_schema,
                weighted_pool=weighted_pool,
                weighted_eval_pools=weighted_eval_pools,
                feature_pipeline=feature_pipeline,
                resolved_init_model=resolved_init_model,
                resolved_snapshot_path=resolved_snapshot_path,
                snapshot_model_format=snapshot_model_format,
                native_eval_metric=metric_runtime["native_eval_metric"],
            )
        trained_booster = _train_with_python_surface(
            native_params=native_params,
            iterations=iterations,
            init_state=init_state,
            distributed_quantization_schema=distributed_quantization_schema,
            weighted_pool=weighted_pool,
            weighted_eval_pools=weighted_eval_pools,
            resolved_eval_names=resolved_eval_names,
            metric_runtime=metric_runtime,
            feature_pipeline=feature_pipeline,
            resolved_init_model=resolved_init_model,
            resolved_learning_rate_schedule=resolved_learning_rate_schedule,
            distributed_config=distributed_config,
        )
        return _finalize_early_stopping(trained_booster) if early_stopping > 0 else trained_booster
