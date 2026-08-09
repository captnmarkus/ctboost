"""Native training execution for ctboost.training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import _core
from ..core import Pool
from .booster import Booster
from .resume import _initial_evals_result_from_model, _initial_learning_rate_history_from_model
from .schema import _pool_schema_metadata


def _make_native_booster(
    native_params: Dict[str, Any],
    iterations: int,
    *,
    native_eval_metric: str,
    state: Optional[Dict[str, Any]] = None,
    distributed_quantization_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    native_booster = _core.GradientBooster(
        objective=native_params["objective"],
        iterations=iterations,
        learning_rate=native_params["learning_rate"],
        max_depth=native_params["max_depth"],
        alpha=native_params["alpha"],
        lambda_l2=native_params["lambda_l2"],
        subsample=native_params["subsample"],
        bootstrap_type=native_params["bootstrap_type"],
        bagging_temperature=native_params["bagging_temperature"],
        boosting_type=native_params["boosting_type"],
        drop_rate=native_params["drop_rate"],
        skip_drop=native_params["skip_drop"],
        max_drop=native_params["max_drop"],
        monotone_constraints=native_params["monotone_constraints"],
        interaction_constraints=native_params["interaction_constraints"],
        colsample_bytree=native_params["colsample_bytree"],
        feature_weights=native_params["feature_weights"],
        first_feature_use_penalties=native_params["first_feature_use_penalties"],
        random_strength=native_params["random_strength"],
        grow_policy=native_params["grow_policy"],
        max_leaves=native_params["max_leaves"],
        min_samples_split=native_params["min_samples_split"],
        min_data_in_leaf=native_params["min_data_in_leaf"],
        min_child_weight=native_params["min_child_weight"],
        gamma=native_params["gamma"],
        max_leaf_weight=native_params["max_leaf_weight"],
        num_classes=native_params["num_classes"],
        max_bins=native_params["max_bins"],
        nan_mode=native_params["nan_mode"],
        max_bin_by_feature=native_params["max_bin_by_feature"],
        border_selection_method=native_params["border_selection_method"],
        nan_mode_by_feature=native_params["nan_mode_by_feature"],
        feature_borders=native_params["feature_borders"],
        external_memory=native_params["native_external_memory"],
        external_memory_dir=native_params["native_external_memory_dir"],
        eval_metric=native_eval_metric,
        quantile_alpha=native_params["quantile_alpha"],
        huber_delta=native_params["huber_delta"],
        tweedie_variance_power=native_params["tweedie_variance_power"],
        task_type=native_params["task_type"],
        devices=native_params["devices"],
        distributed_world_size=native_params["distributed_world_size"],
        distributed_rank=native_params["distributed_rank"],
        distributed_root=native_params["distributed_root"],
        distributed_run_id=native_params["distributed_run_id"],
        distributed_timeout=native_params["distributed_timeout"],
        random_seed=native_params["random_seed"],
        verbose=native_params["verbose"],
    )
    if state is not None:
        # Persisted model state deliberately excludes the ephemeral TCP bearer
        # token.  Runtime distributed settings are authoritative when a
        # snapshot is resumed and must survive load_state().
        runtime_state = dict(state)
        for key in (
            "distributed_world_size",
            "distributed_rank",
            "distributed_root",
            "distributed_run_id",
            "distributed_timeout",
        ):
            runtime_state[key] = native_params[key]
        native_booster.load_state(runtime_state)
    elif distributed_quantization_schema is not None:
        native_booster.load_quantization_schema(distributed_quantization_schema)
    return native_booster


def _train_native_only(
    *,
    native_params: Dict[str, Any],
    iterations: int,
    early_stopping: int,
    init_state: Optional[Dict[str, Any]],
    distributed_quantization_schema: Optional[Dict[str, Any]],
    weighted_pool: Pool,
    weighted_eval_pools: List[Pool],
    feature_pipeline: Any,
    resolved_init_model: Any,
    resolved_snapshot_path: Optional[Path],
    snapshot_model_format: Optional[str],
    native_eval_metric: str,
    native_eval_name: Optional[str],
    reported_eval_metric: str,
) -> Booster:
    booster = _make_native_booster(
        native_params,
        iterations,
        native_eval_metric=native_eval_metric,
        state=init_state,
        distributed_quantization_schema=distributed_quantization_schema,
    )
    native_eval_pool = weighted_eval_pools[0] if weighted_eval_pools else None
    begin_iteration = int(booster.num_iterations_trained())
    seeded_learning_rate_history = _initial_learning_rate_history_from_model(resolved_init_model)
    seeded_evals_result = _initial_evals_result_from_model(resolved_init_model)
    booster.fit(
        weighted_pool._handle,
        None if native_eval_pool is None else native_eval_pool._handle,
        early_stopping,
        init_state is not None,
    )
    trained_booster = Booster(
        booster,
        feature_pipeline=feature_pipeline,
        training_metadata=None,
    )
    if seeded_learning_rate_history is None:
        learning_rate_history = [trained_booster.learning_rate] * trained_booster.num_iterations_trained
    else:
        learning_rate_history = [
            float(value)
            for value in seeded_learning_rate_history[:begin_iteration]
        ]
        learning_rate_history.extend(
            [trained_booster.learning_rate] * max(trained_booster.num_iterations_trained - len(learning_rate_history), 0)
        )
        learning_rate_history = learning_rate_history[:trained_booster.num_iterations_trained]
    train_loss_history = [float(value) for value in booster.loss_history()]
    metadata: Dict[str, Any] = {
        "evals_result": {"learn": {"loss": train_loss_history}},
        "eval_loss_history": [],
        "best_iteration": int(booster.best_iteration()),
        "eval_metric_name": reported_eval_metric,
        "data_schema": _pool_schema_metadata(weighted_pool),
        "learning_rate_history": learning_rate_history,
    }
    if native_eval_pool is not None:
        resolved_eval_name = native_eval_name or "validation"
        result_metric_name = (
            "loss"
            if native_eval_name is None
            and reported_eval_metric.lower() == native_params["objective"].lower()
            else reported_eval_metric
        )
        native_eval_history = [float(value) for value in booster.eval_loss_history()]
        trained_iterations = int(booster.num_iterations_trained())
        seeded_eval_history: List[float] = []
        if seeded_evals_result is not None and resolved_eval_name in seeded_evals_result:
            seeded_metrics = seeded_evals_result[resolved_eval_name]
            matching_metric_name = next(
                (
                    metric_name
                    for metric_name in (reported_eval_metric, result_metric_name)
                    if metric_name in seeded_metrics
                ),
                None,
            )
            if matching_metric_name is None:
                matching_metric_name = next(
                    (
                        metric_name
                        for metric_name in seeded_metrics
                        if metric_name.lower() == reported_eval_metric.lower()
                    ),
                    None,
                )
            if matching_metric_name is not None:
                seeded_eval_history = [float(value) for value in seeded_metrics[matching_metric_name]]

        if len(native_eval_history) >= trained_iterations:
            eval_loss_history = native_eval_history[:trained_iterations]
        elif seeded_eval_history:
            newly_trained_iterations = max(trained_iterations - begin_iteration, 0)
            new_eval_history = (
                native_eval_history[-newly_trained_iterations:]
                if newly_trained_iterations > 0
                else []
            )
            eval_loss_history = seeded_eval_history[:begin_iteration] + new_eval_history
        else:
            eval_loss_history = native_eval_history

        metadata.update(
            evals_result={
                "learn": {"loss": train_loss_history},
                resolved_eval_name: {result_metric_name: eval_loss_history},
            },
            eval_loss_history=eval_loss_history,
        )
    best_iteration = int(booster.best_iteration())
    score_history = metadata["eval_loss_history"] if native_eval_pool is not None else train_loss_history
    if 0 <= best_iteration < len(score_history):
        metadata["best_score"] = float(score_history[best_iteration])
    trained_booster._set_training_metadata(**metadata)
    if resolved_snapshot_path is not None:
        trained_booster.save_model(resolved_snapshot_path, model_format=snapshot_model_format)
    return trained_booster
