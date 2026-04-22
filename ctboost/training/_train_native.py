"""Native training execution for ctboost.training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import _core
from ..core import Pool
from .booster import Booster
from .resume import _initial_learning_rate_history_from_model
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
        native_booster.load_state(state)
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
) -> Booster:
    booster = _make_native_booster(
        native_params,
        iterations,
        native_eval_metric=native_eval_metric,
        state=init_state,
        distributed_quantization_schema=distributed_quantization_schema,
    )
    native_eval_pool = weighted_eval_pools[0] if weighted_eval_pools else None
    seeded_learning_rate_history = _initial_learning_rate_history_from_model(resolved_init_model)
    booster.fit(
        weighted_pool._handle,
        None if native_eval_pool is None else native_eval_pool._handle,
        early_stopping,
        init_state is not None,
    )
    trained_booster = Booster(
        booster,
        feature_pipeline=feature_pipeline,
        training_metadata=getattr(resolved_init_model, "_training_metadata", None),
    )
    if seeded_learning_rate_history is None:
        learning_rate_history = [trained_booster.learning_rate] * trained_booster.num_iterations_trained
    else:
        learning_rate_history = [float(value) for value in seeded_learning_rate_history]
        learning_rate_history.extend(
            [trained_booster.learning_rate] * max(trained_booster.num_iterations_trained - len(learning_rate_history), 0)
        )
    trained_booster._set_training_metadata(
        data_schema=_pool_schema_metadata(weighted_pool),
        learning_rate_history=learning_rate_history,
    )
    if resolved_snapshot_path is not None:
        trained_booster.save_model(resolved_snapshot_path, model_format=snapshot_model_format)
    return trained_booster
