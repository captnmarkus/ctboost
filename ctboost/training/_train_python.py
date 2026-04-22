"""Python-surface training loop for ctboost.training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core import Pool, _clone_pool
from ._distributed_payloads import _distributed_allgather_value, _distributed_broadcast_value
from ._early_stopping import _prune_booster_to_best_iteration
from ._eval_runtime import (
    _collect_prediction_metric_inputs,
    _evaluate_metric_from_inputs,
    _merge_prediction_metric_inputs,
)
from ._train_native import _make_native_booster
from .booster import Booster
from .callbacks import TrainingCallbackEnv
from .resume import (
    _initial_evals_result_from_model,
    _initial_learning_rate_history_from_model,
    _resolve_learning_rate_for_iteration,
)
from .schema import _pool_schema_metadata


def _train_with_python_surface(
    *,
    native_params: Dict[str, Any],
    iterations: int,
    init_state: Optional[Dict[str, Any]],
    distributed_quantization_schema: Optional[Dict[str, Any]],
    weighted_pool: Pool,
    weighted_eval_pools: List[Pool],
    resolved_eval_names: List[str],
    metric_runtime: Dict[str, Any],
    feature_pipeline: Any,
    resolved_init_model: Any,
    resolved_learning_rate_schedule: Any,
    distributed_config: Optional[Dict[str, Any]],
) -> Booster:
    booster = _make_native_booster(
        native_params,
        iterations,
        native_eval_metric=metric_runtime["native_eval_metric"],
        state=init_state,
        distributed_quantization_schema=distributed_quantization_schema,
    )
    wrapped_booster = Booster(
        booster,
        feature_pipeline=feature_pipeline,
        training_metadata=getattr(resolved_init_model, "_training_metadata", None),
    )
    train_pool_template = _clone_pool(weighted_pool, releasable_feature_storage=True)
    seeded_evals_result = _initial_evals_result_from_model(resolved_init_model)
    seeded_learning_rate_history = _initial_learning_rate_history_from_model(resolved_init_model)
    evals_result: Dict[str, Dict[str, List[float]]] = {
        "learn": {"loss": [float(value) for value in booster.loss_history()]}
    }
    for eval_name in resolved_eval_names:
        evals_result[eval_name] = {}
        for metric_name in metric_runtime["eval_metric_names"]:
            if (
                seeded_evals_result is not None
                and eval_name in seeded_evals_result
                and metric_name in seeded_evals_result[eval_name]
            ):
                history = seeded_evals_result[eval_name][metric_name]
            elif (
                len(weighted_eval_pools) == 1
                and eval_name == "validation"
                and metric_name.lower() == metric_runtime["primary_eval_metric"].lower()
                and booster.eval_loss_history()
            ):
                history = booster.eval_loss_history()
            else:
                history = []
            evals_result[eval_name][metric_name] = [float(value) for value in history]
    begin_iteration = int(booster.num_iterations_trained())
    end_iteration = begin_iteration + iterations
    learning_rate_history = (
        [float(value) for value in seeded_learning_rate_history]
        if seeded_learning_rate_history is not None
        else [wrapped_booster.learning_rate] * begin_iteration
    )
    if resolved_learning_rate_schedule is not None and begin_iteration < end_iteration:
        wrapped_booster.set_learning_rate(
            _resolve_learning_rate_for_iteration(
                resolved_learning_rate_schedule,
                iteration=begin_iteration,
                total_iterations=end_iteration,
            )
        )
    if metric_runtime["primary_eval_name"] is not None:
        monitor_history = evals_result[metric_runtime["primary_eval_name"]][metric_runtime["primary_eval_metric"]]
        best_iteration = -1
        best_score = float("-inf") if metric_runtime["primary_metric_direction"] else float("inf")
        for history_index, history_value in enumerate(monitor_history):
            improved = (
                best_iteration < 0
                or (metric_runtime["primary_metric_direction"] and history_value > best_score)
                or ((not metric_runtime["primary_metric_direction"]) and history_value < best_score)
            )
            if improved:
                best_iteration = history_index
                best_score = float(history_value)
    else:
        monitor_history = []
        best_iteration = begin_iteration - 1 if begin_iteration > 0 else -1
        best_score = float(evals_result["learn"]["loss"][best_iteration]) if best_iteration >= 0 and evals_result["learn"]["loss"] else None
    wrapped_booster._set_training_metadata(
        evals_result=evals_result,
        eval_loss_history=monitor_history,
        best_iteration=best_iteration,
        best_score=best_score,
        eval_metric_name=metric_runtime["primary_eval_metric"],
        data_schema=_pool_schema_metadata(weighted_pool),
        learning_rate_history=learning_rate_history,
    )
    stopped_by_early_stopping = False
    callback_stop_requested = False
    for _ in range(iterations):
        booster = wrapped_booster._handle
        current_learning_rate = float(wrapped_booster.learning_rate)
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
        learning_rate_history.append(current_learning_rate)
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
                        f"{native_params['distributed_run_id']}/py_eval/{current_iteration}/{eval_name}",
                        prediction_inputs,
                    )
                )
            for metric_spec in metric_runtime["eval_metric_specs"]:
                metric_name = str(metric_spec["name"])
                metric_value = _evaluate_metric_from_inputs(
                    metric_spec,
                    prediction_inputs,
                    num_classes=native_params["num_classes"],
                    params=native_params,
                )
                evals_result[eval_name][metric_name].append(metric_value)
                evaluation_result_list.append(
                    (
                        eval_name,
                        metric_name,
                        float(metric_value),
                        bool(metric_runtime["metric_directions"][metric_name]),
                    )
                )
        if metric_runtime["primary_eval_name"] is not None:
            current_score = float(
                evals_result[metric_runtime["primary_eval_name"]][metric_runtime["primary_eval_metric"]][-1]
            )
            improved = (
                best_iteration < 0
                or (metric_runtime["primary_metric_direction"] and current_score > float(best_score))
                or ((not metric_runtime["primary_metric_direction"]) and current_score < float(best_score))
            )
            if improved:
                best_iteration = current_iteration
                best_score = current_score
            elif current_iteration - best_iteration >= native_params["early_stopping"]:
                stopped_by_early_stopping = native_params["early_stopping"] > 0
        else:
            best_iteration = current_iteration
            best_score = float(evals_result["learn"]["loss"][-1])
        wrapped_booster._set_training_metadata(
            evals_result=evals_result,
            eval_loss_history=(
                evals_result[metric_runtime["primary_eval_name"]][metric_runtime["primary_eval_metric"]]
                if metric_runtime["primary_eval_name"] is not None
                else []
            ),
            best_iteration=best_iteration,
            best_score=best_score,
            eval_metric_name=metric_runtime["primary_eval_metric"],
            data_schema=_pool_schema_metadata(weighted_pool),
            learning_rate_history=learning_rate_history,
        )
        if metric_runtime["callback_list"]:
            env = TrainingCallbackEnv(
                model=wrapped_booster,
                iteration=current_iteration,
                begin_iteration=begin_iteration,
                end_iteration=end_iteration,
                evaluation_result_list=evaluation_result_list,
                history=wrapped_booster.evals_result_,
                best_iteration=best_iteration,
                best_score=None if best_score is None else float(best_score),
                learning_rate=current_learning_rate,
            )
            if distributed_config is None or int(distributed_config["rank"]) == 0:
                for callback in metric_runtime["callback_list"]:
                    if callback(env):
                        callback_stop_requested = True
                        break
            if distributed_config is not None:
                callback_stop_requested = bool(
                    _distributed_broadcast_value(
                        distributed_config,
                        f"{native_params['distributed_run_id']}/py_callback_stop/{current_iteration}",
                        value=callback_stop_requested,
                    )
                )
        next_iteration = current_iteration + 1
        if resolved_learning_rate_schedule is not None and next_iteration < end_iteration:
            wrapped_booster.set_learning_rate(
                _resolve_learning_rate_for_iteration(
                    resolved_learning_rate_schedule,
                    iteration=next_iteration,
                    total_iterations=end_iteration,
                )
            )
        if stopped_by_early_stopping or callback_stop_requested:
            break
    booster = wrapped_booster._handle
    booster.set_iterations(iterations)
    if native_params["early_stopping"] > 0 and best_iteration >= 0 and best_iteration + 1 < booster.num_iterations_trained():
        wrapped_booster._handle = _prune_booster_to_best_iteration(booster, best_iteration)
        booster = wrapped_booster._handle
        retained_iterations = best_iteration + 1
        learning_rate_history = learning_rate_history[:retained_iterations]
        evals_result["learn"]["loss"] = evals_result["learn"]["loss"][:retained_iterations]
        for eval_name in resolved_eval_names:
            for metric_name in metric_runtime["eval_metric_names"]:
                evals_result[eval_name][metric_name] = evals_result[eval_name][metric_name][:retained_iterations]
    wrapped_booster._set_training_metadata(
        evals_result=evals_result,
        eval_loss_history=(
            evals_result[metric_runtime["primary_eval_name"]][metric_runtime["primary_eval_metric"]]
            if metric_runtime["primary_eval_name"] is not None
            else []
        ),
        best_iteration=best_iteration,
        best_score=best_score,
        eval_metric_name=metric_runtime["primary_eval_metric"],
        data_schema=_pool_schema_metadata(weighted_pool),
        learning_rate_history=learning_rate_history,
    )
    return wrapped_booster
