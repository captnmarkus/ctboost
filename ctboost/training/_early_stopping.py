"""Early-stopping pruning helpers for ctboost.training."""

from __future__ import annotations

from typing import Any

from .. import _core
from .booster import Booster


def _prune_booster_to_best_iteration(handle: Any, best_iteration: int) -> Any:
    state = dict(handle.export_state())
    retained_iterations = int(best_iteration) + 1
    prediction_dimension = int(handle.prediction_dimension())
    state["trees"] = state["trees"][: retained_iterations * prediction_dimension]
    if "tree_learning_rates" in state:
        state["tree_learning_rates"] = state["tree_learning_rates"][:retained_iterations]
    state["loss_history"] = state["loss_history"][:retained_iterations]
    state["eval_loss_history"] = state["eval_loss_history"][:retained_iterations]
    state["best_iteration"] = int(best_iteration)
    if retained_iterations > 0:
        eval_history = state["eval_loss_history"]
        if eval_history:
            state["best_score"] = float(eval_history[retained_iterations - 1])
        else:
            state["best_score"] = float(state["loss_history"][retained_iterations - 1])
    return _core.GradientBooster.from_state(state)


def _finalize_early_stopping(booster: Booster) -> Booster:
    best_iteration = booster.best_iteration
    if best_iteration < 0 or best_iteration + 1 >= booster.num_iterations_trained:
        return booster
    retained_iterations = best_iteration + 1
    booster._handle = _prune_booster_to_best_iteration(booster._handle, best_iteration)
    metadata = getattr(booster, "_training_metadata", None)
    if metadata is None:
        return booster
    trimmed = dict(metadata)
    if "eval_loss_history" in trimmed:
        trimmed["eval_loss_history"] = list(trimmed["eval_loss_history"])[:retained_iterations]
    if "learning_rate_history" in trimmed:
        trimmed["learning_rate_history"] = list(trimmed["learning_rate_history"])[:retained_iterations]
    if "evals_result" in trimmed:
        trimmed["evals_result"] = {
            dataset_name: {
                metric_name: list(history)[:retained_iterations]
                for metric_name, history in metric_histories.items()
            }
            for dataset_name, metric_histories in trimmed["evals_result"].items()
        }
    booster._training_metadata = trimmed
    return booster
