"""Training callbacks for CTBoost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .booster import Booster

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
    learning_rate: float

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

