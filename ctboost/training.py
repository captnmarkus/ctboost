"""Training entry points for the Python API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Optional

import numpy as np

from . import _core
from .core import Pool


def _pool_from_data_and_label(data: Any, label: Any) -> Pool:
    if isinstance(data, Pool):
        resolved_label = data.label if label is None else label
        return Pool(
            data=data.data,
            label=resolved_label,
            cat_features=data.cat_features,
            weight=data.weight,
            feature_names=data.feature_names,
        )

    if label is None:
        raise ValueError("y must be provided when data is not a Pool")
    return Pool(data=data, label=label)


def _normalize_eval_set(eval_set: Any) -> Optional[Any]:
    if eval_set is None:
        return None
    if isinstance(eval_set, Pool):
        return eval_set
    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        return eval_set
    if (
        isinstance(eval_set, list)
        and len(eval_set) == 1
        and isinstance(eval_set[0], tuple)
        and len(eval_set[0]) == 2
    ):
        return eval_set[0]
    raise TypeError(
        "eval_set must be a Pool, a single (X, y) tuple, or a list containing one (X, y) tuple"
    )


def _resolve_eval_pool(eval_set: Any) -> Optional[Pool]:
    normalized = _normalize_eval_set(eval_set)
    if normalized is None or isinstance(normalized, Pool):
        return normalized

    X_eval, y_eval = normalized
    return _pool_from_data_and_label(X_eval, y_eval)


class Booster:
    """Small Python wrapper around the native gradient booster."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def predict(self, data: Any) -> np.ndarray:
        pool = data if isinstance(data, Pool) else Pool(
            data=data,
            label=np.zeros(len(data), dtype=np.float32),
        )
        raw = np.asarray(self._handle.predict(pool._handle), dtype=np.float32)
        prediction_dimension = int(self._handle.prediction_dimension())
        if prediction_dimension > 1:
            return raw.reshape((pool.num_rows, prediction_dimension))
        return raw

    @property
    def loss_history(self) -> List[float]:
        return list(self._handle.loss_history())

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._handle.feature_importances(), dtype=np.float32)

    @property
    def num_classes(self) -> int:
        return int(self._handle.num_classes())

    @property
    def prediction_dimension(self) -> int:
        return int(self._handle.prediction_dimension())

    @property
    def best_iteration(self) -> int:
        return int(self._handle.best_iteration())


def train(
    pool: Pool,
    params: Mapping[str, Any],
    num_boost_round: Optional[int] = None,
    *,
    eval_set: Any = None,
    early_stopping_rounds: Optional[int] = None,
) -> Booster:
    if not isinstance(pool, Pool):
        raise TypeError("pool must be an instance of ctboost.Pool")

    eval_pool = _resolve_eval_pool(eval_set)
    if early_stopping_rounds is not None:
        if early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be a positive integer")
        if eval_pool is None:
            raise ValueError("early_stopping_rounds requires eval_set")
    early_stopping = 0 if early_stopping_rounds is None else int(early_stopping_rounds)

    config = dict(params)
    iterations = int(num_boost_round if num_boost_round is not None else config.get("iterations", 100))
    objective = str(config.get("objective", config.get("loss_function", "RMSE")))
    learning_rate = float(config.get("learning_rate", 0.1))
    max_depth = int(config.get("max_depth", 6))
    alpha = float(config.get("alpha", 0.05))
    lambda_l2 = float(config.get("lambda", config.get("lambda_l2", 1.0)))
    if "num_classes" in config:
        num_classes = int(config["num_classes"])
    elif objective.lower() in {"multiclass", "softmax", "softmaxloss"}:
        num_classes = int(np.unique(pool.label).size)
    else:
        num_classes = 1
    max_bins = int(config.get("max_bins", 256))
    task_type = str(config.get("task_type", "CPU"))
    devices = str(config.get("devices", "0"))

    booster = _core.GradientBooster(
        objective=objective,
        iterations=iterations,
        learning_rate=learning_rate,
        max_depth=max_depth,
        alpha=alpha,
        lambda_l2=lambda_l2,
        num_classes=num_classes,
        max_bins=max_bins,
        task_type=task_type,
        devices=devices,
    )
    booster.fit(
        pool._handle,
        None if eval_pool is None else eval_pool._handle,
        early_stopping,
    )
    return Booster(booster)
