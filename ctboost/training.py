"""Training entry points for the Python API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _core
from .core import Pool


class Booster:
    """Small Python wrapper around the native gradient booster."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def predict(self, data: Pool | Any) -> np.ndarray:
        pool = data if isinstance(data, Pool) else Pool(
            data=data,
            label=np.zeros(np.asarray(data).shape[0], dtype=np.float32),
        )
        return self._handle.predict(pool._handle)

    @property
    def loss_history(self) -> list[float]:
        return list(self._handle.loss_history())


def train(
    pool: Pool,
    params: Mapping[str, Any],
    num_boost_round: int | None = None,
    *,
    eval_set: Any = None,
    early_stopping_rounds: int | None = None,
) -> Booster:
    if not isinstance(pool, Pool):
        raise TypeError("pool must be an instance of ctboost.Pool")
    if eval_set is not None:
        raise NotImplementedError("eval_set is not implemented yet.")
    if early_stopping_rounds is not None:
        raise NotImplementedError("early_stopping_rounds is not implemented yet.")

    config = dict(params)
    iterations = int(num_boost_round if num_boost_round is not None else config.get("iterations", 100))
    objective = str(config.get("objective", config.get("loss_function", "RMSE")))
    learning_rate = float(config.get("learning_rate", 0.1))
    max_depth = int(config.get("max_depth", 6))
    alpha = float(config.get("alpha", 0.05))
    lambda_l2 = float(config.get("lambda", config.get("lambda_l2", 1.0)))
    max_bins = int(config.get("max_bins", 256))

    booster = _core.GradientBooster(
        objective=objective,
        iterations=iterations,
        learning_rate=learning_rate,
        max_depth=max_depth,
        alpha=alpha,
        lambda_l2=lambda_l2,
        max_bins=max_bins,
    )
    booster.fit(pool._handle)
    return Booster(booster)
