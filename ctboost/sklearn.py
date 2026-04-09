"""scikit-learn compatible estimator shells for CTBoost."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .core import Pool
from .training import train


class _BaseCTBoost(BaseEstimator):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        loss_function: str | None = None,
        task_type: str = "CPU",
        devices: str | None = None,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.loss_function = loss_function
        self.task_type = task_type
        self.devices = devices

    def _fit_impl(
        self,
        X: Any,
        y: Any = None,
        *,
        eval_set: Any = None,
        early_stopping_rounds: int | None = None,
    ) -> "_BaseCTBoost":
        train_pool = X if isinstance(X, Pool) else Pool(data=X, label=y)
        self._booster = train(
            train_pool,
            {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "alpha": self.alpha,
                "loss_function": self.loss_function,
                "task_type": self.task_type,
                "devices": self.devices,
            },
            num_boost_round=self.iterations,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self

    @staticmethod
    def _prediction_pool(X: Any) -> Pool:
        if isinstance(X, Pool):
            return X
        data = np.asarray(X, dtype=np.float32)
        return Pool(data=data, label=np.zeros(data.shape[0], dtype=np.float32))


class CTBoostClassifier(_BaseCTBoost, ClassifierMixin):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        loss_function: str = "Logloss",
        task_type: str = "CPU",
        devices: str | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            loss_function=loss_function,
            task_type=task_type,
            devices=devices,
        )

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        eval_set: Any = None,
        early_stopping_rounds: int | None = None,
    ) -> "CTBoostClassifier":
        fitted = self._fit_impl(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        labels = np.asarray(y if y is not None else self._prediction_pool(X).label)
        self.classes_ = np.unique(labels)
        if self.classes_.size != 2:
            raise ValueError("CTBoostClassifier currently supports binary classification only.")
        return fitted

    def predict(self, X: Any) -> Any:
        pool = self._prediction_pool(X)
        scores = np.asarray(self._booster.predict(pool), dtype=np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        indices = (probabilities >= 0.5).astype(np.int64)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> Any:
        pool = self._prediction_pool(X)
        scores = np.asarray(self._booster.predict(pool), dtype=np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probabilities, probabilities])


class CTBoostRegressor(_BaseCTBoost, RegressorMixin):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        loss_function: str = "RMSE",
        task_type: str = "CPU",
        devices: str | None = None,
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            loss_function=loss_function,
            task_type=task_type,
            devices=devices,
        )

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        eval_set: Any = None,
        early_stopping_rounds: int | None = None,
    ) -> "CTBoostRegressor":
        return self._fit_impl(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )

    def predict(self, X: Any) -> Any:
        pool = self._prediction_pool(X)
        return np.asarray(self._booster.predict(pool), dtype=np.float32)
