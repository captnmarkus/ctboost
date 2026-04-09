"""scikit-learn compatible estimator shells for CTBoost."""

from __future__ import annotations

from typing import Any

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
            {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "alpha": self.alpha,
                "loss_function": self.loss_function,
                "task_type": self.task_type,
                "devices": self.devices,
            },
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        return self


class CTBoostClassifier(_BaseCTBoost, ClassifierMixin):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("loss_function", "Logloss")
        super().__init__(**kwargs)

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        eval_set: Any = None,
        early_stopping_rounds: int | None = None,
    ) -> "CTBoostClassifier":
        return self._fit_impl(
            X,
            y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )

    def predict(self, X: Any) -> Any:
        raise NotImplementedError("Prediction is not implemented yet.")

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("Probability prediction is not implemented yet.")


class CTBoostRegressor(_BaseCTBoost, RegressorMixin):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("loss_function", "RMSE")
        super().__init__(**kwargs)

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
        raise NotImplementedError("Prediction is not implemented yet.")
