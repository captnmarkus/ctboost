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
        lambda_l2: float = 1.0,
        loss_function: str | None = None,
        task_type: str = "CPU",
        devices: str = "0",
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.lambda_l2 = lambda_l2
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
        objective: str,
        num_classes: int = 1,
    ) -> "_BaseCTBoost":
        train_pool = X if isinstance(X, Pool) else Pool(data=X, label=y)
        resolved_objective = objective if num_classes > 2 else (self.loss_function or objective)
        self._booster = train(
            train_pool,
            {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "alpha": self.alpha,
                "lambda_l2": self.lambda_l2,
                "objective": resolved_objective,
                "num_classes": num_classes,
                "task_type": self.task_type,
                "devices": self.devices,
            },
            num_boost_round=self.iterations,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.n_features_in_ = train_pool.num_cols
        return self

    @staticmethod
    def _prediction_pool(X: Any) -> Pool:
        if isinstance(X, Pool):
            return X
        data = np.asarray(X, dtype=np.float32)
        return Pool(data=data, label=np.zeros(data.shape[0], dtype=np.float32))

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._booster.feature_importances_, dtype=np.float32)


class CTBoostClassifier(_BaseCTBoost, ClassifierMixin):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        lambda_l2: float = 1.0,
        loss_function: str = "Logloss",
        task_type: str = "CPU",
        devices: str = "0",
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            lambda_l2=lambda_l2,
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
        if isinstance(X, Pool):
            raw_labels = np.asarray(X.label if y is None else y)
            if raw_labels.ndim != 1:
                raise ValueError("classification labels must be a 1D array")
            classes, inverse = np.unique(raw_labels, return_inverse=True)
            encoded_labels = inverse.astype(np.float32, copy=False)
            train_pool = Pool(
                data=X.data,
                label=encoded_labels,
                cat_features=X.cat_features,
                weight=X.weight,
                feature_names=X.feature_names,
            )
        else:
            if y is None:
                raise ValueError("y must be provided when X is not a Pool")
            raw_labels = np.asarray(y)
            if raw_labels.ndim != 1:
                raise ValueError("classification labels must be a 1D array")
            classes, inverse = np.unique(raw_labels, return_inverse=True)
            encoded_labels = inverse.astype(np.float32, copy=False)
            train_pool = Pool(data=X, label=encoded_labels)

        self.classes_ = classes
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError("CTBoostClassifier requires at least two distinct classes.")

        objective = "MultiClass" if self.n_classes_ > 2 else "Logloss"
        fitted = self._fit_impl(
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            objective=objective,
            num_classes=self.n_classes_,
        )
        return fitted

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> Any:
        pool = self._prediction_pool(X)
        probabilities = self.predict_proba(pool)
        if self.n_classes_ > 2:
            indices = np.argmax(probabilities, axis=1)
        else:
            indices = (probabilities[:, 1] >= 0.5).astype(np.int64)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> Any:
        pool = self._prediction_pool(X)
        scores = np.asarray(self._booster.predict(pool), dtype=np.float32)
        if self.n_classes_ > 2:
            if scores.ndim == 1:
                scores = scores.reshape((pool.num_rows, self.n_classes_))
            return self._softmax(scores)
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
        lambda_l2: float = 1.0,
        loss_function: str = "RMSE",
        task_type: str = "CPU",
        devices: str = "0",
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            lambda_l2=lambda_l2,
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
            objective="SquaredError",
        )

    def predict(self, X: Any) -> Any:
        pool = self._prediction_pool(X)
        return np.asarray(self._booster.predict(pool), dtype=np.float32)


CBoostClassifier = CTBoostClassifier
CBoostRegressor = CTBoostRegressor
