"""scikit-learn compatible estimator shells for CTBoost."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .core import Pool
from .training import _apply_sample_weight_to_pool, _normalize_eval_set, _pool_from_data_and_label, train

PathLike = Union[str, Path]


def _encode_class_labels(raw_labels: Any, label_to_index: Dict[Any, int], label_name: str) -> np.ndarray:
    label_array = np.asarray(raw_labels)
    if label_array.ndim != 1:
        raise ValueError("classification labels must be a 1D array")

    try:
        encoded = [label_to_index[label] for label in label_array]
    except KeyError as exc:
        raise ValueError(f"{label_name} contains labels not seen in the training data") from exc
    return np.asarray(encoded, dtype=np.float32)


def _resolve_classifier_eval_pool(
    eval_set: Any, label_to_index: Dict[Any, int]
) -> Optional[Pool]:
    normalized = _normalize_eval_set(eval_set)
    if normalized is None or isinstance(normalized, Pool):
        return normalized

    X_eval, y_eval = normalized
    encoded_labels = _encode_class_labels(y_eval, label_to_index, "eval_set")
    return _pool_from_data_and_label(X_eval, encoded_labels)


class _BaseCTBoost(BaseEstimator):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        lambda_l2: float = 1.0,
        loss_function: Optional[str] = None,
        eval_metric: Optional[str] = None,
        nan_mode: str = "Min",
        task_type: str = "CPU",
        devices: str = "0",
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.lambda_l2 = lambda_l2
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.nan_mode = nan_mode
        self.task_type = task_type
        self.devices = devices

    def _fit_impl(
        self,
        X: Any,
        y: Any = None,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        early_stopping_rounds: Optional[int] = None,
        objective: str,
        num_classes: int = 1,
        extra_train_params: Optional[Dict[str, Any]] = None,
    ) -> "_BaseCTBoost":
        train_pool = X if isinstance(X, Pool) and y is None else _pool_from_data_and_label(X, y)
        train_pool = _apply_sample_weight_to_pool(train_pool, sample_weight)
        resolved_objective = objective if num_classes > 2 else (self.loss_function or objective)
        train_params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "alpha": self.alpha,
            "lambda_l2": self.lambda_l2,
            "objective": resolved_objective,
            "num_classes": num_classes,
            "nan_mode": self.nan_mode,
            "task_type": self.task_type,
            "devices": self.devices,
        }
        if self.eval_metric is not None:
            train_params["eval_metric"] = self.eval_metric
        if extra_train_params:
            train_params.update(extra_train_params)
        self._booster = train(
            train_pool,
            train_params,
            num_boost_round=self.iterations,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
        )
        self.n_features_in_ = train_pool.num_cols
        self.best_iteration_ = self._booster.best_iteration
        self.evals_result_ = self._booster.evals_result_
        self.best_score_ = self._compute_best_score()
        return self

    @staticmethod
    def _prediction_pool(X: Any) -> Pool:
        if isinstance(X, Pool):
            return X
        return Pool(data=X, label=np.zeros(len(X), dtype=np.float32))

    def _compute_best_score(self) -> Dict[str, float]:
        best_score: Dict[str, float] = {}
        if self._booster.loss_history:
            best_index = self.best_iteration_ if self.best_iteration_ >= 0 else len(self._booster.loss_history) - 1
            best_score["learn"] = float(self._booster.loss_history[best_index])
        if self._booster.eval_loss_history:
            best_index = self.best_iteration_ if self.best_iteration_ >= 0 else len(self._booster.eval_loss_history) - 1
            best_score["validation"] = float(self._booster.eval_loss_history[best_index])
        return best_score

    def save_model(self, path: PathLike) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, path: PathLike) -> "_BaseCTBoost":
        with Path(path).open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError(f"serialized model does not contain a {cls.__name__} instance")
        return model

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
        class_weight: Optional[Any] = None,
        scale_pos_weight: Optional[float] = None,
        eval_metric: Optional[str] = None,
        nan_mode: str = "Min",
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
            eval_metric=eval_metric,
            nan_mode=nan_mode,
            task_type=task_type,
            devices=devices,
        )
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> "CTBoostClassifier":
        if isinstance(X, Pool):
            raw_labels = np.asarray(X.label if y is None else y)
        else:
            if y is None:
                raise ValueError("y must be provided when X is not a Pool")
            raw_labels = np.asarray(y)
        if raw_labels.ndim != 1:
            raise ValueError("classification labels must be a 1D array")

        classes, inverse = np.unique(raw_labels, return_inverse=True)
        encoded_labels = inverse.astype(np.float32, copy=False)
        train_pool = _pool_from_data_and_label(X, encoded_labels)

        self.classes_ = classes
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError("CTBoostClassifier requires at least two distinct classes.")

        label_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        eval_pool = _resolve_classifier_eval_pool(eval_set, label_to_index)
        objective = "MultiClass" if self.n_classes_ > 2 else "Logloss"
        train_params: Dict[str, Any] = {}
        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight == "balanced":
                train_params["auto_class_weights"] = "balanced"
            elif isinstance(self.class_weight, dict):
                encoded_class_weights = {}
                for label, weight in self.class_weight.items():
                    if label not in label_to_index:
                        raise ValueError("class_weight contains labels not seen in the training data")
                    encoded_class_weights[label_to_index[label]] = weight
                train_params["class_weights"] = encoded_class_weights
            else:
                train_params["class_weights"] = self.class_weight
        if self.scale_pos_weight is not None:
            train_params["scale_pos_weight"] = self.scale_pos_weight
        fitted = self._fit_impl(
            train_pool,
            sample_weight=sample_weight,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds,
            objective=objective,
            num_classes=self.n_classes_,
            extra_train_params=train_params,
        )
        return fitted

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._prediction_pool(X)
        probabilities = self.predict_proba(pool, num_iteration=num_iteration)
        if self.n_classes_ > 2:
            indices = np.argmax(probabilities, axis=1)
        else:
            indices = (probabilities[:, 1] >= 0.5).astype(np.int64)
        return self.classes_[indices]

    def predict_proba(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._prediction_pool(X)
        scores = np.asarray(self._booster.predict(pool, num_iteration=num_iteration), dtype=np.float32)
        if self.n_classes_ > 2:
            if scores.ndim == 1:
                scores = scores.reshape((pool.num_rows, self.n_classes_))
            return self._softmax(scores)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probabilities, probabilities])

    def staged_predict_proba(self, X: Any) -> Iterable[Any]:
        pool = self._prediction_pool(X)
        for scores in self._booster.staged_predict(pool):
            scores = np.asarray(scores, dtype=np.float32)
            if self.n_classes_ > 2:
                if scores.ndim == 1:
                    scores = scores.reshape((pool.num_rows, self.n_classes_))
                yield self._softmax(scores)
            else:
                probabilities = 1.0 / (1.0 + np.exp(-scores))
                yield np.column_stack([1.0 - probabilities, probabilities])

    def staged_predict(self, X: Any) -> Iterable[Any]:
        for probabilities in self.staged_predict_proba(X):
            if self.n_classes_ > 2:
                indices = np.argmax(probabilities, axis=1)
            else:
                indices = (probabilities[:, 1] >= 0.5).astype(np.int64)
            yield self.classes_[indices]


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
        eval_metric: Optional[str] = None,
        nan_mode: str = "Min",
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
            eval_metric=eval_metric,
            nan_mode=nan_mode,
            task_type=task_type,
            devices=devices,
        )

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        sample_weight: Any = None,
        eval_set: Any = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> "CTBoostRegressor":
        return self._fit_impl(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            objective="SquaredError",
        )

    def predict(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._prediction_pool(X)
        return np.asarray(self._booster.predict(pool, num_iteration=num_iteration), dtype=np.float32)

    def staged_predict(self, X: Any) -> Iterable[Any]:
        pool = self._prediction_pool(X)
        yield from self._booster.staged_predict(pool)


CBoostClassifier = CTBoostClassifier
CBoostRegressor = CTBoostRegressor
