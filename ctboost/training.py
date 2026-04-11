"""Training entry points for the Python API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from . import _core
from .core import Pool

PathLike = Union[str, Path]


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


def _prediction_pool(data: Any) -> Pool:
    if isinstance(data, Pool):
        return data
    return Pool(data=data, label=np.zeros(len(data), dtype=np.float32))


def _resolve_num_iteration(num_iteration: Optional[int]) -> int:
    if num_iteration is None:
        return -1
    if num_iteration <= 0:
        raise ValueError("num_iteration must be a positive integer")
    return int(num_iteration)


def _objective_name(params: Mapping[str, Any]) -> str:
    return str(params.get("objective", params.get("loss_function", "RMSE")))


def _is_binary_classification_objective(objective_name: str) -> bool:
    return objective_name.lower() in {"logloss", "binary_logloss", "binary:logistic"}


def _is_classification_objective(objective_name: str) -> bool:
    return objective_name.lower() in {
        "logloss",
        "binary_logloss",
        "binary:logistic",
        "multiclass",
        "softmax",
        "softmaxloss",
    }


def _combine_weight_arrays(*arrays: Optional[np.ndarray]) -> Optional[np.ndarray]:
    combined: Optional[np.ndarray] = None
    for array in arrays:
        if array is None:
            continue
        resolved = np.asarray(array, dtype=np.float32)
        combined = resolved.copy() if combined is None else combined * resolved
    return combined


def _apply_sample_weight_to_pool(pool: Pool, sample_weight: Optional[Any]) -> Pool:
    if sample_weight is None:
        return pool

    resolved_weight = np.asarray(sample_weight, dtype=np.float32)
    if resolved_weight.ndim != 1:
        raise ValueError("sample_weight must be a 1D array")
    if resolved_weight.shape[0] != len(pool):
        raise ValueError("sample_weight size must match the number of rows")

    combined_weight = _combine_weight_arrays(pool.weight, resolved_weight)
    return Pool(
        data=pool.data,
        label=pool.label,
        cat_features=pool.cat_features,
        weight=combined_weight,
        feature_names=pool.feature_names,
    )


def _balanced_class_weight(labels: np.ndarray) -> Dict[Any, float]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_count = unique_labels.size
    sample_count = labels.shape[0]
    return {
        label.item() if hasattr(label, "item") else label: float(sample_count / (class_count * count))
        for label, count in zip(unique_labels, counts)
    }


def _resolve_class_weight_map(
    labels: np.ndarray,
    class_weights: Optional[Any],
    auto_class_weights: Optional[str],
) -> Optional[Dict[Any, float]]:
    if class_weights is not None and auto_class_weights is not None:
        raise ValueError("class_weights and auto_class_weights cannot be used together")

    if auto_class_weights is not None:
        normalized = str(auto_class_weights).lower()
        if normalized != "balanced":
            raise ValueError("auto_class_weights must be 'balanced'")
        return _balanced_class_weight(labels)

    if class_weights is None:
        return None

    if isinstance(class_weights, Mapping):
        return {
            key.item() if hasattr(key, "item") else key: float(value)
            for key, value in class_weights.items()
        }

    weight_values = np.asarray(class_weights, dtype=np.float32).reshape(-1)
    unique_labels = np.unique(labels)
    if weight_values.shape[0] != unique_labels.shape[0]:
        raise ValueError("class_weights must match the number of unique labels")
    return {
        label.item() if hasattr(label, "item") else label: float(weight)
        for label, weight in zip(unique_labels.tolist(), weight_values.tolist())
    }


def _classification_weight_multiplier(
    labels: np.ndarray,
    objective_name: str,
    *,
    class_weights: Optional[Any] = None,
    auto_class_weights: Optional[str] = None,
    scale_pos_weight: Optional[float] = None,
) -> Optional[np.ndarray]:
    if (
        class_weights is None
        and auto_class_weights is None
        and scale_pos_weight is None
    ):
        return None

    if not _is_classification_objective(objective_name):
        raise ValueError("class weighting parameters require a classification objective")

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("classification labels must be a 1D array")

    multiplier = np.ones(labels.shape[0], dtype=np.float32)
    class_weight_map = _resolve_class_weight_map(labels, class_weights, auto_class_weights)
    if class_weight_map is not None:
        unique_labels = {
            label.item() if hasattr(label, "item") else label for label in np.unique(labels).tolist()
        }
        if set(class_weight_map) != unique_labels:
            raise ValueError("class_weights must provide a weight for every class in the data")
        for label_value, weight in class_weight_map.items():
            if weight <= 0.0:
                raise ValueError("class weights must be positive")
            multiplier[labels == label_value] *= float(weight)

    if scale_pos_weight is not None:
        if not _is_binary_classification_objective(objective_name):
            raise ValueError("scale_pos_weight is only supported for binary classification")
        scale_value = float(scale_pos_weight)
        if scale_value <= 0.0:
            raise ValueError("scale_pos_weight must be positive")
        positive_mask = labels == 1
        multiplier[positive_mask] *= scale_value

    return multiplier


def _apply_objective_weights(pool: Pool, params: Mapping[str, Any], objective_name: str) -> Pool:
    objective_weight = _classification_weight_multiplier(
        np.asarray(pool.label),
        objective_name,
        class_weights=params.get("class_weights"),
        auto_class_weights=params.get("auto_class_weights"),
        scale_pos_weight=params.get("scale_pos_weight"),
    )
    if objective_weight is None:
        return pool

    combined_weight = _combine_weight_arrays(pool.weight, objective_weight)
    return Pool(
        data=pool.data,
        label=pool.label,
        cat_features=pool.cat_features,
        weight=combined_weight,
        feature_names=pool.feature_names,
    )


def _slice_pool(pool: Pool, indices: Iterable[int]) -> Pool:
    index_array = np.asarray(list(indices), dtype=np.int64)
    weight = None if pool.weight is None else pool.weight[index_array]
    return Pool(
        data=pool.data[index_array],
        label=pool.label[index_array],
        cat_features=pool.cat_features,
        weight=weight,
        feature_names=pool.feature_names,
    )


def _stack_histories(histories: List[np.ndarray]) -> np.ndarray:
    if not histories:
        return np.empty((0, 0), dtype=np.float32)
    max_length = max(history.shape[0] for history in histories)
    stacked = np.full((max_length, len(histories)), np.nan, dtype=np.float32)
    for column_index, history in enumerate(histories):
        stacked[: history.shape[0], column_index] = history
    return stacked


class Booster:
    """Small Python wrapper around the native gradient booster."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def _predict_raw(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = _prediction_pool(data)
        raw = np.asarray(
            self._handle.predict(pool._handle, _resolve_num_iteration(num_iteration)),
            dtype=np.float32,
        )
        prediction_dimension = int(self._handle.prediction_dimension())
        if prediction_dimension > 1:
            return raw.reshape((pool.num_rows, prediction_dimension))
        return raw

    def predict(self, data: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        return self._predict_raw(data, num_iteration=num_iteration)

    def staged_predict(self, data: Any) -> Iterable[np.ndarray]:
        pool = _prediction_pool(data)
        for iteration in range(1, self.num_iterations_trained + 1):
            yield self._predict_raw(pool, num_iteration=iteration)

    def save_model(self, path: PathLike) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            pickle.dump(self._handle, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model(cls, path: PathLike) -> "Booster":
        with Path(path).open("rb") as stream:
            handle = pickle.load(stream)
        if not isinstance(handle, _core.GradientBooster):
            raise TypeError("serialized model does not contain a CTBoost booster")
        return cls(handle)

    @property
    def loss_history(self) -> List[float]:
        return list(self._handle.loss_history())

    @property
    def eval_loss_history(self) -> List[float]:
        return list(self._handle.eval_loss_history())

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._handle.feature_importances(), dtype=np.float32)

    @property
    def evals_result_(self) -> Dict[str, Dict[str, List[float]]]:
        result = {"learn": {"loss": self.loss_history}}
        if self.eval_loss_history:
            metric_name = "loss" if self.eval_metric_name.lower() == self.objective_name.lower() else self.eval_metric_name
            result["validation"] = {metric_name: self.eval_loss_history}
        return result

    @property
    def num_classes(self) -> int:
        return int(self._handle.num_classes())

    @property
    def prediction_dimension(self) -> int:
        return int(self._handle.prediction_dimension())

    @property
    def num_iterations_trained(self) -> int:
        return int(self._handle.num_iterations_trained())

    @property
    def best_iteration(self) -> int:
        return int(self._handle.best_iteration())

    @property
    def objective_name(self) -> str:
        return str(self._handle.objective_name())

    @property
    def eval_metric_name(self) -> str:
        return str(self._handle.eval_metric_name())


def load_model(path: PathLike) -> Booster:
    return Booster.load_model(path)


def cv(
    pool: Pool,
    params: Mapping[str, Any],
    num_boost_round: Optional[int] = None,
    *,
    nfold: int = 3,
    shuffle: bool = True,
    random_state: Optional[int] = 0,
    stratified: Optional[bool] = None,
    early_stopping_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(pool, Pool):
        raise TypeError("pool must be an instance of ctboost.Pool")
    if nfold < 2:
        raise ValueError("nfold must be at least 2")

    objective = _objective_name(params)
    labels = np.asarray(pool.label)
    use_stratified = _is_classification_objective(objective) if stratified is None else bool(stratified)

    if use_stratified:
        splitter = StratifiedKFold(
            n_splits=nfold,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32), labels)
    else:
        splitter = KFold(
            n_splits=nfold,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        split_iterator = splitter.split(np.zeros(labels.shape[0], dtype=np.float32))

    train_histories: List[np.ndarray] = []
    valid_histories: List[np.ndarray] = []
    best_iterations: List[int] = []

    for train_index, valid_index in split_iterator:
        train_pool = _slice_pool(pool, train_index)
        valid_pool = _slice_pool(pool, valid_index)
        booster = train(
            train_pool,
            params,
            num_boost_round=num_boost_round,
            eval_set=valid_pool,
            early_stopping_rounds=early_stopping_rounds,
        )
        train_histories.append(np.asarray(booster.loss_history, dtype=np.float32))
        valid_histories.append(np.asarray(booster.eval_loss_history, dtype=np.float32))
        best_iterations.append(booster.best_iteration)

    train_matrix = _stack_histories(train_histories)
    valid_matrix = _stack_histories(valid_histories)
    iteration_count = train_matrix.shape[0]

    return {
        "iterations": np.arange(1, iteration_count + 1, dtype=np.int32),
        "train_loss_mean": np.nanmean(train_matrix, axis=1),
        "train_loss_std": np.nanstd(train_matrix, axis=1),
        "valid_loss_mean": np.nanmean(valid_matrix, axis=1),
        "valid_loss_std": np.nanstd(valid_matrix, axis=1),
        "best_iteration_mean": float(np.mean(best_iterations)) if best_iterations else float("nan"),
        "best_iteration_std": float(np.std(best_iterations)) if best_iterations else float("nan"),
    }


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
    nan_mode = str(config.get("nan_mode", "Min"))
    eval_metric = str(config.get("eval_metric", ""))
    quantile_alpha = float(config.get("quantile_alpha", 0.5))
    huber_delta = float(config.get("huber_delta", 1.0))
    task_type = str(config.get("task_type", "CPU"))
    devices = str(config.get("devices", "0"))
    weighted_pool = _apply_objective_weights(pool, config, objective)
    weighted_eval_pool = (
        None if eval_pool is None else _apply_objective_weights(eval_pool, config, objective)
    )

    booster = _core.GradientBooster(
        objective=objective,
        iterations=iterations,
        learning_rate=learning_rate,
        max_depth=max_depth,
        alpha=alpha,
        lambda_l2=lambda_l2,
        num_classes=num_classes,
        max_bins=max_bins,
        nan_mode=nan_mode,
        eval_metric=eval_metric,
        quantile_alpha=quantile_alpha,
        huber_delta=huber_delta,
        task_type=task_type,
        devices=devices,
    )
    booster.fit(
        weighted_pool._handle,
        None if weighted_eval_pool is None else weighted_eval_pool._handle,
        early_stopping,
    )
    return Booster(booster)
