"""scikit-learn compatible estimator shells for CTBoost."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from . import _core
from ._serialization import load_estimator_document, save_estimator
from .core import Pool
from .feature_pipeline import FeaturePipeline
from .training import (
    Booster,
    _apply_sample_weight_to_pool,
    _normalize_eval_sets,
    _pool_from_data_and_label,
    train,
)

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
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    resolved = []
    for entry in normalized:
        if isinstance(entry, Pool):
            encoded_labels = _encode_class_labels(entry.label, label_to_index, "eval_set")
            resolved.append(_pool_from_data_and_label(entry, encoded_labels))
            continue
        if len(entry) == 2:
            X_eval, y_eval = entry
            resolved.append(
                _pool_from_data_and_label(
                    X_eval,
                    _encode_class_labels(y_eval, label_to_index, "eval_set"),
                )
            )
        else:
            X_eval, y_eval, group_id = entry
            resolved.append(
                _pool_from_data_and_label(
                    X_eval,
                    _encode_class_labels(y_eval, label_to_index, "eval_set"),
                    group_id=group_id,
                )
            )
    return resolved[0] if len(resolved) == 1 else resolved


def _encode_classifier_eval_set(eval_set: Any, label_to_index: Dict[Any, int]) -> Optional[Any]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    resolved = []
    for entry in normalized:
        if isinstance(entry, Pool):
            resolved.append(entry)
            continue
        if len(entry) == 2:
            X_eval, y_eval = entry
            resolved.append((X_eval, _encode_class_labels(y_eval, label_to_index, "eval_set")))
        else:
            X_eval, y_eval, group_id = entry
            resolved.append((X_eval, _encode_class_labels(y_eval, label_to_index, "eval_set"), group_id))
    return resolved[0] if len(resolved) == 1 else resolved


def _resolve_ranker_eval_pool(eval_set: Any) -> Optional[Pool]:
    normalized = _normalize_eval_sets(eval_set)
    if not normalized:
        return None
    resolved = []
    for entry in normalized:
        if isinstance(entry, Pool):
            resolved.append(entry)
            continue
        if len(entry) != 3:
            raise TypeError("ranking eval_set entries must be Pools or (X, y, group_id) tuples")
        X_eval, y_eval, group_id = entry
        resolved.append(_pool_from_data_and_label(X_eval, y_eval, group_id=group_id))
    return resolved[0] if len(resolved) == 1 else resolved


def _serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {
            "__ctboost_type__": "dict",
            "items": [[_serialize_value(key), _serialize_value(item)] for key, item in value.items()],
        }
    if isinstance(value, tuple):
        return {
            "__ctboost_type__": "tuple",
            "items": [_serialize_value(item) for item in value],
        }
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    marker = value.get("__ctboost_type__")
    if marker == "dict":
        return {
            _deserialize_value(key): _deserialize_value(item)
            for key, item in value["items"]
        }
    if marker == "tuple":
        return tuple(_deserialize_value(item) for item in value["items"])
    return {key: _deserialize_value(item) for key, item in value.items()}


class _BaseCTBoost(BaseEstimator):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        lambda_l2: float = 1.0,
        subsample: float = 1.0,
        bootstrap_type: str = "No",
        bagging_temperature: float = 0.0,
        boosting_type: str = "GradientBoosting",
        drop_rate: float = 0.1,
        skip_drop: float = 0.5,
        max_drop: int = 0,
        ordered_ctr: bool = False,
        one_hot_max_size: int = 0,
        max_cat_threshold: int = 0,
        cat_features: Optional[Any] = None,
        categorical_combinations: Optional[Any] = None,
        pairwise_categorical_combinations: bool = False,
        simple_ctr: Optional[Any] = None,
        combinations_ctr: Optional[Any] = None,
        per_feature_ctr: Optional[Any] = None,
        text_features: Optional[Any] = None,
        text_hash_dim: int = 64,
        embedding_features: Optional[Any] = None,
        embedding_stats: Any = ("mean", "std", "min", "max", "l2"),
        ctr_prior_strength: float = 1.0,
        monotone_constraints: Optional[Any] = None,
        interaction_constraints: Optional[Any] = None,
        colsample_bytree: float = 1.0,
        feature_weights: Optional[Any] = None,
        first_feature_use_penalties: Optional[Any] = None,
        random_strength: float = 0.0,
        grow_policy: str = "DepthWise",
        max_leaves: int = 0,
        min_samples_split: int = 2,
        min_data_in_leaf: int = 0,
        min_child_weight: float = 0.0,
        gamma: float = 0.0,
        max_leaf_weight: float = 0.0,
        max_bins: int = 256,
        max_bin_by_feature: Optional[Any] = None,
        border_selection_method: str = "Quantile",
        nan_mode_by_feature: Optional[Any] = None,
        feature_borders: Optional[Any] = None,
        random_seed: int = 0,
        loss_function: Optional[str] = None,
        eval_metric: Optional[str] = None,
        quantile_alpha: float = 0.5,
        huber_delta: float = 1.0,
        tweedie_variance_power: float = 1.5,
        nan_mode: str = "Min",
        warm_start: bool = False,
        task_type: str = "CPU",
        devices: str = "0",
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_root: str = "",
        distributed_run_id: str = "default",
        distributed_timeout: float = 600.0,
        verbose: bool = False,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.alpha = alpha
        self.lambda_l2 = lambda_l2
        self.subsample = subsample
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.boosting_type = boosting_type
        self.drop_rate = drop_rate
        self.skip_drop = skip_drop
        self.max_drop = max_drop
        self.ordered_ctr = ordered_ctr
        self.one_hot_max_size = one_hot_max_size
        self.max_cat_threshold = max_cat_threshold
        self.cat_features = cat_features
        self.categorical_combinations = categorical_combinations
        self.pairwise_categorical_combinations = pairwise_categorical_combinations
        self.simple_ctr = simple_ctr
        self.combinations_ctr = combinations_ctr
        self.per_feature_ctr = per_feature_ctr
        self.text_features = text_features
        self.text_hash_dim = text_hash_dim
        self.embedding_features = embedding_features
        self.embedding_stats = embedding_stats
        self.ctr_prior_strength = ctr_prior_strength
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.colsample_bytree = colsample_bytree
        self.feature_weights = feature_weights
        self.first_feature_use_penalties = first_feature_use_penalties
        self.random_strength = random_strength
        self.grow_policy = grow_policy
        self.max_leaves = max_leaves
        self.min_samples_split = min_samples_split
        self.min_data_in_leaf = min_data_in_leaf
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.max_leaf_weight = max_leaf_weight
        self.max_bins = max_bins
        self.max_bin_by_feature = max_bin_by_feature
        self.border_selection_method = border_selection_method
        self.nan_mode_by_feature = nan_mode_by_feature
        self.feature_borders = feature_borders
        self.random_seed = random_seed
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.quantile_alpha = quantile_alpha
        self.huber_delta = huber_delta
        self.tweedie_variance_power = tweedie_variance_power
        self.nan_mode = nan_mode
        self.warm_start = warm_start
        self.task_type = task_type
        self.devices = devices
        self.distributed_world_size = distributed_world_size
        self.distributed_rank = distributed_rank
        self.distributed_root = distributed_root
        self.distributed_run_id = distributed_run_id
        self.distributed_timeout = distributed_timeout
        self.verbose = verbose
        self._feature_pipeline: Optional[FeaturePipeline] = None

    def _uses_feature_pipeline(self) -> bool:
        return bool(
            self.ordered_ctr
            or self.cat_features
            or self.one_hot_max_size
            or self.max_cat_threshold
            or self.categorical_combinations
            or self.pairwise_categorical_combinations
            or self.simple_ctr
            or self.combinations_ctr
            or self.per_feature_ctr
            or self.text_features
            or self.embedding_features
        )

    def _build_feature_pipeline(self) -> FeaturePipeline:
        return FeaturePipeline(
            cat_features=self.cat_features,
            ordered_ctr=self.ordered_ctr,
            one_hot_max_size=self.one_hot_max_size,
            max_cat_threshold=self.max_cat_threshold,
            categorical_combinations=self.categorical_combinations,
            pairwise_categorical_combinations=self.pairwise_categorical_combinations,
            simple_ctr=self.simple_ctr,
            combinations_ctr=self.combinations_ctr,
            per_feature_ctr=self.per_feature_ctr,
            text_features=self.text_features,
            text_hash_dim=self.text_hash_dim,
            embedding_features=self.embedding_features,
            embedding_stats=self.embedding_stats,
            ctr_prior_strength=self.ctr_prior_strength,
            random_seed=self.random_seed,
        )

    def _fit_impl(
        self,
        X: Any,
        y: Any = None,
        *,
        group_id: Any = None,
        init_model: Any = None,
        sample_weight: Any = None,
        eval_set: Any = None,
        eval_names: Any = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        early_stopping_name: Optional[str] = None,
        callbacks: Optional[Iterable[Any]] = None,
        objective: str,
        num_classes: int = 1,
        extra_train_params: Optional[Dict[str, Any]] = None,
    ) -> "_BaseCTBoost":
        resolved_eval_sets = _normalize_eval_sets(eval_set)
        if self._uses_feature_pipeline():
            if isinstance(X, Pool):
                raise ValueError("feature pipeline parameters require raw array or DataFrame input")
            if y is None:
                raise ValueError("y must be provided when feature pipeline parameters are enabled")
            if any(isinstance(entry, Pool) for entry in resolved_eval_sets):
                raise ValueError("feature pipeline parameters require raw eval_set input, not a Pool")
            self._feature_pipeline = self._build_feature_pipeline()
            transformed_X, transformed_cat_features, transformed_feature_names = (
                self._feature_pipeline.fit_transform_array(X, y)
            )
            train_pool = Pool(
                data=transformed_X,
                label=y,
                cat_features=transformed_cat_features,
                group_id=group_id,
                feature_names=transformed_feature_names,
                _releasable_feature_storage=True,
            )
            eval_pools = []
            for resolved_eval_set in resolved_eval_sets:
                if len(resolved_eval_set) == 2:
                    X_eval, y_eval = resolved_eval_set
                    group_eval = None
                else:
                    X_eval, y_eval, group_eval = resolved_eval_set
                eval_pools.append(
                    self._feature_pipeline.transform_pool(
                        X_eval,
                        y_eval,
                        group_id=group_eval,
                    )
                )
        else:
            self._feature_pipeline = None
            train_pool = (
                X
                if isinstance(X, Pool) and y is None and group_id is None
                else _pool_from_data_and_label(X, y, group_id=group_id)
            )
            eval_pools = []
            for resolved_eval_set in resolved_eval_sets:
                if isinstance(resolved_eval_set, Pool):
                    eval_pools.append(resolved_eval_set)
                elif len(resolved_eval_set) == 2:
                    X_eval, y_eval = resolved_eval_set
                    eval_pools.append(_pool_from_data_and_label(X_eval, y_eval))
                else:
                    X_eval, y_eval, group_eval = resolved_eval_set
                    eval_pools.append(_pool_from_data_and_label(X_eval, y_eval, group_id=group_eval))

        train_pool = _apply_sample_weight_to_pool(train_pool, sample_weight)
        if not eval_pools:
            eval_arg = None
        elif len(eval_pools) == 1:
            eval_arg = eval_pools[0]
        else:
            eval_arg = eval_pools
        resolved_objective = objective if num_classes > 2 else (self.loss_function or objective)
        train_params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "alpha": self.alpha,
            "lambda_l2": self.lambda_l2,
            "subsample": self.subsample,
            "bootstrap_type": self.bootstrap_type,
            "bagging_temperature": self.bagging_temperature,
            "boosting_type": self.boosting_type,
            "drop_rate": self.drop_rate,
            "skip_drop": self.skip_drop,
            "max_drop": self.max_drop,
            "monotone_constraints": self.monotone_constraints,
            "interaction_constraints": self.interaction_constraints,
            "colsample_bytree": self.colsample_bytree,
            "feature_weights": self.feature_weights,
            "first_feature_use_penalties": self.first_feature_use_penalties,
            "random_strength": self.random_strength,
            "grow_policy": self.grow_policy,
            "max_leaves": self.max_leaves,
            "min_samples_split": self.min_samples_split,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "max_leaf_weight": self.max_leaf_weight,
            "max_bins": self.max_bins,
            "max_bin_by_feature": self.max_bin_by_feature,
            "border_selection_method": self.border_selection_method,
            "nan_mode_by_feature": self.nan_mode_by_feature,
            "feature_borders": self.feature_borders,
            "random_seed": self.random_seed,
            "objective": resolved_objective,
            "num_classes": num_classes,
            "quantile_alpha": self.quantile_alpha,
            "huber_delta": self.huber_delta,
            "tweedie_variance_power": self.tweedie_variance_power,
            "nan_mode": self.nan_mode,
            "task_type": self.task_type,
            "devices": self.devices,
            "distributed_world_size": self.distributed_world_size,
            "distributed_rank": self.distributed_rank,
            "distributed_root": self.distributed_root,
            "distributed_run_id": self.distributed_run_id,
            "distributed_timeout": self.distributed_timeout,
            "verbose": self.verbose,
        }
        if self.eval_metric is not None:
            train_params["eval_metric"] = self.eval_metric
        if extra_train_params:
            train_params.update(extra_train_params)
        resolved_init_model = init_model
        if resolved_init_model is None and self.warm_start and hasattr(self, "_booster"):
            resolved_init_model = self._booster
        self._booster = train(
            train_pool,
            train_params,
            num_boost_round=self.iterations,
            eval_set=eval_arg,
            eval_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callbacks,
            init_model=resolved_init_model,
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
        num_rows = int(X.shape[0]) if hasattr(X, "shape") else len(X)
        return Pool(data=X, label=np.zeros(num_rows, dtype=np.float32))

    def _transform_prediction_pool(self, X: Any) -> Pool:
        if isinstance(X, Pool):
            return X
        if self._feature_pipeline is None:
            return self._prediction_pool(X)
        num_rows = int(X.shape[0]) if hasattr(X, "shape") else len(X)
        transformed, cat_features, feature_names = self._feature_pipeline.transform_array(X)
        return Pool(
            data=transformed,
            label=np.zeros(num_rows, dtype=np.float32),
            cat_features=cat_features,
            feature_names=feature_names,
        )

    def _compute_best_score(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        evals_result = self._booster.evals_result_
        best_index = self.best_iteration_ if self.best_iteration_ >= 0 else len(self._booster.loss_history) - 1
        if best_index < 0:
            return result
        for dataset_name, metric_histories in evals_result.items():
            dataset_scores = {}
            for metric_name, history in metric_histories.items():
                if not history:
                    continue
                resolved_index = min(best_index, len(history) - 1)
                dataset_scores[metric_name] = float(history[resolved_index])
            if not dataset_scores:
                continue
            if dataset_name == "learn" and list(dataset_scores) == ["loss"]:
                result[dataset_name] = dataset_scores["loss"]
            elif len(dataset_scores) == 1 and list(dataset_scores) == [self._booster.eval_metric_name]:
                result[dataset_name] = dataset_scores[self._booster.eval_metric_name]
            else:
                result[dataset_name] = dataset_scores
        return result

    def _extra_serialized_state(self) -> Dict[str, Any]:
        return {}

    def _load_extra_serialized_state(self, fitted_state: Dict[str, Any]) -> None:
        return None

    def save_model(self, path: PathLike, *, model_format: Optional[str] = None) -> None:
        destination = Path(path)
        fitted_state = {
            "booster_state": dict(self._booster._handle.export_state()),
            "n_features_in_": int(self.n_features_in_),
            "best_iteration_": int(self.best_iteration_),
            "evals_result_": _serialize_value(self.evals_result_),
            "best_score_": _serialize_value(self.best_score_),
            "python_object": self,
        }
        if self._feature_pipeline is not None:
            fitted_state["feature_pipeline_state"] = self._feature_pipeline.to_state()
        fitted_state.update(self._extra_serialized_state())
        save_estimator(
            destination,
            estimator_class=type(self).__name__,
            init_params={key: _serialize_value(value) for key, value in self.get_params(deep=False).items()},
            fitted_state=fitted_state,
            model_format=model_format,
        )

    def export_model(
        self,
        path: PathLike,
        *,
        export_format: Optional[str] = None,
        prepared_features: bool = False,
    ) -> None:
        self._booster.export_model(
            path,
            export_format=export_format,
            prepared_features=prepared_features,
        )

    @classmethod
    def load_model(cls, path: PathLike) -> "_BaseCTBoost":
        destination = Path(path)
        document = load_estimator_document(destination)
        if document is None:
            import pickle

            with destination.open("rb") as stream:
                model = pickle.load(stream)
            if not isinstance(model, cls):
                raise TypeError(f"serialized model does not contain a {cls.__name__} instance")
            return model

        if document["estimator_class"] != cls.__name__:
            raise TypeError(f"serialized model does not contain a {cls.__name__} instance")

        model = cls(**{key: _deserialize_value(value) for key, value in document["init_params"].items()})
        fitted_state = document["fitted_state"]
        model._booster = Booster(_core.GradientBooster.from_state(fitted_state["booster_state"]))
        model.n_features_in_ = int(fitted_state["n_features_in_"])
        model.best_iteration_ = int(fitted_state["best_iteration_"])
        model.evals_result_ = _deserialize_value(fitted_state["evals_result_"])
        model.best_score_ = _deserialize_value(fitted_state["best_score_"])
        pipeline_state = fitted_state.get("feature_pipeline_state")
        model._feature_pipeline = None if pipeline_state is None else FeaturePipeline.from_state(pipeline_state)
        model._load_extra_serialized_state(fitted_state)
        return model

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.asarray(self._booster.feature_importances_, dtype=np.float32)

    def predict_leaf_index(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._transform_prediction_pool(X)
        return self._booster.predict_leaf_index(pool, num_iteration=num_iteration)

    def predict_contrib(self, X: Any, *, num_iteration: Optional[int] = None) -> np.ndarray:
        pool = self._transform_prediction_pool(X)
        return self._booster.predict_contrib(pool, num_iteration=num_iteration)


class CTBoostClassifier(_BaseCTBoost, ClassifierMixin):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        lambda_l2: float = 1.0,
        subsample: float = 1.0,
        bootstrap_type: str = "No",
        bagging_temperature: float = 0.0,
        boosting_type: str = "GradientBoosting",
        drop_rate: float = 0.1,
        skip_drop: float = 0.5,
        max_drop: int = 0,
        ordered_ctr: bool = False,
        one_hot_max_size: int = 0,
        max_cat_threshold: int = 0,
        cat_features: Optional[Any] = None,
        categorical_combinations: Optional[Any] = None,
        pairwise_categorical_combinations: bool = False,
        simple_ctr: Optional[Any] = None,
        combinations_ctr: Optional[Any] = None,
        per_feature_ctr: Optional[Any] = None,
        text_features: Optional[Any] = None,
        text_hash_dim: int = 64,
        embedding_features: Optional[Any] = None,
        embedding_stats: Any = ("mean", "std", "min", "max", "l2"),
        ctr_prior_strength: float = 1.0,
        monotone_constraints: Optional[Any] = None,
        interaction_constraints: Optional[Any] = None,
        colsample_bytree: float = 1.0,
        feature_weights: Optional[Any] = None,
        first_feature_use_penalties: Optional[Any] = None,
        random_strength: float = 0.0,
        grow_policy: str = "DepthWise",
        max_leaves: int = 0,
        min_samples_split: int = 2,
        min_data_in_leaf: int = 0,
        min_child_weight: float = 0.0,
        gamma: float = 0.0,
        max_leaf_weight: float = 0.0,
        max_bins: int = 256,
        max_bin_by_feature: Optional[Any] = None,
        border_selection_method: str = "Quantile",
        nan_mode_by_feature: Optional[Any] = None,
        feature_borders: Optional[Any] = None,
        random_seed: int = 0,
        loss_function: str = "Logloss",
        class_weight: Optional[Any] = None,
        scale_pos_weight: Optional[float] = None,
        eval_metric: Optional[str] = None,
        quantile_alpha: float = 0.5,
        huber_delta: float = 1.0,
        tweedie_variance_power: float = 1.5,
        nan_mode: str = "Min",
        warm_start: bool = False,
        task_type: str = "CPU",
        devices: str = "0",
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_root: str = "",
        distributed_run_id: str = "default",
        distributed_timeout: float = 600.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            lambda_l2=lambda_l2,
            subsample=subsample,
            bootstrap_type=bootstrap_type,
            bagging_temperature=bagging_temperature,
            boosting_type=boosting_type,
            drop_rate=drop_rate,
            skip_drop=skip_drop,
            max_drop=max_drop,
            ordered_ctr=ordered_ctr,
            one_hot_max_size=one_hot_max_size,
            max_cat_threshold=max_cat_threshold,
            cat_features=cat_features,
            categorical_combinations=categorical_combinations,
            pairwise_categorical_combinations=pairwise_categorical_combinations,
            simple_ctr=simple_ctr,
            combinations_ctr=combinations_ctr,
            per_feature_ctr=per_feature_ctr,
            text_features=text_features,
            text_hash_dim=text_hash_dim,
            embedding_features=embedding_features,
            embedding_stats=embedding_stats,
            ctr_prior_strength=ctr_prior_strength,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            colsample_bytree=colsample_bytree,
            feature_weights=feature_weights,
            first_feature_use_penalties=first_feature_use_penalties,
            random_strength=random_strength,
            grow_policy=grow_policy,
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            min_data_in_leaf=min_data_in_leaf,
            min_child_weight=min_child_weight,
            gamma=gamma,
            max_leaf_weight=max_leaf_weight,
            max_bins=max_bins,
            max_bin_by_feature=max_bin_by_feature,
            border_selection_method=border_selection_method,
            nan_mode_by_feature=nan_mode_by_feature,
            feature_borders=feature_borders,
            random_seed=random_seed,
            loss_function=loss_function,
            eval_metric=eval_metric,
            quantile_alpha=quantile_alpha,
            huber_delta=huber_delta,
            tweedie_variance_power=tweedie_variance_power,
            nan_mode=nan_mode,
            warm_start=warm_start,
            task_type=task_type,
            devices=devices,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_root=distributed_root,
            distributed_run_id=distributed_run_id,
            distributed_timeout=distributed_timeout,
            verbose=verbose,
        )
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        init_model: Any = None,
        sample_weight: Any = None,
        eval_set: Any = None,
        eval_names: Any = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        early_stopping_name: Optional[str] = None,
        callbacks: Optional[Iterable[Any]] = None,
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

        self.classes_ = classes
        self.n_classes_ = int(self.classes_.size)
        if self.n_classes_ < 2:
            raise ValueError("CTBoostClassifier requires at least two distinct classes.")

        label_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        eval_pool = (
            _encode_classifier_eval_set(eval_set, label_to_index)
            if self._uses_feature_pipeline()
            else _resolve_classifier_eval_pool(eval_set, label_to_index)
        )
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
        train_input = (
            X
            if self._uses_feature_pipeline() or isinstance(X, Pool)
            else _pool_from_data_and_label(X, encoded_labels)
        )
        fitted = self._fit_impl(
            train_input,
            None if isinstance(train_input, Pool) else encoded_labels,
            init_model=init_model,
            sample_weight=sample_weight,
            eval_set=eval_pool,
            eval_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callbacks,
            objective=objective,
            num_classes=self.n_classes_,
            extra_train_params=train_params,
        )
        return fitted

    def _extra_serialized_state(self) -> Dict[str, Any]:
        return {
            "classes_": _serialize_value(self.classes_),
            "n_classes_": int(self.n_classes_),
        }

    def _load_extra_serialized_state(self, fitted_state: Dict[str, Any]) -> None:
        self.classes_ = np.asarray(fitted_state["classes_"])
        self.n_classes_ = int(fitted_state["n_classes_"])

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._transform_prediction_pool(X)
        probabilities = self.predict_proba(pool, num_iteration=num_iteration)
        if self.n_classes_ > 2:
            indices = np.argmax(probabilities, axis=1)
        else:
            indices = (probabilities[:, 1] >= 0.5).astype(np.int64)
        return self.classes_[indices]

    def predict_proba(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._transform_prediction_pool(X)
        scores = np.asarray(self._booster.predict(pool, num_iteration=num_iteration), dtype=np.float32)
        if self.n_classes_ > 2:
            if scores.ndim == 1:
                scores = scores.reshape((pool.num_rows, self.n_classes_))
            return self._softmax(scores)
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probabilities, probabilities])

    def staged_predict_proba(self, X: Any) -> Iterable[Any]:
        pool = self._transform_prediction_pool(X)
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
        subsample: float = 1.0,
        bootstrap_type: str = "No",
        bagging_temperature: float = 0.0,
        boosting_type: str = "GradientBoosting",
        drop_rate: float = 0.1,
        skip_drop: float = 0.5,
        max_drop: int = 0,
        ordered_ctr: bool = False,
        one_hot_max_size: int = 0,
        max_cat_threshold: int = 0,
        cat_features: Optional[Any] = None,
        categorical_combinations: Optional[Any] = None,
        pairwise_categorical_combinations: bool = False,
        simple_ctr: Optional[Any] = None,
        combinations_ctr: Optional[Any] = None,
        per_feature_ctr: Optional[Any] = None,
        text_features: Optional[Any] = None,
        text_hash_dim: int = 64,
        embedding_features: Optional[Any] = None,
        embedding_stats: Any = ("mean", "std", "min", "max", "l2"),
        ctr_prior_strength: float = 1.0,
        monotone_constraints: Optional[Any] = None,
        interaction_constraints: Optional[Any] = None,
        colsample_bytree: float = 1.0,
        feature_weights: Optional[Any] = None,
        first_feature_use_penalties: Optional[Any] = None,
        random_strength: float = 0.0,
        grow_policy: str = "DepthWise",
        max_leaves: int = 0,
        min_samples_split: int = 2,
        min_data_in_leaf: int = 0,
        min_child_weight: float = 0.0,
        gamma: float = 0.0,
        max_leaf_weight: float = 0.0,
        max_bins: int = 256,
        max_bin_by_feature: Optional[Any] = None,
        border_selection_method: str = "Quantile",
        nan_mode_by_feature: Optional[Any] = None,
        feature_borders: Optional[Any] = None,
        random_seed: int = 0,
        loss_function: str = "RMSE",
        eval_metric: Optional[str] = None,
        quantile_alpha: float = 0.5,
        huber_delta: float = 1.0,
        tweedie_variance_power: float = 1.5,
        nan_mode: str = "Min",
        warm_start: bool = False,
        task_type: str = "CPU",
        devices: str = "0",
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_root: str = "",
        distributed_run_id: str = "default",
        distributed_timeout: float = 600.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            lambda_l2=lambda_l2,
            subsample=subsample,
            bootstrap_type=bootstrap_type,
            bagging_temperature=bagging_temperature,
            boosting_type=boosting_type,
            drop_rate=drop_rate,
            skip_drop=skip_drop,
            max_drop=max_drop,
            ordered_ctr=ordered_ctr,
            one_hot_max_size=one_hot_max_size,
            max_cat_threshold=max_cat_threshold,
            cat_features=cat_features,
            categorical_combinations=categorical_combinations,
            pairwise_categorical_combinations=pairwise_categorical_combinations,
            simple_ctr=simple_ctr,
            combinations_ctr=combinations_ctr,
            per_feature_ctr=per_feature_ctr,
            text_features=text_features,
            text_hash_dim=text_hash_dim,
            embedding_features=embedding_features,
            embedding_stats=embedding_stats,
            ctr_prior_strength=ctr_prior_strength,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            colsample_bytree=colsample_bytree,
            feature_weights=feature_weights,
            first_feature_use_penalties=first_feature_use_penalties,
            random_strength=random_strength,
            grow_policy=grow_policy,
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            min_data_in_leaf=min_data_in_leaf,
            min_child_weight=min_child_weight,
            gamma=gamma,
            max_leaf_weight=max_leaf_weight,
            max_bins=max_bins,
            max_bin_by_feature=max_bin_by_feature,
            border_selection_method=border_selection_method,
            nan_mode_by_feature=nan_mode_by_feature,
            feature_borders=feature_borders,
            random_seed=random_seed,
            loss_function=loss_function,
            eval_metric=eval_metric,
            quantile_alpha=quantile_alpha,
            huber_delta=huber_delta,
            tweedie_variance_power=tweedie_variance_power,
            nan_mode=nan_mode,
            warm_start=warm_start,
            task_type=task_type,
            devices=devices,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_root=distributed_root,
            distributed_run_id=distributed_run_id,
            distributed_timeout=distributed_timeout,
            verbose=verbose,
        )

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        init_model: Any = None,
        sample_weight: Any = None,
        eval_set: Any = None,
        eval_names: Any = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        early_stopping_name: Optional[str] = None,
        callbacks: Optional[Iterable[Any]] = None,
    ) -> "CTBoostRegressor":
        return self._fit_impl(
            X,
            y,
            init_model=init_model,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callbacks,
            objective="SquaredError",
        )

    def predict(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._transform_prediction_pool(X)
        return np.asarray(self._booster.predict(pool, num_iteration=num_iteration), dtype=np.float32)

    def staged_predict(self, X: Any) -> Iterable[Any]:
        pool = self._transform_prediction_pool(X)
        yield from self._booster.staged_predict(pool)


class CTBoostRanker(_BaseCTBoost):
    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        alpha: float = 0.05,
        lambda_l2: float = 1.0,
        subsample: float = 1.0,
        bootstrap_type: str = "No",
        bagging_temperature: float = 0.0,
        boosting_type: str = "GradientBoosting",
        drop_rate: float = 0.1,
        skip_drop: float = 0.5,
        max_drop: int = 0,
        ordered_ctr: bool = False,
        one_hot_max_size: int = 0,
        max_cat_threshold: int = 0,
        cat_features: Optional[Any] = None,
        categorical_combinations: Optional[Any] = None,
        pairwise_categorical_combinations: bool = False,
        simple_ctr: Optional[Any] = None,
        combinations_ctr: Optional[Any] = None,
        per_feature_ctr: Optional[Any] = None,
        text_features: Optional[Any] = None,
        text_hash_dim: int = 64,
        embedding_features: Optional[Any] = None,
        embedding_stats: Any = ("mean", "std", "min", "max", "l2"),
        ctr_prior_strength: float = 1.0,
        monotone_constraints: Optional[Any] = None,
        interaction_constraints: Optional[Any] = None,
        colsample_bytree: float = 1.0,
        feature_weights: Optional[Any] = None,
        first_feature_use_penalties: Optional[Any] = None,
        random_strength: float = 0.0,
        grow_policy: str = "DepthWise",
        max_leaves: int = 0,
        min_samples_split: int = 2,
        min_data_in_leaf: int = 0,
        min_child_weight: float = 0.0,
        gamma: float = 0.0,
        max_leaf_weight: float = 0.0,
        max_bins: int = 256,
        max_bin_by_feature: Optional[Any] = None,
        border_selection_method: str = "Quantile",
        nan_mode_by_feature: Optional[Any] = None,
        feature_borders: Optional[Any] = None,
        random_seed: int = 0,
        loss_function: str = "PairLogit",
        eval_metric: Optional[str] = "NDCG",
        quantile_alpha: float = 0.5,
        huber_delta: float = 1.0,
        tweedie_variance_power: float = 1.5,
        nan_mode: str = "Min",
        warm_start: bool = False,
        task_type: str = "CPU",
        devices: str = "0",
        distributed_world_size: int = 1,
        distributed_rank: int = 0,
        distributed_root: str = "",
        distributed_run_id: str = "default",
        distributed_timeout: float = 600.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            iterations=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            alpha=alpha,
            lambda_l2=lambda_l2,
            subsample=subsample,
            bootstrap_type=bootstrap_type,
            bagging_temperature=bagging_temperature,
            boosting_type=boosting_type,
            drop_rate=drop_rate,
            skip_drop=skip_drop,
            max_drop=max_drop,
            ordered_ctr=ordered_ctr,
            one_hot_max_size=one_hot_max_size,
            max_cat_threshold=max_cat_threshold,
            cat_features=cat_features,
            categorical_combinations=categorical_combinations,
            pairwise_categorical_combinations=pairwise_categorical_combinations,
            simple_ctr=simple_ctr,
            combinations_ctr=combinations_ctr,
            per_feature_ctr=per_feature_ctr,
            text_features=text_features,
            text_hash_dim=text_hash_dim,
            embedding_features=embedding_features,
            embedding_stats=embedding_stats,
            ctr_prior_strength=ctr_prior_strength,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            colsample_bytree=colsample_bytree,
            feature_weights=feature_weights,
            first_feature_use_penalties=first_feature_use_penalties,
            random_strength=random_strength,
            grow_policy=grow_policy,
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            min_data_in_leaf=min_data_in_leaf,
            min_child_weight=min_child_weight,
            gamma=gamma,
            max_leaf_weight=max_leaf_weight,
            max_bins=max_bins,
            max_bin_by_feature=max_bin_by_feature,
            border_selection_method=border_selection_method,
            nan_mode_by_feature=nan_mode_by_feature,
            feature_borders=feature_borders,
            random_seed=random_seed,
            loss_function=loss_function,
            eval_metric=eval_metric,
            quantile_alpha=quantile_alpha,
            huber_delta=huber_delta,
            tweedie_variance_power=tweedie_variance_power,
            nan_mode=nan_mode,
            warm_start=warm_start,
            task_type=task_type,
            devices=devices,
            distributed_world_size=distributed_world_size,
            distributed_rank=distributed_rank,
            distributed_root=distributed_root,
            distributed_run_id=distributed_run_id,
            distributed_timeout=distributed_timeout,
            verbose=verbose,
        )

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        group_id: Any = None,
        init_model: Any = None,
        sample_weight: Any = None,
        eval_set: Any = None,
        eval_names: Any = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[str] = None,
        early_stopping_name: Optional[str] = None,
        callbacks: Optional[Iterable[Any]] = None,
    ) -> "CTBoostRanker":
        if isinstance(X, Pool):
            if group_id is not None:
                raise ValueError("group_id should be provided through the Pool when X is already a Pool")
            train_pool = X if y is None else _pool_from_data_and_label(X, y)
            resolved_group_id = train_pool.group_id
        else:
            if y is None:
                raise ValueError("y must be provided when X is not a Pool")
            resolved_group_id = group_id
            train_pool = _pool_from_data_and_label(X, y, group_id=group_id)

        if resolved_group_id is None:
            raise ValueError("CTBoostRanker requires group_id")

        eval_pool = eval_set if self._uses_feature_pipeline() else _resolve_ranker_eval_pool(eval_set)
        fitted = self._fit_impl(
            train_pool,
            group_id=resolved_group_id,
            init_model=init_model,
            sample_weight=sample_weight,
            eval_set=eval_pool,
            eval_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callbacks,
            objective="PairLogit",
            num_classes=1,
        )
        return fitted

    def predict(self, X: Any, *, num_iteration: Optional[int] = None) -> Any:
        pool = self._transform_prediction_pool(X)
        return np.asarray(self._booster.predict(pool, num_iteration=num_iteration), dtype=np.float32)

    def staged_predict(self, X: Any) -> Iterable[Any]:
        pool = self._transform_prediction_pool(X)
        yield from self._booster.staged_predict(pool)


CBoostClassifier = CTBoostClassifier
CBoostRegressor = CTBoostRegressor
CBoostRanker = CTBoostRanker
