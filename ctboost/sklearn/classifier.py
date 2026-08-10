"""Classifier estimator for ctboost.sklearn."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.base import ClassifierMixin

from ..core import Pool
from ..core.columnar import _columnar_vector_to_numpy
from ..training import _pool_from_data_and_label
from .base import _BaseCTBoost
from .labels import _encode_classifier_eval_set, _resolve_classifier_eval_pool
from .serialization import PathLike, _serialize_value

class CTBoostClassifier(ClassifierMixin, _BaseCTBoost):
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
        text_tokenizer: str = "word",
        text_ngram_range: Any = (1, 1),
        text_lowercase: bool = True,
        text_min_token_count: int = 1,
        text_max_dictionary_size: int = 0,
        text_feature_calcer: str = "count",
        embedding_features: Optional[Any] = None,
        embedding_stats: Any = ("mean", "std", "min", "max", "l2"),
        embedding_target_features: bool = False,
        embedding_target_regularization: float = 1.0,
        embedding_target_mode: str = "auto",
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
        leaf_estimation_iterations: int = 1,
        max_bins: int = 256,
        max_bin_by_feature: Optional[Any] = None,
        border_selection_method: str = "Quantile",
        nan_mode_by_feature: Optional[Any] = None,
        feature_borders: Optional[Any] = None,
        boost_from_average: bool = True,
        base_score: Optional[Any] = None,
        random_seed: int = 0,
        loss_function: str = "Logloss",
        class_weight: Optional[Any] = None,
        scale_pos_weight: Optional[float] = None,
        eval_metric: Optional[Any] = None,
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
        n_estimators: Optional[int] = None,
        depth: Optional[int] = None,
        reg_lambda: Optional[float] = None,
        l2_leaf_reg: Optional[float] = None,
        random_state: Optional[int] = None,
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
            text_tokenizer=text_tokenizer,
            text_ngram_range=text_ngram_range,
            text_lowercase=text_lowercase,
            text_min_token_count=text_min_token_count,
            text_max_dictionary_size=text_max_dictionary_size,
            text_feature_calcer=text_feature_calcer,
            embedding_features=embedding_features,
            embedding_stats=embedding_stats,
            embedding_target_features=embedding_target_features,
            embedding_target_regularization=embedding_target_regularization,
            embedding_target_mode=embedding_target_mode,
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
            leaf_estimation_iterations=leaf_estimation_iterations,
            max_bins=max_bins,
            max_bin_by_feature=max_bin_by_feature,
            border_selection_method=border_selection_method,
            nan_mode_by_feature=nan_mode_by_feature,
            feature_borders=feature_borders,
            boost_from_average=boost_from_average,
            base_score=base_score,
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
            n_estimators=n_estimators,
            depth=depth,
            reg_lambda=reg_lambda,
            l2_leaf_reg=l2_leaf_reg,
            random_state=random_state,
        )
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight

    def fit(
        self,
        X: Any,
        y: Any = None,
        *,
        baseline: Any = None,
        init_model: Any = None,
        sample_weight: Any = None,
        eval_set: Any = None,
        eval_names: Any = None,
        early_stopping_rounds: Optional[int] = None,
        early_stopping_metric: Optional[Any] = None,
        early_stopping_name: Optional[str] = None,
        callbacks: Optional[Iterable[Any]] = None,
        learning_rate_schedule: Any = None,
        snapshot_path: Optional[PathLike] = None,
        snapshot_interval: int = 1,
        snapshot_save_best_only: bool = False,
        snapshot_model_format: Optional[str] = None,
        resume_from_snapshot: Any = None,
    ) -> "CTBoostClassifier":
        if isinstance(X, Pool):
            raw_labels = np.asarray(
                X.label if y is None else _columnar_vector_to_numpy(y)
            )
        else:
            if y is None:
                raise ValueError("y must be provided when X is not a Pool")
            raw_labels = np.asarray(_columnar_vector_to_numpy(y))
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
        if isinstance(X, Pool):
            train_input = _pool_from_data_and_label(X, encoded_labels, baseline=baseline)
            train_labels = None
            fit_baseline = None
        elif self._uses_feature_pipeline():
            train_input = X
            train_labels = encoded_labels
            fit_baseline = baseline
        else:
            train_input = _pool_from_data_and_label(X, encoded_labels, baseline=baseline)
            train_labels = None
            fit_baseline = None
        fitted = self._fit_impl(
            train_input,
            train_labels,
            baseline=fit_baseline,
            init_model=init_model,
            sample_weight=sample_weight,
            eval_set=eval_pool,
            eval_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_metric=early_stopping_metric,
            early_stopping_name=early_stopping_name,
            callbacks=callbacks,
            learning_rate_schedule=learning_rate_schedule,
            snapshot_path=snapshot_path,
            snapshot_interval=snapshot_interval,
            snapshot_save_best_only=snapshot_save_best_only,
            snapshot_model_format=snapshot_model_format,
            resume_from_snapshot=resume_from_snapshot,
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

    @staticmethod
    def _sigmoid(scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float32)
        probabilities = np.empty_like(values, dtype=np.float32)
        nonnegative = values >= 0.0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
        exp_values = np.exp(values[~nonnegative])
        probabilities[~nonnegative] = exp_values / (1.0 + exp_values)
        return probabilities

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
        probabilities = self._sigmoid(scores)
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
                probabilities = self._sigmoid(scores)
                yield np.column_stack([1.0 - probabilities, probabilities])

    def staged_predict(self, X: Any) -> Iterable[Any]:
        for probabilities in self.staged_predict_proba(X):
            if self.n_classes_ > 2:
                indices = np.argmax(probabilities, axis=1)
            else:
                indices = (probabilities[:, 1] >= 0.5).astype(np.int64)
            yield self.classes_[indices]
