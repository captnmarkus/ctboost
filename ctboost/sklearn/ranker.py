"""Ranker estimator for ctboost.sklearn."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from ..core import Pool
from ..training import _pool_from_data_and_label
from .base import _BaseCTBoost
from .labels import _resolve_ranker_eval_pool

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
        eval_metric: Optional[Any] = "NDCG",
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
        group_weight: Any = None,
        subgroup_id: Any = None,
        baseline: Any = None,
        pairs: Any = None,
        pairs_weight: Any = None,
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
    ) -> "CTBoostRanker":
        if isinstance(X, Pool):
            if (
                group_id is not None
                or group_weight is not None
                or subgroup_id is not None
                or baseline is not None
                or pairs is not None
                or pairs_weight is not None
            ):
                raise ValueError(
                    "group_id, group_weight, subgroup_id, baseline, pairs, and pairs_weight "
                    "should be provided through the Pool when X is already a Pool"
                )
            train_pool = X if y is None else _pool_from_data_and_label(X, y)
            resolved_group_id = train_pool.group_id
        else:
            if y is None:
                raise ValueError("y must be provided when X is not a Pool")
            resolved_group_id = group_id
            train_pool = _pool_from_data_and_label(
                X,
                y,
                group_id=group_id,
                group_weight=group_weight,
                subgroup_id=subgroup_id,
                baseline=baseline,
                pairs=pairs,
                pairs_weight=pairs_weight,
            )

        if resolved_group_id is None:
            raise ValueError("CTBoostRanker requires group_id")

        eval_pool = eval_set if self._uses_feature_pipeline() else _resolve_ranker_eval_pool(eval_set)
        fitted = self._fit_impl(
            train_pool,
            group_id=resolved_group_id,
            group_weight=group_weight,
            subgroup_id=subgroup_id,
            baseline=baseline,
            pairs=pairs,
            pairs_weight=pairs_weight,
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
