"""Fit-time mixin for ctboost.sklearn."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..core import Pool
from ..feature_pipeline import FeaturePipeline
from ..training import _apply_sample_weight_to_pool, _normalize_eval_sets, _pool_from_data_and_label, train


class _BaseFitMixin:
        def _fit_impl(
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
                    group_weight=group_weight,
                    subgroup_id=subgroup_id,
                    baseline=baseline,
                    pairs=pairs,
                    pairs_weight=pairs_weight,
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
                    if (
                        isinstance(X, Pool)
                        and y is None
                        and group_id is None
                        and group_weight is None
                        and subgroup_id is None
                        and baseline is None
                        and pairs is None
                        and pairs_weight is None
                    )
                    else _pool_from_data_and_label(
                        X,
                        y,
                        group_id=group_id,
                        group_weight=group_weight,
                        subgroup_id=subgroup_id,
                        baseline=baseline,
                        pairs=pairs,
                        pairs_weight=pairs_weight,
                    )
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
                learning_rate_schedule=learning_rate_schedule,
                snapshot_path=snapshot_path,
                snapshot_interval=snapshot_interval,
                snapshot_save_best_only=snapshot_save_best_only,
                snapshot_model_format=snapshot_model_format,
                resume_from_snapshot=resume_from_snapshot,
                init_model=resolved_init_model,
            )
            self.n_features_in_ = train_pool.num_cols
            self.best_iteration_ = self._booster.best_iteration
            self.evals_result_ = self._booster.evals_result_
            self.best_score_ = self._compute_best_score()
            return self
        @property
        def data_schema_(self) -> Dict[str, Any]:
            return self._booster.data_schema
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
