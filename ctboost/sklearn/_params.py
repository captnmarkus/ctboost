"""Parameter and pipeline mixin for ctboost.sklearn."""

from __future__ import annotations

from typing import Any, Optional

from ..feature_pipeline import FeaturePipeline


class _BaseInitMixin:
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
