"""Parameter and pipeline mixin for ctboost.sklearn."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..feature_pipeline import FeaturePipeline


_COMPATIBILITY_ALIASES = {
    "iterations": ("n_estimators",),
    "max_depth": ("depth",),
    "lambda_l2": ("reg_lambda", "l2_leaf_reg"),
    "random_seed": ("random_state",),
}


def _resolve_constructor_alias(
    canonical_name: str,
    canonical_value: Any,
    canonical_default: Any,
    aliases: Dict[str, Any],
) -> Any:
    provided = {name: value for name, value in aliases.items() if value is not None}
    if not provided:
        return canonical_value
    values = list(provided.values())
    if any(value != values[0] for value in values[1:]):
        raise ValueError(
            f"conflicting aliases for {canonical_name}: {', '.join(provided)}"
        )
    alias_value = values[0]
    if canonical_value != canonical_default and canonical_value != alias_value:
        raise ValueError(
            f"{canonical_name} conflicts with its compatibility alias"
        )
    return alias_value


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
            multi_strategy: str = "one_output_per_tree",
            feature_test: str = "quadratic",
            feature_test_bins: int = 8,
            feature_test_adjustment: str = "none",
            max_bins: int = 256,
            max_bin_by_feature: Optional[Any] = None,
            border_selection_method: str = "Quantile",
            nan_mode_by_feature: Optional[Any] = None,
            feature_borders: Optional[Any] = None,
            boost_from_average: bool = True,
            base_score: Optional[Any] = None,
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
            n_estimators: Optional[int] = None,
            depth: Optional[int] = None,
            reg_lambda: Optional[float] = None,
            l2_leaf_reg: Optional[float] = None,
            random_state: Optional[int] = None,
        ) -> None:
            iterations = _resolve_constructor_alias(
                "iterations", iterations, 100, {"n_estimators": n_estimators}
            )
            max_depth = _resolve_constructor_alias(
                "max_depth", max_depth, 6, {"depth": depth}
            )
            lambda_l2 = _resolve_constructor_alias(
                "lambda_l2",
                lambda_l2,
                1.0,
                {"reg_lambda": reg_lambda, "l2_leaf_reg": l2_leaf_reg},
            )
            random_seed = _resolve_constructor_alias(
                "random_seed", random_seed, 0, {"random_state": random_state}
            )
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
            self.text_tokenizer = text_tokenizer
            self.text_ngram_range = text_ngram_range
            self.text_lowercase = text_lowercase
            self.text_min_token_count = text_min_token_count
            self.text_max_dictionary_size = text_max_dictionary_size
            self.text_feature_calcer = text_feature_calcer
            self.embedding_features = embedding_features
            self.embedding_stats = embedding_stats
            self.embedding_target_features = embedding_target_features
            self.embedding_target_regularization = embedding_target_regularization
            self.embedding_target_mode = embedding_target_mode
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
            self.leaf_estimation_iterations = leaf_estimation_iterations
            self.multi_strategy = multi_strategy
            self.feature_test = feature_test
            self.feature_test_bins = feature_test_bins
            self.feature_test_adjustment = feature_test_adjustment
            self.max_bins = max_bins
            self.max_bin_by_feature = max_bin_by_feature
            self.border_selection_method = border_selection_method
            self.nan_mode_by_feature = nan_mode_by_feature
            self.feature_borders = feature_borders
            self.boost_from_average = boost_from_average
            self.base_score = base_score
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
            self.n_estimators = n_estimators
            self.depth = depth
            self.reg_lambda = reg_lambda
            self.l2_leaf_reg = l2_leaf_reg
            self.random_state = random_state
            self._feature_pipeline: Optional[FeaturePipeline] = None
        def set_params(self, **params: Any) -> "_BaseInitMixin":
            for canonical_name, alias_names in _COMPATIBILITY_ALIASES.items():
                provided_aliases = {
                    name: params[name]
                    for name in alias_names
                    if name in params and params[name] is not None
                }
                if not provided_aliases:
                    if canonical_name in params:
                        for alias_name in alias_names:
                            params.setdefault(alias_name, None)
                    continue
                alias_values = list(provided_aliases.values())
                if any(value != alias_values[0] for value in alias_values[1:]):
                    raise ValueError(
                        f"conflicting aliases for {canonical_name}: "
                        f"{', '.join(provided_aliases)}"
                    )
                alias_value = alias_values[0]
                if canonical_name in params and params[canonical_name] != alias_value:
                    raise ValueError(
                        f"{canonical_name} conflicts with its compatibility alias"
                    )
                params[canonical_name] = alias_value
                for alias_name in alias_names:
                    if alias_name not in provided_aliases:
                        params[alias_name] = None
            return super().set_params(**params)
        def _synchronize_compatibility_aliases(self) -> None:
            """Restore the parameter identity invariant required by sklearn.clone."""
            for canonical_name, alias_names in _COMPATIBILITY_ALIASES.items():
                for alias_name in alias_names:
                    if not hasattr(self, alias_name):
                        setattr(self, alias_name, None)
                    alias_value = getattr(self, alias_name, None)
                    if alias_value is not None:
                        setattr(self, canonical_name, alias_value)
                        break
        def __setstate__(self, state: Dict[str, Any]) -> None:
            super().__setstate__(state)
            preprocessing_defaults = {
                "text_tokenizer": "word",
                "text_ngram_range": (1, 1),
                "text_lowercase": True,
                "text_min_token_count": 1,
                "text_max_dictionary_size": 0,
                "text_feature_calcer": "count",
                "embedding_target_features": False,
                "embedding_target_regularization": 1.0,
                "embedding_target_mode": "auto",
                "boost_from_average": True,
                "base_score": None,
                "leaf_estimation_iterations": 1,
                "multi_strategy": "one_output_per_tree",
                "feature_test": "quadratic",
                "feature_test_bins": 8,
                "feature_test_adjustment": "none",
            }
            for name, default in preprocessing_defaults.items():
                if not hasattr(self, name):
                    setattr(self, name, default)
            self._synchronize_compatibility_aliases()
            if hasattr(self, "_booster"):
                self._booster._feature_pipeline = getattr(self, "_feature_pipeline", None)
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
            embedding_target_mode = self.embedding_target_mode
            if embedding_target_mode == "auto" and self.embedding_target_features:
                embedding_target_mode = (
                    "classification"
                    if getattr(self, "_estimator_type", None) == "classifier"
                    else "regression"
                )
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
                text_tokenizer=self.text_tokenizer,
                text_ngram_range=self.text_ngram_range,
                text_lowercase=self.text_lowercase,
                text_min_token_count=self.text_min_token_count,
                text_max_dictionary_size=self.text_max_dictionary_size,
                text_feature_calcer=self.text_feature_calcer,
                embedding_features=self.embedding_features,
                embedding_stats=self.embedding_stats,
                embedding_target_features=self.embedding_target_features,
                embedding_target_regularization=self.embedding_target_regularization,
                embedding_target_mode=embedding_target_mode,
                ctr_prior_strength=self.ctr_prior_strength,
                random_seed=self.random_seed,
            )
