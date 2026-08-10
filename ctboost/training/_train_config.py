"""Compatibility normalization and validation for low-level train parameters."""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Mapping


_TRAIN_PARAM_ALIASES = {
    "n_estimators": "iterations",
    "num_trees": "iterations",
    "eta": "learning_rate",
    "depth": "max_depth",
    "reg_lambda": "lambda_l2",
    "l2_leaf_reg": "lambda_l2",
    "lambda": "lambda_l2",
    "random_state": "random_seed",
    "seed": "random_seed",
    "max_bin": "max_bins",
}

_NATIVE_TRAIN_PARAM_KEYS = {
    "alpha",
    "bagging_temperature",
    "boosting_type",
    "bootstrap_type",
    "boost_from_average",
    "base_score",
    "border_selection_method",
    "colsample_bytree",
    "devices",
    "drop_rate",
    "eval_metric",
    "feature_borders",
    "feature_weights",
    "first_feature_use_penalties",
    "gamma",
    "grow_policy",
    "huber_delta",
    "interaction_constraints",
    "iterations",
    "lambda_l2",
    "leaf_estimation_iterations",
    "learning_rate",
    "loss_function",
    "max_bin_by_feature",
    "max_bins",
    "max_depth",
    "max_drop",
    "max_leaf_weight",
    "max_leaves",
    "min_child_weight",
    "min_data_in_leaf",
    "min_samples_split",
    "monotone_constraints",
    "nan_mode",
    "nan_mode_by_feature",
    "num_classes",
    "objective",
    "quantile_alpha",
    "random_seed",
    "random_strength",
    "skip_drop",
    "subsample",
    "task_type",
    "tweedie_variance_power",
    "verbose",
}

_PREPROCESSING_PARAM_KEYS = {
    "cat_features",
    "categorical_combinations",
    "combinations_ctr",
    "ctr_prior_strength",
    "embedding_features",
    "embedding_stats",
    "embedding_target_features",
    "embedding_target_regularization",
    "embedding_target_mode",
    "max_cat_threshold",
    "max_cat_to_onehot",
    "one_hot_max_size",
    "ordered_ctr",
    "pairwise_categorical_combinations",
    "per_feature_ctr",
    "simple_ctr",
    "text_features",
    "text_feature_calcer",
    "text_hash_dim",
    "text_lowercase",
    "text_max_dictionary_size",
    "text_min_token_count",
    "text_ngram_range",
    "text_tokenizer",
}

_EXTERNAL_MEMORY_PARAM_KEYS = {
    "eval_external_memory",
    "eval_external_memory_dir",
    "external_memory",
    "external_memory_dir",
}

_DISTRIBUTED_PARAM_KEYS = {
    "distributed_rank",
    "distributed_root",
    "distributed_run_id",
    "distributed_timeout",
    "distributed_world_size",
}

_WEIGHTING_PARAM_KEYS = {
    "auto_class_weights",
    "class_weights",
    "scale_pos_weight",
}

_SUPPORTED_TRAIN_PARAM_KEYS = (
    _NATIVE_TRAIN_PARAM_KEYS
    | _PREPROCESSING_PARAM_KEYS
    | _EXTERNAL_MEMORY_PARAM_KEYS
    | _DISTRIBUTED_PARAM_KEYS
    | _WEIGHTING_PARAM_KEYS
)


def _normalize_training_config(params: Mapping[str, Any]) -> Dict[str, Any]:
    config = dict(params)
    if any(not isinstance(key, str) for key in config):
        raise TypeError("training parameter names must be strings")

    aliases_by_canonical: Dict[str, List[str]] = {}
    for alias, canonical_name in _TRAIN_PARAM_ALIASES.items():
        if alias in config:
            aliases_by_canonical.setdefault(canonical_name, []).append(alias)

    for canonical_name, aliases in aliases_by_canonical.items():
        supplied_names = ([canonical_name] if canonical_name in config else []) + aliases
        if len(supplied_names) > 1:
            formatted_names = ", ".join(repr(name) for name in supplied_names)
            raise ValueError(
                f"training parameters {formatted_names} are aliases for {canonical_name!r} "
                "and cannot be used together"
            )
        alias = aliases[0]
        config[canonical_name] = config.pop(alias)

    unknown_names = sorted(set(config) - _SUPPORTED_TRAIN_PARAM_KEYS)
    if unknown_names:
        accepted_names = sorted(_SUPPORTED_TRAIN_PARAM_KEYS | set(_TRAIN_PARAM_ALIASES))
        details = []
        for name in unknown_names:
            matches = difflib.get_close_matches(name, accepted_names, n=1, cutoff=0.6)
            suggestion = "" if not matches else f"; did you mean {matches[0]!r}?"
            details.append(f"{name!r}{suggestion}")
        noun = "parameter" if len(details) == 1 else "parameters"
        raise ValueError(f"unknown training {noun}: {', '.join(details)}")
    return config
