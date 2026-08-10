"""AutoGluon/TabArena adapter for CTBoost.

The adapter intentionally lives outside the runtime package.  TabArena owns
dataset splitting, resource accounting, bagging, metrics, and result formats;
this module only translates its ``AbstractModel`` contract into CTBoost's
scikit-learn API.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import os
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

try:
    from autogluon.core.models import AbstractModel
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in benchmark env
    _AUTOGLUON_IMPORT_ERROR: Optional[ModuleNotFoundError] = exc

    class AbstractModel:  # type: ignore[no-redef]
        """Import-time placeholder that gives users an actionable error."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "The CTBoost TabArena adapter requires AutoGluon and TabArena. "
                "Follow benchmarks/tabarena/README.md to create the benchmark environment."
            ) from _AUTOGLUON_IMPORT_ERROR

else:
    _AUTOGLUON_IMPORT_ERROR = None

_MISSING_CATEGORY = "__CTBOOST_MISSING__"
_HISTOGRAM_THREAD_ENV = "CTBOOST_HIST_THREADS"
_HISTOGRAM_THREAD_LOCK = threading.Lock()
_MIN_TRAINING_BUDGET_FRACTION = 0.4


def _stopping_metric_name(metric: Any) -> Optional[str]:
    """Return AutoGluon's scorer name in a stable, comparison-friendly form."""
    if metric is None:
        return None
    name = getattr(metric, "name", None)
    if callable(name):
        try:
            name = name()
        except TypeError:
            name = None
    if not isinstance(name, str):
        if isinstance(metric, str):
            name = metric
        else:
            name = getattr(metric, "__name__", None)
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _ctboost_eval_metric(problem_type: Any, stopping_metric: Any) -> Optional[str]:
    """Translate supported AutoGluon stopping scorers to CTBoost metrics."""
    resolved_problem_type = str(problem_type).strip().lower()
    resolved_metric = _stopping_metric_name(stopping_metric)
    mappings = {
        "binary": {"roc_auc": "AUC", "log_loss": "Logloss"},
        "multiclass": {"log_loss": "MultiClass"},
        "regression": {
            "rmse": "RMSE",
            "root_mean_squared_error": "RMSE",
        },
    }
    return mappings.get(resolved_problem_type, {}).get(resolved_metric)


def _resolve_time_limit(time_limit: Optional[float]) -> Optional[float]:
    if time_limit is None:
        return None
    resolved = float(time_limit)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("time_limit must be finite and positive when provided")
    return resolved


def _callback_list(callbacks: Any) -> list[Any]:
    if callbacks is None:
        return []
    if callable(callbacks):
        return [callbacks]
    try:
        return list(callbacks)
    except TypeError as exc:
        raise TypeError("callbacks must be callable or an iterable of callables") from exc


def _raise_time_limit_exceeded() -> None:
    message = "Insufficient AutoGluon fit budget remains after CTBoost adapter setup"
    try:
        from autogluon.core.utils.exceptions import TimeLimitExceeded
    except ModuleNotFoundError as exc:  # pragma: no cover - adapter requires AutoGluon
        raise RuntimeError(
            message + "; AutoGluon's TimeLimitExceeded exception is unavailable"
        ) from exc
    raise TimeLimitExceeded(message)


def _deadline_callback(deadline: float, training_started_at: float) -> Any:
    """Stop when the budget cannot safely fit two average tree iterations."""

    def _stop_at_deadline(env: Any) -> bool:
        now = time.monotonic()
        if now >= deadline:
            return True
        completed_iterations = max(1, int(env.iteration) - int(env.begin_iteration) + 1)
        average_iteration_time = max(
            0.0,
            (now - training_started_at) / completed_iterations,
        )
        return now + 2.0 * average_iteration_time >= deadline

    return _stop_at_deadline


@contextmanager
def _ctboost_histogram_threads(num_cpus: Any) -> Any:
    """Keep native histogram workers inside TabArena's per-fit CPU budget."""
    resolved = max(1, int(num_cpus))
    with _HISTOGRAM_THREAD_LOCK:
        previous = os.environ.get(_HISTOGRAM_THREAD_ENV)
        os.environ[_HISTOGRAM_THREAD_ENV] = str(resolved)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_HISTOGRAM_THREAD_ENV, None)
            else:
                os.environ[_HISTOGRAM_THREAD_ENV] = previous


def _categorical_columns(frame: Any) -> list[str]:
    """Return columns that should use CTBoost's native categorical pipeline."""
    columns: list[str] = []
    for name, dtype in frame.dtypes.items():
        dtype_name = str(dtype).lower()
        if dtype_name in {"object", "category", "string"} or dtype_name.startswith("string["):
            columns.append(str(name))
    return columns


def normalize_tabarena_frame(
    frame: Any,
    *,
    categorical_columns: Optional[Sequence[str]] = None,
) -> tuple[Any, list[str]]:
    """Normalize a TabArena frame without ordinal-encoding categoricals.

    CatBoost and XGBoost are evaluated with categorical awareness in TabArena.
    Keeping category values as strings lets CTBoost exercise its own fitted CTR
    and unknown-category handling instead of receiving an unfair pre-encoded
    representation.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - benchmark dependency
        raise ModuleNotFoundError("TabArena benchmarking requires pandas") from exc

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    normalized = frame.copy()
    normalized.columns = [str(name) for name in normalized.columns]

    resolved_categoricals = (
        _categorical_columns(normalized)
        if categorical_columns is None
        else [str(name) for name in categorical_columns]
    )
    missing_columns = [name for name in resolved_categoricals if name not in normalized.columns]
    if missing_columns:
        raise ValueError(
            "TabArena prediction data is missing categorical columns seen during fit: "
            + ", ".join(missing_columns)
        )

    for name in resolved_categoricals:
        values = normalized[name].astype(object)
        values = values.where(~pd.isna(values), _MISSING_CATEGORY)
        normalized[name] = values.map(str)

    for name in normalized.columns:
        if name in resolved_categoricals:
            continue
        if str(normalized[name].dtype).lower() in {"bool", "boolean"}:
            normalized[name] = normalized[name].astype(np.int8)

    return normalized, resolved_categoricals


class CTBoostTabArenaModel(AbstractModel):
    """TabArena execution wrapper around CTBoost's sklearn estimators."""

    ag_key = "CTB"
    ag_name = "CTBoost"
    ag_priority = 65
    seed_name = "random_seed"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = {
        "valid_raw_types": ["bool", "int", "float", "category", "object"],
        "ignored_type_group_raw": ["datetime_as_object"],
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ctboost_categorical_columns: list[str] = []

    def _preprocess(self, X: Any, is_train: bool = False, **kwargs: Any) -> Any:
        X = super()._preprocess(X, **kwargs)
        categorical_columns = None if is_train else self._ctboost_categorical_columns
        X, resolved = normalize_tabarena_frame(
            X,
            categorical_columns=categorical_columns,
        )
        if is_train:
            self._ctboost_categorical_columns = resolved
        return X

    def _fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        y_val: Any = None,
        sample_weight: Any = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        time_limit: Optional[float] = None,
        callbacks: Any = None,
        **kwargs: Any,
    ) -> None:
        resolved_time_limit = _resolve_time_limit(time_limit)
        training_deadline = None
        if resolved_time_limit is not None:
            fit_started_at = time.monotonic()
            training_deadline = fit_started_at + resolved_time_limit
        from ctboost import CTBoostClassifier, CTBoostRegressor

        X_train = self.preprocess(X, y=y, is_train=True)
        X_validation = None
        if X_val is not None:
            X_validation = self.preprocess(X_val, is_train=False)

        params = dict(self._get_model_params())
        early_stopping_rounds = int(params.pop("early_stopping_rounds", 50))
        configured_callbacks = _callback_list(params.pop("callbacks", None))
        configured_callbacks.extend(_callback_list(callbacks))
        if "eval_metric" not in params:
            eval_metric = _ctboost_eval_metric(
                self.problem_type,
                getattr(self, "stopping_metric", None),
            )
            if eval_metric is not None:
                params["eval_metric"] = eval_metric
        params["cat_features"] = self._ctboost_categorical_columns or None
        if num_gpus and "task_type" not in params:
            params["task_type"] = "GPU"

        if self.problem_type == "regression":
            self.model = CTBoostRegressor(**params)
        else:
            self.model = CTBoostClassifier(**params)

        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
        if X_validation is not None and y_val is not None:
            fit_kwargs.update(
                {
                    "eval_set": (X_validation, np.asarray(y_val)),
                    "early_stopping_rounds": early_stopping_rounds,
                }
            )
        with _ctboost_histogram_threads(num_cpus):
            if training_deadline is not None:
                training_started_at = time.monotonic()
                remaining_time = training_deadline - training_started_at
                if remaining_time <= resolved_time_limit * _MIN_TRAINING_BUDGET_FRACTION:
                    _raise_time_limit_exceeded()
                configured_callbacks.append(
                    _deadline_callback(training_deadline, training_started_at)
                )
            if configured_callbacks:
                fit_kwargs["callbacks"] = configured_callbacks
            self.model.fit(X_train, np.asarray(y), **fit_kwargs)

    def _set_default_params(self) -> None:
        defaults = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "alpha": 0.05,
            "lambda_l2": 1.0,
            "subsample": 0.8,
            "bootstrap_type": "Bernoulli",
            "ordered_ctr": True,
            "max_cat_threshold": 64,
            "early_stopping_rounds": 50,
            "verbose": False,
        }
        for name, value in defaults.items():
            self._set_default_param_value(name, value)

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: Any,
        hyperparameters: Optional[dict[str, Any]] = None,
        num_classes: Optional[int] = 1,
        **kwargs: Any,
    ) -> int:
        """Conservatively estimate peak fit memory for fold scheduling.

        CTBoost keeps the input frame, quantized bins, gradient/statistic buffers,
        and the fitted trees resident during training.  Including the worst-case
        tree payload is important for multiclass datasets, where a depth-only
        estimate can otherwise let AutoGluon start too many folds in parallel.
        """
        del kwargs
        params = dict(hyperparameters or {})
        rows, columns = (int(value) for value in X.shape)
        classes = max(1, int(num_classes or 1))
        try:
            from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

            input_bytes = int(get_approximate_df_mem_usage(X).sum())
        except (ImportError, AttributeError, TypeError):
            memory_usage = getattr(X, "memory_usage", None)
            input_bytes = (
                int(memory_usage(index=True, deep=True).sum())
                if callable(memory_usage)
                else rows * columns * 8
            )

        max_bins = max(1, int(params.get("max_bins", 256)))
        bin_width = 1 if max_bins <= 256 else 2 if max_bins <= 65_535 else 4
        quantized_bytes = rows * columns * bin_width
        statistic_bytes = rows * classes * 8 * 6
        histogram_bytes = columns * max_bins * classes * 8 * 3

        depth = max(0, int(params.get("max_depth", params.get("depth", 6))))
        depth_leaves = 1 << min(depth, 20)
        configured_leaves = int(params.get("max_leaves", 0) or 0)
        leaves = min(depth_leaves, configured_leaves) if configured_leaves > 0 else depth_leaves
        iterations = max(1, int(params.get("iterations", params.get("n_estimators", 1000))))
        tree_bytes = iterations * classes * max(1, 2 * leaves - 1) * 64

        baseline_bytes = 512 * 1024 * 1024
        return int(
            baseline_bytes
            + 4 * input_bytes
            + 2 * quantized_bytes
            + statistic_bytes
            + histogram_bytes
            + tree_bytes
        )


class CTBoostTabArenaGPUModel(CTBoostTabArenaModel):
    """GPU-pinned TabArena variant with a distinct leaderboard identity."""

    ag_key = "CTB_GPU"
    ag_name = "CTBoostGPU"
    default_num_gpus = 1
    minimum_num_gpus = 1
    gpu_required = True


def _finalize_search_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve internal conditional knobs into valid CTBoost parameters."""
    resolved = {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in config.items()
    }
    leaf_fraction = resolved.pop("__leaf_fraction", None)
    if leaf_fraction is None:
        # A leaf cap makes depth-first growth stop part-way through a level.  It
        # is useful only for the best-first LeafWise policy in this search.
        resolved["max_leaves"] = 0
    else:
        full_leaf_count = 1 << int(resolved["max_depth"])
        resolved["max_leaves"] = max(
            4,
            min(full_leaf_count - 1, round(float(leaf_fraction) * full_leaf_count)),
        )
    return resolved


def generate_configs_ctboost(num_random_configs: int = 200) -> list[dict[str, Any]]:
    """Generate a deterministic, task-safe 200-config TabArena search.

    The tree-count cap remains fixed at 1,000 and every TabArena fold supplies a
    validation set for early stopping.  Consequently the learning-rate floor is
    deliberately 0.02: lower values consumed a large part of the old 200-config
    budget while frequently reaching the cap before converging.  Depth is capped
    at eight to keep multiclass CPU fits and model artifacts within the official
    per-fit resource envelope.
    """
    count = int(num_random_configs)
    if count < 0:
        raise ValueError("num_random_configs must be non-negative")
    if count == 0:
        return []

    from ConfigSpace import Categorical, ConfigurationSpace, EqualsCondition, Float, Integer

    search_space = ConfigurationSpace(seed=1234)
    learning_rate = Float("learning_rate", (2e-2, 2e-1), log=True, default=5e-2)
    max_depth = Integer("max_depth", (3, 8), default=6)
    alpha = Float("alpha", (5e-3, 5e-1), log=True, default=5e-2)
    lambda_l2 = Float("lambda_l2", (1e-4, 10.0), log=True, default=1.0)
    subsample = Float("subsample", (0.6, 1.0), default=0.8)
    colsample_bytree = Float("colsample_bytree", (0.6, 1.0), default=1.0)
    grow_policy = Categorical("grow_policy", ["DepthWise", "LeafWise"], default="DepthWise")
    leaf_fraction = Categorical("__leaf_fraction", [0.25, 0.5, 0.75], default=0.5)
    min_data_in_leaf = Integer("min_data_in_leaf", (1, 64), log=True, default=1)
    min_child_weight = Categorical(
        "min_child_weight",
        [0.0, 0.01, 0.1, 1.0, 5.0],
        default=0.0,
    )
    one_hot_max_size = Categorical("one_hot_max_size", [2, 4, 16, 64], default=4)
    max_cat_threshold = Categorical("max_cat_threshold", [16, 64, 256], default=64)
    ordered_ctr = Categorical("ordered_ctr", [False, True], default=True)
    ctr_prior_strength = Float("ctr_prior_strength", (0.1, 10.0), log=True, default=1.0)
    random_strength = Categorical("random_strength", [0.0, 0.01, 0.1, 1.0], default=0.0)
    max_bins = Categorical("max_bins", [128, 256], default=256)
    search_space.add(
        [
            learning_rate,
            max_depth,
            alpha,
            lambda_l2,
            subsample,
            colsample_bytree,
            grow_policy,
            leaf_fraction,
            min_data_in_leaf,
            min_child_weight,
            one_hot_max_size,
            max_cat_threshold,
            ordered_ctr,
            ctr_prior_strength,
            random_strength,
            max_bins,
        ]
    )
    search_space.add(
        [
            EqualsCondition(leaf_fraction, grow_policy, "LeafWise"),
            EqualsCondition(ctr_prior_strength, ordered_ctr, True),
        ]
    )

    sampled = search_space.sample_configuration(count)
    configs = [sampled] if count == 1 else sampled
    return [_finalize_search_config(dict(config)) for config in configs]


def _build_config_generator(model_cls: Any = CTBoostTabArenaModel) -> Any:
    if _AUTOGLUON_IMPORT_ERROR is not None:
        return None
    try:
        from tabarena.utils.config_utils import CustomAGConfigGenerator
    except ModuleNotFoundError:
        return None
    return CustomAGConfigGenerator(
        model_cls=model_cls,
        search_space_func=generate_configs_ctboost,
        manual_configs=[{}],
    )


gen_ctboost_cpu = _build_config_generator(CTBoostTabArenaModel)
gen_ctboost_gpu = _build_config_generator(CTBoostTabArenaGPUModel)
gen_ctboost = gen_ctboost_cpu


__all__ = [
    "CTBoostTabArenaGPUModel",
    "CTBoostTabArenaModel",
    "gen_ctboost",
    "gen_ctboost_cpu",
    "gen_ctboost_gpu",
    "generate_configs_ctboost",
    "normalize_tabarena_frame",
]
