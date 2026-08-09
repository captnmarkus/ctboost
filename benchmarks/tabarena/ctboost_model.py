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
    seed_name = "random_seed"
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

    def _get_default_auxiliary_params(self) -> dict[str, Any]:
        params = super()._get_default_auxiliary_params()
        params.update(
            {
                "valid_raw_types": ["bool", "int", "float", "category", "object"],
                "ignored_type_group_raw": ["datetime_as_object"],
            }
        )
        return params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]


class CTBoostTabArenaGPUModel(CTBoostTabArenaModel):
    """GPU-pinned TabArena variant with a distinct leaderboard identity."""

    ag_key = "CTB_GPU"
    ag_name = "CTBoostGPU"
    default_num_gpus = 1
    minimum_num_gpus = 1
    gpu_required = True


def generate_configs_ctboost(num_random_configs: int = 200) -> list[dict[str, Any]]:
    """Generate the author-controlled CTBoost search space for TabArena."""
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 2e-1), log=True),
            Integer("max_depth", (3, 10)),
            Float("alpha", (1e-3, 2e-1), log=True),
            Float("lambda_l2", (1e-4, 10.0), log=True),
            Float("subsample", (0.6, 1.0)),
            Float("colsample_bytree", (0.6, 1.0)),
            Categorical("grow_policy", ["DepthWise", "LeafWise"]),
            Integer("max_leaves", (8, 1024), log=True),
            Integer("min_data_in_leaf", (1, 100), log=True),
            Float("min_child_weight", (1e-3, 10.0), log=True),
            Integer("one_hot_max_size", (2, 100), log=True),
            Integer("max_cat_threshold", (8, 256), log=True),
            Categorical("ordered_ctr", [False, True]),
        ],
        seed=1234,
    )
    configs = search_space.sample_configuration(int(num_random_configs))
    if int(num_random_configs) == 1:
        configs = [configs]
    return [
        {
            name: value.item() if isinstance(value, np.generic) else value
            for name, value in dict(config).items()
        }
        for config in configs
    ]


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
