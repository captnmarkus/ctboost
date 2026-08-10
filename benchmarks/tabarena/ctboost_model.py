"""AutoGluon/TabArena adapter for CTBoost.

The adapter intentionally lives outside the runtime package.  TabArena owns
dataset splitting, resource accounting, bagging, metrics, and result formats;
this module only translates its ``AbstractModel`` contract into CTBoost's
scikit-learn API.
"""

from __future__ import annotations

import math
import os
import threading
import time
from contextlib import contextmanager
from itertools import combinations
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
                "Follow benchmarks/tabarena/README.md to create the benchmark "
                "environment."
            ) from _AUTOGLUON_IMPORT_ERROR

else:
    _AUTOGLUON_IMPORT_ERROR = None

_HISTOGRAM_THREAD_ENV = "CTBOOST_HIST_THREADS"
_HISTOGRAM_THREAD_LOCK = threading.Lock()
_MIN_TRAINING_BUDGET_FRACTION = 0.4
TABARENA_SEARCH_PORTFOLIO_SIZE = 200
_SEARCH_PORTFOLIO_SEED = 1234
_PAIR_BUDGET_PARAM = "tabarena_categorical_pair_budget"
_PAIR_BUDGET_LIMIT = 4
_PAIR_CANDIDATE_COLUMN_LIMIT = 16
_PAIR_JOINT_CARDINALITY_LIMIT = 4096


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
        raise TypeError(
            "callbacks must be callable or an iterable of callables"
        ) from exc


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
    # AutoGluon resolves ``None`` before calling a model.  Retaining a
    # conservative fallback makes direct FitHelper/adapter use safe as well.
    resolved = 1 if num_cpus is None else max(1, int(num_cpus))
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
        if dtype_name in {"object", "category", "string"} or dtype_name.startswith(
            "string["
        ):
            columns.append(str(name))
    return columns


def _resolve_categorical_pair_budget(value: Any) -> int:
    """Validate the small adapter-only categorical-pair budget."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{_PAIR_BUDGET_PARAM} must be an integer")
    resolved = int(value)
    if not 0 <= resolved <= _PAIR_BUDGET_LIMIT:
        raise ValueError(
            f"{_PAIR_BUDGET_PARAM} must be between 0 and {_PAIR_BUDGET_LIMIT}"
        )
    return resolved


def normalize_tabarena_frame(
    frame: Any,
    *,
    categorical_columns: Optional[Sequence[str]] = None,
) -> tuple[Any, list[str]]:
    """Preserve a TabArena frame for CTBoost's native data handling.

    CatBoost and XGBoost are evaluated with categorical awareness in TabArena.
    CTBoost natively handles pandas categorical/object/string values, missing
    values, booleans, and unseen categories.  The adapter therefore records the
    training schema without stringifying or replacing values; doing so could
    alias a literal category with a missing-value sentinel.
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
    missing_columns = [
        name for name in resolved_categoricals if name not in normalized.columns
    ]
    if missing_columns:
        raise ValueError(
            "TabArena prediction data is missing categorical columns seen during fit: "
            + ", ".join(missing_columns)
        )

    return normalized, resolved_categoricals


def _bounded_categorical_pairs(
    frame: Any,
    categorical_columns: Sequence[str],
    *,
    max_pairs: int,
    max_joint_cardinality: int = _PAIR_JOINT_CARDINALITY_LIMIT,
) -> list[list[str]]:
    """Select a small deterministic set of inexpensive categorical pairs.

    CTBoost can generate every pair automatically, but that is quadratic in the
    number of categorical columns.  TabArena configurations instead opt into at
    most four explicit pairs.  Candidate columns and cardinality products are
    bounded using the training fold only; validation/test values never influence
    feature construction.
    """
    budget = max(0, int(max_pairs))
    if budget == 0 or len(categorical_columns) < 2:
        return []

    column_order = {name: index for index, name in enumerate(categorical_columns)}
    cardinalities: list[tuple[int, int, str]] = []
    for name in categorical_columns:
        cardinality = int(frame[name].nunique(dropna=False))
        # Constant columns cannot add information to a combination.  Very high
        # cardinality columns are excluded before forming the bounded candidate
        # set so the pair construction cannot unexpectedly inflate a fold.
        if 1 < cardinality <= max_joint_cardinality:
            cardinalities.append((cardinality, column_order[name], name))

    cardinalities.sort()
    candidates = cardinalities[:_PAIR_CANDIDATE_COLUMN_LIMIT]
    ranked_pairs: list[tuple[int, int, int, str, str]] = []
    for left, right in combinations(candidates, 2):
        joint_upper_bound = left[0] * right[0]
        if joint_upper_bound <= max_joint_cardinality:
            left_order, right_order = sorted((left[1], right[1]))
            left_name = categorical_columns[left_order]
            right_name = categorical_columns[right_order]
            ranked_pairs.append(
                (joint_upper_bound, left_order, right_order, left_name, right_name)
            )

    ranked_pairs.sort()
    return [[left, right] for _, _, _, left, right in ranked_pairs[:budget]]


class CTBoostTabArenaModel(AbstractModel):
    """TabArena execution wrapper around CTBoost's sklearn estimators."""

    ag_key = "CTB"
    ag_name = "CTBoost"
    ag_priority = 65
    seed_name = "random_seed"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = {
        "valid_raw_types": ["bool", "int", "float", "category", "object"],
        "ignored_type_group_special": ["datetime_as_object"],
    }
    default_resources_physical_cores_only = True
    default_num_gpus = 0
    _ctboost_task_type = "CPU"

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
        categorical_pair_budget = _resolve_categorical_pair_budget(
            params.pop(_PAIR_BUDGET_PARAM, 0)
        )
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
        if (
            categorical_pair_budget
            and "categorical_combinations" not in params
            and not params.get("pairwise_categorical_combinations", False)
        ):
            categorical_pairs = _bounded_categorical_pairs(
                X_train,
                self._ctboost_categorical_columns,
                max_pairs=categorical_pair_budget,
            )
            if categorical_pairs:
                params["categorical_combinations"] = categorical_pairs
        # Resource allocation and model identity are separate contracts.  In
        # particular, a mixed CPU/GPU TabArena run must never turn the CPU
        # leaderboard entry into a GPU fit merely because a GPU is available.
        params["task_type"] = self._ctboost_task_type

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
                if (
                    remaining_time
                    <= resolved_time_limit * _MIN_TRAINING_BUDGET_FRACTION
                ):
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
        rows, raw_columns = (int(value) for value in X.shape)
        classes = max(1, int(num_classes or 1))
        try:
            from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

            input_bytes = int(get_approximate_df_mem_usage(X).sum())
        except (ImportError, AttributeError, TypeError):
            memory_usage = getattr(X, "memory_usage", None)
            input_bytes = (
                int(memory_usage(index=True, deep=True).sum())
                if callable(memory_usage)
                else rows * raw_columns * 8
            )

        pair_budget = _resolve_categorical_pair_budget(
            params.get(_PAIR_BUDGET_PARAM, 0)
        )
        columns = raw_columns + min(pair_budget, _PAIR_BUDGET_LIMIT)
        max_bins = max(1, int(params.get("max_bins", 256)))
        bin_width = 1 if max_bins <= 256 else 2 if max_bins <= 65_535 else 4
        quantized_bytes = rows * columns * bin_width
        statistic_bytes = rows * classes * 8 * 6
        histogram_bytes = columns * max_bins * classes * 8 * 3

        depth = max(0, int(params.get("max_depth", params.get("depth", 6))))
        depth_leaves = 1 << min(depth, 20)
        configured_leaves = int(params.get("max_leaves", 0) or 0)
        leaves = (
            min(depth_leaves, configured_leaves)
            if configured_leaves > 0
            else depth_leaves
        )
        iterations = max(
            1, int(params.get("iterations", params.get("n_estimators", 1000)))
        )
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
    _ctboost_task_type = "GPU"


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

    pair_budget = int(resolved.pop("__categorical_pair_budget", 0))
    if pair_budget and resolved.get("ordered_ctr", False):
        resolved[_PAIR_BUDGET_PARAM] = pair_budget

    learning_rate = resolved.get("learning_rate")
    if learning_rate is not None:
        iterations, early_stopping_rounds = _adaptive_training_budget(
            float(learning_rate)
        )
        resolved.setdefault("iterations", iterations)
        resolved.setdefault("early_stopping_rounds", early_stopping_rounds)
    return resolved


def _adaptive_training_budget(learning_rate: float) -> tuple[int, int]:
    """Return a bounded tree cap and validation patience for a learning rate.

    The cap preserves roughly the default shrinkage path (``1000 * 0.05``),
    while clamps prevent very low-rate configurations from monopolizing the
    one-hour TabArena fit budget.  Early stopping still selects the retained
    tree count; this only provides appropriately scaled headroom.
    """
    resolved = float(learning_rate)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    iterations = int(round(min(1600.0, max(400.0, 50.0 / resolved)) / 50.0) * 50)
    patience = round(min(80.0, max(30.0, 0.075 * iterations)))
    return iterations, patience


def _stratified_unit_samples(count: int, dimensions: int) -> np.ndarray:
    """Create a progressively ordered deterministic Latin-hypercube design."""
    rng = np.random.default_rng(_SEARCH_PORTFOLIO_SEED)
    samples = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = rng.permutation(count)
        samples[:, dimension] = (strata + rng.random(count)) / count

    # Greedy maximin ordering makes small smoke/HPO prefixes cover the full
    # design instead of inheriting an arbitrary random row order.  Starting
    # from the centre also complements the separately evaluated manual default.
    remaining = np.ones(count, dtype=bool)
    minimum_distance = np.sum(np.square(samples - 0.5), axis=1)
    order: list[int] = []
    for _ in range(count):
        candidate_scores = np.where(remaining, minimum_distance, -1.0)
        selected = int(np.argmax(candidate_scores))
        order.append(selected)
        remaining[selected] = False
        distances = np.sum(np.square(samples - samples[selected]), axis=1)
        minimum_distance = np.minimum(minimum_distance, distances)
    return samples[order]


def _linear_sample(value: float, lower: float, upper: float) -> float:
    return float(lower * (1.0 - value) + upper * value)


def _log_sample(value: float, lower: float, upper: float) -> float:
    return float(math.exp(math.log(lower) * (1.0 - value) + math.log(upper) * value))


def _integer_sample(value: float, lower: int, upper: int) -> int:
    return min(upper, lower + int(value * (upper - lower + 1)))


def _log_integer_sample(value: float, lower: int, upper: int) -> int:
    return min(upper, max(lower, round(_log_sample(value, lower, upper))))


def _categorical_sample(value: float, choices: Sequence[Any]) -> Any:
    return choices[min(len(choices) - 1, int(value * len(choices)))]


def generate_configs_ctboost(num_random_configs: int = 200) -> list[dict[str, Any]]:
    """Generate the frozen, stratified 200-config TabArena portfolio.

    TabArena's :class:`CustomAGConfigGenerator` accepts any deterministic
    ``num_configs -> list[dict]`` callable; ConfigSpace is not part of that
    contract.  A fixed Latin-hypercube design gives each numeric dimension even
    coverage and exact categorical balance instead of spending the small budget
    on clusters from an unconstrained random draw.  Conditional parameters are
    omitted when inactive, every configuration remains valid for regression,
    binary, and multiclass tasks, and no model seed is hard-coded (AutoGluon owns
    the fold/config seed).
    """
    count = int(num_random_configs)
    if count < 0:
        raise ValueError("num_random_configs must be non-negative")
    if count == 0:
        return []
    if count > TABARENA_SEARCH_PORTFOLIO_SIZE:
        raise ValueError(
            "num_random_configs cannot exceed the frozen "
            f"{TABARENA_SEARCH_PORTFOLIO_SIZE}-config portfolio"
        )

    samples = _stratified_unit_samples(TABARENA_SEARCH_PORTFOLIO_SIZE, dimensions=17)[
        :count
    ]
    configs: list[dict[str, Any]] = []
    for values in samples:
        ordered_ctr = bool(values[12] >= 0.25)
        grow_policy = str(_categorical_sample(values[6], ["DepthWise", "LeafWise"]))
        config: dict[str, Any] = {
            "learning_rate": _log_sample(values[0], 2e-2, 2e-1),
            "max_depth": _integer_sample(values[1], 3, 8),
            "alpha": _log_sample(values[2], 5e-3, 5e-1),
            "lambda_l2": _log_sample(values[3], 1e-4, 10.0),
            "subsample": _linear_sample(values[4], 0.6, 1.0),
            "colsample_bytree": _linear_sample(values[5], 0.6, 1.0),
            "grow_policy": grow_policy,
            "min_data_in_leaf": _log_integer_sample(values[8], 1, 64),
            "min_child_weight": _categorical_sample(
                values[9], [0.0, 0.01, 0.1, 1.0, 5.0]
            ),
            "one_hot_max_size": _categorical_sample(values[10], [2, 4, 16, 64]),
            "max_cat_threshold": _categorical_sample(values[11], [16, 64, 256]),
            "ordered_ctr": ordered_ctr,
            "random_strength": _categorical_sample(values[14], [0.0, 0.01, 0.1, 1.0]),
            "max_bins": _categorical_sample(values[15], [128, 256]),
            "bootstrap_type": "Bernoulli",
        }
        if grow_policy == "LeafWise":
            config["__leaf_fraction"] = _categorical_sample(
                values[7], [0.25, 0.5, 0.75]
            )
        if ordered_ctr:
            config["ctr_prior_strength"] = _log_sample(values[13], 0.1, 10.0)
            config["__categorical_pair_budget"] = _categorical_sample(
                values[16], [0, 0, 0, 2, 4]
            )
        configs.append(_finalize_search_config(config))

    return configs


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
    "TABARENA_SEARCH_PORTFOLIO_SIZE",
    "CTBoostTabArenaGPUModel",
    "CTBoostTabArenaModel",
    "gen_ctboost",
    "gen_ctboost_cpu",
    "gen_ctboost_gpu",
    "generate_configs_ctboost",
    "normalize_tabarena_frame",
]
