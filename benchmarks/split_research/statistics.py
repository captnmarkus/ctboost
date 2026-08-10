"""Pure-Python reference statistics for conditional-split research.

This module deliberately lives under :mod:`benchmarks`: it is an executable
specification for experiments, not part of CTBoost's production split engine.
Frequency weights are interpreted as counts of exchangeable observations.
"""

# ruff: noqa: UP006, UP007, UP045
# CTBoost supports Python 3.8, where built-in generic, union, and optional
# shorthand syntax is not parseable even when annotations are postponed.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class TestResult:
    """Result of one conditional feature test."""

    method: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    n_observations: int
    weight_sum: float
    details: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BonferroniDecision:
    """Global-null decision used before a conditional tree is allowed to split."""

    selected_index: Optional[int]
    raw_p_value: float
    adjusted_p_value: float
    alpha: float
    n_tests: int
    should_split: bool


def _regularized_gamma_q(shape: float, x: float) -> float:
    """Regularized upper incomplete gamma, using stable series/fraction forms."""

    if shape <= 0.0:
        raise ValueError("shape must be positive")
    if x <= 0.0:
        return 1.0
    if not math.isfinite(x):
        return 0.0

    epsilon = 1e-14
    tiny = 1e-300
    max_iterations = 10000
    log_scale = -x + shape * math.log(x) - math.lgamma(shape)

    if x < shape + 1.0:
        term = 1.0 / shape
        series = term
        cursor = shape
        for _ in range(max_iterations):
            cursor += 1.0
            term *= x / cursor
            series += term
            if abs(term) <= abs(series) * epsilon:
                break
        lower = series * math.exp(log_scale)
        return min(1.0, max(0.0, 1.0 - lower))

    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / max(abs(b), tiny)
    if b < 0.0:
        d = -d
    fraction = d
    for iteration in range(1, max_iterations + 1):
        coefficient = -float(iteration) * (float(iteration) - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        fraction *= delta
        if abs(delta - 1.0) <= epsilon:
            break
    return min(1.0, max(0.0, math.exp(log_scale) * fraction))


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Survival function of a chi-square random variable without SciPy."""

    if degrees_of_freedom <= 0 or statistic <= 0.0:
        return 1.0
    return _regularized_gamma_q(0.5 * degrees_of_freedom, 0.5 * statistic)


def _missing_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(values.shape[0], dtype=bool)
    for index, value in enumerate(values):
        if value is None:
            mask[index] = True
            continue
        try:
            mask[index] = bool(np.isnan(value))
        except (TypeError, ValueError):
            mask[index] = False
    return mask


def _validate_inputs(
    scores: ArrayLike,
    feature: Sequence[Any],
    frequency_weights: Optional[ArrayLike],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    feature_array = np.asarray(feature)
    if score_array.ndim != 1 or feature_array.ndim != 1:
        raise ValueError("scores and feature must be one-dimensional")
    if score_array.shape[0] != feature_array.shape[0]:
        raise ValueError("scores and feature must have the same length")
    if frequency_weights is None:
        weight_array = np.ones(score_array.shape[0], dtype=np.float64)
    else:
        weight_array = np.asarray(frequency_weights, dtype=np.float64)
        if weight_array.ndim != 1 or weight_array.shape != score_array.shape:
            raise ValueError("frequency_weights must have the same shape as scores")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("frequency_weights must be finite and non-negative")

    keep = (~_missing_mask(feature_array)) & (weight_array > 0.0)
    return score_array[keep], feature_array[keep], weight_array[keep]


def _factorize(values: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    codes = np.empty(values.shape[0], dtype=np.int64)
    levels: List[Any] = []
    lookup = {}
    for index, raw_value in enumerate(values.tolist()):
        try:
            key = (type(raw_value), raw_value)
            code = lookup.get(key)
        except TypeError as exc:
            raise ValueError("nominal feature levels must be hashable") from exc
        if code is None:
            code = len(levels)
            lookup[key] = code
            levels.append(raw_value)
        codes[index] = code
    return codes, levels


def nominal_quadratic_test(
    scores: ArrayLike,
    feature: Sequence[Any],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    epsilon: float = 1e-7,
    missing_policy: str = "production_bin",
) -> TestResult:
    """Reproduce CTBoost's current nominal ``k - 1`` quadratic feature test.

    The covariance and variance conventions intentionally mirror
    ``src/core/statistics.cpp``. ``missing_policy="production_bin"`` mirrors
    CTBoost quantization by treating missing values as a nominal level;
    ``"ctree_omit"`` follows Hothorn et al. (2006) by giving those rows zero
    weight for this feature test.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if missing_policy not in {"production_bin", "ctree_omit"}:
        raise ValueError("missing_policy must be 'production_bin' or 'ctree_omit'")

    score_array = np.asarray(scores, dtype=np.float64)
    feature_array = np.asarray(feature, dtype=object)
    if score_array.ndim != 1 or feature_array.ndim != 1:
        raise ValueError("scores and feature must be one-dimensional")
    if score_array.shape[0] != feature_array.shape[0]:
        raise ValueError("scores and feature must have the same length")
    if frequency_weights is None:
        weights = np.ones(score_array.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(frequency_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape != score_array.shape:
            raise ValueError("frequency_weights must have the same shape as scores")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("frequency_weights must be finite and non-negative")
    missing = _missing_mask(feature_array)
    keep = weights > 0.0
    if missing_policy == "ctree_omit":
        keep &= ~missing
        feature_array = feature_array[keep]
    else:
        feature_array = feature_array.copy()
        missing_level = object()
        feature_array[missing] = missing_level
        feature_array = feature_array[keep]
    score_array = score_array[keep]
    weights = weights[keep]
    n_observations = int(score_array.shape[0])
    weight_sum = float(np.sum(weights))
    if n_observations == 0:
        return TestResult(
            "nominal_quadratic",
            0.0,
            1.0,
            0,
            0,
            0.0,
            {"active_levels": 0, "missing_policy": missing_policy},
        )

    codes, levels = _factorize(feature_array)
    level_count = len(levels)
    if weight_sum <= 1.0 or level_count <= 1:
        return TestResult(
            "nominal_quadratic",
            0.0,
            1.0,
            0,
            n_observations,
            weight_sum,
            {"active_levels": level_count, "missing_policy": missing_policy},
        )

    bin_weights = np.bincount(codes, weights=weights, minlength=level_count)
    bin_sums = np.bincount(codes, weights=weights * score_array, minlength=level_count)
    total_score = float(np.dot(weights, score_array))
    mean = total_score / weight_sum
    variance = float(np.dot(weights, np.square(score_array - mean)) / weight_sum)
    degrees_of_freedom = level_count - 1
    if variance <= np.finfo(np.float64).eps:
        return TestResult(
            "nominal_quadratic",
            0.0,
            1.0,
            degrees_of_freedom,
            n_observations,
            weight_sum,
            {"active_levels": level_count, "missing_policy": missing_policy},
        )

    reduced_weights = bin_weights[:degrees_of_freedom]
    difference = bin_sums[:degrees_of_freedom] - reduced_weights * mean
    diagonal_scale = weight_sum / (weight_sum - 1.0) * variance
    outer_scale = variance / (weight_sum - 1.0)
    covariance = -outer_scale * np.outer(reduced_weights, reduced_weights)
    covariance[np.diag_indices_from(covariance)] += (
        diagonal_scale * reduced_weights + epsilon
    )
    try:
        solved = np.linalg.solve(covariance, difference)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance, hermitian=True) @ difference
    statistic = max(0.0, float(np.dot(difference, solved)))
    return TestResult(
        "nominal_quadratic",
        statistic,
        chi_square_survival(statistic, degrees_of_freedom),
        degrees_of_freedom,
        n_observations,
        weight_sum,
        {"active_levels": level_count, "missing_policy": missing_policy},
    )


def weighted_midranks(
    values: Sequence[float], frequency_weights: Optional[ArrayLike] = None
) -> np.ndarray:
    """Return normalized weighted midranks, with equal values sharing a score."""

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim != 1 or not np.all(np.isfinite(value_array)):
        raise ValueError("values must be a finite one-dimensional array")
    if frequency_weights is None:
        weights = np.ones(value_array.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(frequency_weights, dtype=np.float64)
        if weights.shape != value_array.shape:
            raise ValueError("frequency_weights must have the same shape as values")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("frequency_weights must be finite and positive")
    if value_array.size == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(value_array, kind="stable")
    sorted_values = value_array[order]
    sorted_weights = weights[order]
    ranks_sorted = np.empty(value_array.shape[0], dtype=np.float64)
    total_weight = float(np.sum(weights))
    cumulative_weight = 0.0
    start = 0
    while start < sorted_values.shape[0]:
        stop = start + 1
        while (
            stop < sorted_values.shape[0]
            and sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        tie_weight = float(np.sum(sorted_weights[start:stop]))
        ranks_sorted[start:stop] = (cumulative_weight + 0.5 * tie_weight) / total_weight
        cumulative_weight += tie_weight
        start = stop
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def ordered_linear_test(
    scores: ArrayLike,
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    missing_policy: str = "ctree_omit",
) -> TestResult:
    """One-df linear statistic using ordered weighted-midrank feature scores.

    The reference candidate intentionally uses feature-wise CTree omission for
    missing ordered values. Other policies must be researched explicitly.
    """

    if missing_policy != "ctree_omit":
        raise ValueError(
            "ordered midrank currently supports only missing_policy='ctree_omit'"
        )

    score_array, feature_array, weights = _validate_inputs(
        scores, ordered_feature, frequency_weights
    )
    try:
        numeric_feature = feature_array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordered_feature must be numeric") from exc
    if not np.all(np.isfinite(numeric_feature)):
        raise ValueError("ordered_feature must be finite after omitting missing values")

    n_observations = int(score_array.shape[0])
    weight_sum = float(np.sum(weights))
    if n_observations <= 1 or weight_sum <= 1.0:
        return TestResult(
            "ordered_midrank", 0.0, 1.0, 0, n_observations, weight_sum, {}
        )

    ranks = weighted_midranks(numeric_feature, weights)
    rank_mean = float(np.dot(weights, ranks) / weight_sum)
    score_mean = float(np.dot(weights, score_array) / weight_sum)
    centered_rank = ranks - rank_mean
    centered_score = score_array - score_mean
    cross_sum = float(np.dot(weights, centered_rank * centered_score))
    rank_sum_squares = float(np.dot(weights, np.square(centered_rank)))
    score_sum_squares = float(np.dot(weights, np.square(centered_score)))
    permutation_variance = rank_sum_squares * score_sum_squares / (weight_sum - 1.0)
    if permutation_variance <= np.finfo(np.float64).eps:
        return TestResult(
            "ordered_midrank", 0.0, 1.0, 1, n_observations, weight_sum, {}
        )

    statistic = cross_sum * cross_sum / permutation_variance
    return TestResult(
        "ordered_midrank",
        statistic,
        chi_square_survival(statistic, 1),
        1,
        n_observations,
        weight_sum,
        {},
    )


def equal_weight_ordered_groups(
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_groups: int = 8,
) -> np.ndarray:
    """Collapse ordered levels into contiguous, approximately equal-weight groups.

    Existing quantization levels are kept atomic. Only the feature-selection
    statistic sees these groups; a future learner prototype would continue to
    search the eventual gain split over every original histogram bin.
    """

    values = np.asarray(ordered_feature, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("ordered_feature must be finite and one-dimensional")
    if n_groups < 2:
        raise ValueError("n_groups must be at least two")
    if frequency_weights is None:
        weights = np.ones(values.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(frequency_weights, dtype=np.float64)
        if weights.shape != values.shape:
            raise ValueError(
                "frequency_weights must have the same shape as ordered_feature"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("frequency_weights must be finite and positive")
    if values.size == 0:
        return np.empty(0, dtype=np.int64)

    levels, inverse = np.unique(values, return_inverse=True)
    level_weights = np.bincount(inverse, weights=weights, minlength=levels.shape[0])
    cumulative_before = np.cumsum(level_weights) - level_weights
    midpoints = cumulative_before + 0.5 * level_weights
    level_groups = np.floor(n_groups * midpoints / np.sum(level_weights)).astype(
        np.int64
    )
    level_groups = np.clip(level_groups, 0, n_groups - 1)
    # Remove gaps caused by a single very heavy atomic level.
    _, compact = np.unique(level_groups, return_inverse=True)
    return compact[inverse]


def grouped_ordered_quadratic_test(
    scores: ArrayLike,
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_groups: int = 8,
    missing_policy: str = "production_bin",
    epsilon: float = 1e-7,
) -> TestResult:
    """Quadratic feature test after selection-only ordered-bin compression."""

    if missing_policy not in {"production_bin", "ctree_omit"}:
        raise ValueError("missing_policy must be 'production_bin' or 'ctree_omit'")
    score_array = np.asarray(scores, dtype=np.float64)
    feature_array = np.asarray(ordered_feature)
    if (
        score_array.ndim != 1
        or feature_array.ndim != 1
        or score_array.shape != feature_array.shape
    ):
        raise ValueError(
            "scores and ordered_feature must be one-dimensional with equal length"
        )
    if frequency_weights is None:
        weights = np.ones(score_array.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(frequency_weights, dtype=np.float64)
        if weights.shape != score_array.shape:
            raise ValueError("frequency_weights must have the same shape as scores")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("frequency_weights must be finite and non-negative")
    missing = _missing_mask(feature_array)
    keep = weights > 0.0
    complete = keep & ~missing
    try:
        numeric_complete = feature_array[complete].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordered_feature must be numeric") from exc
    if not np.all(np.isfinite(numeric_complete)):
        raise ValueError(
            "ordered_feature must be finite after identifying missing values"
        )
    complete_groups = equal_weight_ordered_groups(
        numeric_complete, weights[complete], n_groups=n_groups
    )
    nonmissing_group_count = int(np.unique(complete_groups).shape[0])
    if missing_policy == "ctree_omit":
        score_array = score_array[complete]
        weights = weights[complete]
        groups = complete_groups
    else:
        score_array = score_array[keep]
        weights = weights[keep]
        kept_missing = missing[keep]
        groups = np.empty(score_array.shape[0], dtype=np.int64)
        groups[~kept_missing] = complete_groups
        groups[kept_missing] = nonmissing_group_count
    base = nominal_quadratic_test(
        score_array,
        groups,
        weights,
        epsilon=epsilon,
        missing_policy="production_bin",
    )
    details = dict(base.details)
    details.update(
        {
            "requested_groups": n_groups,
            "effective_groups": int(np.unique(groups).shape[0]),
            "nonmissing_groups": nonmissing_group_count,
            "source_levels": int(np.unique(numeric_complete).shape[0]),
            "missing_level_present": bool(np.any(missing & keep)),
            "missing_policy": missing_policy,
            "selection_only_grouping": True,
        }
    )
    return TestResult(
        "grouped_ordered_quadratic",
        base.statistic,
        base.p_value,
        base.degrees_of_freedom,
        base.n_observations,
        base.weight_sum,
        details,
    )


def ordered_grouped_hybrid_test(
    scores: ArrayLike,
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_groups: int = 8,
    missing_policy: str = "production_bin",
) -> TestResult:
    """Bonferroni-safe min-p hybrid of linear and grouped quadratic tests."""

    if missing_policy not in {"production_bin", "ctree_omit"}:
        raise ValueError("missing_policy must be 'production_bin' or 'ctree_omit'")
    linear = ordered_linear_test(
        scores,
        ordered_feature,
        frequency_weights,
        missing_policy="ctree_omit",
    )
    grouped = grouped_ordered_quadratic_test(
        scores,
        ordered_feature,
        frequency_weights,
        n_groups=n_groups,
        missing_policy=missing_policy,
    )
    minimum_p = min(linear.p_value, grouped.p_value)
    adjusted_p = min(1.0, 2.0 * minimum_p)
    return TestResult(
        "ordered_grouped_hybrid",
        max(linear.statistic, grouped.statistic),
        adjusted_p,
        0,
        grouped.n_observations,
        grouped.weight_sum,
        {
            "ordered_p_value": linear.p_value,
            "grouped_p_value": grouped.p_value,
            "within_feature_adjustment": "bonferroni_2",
            "requested_groups": n_groups,
            "ordered_missing_policy": "ctree_omit",
            "grouped_missing_policy": missing_policy,
        },
    )


def _integer_frequency_weights(weights: np.ndarray, max_expanded_n: int) -> np.ndarray:
    rounded = np.rint(weights)
    if not np.allclose(weights, rounded, rtol=0.0, atol=1e-10):
        raise ValueError(
            "permutation maxstat requires integer frequency weights; "
            "non-integer case weights do not define exchangeable replicas"
        )
    counts = rounded.astype(np.int64)
    expanded_n = int(np.sum(counts))
    if expanded_n > max_expanded_n:
        raise ValueError(
            f"expanded frequency-weight sample exceeds max_expanded_n={max_expanded_n}"
        )
    return counts


def _maxstat_values(
    sorted_centered_scores: np.ndarray,
    cut_positions: np.ndarray,
    total_sum_squares: float,
) -> np.ndarray:
    n_observations = sorted_centered_scores.shape[-1]
    cumulative = np.cumsum(sorted_centered_scores, axis=-1)
    left_sums = np.take(cumulative, cut_positions - 1, axis=-1)
    variances = (
        cut_positions
        * (n_observations - cut_positions)
        / (n_observations * (n_observations - 1.0))
        * total_sum_squares
    )
    standardized_squared = np.square(left_sums) / variances
    return np.max(standardized_squared, axis=-1)


def permutation_maxstat_test(
    scores: ArrayLike,
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_permutations: int = 199,
    min_fraction: float = 0.1,
    min_samples_leaf: int = 1,
    random_state: Union[int, np.random.Generator] = 0,
    max_expanded_n: int = 100000,
    permutation_batch_size: int = 256,
    missing_policy: str = "ctree_omit",
) -> TestResult:
    """Permutation-calibrated maximally selected statistic.

    Candidate cutpoints are trimmed by both ``min_fraction`` and
    ``min_samples_leaf``. The Monte Carlo p-value is
    ``(1 + count(T_perm >= T_obs)) / (B + 1)``; consequently it is never zero.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    if missing_policy != "ctree_omit":
        raise ValueError(
            "permutation maxstat supports only missing_policy='ctree_omit'"
        )
    if not 0.0 < min_fraction < 0.5:
        raise ValueError("min_fraction must be between 0 and 0.5")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be positive")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be positive")

    score_array, feature_array, weights = _validate_inputs(
        scores, ordered_feature, frequency_weights
    )
    try:
        numeric_feature = feature_array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordered_feature must be numeric") from exc
    if not np.all(np.isfinite(numeric_feature)):
        raise ValueError("ordered_feature must be finite after omitting missing values")

    counts = _integer_frequency_weights(weights, max_expanded_n)
    expanded_scores = np.repeat(score_array, counts)
    expanded_feature = np.repeat(numeric_feature, counts)
    n_observations = int(expanded_scores.shape[0])
    if n_observations <= 1:
        return TestResult(
            "permutation_maxstat",
            0.0,
            1.0,
            0,
            n_observations,
            float(n_observations),
            {},
        )

    order = np.argsort(expanded_feature, kind="stable")
    sorted_feature = expanded_feature[order]
    centered_scores = expanded_scores - float(np.mean(expanded_scores))
    sorted_centered_scores = centered_scores[order]
    minimum_count = max(min_samples_leaf, math.ceil(min_fraction * n_observations))
    all_positions = np.arange(1, n_observations, dtype=np.int64)
    eligible = (
        (all_positions >= minimum_count)
        & ((n_observations - all_positions) >= minimum_count)
        & (sorted_feature[:-1] < sorted_feature[1:])
    )
    cut_positions = all_positions[eligible]
    if cut_positions.size == 0:
        return TestResult(
            "permutation_maxstat",
            0.0,
            1.0,
            0,
            n_observations,
            float(n_observations),
            {"eligible_cutpoints": 0, "minimum_count": minimum_count},
        )

    total_sum_squares = float(np.dot(centered_scores, centered_scores))
    if total_sum_squares <= np.finfo(np.float64).eps:
        return TestResult(
            "permutation_maxstat",
            0.0,
            1.0,
            1,
            n_observations,
            float(n_observations),
            {
                "eligible_cutpoints": int(cut_positions.size),
                "minimum_count": minimum_count,
            },
        )

    observed_values = np.square(
        np.cumsum(sorted_centered_scores)[cut_positions - 1]
    ) / (
        cut_positions
        * (n_observations - cut_positions)
        / (n_observations * (n_observations - 1.0))
        * total_sum_squares
    )
    best_index = int(np.argmax(observed_values))
    observed = float(observed_values[best_index])
    best_position = int(cut_positions[best_index])
    threshold = float(
        sorted_feature[best_position - 1]
        + 0.5 * (sorted_feature[best_position] - sorted_feature[best_position - 1])
    )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    exceedances = 0
    completed = 0
    while completed < n_permutations:
        batch_size = min(permutation_batch_size, n_permutations - completed)
        permutation_keys = rng.random((batch_size, n_observations))
        permutation_indices = np.argsort(permutation_keys, axis=1)
        permuted_scores = centered_scores[permutation_indices]
        maxima = _maxstat_values(permuted_scores, cut_positions, total_sum_squares)
        exceedances += int(np.count_nonzero(maxima >= observed - 1e-14))
        completed += batch_size
    p_value = (1.0 + exceedances) / (n_permutations + 1.0)
    return TestResult(
        "permutation_maxstat",
        observed,
        p_value,
        1,
        n_observations,
        float(n_observations),
        {
            "eligible_cutpoints": int(cut_positions.size),
            "minimum_count": minimum_count,
            "best_cut_position": best_position,
            "best_threshold": threshold,
            "n_permutations": n_permutations,
            "exceedances": exceedances,
        },
    )


def global_bonferroni_stop(
    p_values: Iterable[float], *, alpha: float = 0.05, n_tests: Optional[int] = None
) -> BonferroniDecision:
    """Select the smallest p-value and apply a global Bonferroni stop test."""

    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must contain at least one value")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be within [0, 1]")
    values = np.where(np.isfinite(values), values, 1.0)
    multiplicity = int(values.size if n_tests is None else n_tests)
    if multiplicity < values.size:
        raise ValueError("n_tests cannot be smaller than the number of p-values")
    selected_index = int(np.argmin(values))
    raw_p_value = float(values[selected_index])
    adjusted_p_value = min(1.0, multiplicity * raw_p_value)
    should_split = adjusted_p_value <= alpha
    return BonferroniDecision(
        selected_index if should_split else None,
        raw_p_value,
        adjusted_p_value,
        alpha,
        multiplicity,
        should_split,
    )


def equal_frequency_bins(values: Sequence[float], n_bins: int = 8) -> np.ndarray:
    """Deterministically bin a numeric feature for the nominal reference test."""

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    missing = ~np.isfinite(value_array)
    complete = value_array[~missing]
    output = np.full(value_array.shape, np.nan, dtype=np.float64)
    if complete.size == 0:
        return output
    probabilities = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    try:
        cutpoints = np.quantile(complete, probabilities, method="linear")
    except TypeError:  # NumPy < 1.22
        cutpoints = np.quantile(complete, probabilities, interpolation="linear")
    cutpoints = np.unique(cutpoints)
    output[~missing] = np.searchsorted(cutpoints, complete, side="right")
    return output
