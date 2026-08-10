"""Reference category orderings for binary objectives."""

# ruff: noqa: UP006, UP045
# CTBoost supports Python 3.8, where built-in generic and ``X | None`` syntax
# is not parseable even when annotations are postponed.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CategoryOrderingComparison:
    categories: Tuple[Any, ...]
    newton_order: Tuple[Any, ...]
    woe_order: Tuple[Any, ...]
    pairwise_agreement: float
    matched_smoothing: float


def _validate_binary_inputs(
    categories: Sequence[Any],
    target: Sequence[float],
    frequency_weights: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    category_array = np.asarray(categories, dtype=object)
    target_array = np.asarray(target, dtype=np.float64)
    if category_array.ndim != 1 or target_array.ndim != 1:
        raise ValueError("categories and target must be one-dimensional")
    if category_array.shape != target_array.shape:
        raise ValueError("categories and target must have the same shape")
    if not np.all(np.isfinite(target_array)) or np.any(
        (target_array < 0.0) | (target_array > 1.0)
    ):
        raise ValueError("target must contain finite binary probabilities in [0, 1]")
    if frequency_weights is None:
        weights = np.ones(target_array.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(frequency_weights, dtype=np.float64)
        if weights.shape != target_array.shape:
            raise ValueError("frequency_weights must have the same shape as target")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("frequency_weights must be finite and positive")
    return category_array, target_array, weights


def _category_totals(
    categories: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    levels: List[Any] = []
    lookup: Dict[Tuple[type, Any], int] = {}
    counts: List[float] = []
    successes: List[float] = []
    for category, outcome, weight in zip(categories.tolist(), target, weights):
        try:
            key = (type(category), category)
            index = lookup.get(key)
        except TypeError as exc:
            raise ValueError("categories must be hashable") from exc
        if index is None:
            index = len(levels)
            lookup[key] = index
            levels.append(category)
            counts.append(0.0)
            successes.append(0.0)
        counts[index] += float(weight)
        successes[index] += float(weight * outcome)
    return levels, np.asarray(counts), np.asarray(successes)


def _resolve_prior(
    target: np.ndarray, weights: np.ndarray, prior: Optional[float]
) -> float:
    resolved = (
        float(np.dot(target, weights) / np.sum(weights))
        if prior is None
        else float(prior)
    )
    if not 0.0 < resolved < 1.0:
        raise ValueError("prior must be strictly between zero and one")
    return resolved


def binary_category_newton_scores(
    categories: Sequence[Any],
    target: Sequence[float],
    frequency_weights: Optional[Sequence[float]] = None,
    *,
    prior: Optional[float] = None,
    l2: float = 1.0,
) -> Mapping[Any, float]:
    """Order categories by their regularized intercept-only Newton update."""

    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    category_array, target_array, weights = _validate_binary_inputs(
        categories, target, frequency_weights
    )
    resolved_prior = _resolve_prior(target_array, weights, prior)
    levels, counts, successes = _category_totals(category_array, target_array, weights)
    numerator = successes - counts * resolved_prior
    denominator = counts * resolved_prior * (1.0 - resolved_prior) + l2
    scores = numerator / denominator
    return {level: float(score) for level, score in zip(levels, scores)}


def smoothed_woe_scores(
    categories: Sequence[Any],
    target: Sequence[float],
    frequency_weights: Optional[Sequence[float]] = None,
    *,
    prior: Optional[float] = None,
    smoothing: float = 1.0,
) -> Mapping[Any, float]:
    """Return prior-centered, smoothed weight-of-evidence category scores."""

    if smoothing < 0.0:
        raise ValueError("smoothing must be non-negative")
    category_array, target_array, weights = _validate_binary_inputs(
        categories, target, frequency_weights
    )
    resolved_prior = _resolve_prior(target_array, weights, prior)
    levels, counts, successes = _category_totals(category_array, target_array, weights)
    probabilities = (successes + smoothing * resolved_prior) / (counts + smoothing)
    epsilon = np.finfo(np.float64).eps
    probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
    prior_logit = math.log(resolved_prior / (1.0 - resolved_prior))
    scores = np.log(probabilities / (1.0 - probabilities)) - prior_logit
    return {level: float(score) for level, score in zip(levels, scores)}


def compare_newton_and_woe_ordering(
    categories: Sequence[Any],
    target: Sequence[float],
    frequency_weights: Optional[Sequence[float]] = None,
    *,
    prior: Optional[float] = None,
    l2: float = 1.0,
) -> CategoryOrderingComparison:
    """Compare Newton and algebraically matched smoothed-WoE orderings.

    For a common binary-logloss prior ``p``, setting WoE smoothing to
    ``l2 / (p * (1 - p))`` makes both scores monotone transforms of the same
    smoothed category probability. Their rankings should therefore agree,
    including under unequal category frequencies.
    """

    category_array, target_array, weights = _validate_binary_inputs(
        categories, target, frequency_weights
    )
    resolved_prior = _resolve_prior(target_array, weights, prior)
    smoothing = l2 / (resolved_prior * (1.0 - resolved_prior))
    newton = binary_category_newton_scores(
        category_array, target_array, weights, prior=resolved_prior, l2=l2
    )
    woe = smoothed_woe_scores(
        category_array,
        target_array,
        weights,
        prior=resolved_prior,
        smoothing=smoothing,
    )
    levels = tuple(newton)
    newton_order = tuple(sorted(levels, key=lambda item: (newton[item], repr(item))))
    woe_order = tuple(sorted(levels, key=lambda item: (woe[item], repr(item))))
    comparable = 0
    agreements = 0
    for left in range(len(levels)):
        for right in range(left + 1, len(levels)):
            first = levels[left]
            second = levels[right]
            newton_sign = np.sign(newton[first] - newton[second])
            woe_sign = np.sign(woe[first] - woe[second])
            if newton_sign == 0.0 and woe_sign == 0.0:
                continue
            comparable += 1
            agreements += int(newton_sign == woe_sign)
    agreement = 1.0 if comparable == 0 else agreements / comparable
    return CategoryOrderingComparison(
        levels, newton_order, woe_order, agreement, smoothing
    )


def cross_fitted_binary_category_scores(
    categories: Sequence[Any],
    target: Sequence[float],
    frequency_weights: Optional[Sequence[float]] = None,
    *,
    n_splits: int = 5,
    random_state: int = 0,
    method: str = "woe",
    smoothing: float = 10.0,
    l2: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold category scores without using a row's own target.

    Unknown categories receive the training fold's prior-centered score zero.
    The helper is a leakage-safety reference, not a production encoder.
    """

    category_array, target_array, weights = _validate_binary_inputs(
        categories, target, frequency_weights
    )
    if n_splits < 2 or n_splits > target_array.shape[0]:
        raise ValueError("n_splits must be between two and the sample count")
    if method not in {"woe", "newton"}:
        raise ValueError("method must be 'woe' or 'newton'")

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(target_array.shape[0])
    fold_ids = np.empty(target_array.shape[0], dtype=np.int64)
    fold_ids[shuffled] = np.arange(target_array.shape[0]) % n_splits
    encoded = np.zeros(target_array.shape[0], dtype=np.float64)
    for fold in range(n_splits):
        validation = fold_ids == fold
        training = ~validation
        prior = _resolve_prior(target_array[training], weights[training], None)
        if method == "woe":
            table = smoothed_woe_scores(
                category_array[training],
                target_array[training],
                weights[training],
                prior=prior,
                smoothing=smoothing,
            )
        else:
            table = binary_category_newton_scores(
                category_array[training],
                target_array[training],
                weights[training],
                prior=prior,
                l2=l2,
            )
        encoded[validation] = [
            table.get(value, 0.0) for value in category_array[validation]
        ]
    return encoded, fold_ids
