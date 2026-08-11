"""Reference-only multiclass score statistics and numeric cut gains.

The functions in this module are executable specifications for research.  They
are deliberately isolated from :mod:`ctboost` and do not alter the native
conditional-inference tree, public API, serialization, CPU, or CUDA paths.

Integer frequency weights are interpreted as literal replicated observations.
Non-integer non-negative weights are accepted to mirror CTBoost's working
asymptotic convention, but do not define an exact conditional permutation
test.  The full-score test follows the conditional covariance construction of
Strasser and Weber and the quadratic statistic used by Hothorn et al.  The
numeric gain is a separate, post-selection diagnostic.
"""

# ruff: noqa: UP006, UP007, UP045
# CTBoost supports Python 3.8, where built-in generic, union, and optional
# shorthand syntax is not parseable even when annotations are postponed.

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .statistics import (
    TestResult,
    chi_square_survival,
    equal_weight_ordered_groups,
    nominal_quadratic_test,
)

ArrayLike = Union[Sequence[float], np.ndarray]


# The full gain is a sum over score coordinates.  Column permutations can
# change only floating-point summation noise, so comparisons within this
# relative envelope are ties rather than evidence for a different cut.
NUMERIC_GAIN_TIE_RELATIVE_TOLERANCE = 64.0 * np.finfo(np.float64).eps


class _MissingLevel:
    pass


_MISSING_LEVEL = _MissingLevel()


@dataclass(frozen=True)
class MulticlassTestResult:
    """Result of one legacy or full-score multiclass feature test."""

    method: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    response_rank: int
    active_bins: int
    n_observations: int
    weight_sum: float
    structure_class: Optional[int]
    details: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PermutationCalibrationResult:
    """Inclusive-plus-one permutation calibration of a multiclass statistic."""

    method: str
    statistic: float
    asymptotic_p_value: float
    permutation_p_value: float
    exceedances: int
    n_permutations: int
    degrees_of_freedom: int
    response_rank: int

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NumericSplitResult:
    """Best ordered cut found after a feature has already been selected."""

    method: str
    gain: float
    threshold: Optional[float]
    missing_go_left: bool
    left_weight: float
    right_weight: float
    evaluated_cutpoints: int
    structure_class: Optional[int]

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


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


def _as_score_matrix(scores: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("scores must have shape (n_observations, n_score_dimensions)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scores must be finite")
    return matrix


def _as_weights(n_observations: int, frequency_weights: Optional[ArrayLike]) -> np.ndarray:
    if frequency_weights is None:
        weights = np.ones(n_observations, dtype=np.float64)
    else:
        weights = np.asarray(frequency_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != n_observations:
            raise ValueError("frequency_weights must have one entry per observation")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("frequency_weights must be finite and non-negative")
    return weights


def _factorize(values: np.ndarray) -> Tuple[np.ndarray, int]:
    levels = []
    lookup = {}
    for raw_value in values.tolist():
        try:
            key = (type(raw_value), raw_value)
            present = key in lookup
        except TypeError as exc:
            raise ValueError("feature levels must be hashable") from exc
        if not present:
            lookup[key] = len(levels)
            levels.append(raw_value)

    def level_sort_key(value: Any) -> Tuple[int, Any, str]:
        if value is _MISSING_LEVEL:
            return (2, 0.0, "")
        if isinstance(value, (int, float, np.integer, np.floating)):
            return (0, float(value), "")
        return (
            1,
            f"{type(value).__module__}.{type(value).__qualname__}",
            repr(value),
        )

    ordered_levels = sorted(levels, key=level_sort_key)
    ordered_lookup = {
        (type(raw_value), raw_value): code
        for code, raw_value in enumerate(ordered_levels)
    }
    codes = np.asarray(
        [ordered_lookup[(type(raw_value), raw_value)] for raw_value in values.tolist()],
        dtype=np.int64,
    )
    return codes, len(levels)


def _weighted_score_inverse(
    score_matrix: np.ndarray,
    weights: np.ndarray,
    eigen_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    weight_sum = float(np.sum(weights))
    mean_score = np.sum(weights[:, None] * score_matrix, axis=0) / weight_sum
    centered = score_matrix - mean_score
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("score covariance produced non-finite eigenvalues")
    largest = float(eigenvalues[-1])
    if largest <= 0.0:
        retained = np.zeros(eigenvalues.shape[0], dtype=bool)
        threshold = 0.0
    else:
        threshold = eigen_tolerance * largest
        if float(eigenvalues[0]) < -threshold:
            raise FloatingPointError(
                "score covariance is not positive semidefinite within tolerance"
            )
        retained = eigenvalues > threshold
    response_rank = int(np.count_nonzero(retained))
    if response_rank == 0:
        inverse_covariance = np.zeros_like(covariance)
    else:
        retained_vectors = eigenvectors[:, retained]
        inverse_covariance = (
            retained_vectors / eigenvalues[retained][None, :]
        ) @ retained_vectors.T
    return mean_score, inverse_covariance, eigenvalues, response_rank, threshold


def _prepare_feature_test_inputs(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    feature: Sequence[Any],
    frequency_weights: Optional[ArrayLike],
    missing_policy: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if missing_policy not in {"production_bin", "ctree_omit"}:
        raise ValueError("missing_policy must be 'production_bin' or 'ctree_omit'")
    score_matrix = _as_score_matrix(scores)
    feature_array = np.asarray(feature, dtype=object)
    if feature_array.ndim != 1 or feature_array.shape[0] != score_matrix.shape[0]:
        raise ValueError("feature must have one entry per score row")
    weights = _as_weights(score_matrix.shape[0], frequency_weights)
    missing = _missing_mask(feature_array)
    keep = weights > 0.0
    if missing_policy == "ctree_omit":
        keep &= ~missing
        prepared_feature = feature_array[keep]
    else:
        prepared_feature = feature_array.copy()
        prepared_feature[missing] = _MISSING_LEVEL
        prepared_feature = prepared_feature[keep]
    return score_matrix[keep], prepared_feature, weights[keep]


def softmax_score_matrix(
    labels: Sequence[int], probabilities: Union[Sequence[float], np.ndarray]
) -> np.ndarray:
    """Return softmax objective scores ``p - one_hot(label)`` in float64."""

    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim == 1:
        probability_array = np.broadcast_to(
            probability_array, (label_array.shape[0], probability_array.shape[0])
        ).copy()
    if (
        probability_array.ndim != 2
        or probability_array.shape[0] != label_array.shape[0]
        or probability_array.shape[1] < 2
    ):
        raise ValueError("probabilities must have shape (K,) or (n_observations, K)")
    if (
        not np.all(np.isfinite(probability_array))
        or np.any(probability_array < 0.0)
        or not np.allclose(np.sum(probability_array, axis=1), 1.0, atol=1e-12)
    ):
        raise ValueError("probability rows must be finite, non-negative, and sum to one")
    n_classes = probability_array.shape[1]
    if np.any(label_array < 0) or np.any(label_array >= n_classes):
        raise ValueError("labels must be valid probability-column indices")
    scores = probability_array.copy()
    scores[np.arange(label_array.shape[0]), label_array] -= 1.0
    return scores


def softmax_diagonal_hessians(
    probabilities: Union[Sequence[float], np.ndarray], n_observations: Optional[int] = None
) -> np.ndarray:
    """Return CTBoost's diagonal softmax Hessian approximation ``p * (1-p)``."""

    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim == 1:
        if n_observations is None:
            raise ValueError("n_observations is required for one-dimensional probabilities")
        probability_array = np.broadcast_to(
            probability_array, (n_observations, probability_array.shape[0])
        ).copy()
    if probability_array.ndim != 2:
        raise ValueError("probabilities must be one- or two-dimensional")
    if (
        not np.all(np.isfinite(probability_array))
        or np.any(probability_array < 0.0)
        or np.any(probability_array > 1.0)
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return probability_array * (1.0 - probability_array)


def select_legacy_structure_class(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    frequency_weights: Optional[ArrayLike] = None,
) -> Tuple[int, np.ndarray]:
    """Reproduce CTBoost's global highest-variance class-coordinate choice.

    ``numpy.argmax`` returns the first maximum, matching the native strict-``>``
    update and therefore exposing the legacy label-order tie behavior.
    """

    score_matrix = _as_score_matrix(scores)
    weights = _as_weights(score_matrix.shape[0], frequency_weights)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        variances = np.zeros(score_matrix.shape[1], dtype=np.float64)
    else:
        # Preserve the native row-major accumulation order instead of relying
        # on a platform-dependent vectorized reduction.
        sums = np.zeros(score_matrix.shape[1], dtype=np.float64)
        square_sums = np.zeros(score_matrix.shape[1], dtype=np.float64)
        for row in range(score_matrix.shape[0]):
            sample_weight = float(weights[row])
            for dimension in range(score_matrix.shape[1]):
                score = float(score_matrix[row, dimension])
                sums[dimension] += sample_weight * score
                square_sums[dimension] += sample_weight * score * score
        means = sums / weight_sum
        variances = np.maximum(0.0, square_sums / weight_sum - np.square(means))
    return int(np.argmax(variances)), variances


def _legacy_scalar_quadratic_test(
    scores: np.ndarray,
    feature: Sequence[Any],
    weights: np.ndarray,
    *,
    missing_policy: str,
    epsilon: float,
) -> TestResult:
    selected_scores, prepared_feature, test_weights = _prepare_feature_test_inputs(
        scores[:, None], feature, weights, missing_policy
    )
    scalar_scores = selected_scores[:, 0]
    n_observations = int(scalar_scores.shape[0])
    weight_sum = float(np.sum(test_weights))
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

    codes, active_bins = _factorize(prepared_feature)
    if weight_sum <= 1.0 or active_bins <= 1:
        return TestResult(
            "nominal_quadratic",
            0.0,
            1.0,
            0,
            n_observations,
            weight_sum,
            {"active_levels": active_bins, "missing_policy": missing_policy},
        )

    bin_weights = np.bincount(codes, weights=test_weights, minlength=active_bins)
    bin_sums = np.bincount(
        codes,
        weights=test_weights * scalar_scores,
        minlength=active_bins,
    )
    total_score = float(np.dot(test_weights, scalar_scores))
    mean_score = total_score / weight_sum
    variance = max(
        0.0,
        float(np.dot(test_weights, np.square(scalar_scores)) / weight_sum)
        - mean_score * mean_score,
    )
    degrees_of_freedom = active_bins - 1
    if variance <= np.finfo(np.float64).eps:
        return TestResult(
            "nominal_quadratic",
            0.0,
            1.0,
            degrees_of_freedom,
            n_observations,
            weight_sum,
            {"active_levels": active_bins, "missing_policy": missing_policy},
        )

    # This is the O(B) Sherman-Morrison evaluation used by
    # src/core/statistics.cpp, including its diagonal epsilon and
    # deterministic omission of the final active bin.
    reduced_weights = bin_weights[:degrees_of_freedom]
    differences = bin_sums[:degrees_of_freedom] - reduced_weights * mean_score
    diagonal_scale = weight_sum / (weight_sum - 1.0) * variance
    outer_scale = variance / (weight_sum - 1.0)
    diagonal = diagonal_scale * reduced_weights + epsilon
    difference_quadratic = float(np.sum(np.square(differences) / diagonal))
    weighted_projection = float(
        np.sum(reduced_weights * differences / diagonal)
    )
    diagonal_projection = float(
        np.sum(np.square(reduced_weights) / diagonal)
    )
    denominator = 1.0 - outer_scale * diagonal_projection
    if denominator <= epsilon:
        # Retain the existing auditable dense reference only for the same
        # ill-conditioned fallback condition as native.
        return nominal_quadratic_test(
            scores,
            feature,
            weights,
            epsilon=epsilon,
            missing_policy=missing_policy,
        )

    statistic = max(
        0.0,
        difference_quadratic
        + outer_scale * weighted_projection * weighted_projection / denominator,
    )
    return TestResult(
        "nominal_quadratic",
        statistic,
        chi_square_survival(statistic, degrees_of_freedom),
        degrees_of_freedom,
        n_observations,
        weight_sum,
        {"active_levels": active_bins, "missing_policy": missing_policy},
    )


def legacy_multiclass_quadratic_test(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    feature: Sequence[Any],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    structure_class: Optional[int] = None,
    missing_policy: str = "production_bin",
    epsilon: float = 1e-7,
) -> MulticlassTestResult:
    """Apply the native-like scalar test to the legacy structure coordinate.

    When ``structure_class`` is omitted, the choice is made over every supplied
    row before feature-wise missing handling, as it is at the root of the
    current learner.  Callers modelling deeper nodes may pass the iteration's
    already selected coordinate explicitly.
    """

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    score_matrix = _as_score_matrix(scores)
    weights = _as_weights(score_matrix.shape[0], frequency_weights)
    if structure_class is None:
        selected, variances = select_legacy_structure_class(score_matrix, weights)
    else:
        selected = int(structure_class)
        if selected < 0 or selected >= score_matrix.shape[1]:
            raise ValueError("structure_class is out of range")
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            variances = np.zeros(score_matrix.shape[1], dtype=np.float64)
        else:
            means = np.sum(weights[:, None] * score_matrix, axis=0) / weight_sum
            variances = np.maximum(
                0.0,
                np.sum(weights[:, None] * np.square(score_matrix), axis=0)
                / weight_sum
                - np.square(means),
            )
    scalar = _legacy_scalar_quadratic_test(
        score_matrix[:, selected],
        feature,
        weights,
        missing_policy=missing_policy,
        epsilon=epsilon,
    )
    return MulticlassTestResult(
        method="legacy_highest_variance_class",
        statistic=scalar.statistic,
        p_value=scalar.p_value,
        degrees_of_freedom=scalar.degrees_of_freedom,
        response_rank=int(variances[selected] > np.finfo(np.float64).eps),
        active_bins=int(scalar.details.get("active_levels", 0)),
        n_observations=scalar.n_observations,
        weight_sum=scalar.weight_sum,
        structure_class=selected,
        details={
            "class_variances": variances.tolist(),
            "missing_policy": missing_policy,
            "scalar_epsilon": epsilon,
            "scalar_evaluation": "native_o_b_sherman_morrison",
        },
    )


def full_multiclass_quadratic_test(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    feature: Sequence[Any],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    missing_policy: str = "production_bin",
    eigen_tolerance: float = float(np.sqrt(np.finfo(np.float64).eps)),
) -> MulticlassTestResult:
    """Test all independent score directions with a quadratic linear statistic.

    For one-hot feature bins the general Moore-Penrose quadratic form simplifies
    to ``(W-1)/W * sum_b D_b' V+ D_b / n_b``.  Keeping all score columns avoids
    a label-dependent reference class; the pseudoinverse removes the softmax
    sum-to-zero direction and any additional rank deficiency.
    """

    if not np.isfinite(eigen_tolerance) or eigen_tolerance <= 0.0:
        raise ValueError("eigen_tolerance must be finite and positive")
    score_matrix, prepared_feature, weights = _prepare_feature_test_inputs(
        scores, feature, frequency_weights, missing_policy
    )
    n_observations = int(score_matrix.shape[0])
    weight_sum = float(np.sum(weights))
    if n_observations == 0:
        return MulticlassTestResult(
            "full_k_minus_one_quadratic",
            0.0,
            1.0,
            0,
            0,
            0,
            0,
            0.0,
            None,
            {"missing_policy": missing_policy, "eigenvalues": []},
        )

    codes, active_bins = _factorize(prepared_feature)
    if weight_sum <= 1.0 or active_bins <= 1:
        return MulticlassTestResult(
            "full_k_minus_one_quadratic",
            0.0,
            1.0,
            0,
            0,
            active_bins,
            n_observations,
            weight_sum,
            None,
            {"missing_policy": missing_policy, "eigenvalues": []},
        )

    (
        mean_score,
        inverse_covariance,
        eigenvalues,
        response_rank,
        threshold,
    ) = _weighted_score_inverse(score_matrix, weights, eigen_tolerance)
    if response_rank == 0:
        return MulticlassTestResult(
            "full_k_minus_one_quadratic",
            0.0,
            1.0,
            0,
            0,
            active_bins,
            n_observations,
            weight_sum,
            None,
            {
                "missing_policy": missing_policy,
                "eigenvalues": eigenvalues.tolist(),
                "absolute_eigenvalue_threshold": threshold,
            },
        )

    bin_weights = np.bincount(codes, weights=weights, minlength=active_bins)
    bin_score_sums = np.zeros((active_bins, score_matrix.shape[1]), dtype=np.float64)
    np.add.at(bin_score_sums, codes, weights[:, None] * score_matrix)
    differences = bin_score_sums - bin_weights[:, None] * mean_score
    mahalanobis = np.einsum(
        "bi,ij,bj->b", differences, inverse_covariance, differences, optimize=True
    )
    statistic = float(
        (weight_sum - 1.0) / weight_sum * np.sum(mahalanobis / bin_weights)
    )
    numerical_scale = max(1.0, float(np.sum(np.abs(mahalanobis / bin_weights))))
    if statistic < -1e-12 * numerical_scale:
        raise FloatingPointError("quadratic statistic is negative beyond tolerance")
    statistic = max(0.0, statistic)
    degrees_of_freedom = (active_bins - 1) * response_rank
    return MulticlassTestResult(
        method="full_k_minus_one_quadratic",
        statistic=statistic,
        p_value=chi_square_survival(statistic, degrees_of_freedom),
        degrees_of_freedom=degrees_of_freedom,
        response_rank=response_rank,
        active_bins=active_bins,
        n_observations=n_observations,
        weight_sum=weight_sum,
        structure_class=None,
        details={
            "missing_policy": missing_policy,
            "eigenvalues": eigenvalues.tolist(),
            "relative_eigenvalue_tolerance": eigen_tolerance,
            "absolute_eigenvalue_threshold": threshold,
            "quadratic_form": "one_hot_kronecker_closed_form",
        },
    )


def grouped_ordered_selection_bins(
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_groups: int = 8,
) -> np.ndarray:
    """Return selection-only contiguous groups while preserving a missing bin."""

    feature_array = np.asarray(ordered_feature, dtype=object)
    if feature_array.ndim != 1:
        raise ValueError("ordered_feature must be one-dimensional")
    weights = _as_weights(feature_array.shape[0], frequency_weights)
    missing = _missing_mask(feature_array)
    complete = (weights > 0.0) & ~missing
    try:
        numeric_complete = feature_array[complete].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ordered_feature must be numeric outside missing rows") from exc
    if not np.all(np.isfinite(numeric_complete)):
        raise ValueError("ordered_feature must be finite outside missing rows")
    complete_groups = equal_weight_ordered_groups(
        numeric_complete, weights[complete], n_groups=n_groups
    )
    groups = np.empty(feature_array.shape[0], dtype=object)
    groups[:] = np.nan
    groups[complete] = complete_groups.astype(object)
    return groups


def grouped_full_multiclass_quadratic_test(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_groups: int = 8,
    missing_policy: str = "production_bin",
    eigen_tolerance: float = float(np.sqrt(np.finfo(np.float64).eps)),
) -> MulticlassTestResult:
    """Apply the full test to selection-only contiguous ordered groups."""

    groups = grouped_ordered_selection_bins(
        ordered_feature, frequency_weights, n_groups=n_groups
    )
    return full_multiclass_quadratic_test(
        scores,
        groups,
        frequency_weights,
        missing_policy=missing_policy,
        eigen_tolerance=eigen_tolerance,
    )


def grouped_legacy_multiclass_quadratic_test(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    n_groups: int = 8,
    structure_class: Optional[int] = None,
    missing_policy: str = "production_bin",
    epsilon: float = 1e-7,
) -> MulticlassTestResult:
    """Apply the legacy scalar response rule to the same ordered groups."""

    groups = grouped_ordered_selection_bins(
        ordered_feature, frequency_weights, n_groups=n_groups
    )
    return legacy_multiclass_quadratic_test(
        scores,
        groups,
        frequency_weights,
        structure_class=structure_class,
        missing_policy=missing_policy,
        epsilon=epsilon,
    )


def permutation_calibrated_multiclass_test(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    feature: Sequence[Any],
    *,
    method: str = "full",
    n_permutations: int = 499,
    random_state: Union[int, np.random.Generator] = 0,
) -> PermutationCalibrationResult:
    """Calibrate a root-node statistic by permuting whole score vectors.

    The bounded oracle intentionally accepts unit weights only.  Literal
    frequency weights can be expanded before calling it; arbitrary real weights
    do not define exchangeable replicated observations.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    score_matrix = _as_score_matrix(scores)
    feature_array = np.asarray(feature, dtype=object)
    if feature_array.ndim != 1 or feature_array.shape[0] != score_matrix.shape[0]:
        raise ValueError("feature must have one entry per score row")
    score_matrix, prepared_feature, unit_weights = _prepare_feature_test_inputs(
        score_matrix,
        feature_array,
        None,
        "production_bin",
    )
    if score_matrix.shape[0] < 2:
        raise ValueError("permutation calibration requires at least two observations")
    codes, active_bins = _factorize(prepared_feature)
    membership = np.zeros((active_bins, score_matrix.shape[0]), dtype=np.float64)
    membership[codes, np.arange(score_matrix.shape[0])] = 1.0
    bin_weights = np.sum(membership, axis=1)
    weight_sum = float(score_matrix.shape[0])

    if method == "full":
        observed = full_multiclass_quadratic_test(score_matrix, prepared_feature)
        (
            mean_score,
            inverse_covariance,
            _,
            response_rank,
            _,
        ) = _weighted_score_inverse(
            score_matrix,
            unit_weights,
            float(np.sqrt(np.finfo(np.float64).eps)),
        )

        def permutation_statistic(permutation):
            if response_rank == 0 or active_bins <= 1 or weight_sum <= 1.0:
                return 0.0
            bin_sums = membership @ score_matrix[permutation]
            differences = bin_sums - bin_weights[:, None] * mean_score
            mahalanobis = np.einsum(
                "bi,ij,bj->b",
                differences,
                inverse_covariance,
                differences,
                optimize=True,
            )
            return max(
                0.0,
                float(
                    (weight_sum - 1.0)
                    / weight_sum
                    * np.sum(mahalanobis / bin_weights)
                ),
            )

    elif method == "legacy":
        structure_class, _ = select_legacy_structure_class(score_matrix)
        observed = legacy_multiclass_quadratic_test(
            score_matrix,
            prepared_feature,
            structure_class=structure_class,
        )
        scalar_scores = score_matrix[:, structure_class]
        total_score = float(np.sum(scalar_scores))
        mean_score = total_score / weight_sum
        variance = max(
            0.0,
            float(np.dot(scalar_scores, scalar_scores) / weight_sum)
            - mean_score * mean_score,
        )
        degrees_of_freedom = active_bins - 1
        reduced_weights = bin_weights[:degrees_of_freedom]
        diagonal_scale = (
            weight_sum / (weight_sum - 1.0) * variance
            if weight_sum > 1.0
            else 0.0
        )
        outer_scale = variance / (weight_sum - 1.0) if weight_sum > 1.0 else 0.0
        diagonal = diagonal_scale * reduced_weights + 1e-7
        diagonal_projection = float(
            np.sum(np.square(reduced_weights) / diagonal)
        )
        denominator = 1.0 - outer_scale * diagonal_projection

        def permutation_statistic(permutation):
            if (
                variance <= np.finfo(np.float64).eps
                or active_bins <= 1
                or weight_sum <= 1.0
            ):
                return 0.0
            if denominator <= 1e-7:
                return legacy_multiclass_quadratic_test(
                    score_matrix[permutation],
                    prepared_feature,
                    structure_class=structure_class,
                ).statistic
            bin_sums = membership @ scalar_scores[permutation]
            differences = (
                bin_sums[:degrees_of_freedom]
                - reduced_weights * mean_score
            )
            difference_quadratic = float(
                np.sum(np.square(differences) / diagonal)
            )
            weighted_projection = float(
                np.sum(reduced_weights * differences / diagonal)
            )
            return max(
                0.0,
                difference_quadratic
                + outer_scale
                * weighted_projection
                * weighted_projection
                / denominator,
            )
    else:
        raise ValueError("method must be 'full' or 'legacy'")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    exceedances = 0
    for _ in range(n_permutations):
        statistic = permutation_statistic(rng.permutation(score_matrix.shape[0]))
        exceedances += int(statistic >= observed.statistic - 1e-12)
    return PermutationCalibrationResult(
        method=observed.method,
        statistic=observed.statistic,
        asymptotic_p_value=observed.p_value,
        permutation_p_value=(1.0 + exceedances) / (n_permutations + 1.0),
        exceedances=exceedances,
        n_permutations=n_permutations,
        degrees_of_freedom=observed.degrees_of_freedom,
        response_rank=observed.response_rank,
    )


def _validated_gain_inputs(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    hessians: Union[Sequence[Sequence[float]], np.ndarray],
    frequency_weights: Optional[ArrayLike],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_matrix = _as_score_matrix(scores)
    hessian_matrix = np.asarray(hessians, dtype=np.float64)
    if hessian_matrix.shape != score_matrix.shape:
        raise ValueError("hessians must have the same shape as scores")
    if not np.all(np.isfinite(hessian_matrix)) or np.any(hessian_matrix < 0.0):
        raise ValueError("hessians must be finite and non-negative")
    weights = _as_weights(score_matrix.shape[0], frequency_weights)
    keep = weights > 0.0
    return score_matrix[keep], hessian_matrix[keep], weights[keep]


def multiclass_partition_gain(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    hessians: Union[Sequence[Sequence[float]], np.ndarray],
    left_mask: Sequence[bool],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    lambda_l2: float = 1.0,
    class_index: Optional[int] = None,
) -> float:
    """Return the diagonal-Newton improvement for one fixed partition."""

    if not np.isfinite(lambda_l2) or lambda_l2 < 0.0:
        raise ValueError("lambda_l2 must be finite and non-negative")
    original_scores = _as_score_matrix(scores)
    original_weights = _as_weights(original_scores.shape[0], frequency_weights)
    mask = np.asarray(left_mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != original_scores.shape[0]:
        raise ValueError("left_mask must have one entry per score row")
    keep = original_weights > 0.0
    score_matrix, hessian_matrix, weights = _validated_gain_inputs(
        original_scores, hessians, original_weights
    )
    mask = mask[keep]
    if class_index is not None:
        selected = int(class_index)
        if selected < 0 or selected >= score_matrix.shape[1]:
            raise ValueError("class_index is out of range")
        score_matrix = score_matrix[:, selected : selected + 1]
        hessian_matrix = hessian_matrix[:, selected : selected + 1]

    weighted_scores = weights[:, None] * score_matrix
    weighted_hessians = weights[:, None] * hessian_matrix
    total_gradient = np.sum(weighted_scores, axis=0)
    total_hessian = np.sum(weighted_hessians, axis=0)
    left_gradient = np.sum(weighted_scores[mask], axis=0)
    left_hessian = np.sum(weighted_hessians[mask], axis=0)
    right_gradient = total_gradient - left_gradient
    right_hessian = total_hessian - left_hessian

    def score(gradient: np.ndarray, hessian: np.ndarray) -> float:
        denominator = hessian + lambda_l2
        terms = np.divide(
            np.square(gradient),
            denominator,
            out=np.zeros_like(gradient),
            where=denominator > np.finfo(np.float64).eps,
        )
        return float(np.sum(terms))

    return score(left_gradient, left_hessian) + score(
        right_gradient, right_hessian
    ) - score(total_gradient, total_hessian)


def best_numeric_split_gain(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    hessians: Union[Sequence[Sequence[float]], np.ndarray],
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    *,
    lambda_l2: float = 1.0,
    min_data_in_leaf: float = 1.0,
    min_child_weight: float = 0.0,
    structure_class: Optional[int] = None,
) -> NumericSplitResult:
    """Search one already-selected numeric feature with scalar or full gain.

    ``structure_class=None`` sums the diagonal-Newton improvement over every
    score coordinate.  Passing a class index reproduces the legacy scalar cut
    objective.  Missing rows are considered all-right and all-left for every
    non-missing cutpoint.
    """

    if not np.isfinite(lambda_l2) or lambda_l2 < 0.0:
        raise ValueError("lambda_l2 must be finite and non-negative")
    if min_data_in_leaf < 0.0 or min_child_weight < 0.0:
        raise ValueError("minimum leaf constraints must be non-negative")
    original_scores = _as_score_matrix(scores)
    original_weights = _as_weights(original_scores.shape[0], frequency_weights)
    feature = np.asarray(ordered_feature, dtype=np.float64)
    if feature.ndim != 1 or feature.shape[0] != original_scores.shape[0]:
        raise ValueError("ordered_feature must have one entry per score row")
    keep = original_weights > 0.0
    score_matrix, hessian_matrix, weights = _validated_gain_inputs(
        original_scores, hessians, original_weights
    )
    feature = feature[keep]
    if np.any(np.isinf(feature)):
        raise ValueError("ordered_feature may contain finite values or missing values")
    if structure_class is not None:
        selected = int(structure_class)
        if selected < 0 or selected >= score_matrix.shape[1]:
            raise ValueError("structure_class is out of range")
        score_matrix = score_matrix[:, selected : selected + 1]
        hessian_matrix = hessian_matrix[:, selected : selected + 1]
        method = "legacy_single_class_numeric_gain"
    else:
        selected = None
        method = "full_class_invariant_numeric_gain"

    missing = np.isnan(feature)
    nonmissing_indices = np.flatnonzero(~missing)
    missing_indices = np.flatnonzero(missing)
    if nonmissing_indices.shape[0] < 2:
        return NumericSplitResult(method, 0.0, None, False, 0.0, 0.0, 0, selected)
    order = nonmissing_indices[np.argsort(feature[nonmissing_indices], kind="stable")]
    sorted_feature = feature[order]
    if np.unique(sorted_feature).shape[0] < 2:
        return NumericSplitResult(method, 0.0, None, False, 0.0, 0.0, 0, selected)

    weighted_gradient = weights[:, None] * score_matrix
    weighted_hessian = weights[:, None] * hessian_matrix
    total_gradient = np.sum(weighted_gradient, axis=0)
    total_hessian = np.sum(weighted_hessian, axis=0)
    total_weight = float(np.sum(weights))
    missing_gradient = np.sum(weighted_gradient[missing_indices], axis=0)
    missing_hessian = np.sum(weighted_hessian[missing_indices], axis=0)
    missing_weight = float(np.sum(weights[missing_indices]))
    cumulative_gradient = np.cumsum(weighted_gradient[order], axis=0)
    cumulative_hessian = np.cumsum(weighted_hessian[order], axis=0)
    cumulative_weight = np.cumsum(weights[order])

    def node_score(gradient: np.ndarray, hessian: np.ndarray) -> float:
        denominator = hessian + lambda_l2
        terms = np.divide(
            np.square(gradient),
            denominator,
            out=np.zeros_like(gradient),
            where=denominator > np.finfo(np.float64).eps,
        )
        # ``math.fsum`` makes the score independent of class-column order for
        # the same finite terms.  The scale-aware comparison below still
        # handles roundoff accumulated while constructing those terms.
        return float(math.fsum(float(term) for term in terms))

    parent_score = node_score(total_gradient, total_hessian)
    best_gain = -np.inf
    best_threshold = None
    best_missing_left = False
    best_left_weight = 0.0
    evaluated = 0
    for position in range(1, order.shape[0]):
        if sorted_feature[position - 1] == sorted_feature[position]:
            continue
        base_gradient = cumulative_gradient[position - 1]
        base_hessian = cumulative_hessian[position - 1]
        base_weight = float(cumulative_weight[position - 1])
        threshold = float(
            sorted_feature[position - 1]
            + 0.5 * (sorted_feature[position] - sorted_feature[position - 1])
        )
        for missing_go_left in ((False, True) if missing_weight > 0.0 else (False,)):
            left_gradient = base_gradient + (
                missing_gradient if missing_go_left else 0.0
            )
            left_hessian = base_hessian + (
                missing_hessian if missing_go_left else 0.0
            )
            left_weight = base_weight + (missing_weight if missing_go_left else 0.0)
            right_gradient = total_gradient - left_gradient
            right_hessian = total_hessian - left_hessian
            right_weight = total_weight - left_weight
            if left_weight < min_data_in_leaf or right_weight < min_data_in_leaf:
                continue
            if (
                float(np.sum(left_hessian)) < min_child_weight
                or float(np.sum(right_hessian)) < min_child_weight
            ):
                continue
            gain = (
                node_score(left_gradient, left_hessian)
                + node_score(right_gradient, right_hessian)
                - parent_score
            )
            evaluated += 1
            tie_key = (threshold, int(missing_go_left))
            best_tie_key = (
                (best_threshold if best_threshold is not None else np.inf),
                int(best_missing_left),
            )
            if best_threshold is None:
                gain_tolerance = 0.0
            elif selected is not None:
                # Preserve the scalar legacy reference exactly; class-order
                # roundoff cannot occur when only one coordinate is scored.
                gain_tolerance = 1e-15
            else:
                gain_tolerance = NUMERIC_GAIN_TIE_RELATIVE_TOLERANCE * max(
                    1.0, abs(gain), abs(best_gain)
                )
            if gain > best_gain + gain_tolerance or (
                abs(gain - best_gain) <= gain_tolerance
                and tie_key < best_tie_key
            ):
                best_gain = gain
                best_threshold = threshold
                best_missing_left = missing_go_left
                best_left_weight = left_weight
    if best_threshold is None:
        return NumericSplitResult(method, 0.0, None, False, 0.0, 0.0, evaluated, selected)
    return NumericSplitResult(
        method=method,
        gain=float(best_gain),
        threshold=best_threshold,
        missing_go_left=best_missing_left,
        left_weight=best_left_weight,
        right_weight=total_weight - best_left_weight,
        evaluated_cutpoints=evaluated,
        structure_class=selected,
    )


def best_legacy_numeric_split_gain(
    scores: Union[Sequence[Sequence[float]], np.ndarray],
    hessians: Union[Sequence[Sequence[float]], np.ndarray],
    ordered_feature: Sequence[float],
    frequency_weights: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> NumericSplitResult:
    """Search a numeric feature with the current highest-variance coordinate."""

    selected, _ = select_legacy_structure_class(scores, frequency_weights)
    return best_numeric_split_gain(
        scores,
        hessians,
        ordered_feature,
        frequency_weights,
        structure_class=selected,
        **kwargs,
    )
