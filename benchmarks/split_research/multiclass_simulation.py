"""Deterministic bounded screen for multiclass conditional score statistics."""

# ruff: noqa: UP006, UP045
# CTBoost supports Python 3.8, where built-in generic and ``X | None`` syntax
# is not parseable even when annotations are postponed.

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .multiclass import (
    NUMERIC_GAIN_TIE_RELATIVE_TOLERANCE,
    best_numeric_split_gain,
    full_multiclass_quadratic_test,
    grouped_ordered_selection_bins,
    legacy_multiclass_quadratic_test,
    permutation_calibrated_multiclass_test,
    select_legacy_structure_class,
    softmax_diagonal_hessians,
    softmax_score_matrix,
)
from .statistics import equal_frequency_bins


@dataclass(frozen=True)
class MulticlassScenario:
    """One fixed null or power data-generating mechanism."""

    name: str
    family: str
    n_classes: int
    relationship: str
    true_boundary: Optional[float] = None


@dataclass(frozen=True)
class GeneratedScenario:
    feature: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    hessians: np.ndarray
    probabilities: np.ndarray


SCENARIOS = (
    MulticlassScenario("null_balanced_k3", "null", 3, "independent_balanced"),
    MulticlassScenario("null_long_tail_k5", "null", 5, "independent_long_tail"),
    MulticlassScenario(
        "power_exact_tie_k4",
        "power",
        4,
        "class_1_2_midpoint_swap_dyadic_equal_margins",
        true_boundary=0.5,
    ),
    MulticlassScenario(
        "power_imbalanced_hidden_k3",
        "power",
        3,
        "class_1_2_midpoint_swap_class_0_constant",
        true_boundary=0.5,
    ),
    MulticlassScenario(
        "power_diffuse_k5",
        "power",
        5,
        "rotating_quartile_signal_class_0_constant",
    ),
    MulticlassScenario(
        "power_aligned_k3",
        "power",
        3,
        "highest_variance_class_midpoint_shift",
        true_boundary=0.5,
    ),
    MulticlassScenario(
        "power_rare_k4",
        "power",
        4,
        "rare_class_midpoint_swap_class_0_1_constant",
        true_boundary=0.5,
    ),
)

BIN_PROFILES = (
    ("raw_2", 2, False),
    ("raw_8", 8, False),
    ("raw_32", 32, False),
    ("raw_255", 255, False),
    ("grouped8_from_raw255", 255, True),
)

ProgressCallback = Optional[Callable[[str], None]]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a ledger only after a complete, finite JSON snapshot is durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_common_config(n_observations: int, alpha: float) -> None:
    if n_observations <= 0 or n_observations % 120 != 0:
        raise ValueError("n_observations must be a positive multiple of 120")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and lie strictly between zero and one")


def _counts_from_probabilities(total: int, probabilities: Sequence[float]) -> np.ndarray:
    probability_array = np.asarray(probabilities, dtype=np.float64)
    expected = total * probability_array / np.sum(probability_array)
    counts = np.floor(expected).astype(np.int64)
    remainder = total - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _assign_counts(
    labels: np.ndarray,
    indices: np.ndarray,
    counts: Sequence[int],
    rng: np.random.Generator,
) -> None:
    count_array = np.asarray(counts, dtype=np.int64)
    if int(np.sum(count_array)) != indices.shape[0] or np.any(count_array < 0):
        raise ValueError("class counts must be non-negative and fill the stratum")
    values = np.repeat(np.arange(count_array.shape[0], dtype=np.int64), count_array)
    rng.shuffle(values)
    labels[indices] = values


def _stratified_uniform_feature(
    rng: np.random.Generator, n_observations: int
) -> np.ndarray:
    feature = (np.arange(n_observations) + rng.random(n_observations)) / n_observations
    rng.shuffle(feature)
    return feature


def _generate_scenario(
    scenario: MulticlassScenario,
    rng: np.random.Generator,
    n_observations: int,
) -> GeneratedScenario:
    _validate_common_config(n_observations, 0.05)
    feature = _stratified_uniform_feature(rng, n_observations)
    labels = np.empty(n_observations, dtype=np.int64)
    all_indices = np.arange(n_observations)
    left_indices = np.flatnonzero(feature < 0.5)
    right_indices = np.flatnonzero(feature >= 0.5)

    if scenario.name == "null_balanced_k3":
        _assign_counts(
            labels,
            all_indices,
            _counts_from_probabilities(n_observations, (1.0, 1.0, 1.0)),
            rng,
        )
    elif scenario.name == "null_long_tail_k5":
        _assign_counts(
            labels,
            all_indices,
            _counts_from_probabilities(n_observations, (0.55, 0.25, 0.12, 0.06, 0.02)),
            rng,
        )
    elif scenario.name == "power_exact_tie_k4":
        class_total = n_observations // 4
        class_zero_half = class_total // 2
        class_three_half = class_total // 2
        class_one_left = int(rng.binomial(class_total, 0.62))
        _assign_counts(
            labels,
            left_indices,
            (
                class_zero_half,
                class_one_left,
                class_total - class_one_left,
                class_three_half,
            ),
            rng,
        )
        _assign_counts(
            labels,
            right_indices,
            (
                class_zero_half,
                class_total - class_one_left,
                class_one_left,
                class_three_half,
            ),
            rng,
        )
    elif scenario.name == "power_imbalanced_hidden_k3":
        secondary_total = n_observations // 4
        class_one_left = int(rng.binomial(secondary_total, 0.68))
        _assign_counts(
            labels,
            left_indices,
            (
                n_observations // 4,
                class_one_left,
                secondary_total - class_one_left,
            ),
            rng,
        )
        _assign_counts(
            labels,
            right_indices,
            (
                n_observations // 4,
                secondary_total - class_one_left,
                class_one_left,
            ),
            rng,
        )
    elif scenario.name == "power_diffuse_k5":
        quartile = np.minimum((4.0 * feature).astype(np.int64), 3)
        for group in range(4):
            indices = np.flatnonzero(quartile == group)
            class_zero_count = round(0.40 * indices.shape[0])
            remaining = indices.shape[0] - class_zero_count
            lower_probabilities = np.repeat(0.22, 4)
            lower_probabilities[group] = 0.34
            lower_counts = rng.multinomial(
                remaining, lower_probabilities / np.sum(lower_probabilities)
            )
            _assign_counts(
                labels,
                indices,
                np.r_[class_zero_count, lower_counts],
                rng,
            )
    elif scenario.name == "power_aligned_k3":
        left_probabilities = np.asarray([0.58, 0.25, 0.17])
        right_probabilities = np.asarray([0.42, 0.35, 0.23])
        _assign_counts(
            labels,
            left_indices,
            rng.multinomial(left_indices.shape[0], left_probabilities),
            rng,
        )
        _assign_counts(
            labels,
            right_indices,
            rng.multinomial(right_indices.shape[0], right_probabilities),
            rng,
        )
    elif scenario.name == "power_rare_k4":
        rare_total = n_observations // 20
        rare_left = int(rng.binomial(rare_total, 0.20))
        constant_zero = _counts_from_probabilities(n_observations, (0.45, 0.30, 0.20, 0.05))[0]
        constant_one = _counts_from_probabilities(n_observations, (0.45, 0.30, 0.20, 0.05))[1]
        zero_half = int(constant_zero // 2)
        one_half = int(constant_one // 2)
        remaining_half = left_indices.shape[0] - zero_half - one_half
        _assign_counts(
            labels,
            left_indices,
            (zero_half, one_half, remaining_half - rare_left, rare_left),
            rng,
        )
        _assign_counts(
            labels,
            right_indices,
            (
                constant_zero - zero_half,
                constant_one - one_half,
                remaining_half - (rare_total - rare_left),
                rare_total - rare_left,
            ),
            rng,
        )
    else:
        raise ValueError(f"unknown scenario: {scenario.name}")

    counts = np.bincount(labels, minlength=scenario.n_classes).astype(np.float64)
    probabilities = counts / n_observations
    scores = softmax_score_matrix(labels, probabilities)
    hessians = softmax_diagonal_hessians(probabilities, n_observations)
    return GeneratedScenario(feature, labels, scores, hessians, probabilities)


def _profile_feature(feature: np.ndarray, raw_bins: int, grouped: bool) -> np.ndarray:
    raw = equal_frequency_bins(feature, min(raw_bins, feature.shape[0]))
    if not grouped:
        return raw
    return grouped_ordered_selection_bins(raw, n_groups=8)


def _legacy_test(scores: np.ndarray, feature: np.ndarray):
    return legacy_multiclass_quadratic_test(scores, feature)


def run_main_screen(
    *,
    repetitions: int = 400,
    n_observations: int = 480,
    alpha: float = 0.05,
    seed: int = 20260820,
    scenarios: Sequence[MulticlassScenario] = SCENARIOS,
    bin_profiles: Sequence[Tuple[str, int, bool]] = BIN_PROFILES,
    progress_callback: ProgressCallback = None,
) -> Mapping[str, Any]:
    """Run paired asymptotic null/power tests over every frozen bin profile."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not scenarios or not bin_profiles:
        raise ValueError("scenarios and bin_profiles must be non-empty")
    _validate_common_config(n_observations, alpha)
    rng = np.random.default_rng(seed)
    rows: List[Mapping[str, Any]] = []
    elapsed = {
        (method, profile): 0.0
        for method in ("legacy", "full")
        for profile, _, _ in bin_profiles
    }
    calls = {key: 0 for key in elapsed}
    for scenario_index, scenario in enumerate(scenarios):
        collected = {
            profile: {
                "legacy": [],
                "full": [],
                "rank": [],
                "legacy_df": [],
                "full_df": [],
                "bins": [],
            }
            for profile, _, _ in bin_profiles
        }
        selected_counts: Dict[str, int] = {}
        for _ in range(repetitions):
            generated = _generate_scenario(scenario, rng, n_observations)
            for profile, raw_bins, grouped in bin_profiles:
                selection_feature = _profile_feature(
                    generated.feature, raw_bins, grouped
                )
                started = time.perf_counter()
                legacy = _legacy_test(generated.scores, selection_feature)
                elapsed[("legacy", profile)] += time.perf_counter() - started
                calls[("legacy", profile)] += 1
                started = time.perf_counter()
                full = full_multiclass_quadratic_test(
                    generated.scores, selection_feature
                )
                elapsed[("full", profile)] += time.perf_counter() - started
                calls[("full", profile)] += 1
                collected[profile]["legacy"].append(legacy.p_value)
                collected[profile]["full"].append(full.p_value)
                collected[profile]["rank"].append(full.response_rank)
                collected[profile]["legacy_df"].append(legacy.degrees_of_freedom)
                collected[profile]["full_df"].append(full.degrees_of_freedom)
                collected[profile]["bins"].append(full.active_bins)
                key = str(legacy.structure_class)
                selected_counts[key] = selected_counts.get(key, 0) + 1
        for profile, _, _ in bin_profiles:
            legacy_values = np.asarray(collected[profile]["legacy"])
            full_values = np.asarray(collected[profile]["full"])
            legacy_rate = float(np.mean(legacy_values <= alpha))
            full_rate = float(np.mean(full_values <= alpha))
            rows.append(
                {
                    "scenario": scenario.name,
                    "family": scenario.family,
                    "n_classes": scenario.n_classes,
                    "profile": profile,
                    "active_bins": int(np.median(collected[profile]["bins"])),
                    "full_response_rank": int(np.median(collected[profile]["rank"])),
                    "legacy_df": int(np.median(collected[profile]["legacy_df"])),
                    "full_df": int(np.median(collected[profile]["full_df"])),
                    "legacy_rejection_rate": legacy_rate,
                    "full_rejection_rate": full_rate,
                    "full_minus_legacy": full_rate - legacy_rate,
                    "legacy_median_p_value": float(np.median(legacy_values)),
                    "full_median_p_value": float(np.median(full_values)),
                    "legacy_structure_class_counts_across_profile_calls": (
                        selected_counts if profile == bin_profiles[0][0] else None
                    ),
                }
            )
        if progress_callback is not None:
            progress_callback(
                f"main screen scenario {scenario_index + 1}/{len(scenarios)}: "
                f"{scenario.name}"
            )
    timing_rows = []
    for method in ("legacy", "full"):
        for profile, _, _ in bin_profiles:
            timing_rows.append(
                {
                    "method": method,
                    "profile": profile,
                    "calls": calls[(method, profile)],
                    "mean_microseconds_per_test": (
                        1e6 * elapsed[(method, profile)] / calls[(method, profile)]
                    ),
                }
            )
    return {
        "repetitions": repetitions,
        "n_observations": n_observations,
        "alpha": alpha,
        "rows": rows,
        "timing_rows": timing_rows,
    }


def run_permutation_oracle_screen(
    *,
    repetitions: int = 80,
    n_observations: int = 480,
    n_permutations: int = 499,
    alpha: float = 0.05,
    seed: int = 20260920,
    progress_callback: ProgressCallback = None,
) -> Mapping[str, Any]:
    """Compare asymptotic and score-vector permutation null rejection rates."""

    if repetitions < 1 or n_permutations < 1:
        raise ValueError("repetitions and n_permutations must be positive")
    _validate_common_config(n_observations, alpha)
    rng = np.random.default_rng(seed)
    nulls = [scenario for scenario in SCENARIOS if scenario.family == "null"]
    profiles = (("raw_8", 8, False), ("grouped8_from_raw255", 255, True))
    storage = {
        (scenario.name, profile, method): {"asymptotic": [], "permutation": []}
        for scenario in nulls
        for profile, _, _ in profiles
        for method in ("legacy", "full")
    }
    started = time.perf_counter()
    for scenario in nulls:
        for repetition in range(repetitions):
            generated = _generate_scenario(scenario, rng, n_observations)
            for profile_index, (profile, raw_bins, grouped) in enumerate(profiles):
                selection_feature = _profile_feature(
                    generated.feature, raw_bins, grouped
                )
                permutation_seed = (
                    seed
                    + 100000 * (1 + nulls.index(scenario))
                    + 1000 * profile_index
                    + repetition
                )
                for method in ("legacy", "full"):
                    result = permutation_calibrated_multiclass_test(
                        generated.scores,
                        selection_feature,
                        method=method,
                        n_permutations=n_permutations,
                        random_state=permutation_seed,
                    )
                    values = storage[(scenario.name, profile, method)]
                    values["asymptotic"].append(result.asymptotic_p_value)
                    values["permutation"].append(result.permutation_p_value)
        if progress_callback is not None:
            completed_cells = 2 * (nulls.index(scenario) + 1)
            progress_callback(
                f"permutation oracle cells {completed_cells}/4: "
                f"{scenario.name} raw8 + grouped8"
            )
    rows = []
    for scenario in nulls:
        for profile, _, _ in profiles:
            for method in ("legacy", "full"):
                values = storage[(scenario.name, profile, method)]
                asymptotic = np.asarray(values["asymptotic"])
                permutation = np.asarray(values["permutation"])
                asymptotic_rate = float(np.mean(asymptotic <= alpha))
                permutation_rate = float(np.mean(permutation <= alpha))
                rows.append(
                    {
                        "scenario": scenario.name,
                        "profile": profile,
                        "method": method,
                        "asymptotic_rejection_rate": asymptotic_rate,
                        "permutation_rejection_rate": permutation_rate,
                        "absolute_rejection_difference": abs(
                            asymptotic_rate - permutation_rate
                        ),
                        "median_absolute_p_difference": float(
                            np.median(np.abs(asymptotic - permutation))
                        ),
                    }
                )
    return {
        "repetitions": repetitions,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_class_permutation_screen(
    *,
    repetitions: int = 200,
    n_observations: int = 480,
    alpha: float = 0.05,
    seed: int = 20261020,
    progress_callback: ProgressCallback = None,
) -> Mapping[str, Any]:
    """Evaluate every K=4 label order on paired exact-tie-trap draws."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    _validate_common_config(n_observations, alpha)
    scenario = next(item for item in SCENARIOS if item.name == "power_exact_tie_k4")
    permutations = list(itertools.permutations(range(4)))
    rng = np.random.default_rng(seed)
    p_values = {
        permutation: {"legacy": [], "full": []} for permutation in permutations
    }
    maximum_full_statistic_difference = 0.0
    progress_interval = max(1, repetitions // 4)
    for repetition in range(repetitions):
        generated = _generate_scenario(scenario, rng, n_observations)
        feature = _profile_feature(generated.feature, 8, False)
        identity_statistic = None
        for permutation in permutations:
            permuted_scores = generated.scores[:, permutation]
            legacy = legacy_multiclass_quadratic_test(permuted_scores, feature)
            full = full_multiclass_quadratic_test(permuted_scores, feature)
            p_values[permutation]["legacy"].append(legacy.p_value)
            p_values[permutation]["full"].append(full.p_value)
            if identity_statistic is None:
                identity_statistic = full.statistic
            maximum_full_statistic_difference = max(
                maximum_full_statistic_difference,
                abs(full.statistic - identity_statistic),
            )
        if progress_callback is not None and (
            (repetition + 1) % progress_interval == 0
            or repetition + 1 == repetitions
        ):
            progress_callback(
                f"class-permutation draws {repetition + 1}/{repetitions}"
            )
    rows = []
    identity_rejections = (
        np.asarray(p_values[permutations[0]]["full"]) <= alpha
    )
    maximum_paired_rejection_mismatches = 0
    total_paired_rejection_mismatches = 0
    for permutation in permutations:
        legacy = np.asarray(p_values[permutation]["legacy"])
        full = np.asarray(p_values[permutation]["full"])
        paired_rejection_mismatches = int(
            np.count_nonzero((full <= alpha) != identity_rejections)
        )
        maximum_paired_rejection_mismatches = max(
            maximum_paired_rejection_mismatches,
            paired_rejection_mismatches,
        )
        total_paired_rejection_mismatches += paired_rejection_mismatches
        rows.append(
            {
                "label_order": list(permutation),
                "legacy_selected_original_class": permutation[0],
                "legacy_rejection_rate": float(np.mean(legacy <= alpha)),
                "full_rejection_rate": float(np.mean(full <= alpha)),
                "paired_full_rejection_mismatch_count": (
                    paired_rejection_mismatches
                ),
                "legacy_median_p_value": float(np.median(legacy)),
                "full_median_p_value": float(np.median(full)),
            }
        )
    return {
        "scenario": scenario.name,
        "profile": "raw_8",
        "repetitions": repetitions,
        "rows": rows,
        "maximum_paired_full_statistic_difference": (
            maximum_full_statistic_difference
        ),
        "maximum_paired_full_rejection_mismatch_count": (
            maximum_paired_rejection_mismatches
        ),
        "total_paired_full_rejection_mismatch_count": (
            total_paired_rejection_mismatches
        ),
    }


def run_numeric_cut_screen(
    *,
    repetitions: int = 200,
    n_observations: int = 480,
    boundary_tolerance: float = 0.05,
    seed: int = 20261120,
    progress_callback: ProgressCallback = None,
) -> Mapping[str, Any]:
    """Compare scalar and full-class gains after a numeric feature is fixed."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not np.isfinite(boundary_tolerance) or boundary_tolerance < 0.0:
        raise ValueError("boundary_tolerance must be finite and non-negative")
    _validate_common_config(n_observations, 0.05)
    alternatives = [scenario for scenario in SCENARIOS if scenario.family == "power"]
    rng = np.random.default_rng(seed)
    storage = {
        scenario.name: {
            "legacy_gain": [],
            "full_gain": [],
            "legacy_threshold": [],
            "full_threshold": [],
            "legacy_hit": [],
            "full_hit": [],
        }
        for scenario in alternatives
    }
    maximum_permuted_gain_difference = 0.0
    maximum_permuted_threshold_difference = 0.0
    started = time.perf_counter()
    for scenario_index, scenario in enumerate(alternatives):
        for _ in range(repetitions):
            generated = _generate_scenario(scenario, rng, n_observations)
            raw_bins = equal_frequency_bins(generated.feature, 255).astype(np.float64)
            normalized_bins = raw_bins / 255.0
            selected, _ = select_legacy_structure_class(generated.scores)
            legacy = best_numeric_split_gain(
                generated.scores,
                generated.hessians,
                normalized_bins,
                structure_class=selected,
            )
            full = best_numeric_split_gain(
                generated.scores, generated.hessians, normalized_bins
            )
            permutation = np.arange(scenario.n_classes)[::-1]
            permuted = best_numeric_split_gain(
                generated.scores[:, permutation],
                generated.hessians[:, permutation],
                normalized_bins,
            )
            values = storage[scenario.name]
            values["legacy_gain"].append(legacy.gain)
            values["full_gain"].append(full.gain)
            values["legacy_threshold"].append(legacy.threshold)
            values["full_threshold"].append(full.threshold)
            if scenario.true_boundary is not None:
                values["legacy_hit"].append(
                    abs(float(legacy.threshold) - scenario.true_boundary)
                    <= boundary_tolerance
                )
                values["full_hit"].append(
                    abs(float(full.threshold) - scenario.true_boundary)
                    <= boundary_tolerance
                )
            maximum_permuted_gain_difference = max(
                maximum_permuted_gain_difference, abs(full.gain - permuted.gain)
            )
            maximum_permuted_threshold_difference = max(
                maximum_permuted_threshold_difference,
                abs(float(full.threshold) - float(permuted.threshold)),
            )
        if progress_callback is not None:
            progress_callback(
                f"numeric-cut scenario {scenario_index + 1}/{len(alternatives)}: "
                f"{scenario.name}"
            )
    rows = []
    for scenario in alternatives:
        values = storage[scenario.name]
        rows.append(
            {
                "scenario": scenario.name,
                "legacy_mean_gain": float(np.mean(values["legacy_gain"])),
                "full_mean_gain": float(np.mean(values["full_gain"])),
                "legacy_median_threshold": float(
                    np.median(values["legacy_threshold"])
                ),
                "full_median_threshold": float(np.median(values["full_threshold"])),
                "legacy_boundary_hit_rate": (
                    float(np.mean(values["legacy_hit"]))
                    if values["legacy_hit"]
                    else None
                ),
                "full_boundary_hit_rate": (
                    float(np.mean(values["full_hit"]))
                    if values["full_hit"]
                    else None
                ),
            }
        )
    return {
        "repetitions": repetitions,
        "raw_bins": 255,
        "normalized_boundary_tolerance": boundary_tolerance,
        "rows": rows,
        "maximum_class_permuted_gain_difference": maximum_permuted_gain_difference,
        "maximum_class_permuted_threshold_difference": (
            maximum_permuted_threshold_difference
        ),
        "full_numeric_gain_tie_relative_tolerance": float(
            NUMERIC_GAIN_TIE_RELATIVE_TOLERANCE
        ),
        "full_numeric_gain_tie_scale_floor": 1.0,
        "full_numeric_gain_tie_break": "lowest_threshold_then_missing_right",
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_bounded_gate(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Apply the corrected gate frozen in ``MULTICLASS_PROTOCOL_V2.md``."""

    main_rows = result["main_screen"]["rows"]
    gated_profiles = {"raw_2", "raw_8", "raw_32", "grouped8_from_raw255"}
    null_rows = [
        row
        for row in main_rows
        if row["family"] == "null" and row["profile"] in gated_profiles
    ]
    null_calibrated = all(
        0.02 <= row["full_rejection_rate"] <= 0.08 for row in null_rows
    )
    targeted_names = {
        "power_exact_tie_k4",
        "power_imbalanced_hidden_k3",
        "power_diffuse_k5",
    }
    targeted = [
        row
        for row in main_rows
        if row["scenario"] in targeted_names
        and row["profile"] == "grouped8_from_raw255"
    ]
    targeted_improvements = sum(row["full_minus_legacy"] >= 0.10 for row in targeted)
    aligned = next(
        row
        for row in main_rows
        if row["scenario"] == "power_aligned_k3"
        and row["profile"] == "grouped8_from_raw255"
    )
    aligned_guard = aligned["full_minus_legacy"] >= -0.10

    permutation_screen = result["class_permutation"]
    class_invariant = (
        permutation_screen["maximum_paired_full_rejection_mismatch_count"] == 0
        and permutation_screen["maximum_paired_full_statistic_difference"] <= 1e-10
    )
    full_oracle_rows = [
        row for row in result["permutation_oracle"]["rows"] if row["method"] == "full"
    ]
    oracle_calibrated = all(
        0.01 <= row["asymptotic_rejection_rate"] <= 0.09
        and 0.01 <= row["permutation_rejection_rate"] <= 0.09
        and row["absolute_rejection_difference"] <= 0.04
        for row in full_oracle_rows
    )
    cut = result["numeric_cut"]
    aligned_cut = next(
        row for row in cut["rows"] if row["scenario"] == "power_aligned_k3"
    )
    cut_invariant = cut["maximum_class_permuted_gain_difference"] <= 1e-10
    aligned_cut_guard = (
        aligned_cut["full_boundary_hit_rate"]
        >= aligned_cut["legacy_boundary_hit_rate"]
    )
    checks = {
        "full_null_calibration": null_calibrated,
        "targeted_grouped8_improvement_count_at_least_two": (
            targeted_improvements >= 2
        ),
        "aligned_grouped8_power_guard": aligned_guard,
        "class_permutation_invariance": class_invariant,
        "bounded_permutation_oracle": oracle_calibrated,
        "numeric_cut_class_permutation_invariance": cut_invariant,
        "aligned_numeric_cut_guard": aligned_cut_guard,
    }
    return {
        "checks": checks,
        "diagnostics_not_gated": {
            "maximum_class_permuted_threshold_difference": cut[
                "maximum_class_permuted_threshold_difference"
            ],
            "reason": (
                "The frozen protocol gates numeric gain invariance only; "
                "threshold equality remains a reported diagnostic."
            ),
        },
        "targeted_grouped8_improvement_count": targeted_improvements,
        "statistical_screen_passed": all(checks.values()),
        "full_run_warranted_if_correctness_tests_pass": all(checks.values()),
        "qualification": (
            "Passing this bounded screen only warrants the full pre-registered "
            "synthetic run; it does not warrant native integration or a default change."
        ),
    }


def run_reference_screen(
    *,
    repetitions: int = 400,
    oracle_repetitions: int = 80,
    n_permutations: int = 499,
    class_permutation_repetitions: int = 200,
    cut_repetitions: int = 200,
    n_observations: int = 480,
    alpha: float = 0.05,
    seed: int = 20260820,
    progress_callback: ProgressCallback = None,
    checkpoint_callback: Optional[
        Callable[[str, Mapping[str, Any]], None]
    ] = None,
) -> Mapping[str, Any]:
    """Run the complete bounded screen and return a JSON-safe result ledger."""

    if min(
        repetitions,
        oracle_repetitions,
        n_permutations,
        class_permutation_repetitions,
        cut_repetitions,
    ) < 1:
        raise ValueError("all repetition and permutation counts must be positive")
    _validate_common_config(n_observations, alpha)
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "schema_version": 2,
        "protocol": "multiclass-bounded-screen-v2-post-run-audit",
        "status": "reference_only_not_integrated",
        "config": {
            "repetitions": repetitions,
            "oracle_repetitions": oracle_repetitions,
            "n_permutations": n_permutations,
            "class_permutation_repetitions": class_permutation_repetitions,
            "cut_repetitions": cut_repetitions,
            "n_observations": n_observations,
            "alpha": alpha,
            "seed": seed,
            "bin_profiles": [profile for profile, _, _ in BIN_PROFILES],
            "weights": "unit; integer/zero-weight semantics covered by unit tests",
            "missing_policy": "production_bin; missing behavior covered by unit tests",
            "permutation_calibration": (
                "whole score-vector permutations with inclusive plus-one p-values"
            ),
        },
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }
    result["main_screen"] = run_main_screen(
        repetitions=repetitions,
        n_observations=n_observations,
        alpha=alpha,
        seed=seed,
        progress_callback=progress_callback,
    )
    if checkpoint_callback is not None:
        checkpoint_callback("main_screen", result)
    result["permutation_oracle"] = run_permutation_oracle_screen(
        repetitions=oracle_repetitions,
        n_observations=n_observations,
        n_permutations=n_permutations,
        alpha=alpha,
        seed=seed + 100,
        progress_callback=progress_callback,
    )
    if checkpoint_callback is not None:
        checkpoint_callback("permutation_oracle", result)
    result["class_permutation"] = run_class_permutation_screen(
        repetitions=class_permutation_repetitions,
        n_observations=n_observations,
        alpha=alpha,
        seed=seed + 200,
        progress_callback=progress_callback,
    )
    if checkpoint_callback is not None:
        checkpoint_callback("class_permutation", result)
    result["numeric_cut"] = run_numeric_cut_screen(
        repetitions=cut_repetitions,
        n_observations=n_observations,
        seed=seed + 300,
        progress_callback=progress_callback,
    )
    if checkpoint_callback is not None:
        checkpoint_callback("numeric_cut", result)
    result["decision"] = evaluate_bounded_gate(result)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def render_markdown_tables(result: Mapping[str, Any]) -> str:
    """Render every requested rejection, permutation, cut, and runtime table."""

    lines = [
        "### Rejection and power",
        "",
        "| scenario | profile | legacy | full K-1 | delta | full df |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["main_screen"]["rows"]:
        lines.append(
            "| {scenario} | {profile} | {legacy:.3f} | {full:.3f} | {delta:+.3f} | {df} |".format(
                scenario=row["scenario"],
                profile=row["profile"],
                legacy=row["legacy_rejection_rate"],
                full=row["full_rejection_rate"],
                delta=row["full_minus_legacy"],
                df=row["full_df"],
            )
        )
    lines.extend(
        [
            "",
            "### Permutation oracle",
            "",
            "| null | profile | method | asymptotic | permutation | absolute delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in result["permutation_oracle"]["rows"]:
        lines.append(
            "| {scenario} | {profile} | {method} | {asymptotic:.3f} | {permutation:.3f} | {delta:.3f} |".format(
                scenario=row["scenario"],
                profile=row["profile"],
                method=row["method"],
                asymptotic=row["asymptotic_rejection_rate"],
                permutation=row["permutation_rejection_rate"],
                delta=row["absolute_rejection_difference"],
            )
        )
    lines.extend(
        [
            "",
            "### Class-label permutations (tie trap, raw 8)",
            "",
            "| label order | legacy selected original | legacy | full K-1 | paired full mismatches |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["class_permutation"]["rows"]:
        lines.append(
            "| {order} | {selected} | {legacy:.3f} | {full:.3f} | {mismatches} |".format(
                order="-".join(str(value) for value in row["label_order"]),
                selected=row["legacy_selected_original_class"],
                legacy=row["legacy_rejection_rate"],
                full=row["full_rejection_rate"],
                mismatches=row["paired_full_rejection_mismatch_count"],
            )
        )
    lines.extend(
        [
            "",
            "### Numeric cut gain (raw 255)",
            "",
            "| scenario | legacy mean gain | full mean gain | legacy midpoint hit | full midpoint hit |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["numeric_cut"]["rows"]:
        legacy_hit = (
            "n/a"
            if row["legacy_boundary_hit_rate"] is None
            else "{:.3f}".format(row["legacy_boundary_hit_rate"])
        )
        full_hit = (
            "n/a"
            if row["full_boundary_hit_rate"] is None
            else "{:.3f}".format(row["full_boundary_hit_rate"])
        )
        lines.append(
            "| {scenario} | {legacy:.6f} | {full:.6f} | {legacy_hit} | {full_hit} |".format(
                scenario=row["scenario"],
                legacy=row["legacy_mean_gain"],
                full=row["full_mean_gain"],
                legacy_hit=legacy_hit,
                full_hit=full_hit,
            )
        )
    lines.extend(
        [
            "",
            "### Python reference timing",
            "",
            "| method | profile | mean microseconds/test | calls |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["main_screen"]["timing_rows"]:
        lines.append(
            "| {method} | {profile} | {time:.3f} | {calls} |".format(
                method=row["method"],
                profile=row["profile"],
                time=row["mean_microseconds_per_test"],
                calls=row["calls"],
            )
        )
    lines.extend(
        [
            "",
            "Bounded gate: **{}**".format(
                "PASS" if result["decision"]["statistical_screen_passed"] else "FAIL"
            ),
            "",
            "Paired full-test rejection mismatches: {}".format(
                result["class_permutation"][
                    "total_paired_full_rejection_mismatch_count"
                ]
            ),
            "",
            "Maximum class-permuted numeric-threshold difference "
            "(diagnostic only): {:.12g}".format(
                result["numeric_cut"][
                    "maximum_class_permuted_threshold_difference"
                ]
            ),
            "",
            "Total runtime: {:.3f} seconds".format(result["elapsed_seconds"]),
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=400)
    parser.add_argument("--oracle-repetitions", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--class-permutation-repetitions", type=int, default=200)
    parser.add_argument("--cut-repetitions", type=int, default=200)
    parser.add_argument("--n-observations", type=int, default=480)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()

    def report_progress(message: str) -> None:
        print(f"[multiclass-reference] {message}", flush=True)

    def write_checkpoint(stage: str, partial: Mapping[str, Any]) -> None:
        if args.output is None:
            return
        checkpoint = dict(partial)
        checkpoint["completion"] = {
            "complete": False,
            "last_completed_stage": stage,
        }
        checkpoint["elapsed_seconds"] = time.perf_counter() - started
        _write_json_atomic(args.output, checkpoint)
        report_progress(f"checkpoint written after {stage}")

    result = run_reference_screen(
        repetitions=args.repetitions,
        oracle_repetitions=args.oracle_repetitions,
        n_permutations=args.permutations,
        class_permutation_repetitions=args.class_permutation_repetitions,
        cut_repetitions=args.cut_repetitions,
        n_observations=args.n_observations,
        alpha=args.alpha,
        seed=args.seed,
        progress_callback=report_progress,
        checkpoint_callback=write_checkpoint,
    )
    result["completion"] = {
        "complete": True,
        "last_completed_stage": "decision",
    }
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(render_markdown_tables(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
