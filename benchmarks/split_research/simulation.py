"""Deterministic null-calibration and power simulations for split statistics."""

# ruff: noqa: UP006, UP045
# CTBoost supports Python 3.8, where built-in generic and ``X | None`` syntax
# is not parseable even when annotations are postponed.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .categorical import (
    compare_newton_and_woe_ordering,
    cross_fitted_binary_category_scores,
    smoothed_woe_scores,
)
from .statistics import (
    equal_frequency_bins,
    global_bonferroni_stop,
    grouped_ordered_quadratic_test,
    nominal_quadratic_test,
    ordered_grouped_hybrid_test,
    ordered_linear_test,
    permutation_maxstat_test,
)

METHODS = ("nominal_quadratic", "ordered_midrank", "permutation_maxstat")


@dataclass(frozen=True)
class Scenario:
    name: str
    family: str
    relationship: str
    cardinality: Optional[int] = None
    missing_rate: float = 0.0
    frequency_weighted: bool = False


DEFAULT_SCENARIOS = (
    Scenario("null_cardinality_2", "null", "independent", cardinality=2),
    Scenario("null_cardinality_8", "null", "independent", cardinality=8),
    Scenario("null_cardinality_32", "null", "independent", cardinality=32),
    Scenario("null_cardinality_64", "null", "independent", cardinality=64),
    Scenario("null_cardinality_255", "null", "independent", cardinality=255),
    Scenario("null_missing_0", "null", "independent", cardinality=8),
    Scenario(
        "null_missing_30", "null", "independent", cardinality=8, missing_rate=0.30
    ),
    Scenario(
        "null_frequency_weights",
        "null",
        "independent_binary_aggregate",
        cardinality=8,
        frequency_weighted=True,
    ),
    Scenario("power_smooth", "power", "smooth"),
    Scenario("power_abrupt", "power", "abrupt"),
    Scenario("power_u_shaped", "power", "u_shaped"),
)


def _generate_scenario(
    scenario: Scenario,
    rng: np.random.Generator,
    n_observations: int,
    effect_size: float,
    n_nominal_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scenario.frequency_weighted:
        expanded_feature = rng.integers(
            0, int(scenario.cardinality), size=n_observations, dtype=np.int64
        )
        expanded_target = rng.integers(0, 2, size=n_observations, dtype=np.int64)
        combinations, counts = np.unique(
            np.column_stack((expanded_feature, expanded_target)),
            axis=0,
            return_counts=True,
        )
        feature = combinations[:, 0].astype(np.float64)
        score = combinations[:, 1].astype(np.float64)
        weights = counts.astype(np.float64)
        return feature, score, weights, feature.copy()

    if scenario.family == "null" and scenario.cardinality is not None:
        feature = rng.integers(0, scenario.cardinality, size=n_observations).astype(
            np.float64
        )
        score = rng.normal(size=n_observations)
        if scenario.missing_rate > 0.0:
            feature[rng.random(n_observations) < scenario.missing_rate] = np.nan
        return feature, score, np.ones(n_observations), feature.copy()

    feature = rng.random(n_observations)
    noise = rng.normal(size=n_observations)
    if scenario.relationship == "smooth":
        signal = np.sqrt(12.0) * (feature - 0.5)
    elif scenario.relationship == "abrupt":
        indicator = (feature > 0.65).astype(np.float64)
        signal = (indicator - np.mean(indicator)) / np.std(indicator)
    elif scenario.relationship == "u_shaped":
        raw_signal = np.square(feature - 0.5)
        signal = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
    else:
        raise ValueError(f"unknown relationship: {scenario.relationship}")
    score = effect_size * signal + noise
    nominal_feature = equal_frequency_bins(feature, n_nominal_bins)
    return feature, score, np.ones(n_observations), nominal_feature


def _summarize_p_values(
    p_values: Mapping[str, List[float]], alpha: float
) -> Mapping[str, Any]:
    summary: Dict[str, Any] = {}
    for method, values in p_values.items():
        array = np.asarray(values, dtype=np.float64)
        summary[method] = {
            "rejection_rate": float(np.mean(array <= alpha)),
            "median_p_value": float(np.median(array)),
            "p10": float(np.quantile(array, 0.10)),
            "p90": float(np.quantile(array, 0.90)),
        }
    return summary


def run_scenario_experiment(
    *,
    repetitions: int = 250,
    n_observations: int = 320,
    n_permutations: int = 199,
    alpha: float = 0.05,
    effect_size: float = 0.6,
    n_nominal_bins: int = 8,
    min_fraction: float = 0.1,
    seed: int = 20260810,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
) -> List[Mapping[str, Any]]:
    """Run repeated per-feature null and power experiments."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = np.random.default_rng(seed)
    output: List[Mapping[str, Any]] = []
    for scenario in scenarios:
        p_values = {method: [] for method in METHODS}
        for _ in range(repetitions):
            feature, score, weights, nominal_feature = _generate_scenario(
                scenario, rng, n_observations, effect_size, n_nominal_bins
            )
            nominal = nominal_quadratic_test(score, nominal_feature, weights)
            ordered = ordered_linear_test(score, feature, weights)
            maxstat = permutation_maxstat_test(
                score,
                feature,
                weights,
                n_permutations=n_permutations,
                min_fraction=min_fraction,
                random_state=rng,
            )
            p_values["nominal_quadratic"].append(nominal.p_value)
            p_values["ordered_midrank"].append(ordered.p_value)
            p_values["permutation_maxstat"].append(maxstat.p_value)
        output.append(
            {
                "scenario": asdict(scenario),
                "repetitions": repetitions,
                "methods": _summarize_p_values(p_values, alpha),
            }
        )
    return output


def run_global_null_experiment(
    *,
    repetitions: int = 200,
    n_observations: int = 320,
    cardinalities: Sequence[int] = (2, 8, 32, 64, 255),
    n_permutations: int = 199,
    alpha: float = 0.05,
    min_fraction: float = 0.1,
    seed: int = 20260811,
) -> Mapping[str, Any]:
    """Measure family-wise error after global Bonferroni stopping."""

    rng = np.random.default_rng(seed)
    rejected = {method: 0 for method in METHODS}
    selected_counts = {
        method: {str(cardinality): 0 for cardinality in cardinalities}
        for method in METHODS
    }
    for _ in range(repetitions):
        score = rng.normal(size=n_observations)
        feature_p_values = {method: [] for method in METHODS}
        for cardinality in cardinalities:
            feature = rng.integers(0, cardinality, size=n_observations).astype(
                np.float64
            )
            weights = np.ones(n_observations)
            feature_p_values["nominal_quadratic"].append(
                nominal_quadratic_test(score, feature, weights).p_value
            )
            feature_p_values["ordered_midrank"].append(
                ordered_linear_test(score, feature, weights).p_value
            )
            feature_p_values["permutation_maxstat"].append(
                permutation_maxstat_test(
                    score,
                    feature,
                    weights,
                    n_permutations=n_permutations,
                    min_fraction=min_fraction,
                    random_state=rng,
                ).p_value
            )
        for method, p_values in feature_p_values.items():
            decision = global_bonferroni_stop(p_values, alpha=alpha)
            rejected[method] += int(decision.should_split)
            if decision.selected_index is not None:
                key = str(cardinalities[decision.selected_index])
                selected_counts[method][key] += 1
    return {
        "repetitions": repetitions,
        "cardinalities": list(cardinalities),
        "alpha": alpha,
        "methods": {
            method: {
                "familywise_rejection_rate": rejected[method] / repetitions,
                "selected_counts_after_stop": selected_counts[method],
            }
            for method in METHODS
        },
    }


def run_numeric_quantization_experiment(
    *,
    repetitions: int = 200,
    n_observations: int = 320,
    requested_bin_counts: Sequence[int] = (8, 32, 64, 255),
    n_permutations: int = 199,
    alpha: float = 0.05,
    effect_size: float = 0.6,
    min_fraction: float = 0.1,
    seed: int = 20260813,
) -> Mapping[str, Any]:
    """Measure the k-1 df test's power as numeric quantization gets finer."""

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    bin_counts = sorted(
        {max(2, min(int(count), n_observations)) for count in requested_bin_counts}
    )
    rng = np.random.default_rng(seed)
    relationships = ("null", "smooth", "abrupt", "u_shaped")
    nominal_p_values = {
        relationship: {count: [] for count in bin_counts}
        for relationship in relationships
    }
    alternatives = {
        relationship: {"ordered_midrank": [], "permutation_maxstat": []}
        for relationship in relationships
    }
    for relationship in relationships:
        for _ in range(repetitions):
            feature = rng.random(n_observations)
            noise = rng.normal(size=n_observations)
            if relationship == "null":
                score = noise
            elif relationship == "smooth":
                score = effect_size * np.sqrt(12.0) * (feature - 0.5) + noise
            elif relationship == "abrupt":
                indicator = (feature > 0.65).astype(np.float64)
                standardized = (indicator - np.mean(indicator)) / np.std(indicator)
                score = effect_size * standardized + noise
            else:
                raw_signal = np.square(feature - 0.5)
                standardized = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
                score = effect_size * standardized + noise

            for bin_count in bin_counts:
                binned = equal_frequency_bins(feature, bin_count)
                nominal_p_values[relationship][bin_count].append(
                    nominal_quadratic_test(score, binned).p_value
                )
            alternatives[relationship]["ordered_midrank"].append(
                ordered_linear_test(score, feature).p_value
            )
            alternatives[relationship]["permutation_maxstat"].append(
                permutation_maxstat_test(
                    score,
                    feature,
                    n_permutations=n_permutations,
                    min_fraction=min_fraction,
                    random_state=rng,
                ).p_value
            )

    rows: List[Mapping[str, Any]] = []
    for relationship in relationships:
        alternative_rates = {
            method: float(np.mean(np.asarray(values) <= alpha))
            for method, values in alternatives[relationship].items()
        }
        for bin_count in bin_counts:
            values = np.asarray(nominal_p_values[relationship][bin_count])
            rows.append(
                {
                    "relationship": relationship,
                    "requested_bins": bin_count,
                    "nominal_rejection_rate": float(np.mean(values <= alpha)),
                    "nominal_median_p_value": float(np.median(values)),
                    **alternative_rates,
                }
            )
    return {
        "repetitions": repetitions,
        "n_observations": n_observations,
        "requested_bin_counts": bin_counts,
        "note": (
            "For continuous inputs, the final setting is min(255, n_observations); "
            "ordered and maxstat rates are computed once per data replicate and repeated "
            "for comparison, not retuned per quantization."
        ),
        "rows": rows,
    }


def run_missing_policy_experiment(
    *,
    repetitions: int = 2000,
    n_observations: int = 320,
    cardinality: int = 8,
    missing_rate: float = 0.30,
    alpha: float = 0.05,
    seed: int = 20260814,
) -> Mapping[str, Any]:
    """Compare CTBoost's missing-bin baseline with CTree feature-wise omission."""

    rng = np.random.default_rng(seed)
    p_values = {"production_bin": [], "ctree_omit": []}
    for _ in range(repetitions):
        feature = rng.integers(0, cardinality, size=n_observations).astype(np.float64)
        feature[rng.random(n_observations) < missing_rate] = np.nan
        score = rng.normal(size=n_observations)
        for policy, values in p_values.items():
            values.append(
                nominal_quadratic_test(score, feature, missing_policy=policy).p_value
            )
    return {
        "repetitions": repetitions,
        "n_observations": n_observations,
        "cardinality": cardinality,
        "missing_rate": missing_rate,
        "alpha": alpha,
        "policies": {
            policy: {
                "rejection_rate": float(np.mean(np.asarray(values) <= alpha)),
                "median_p_value": float(np.median(values)),
            }
            for policy, values in p_values.items()
        },
    }


def run_grouped_hybrid_experiment(
    *,
    repetitions: int = 250,
    n_observations: int = 320,
    raw_bins: int = 255,
    n_permutations: int = 199,
    alpha: float = 0.05,
    effect_size: float = 0.6,
    seed: int = 20260815,
) -> Mapping[str, Any]:
    """Evaluate cheap grouped/hybrid tests at a high raw-bin cardinality."""

    rng = np.random.default_rng(seed)
    relationships = ("null", "smooth", "abrupt", "u_shaped", "missing_signal")
    methods = (
        "raw_nominal_255",
        "grouped_quadratic_8",
        "grouped_quadratic_16",
        "ordered_midrank",
        "hybrid_8",
        "hybrid_16",
        "permutation_maxstat",
    )
    p_values = {
        relationship: {method: [] for method in methods}
        for relationship in relationships
    }
    elapsed = {method: 0.0 for method in methods}
    for relationship in relationships:
        for _ in range(repetitions):
            feature = rng.random(n_observations)
            noise = rng.normal(size=n_observations)
            missing_mask = np.zeros(n_observations, dtype=bool)
            if relationship == "null":
                score = noise
            elif relationship == "smooth":
                score = effect_size * np.sqrt(12.0) * (feature - 0.5) + noise
            elif relationship == "abrupt":
                indicator = (feature > 0.65).astype(np.float64)
                signal = (indicator - np.mean(indicator)) / np.std(indicator)
                score = effect_size * signal + noise
            elif relationship == "u_shaped":
                raw_signal = np.square(feature - 0.5)
                signal = (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)
                score = effect_size * signal + noise
            else:
                missing_mask = rng.random(n_observations) < 0.30
                signal = (
                    missing_mask.astype(np.float64) - np.mean(missing_mask)
                ) / np.std(missing_mask)
                score = effect_size * signal + noise
            binned = equal_frequency_bins(feature, min(raw_bins, n_observations))
            binned[missing_mask] = np.nan

            calls = (
                ("raw_nominal_255", nominal_quadratic_test, (score, binned), {}),
                (
                    "grouped_quadratic_8",
                    grouped_ordered_quadratic_test,
                    (score, binned),
                    {"n_groups": 8},
                ),
                (
                    "grouped_quadratic_16",
                    grouped_ordered_quadratic_test,
                    (score, binned),
                    {"n_groups": 16},
                ),
                ("ordered_midrank", ordered_linear_test, (score, binned), {}),
                (
                    "hybrid_8",
                    ordered_grouped_hybrid_test,
                    (score, binned),
                    {"n_groups": 8},
                ),
                (
                    "hybrid_16",
                    ordered_grouped_hybrid_test,
                    (score, binned),
                    {"n_groups": 16},
                ),
                (
                    "permutation_maxstat",
                    permutation_maxstat_test,
                    (score, binned),
                    {"n_permutations": n_permutations, "random_state": rng},
                ),
            )
            for method, call, positional, keywords in calls:
                started = time.perf_counter()
                result = call(*positional, **keywords)
                elapsed[method] += time.perf_counter() - started
                p_values[relationship][method].append(result.p_value)

    rows = []
    for relationship in relationships:
        rows.append(
            {
                "relationship": relationship,
                "rejection_rates": {
                    method: float(np.mean(np.asarray(values) <= alpha))
                    for method, values in p_values[relationship].items()
                },
            }
        )
    total_calls_per_method = repetitions * len(relationships)
    return {
        "repetitions_per_relationship": repetitions,
        "n_observations": n_observations,
        "raw_bins": min(raw_bins, n_observations),
        "alpha": alpha,
        "rows": rows,
        "mean_microseconds_per_test": {
            method: 1e6 * seconds / total_calls_per_method
            for method, seconds in elapsed.items()
        },
        "guardrail": (
            "Grouping affects feature selection only; any later gain/cut search would retain "
            "all raw bins. Hybrid p-values include a within-feature Bonferroni factor of two."
        ),
    }


def run_categorical_diagnostics(seed: int = 20260812) -> Mapping[str, Any]:
    """Check matched Newton/WoE ordering and a high-cardinality leakage control."""

    rng = np.random.default_rng(seed)
    category_counts = np.asarray([7, 13, 29, 51, 83], dtype=np.int64)
    probabilities = np.asarray([0.08, 0.22, 0.48, 0.71, 0.91])
    categories = np.concatenate(
        [
            np.repeat(f"level_{index}", count)
            for index, count in enumerate(category_counts)
        ]
    )
    target = np.concatenate(
        [
            rng.binomial(1, probability, size=count)
            for count, probability in zip(category_counts, probabilities)
        ]
    ).astype(np.float64)
    comparison = compare_newton_and_woe_ordering(categories, target, l2=2.0)

    unique_categories = np.asarray(
        [f"row_{index}" for index in range(240)], dtype=object
    )
    null_target = rng.integers(0, 2, size=unique_categories.shape[0]).astype(np.float64)
    prior = float(np.mean(null_target))
    in_sample_table = smoothed_woe_scores(
        unique_categories, null_target, prior=prior, smoothing=1.0
    )
    in_sample = np.asarray([in_sample_table[value] for value in unique_categories])
    cross_fitted, _ = cross_fitted_binary_category_scores(
        unique_categories,
        null_target,
        n_splits=5,
        random_state=seed,
        method="woe",
        smoothing=1.0,
    )
    in_sample_correlation = float(np.corrcoef(in_sample, null_target)[0, 1])
    cross_fitted_correlation = (
        0.0
        if np.std(cross_fitted) == 0.0
        else float(np.corrcoef(cross_fitted, null_target)[0, 1])
    )
    return {
        "newton_woe": {
            "pairwise_order_agreement": comparison.pairwise_agreement,
            "orders_equal": comparison.newton_order == comparison.woe_order,
            "matched_smoothing": comparison.matched_smoothing,
            "newton_order": list(comparison.newton_order),
            "woe_order": list(comparison.woe_order),
        },
        "unique_category_null": {
            "n_observations": int(unique_categories.shape[0]),
            "in_sample_target_correlation": in_sample_correlation,
            "cross_fitted_target_correlation": cross_fitted_correlation,
            "cross_fitted_all_unseen_are_zero": bool(np.all(cross_fitted == 0.0)),
        },
    }


def run_reference_experiment(
    *,
    repetitions: int = 250,
    global_repetitions: int = 200,
    n_observations: int = 320,
    n_permutations: int = 199,
    alpha: float = 0.05,
    effect_size: float = 0.6,
    seed: int = 20260810,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    scenarios = run_scenario_experiment(
        repetitions=repetitions,
        n_observations=n_observations,
        n_permutations=n_permutations,
        alpha=alpha,
        effect_size=effect_size,
        seed=seed,
    )
    global_null = run_global_null_experiment(
        repetitions=global_repetitions,
        n_observations=n_observations,
        n_permutations=n_permutations,
        alpha=alpha,
        seed=seed + 1,
    )
    numeric_quantization = run_numeric_quantization_experiment(
        repetitions=repetitions,
        n_observations=n_observations,
        n_permutations=n_permutations,
        alpha=alpha,
        effect_size=effect_size,
        seed=seed + 2,
    )
    missing_policy = run_missing_policy_experiment(
        repetitions=max(2000, repetitions),
        n_observations=n_observations,
        alpha=alpha,
        seed=seed + 3,
    )
    grouped_hybrid = run_grouped_hybrid_experiment(
        repetitions=repetitions,
        n_observations=n_observations,
        n_permutations=n_permutations,
        alpha=alpha,
        effect_size=effect_size,
        seed=seed + 4,
    )
    categorical = run_categorical_diagnostics(seed + 5)
    elapsed_seconds = time.perf_counter() - started
    return {
        "schema_version": 1,
        "status": "reference_only_not_integrated",
        "config": {
            "repetitions": repetitions,
            "global_repetitions": global_repetitions,
            "n_observations": n_observations,
            "n_permutations": n_permutations,
            "alpha": alpha,
            "effect_size": effect_size,
            "seed": seed,
            "maxstat_calibration": (
                "one shared response-permutation draw per feature test is evaluated "
                "across every eligible cutpoint; inclusive plus-one Monte Carlo p-value"
            ),
        },
        "scenarios": scenarios,
        "global_null": global_null,
        "numeric_quantization": numeric_quantization,
        "missing_policy": missing_policy,
        "grouped_hybrid": grouped_hybrid,
        "categorical_diagnostics": categorical,
        "elapsed_seconds": elapsed_seconds,
    }


def render_markdown_table(result: Mapping[str, Any]) -> str:
    lines = [
        "| scenario | nominal quadratic | ordered midrank | permutation maxstat |",
        "|---|---:|---:|---:|",
    ]
    for row in result["scenarios"]:
        methods = row["methods"]
        lines.append(
            "| {name} | {nominal:.3f} | {ordered:.3f} | {maxstat:.3f} |".format(
                name=row["scenario"]["name"],
                nominal=methods["nominal_quadratic"]["rejection_rate"],
                ordered=methods["ordered_midrank"]["rejection_rate"],
                maxstat=methods["permutation_maxstat"]["rejection_rate"],
            )
        )
    lines.append("")
    lines.append(f"Runtime: {result['elapsed_seconds']:.2f} seconds")
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=250)
    parser.add_argument("--global-repetitions", type=int, default=200)
    parser.add_argument("--n-observations", type=int, default=320)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--effect-size", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = run_reference_experiment(
        repetitions=args.repetitions,
        global_repetitions=args.global_repetitions,
        n_observations=args.n_observations,
        n_permutations=args.permutations,
        alpha=args.alpha,
        effect_size=args.effect_size,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(render_markdown_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
