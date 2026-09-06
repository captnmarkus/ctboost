"""Calibrate the opt-in joint conditional statistic, not a leaderboard model.

The reference explicitly constructs the full conditional Kronecker covariance
and uses NumPy's symmetric pseudoinverse, independently of the native binwise
whitening shortcut. The chi-square tail is asymptotic. Integer weights represent
literal repeated observations; fractional weights retain the frequency formula
and do not have a general finite-sample or type-I error guarantee. Multiplying
weights changes the assumed sample mass and therefore changes significance.

The integer-frequency null below compresses an independently generated expanded
contingency table. It does not generate correlated duplicates and then pretend
they are independent observations. Fractional-weight runs are diagnostics only.

Reference: Hothorn, Hornik and Zeileis (2006), equations (1)-(2), quadratic form
and asymptotic rank-df discussion, https://www.zeileis.org/papers/Hothorn+Hornik+Zeileis-2006.pdf
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
from scipy.stats import chi2


def reference_statistic(gradients, bins, weights=None, *, groups=0, missing_bin=-1):
    responses = np.asarray(gradients, dtype=np.float32).astype(np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    weights = (np.ones(len(bins)) if weights is None else np.asarray(weights, dtype=np.float32).astype(np.float64))
    total = weights.sum()
    active = np.unique(bins[weights > 0])
    mapped = bins.copy()
    if groups:
        nonmissing = [level for level in active if level != missing_bin]
        mass = sum(weights[bins == level].sum() for level in nonmissing)
        before = 0.0
        for level in nonmissing:
            count = weights[bins == level].sum()
            mapped[bins == level] = min(groups - 1, int(groups * (before + count / 2) / mass))
            before += count
        if missing_bin in active:
            mapped[bins == missing_bin] = groups
        active = np.unique(mapped[weights > 0])
    if total <= 1 or len(active) <= 1:
        return {"chi_square": 0.0, "p_value": 1.0, "degrees_of_freedom": 0, "response_rank": 0}
    mean = np.average(responses, axis=0, weights=weights)
    centered = responses - mean
    covariance = (centered * weights[:, None]).T @ centered / total
    eigenvalues = np.linalg.eigvalsh(covariance)
    rank = int(np.count_nonzero(eigenvalues > eigenvalues[-1] * 1e-10)) if eigenvalues[-1] > 0 else 0
    if not rank:
        return {"chi_square": 0.0, "p_value": 1.0, "degrees_of_freedom": 0, "response_rank": 0}
    indicator = np.column_stack([mapped == level for level in active]).astype(float)
    counts = indicator.T @ weights
    observed = indicator.T @ (weights[:, None] * responses)
    residual = observed - counts[:, None] * mean
    feature_covariance = (total * np.diag(counts) - np.outer(counts, counts)) / (total - 1)
    conditional_covariance = np.kron(feature_covariance, covariance)
    inverse = np.linalg.pinv(conditional_covariance, rcond=1e-10, hermitian=True)
    quadratic = float(residual.ravel() @ inverse @ residual.ravel())
    df = (len(active) - 1) * rank
    return {"chi_square": quadratic, "p_value": float(chi2.sf(quadratic, df)),
            "degrees_of_freedom": df, "response_rank": rank,
            "conditional_covariance": conditional_covariance, "observed": observed}


def native_statistic(gradients, bins, weights=None, *, groups=0, missing_bin=-1):
    import ctboost._core as core

    return core._debug_compute_multivariate_pvalue(
        np.asarray(gradients, dtype=np.float32), np.asarray(bins, dtype=np.int64),
        weights=None if weights is None else np.asarray(weights, dtype=np.float32),
        groups=groups, missing_bin=missing_bin,
    )


def small_permutation_oracle():
    responses = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    bins = np.array([0, 0, 1, 1])
    reference = reference_statistic(responses, bins)
    observed = []
    scores = []
    for permutation in itertools.permutations(range(4)):
        result = reference_statistic(responses[list(permutation)], bins)
        observed.append(result["observed"].ravel())
        scores.append(result["chi_square"])
    empirical_covariance = np.cov(np.asarray(observed), rowvar=False, bias=True)
    np.testing.assert_allclose(empirical_covariance, reference["conditional_covariance"], atol=1e-14)
    native = native_statistic(responses, bins)
    np.testing.assert_allclose(native["chi_square"], reference["chi_square"], atol=1e-12)
    return {"permutations": len(scores), "covariance_max_error": float(np.max(np.abs(
                empirical_covariance - reference["conditional_covariance"]))),
            "chi_square_tail": float(native["p_value"]),
            "exact_permutation_tail": float(np.mean(np.asarray(scores) >= reference["chi_square"] - 1e-12))}


def run_calibration(*, repetitions=500, rows=480, classes=3, bins_count=4, seed=159):
    from ctboost import _core

    rng = np.random.default_rng(seed)
    scenarios = {name: [] for name in ("unweighted", "integer_frequency", "fractional_frequency_approximation")}
    for _ in range(repetitions):
        labels = rng.integers(classes, size=rows)
        bins = rng.integers(bins_count, size=rows)
        responses = (1.0 / classes - np.eye(classes)[labels]).astype(np.float32)
        scenarios["unweighted"].append(native_statistic(responses, bins)["p_value"])
        table = np.bincount(bins * classes + labels, minlength=bins_count * classes)
        occupied = np.flatnonzero(table)
        frequency_bins, frequency_labels = np.divmod(occupied, classes)
        compressed = (1.0 / classes - np.eye(classes)[frequency_labels]).astype(np.float32)
        frequency = native_statistic(compressed, frequency_bins, table[occupied])
        np.testing.assert_allclose(frequency["p_value"], scenarios["unweighted"][-1], rtol=1e-8, atol=1e-12)
        scenarios["integer_frequency"].append(frequency["p_value"])
        fractional = rng.uniform(0.25, 2.0, size=rows).astype(np.float32)
        result = native_statistic(responses, bins, fractional)
        if result["frequency_weights"]:
            raise AssertionError("Fractional weights must be marked as an approximation")
        scenarios["fractional_frequency_approximation"].append(result["p_value"])
    results = {}
    for name, values in scenarios.items():
        values = np.asarray(values)
        results[name] = {"rejections_at_0_05": int(np.count_nonzero(values <= 0.05)),
                         "rejection_rate": float(np.mean(values <= 0.05)),
                         "finite_pvalues": bool(np.isfinite(values).all()),
                         "median_pvalue": float(np.median(values))}
    return {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "build_info": _core.build_info(),
                            "native_extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest()},
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": seed, "repetitions": repetitions, "rows": rows, "classes": classes,
            "feature_bins": bins_count, "scenarios": results,
            "permutation_oracle": small_permutation_oracle(),
            "limits": ["Chi-square p-values are asymptotic, including for integer frequency weights.",
                       "Fractional weights have no general type-I error guarantee.",
                       "This is a fixed-feature null simulation, not a guarantee for adaptive boosted trees.",
                       "Multiple candidate features require a separate multiplicity policy.",
                       "No predictive or leaderboard improvement is inferred from this calibration."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--rows", type=int, default=480)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=159)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.rows < 2 or not 2 <= args.classes <= 32 or args.bins < 2:
        parser.error("require positive repetitions, rows >= 2, 2 <= classes <= 32 and bins >= 2")
    result = run_calibration(repetitions=args.repetitions, rows=args.rows, classes=args.classes,
                             bins_count=args.bins, seed=args.seed)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
