"""Frozen, node-level calibration diagnostics for the unchanged grouped score.

This is not a tree-training benchmark or a test of adaptive-tree family-wise
error. Fractional weights are sensitivity diagnostics, not frequency samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

PROTOCOL = {
    "name": "ctboost-grouped-node-calibration-v1",
    "seed": 20260907,
    "repetitions": 2000,
    "alpha": 0.05,
    "groups": 8,
    "cases": [
        {"name": "small_24", "rows": 24, "bins": 255},
        {"name": "small_96", "rows": 96, "bins": 255},
        {"name": "reference_320", "rows": 320, "bins": 255},
        {"name": "sparse_96", "rows": 96, "bins": 255, "sparse": True},
        {"name": "missing_min_96", "rows": 96, "bins": 255, "missing": "min"},
        {"name": "missing_max_96", "rows": 96, "bins": 255, "missing": "max"},
        {"name": "frequency_320", "rows": 320, "bins": 32, "weights": "frequency"},
        {"name": "fractional_unit", "rows": 96, "bins": 255, "weight_scale": 1.0},
        {"name": "fractional_small", "rows": 96, "bins": 255, "weight_scale": 0.1},
        {"name": "fractional_large", "rows": 96, "bins": 255, "weight_scale": 10.0},
    ],
    "interval": "95% Wilson interval for each scenario's marginal rejection rate",
    "decision": "Descriptive diagnostics only; no post-hoc default or threshold changes",
}


def protocol_hash():
    return hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generate_case(case, repetition):
    # The three fractional-weight scales deliberately share the same draws.
    case_index = 7 if "weight_scale" in case else PROTOCOL["cases"].index(case)
    rng = np.random.default_rng(
        np.random.SeedSequence([PROTOCOL["seed"], case_index, repetition])
    )
    rows, levels = case["rows"], case["bins"]
    bins = rng.integers(0, levels, size=rows, dtype=np.int64)
    scores = rng.normal(size=rows).astype(np.float32)
    weights = np.ones(rows, dtype=np.float32)
    if case.get("sparse"):
        bins[rng.random(rows) < 0.9] = 0
    missing_bin = -1
    if "missing" in case:
        missing = rng.random(rows) < 0.3
        if case["missing"] == "min":
            bins += 1
            missing_bin = 0
        else:
            missing_bin = levels
        bins[missing] = missing_bin
    if case.get("weights") == "frequency":
        # Start with independent observations, then losslessly collapse identical
        # (feature, binary-score) pairs. Do not duplicate one random score w times.
        binary_scores = rng.integers(0, 2, size=rows)
        pairs, counts = np.unique(
            np.column_stack((bins, binary_scores)), axis=0, return_counts=True
        )
        bins = pairs[:, 0].astype(np.int64)
        scores = pairs[:, 1].astype(np.float32)
        weights = counts.astype(np.float32)
    if "weight_scale" in case:
        raw_weights = rng.lognormal(mean=0.0, sigma=1.0, size=rows)
        weights = (raw_weights / raw_weights.mean() * case["weight_scale"]).astype(
            np.float32
        )
    return scores, bins, weights, missing_bin


def wilson_interval(successes, count):
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count**2))
        / denominator
    )
    return [
        0.0 if successes == 0 else max(0.0, center - half),
        1.0 if successes == count else min(1.0, center + half),
    ]


def run_diagnostics(*, repetitions=None):
    import ctboost._core as core

    import ctboost

    count = PROTOCOL["repetitions"] if repetitions is None else repetitions
    if count < 1:
        raise ValueError("repetitions must be positive")
    summaries = []
    for case in PROTOCOL["cases"]:
        p_values, degrees, effective_sizes = [], [], []
        for repetition in range(count):
            scores, bins, weights, missing_bin = generate_case(case, repetition)
            result = core._debug_compute_grouped_pvalue(
                scores, bins, weights, PROTOCOL["groups"], missing_bin
            )
            for key in ("p_value", "chi_square", "degrees_of_freedom"):
                if result[key] != result["optimized_" + key]:
                    raise AssertionError(
                        f"Grouped score changed: {case['name']} / {key}"
                    )
            p_value = result["optimized_p_value"]
            if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                raise AssertionError("Non-finite or invalid p-value")
            p_values.append(p_value)
            degrees.append(result["optimized_degrees_of_freedom"])
            weights64 = weights.astype(np.float64)
            effective_sizes.append(
                float(weights64.sum() ** 2 / np.dot(weights64, weights64))
            )
        rejected = sum(value <= PROTOCOL["alpha"] for value in p_values)
        summaries.append(
            {
                "name": case["name"],
                "replicates": count,
                "rejections": rejected,
                "rejection_rate": rejected / count,
                "rejection_rate_interval": wilson_interval(rejected, count),
                "median_p_value": float(np.median(p_values)),
                "degrees_of_freedom_range": [min(degrees), max(degrees)],
                "median_weight_effective_size_diagnostic": float(
                    np.median(effective_sizes)
                ),
                "legacy_optimized_exact": True,
                "interpretation": (
                    "Fractional-weight sensitivity only; not a calibrated frequency null"
                    if "weight_scale" in case
                    else "Marginal fixed-node null; not adaptive-tree FWER"
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "protocol_sha256": protocol_hash(),
        "registered_repetitions_used": count == PROTOCOL["repetitions"],
        "ctboost_version": ctboost.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "build_info": ctboost.build_info(),
        "native_extension_sha256": hashlib.sha256(
            Path(core.__file__).read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenarios": summaries,
        "scope": "Fixed-node diagnostics; no fits, no topology changes, no full-tree FWER claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = (
        {"protocol": PROTOCOL, "protocol_sha256": protocol_hash()}
        if args.protocol_only
        else run_diagnostics()
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
