"""Focused native training microbenchmark for feature-test dispatch overhead.

This benchmark is intentionally separate from TabArena. It guards the legacy
default path while the grouped statistic remains behind an opt-in external
validation gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Dict, List, Tuple

import numpy as np

import ctboost


def _fit(
    X: np.ndarray, y: np.ndarray, params: Dict[str, object]
) -> Tuple[float, np.ndarray, object]:
    started = time.perf_counter()
    model = ctboost.train(X, params, label=y, num_boost_round=int(params["iterations"]))
    elapsed = time.perf_counter() - started
    state = model._handle.export_state()
    return elapsed, model.predict(X), state["trees"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-explicit-default-ratio", type=float, default=1.15)
    args = parser.parse_args()
    if args.rows < 100 or args.features < 2 or args.iterations < 1 or args.repeats < 2:
        parser.error("rows/features/iterations/repeats are below the benchmark minimum")

    rng = np.random.default_rng(20260810)
    X = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    y = (
        1.4 * X[:, 0]
        - 0.8 * X[:, 1]
        + 0.6 * np.square(X[:, 2])
        + 0.1 * rng.normal(size=args.rows)
    ).astype(np.float32)
    common: Dict[str, object] = {
        "objective": "RMSE",
        "iterations": args.iterations,
        "learning_rate": 0.15,
        "max_depth": 3,
        "alpha": 0.05,
        "max_bins": 255,
        "random_seed": 41,
    }
    cases = {
        "implicit_default": common,
        "explicit_quadratic": {
            **common,
            "feature_test": "quadratic",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
        "grouped8": {
            **common,
            "feature_test": "grouped",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
    }
    timings: Dict[str, List[float]] = {name: [] for name in cases}
    predictions = {}
    trees = {}

    # One untimed fit per branch warms imports, allocation arenas, and code pages.
    for name, params in cases.items():
        _, predictions[name], trees[name] = _fit(X, y, params)
    order_rng = np.random.default_rng(91)
    names = list(cases)
    for _ in range(args.repeats):
        for index in order_rng.permutation(len(names)):
            name = names[int(index)]
            elapsed, predictions[name], trees[name] = _fit(X, y, cases[name])
            timings[name].append(elapsed)

    medians = {name: statistics.median(values) for name, values in timings.items()}
    implicit = medians["implicit_default"]
    report = {
        "rows": args.rows,
        "features": args.features,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "median_seconds": medians,
        "ratio_to_implicit": {name: value / implicit for name, value in medians.items()},
        "explicit_default_predictions_equal": bool(
            np.array_equal(predictions["implicit_default"], predictions["explicit_quadratic"])
        ),
        "explicit_default_trees_equal": trees["implicit_default"] == trees["explicit_quadratic"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["explicit_default_predictions_equal"] or not report["explicit_default_trees_equal"]:
        return 2
    if report["ratio_to_implicit"]["explicit_quadratic"] > args.max_explicit_default_ratio:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
