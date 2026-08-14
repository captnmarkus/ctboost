"""Measure deterministic per-node histogram scaling across CPU thread counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from typing import Any

import numpy as np

import ctboost


def _fit_once(
    pool: Any,
    *,
    node_threads: int,
    quantization_threads: int,
    minimum_parallel_values: int,
    minimum_values_per_worker: int,
    iterations: int,
    max_depth: int,
) -> tuple[float, str]:
    # Keep quantization parallelism fixed so the comparison isolates the new
    # per-node histogram path as closely as an end-to-end fit can.
    os.environ["CTBOOST_HIST_THREADS"] = str(quantization_threads)
    os.environ["CTBOOST_NODE_HIST_THREADS"] = str(node_threads)
    os.environ["CTBOOST_NODE_HIST_MIN_PARALLEL_VALUES"] = str(
        minimum_parallel_values
    )
    os.environ["CTBOOST_NODE_HIST_MIN_VALUES_PER_WORKER"] = str(
        minimum_values_per_worker
    )
    started = time.perf_counter()
    model = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "iterations": iterations,
            "max_depth": max_depth,
            "max_bins": 64,
            "alpha": 1.0,
            "learning_rate": 0.1,
            "random_seed": 20260814,
        },
        num_boost_round=iterations,
    )
    elapsed = time.perf_counter() - started
    state = json.dumps(
        model._handle.export_state(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return elapsed, hashlib.sha256(state).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--quantization-threads", type=int, default=1)
    parser.add_argument("--minimum-parallel-values", type=int, default=1_048_576)
    parser.add_argument("--minimum-values-per-worker", type=int, default=262_144)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    args = parser.parse_args()
    if min(
        args.rows,
        args.threads,
        args.quantization_threads,
        args.minimum_values_per_worker,
        args.iterations,
        args.max_depth,
    ) < 1:
        parser.error(
            "rows, features, threads, quantization-threads, iterations, and max-depth "
            "must be positive"
        )
    if args.features < 4:
        parser.error("features must be at least 4 for the fixed benchmark signal")
    if args.minimum_parallel_values < 0:
        parser.error("minimum-parallel-values must be non-negative")
    if args.threads < 2:
        parser.error("threads must be at least 2")
    if args.repeats < 2:
        parser.error("repeats must be at least 2")

    rng = np.random.default_rng(20260814)
    data = rng.normal(size=(args.rows, args.features)).astype(np.float32)
    label = (
        1.4 * data[:, 0]
        - 0.9 * data[:, 1]
        + 0.5 * data[:, 2] * data[:, 3]
        + rng.normal(scale=0.2, size=args.rows)
    ).astype(np.float32)
    pool = ctboost.Pool(data, label)

    timings: dict[int, list[float]] = {1: [], args.threads: []}
    state_hashes: set[str] = set()
    # Interleave the variants so thermal and allocation drift do not favor one side.
    for repeat in range(args.repeats):
        order = (1, args.threads) if repeat % 2 == 0 else (args.threads, 1)
        for thread_count in order:
            elapsed, state_hash = _fit_once(
                pool,
                node_threads=thread_count,
                quantization_threads=args.quantization_threads,
                minimum_parallel_values=args.minimum_parallel_values,
                minimum_values_per_worker=args.minimum_values_per_worker,
                iterations=args.iterations,
                max_depth=args.max_depth,
            )
            timings[thread_count].append(elapsed)
            state_hashes.add(state_hash)

    single_median = statistics.median(timings[1])
    parallel_median = statistics.median(timings[args.threads])
    speedup = single_median / parallel_median
    report: dict[str, Any] = {
        "rows": args.rows,
        "features": args.features,
        "iterations": args.iterations,
        "max_depth": args.max_depth,
        "parallel_threads": args.threads,
        "quantization_threads": args.quantization_threads,
        "minimum_parallel_values": args.minimum_parallel_values,
        "minimum_values_per_worker": args.minimum_values_per_worker,
        "single_thread_seconds": timings[1],
        "parallel_seconds": timings[args.threads],
        "single_thread_median_seconds": single_median,
        "parallel_median_seconds": parallel_median,
        "speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "model_state_hashes": sorted(state_hashes),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if len(state_hashes) == 1 and speedup >= args.minimum_speedup else 2


if __name__ == "__main__":
    raise SystemExit(main())
