"""Measure sparse CSC quantization/materialization against equivalent dense data.

Pool construction and CSC canonicalization are deliberately outside the timer.
This benchmark covers initial quantization and bin materialization only; it
does not claim full sparse training or memory proportional to ``nnz`` because
the fitted histogram matrix remains dense and compactly binned.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import numpy as np

import ctboost


def _time_histogram(pool: Any, *, max_bins: int) -> float:
    started = time.perf_counter()
    ctboost._core._debug_build_histogram(
        pool._handle,
        max_bins=max_bins,
        nan_mode="Min",
    )
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-sparse-to-dense-ratio", type=float, default=1.5)
    args = parser.parse_args()
    if args.rows < 1 or args.features < 1 or args.threads < 1 or not 0.0 < args.density <= 1.0:
        parser.error("rows/features must be positive and density must be in (0, 1]")
    if args.repeats < 2:
        parser.error("repeats must be at least 2")

    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - benchmark dependency
        raise SystemExit("install scipy to run the sparse histogram benchmark") from exc

    rng = np.random.default_rng(20260814)
    matrix = sparse.random(
        args.rows,
        args.features,
        density=args.density,
        format="csc",
        dtype=np.float32,
        random_state=rng,
        data_rvs=lambda count: rng.normal(size=count).astype(np.float32),
    )
    labels = np.zeros(args.rows, dtype=np.float32)
    sparse_pool = ctboost.Pool(matrix, labels)
    dense_pool = ctboost.Pool(matrix.toarray(), labels)
    os.environ["CTBOOST_HIST_THREADS"] = str(args.threads)

    # Warm common allocation paths, then interleave variants so drift does not
    # systematically favor dense or sparse input.
    _time_histogram(dense_pool, max_bins=args.max_bins)
    _time_histogram(sparse_pool, max_bins=args.max_bins)
    dense_seconds: list[float] = []
    sparse_seconds: list[float] = []
    for repeat in range(args.repeats):
        variants = (
            (("dense", dense_pool), ("sparse", sparse_pool))
            if repeat % 2 == 0
            else (("sparse", sparse_pool), ("dense", dense_pool))
        )
        for name, pool in variants:
            elapsed = _time_histogram(pool, max_bins=args.max_bins)
            (dense_seconds if name == "dense" else sparse_seconds).append(elapsed)
    dense_median = statistics.median(dense_seconds)
    sparse_median = statistics.median(sparse_seconds)
    ratio = sparse_median / dense_median
    report = {
        "rows": args.rows,
        "features": args.features,
        "density": args.density,
        "nnz": int(matrix.nnz),
        "max_bins": args.max_bins,
        "repeats": args.repeats,
        "threads": args.threads,
        "dense_seconds": dense_seconds,
        "sparse_seconds": sparse_seconds,
        "dense_median_seconds": dense_median,
        "sparse_median_seconds": sparse_median,
        "sparse_to_dense_ratio": ratio,
        "max_sparse_to_dense_ratio": args.max_sparse_to_dense_ratio,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ratio <= args.max_sparse_to_dense_ratio else 2


if __name__ == "__main__":
    raise SystemExit(main())
