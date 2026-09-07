"""Synthetic equal-work CPU throughput probe; never reads official benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Immutable specification: parse a fresh copy instead of exposing mutable defaults.
WORKLOAD_SPEC_JSON = """{
  "schema_version": 1,
  "name": "ctboost_0160_synthetic_worker_throughput_v2",
  "ctboost_version": "0.1.60",
  "data_seed": 160,
  "model_seed_base": 160000,
  "prediction_rows": 256,
  "parameters": {
    "iterations": 128, "learning_rate": 0.08, "max_depth": 5,
    "alpha": 0.05, "lambda_l2": 1.0, "min_data_in_leaf": 16,
    "max_bins": 64, "subsample": 0.8, "bootstrap_type": "Bernoulli",
    "feature_test": "quadratic", "task_type": "CPU", "verbose": false,
    "leaf_estimation_iterations": 1, "leaf_estimation_backtracking": false,
    "multiclass_leaf_solver": "diagonal", "multiclass_feature_test": "single",
    "multi_strategy": "one_output_per_tree", "ordered_ctr": true,
    "one_hot_max_size": 2, "max_cat_threshold": 64, "ctr_prior_strength": 0.5
  },
  "profiles": [
    {"name": "numeric_binary", "problem_type": "binary", "rows": 32768, "numeric_columns": 96, "categorical_columns": 0, "classes": 2},
    {"name": "numeric_regression", "problem_type": "regression", "rows": 8192, "numeric_columns": 48, "categorical_columns": 0, "classes": 0},
    {"name": "numeric_multiclass", "problem_type": "multiclass", "rows": 4096, "numeric_columns": 32, "categorical_columns": 0, "classes": 6},
    {"name": "categorical_binary", "problem_type": "binary", "rows": 8192, "numeric_columns": 16, "categorical_columns": 8, "classes": 2}
  ],
  "categorical_levels": 64,
  "warmup": {"profile": "numeric_binary", "rows": 128, "iterations": 4, "max_depth": 3, "model_seed": 159999},
  "allowed_job_counts": [32, 64],
  "early_stopping": false,
  "validation_data": false
}"""


def specification():
    return json.loads(WORKLOAD_SPEC_JSON)


def spec_hash():
    payload = json.dumps(specification(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def configure_threads(histogram_threads):
    if histogram_threads not in (1, 2):
        raise ValueError("The comparison permits only one or two histogram threads")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["CTBOOST_HIST_THREADS"] = str(histogram_threads)


def runtime_provenance():
    import ctboost

    package = Path(ctboost.__file__).resolve()
    if (
        not sys.flags.isolated
        or ctboost.__version__ != "0.1.60"
        or Path(__file__).resolve().parents[2] / "ctboost" in package.parents
    ):
        raise RuntimeError(
            "Use python -I with the installed public CTBoost 0.1.60 wheel"
        )
    native = Path(ctboost._core.__file__)
    return {
        "ctboost_version": ctboost.__version__,
        "package_path": str(package),
        "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "python": sys.version,
        "spec_sha256": spec_hash(),
    }


def make_jobs(count=32):
    spec = specification()
    if count not in spec["allowed_job_counts"]:
        raise ValueError("Use the same 32 or 64 jobs for each comparison layout")
    return [
        {
            "job_id": index,
            "profile": spec["profiles"][index % 4]["name"],
            "model_seed": spec["model_seed_base"] + index,
        }
        for index in range(count)
    ]


def warmup_job():
    spec = specification()["warmup"]
    return {"job_id": -1, "profile": spec["profile"], "model_seed": spec["model_seed"]}


def prepare_profiles():
    """Generate each profile once per worker, outside the individual fit timer."""
    import numpy as np
    import pandas as pd

    spec = specification()
    prepared = {}
    for index, profile in enumerate(spec["profiles"]):
        rng = np.random.default_rng(spec["data_seed"] + index)
        rows, columns = profile["rows"], profile["numeric_columns"]
        numeric = rng.normal(size=(rows, columns))
        features = pd.DataFrame(numeric, columns=[f"x{i}" for i in range(columns)])
        signal = numeric[:, :6] @ np.array([1.2, -1.0, 0.8, -0.6, 0.4, 0.3])
        signal += 0.7 * np.sin(numeric[:, 6]) + 0.6 * numeric[:, 7] * numeric[:, 8]
        categories = []
        for cat in range(profile["categorical_columns"]):
            codes = rng.integers(spec["categorical_levels"], size=rows)
            name = f"cat{cat}"
            features[name] = pd.Categorical([f"value-{value}" for value in codes])
            categories.append(name)
            signal += np.sin(codes * (cat + 1)) * 0.8
        if profile["problem_type"] == "regression":
            labels = signal + rng.normal(scale=0.4, size=rows)
        elif profile["problem_type"] == "multiclass":
            logits = numeric[:, :12] @ rng.normal(size=(12, profile["classes"]))
            labels = (logits + rng.normal(scale=0.5, size=logits.shape)).argmax(axis=1)
        else:
            labels = (rng.random(rows) < 1 / (1 + np.exp(-signal))).astype(np.int64)
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(features, index=True).values.tobytes()
        )
        digest.update(np.ascontiguousarray(labels).tobytes())
        prepared[profile["name"]] = {
            "X": features,
            "y": labels,
            "cat_features": categories,
            "profile": profile,
            "data_sha256": digest.hexdigest(),
        }
    return prepared


def fit_job(job, prepared, *, histogram_threads):
    """Execute exactly one fixed fit; prediction digests check equal work, not scores."""
    configure_threads(histogram_threads)
    import numpy as np
    from threadpoolctl import threadpool_limits

    import ctboost

    is_warmup = job == warmup_job()
    if not is_warmup and job not in make_jobs(64):
        raise ValueError("Unknown or modified workload job")
    data = prepared[job["profile"]]
    spec = specification()
    params = spec["parameters"]
    features, labels = data["X"], data["y"]
    if is_warmup:
        params.update(
            {name: spec["warmup"][name] for name in ("iterations", "max_depth")}
        )
        features = features.iloc[: spec["warmup"]["rows"]]
        labels = labels[: spec["warmup"]["rows"]]
    model_cls = (
        ctboost.CTBoostRegressor
        if data["profile"]["problem_type"] == "regression"
        else ctboost.CTBoostClassifier
    )
    model = model_cls(
        **params,
        random_seed=job["model_seed"],
        cat_features=data["cat_features"] or None,
    )
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        model.fit(features, labels)
        fit_seconds = time.perf_counter() - started
        rounds = int(model.get_booster().num_iterations_trained)
        if rounds != params["iterations"]:
            raise RuntimeError("A synthetic job did not finish its declared iterations")
        predict = (
            model.predict
            if data["profile"]["problem_type"] == "regression"
            else model.predict_proba
        )
        values = np.ascontiguousarray(
            predict(data["X"].iloc[: spec["prediction_rows"]]), dtype=np.float64
        )
    return {
        **job,
        "status": "complete",
        "is_warmup": is_warmup,
        "spec_sha256": spec_hash(),
        "data_sha256": data["data_sha256"],
        "histogram_threads": histogram_threads,
        "histogram_threads_env": os.environ["CTBOOST_HIST_THREADS"],
        "fit_seconds": fit_seconds,
        "iterations_trained": rounds,
        "prediction_shape": list(values.shape),
        "prediction_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "prediction_values": values.tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs", required=True, help="Comma-separated job indices from 0 through 63"
    )
    parser.add_argument("--histogram-threads", type=int, choices=[1, 2], required=True)
    args = parser.parse_args(argv)
    ids = [int(value) for value in args.jobs.split(",")]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not all(0 <= index < 64 for index in ids)
    ):
        parser.error("Jobs must be distinct indices from 0 through 63")
    started = time.perf_counter()
    configure_threads(args.histogram_threads)
    runtime = runtime_provenance()
    preparation = time.perf_counter()
    prepared = prepare_profiles()
    preparation_seconds = time.perf_counter() - preparation
    warmup = fit_job(warmup_job(), prepared, histogram_threads=args.histogram_threads)
    print(
        json.dumps(
            {
                "event": "ready",
                "runtime": runtime,
                "preparation_seconds": preparation_seconds,
                "warmup": warmup,
            }
        ),
        flush=True,
    )
    jobs = make_jobs(64)
    for index in ids:
        print(
            json.dumps(
                fit_job(jobs[index], prepared, histogram_threads=args.histogram_threads)
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "worker_complete",
                "job_count": len(ids),
                "worker_wall_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
