"""Compare expectile and squared-error objectives on cached TabArena regression tasks."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Each fit uses one CPU thread, including when the two workers run concurrently.
for variable in (
    "CTBOOST_HIST_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

from quantile_ties_scout import STUDIES, load_role, verified_bytes

TAU = 0.8
SEEDS = (0, 1)


def evaluate(case_info):
    import numpy as np
    import pandas as pd

    import ctboost

    directory, case, seed = case_info
    train = load_role(directory, case, "train")
    stop = load_role(directory, case, "stop")
    development = load_role(directory, case, "development")
    categorical = [
        name
        for name, dtype in train["X"].dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    ]
    objective, metric = ctboost.make_expectile_loss(TAU)
    start = float(np.mean(train["y"], dtype=np.float64))
    target = np.asarray(development["y"], dtype=np.float64)
    result = {
        "dataset": case["dataset_name"],
        "seed": seed,
        "train_rows": len(train["y"]),
        "stopping_rows": len(stop["y"]),
        "development_rows": len(target),
        "base_score": start,
    }
    for arm, loss in (("squared_error", "RMSE"), ("expectile", objective)):
        estimator = ctboost.CTBoostRegressor(
            loss_function=loss,
            eval_metric=metric,
            iterations=200,
            learning_rate=0.05,
            max_depth=6,
            alpha=0.05,
            lambda_l2=1,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            ordered_ctr=True,
            max_cat_threshold=64,
            cat_features=categorical or None,
            random_seed=seed,
            base_score=start,
        )
        estimator.fit(
            train["X"],
            train["y"],
            eval_set=(stop["X"], stop["y"]),
            early_stopping_rounds=25,
        )
        prediction = estimator.predict(development["X"])
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid {arm} predictions on {case['dataset_name']}")
        residual = target - prediction.astype(np.float64)
        squared = np.square(residual)
        # Independent of the public metric callback used for stopping.
        asymmetric_mse = np.mean(
            np.where(residual > 0, TAU * squared, (1 - TAU) * squared)
        )
        result[arm] = {
            "asymmetric_mse": float(asymmetric_mse),
            "rmse": float(np.sqrt(np.mean(squared))),
            "selected_rounds": estimator.get_best_iteration() + 1,
            "predictions_sha256": hashlib.sha256(
                np.ascontiguousarray(prediction).tobytes()
            ).hexdigest(),
        }
    return result


def summarize(results):
    summary = {}
    for metric in ("asymmetric_mse", "rmse"):
        changes = [
            1 - row["expectile"][metric] / row["squared_error"][metric]
            for row in results
        ]
        summary[metric] = {
            "mean_relative_reduction": statistics.mean(changes),
            "median_relative_reduction": statistics.median(changes),
            "wins": sum(change > 0 for change in changes),
            "ties": changes.count(0),
            "losses": sum(change < 0 for change in changes),
        }
    return summary


def make_report(results):
    import ctboost

    grouped = {}
    for row in results:
        grouped.setdefault(row["dataset"], []).append(row)
    averages = []
    for dataset, rows in grouped.items():
        if sorted(row["seed"] for row in rows) != list(SEEDS):
            raise ValueError(f"Incomplete paired results for {dataset}")
        averaged = {"dataset": dataset}
        for arm in ("squared_error", "expectile"):
            averaged[arm] = {
                metric: statistics.mean(row[arm][metric] for row in rows)
                for metric in ("asymmetric_mse", "rmse")
            }
        averages.append(averaged)
    return {
        "ctboost": ctboost.__version__,
        "python": platform.python_version(),
        "native_sha256": hashlib.sha256(
            Path(ctboost._core.__file__).read_bytes()
        ).hexdigest(),
        "source_plan_sha256": STUDIES,
        "protocol": {
            "tau": TAU,
            "seeds": list(SEEDS),
            "max_rounds": 200,
            "patience": 25,
            "threads_per_fit": 1,
            "base_score": "arithmetic mean of training labels",
            "selection_metric": "mean asymmetric squared error",
        },
        "summary_by_seed": {
            seed: summarize([r for r in results if r["seed"] == seed]) for seed in SEEDS
        },
        "summary": summarize(averages),
        "mean_errors_by_dataset": averages,
        "tasks": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tasks = []
    for study, digest in STUDIES.items():
        directory = args.cache / study
        plan = json.loads(verified_bytes(directory / "plan.json", digest))
        tasks.extend(
            (directory, case)
            for case in plan["datasets"]
            if case["problem_type"] == "regression"
        )
    if len(tasks) != 13 or len({case["dataset_name"] for _, case in tasks}) != len(
        tasks
    ):
        raise ValueError(
            "Expected all 13 regression tasks from the two verified source plans"
        )
    results = []
    with ProcessPoolExecutor(max_workers=2) as workers:
        jobs = [(directory, case, seed) for seed in SEEDS for directory, case in tasks]
        for result in workers.map(evaluate, jobs):
            print(result["seed"], result["dataset"], file=sys.stderr, flush=True)
            results.append(result)
    report = make_report(results)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
