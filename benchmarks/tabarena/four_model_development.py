"""Compare single CPU fits on the verified training, stopping and development splits."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

# Allow use with python -I and installed model wheels, from outside the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantile_ties_scout import (
    STUDIES,
    fit as fit_ctboost,
    load_role,
    verified_bytes,
)

LIBRARIES = ("ctboost", "catboost", "lightgbm", "xgboost")


def summarize(directory, output):
    import numpy as np
    from scipy.stats import rankdata

    metadata = {}
    tasks = {}
    for library in LIBRARIES:
        source = directory / (library + ".jsonl")
        header, *rows = [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        ]
        if (
            header["library"] != library
            or len(rows) != 51
            or len({r["dataset"] for r in rows}) != 51
        ):
            raise ValueError(f"Incomplete or duplicate task records in {source}")
        metadata[library] = header
        for row in rows:
            name = row.pop("dataset")
            problem, n = row.pop("problem"), row.pop("development_rows")
            task = tasks.setdefault(
                name,
                {
                    "dataset": name,
                    "problem": problem,
                    "development_rows": n,
                    "models": {},
                },
            )
            if task["problem"] != problem or task["development_rows"] != n:
                raise ValueError(f"Inconsistent task metadata on {name}")
            task["models"][library] = row
    if len(tasks) != 51:
        raise ValueError("The four libraries covered different task rosters")
    for task in tasks.values():
        models = task["models"]
        if all("error" in entry for entry in models.values()):
            ranks = rankdata(
                [models[lib]["error"] for lib in LIBRARIES], method="average"
            )
            for library, rank in zip(LIBRARIES, ranks):
                models[library]["rank"] = float(rank)
    groups = {}
    for group in ("all", "binary", "multiclass", "regression"):
        rated = [
            t
            for t in tasks.values()
            if (group == "all" or t["problem"] == group)
            and "rank" in t["models"]["ctboost"]
        ]
        if not rated:
            continue
        means = {
            lib: float(np.mean([t["models"][lib]["rank"] for t in rated]))
            for lib in LIBRARIES
        }
        pairs = {}
        for lib in LIBRARIES[1:]:
            differences = [
                t["models"]["ctboost"]["error"] - t["models"][lib]["error"]
                for t in rated
            ]
            pairs[lib] = {
                "wins": sum(d < 0 for d in differences),
                "ties": sum(d == 0 for d in differences),
                "losses": sum(d > 0 for d in differences),
            }
        groups[group] = {
            "complete_tasks": len(rated),
            "average_rank": means,
            "ctboost_head_to_head": pairs,
        }
    result = {
        "protocol": {
            "seed": 0,
            "round_cap": 200,
            "patience": 25,
            "threads_per_fit": 1,
            "source_plan_sha256": STUDIES,
            "metrics": {
                "binary": "1 - ROC-AUC",
                "multiclass": "log loss",
                "regression": "RMSE",
            },
        },
        "packages": metadata,
        "groups": groups,
        "tasks": list(tasks.values()),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


def fit_xgboost(case, train, stop, dev, seed):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from development_lightgbm import prepare_native_frames

    problem = case["problem_type"]
    X, X_stop, X_dev = prepare_native_frames(train["X"], stop["X"], dev["X"])

    def matrix(frame, labels=None):
        return xgb.DMatrix(frame, label=labels, enable_categorical=True, nthread=1)

    params = {
        "objective": {
            "binary": "binary:logistic",
            "multiclass": "multi:softprob",
            "regression": "reg:squarederror",
        }[problem],
        "eval_metric": {
            "binary": "logloss",
            "multiclass": "mlogloss",
            "regression": "rmse",
        }[problem],
        "nthread": 1,
        "seed": seed,
        "device": "cpu",
    }
    y_train, y_stop = train["y"], stop["y"]
    if problem != "regression":
        classes, y_train = np.unique(y_train, return_inverse=True)
        y_stop = pd.Index(classes).get_indexer(y_stop)
        if (y_stop < 0).any():
            raise ValueError("Stopping labels are outside the training classes")
        if problem == "multiclass":
            params["num_class"] = len(classes)
    model = xgb.train(
        params,
        matrix(X, y_train),
        num_boost_round=200,
        evals=[(matrix(X_stop, y_stop), "stop")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    retained = model.best_iteration + 1
    return model.predict(matrix(X_dev), iteration_range=(0, retained)), retained


def run_task(library, directory, case):
    import numpy as np
    from sklearn.metrics import log_loss, roc_auc_score

    dev = load_role(directory, case, "development")
    if library == "ctboost":
        model = fit_ctboost(directory, case, seed=0)
        rounds = model.get_best_iteration() + 1
        if case["problem_type"] == "regression":
            prediction = model.predict(dev["X"])
        else:
            classes = model.classes_
            probabilities = model.predict_proba(dev["X"])
            prediction = (
                probabilities[:, 1]
                if case["problem_type"] == "binary"
                else probabilities
            )
    else:
        train = load_role(directory, case, "train")
        stop = load_role(directory, case, "stop")
        if library == "catboost":
            from development_catboost import fit_predict
        elif library == "lightgbm":
            from development_lightgbm import fit_predict
        else:
            fit_predict = fit_xgboost
        prediction, rounds = fit_predict(case, train, stop, {"X": dev["X"]}, seed=0)
        if case["problem_type"] == "multiclass":
            classes = np.unique(train["y"])
    target = np.asarray(dev["y"])
    if len(prediction) != len(target) or not np.isfinite(prediction).all():
        raise ValueError("Invalid development predictions")
    if case["problem_type"] == "binary":
        error = 1 - roc_auc_score(target, prediction)
    elif case["problem_type"] == "multiclass":
        error = log_loss(
            target, np.asarray(prediction, dtype=np.float64), labels=classes
        )
    else:
        error = np.sqrt(np.mean((target.astype(np.float64) - prediction) ** 2))
    digest = hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest()
    return float(error), int(rounds), digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--library", choices=LIBRARIES)
    mode.add_argument("--summarize", type=Path, metavar="RECEIPT_DIRECTORY")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New per-library JSONL or summary JSON",
    )
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.output)
        return
    if not args.cache:
        parser.error("--library requires --cache")
    for variable in (
        "CTBOOST_HIST_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    versions = {
        name: importlib.metadata.version(name)
        for name in (args.library, "numpy", "pandas", "scikit-learn", "scipy")
    }
    failed = 0
    with args.output.open("x", encoding="utf-8") as report:
        print(
            json.dumps(
                {
                    "library": args.library,
                    "versions": versions,
                    "python": platform.python_version(),
                }
            ),
            file=report,
        )
        for study, digest in STUDIES.items():
            directory = args.cache / study
            plan = json.loads(verified_bytes(directory / "plan.json", digest))
            for case in plan["datasets"]:
                row = {
                    "dataset": case["dataset_name"],
                    "problem": case["problem_type"],
                    "development_rows": case["roles"]["development"]["rows"],
                }
                try:
                    row["error"], row["selected_rounds"], row["prediction_sha256"] = (
                        run_task(args.library, directory, case)
                    )
                except Exception as exc:
                    row["failure"] = f"{type(exc).__name__}: {exc}"
                    failed += 1
                print(json.dumps(row, allow_nan=False), file=report, flush=True)
    if failed:
        raise SystemExit(f"{failed} tasks failed")


if __name__ == "__main__":
    main()
