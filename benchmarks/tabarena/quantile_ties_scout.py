"""Measure default quantile cuts on the train, stopping and development roles of cached TabArena tasks."""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

STUDIES = {
    "accuracy-grouped-scout-v1": "d53e13021fb0d5d7f29353876c9f971ca3d56cb7857937c1b3c3afe436904a7b",
    "accuracy-remaining-scout-v1": "07cd7841988b2f32cc034470ef295fb54ad68d90c6dcf39a2b81105db1ac2e94",
}


def verified_bytes(path, expected_hash):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return data


def load_role(directory, case, role):
    path = directory / "data" / case["dataset_name"] / (role + ".pkl")
    return pickle.loads(verified_bytes(path, case["roles"][role]["sha256"]))


def affected_features(frame):
    import numpy as np
    import pandas as pd

    names = []
    for name, series in frame.items():
        if not pd.api.types.is_numeric_dtype(series.dtype):
            continue
        values = series.to_numpy(dtype=np.float32, na_value=np.nan)
        values = values[~np.isnan(values)]
        n = len(values)
        if n < 2:
            continue
        minimum = values.min()
        bins = min(256, n)
        last_cut_index = ((bins - 1) * n) // bins
        # Every sampled cut is the minimum, but larger observed values exist.
        if last_cut_index < np.count_nonzero(values == minimum) < n:
            names.append(str(name))
    return names


def fit(directory, case, seed):
    import pandas as pd

    import ctboost

    train = load_role(directory, case, "train")
    stop = load_role(directory, case, "stop")
    categorical = [
        name
        for name, dtype in train["X"].dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    ]
    problem = case["problem_type"]
    metric = {"binary": "Logloss", "multiclass": "MultiClass", "regression": "RMSE"}[
        problem
    ]
    model_cls = (
        ctboost.CTBoostRegressor
        if problem == "regression"
        else ctboost.CTBoostClassifier
    )
    model = model_cls(
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
        eval_metric=metric,
    )
    model.fit(
        train["X"],
        train["y"],
        eval_set=(stop["X"], stop["y"]),
        early_stopping_rounds=25,
    )
    return model


def score(model, directory, case):
    import numpy as np
    from sklearn.metrics import log_loss, roc_auc_score

    data = load_role(directory, case, "development")
    target = np.asarray(data["y"])
    if case["problem_type"] == "binary":
        prediction = model.predict_proba(data["X"])[:, 1]
        error = 1 - roc_auc_score(target, prediction)
    elif case["problem_type"] == "multiclass":
        prediction = model.predict_proba(data["X"])
        error = log_loss(target, prediction, labels=model.classes_)
    else:
        prediction = model.predict(data["X"])
        error = np.sqrt(np.mean((target.astype(np.float64) - prediction) ** 2))
    return {
        "dataset": case["dataset_name"],
        "n_development": len(target),
        "error": float(error),
        "best_iteration": model.get_best_iteration(),
        "predictions_sha256": hashlib.sha256(
            np.ascontiguousarray(prediction).tobytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Directory containing the two September 2026 scout caches",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=(0, 1),
        default=0,
        help="Learner seed (0 primary, 1 sensitivity)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Extend the scout beyond tasks with affected raw columns",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        help="Write one JSON line per completed task to a new file",
    )
    args = parser.parse_args()
    for variable in (
        "CTBOOST_HIST_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import ctboost

    audit = {}
    tasks = []
    for study, digest in STUDIES.items():
        directory = args.cache / study
        plan = json.loads(verified_bytes(directory / "plan.json", digest))
        for case in plan["datasets"]:
            name = case["dataset_name"]
            audit[name] = affected_features(load_role(directory, case, "train")["X"])
            if args.all_tasks or audit[name]:
                tasks.append((directory, case))

    native_sha = hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest()
    rows = []
    progress = args.progress.open("x", encoding="utf-8") if args.progress else None
    try:
        for directory, case in tasks:
            try:
                model = fit(directory, case, args.seed)
                row = score(model, directory, case)
            except Exception as exc:  # noqa: BLE001 - report failed tasks and continue the fixed panel
                row = {
                    "dataset": case["dataset_name"],
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            if progress:
                progress.write(
                    json.dumps(
                        {"native_sha256": native_sha, "seed": args.seed, **row},
                        allow_nan=False,
                    )
                    + "\n"
                )
                progress.flush()
    finally:
        if progress:
            progress.close()
    print(
        json.dumps(
            {
                "ctboost": ctboost.__version__,
                "native_sha256": native_sha,
                "seed": args.seed,
                "audit": audit,
                "development": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )
    if any("failure" in row for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
