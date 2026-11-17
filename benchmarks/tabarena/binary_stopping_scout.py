"""Compare log loss and ROC-AUC for binary checkpoint selection on cached TabArena data."""

import argparse
import hashlib
import json
import os
import pickle
import platform
from pathlib import Path

STUDIES = {
    "accuracy-grouped-scout-v1": "d53e13021fb0d5d7f29353876c9f971ca3d56cb7857937c1b3c3afe436904a7b",
    "accuracy-remaining-scout-v1": "07cd7841988b2f32cc034470ef295fb54ad68d90c6dcf39a2b81105db1ac2e94",
}
SEEDS = (0, 1)
METRICS = ("Logloss", "AUC")


def verified_bytes(path, expected_hash):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return content


def load_role(directory, case, role):
    path = directory / "data" / case["dataset_name"] / (role + ".pkl")
    return pickle.loads(verified_bytes(path, case["roles"][role]["sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Root of the trusted September 2026 scout caches",
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

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.metrics import roc_auc_score

    import ctboost

    cases = []
    for study, expected_hash in STUDIES.items():
        directory = args.cache / study
        plan = json.loads(verified_bytes(directory / "plan.json", expected_hash))
        for case in plan["datasets"]:
            if (
                case["problem_type"] == "binary"
                and case["original_rows"] <= 2000
                and case["num_features"] <= 128
            ):
                cases.append((directory, case))

    fitted = {}
    for directory, case in cases:
        train = load_role(directory, case, "train")
        stop = load_role(directory, case, "stop")
        categorical = [
            name
            for name, dtype in train["X"].dtypes.items()
            if isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        ]
        for seed in SEEDS:
            for metric in METRICS:
                model = ctboost.CTBoostClassifier(
                    iterations=1000,
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
                    verbose=False,
                )
                model.fit(
                    train["X"],
                    train["y"],
                    eval_set=(stop["X"], stop["y"]),
                    early_stopping_rounds=50,
                )
                fitted[case["dataset_name"], seed, metric] = model

    rows = []
    for directory, case in cases:
        name = case["dataset_name"]
        development = load_role(directory, case, "development")
        row = {"dataset": name, "n_development": len(development["y"]), "metrics": {}}
        for metric in METRICS:
            scores = []
            for seed in SEEDS:
                model = fitted[name, seed, metric]
                probabilities = model.predict_proba(development["X"])[:, 1]
                error = 1 - roc_auc_score(development["y"], probabilities)
                scores.append(
                    {
                        "seed": seed,
                        "best_iteration": model.get_best_iteration(),
                        "error": error,
                    }
                )
            row["metrics"][metric] = {
                "fits": scores,
                "mean_error": np.mean([s["error"] for s in scores]),
            }
        rows.append(row)

    baseline = np.array([row["metrics"]["Logloss"]["mean_error"] for row in rows])
    candidate = np.array([row["metrics"]["AUC"]["mean_error"] for row in rows])
    improvement = baseline - candidate
    print(
        json.dumps(
            {
                "python": platform.python_version(),
                "ctboost": ctboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "native_sha256": hashlib.sha256(
                    Path(ctboost._core.__file__).read_bytes()
                ).hexdigest(),
                "tasks": rows,
                "mean_logloss_policy_error": np.mean(baseline),
                "mean_auc_policy_error": np.mean(candidate),
                "median_relative_error_reduction": np.median(improvement / baseline),
                "auc_wins": int((improvement > 1e-12).sum()),
                "ties": int((np.abs(improvement) <= 1e-12).sum()),
                "auc_losses": int((improvement < -1e-12).sum()),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
