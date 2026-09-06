"""Fixed development validation of CTBoost's opt-in learning options.

Run: python -m benchmarks.learning_options --output learning-options.json

Uses only bundled sklearn datasets, three predeclared stratified 75/25 splits,
and every predeclared variant. No early stopping, tuning, or winner selection.
These small local validation measurements are not new TabArena evidence.
Leaf solvers safeguard the weighted, L2-regularized training objective at
unshrunk leaf increments; this does not guarantee better validation loss.
"""

import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split

import ctboost

SEEDS = (0, 1, 2)
COMMON_PARAMS = {
    "iterations": 40,
    "learning_rate": 0.1,
    "max_depth": 3,
    "max_bins": 64,
    "alpha": 0.05,
    "lambda_l2": 1.0,
    "task_type": "CPU",
    "boost_from_average": True,
}
JOINT = {"multiclass_feature_test": "joint", "feature_test": "grouped", "feature_test_bins": 8}
FULL = {"multiclass_leaf_solver": "full", "leaf_estimation_iterations": 3}


def _objective_loss(margins, labels):
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim == 1:
        return float(np.mean(np.logaddexp(0.0, margins) - labels * margins))
    shifted = margins - margins.max(axis=1, keepdims=True)
    return float(np.mean(
        np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels.astype(int)]
    ))


def run_comparison():
    results = []
    datasets = {}
    for name, loader in (
        ("breast_cancer", load_breast_cancer),
        ("iris", load_iris),
        ("wine", load_wine),
        ("digits", load_digits),
    ):
        data = loader()
        X = np.asarray(data.data, dtype=np.float32)
        labels = np.asarray(data.target, dtype=np.float32)
        classes = len(np.unique(labels))
        data_digest = hashlib.sha256()
        data_digest.update(json.dumps({"shape": X.shape, "dtype": "float32"}, sort_keys=True).encode())
        data_digest.update(X.tobytes(order="C"))
        data_digest.update(labels.tobytes(order="C"))
        datasets[name] = {"rows": len(X), "features": X.shape[1], "classes": classes,
                          "data_sha256": data_digest.hexdigest()}
        variants = (
            {"baseline": {}, "backtracking_3": {
                "leaf_estimation_backtracking": True, "leaf_estimation_iterations": 3,
            }}
            if classes == 2
            else {"baseline": {}, "full_3": FULL, "joint_grouped_8": JOINT,
                  "full_3_joint_grouped_8": {**FULL, **JOINT}}
        )
        for seed in SEEDS:
            train_indices, valid_indices = train_test_split(
                np.arange(len(labels)), test_size=0.25, random_state=seed, stratify=labels,
            )
            X_train, X_valid = X[train_indices], X[valid_indices]
            y_train, y_valid = labels[train_indices], labels[valid_indices]
            pool = ctboost.Pool(X_train, y_train)
            for variant, controls in variants.items():
                params = {
                    **COMMON_PARAMS,
                    "objective": "Logloss" if classes == 2 else "MultiClass",
                    "num_classes": classes,
                    "random_seed": seed,
                    **controls,
                }
                started = time.perf_counter()
                model = ctboost.train(pool, params)
                train_seconds = time.perf_counter() - started
                prediction = model.predict(X_valid)
                predict_timings = []
                for _ in range(3):
                    started = time.perf_counter()
                    model.predict(X_valid)
                    predict_timings.append(time.perf_counter() - started)
                state = model._handle.export_state()
                nodes = sum(len(tree["nodes"]) for tree in state["trees"])
                record = {
                    "dataset": name, "seed": seed, "variant": variant,
                    "train_rows": len(train_indices), "validation_rows": len(valid_indices),
                    "params": params,
                    "train_objective_loss": _objective_loss(model.predict(X_train), y_train),
                    "validation_objective_loss": _objective_loss(prediction, y_valid),
                    "train_seconds": train_seconds,
                    "predict_seconds_median": float(np.median(predict_timings)),
                    "physical_trees": len(state["trees"]), "physical_nodes": nodes,
                    "shared_structure_nodes": nodes if classes == 2 else nodes // classes,
                }
                results.append(record)
                print(
                    f"{name} seed={seed} {variant}: "
                    f"validation_loss={record['validation_objective_loss']:.6f}, "
                    f"train={train_seconds:.3f}s, nodes={nodes}", flush=True,
                )
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["dataset"], result["variant"])].append(result)
    summary = []
    for (dataset, variant), records in grouped.items():
        summary.append({
            "dataset": dataset, "variant": variant, "splits": len(records),
            **{
                f"mean_{key}": float(np.mean([record[key] for record in records]))
                for key in ("train_objective_loss", "validation_objective_loss", "train_seconds",
                            "predict_seconds_median", "physical_nodes", "shared_structure_nodes")
            },
        })
    return {
        "purpose": "Fixed development validation; no tuning, selection, or new TabArena evidence.",
        "leaf_loss_contract": "Weighted training objective plus L2 at unshrunk leaf increments.",
        "split": {"seeds": SEEDS, "validation_fraction": 0.25, "stratified": True},
        "datasets": datasets,
        "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "ctboost": ctboost.__version__, "sklearn": sklearn.__version__,
                        "numpy": np.__version__,
                        "native_module_sha256": hashlib.sha256(
                            Path(ctboost._core.__file__).read_bytes()
                        ).hexdigest()},
        "results": results, "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_comparison()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved all {len(report['results'])} fixed comparisons to {args.output}")


if __name__ == "__main__":
    main()
