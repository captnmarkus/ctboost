"""Compare deterministic model outputs with a separately installed release wheel.

Run with python -I from each isolated environment. New serialization keys are
ignored, while every field present in the baseline trees remains mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CTBOOST_HIST_THREADS", "1")

import numpy as np

import ctboost


def array_record(value):
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": array.tobytes().hex(),
    }


def project_tree(value, reference):
    if isinstance(reference, dict):
        return {
            key: project_tree(value[key], child) for key, child in reference.items()
        }
    if isinstance(reference, list):
        if len(value) != len(reference):
            raise ValueError("Tree/list length changed")
        return [project_tree(child, old) for child, old in zip(value, reference)]
    return value


def run_cases():
    records = {}
    for objective, strategy in [
        ("RMSE", "one_output_per_tree"),
        ("Logloss", "one_output_per_tree"),
        ("MultiClass", "one_output_per_tree"),
        ("MultiClass", "multi_output_tree"),
    ]:
        for feature_test in ("quadratic", "grouped"):
            for variant in (
                "default",
                "weighted_missing_categorical",
                "leafwise",
                "dart",
            ):
                rng = np.random.default_rng(20260907)
                X = rng.normal(size=(192, 7)).astype(np.float32)
                regression = 1.7 * X[:, 0] - 0.8 * X[:, 1] + 0.4 * X[:, 2] ** 2
                regression += rng.normal(scale=0.2, size=len(X))
                labels = (
                    regression
                    if objective == "RMSE"
                    else (regression > 0).astype(np.float32)
                    if objective == "Logloss"
                    else np.argmax(
                        np.column_stack((X[:, 0], X[:, 1], -X[:, 0])), axis=1
                    )
                )
                pool_kwargs = {}
                params = {
                    "objective": objective,
                    "multi_strategy": strategy,
                    "feature_test": feature_test,
                    "feature_test_bins": 8,
                    "iterations": 12,
                    "max_depth": 3,
                    "max_bins": 64,
                    "alpha": 0.05,
                    "learning_rate": 0.12,
                    "random_seed": 19,
                }
                if objective == "MultiClass":
                    params["num_classes"] = 3
                if variant == "weighted_missing_categorical":
                    X[:, 4] = np.round(X[:, 4])
                    X[::7, 2] = np.nan
                    pool_kwargs = {
                        "cat_features": [4],
                        "weight": np.linspace(0.2, 2.0, len(X), dtype=np.float32),
                    }
                    params["nan_mode"] = "Max"
                elif variant == "leafwise":
                    params.update(grow_policy="LeafWise", max_leaves=5)
                elif variant == "dart":
                    params.update(
                        boosting_type="DART", drop_rate=0.3, skip_drop=0.0, max_drop=2
                    )
                pool = ctboost.Pool(X, labels.astype(np.float32), **pool_kwargs)
                model = ctboost.train(pool, params)
                state = model._handle.export_state()
                name = f"{objective}/{strategy}/{feature_test}/{variant}"
                records[name] = {
                    "params": params,
                    "predictions": array_record(model.predict(pool)),
                    "leaf_indices": array_record(model.predict_leaf_index(pool)),
                    "trees": state["trees"],
                    "loss_history": array_record(model.loss_history),
                }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    cases = run_cases()
    report = {
        "version": ctboost.__version__,
        "package_path": str(Path(ctboost.__file__).resolve()),
        "build_info": ctboost.build_info(),
        "numpy_version": np.__version__,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        mismatches = []
        for name, previous in baseline["cases"].items():
            current = cases[name]
            for key in ("predictions", "leaf_indices", "loss_history"):
                if current[key] != previous[key]:
                    mismatches.append({"case": name, "field": key})
            try:
                equal = (
                    project_tree(current["trees"], previous["trees"])
                    == previous["trees"]
                )
            except (KeyError, ValueError):
                equal = False
            if not equal:
                mismatches.append({"case": name, "field": "trees"})
        report["comparison"] = {
            "baseline_version": baseline["version"],
            "case_count": len(cases),
            "mismatches": mismatches,
            "all_exact": not mismatches,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            report.get(
                "comparison", {"version": report["version"], "cases": len(cases)}
            )
        )
    )


if __name__ == "__main__":
    main()
