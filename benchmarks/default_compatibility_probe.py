"""Emit deterministic hashes for cross-checkout default-path regression tests.

Run this file with the same Python/NumPy stack from an unmodified baseline
checkout and from the candidate checkout, then compare the JSON output. Model
metadata is deliberately excluded because candidates may add backward-compatible
state keys; raw predictions and canonical tree JSON are the compatibility gate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

import numpy as np

import ctboost


def _hash_model(model: object, prediction_input: object) -> Dict[str, object]:
    predictions = np.asarray(model.predict(prediction_input), dtype="<f4")
    trees = model._handle.export_state()["trees"]
    tree_json = json.dumps(
        trees, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "tree_json_sha256": hashlib.sha256(tree_json).hexdigest(),
        "tree_count": len(trees),
    }


def _numeric_case() -> Tuple[object, object]:
    rng = np.random.default_rng(1101)
    X = rng.normal(size=(512, 6)).astype(np.float32)
    y = (
        1.3 * X[:, 0]
        - 0.7 * X[:, 2]
        + 0.4 * np.square(X[:, 4])
        + 0.08 * rng.normal(size=X.shape[0])
    ).astype(np.float32)
    params = {
        "objective": "RMSE",
        "learning_rate": 0.13,
        "max_depth": 3,
        "alpha": 0.2,
        "max_bins": 255,
        "random_seed": 71,
    }
    model = ctboost.train(X, params, label=y, num_boost_round=6)
    return model, X


def _categorical_missing_case() -> Tuple[object, object]:
    rng = np.random.default_rng(1102)
    category = rng.integers(0, 7, size=640).astype(np.float32)
    numeric = rng.normal(size=640).astype(np.float32)
    missing = np.arange(640) % 11 == 0
    numeric[missing] = np.nan
    noise = rng.normal(size=640).astype(np.float32)
    X = np.column_stack((category, numeric, noise)).astype(np.float32)
    y = (
        np.isin(category, [1.0, 4.0, 6.0]).astype(np.float32)
        + 0.8 * missing.astype(np.float32)
        + 0.1 * noise
    ).astype(np.float32)
    pool = ctboost.Pool(X, y, cat_features=[0])
    params = {
        "objective": "RMSE",
        "learning_rate": 0.2,
        "max_depth": 3,
        "alpha": 0.15,
        "max_bins": 96,
        "nan_mode": "Max",
        "random_seed": 73,
    }
    model = ctboost.train(pool, params, num_boost_round=5)
    return model, pool


def _constrained_leafwise_case() -> Tuple[object, object]:
    rng = np.random.default_rng(1103)
    X = rng.normal(size=(720, 5)).astype(np.float32)
    y = (
        1.1 * X[:, 0]
        + 0.5 * X[:, 1]
        - 0.3 * X[:, 3]
        + 0.1 * rng.normal(size=X.shape[0])
    ).astype(np.float32)
    params = {
        "objective": "RMSE",
        "learning_rate": 0.1,
        "max_depth": 4,
        "grow_policy": "LeafWise",
        "max_leaves": 7,
        "monotone_constraints": [1, 0, 0, 0, 0],
        "interaction_constraints": [[0, 1, 2], [0, 3, 4]],
        "feature_weights": [1.0, 0.9, 1.1, 1.0, 0.95],
        "first_feature_use_penalties": [0.0, 0.01, 0.0, 0.02, 0.0],
        "random_strength": 0.03,
        "alpha": 0.25,
        "max_bins": 128,
        "random_seed": 79,
    }
    model = ctboost.train(X, params, label=y, num_boost_round=5)
    return model, X


def main() -> int:
    cases = {
        "numeric": _numeric_case(),
        "categorical_missing": _categorical_missing_case(),
        "constraints_leafwise": _constrained_leafwise_case(),
    }
    report = {
        name: _hash_model(model, prediction_input)
        for name, (model, prediction_input) in cases.items()
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
