"""Core-only parity check for the locally rebuilt CPython 3.12 extension."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

for name in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    native_path = args.native.resolve()
    specification = importlib.util.spec_from_file_location("_core", native_path)
    native = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(native)
    info = native.build_info()
    assert info["version"] == "0.1.61", info
    assert not info["cuda_enabled"], info
    rng = np.random.default_rng(20261016)
    train = rng.normal(size=(180, 8)).astype(np.float32)
    regression = (train[:, 0] - 0.8 * train[:, 1] + train[:, 2] * train[:, 3]).astype(np.float32)
    binary = (regression > 0).astype(np.float32)
    multiclass = train[:, :3].argmax(axis=1).astype(np.float32)
    train[::19, 2] = np.nan
    cases = []
    for name, objective, labels, extra in (
        ("regression", "RMSE", regression, {}),
        ("binary_classification", "Logloss", binary, {}),
        ("vector_multiclass", "MultiClass", multiclass,
         {"num_classes": 3, "multi_strategy": "multi_output_tree", "multiclass_leaf_solver": "full",
          "leaf_estimation_iterations": 3}),
    ):
        model = native.GradientBooster(objective=objective, iterations=16, max_depth=3,
                                      max_bins=32, learning_rate=0.075, random_seed=47, **extra)
        model.fit(native.Pool(train, labels))
        state = model.export_state()
        restored = native.GradientBooster.from_state(state)
        dimension = len(state["base_score"])
        per_round = 1 if state["multi_strategy"] == "multi_output_tree" else dimension
        checks = 0
        for count in (0, 1, 64):
            data = train[:count].copy()
            baseline = np.linspace(-0.2, 0.3, count * dimension, dtype=np.float32).reshape(count, dimension)
            pool = native.Pool(data, np.empty(0, dtype=np.float32), baseline=baseline)
            for prefix in (0, 1, 7, -1, 99):
                reference = np.asarray(model._predict_uncached(pool, prefix), dtype=np.float32)
                for _ in range(2):
                    actual = np.asarray(model.predict(pool, prefix), dtype=np.float32)
                    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
                    checks += 1
                restored_prediction = np.asarray(restored.predict(pool, prefix), dtype=np.float32)
                np.testing.assert_array_equal(restored_prediction.view(np.uint32), reference.view(np.uint32))
                leaves = np.asarray(model.predict_leaf_indices(pool, prefix))
                np.testing.assert_array_equal(leaves, restored.predict_leaf_indices(pool, prefix))
                tree_limit = len(state["trees"]) if prefix < 0 else min(len(state["trees"]), prefix * per_round)
                assert leaves.size == count * tree_limit
                assert actual.size == count * dimension
        assert model.export_state() == state
        cases.append({"case": name, "trees": len(state["trees"]), "dimension": dimension,
                      "cached_uncached_bit_parity_checks": checks, "state_reload_parity": True,
                      "leaf_shape_and_reload_parity": True, "model_state_unchanged": True})
    result = {"python": sys.executable, "python_version": platform.python_version(),
              "numpy_version": np.__version__, "native_path": str(native_path),
              "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "build_info": info, "cases": cases, "total_cached_uncached_bit_checks": sum(c["cached_uncached_bit_parity_checks"] for c in cases),
              "scope": "Direct _core import; regression, binary and vector multiclass; rows0/1/64, prefixes0/1/7/all/99, nonzero baselines, missing values, state restoration, leaves. No Python wrapper or sklearn import."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
