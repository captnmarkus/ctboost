"""Synthetic integrity checks for the experimental mandatory union pipeline."""

import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_union_refit.py"
    spec = importlib.util.spec_from_file_location("tested_union", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression", "multiclass"])
def test_fixed_union_pipeline(tmp_path, library, problem):
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    module = runner()
    rng = np.random.default_rng(901)
    values = rng.normal(size=60)
    frame = pd.DataFrame({"value": values, "category": ["a", "b", "c"] * 16 + ["late", "a", "b"] * 4})
    labels = (2.0 * values + rng.normal(scale=.1, size=60) if problem == "regression" else
              (values > 0).astype(int) if problem == "binary" else np.digitize(values, [-.4, .4]))
    roles = {}
    for name, rows in (("train", np.arange(48)), ("stop", np.arange(48, 60))):
        path = tmp_path / (name + ".pkl")
        with path.open("wb") as stream:
            pickle.dump({"X": frame.iloc[rows].copy(), "y": labels[rows], "outer_train_positions": rows}, stream)
        roles[name] = {"path": str(path), "sha256": module.base.digest(path)}
    case = {"dataset_name": "synthetic_union", "problem_type": problem, "roles": roles}
    if library == "ctboost":
        params = {"iterations": 99, "learning_rate": .1, "max_depth": 3, "min_data_in_leaf": 3,
                  "alpha": .5, "lambda_l2": 1, "ordered_ctr": True, "verbose": False,
                  "random_seed": 47, "multi_strategy": "multi_output_tree" if problem == "multiclass" else "one_output_per_tree"}
    elif library == "xgboost":
        params = {"objective": {"binary": "binary:logistic", "regression": "reg:squarederror", "multiclass": "multi:softprob"}[problem],
                  "tree_method": "hist", "nthread": 1, "max_depth": 3, "eta": .1, "seed": 47}
        if problem == "multiclass":
            params["num_class"] = 3
    else:
        params = {"objective": {"binary": "binary", "regression": "regression", "multiclass": "multiclass"}[problem],
                  "num_threads": 1, "num_leaves": 7, "min_data_in_leaf": 3, "learning_rate": .1,
                  "verbosity": -1, "seed": 47, "deterministic": True, "force_col_wise": True}
        if problem == "multiclass":
            params["num_class"] = 3
    correction = {"lambda": 10., "shrinkage": .25, "solver": {"atol": 1e-8, "btol": 1e-8, "iter_lim": 2000}}
    job = {"library": library, "parameters": params, "rounds": 6,
           "leaf_refit": correction if problem == "regression" else None,
           "temperature": .75 if problem == "multiclass" else None}
    payload, details, verification_frame = module.fit_policy(job, case, 947)
    assert "late" in payload["schema"]["category"]
    assert details["union_data"]["union_rows"] == 60
    assert details["union_data"]["train_rows"] == 48
    assert details["union_data"]["stop_rows"] == 12
    assert details["requested_rounds"] == 6
    assert not details["early_stopping"]
    assert details["parameters"]["random_seed" if library == "ctboost" else "seed"] == 947
    restored = pickle.loads(pickle.dumps(payload, protocol=5))
    np.testing.assert_array_equal(module.predict(payload, verification_frame), module.predict(restored, verification_frame))
    if problem == "regression":
        assert details["fixed_leaf_refit"]["ordered_union_check"]["leaf_routing_exact"]
        if library == "ctboost":
            assert details["fixed_leaf_refit"]["preprocessing"]["ordered_training_transform_replayed"]
    if problem == "multiclass":
        raw = module.raw_predict(payload, verification_frame)
        np.testing.assert_array_equal(module.predict(payload, verification_frame), raw.astype(np.float64) / .75)


def test_union_rejects_overlap(tmp_path):
    module = runner()
    roles = {}
    for name in ("train", "stop"):
        path = tmp_path / (name + ".pkl")
        with path.open("wb") as stream:
            pickle.dump({"X": pd.DataFrame({"a": [1.]}), "y": np.array([1.]), "outer_train_positions": np.array([0])}, stream)
        roles[name] = {"path": str(path), "sha256": module.base.digest(path)}
    with pytest.raises(ValueError, match="overlap"):
        module.union_data({"roles": roles})
