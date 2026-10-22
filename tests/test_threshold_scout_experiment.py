"""Synthetic integration of native prefix selection and fixed correction transfer."""

import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression"])
def test_train_correction_then_fixed_union(tmp_path, library, problem):
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    path = Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_threshold_scout.py"
    spec = importlib.util.spec_from_file_location("tested_threshold", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(73)
    values = rng.normal(size=80)
    raw = pd.DataFrame({"number": values, "category": ["a", "b", "c", "d"] * 20})
    labels = (values > 0).astype(int) if problem == "binary" else values * 2 + rng.normal(scale=.1, size=80)
    X, schema = module.base.normalize(raw.iloc[:60])
    Xs, _ = module.base.normalize(raw.iloc[60:], schema)
    if library == "ctboost":
        params = {"learning_rate": .1, "min_data_in_leaf": 3, "max_depth": 3, "alpha": .2}
    elif library == "xgboost":
        params = {"eta": .1, "max_depth": 3}
    else:
        params = {"learning_rate": .1, "num_leaves": 7, "min_data_in_leaf": 3}
    job = {"library": library, "parameters": params, "leaf_refit": None, "binary_refit": None, "temperature": None}
    job["binary_refit" if problem == "binary" else "leaf_refit"] = {
        "lambda": 10., "shrinkage": .25,
        "solver": ({"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6} if problem == "binary"
                   else {"atol": 1e-8, "btol": 1e-8, "iter_lim": 2000})}
    profiles = module.grid(job)
    assert len(profiles) == 4
    if library == "ctboost":
        assert {profile["alpha"] for profile in profiles} == {.5, .8}
    payload, details = module.base.fit_configuration(library, profiles[1],
        {"round_cap": 12, "patience": 3, "seed": 47}, {"problem_type": problem, "num_classes": 2},
        X, labels[:60], Xs, labels[60:], schema)
    corrected, correction = module.correct(payload, job, X, labels[:60], Xs)
    assert correction is not None
    assert np.isfinite(module.base.error(module.predict(corrected, Xs), labels[60:], problem))
    module.persist(corrected, tmp_path / "corrected.pkl", Xs)
    roles = {}
    for role, rows in (("train", np.arange(60)), ("stop", np.arange(60, 80))):
        role_path = tmp_path / (role + ".pkl")
        with role_path.open("wb") as stream:
            pickle.dump({"X": raw.iloc[rows].copy(), "y": labels[rows], "outer_train_positions": rows}, stream)
        roles[role] = {"path": str(role_path), "sha256": module.base.digest(role_path)}
    job.update(parameters=details["parameters"], rounds=details["retained_rounds"])
    union, union_details, frame = module.fit_policy(job, {"problem_type": problem, "roles": roles}, 947)
    assert union_details["requested_rounds"] == details["retained_rounds"]
    assert union_details["early_stopping"] is False
    assert union_details["union_data"]["union_rows"] == 80
    module.persist(union, tmp_path / "union.pkl", frame)
