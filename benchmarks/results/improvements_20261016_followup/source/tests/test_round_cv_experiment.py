"""Synthetic checks of fold-local pipelines and fixed model-length fitting."""

import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_round_cv.py"
    spec = importlib.util.spec_from_file_location("tested_round_cv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression", "multiclass"])
def test_fold_training_preserves_local_vocabulary_and_controls(tmp_path, library, problem):
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    module = runner()
    rng = np.random.default_rng(73)
    values = rng.normal(size=90)
    categories = ["rare"] + ["a", "b", "c"] * 29 + ["a", "b"]
    raw = pd.DataFrame({"value": values, "category": pd.Categorical(categories,
        categories=["a", "b", "c", "rare", "UNOBSERVED_GLOBAL"])})
    labels = (values * 2 if problem == "regression" else np.arange(90) % (3 if problem == "multiclass" else 2))
    case = {"dataset_name": "synthetic", "problem_type": problem, "num_classes": 3 if problem == "multiclass" else 2}
    manifest = module.make_folds(case, labels, np.arange(90), 13013)
    assignments = np.asarray(manifest["assignments"])
    held_fold = assignments[0]
    train, valid = np.flatnonzero(assignments != held_fold), np.flatnonzero(assignments == held_fold)
    if library == "ctboost":
        params = {"learning_rate": .1, "max_depth": 3, "min_data_in_leaf": 3, "alpha": .5,
                  "ordered_ctr": True, "verbose": False, "multi_strategy": "multi_output_tree" if problem == "multiclass" else "one_output_per_tree"}
    elif library == "xgboost":
        params = {"objective": {"binary": "binary:logistic", "regression": "reg:squarederror", "multiclass": "multi:softprob"}[problem],
                  "tree_method": "hist", "eta": .1, "max_depth": 3, "nthread": 1}
        if problem == "multiclass":
            params["num_class"] = 3
    else:
        params = {"objective": {"binary": "binary", "regression": "regression", "multiclass": "multiclass"}[problem],
                  "learning_rate": .1, "num_leaves": 7, "min_data_in_leaf": 3,
                  "num_threads": 1, "verbosity": -1, "deterministic": True, "force_col_wise": True}
        if problem == "multiclass":
            params["num_class"] = 3
    job = {"library": library, "parameters": params, "rounds": 4, "leaf_refit": None,
           "binary_refit": None, "temperature": .75 if problem == "multiclass" else None}
    if problem != "multiclass":
        job["binary_refit" if problem == "binary" else "leaf_refit"] = {"lambda": 10., "shrinkage": .25,
            "solver": ({"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6} if problem == "binary"
                       else {"atol": 1e-8, "btol": 1e-8, "iter_lim": 2000})}
    payload, details = module.fit_fold(job, case, raw.iloc[train], labels[train], raw.iloc[valid], 47)
    assert details["requested_rounds"] == 4
    assert details["early_stopping"] is False
    assert details["fold_category_schema"]["category"] == ["a", "b", "c"]
    predictions = module.predict(payload, raw.iloc[valid])
    assert np.isfinite(module.base.error(predictions, labels[valid], problem))
    restored = pickle.loads(pickle.dumps(payload, protocol=5))
    np.testing.assert_array_equal(predictions, module.predict(restored, raw.iloc[valid]))


def test_rarest_class_exactly_five_has_one_validation_member_per_fold():
    module = runner()
    labels = np.array([0] * 5 + [1] * 30 + [2] * 25)
    manifest = module.make_folds({"dataset_name": "rare", "problem_type": "multiclass", "num_classes": 3},
                                labels, np.arange(60), 13013)
    assert all(fold["train_class_counts"][0] == 4 and fold["validation_class_counts"][0] == 1 for fold in manifest["folds"])
    assert set(manifest["assignments"]) == set(range(5))


def test_missing_validation_class_is_rejected():
    module = runner()
    labels = np.array([0] * 4 + [1] * 30 + [2] * 26)
    with pytest.warns(UserWarning), pytest.raises(AssertionError):
        module.make_folds({"dataset_name": "too_rare", "problem_type": "multiclass", "num_classes": 3},
                          labels, np.arange(60), 13013)
