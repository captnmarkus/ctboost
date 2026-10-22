"""Bounded native toy integration; no real benchmark files or quality gates.

Run with the immutable c5 accuracy package and XGB3.4.1/LGB4.7.0 stack. All fits
below are synthetic, four-round fits, separately counted from any experiment.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("current_cv_test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = load("fair_accuracy_current_round_cv_support")
adapter = load("fair_accuracy_current_round_cv_backend")
scalar = load("fair_accuracy_scalar_numerical_retry")
multiclass = load("fair_accuracy_multiclass_refit")
NAMED_ARRAY_BYTES = 64 * 1024**2  # Toy-test cap only; not a prospective experiment budget.
CASES = [(library, problem, False) for library in support.LIBRARIES for problem in ("regression", "binary")]
CASES += [(library, "multiclass", vector) for library, vector in
          (("ctboost", False), ("ctboost", True), ("xgboost", False), ("xgboost", True), ("lightgbm", False))]


def toy(library, problem, vector, active):
    rng = np.random.default_rng(293)
    frame = pd.DataFrame({"x": rng.normal(size=108), "z": rng.normal(size=108)})
    frame["cat"] = pd.Categorical(np.resize(["a", "b", "c", "d"], 108),
                                  categories=["a", "b", "c", "d", "heldout", "unused"])
    response = 2 * frame.x.to_numpy() - frame.z.to_numpy() + np.resize([0., .5, -.25, .8], 108)
    if problem == "multiclass":
        labels = np.argmax(np.column_stack([response, frame.z, -response]), axis=1).astype(np.int32)
    elif problem == "binary":
        labels = (response > 0).astype(np.int32)
    else:
        labels = response + .1 * rng.normal(size=len(frame))
    train, valid = frame.iloc[:90].copy(), frame.iloc[90:].copy()
    valid["cat"] = pd.Categorical(["heldout"] * len(valid), categories=frame.cat.cat.categories)
    case = {"dataset_name": "synthetic", "problem_type": problem, "num_classes": 3 if problem == "multiclass" else 2}
    if library == "ctboost":
        params = {"iterations": 4, "max_depth": 2, "alpha": 1., "min_data_in_leaf": 2,
            "learning_rate": .15, "random_seed": 47, "ordered_ctr": True, "one_hot_max_size": 1,
            "multi_strategy": "multi_output_tree" if vector else "one_output_per_tree", "verbose": False}
    elif library == "xgboost":
        params = {"objective": {"regression": "reg:squarederror", "binary": "binary:logistic", "multiclass": "multi:softprob"}[problem],
            "max_depth": 2, "eta": .15, "nthread": 1, "seed": 47, "tree_method": "hist",
            "multi_strategy": "multi_output_tree" if vector else "one_output_per_tree"}
        if problem == "multiclass":
            params["num_class"] = 3
    else:
        params = {"objective": problem, "num_leaves": 4, "max_depth": 2, "learning_rate": .15, "num_threads": 1,
            "deterministic": True, "force_col_wise": True, "min_data_in_leaf": 2, "seed": 47, "verbosity": -1}
        if problem == "multiclass":
            params["num_class"] = 3
    solver = ({"variant": "unscaled_lsqr_10000", "iterations": 10000} if problem == "regression" else
              {"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6})
    job = {"dataset": "synthetic", "library": library, "parameters": params, "rounds": 4,
           "temperature": .65 if problem == "multiclass" else None,
           **{field: None for field in support.CORRECTION_FIELDS}}
    if active:
        # Deliberately smaller inherited memory cap proves only that field is overridden.
        job[support.CORRECTION_FIELD[problem]] = {"lambda": 10., "shrinkage": .25,
            "solver": solver, "max_working_bytes": 1}
    return case, job, train, labels[:90], valid, labels[90:]


@pytest.mark.parametrize("library,problem,vector", CASES)
@pytest.mark.parametrize("active", [False, True])
def test_native_fold_local_fixed_policy_oracles_and_original_metric(library, problem, vector, active):
    from scipy.special import logsumexp
    from sklearn.metrics import roc_auc_score
    case, job, train, y, valid, yv = toy(library, problem, vector, active)
    original = copy.deepcopy(job)
    backend = adapter.make_backend(scalar=scalar, multiclass=multiclass, max_working_bytes=NAMED_ARRAY_BYTES)
    fitted, fit_clean = [], backend.fit_clean
    def capture(*args):
        payload, details = fit_clean(*args)
        fitted.append(payload)
        return payload, details
    backend.fit_clean = capture
    candidate, predictions, receipt = support.fit_fold(job, case, train, y, valid, 47, backend)
    assert len(fitted) == 1
    clean = fitted[0]
    assert receipt["native_cv_fresh_fit"] and receipt["fixed_corrections_disabled_before_fit"]
    assert receipt["requested_rounds"] == 4 and 0 < receipt["retained_rounds"] <= 4
    assert receipt["reload_bit_exact"] and receipt["validation_labels_received"] is False
    assert receipt["fold_category_schema"] == {"cat": ["a", "b", "c", "d"]}
    assert candidate["schema"] == clean["schema"] and job == original
    X, _ = backend.normalize(train, candidate["schema"])
    Xv, _ = backend.normalize(valid, candidate["schema"])
    assert Xv.cat.isna().all()
    h = multiclass.helper() if problem == "multiclass" else scalar.helper(problem)
    for frame in (X, Xv):
        np.testing.assert_array_equal(h.leaf_indices(clean, frame), h.leaf_indices(candidate, frame))
    if problem == "multiclass":
        np.testing.assert_array_equal(predictions, h.predict_raw(candidate, Xv) / job["temperature"])
        classes, supports, _ = h._metadata(candidate)
        assert classes == 3 and (all(value == -1 for value in supports) == vector)
        expected_error = np.mean(logsumexp(predictions, axis=1) - predictions[np.arange(len(yv)), yv])
    elif problem == "binary":
        expected_error = 1 - roc_auc_score(yv, predictions)
    else:
        expected_error = np.sqrt(np.mean((predictions - yv) ** 2))
    assert backend.score(predictions, yv, problem) == pytest.approx(expected_error, rel=1e-13, abs=1e-13)
    if active:
        evidence = receipt["correction"]
        assert evidence["independent_sparse_raw_margin_oracle"]
        assert evidence["fold"]["topology_sha256_before"] == evidence["fold"]["topology_sha256_after"]
        assert set(evidence["checks"]) == {"fitting", "validation"}
        resource = evidence["native_cv_adapter"]
        assert resource["resource_override_only"] and resource["effective_named_array_limit"] == NAMED_ARRAY_BYTES
        assert resource["inherited_named_array_limit"] == 1
        if library == "ctboost":
            assert evidence["preprocessing"]["ordered_training_transform_replayed"]
            assert evidence["preprocessing"]["ordered_ctr"]
        if problem == "regression":
            assert evidence["solve"]["success"] and evidence["solve"]["certificate"]["certified"]
        # Fresh metadata guard prevents applying a second fixed correction.
        with pytest.raises(ValueError, match="fresh payload"):
            backend.apply_fixed_correction(candidate, problem, job[support.CORRECTION_FIELD[problem]], X, y, {"validation": Xv})
    else:
        assert receipt["correction"] is None
        assert not any(field in candidate for field in adapter.RECEIPT_FIELDS)
        clean_for_prediction = {**clean, "temperature": job["temperature"]} if problem == "multiclass" else clean
        np.testing.assert_array_equal(predictions, backend.predict(clean_for_prediction, valid))


@pytest.mark.parametrize("field", [*adapter.RECEIPT_FIELDS, *adapter.CORRECTION_FIELDS])
def test_correction_metadata_is_rejected_even_when_empty(field):
    backend = adapter.make_backend(scalar=scalar, multiclass=multiclass, max_working_bytes=NAMED_ARRAY_BYTES)
    with pytest.raises(ValueError, match="fresh payload"):
        backend.apply_fixed_correction({"problem": "regression", field: None}, "regression", {}, None, None, {})


def test_clean_fit_rejects_inherited_controls_without_calling_native_fit():
    def forbidden(*args):
        pytest.fail("A native fit must not start for contaminated controls")
    mock = SimpleNamespace(base=scalar.base, predict=scalar.predict, union=SimpleNamespace(fit_fixed=forbidden))
    backend = adapter.make_backend(scalar=mock, multiclass=multiclass, max_working_bytes=NAMED_ARRAY_BYTES)
    for field in (*adapter.CORRECTION_FIELDS, "temperature"):
        with pytest.raises(ValueError, match="Disable every"):
            backend.fit_clean({field: {"inherited": True}}, {}, None, None, {}, 47)


def test_callback_cannot_mutate_frozen_algorithm_controls():
    def mutation(payload, X, y, controls, frames):
        controls["lambda"] = 1000
        return {**payload, "fixed_leaf_refit": {}}, {}
    mock = SimpleNamespace(base=scalar.base, predict=scalar.predict, corrected=mutation)
    backend = adapter.make_backend(scalar=mock, multiclass=multiclass, max_working_bytes=NAMED_ARRAY_BYTES)
    controls = {"lambda": 10., "shrinkage": .25, "solver": {"fixed": True}}
    with pytest.raises(ValueError, match="changed frozen algorithm"):
        backend.apply_fixed_correction({"problem": "regression"}, "regression", controls, None, None, {})
    assert controls["lambda"] == 10.
