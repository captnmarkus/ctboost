"""Synthetic fresh-refit parity guards; no benchmark models/data are loaded."""
import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from benchmarks import fixed_binary_leaf_refit as binary
from benchmarks import fixed_clean_refit_check as check
from benchmarks import fixed_leaf_refit as regression


def toy(library, problem, categorical=False):
    rng = np.random.default_rng(119)
    frame = pd.DataFrame({"x": rng.normal(size=96).astype(np.float32),
                          "z": rng.normal(size=96).astype(np.float32)})
    raw = 2 * frame.x.to_numpy() + frame.z.to_numpy()
    schema = {}
    if categorical:
        values = np.array(["a", "b", "c", "d"] * 24)
        frame["category"] = pd.Categorical(values, categories=["a", "b", "c", "d"])
        schema = {"category": ["a", "b", "c", "d"]}
        raw += np.tile([.1, -.2, .6, -.7], 24)
    labels = (raw > 0).astype(np.int32) if problem == "binary" else raw + rng.normal(0, .2, len(raw))
    if library == "ctboost":
        import ctboost
        model = ctboost.train(frame, {"objective": "Logloss" if problem == "binary" else "RMSE",
            "iterations": 4, "max_depth": 2, "alpha": 1., "min_data_in_leaf": 2,
            "learning_rate": .15, "random_seed": 119, "ordered_ctr": categorical,
            "cat_features": list(schema) or None, "verbose": False}, label=labels)
    elif library == "xgboost":
        xgb = pytest.importorskip("xgboost")
        model = xgb.train({"objective": "binary:logistic" if problem == "binary" else "reg:squarederror",
            "max_depth": 2, "eta": .15, "nthread": 1, "seed": 119, "tree_method": "hist"},
            xgb.DMatrix(frame, label=labels, enable_categorical=True), num_boost_round=4)
    else:
        lgb = pytest.importorskip("lightgbm")
        model = lgb.train({"objective": "binary" if problem == "binary" else "regression",
            "num_leaves": 4, "learning_rate": .15, "num_threads": 1, "deterministic": True,
            "force_col_wise": True, "min_data_in_leaf": 2, "seed": 119, "verbosity": -1},
            lgb.Dataset(frame, label=labels), num_boost_round=4)
    return {"library": library, "model": model, "problem": problem, "schema": schema}, frame, labels


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression"])
@pytest.mark.parametrize("categorical", [False, True])
def test_fresh_model_matches_corrected_parent_except_leaf_outputs(library, problem, categorical):
    original, frame, labels = toy(library, problem, categorical)
    helper = binary if problem == "binary" else regression
    design = helper.prepare_design(original, frame, labels)
    updates = [{int(leaf): .125 for leaf in leaves} for leaves in design["leaf_ids_by_tree"]]
    parent, _ = helper.fold_corrections(original, updates)
    fresh, _, _ = toy(library, problem, categorical)
    receipt = check.verify_clean_refit(fresh, parent)
    assert receipt["exact_clean_refit_identity_except_scalar_leaf_outputs"]
    assert receipt["clean_native_state_sha256"] != receipt["parent_native_state_sha256"]
    np.testing.assert_array_equal(helper.leaf_indices(fresh, frame), helper.leaf_indices(parent, frame))
    assert not np.array_equal(helper.predict(fresh, frame), helper.predict(parent, frame))
    restored = pickle.loads(pickle.dumps(parent, protocol=5))
    assert check.verify_clean_refit(fresh, restored) == receipt
    if library == "ctboost" and categorical:
        assert design["preprocessing_receipt"]["ordered_training_transform_replayed"]
        assert receipt["feature_pipeline_sha256"] is not None
    with pytest.raises(ValueError, match="inherited correction receipt"):
        check.verify_clean_refit(parent, parent)


@pytest.mark.parametrize("mutation", ["schema", "schema_order", "pipeline", "data_schema", "training_metadata", "bias", "cut", "rate"])
def test_ct_schema_pipeline_and_every_nonleaf_state_change_is_rejected(mutation):
    payload, _, _ = toy("ctboost", "regression", True)
    changed = pickle.loads(pickle.dumps(payload, protocol=5))
    model = changed["model"]
    if mutation == "schema":
        changed["schema"]["category"].reverse()
    elif mutation == "schema_order":
        payload["schema"] = {"first": ["a"], "second": ["b"]}
        changed["schema"] = {"second": ["b"], "first": ["a"]}
    elif mutation == "pipeline":
        state = copy.deepcopy(model._feature_pipeline.to_state())
        state["synthetic_change"] = 1
        model._feature_pipeline = SimpleNamespace(to_state=lambda: state)
    elif mutation == "data_schema":
        model._training_metadata["data_schema"]["synthetic_change"] = 1
    elif mutation == "training_metadata":
        model._training_metadata["synthetic_change"] = 1
    else:
        state = model._handle.export_state()
        if mutation == "bias":
            state["base_score"][0] += .1
        elif mutation == "rate":
            state["tree_learning_rates"][0] += .01
        else:
            branch = next(node for tree in state["trees"] for node in tree["nodes"] if not node["is_leaf"])
            branch["split_bin_index"] += 1
        model._handle = SimpleNamespace(export_state=lambda: state,
            prediction_dimension=lambda: 1)
    with pytest.raises(ValueError, match="differs from parent"):
        check.verify_clean_refit(changed, payload)


@pytest.mark.parametrize("library", ["xgboost", "lightgbm"])
def test_other_native_nonleaf_mutation_is_rejected(library, monkeypatch):
    payload, _, _ = toy(library, "regression")
    parent = pickle.loads(pickle.dumps(payload, protocol=5))
    if library == "xgboost":
        original = check.frozen._xgb_document
        def changed(model):
            state = original(model)
            if model is payload["model"]:
                state["learner"]["learner_model_param"]["base_score"] = "[0.25]"
            return state
        monkeypatch.setattr(check.frozen, "_xgb_document", changed)
    else:
        original = payload["model"].dump_model
        def changed(**kwargs):
            state = original(**kwargs)
            state["feature_names"][0] = "different_feature"
            return state
        monkeypatch.setattr(payload["model"], "dump_model", changed)
    with pytest.raises(ValueError, match="nonleaf_state"):
        check.verify_clean_refit(payload, parent)


def test_nan_safe_hash_is_stable_without_masking_other_values():
    first = {"nested": [float("nan"), float("inf"), -0., 1.]}
    second = json.loads(json.dumps(first))
    assert check._hash(first) == check._hash(second)
    second["nested"][-1] = 2.
    assert check._hash(first) != check._hash(second)


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("problem", ["binary", "regression"])
def test_scalar_preflight_bounds_actual_arrays_and_rejects_without_prediction(library, problem, monkeypatch):
    payload, frame, labels = toy(library, problem)
    helper = binary if problem == "binary" else regression
    receipt = check.memory_preflight(payload, len(frame), max_working_bytes=4 * 1024**3, solver_kind=problem)
    design = helper.prepare_design(payload, frame, labels)
    matrix = design["Z"]
    assert matrix.shape[1] <= receipt["parameters_upper_bound"]
    assert matrix.nnz == receipt["csr_nonzeros"]
    assert (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            <= receipt["array_budgets_bytes"]["resident_csr_data_indices_indptr"])
    assert sum(receipt["array_budgets_bytes"].values()) == receipt["refit_array_budget_bytes"]
    if problem == "binary":
        assert "lbfgsb_float_and_integer_work_arrays" in receipt["array_budgets_bytes"]
    else:
        assert "lsqr_residual_and_vector_workspace_budget" in receipt["array_budgets_bytes"]
    def forbidden(*args, **kwargs):
        raise AssertionError("The memory preflight must not predict leaf indices")
    monkeypatch.setattr(helper, "leaf_indices", forbidden)
    with pytest.raises(MemoryError, match="memory budget exceeded"):
        check.memory_preflight(payload, len(frame), max_working_bytes=1, solver_kind=problem)


def test_problem_temperature_and_solver_mismatch_fail():
    payload, _, _ = toy("ctboost", "regression")
    with pytest.raises(ValueError, match="without temperature"):
        check.verify_clean_refit(dict(payload, temperature=1.), payload)
    with pytest.raises(ValueError, match="matching native objective"):
        check.verify_clean_refit(dict(payload, problem="binary"), payload)
    with pytest.raises(ValueError, match="Invalid frozen"):
        check.memory_preflight(payload, 96, max_working_bytes=2**30, solver_kind="binary")
