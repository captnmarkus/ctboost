"""Tiny synthetic native models only; never loads a benchmark model or dataset."""
import copy
import pickle

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from benchmarks import fixed_multiclass_leaf_refit as refit


def toy(library, vector=False, categorical=False):
    rng = np.random.default_rng(811)
    frame = pd.DataFrame({"x": rng.normal(size=90), "z": rng.normal(size=90)})
    labels = np.argmax(np.column_stack([frame.x - frame.z, frame.z, -frame.x]), axis=1).astype(np.int32)
    if categorical:
        frame["cat"] = pd.Categorical(np.array(["a", "b", "c", "d", "e"] * 18))
        labels[::5] = 1
    if library == "ctboost":
        import ctboost
        model = ctboost.train(frame, {"objective": "MultiClass", "num_classes": 3,
            "iterations": 4, "max_depth": 2, "min_data_in_leaf": 2, "alpha": 1.,
            "learning_rate": .17,
            "ordered_ctr": categorical, "cat_features": ["cat"] if categorical else None,
            "random_seed": 811, "verbose": False,
            "multi_strategy": "multi_output_tree" if vector else "one_output_per_tree"}, label=labels,
            learning_rate_schedule=[.17, .11, .08, .06])
    elif library == "xgboost":
        xgb = pytest.importorskip("xgboost")
        if xgb.__version__ != "3.4.1":
            pytest.skip("Native serialization tests require the frozen XGBoost3.4.1 runtime")
        model = xgb.train({"objective": "multi:softprob", "num_class": 3,
            "max_depth": 2, "eta": .17, "nthread": 1, "tree_method": "hist", "seed": 811,
            "multi_strategy": "multi_output_tree" if vector else "one_output_per_tree"},
            xgb.DMatrix(frame, label=labels, enable_categorical=True), num_boost_round=4)
    else:
        lgb = pytest.importorskip("lightgbm")
        model = lgb.train({"objective": "multiclass", "num_class": 3,
            "num_leaves": 4, "learning_rate": .17, "num_threads": 1,
            "min_data_in_leaf": 2, "verbosity": -1, "seed": 811},
            lgb.Dataset(frame, label=labels), num_boost_round=4)
    return {"library": library, "model": model, "problem": "multiclass", "schema": {},
            "temperature": .65}, frame, labels


CASES = [("ctboost", False), ("ctboost", True), ("xgboost", False),
         ("xgboost", True), ("lightgbm", False)]


@pytest.mark.parametrize("library,vector", CASES)
@pytest.mark.parametrize("categorical", [False, True])
def test_solved_native_margin_probability_and_reload_oracles(library, vector, categorical):
    payload, frame, labels = toy(library, vector, categorical)
    original = refit.predict_raw(payload, frame)
    design = refit.prepare_design(payload, frame, labels)
    coefficients, solve = refit.solve_coefficients(design, labels, 10.)
    assert solve["final_objective"] < solve["initial_objective"]
    updates = refit.corrections_by_tree(design, coefficients, .25)
    candidate, receipt = refit.fold_corrections(payload, updates)
    assert receipt["topology_sha256_before"] == receipt["topology_sha256_after"]
    refit.verify_fold(payload, candidate, frame, updates, receipt, design["training_prepared"])
    refit.verify_fold(payload, candidate, frame, updates, receipt)
    probe = frame.iloc[:17].copy()
    probe["x"] += .7
    if categorical:
        probe["cat"] = pd.Categorical([None] * len(probe), categories=frame["cat"].cat.categories)
    refit.verify_fold(payload, candidate, probe, updates, receipt)
    restored = pickle.loads(pickle.dumps(candidate, protocol=5))
    np.testing.assert_array_equal(refit.predict(restored, frame), refit.predict(candidate, frame))
    np.testing.assert_array_equal(refit.leaf_indices(restored, probe), refit.leaf_indices(candidate, probe))
    np.testing.assert_array_equal(refit.predict_raw(payload, frame), original)
    assert not np.array_equal(refit.predict_raw(candidate, frame), original)
    if library == "ctboost" and categorical:
        assert design["preprocessing_receipt"]["ordered_training_transform_replayed"]
        assert design["preprocessing_receipt"]["ordered_ctr"]


@pytest.mark.parametrize("library,vector", CASES)
def test_zero_fold_exact_invalid_leaf_and_class_support_fail(library, vector):
    payload, frame, labels = toy(library, vector)
    design = refit.prepare_design(payload, frame, labels)
    updates = refit.corrections_by_tree(design, np.zeros(design["offsets"][-1]))
    candidate, _ = refit.fold_corrections(payload, updates)
    np.testing.assert_array_equal(refit.predict(candidate, frame), refit.predict(payload, frame))
    invalid = copy.deepcopy(updates)
    invalid[0][2**30] = np.ones(3) if vector else np.array([1., 0., 0.])
    with pytest.raises(ValueError, match="nonexistent leaf"):
        refit.fold_corrections(payload, invalid)
    if not vector:
        leaf = next(iter(updates[0]))
        updates[0][leaf][1] = .1
        with pytest.raises(ValueError, match="class support"):
            refit.fold_corrections(payload, updates)


def test_xgb_vector_uses_compact_leaf_slot_and_leaves_training_statistics_unchanged():
    payload, frame, labels = toy("xgboost", True)
    design = refit.prepare_design(payload, frame, labels)
    coefficients = np.linspace(-.2, .3, design["offsets"][-1])
    updates = refit.corrections_by_tree(design, coefficients)
    candidate, receipt = refit.fold_corrections(payload, updates)
    _, _, before = refit._metadata(payload)
    _, _, after = refit._metadata(candidate)
    before_trees = before["learner"]["gradient_booster"]["model"]["trees"]
    after_trees = after["learner"]["gradient_booster"]["model"]["trees"]
    for t, (old, new) in enumerate(zip(before_trees, after_trees)):
        assert old["base_weights"] == new["base_weights"]
        assert old["right_children"] == new["right_children"]
        for leaf, delta in updates[t].items():
            start = 3 * old["right_children"][leaf]
            expected = (np.asarray(old["leaf_weights"][start:start + 3], dtype=np.float32).astype(np.float64) + delta).astype(np.float32)
            np.testing.assert_array_equal(np.asarray(new["leaf_weights"][start:start + 3], dtype=np.float32), expected)
    refit.verify_fold(payload, candidate, frame, updates, receipt)


def test_ct_ordered_replay_rejects_different_training_order():
    payload, frame, labels = toy("ctboost", True, True)
    # A changed training population must not silently receive the saved CTR representation.
    with pytest.raises(ValueError, match="Replayed training pipeline"):
        refit.prepare_design(payload, frame.iloc[1:].reset_index(drop=True), labels[1:])


def test_invalid_labels_penalty_temperature_and_solver_failure_rejected():
    design = refit.make_design(np.array([[0], [1], [0]]), np.zeros((3, 3)), (-1,), .8)
    with pytest.raises(ValueError, match="positive ridge"):
        refit.solve_coefficients(design, [0, 1, 2], 0.)
    with pytest.raises(ValueError, match="labels"):
        refit.solve_coefficients(design, [0, 1, 3], 1.)
    with pytest.raises(ValueError, match="temperature"):
        refit.make_design(np.array([[0]]), np.zeros((1, 3)), (-1,), 0.)
    with pytest.raises(RuntimeError, match="did not converge"):
        refit.solve_coefficients(design, [0, 1, 2], 1., maxiter=0)
    with pytest.raises(ValueError, match="Mixed|mixed"):
        refit.make_design(np.array([[0, 0]]), np.zeros((1, 3)), (-1, 1))


def test_extreme_finite_margins_raise_instead_of_silent_nonfinite_loss():
    design = refit.make_design(np.array([[0]]), np.array([[1.7e308, -1.7e308, 0.]]), (-1,))
    with pytest.raises((FloatingPointError, ValueError)):
        refit.softmax_objective(np.zeros(3), design, [1], 1.)


def test_incorrect_runtime_is_rejected(monkeypatch):
    xgb = pytest.importorskip("xgboost")
    payload, _, _ = toy("xgboost", True)
    monkeypatch.setattr(xgb, "__version__", "3.4.2")
    with pytest.raises(ValueError, match="3.4.1"):
        refit.fold_corrections(payload, [])


@pytest.mark.parametrize("library,vector", CASES)
def test_preflight_bounds_actual_sparse_arrays_and_rejects_before_leaf_prediction(library, vector, monkeypatch):
    payload, frame, labels = toy(library, vector)
    estimate = refit.memory_preflight(payload, len(frame), max_working_bytes=2 * 1024**3)
    design = refit.prepare_design(payload, frame, labels, max_working_bytes=2 * 1024**3)
    assert design["preprocessing_receipt"]["memory_preflight"] == estimate
    assert design["offsets"][-1] <= estimate["parameters_upper_bound"]
    unique = {id(matrix): matrix for matrix in design["matrices"]}
    assert len(unique) == estimate["unique_csr_matrices"]
    assert sum(matrix.nnz for matrix in unique.values()) == estimate["csr_nonzeros"]
    storage = sum(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes for matrix in unique.values())
    assert storage <= estimate["array_budgets_bytes"]["resident_csr_data_indices_indptr"]
    assert sum(estimate["array_budgets_bytes"].values()) == estimate["refit_array_budget_bytes"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight rejection must precede leaf prediction")
    monkeypatch.setattr(refit, "leaf_indices", forbidden)
    with pytest.raises(MemoryError, match="memory budget exceeded"):
        refit.prepare_design(payload, frame, labels, max_working_bytes=1)


def test_preflight_large_dimensions_do_not_allocate_leaf_or_dense_design_arrays():
    estimate = refit.sparse_memory_estimate(10**9, 3, (-1,) * 1000, [1024] * 1000)
    assert estimate["csr_nonzeros"] == 10**12
    assert estimate["parameters_upper_bound"] == 3 * 1024 * 1000
    assert estimate["refit_array_budget_bytes"] > 16 * 10**12
    assert estimate["no_dense_hessian"] is True
