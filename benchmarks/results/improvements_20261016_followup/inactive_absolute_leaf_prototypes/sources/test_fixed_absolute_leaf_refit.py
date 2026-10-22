"""Inactive helper tests; generated toy inputs only, no benchmark data/models.

SOURCE PREPARED, NOT EXECUTED. The module fixture plans nine four-round native
fits (three libraries x numeric/categorical/root-only); it is shared by the
native checks. A full-stack, CPU-owned test window must be approved separately.
"""
import copy
import json
import pickle

import numpy as np
import pytest

from benchmarks import fixed_absolute_leaf_refit as refit
from benchmarks.absolute_leaf_ridge_numerics import CertificateLimits, LSQRSettings


SETTINGS = LSQRSettings(iterations=1000, atol=1e-12, btol=1e-12, conlim=1e12)
LIMITS = CertificateLimits(relative_gap=1e-9, normalized_gradient=1e-8,
                           normalized_intercept=1e-9, normalized_dual_feasibility=1e-9)


@pytest.fixture(scope="module", params=[(library, mode)
    for library in ("ctboost", "xgboost", "lightgbm") for mode in ("numeric", "categorical", "root")])
def toy(request):
    pd = pytest.importorskip("pandas")
    library, mode = request.param
    rng = np.random.default_rng(18041)
    X = pd.DataFrame({"x": rng.normal(size=72), "z": rng.normal(size=72)})
    y = 3.0 + 2 * X.x.to_numpy() - .4 * X.z.to_numpy() + rng.normal(0, .05, len(X))
    categorical = mode == "categorical"
    if categorical:
        X["cat"] = pd.Categorical(["a", "b", "c", None] * 18)
        y += np.tile([0., 1., -1., .5], 18)
    if mode == "root":
        X.loc[:, :] = 0.0
    schema = {"cat": ["a", "b", "c"]} if categorical else {}
    if library == "ctboost":
        import ctboost
        model = ctboost.train(X, {"objective": "RMSE", "iterations": 4,
            "max_depth": 2, "min_data_in_leaf": 2, "alpha": 1., "learning_rate": .19,
            "ordered_ctr": categorical, "cat_features": ["cat"] if categorical else None,
            "random_seed": 18041, "verbose": False}, label=y,
            learning_rate_schedule=[.19, .13, .09, .07])
    elif library == "xgboost":
        xgb = pytest.importorskip("xgboost")
        if xgb.__version__ != "3.4.1":
            pytest.skip("Requires prospective exact XGBoost3.4.1 stack")
        model = xgb.train({"objective": "reg:squarederror", "max_depth": 2, "eta": .19,
            "tree_method": "hist", "nthread": 1, "seed": 18041},
            xgb.DMatrix(X, label=y, enable_categorical=True), num_boost_round=4)
    else:
        lgb = pytest.importorskip("lightgbm")
        if lgb.__version__ != "4.7.0":
            pytest.skip("Requires prospective exact LightGBM4.7.0 stack")
        model = lgb.train({"objective": "regression", "num_leaves": 4, "learning_rate": .19,
            "num_threads": 1, "min_data_in_leaf": 2, "verbosity": -1, "seed": 18041},
            lgb.Dataset(X, label=y), num_boost_round=4)
    payload = {"library": library, "model": model, "schema": schema, "problem": "regression"}
    design = refit.prepare_design(payload, X, y, max_working_bytes=64 * 1024**2)
    return payload, X, y, design, mode


def test_complete_map_keeps_unobserved_duplicate_and_root_columns():
    leaves = np.array([[7, 0, 91], [7, 0, 92], [8, 0, 91], [8, 0, 92]])
    maps = [np.array([7, 8, 9]), np.array([0]), np.array([91, 92, 93])]
    Z = refit.indicator_matrix(leaves, maps)
    assert Z.shape == (4, 7) and Z.nnz == 12 and Z.has_canonical_format
    np.testing.assert_array_equal(Z[:, [2, 6]].toarray(), np.zeros((4, 2)))
    np.testing.assert_array_equal(Z.sum(axis=1), np.full((4, 1), 3))
    weights = np.array([1., -2., 0., .4, 3., -4., 0.])
    np.testing.assert_allclose(refit.independent_leaf_sum(leaves, maps, weights, 8.), 8. + Z @ weights)
    with pytest.raises(ValueError, match="absent"):
        refit.indicator_matrix(np.array([[99, 0, 91]]), maps)
    for invalid in (np.array([[1.5]]), np.array([[np.nan]])):
        with pytest.raises(ValueError):
            refit.indicator_matrix(invalid, [np.array([1, 2])])
    with pytest.raises(ValueError):
        refit.independent_leaf_sum(np.array([[1]]), [np.array([1.5])], [1.], 0.)


def test_solver_adapter_matches_independent_dense_centered_solution_and_unseen_zero():
    leaves = np.array([[1, 0], [2, 0], [1, 0], [2, 0], [1, 0]])
    maps = [np.array([1, 2, 3]), np.array([0])]
    Z = refit.indicator_matrix(leaves, maps)
    y = np.array([7., 10., 8., 9., 6.])
    design = {"Z": Z, "labels": y.copy()}
    w, b, receipt = refit.solve_coefficients(design, y, 10., settings=SETTINGS, limits=LIMITS)
    dense = Z.toarray()
    centered = dense - dense.mean(axis=0)
    expected = np.linalg.solve(centered.T @ centered + 10. * np.eye(Z.shape[1]),
                               centered.T @ (y - y.mean()))
    np.testing.assert_allclose(w, expected, rtol=1e-9, atol=1e-11)
    assert b == pytest.approx(np.mean(y - dense @ expected), rel=1e-12)
    assert w[2] == 0 and receipt["success"]
    with pytest.raises(ValueError, match="labels differ"):
        refit.solve_coefficients(design, y + 1, 10., settings=SETTINGS, limits=LIMITS)


def test_native_extraction_replay_full_basis_and_original_reconstruction(toy):
    payload, X, y, design, mode = toy
    native = design["native"]
    assert design["Z"].shape[1] == len(native["theta"])
    expected = refit.independent_leaf_sum(design["training_leaves"], native["leaf_ids_by_tree"],
                                          native["theta"], native["bias"])
    np.testing.assert_allclose(design["baseline"], expected, rtol=5e-5, atol=1e-5)
    if payload["library"] == "ctboost":
        assert native["rates"] == [refit._float32(value) for value in (.19, .13, .09, .07)]
        if mode == "categorical":
            assert design["receipt"]["ordered_training_transform_replayed"]
            assert design["receipt"]["ordered_ctr"]
    if payload["library"] == "lightgbm":
        assert native["bias"] == 0
    if mode == "root":
        assert all(len(ids) == 1 for ids in native["leaf_ids_by_tree"])


def test_native_absolute_solve_blend_routes_bias_and_persistence(toy):
    payload, X, y, design, mode = toy
    native = design["native"]
    original_bytes = pickle.dumps(payload, protocol=5)
    weights, bias, solve = refit.solve_coefficients(design, y, 10., settings=SETTINGS, limits=LIMITS)
    assert solve["success"]
    probe = X.iloc[:17].copy()
    probe["x"] += .7
    probe.loc[probe.index[::4], "z"] = np.nan
    if mode == "categorical":
        import pandas as pd
        probe["cat"] = pd.Categorical([None] * len(probe), categories=X["cat"].cat.categories)
    for q in (.25, 1.):
        candidate, receipt = refit.fold_absolute(payload, design, weights, bias, shrinkage=q)
        assert receipt["nonoutput_sha256_before"] == receipt["nonoutput_sha256_after"]
        refit.verify_fold(payload, candidate, X, design, weights, bias, receipt, design["training_prepared"])
        refit.verify_fold(payload, candidate, X, design, weights, bias, receipt)
        refit.verify_fold(payload, candidate, probe, design, weights, bias, receipt)
        refit.verify_reload(candidate, [X, probe])
        current = refit.extract_native(refit._without_receipt(candidate))
        blended = (1 - q) * native["theta"] + q * weights
        blended_bias = (1 - q) * native["bias"] + q * bias
        if payload["library"] == "lightgbm":
            blended[:len(native["leaf_ids_by_tree"][0])] += blended_bias
            assert current["bias"] == 0
        else:
            assert current["bias"] == refit._float32(blended_bias)
        np.testing.assert_allclose(current["theta"], blended, rtol=5e-7, atol=1e-7)
        assert receipt["inference_only"] and "unsupported" in receipt["historical_statistics"]
    assert pickle.dumps(payload, protocol=5) == original_bytes


def test_q0_is_original_payload_and_exact_serialized_native_identity(toy):
    payload, X, y, design, _ = toy
    original_bytes = pickle.dumps(payload, protocol=5)
    zero = np.zeros(len(design["native"]["theta"]))
    candidate, receipt = refit.fold_absolute(payload, design, zero, 123.456, shrinkage=0.)
    assert candidate is payload
    assert pickle.dumps(candidate, protocol=5) == original_bytes
    refit.verify_fold(payload, candidate, X, design, zero, 123.456, receipt)


def test_all_native_leaves_including_heldout_only_leaf_are_replaced(toy):
    payload, X, y, design, mode = toy
    if mode == "root":
        return  # Root-only basis is checked separately; it has no unseen route.
    # A pure folding fixture with a one-row numerical design. It deliberately
    # does not claim to replay an ordered pipeline on a subset of original TRAIN.
    subset = dict(design, Z=refit.indicator_matrix(design["training_leaves"][:1],
                                                 design["native"]["leaf_ids_by_tree"]))
    unseen = np.asarray(subset["Z"].sum(axis=0)).ravel() == 0
    assert unseen.any()
    weights = np.zeros(len(unseen))
    candidate, receipt = refit.fold_absolute(payload, subset, weights, 2.75, shrinkage=1.)
    current = refit.extract_native(refit._without_receipt(candidate))
    expected = np.zeros(len(unseen))
    if payload["library"] == "lightgbm":
        expected[:len(current["leaf_ids_by_tree"][0])] = 2.75
        assert current["bias"] == 0
    else:
        assert current["bias"] == 2.75
    np.testing.assert_array_equal(current["theta"], expected)
    np.testing.assert_allclose(refit.predict(candidate, X), np.full(len(X), 2.75), rtol=0, atol=1e-6)
    weights[np.flatnonzero(unseen)[0]] = 1.
    with pytest.raises(ValueError, match="Unobserved"):
        refit.fold_absolute(payload, subset, weights, 2.75, shrinkage=.25)


def test_zero_budget_fails_before_any_native_prediction(toy, monkeypatch):
    payload, X, y, _, _ = toy
    monkeypatch.setattr(refit, "leaf_indices", lambda *a, **k: pytest.fail("preflight must precede leaf prediction"))
    with pytest.raises(MemoryError):
        refit.prepare_design(payload, X, y, max_working_bytes=1)


def test_bad_bias_shrinkage_provenance_and_double_application_fail(toy):
    payload, X, y, design, _ = toy
    weights = np.zeros(len(design["native"]["theta"]))
    for q, bias in [(-.1, 1.), (1.1, 1.), (.5, np.inf)]:
        with pytest.raises(ValueError):
            refit.fold_absolute(payload, design, weights, bias, shrinkage=q)
    different = copy.copy(payload)
    different["schema"] = {"different": ["value"]}
    with pytest.raises(ValueError, match="provenance"):
        refit.fold_absolute(different, design, weights, 1., shrinkage=.25)
    malformed = dict(design, native=copy.deepcopy(design["native"]))
    malformed["native"]["leaf_ids_by_tree"][0][0] += 1000000
    with pytest.raises(ValueError, match="leaf map"):
        refit.fold_absolute(payload, malformed, weights, 1., shrinkage=.25)
    candidate, _ = refit.fold_absolute(payload, design, weights, 1., shrinkage=.25)
    with pytest.raises(ValueError, match="twice"):
        refit.extract_native(candidate)
    if payload["library"] != "lightgbm":
        with pytest.raises(ValueError, match="float32"):
            refit.fold_absolute(payload, design, weights, 2 * float(np.finfo(np.float32).max), shrinkage=1.)


def test_separate_fixed_tolerance_is_not_relaxed_by_large_analytical_bound():
    with pytest.raises(AssertionError):
        refit._check_oracle(np.array([1.]), np.array([0.]), 1e10)


def test_xgb_weighted_dart_guard_looks_beyond_gbtree_name():
    document = {"learner": {"objective": {"name": "reg:squarederror"},
        "learner_model_param": {"num_class": "0", "num_target": "1"},
        "gradient_booster": {"name": "gbtree", "model": {"trees": [], "tree_info": [], "weight_drop": [.5]}}}}
    with pytest.raises(ValueError, match="Weighted/DART"):
        refit._xgb_guard(document, {})
    document["learner"]["gradient_booster"]["model"]["weight_drop"] = []
    with pytest.raises(ValueError, match="dropout"):
        refit._xgb_guard(document, {"dart_train_param": {"rate_drop": ".1", "one_drop": "0"}})
    with pytest.raises(ValueError, match="Parallel-tree"):
        refit._xgb_guard(document, {"num_parallel_tree": "2"})


@pytest.mark.parametrize("field,value", [("boosting", "dart"), ("boosting", "rf"),
    ("linear_tree", "1"), ("monotone_constraints", "1,0"), ("reg_sqrt", "1"),
    ("max_delta_step", "1"), ("pred_early_stop", "1")])
def test_lgb_native_text_guards_survive_empty_python_params(field, value):
    document = {"objective": "regression", "num_tree_per_iteration": 1, "num_class": 1,
                "average_output": False, "tree_info": []}
    parameters = {"boosting": "gbdt", "linear_tree": "0", field: value}
    with pytest.raises(ValueError):
        refit._lgb_guard(document, parameters)
    with pytest.raises(ValueError):
        refit._lgb_guard(dict(document, average_output=True), {"boosting": "gbdt", "linear_tree": "0"})


def test_float32_bias_and_rate_conversion_fails_closed():
    for value in (np.nan, np.inf, 2 * float(np.finfo(np.float32).max)):
        with pytest.raises(ValueError, match="float32"):
            refit._float32(value)
    assert refit._float32(1. / 3.) == float(np.float32(1. / 3.))


def test_native_parameter_parser_rejects_ambiguous_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        refit._lgb_parameters("[boosting: gbdt]\n[boosting: dart]\n")
    assert refit._lgb_parameters("[boosting: gbdt]\n[linear_tree: 0]\n") == {
        "boosting": "gbdt", "linear_tree": "0"}
