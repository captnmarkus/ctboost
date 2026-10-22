"""Independent math and native-routing checks for binary post-fit experiments."""
import pickle

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.special import expit
from benchmarks import fixed_binary_leaf_refit as refit


def _payload(library, categorical=False):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(419)
    frame = pd.DataFrame({"x": rng.normal(size=150), "z": rng.normal(size=150)})
    logits = 2 * frame.x.to_numpy() + (frame.z.to_numpy() > 0) - .5
    if categorical:
        frame["cat"] = pd.Categorical(["a", "b", "c"] * 50)
        logits += np.tile([0., 2., -1.], 50)
    target = (rng.random(len(frame)) < expit(logits)).astype(np.int32)
    if library == "ctboost":
        import ctboost
        model = ctboost.train(frame, {"objective": "Logloss", "iterations": 12,
            "max_depth": 3, "min_data_in_leaf": 2, "alpha": .5,
            "learning_rate": .17, "ordered_ctr": categorical,
            "cat_features": ["cat"] if categorical else None, "random_seed": 419,
            "verbose": False}, label=target)
    elif library == "xgboost":
        xgb = pytest.importorskip("xgboost")
        model = xgb.train({"objective": "binary:logistic", "max_depth": 3,
                          "eta": .17, "nthread": 1, "tree_method": "hist"},
                          xgb.DMatrix(frame, label=target, enable_categorical=True),
                          num_boost_round=12)
    else:
        lgb = pytest.importorskip("lightgbm")
        model = lgb.train({"objective": "binary", "num_leaves": 7,
                          "learning_rate": .17, "num_threads": 1,
                          "min_data_in_leaf": 2, "verbosity": -1},
                          lgb.Dataset(frame, label=target), num_boost_round=12)
    return {"library": library, "model": model, "schema": {}, "problem": "binary"}, frame, target


def test_sparse_logistic_fit_matches_independent_dense_newton_solution():
    leaves = np.array([[7, 0, 91], [7, 0, 92], [8, 0, 91], [8, 0, 92], [7, 0, 91]])
    matrix, maps = refit.indicator_matrix(leaves)
    baseline = np.array([-2., 1., -.4, 3., .1])
    labels = np.array([0., 1., 1., 0., 1.])
    penalty = 2.
    actual, receipt = refit.solve_coefficients({"Z": matrix, "baseline": baseline}, labels, penalty)
    dense = matrix.toarray()
    expected = np.zeros(dense.shape[1])
    for _ in range(30):
        probability = expit(baseline + dense @ expected)
        gradient = dense.T @ (probability - labels) + penalty * expected
        hessian = dense.T @ ((probability * (1 - probability))[:, None] * dense) + penalty * np.eye(len(expected))
        expected -= np.linalg.solve(hessian, gradient)
        if np.max(np.abs(gradient)) < 1e-12:
            break
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    assert receipt["final_objective"] < receipt["initial_objective"]
    assert receipt["maximum_absolute_mean_gradient"] < 1e-6
    updates = refit.corrections_by_tree(maps, actual, .25)
    np.testing.assert_allclose(refit.correction_values(leaves, updates), .25 * matrix @ actual)
    assert refit.correction_values(np.array([[999, 999, 999]]), updates)[0] == 0


def test_logistic_gradient_matches_finite_differences_and_extreme_margins_are_finite():
    matrix, _ = refit.indicator_matrix(np.array([[0, 5], [1, 5], [0, 6], [1, 6]]))
    baseline, labels = np.array([.3, -2., 1., 5.]), np.array([0., 1., 1., 0.])
    coefficients = np.array([.1, -.2, .3, -.4])
    _, gradient = refit.logistic_objective(coefficients, matrix, baseline, labels, 10.)
    differences = []
    for j in range(len(coefficients)):
        delta = np.zeros_like(coefficients)
        delta[j] = 1e-5
        plus = refit.logistic_objective(coefficients + delta, matrix, baseline, labels, 10.)[0]
        minus = refit.logistic_objective(coefficients - delta, matrix, baseline, labels, 10.)[0]
        differences.append((plus - minus) / 2e-5)
    np.testing.assert_allclose(gradient, differences, rtol=1e-7, atol=1e-9)
    value, gradient = refit.logistic_objective(coefficients, matrix, np.array([1000., -1000., 1000., -1000.]), labels, 1.)
    assert np.isfinite(value) and np.isfinite(gradient).all()


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("categorical", [False, True])
def test_folded_logit_corrections_preserve_routes_and_survive_pickle(library, categorical):
    payload, frame, target = _payload(library, categorical)
    original_predictions = refit.predict(payload, frame)
    design = refit.prepare_design(payload, frame, target)
    coefficients, _ = refit.solve_coefficients(design, target, 10.)
    updates = refit.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, .25)
    changed, receipt = refit.fold_corrections(payload, updates)
    assert receipt["topology_sha256_before"] == receipt["topology_sha256_after"]
    refit.verify_fold(payload, changed, frame, updates, receipt, design["training_prepared"])
    refit.verify_fold(payload, changed, frame, updates, receipt)
    probe = frame.iloc[:15].copy()
    probe["x"] += .7
    if categorical:
        pd = pytest.importorskip("pandas")
        probe["cat"] = pd.Categorical([None] * len(probe), categories=frame["cat"].cat.categories)
    refit.verify_fold(payload, changed, probe, updates, receipt)
    np.testing.assert_array_equal(refit.predict(payload, frame), original_predictions)
    restored = pickle.loads(pickle.dumps(changed, protocol=5))
    np.testing.assert_array_equal(refit.predict(restored, frame), refit.predict(changed, frame))
    assert not np.array_equal(refit.predict(changed, frame), original_predictions)
    if library == "ctboost" and categorical:
        assert design["preprocessing_receipt"]["ordered_training_transform_replayed"]
        assert design["preprocessing_receipt"]["ordered_ctr"]


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
def test_zero_fold_is_exact_and_invalid_leaf_rejected(library):
    payload, frame, target = _payload(library)
    design = refit.prepare_design(payload, frame, target)
    updates = [{int(leaf): 0. for leaf in leaves} for leaves in design["leaf_ids_by_tree"]]
    changed, _ = refit.fold_corrections(payload, updates)
    np.testing.assert_array_equal(refit.predict(changed, frame), refit.predict(payload, frame))
    updates[0][2**30] = 1.
    with pytest.raises(ValueError, match="nonexistent leaf"):
        refit.fold_corrections(payload, updates)


def test_invalid_solver_inputs_and_nonconvergence_rejected():
    matrix, _ = refit.indicator_matrix(np.array([[0], [1]]))
    design = {"Z": matrix, "baseline": np.array([0., 1.])}
    with pytest.raises(ValueError, match="positive ridge"):
        refit.solve_coefficients(design, np.array([0., 1.]), 0.)
    with pytest.raises(ValueError, match="binary labels"):
        refit.solve_coefficients(design, np.array([0., 2.]), 1.)
    with pytest.raises(RuntimeError, match="did not converge"):
        refit.solve_coefficients(design, np.array([1., 0.]), 1., maxiter=0)
    with pytest.raises(ValueError, match="binary logistic"):
        refit.fold_corrections({"problem": "regression"}, [])


def test_confident_correct_row_does_not_cancel_another_rows_logistic_loss():
    matrix, _ = refit.indicator_matrix(np.array([[0], [1]]))
    value, _ = refit.logistic_objective(np.zeros(2), matrix,
        np.array([1e18, 0.]), np.array([1., 0.]), 1.)
    np.testing.assert_allclose(value, np.log(2.) / 2, rtol=0, atol=1e-15)


def test_nonunit_lightgbm_sigmoid_is_rejected():
    lgb = pytest.importorskip("lightgbm")
    data = np.arange(30, dtype=np.float64)[:, None]
    labels = (data[:, 0] > 15).astype(int)
    model = lgb.train({"objective": "binary", "sigmoid": 2., "num_threads": 1,
                       "min_data_in_leaf": 2, "verbosity": -1},
                      lgb.Dataset(data, label=labels), num_boost_round=2)
    payload = {"library": "lightgbm", "model": model, "problem": "binary"}
    with pytest.raises(ValueError, match="sigmoid must be 1"):
        refit.fold_corrections(payload, [{} for _ in range(model.num_trees())])
