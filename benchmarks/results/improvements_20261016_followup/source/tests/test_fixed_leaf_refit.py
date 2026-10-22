"""Scientific correctness checks for the opt-in benchmark leaf-refit helper."""
import numpy as np
import pytest

lsqr = pytest.importorskip("scipy.sparse.linalg").lsqr

from benchmarks import fixed_leaf_refit as refit


def _payload(library, categorical=False):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(419)
    frame = pd.DataFrame({"x": rng.normal(size=120), "z": rng.normal(size=120)})
    target = 2 * frame.x.to_numpy() + (frame.z.to_numpy() > 0) + rng.normal(0, .1, 120)
    if categorical:
        frame["cat"] = pd.Categorical(["a", "b", "c"] * 40)
        target += np.tile([0., 2., -1.], 40)
    if library == "ctboost":
        import ctboost
        model = ctboost.train(frame, {"objective": "RMSE", "iterations": 12,
            "max_depth": 3, "min_data_in_leaf": 2, "alpha": .5,
            "learning_rate": .17, "ordered_ctr": categorical,
            "cat_features": ["cat"] if categorical else None, "random_seed": 419,
            "verbose": False}, label=target)
    elif library == "xgboost":
        xgb = pytest.importorskip("xgboost")
        model = xgb.train({"objective": "reg:squarederror", "max_depth": 3,
                          "eta": .17, "nthread": 1, "tree_method": "hist"},
                          xgb.DMatrix(frame, label=target, enable_categorical=True),
                          num_boost_round=12)
    else:
        lgb = pytest.importorskip("lightgbm")
        model = lgb.train({"objective": "regression", "num_leaves": 7,
                          "learning_rate": .17, "num_threads": 1,
                          "min_data_in_leaf": 2, "verbosity": -1},
                          lgb.Dataset(frame, label=target), num_boost_round=12)
    return {"library": library, "model": model, "schema": {}, "problem": "regression"}, frame, target


def test_ridge_sparse_design_matches_dense_closed_form_with_duplicate_root_columns():
    leaves = np.array([[7, 0, 91], [7, 0, 92], [8, 0, 91], [8, 0, 92], [7, 0, 91]])
    matrix, maps = refit.indicator_matrix(leaves)
    np.testing.assert_array_equal(matrix.sum(axis=1), np.full((5, 1), 3))
    residual = np.array([1., -2., 3., -4., .5])
    penalty = 10.
    dense = matrix.toarray()
    expected = np.linalg.solve(dense.T @ dense + penalty * np.eye(dense.shape[1]), dense.T @ residual)
    actual = lsqr(matrix, residual, damp=np.sqrt(penalty), atol=1e-12, btol=1e-12)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    updates = refit.corrections_by_tree(maps, actual, .25)
    np.testing.assert_allclose(refit.correction_values(leaves, updates), .25 * matrix @ actual)
    assert refit.correction_values(np.array([[999, 999, 999]]), updates)[0] == 0


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
@pytest.mark.parametrize("categorical", [False, True])
def test_native_leaf_fold_matches_oracle_without_changing_routing_or_original(library, categorical):
    payload, frame, target = _payload(library, categorical)
    old_predictions = refit.predict(payload, frame)
    design = refit.prepare_design(payload, frame, target)
    # A real ridge correction, trained only on the reconstructed training path.
    residual = target - design["baseline"]
    solution = lsqr(design["Z"], residual, damp=np.sqrt(10.), atol=1e-10, btol=1e-10)[0]
    assert np.linalg.norm(design["Z"] @ solution - residual)**2 + 10 * np.dot(solution, solution) <= np.dot(residual, residual) + 1e-8
    updates = refit.corrections_by_tree(design["leaf_ids_by_tree"], solution, .25)
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
    np.testing.assert_array_equal(refit.predict(payload, frame), old_predictions)
    assert not np.array_equal(refit.predict(changed, frame), old_predictions)
    if library == "ctboost" and categorical:
        assert design["preprocessing_receipt"]["ordered_training_transform_replayed"]
        assert design["preprocessing_receipt"]["ordered_ctr"]
        assert not np.array_equal(design["baseline"], old_predictions)


@pytest.mark.parametrize("library", ["ctboost", "xgboost", "lightgbm"])
def test_zero_correction_preserves_predictions_and_bad_leaf_is_rejected(library):
    payload, frame, target = _payload(library)
    design = refit.prepare_design(payload, frame, target)
    updates = [{int(leaf): 0. for leaf in leaves} for leaves in design["leaf_ids_by_tree"]]
    changed, receipt = refit.fold_corrections(payload, updates)
    np.testing.assert_array_equal(refit.predict(changed, frame), refit.predict(payload, frame))
    updates[0][2**30] = 1.
    with pytest.raises(ValueError, match="nonexistent leaf"):
        refit.fold_corrections(payload, updates)


def test_refit_rejects_classification_and_nonfinite_coefficients():
    with pytest.raises(ValueError, match="regression"):
        refit.fold_corrections({"problem": "binary"}, [])
    with pytest.raises(ValueError, match="Invalid correction"):
        refit.corrections_by_tree([np.array([1])], [float("nan")])
