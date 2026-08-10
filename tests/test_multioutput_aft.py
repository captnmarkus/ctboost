import json

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone

import ctboost
from ctboost.sklearn.aft import _aft_bounds, _aft_grad_hess_nll


def _regression_data(seed=712, rows=140):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 6)).astype(np.float32)
    y = np.column_stack(
        [
            1.8 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * X[:, 4],
            -1.4 * X[:, 2] + 0.9 * X[:, 3] - 0.3 * X[:, 5],
        ]
    ).astype(np.float32)
    return X, y


def _multilabel_data(seed=913, rows=150):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 5)).astype(np.float32)
    y = np.column_stack(
        [
            X[:, 0] - 0.4 * X[:, 1] > 0.0,
            X[:, 2] + 0.7 * X[:, 3] - 0.2 * X[:, 4] > 0.0,
        ]
    ).astype(np.int64)
    return X, y


def _aft_data(seed=1121, rows=160, scale=0.45):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 5)).astype(np.float32)
    location = 0.85 * X[:, 0] - 0.55 * X[:, 1] + 0.25 * X[:, 3]
    time = np.exp(location + rng.normal(scale=scale, size=rows))
    bounds = np.column_stack([time, time])
    bounds[0::4, 0] *= 0.8
    bounds[0::4, 1] = np.inf
    bounds[1::4, 0] = 0.0
    bounds[1::4, 1] *= 1.2
    bounds[2::4, 0] *= 0.85
    bounds[2::4, 1] *= 1.15
    return X, bounds, location


def _regressor(iterations=24):
    return ctboost.CTBoostRegressor(
        iterations=iterations,
        learning_rate=0.18,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=19,
    )


def _classifier(iterations=24):
    return ctboost.CTBoostClassifier(
        iterations=iterations,
        learning_rate=0.18,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=23,
    )


def test_multioutput_regressor_sparse_eval_weights_and_exports(tmp_path):
    X, y = _regression_data()
    sample_weight = np.column_stack(
        [np.linspace(0.5, 1.5, 105), np.linspace(1.5, 0.5, 105)]
    ).astype(np.float32)
    model = ctboost.CTBoostMultiOutputRegressor(_regressor(), n_jobs=1)
    model.fit(
        sparse.csr_matrix(X[:105]),
        y[:105],
        sample_weight=sample_weight,
        eval_set=(X[105:], y[105:]),
        eval_names="holdout",
    )

    prediction = model.predict(X[105:])
    assert prediction.shape == (35, 2)
    assert prediction.dtype == np.float64
    assert model.score(X[105:], y[105:]) > 0.0
    assert len(model.get_boosters()) == 2
    assert model.estimators_[0] is not model.estimators_[1]
    assert model.feature_importances_per_output_.shape == (2, X.shape[1])
    assert model.best_iteration_.shape == (2,)
    assert len(model.evals_result_) == 2
    assert all("holdout" in result for result in model.evals_result_)

    model_path = tmp_path / "multioutput.pkl"
    model.save_model(model_path)
    restored = ctboost.CTBoostMultiOutputRegressor.load_model(model_path)
    np.testing.assert_allclose(restored.predict(X[105:]), prediction)

    bundle_path = tmp_path / "multioutput_bundle"
    model.export_model(bundle_path)
    exported = ctboost.CTBoostMultiOutputRegressor.load_exported_model(bundle_path)
    np.testing.assert_allclose(exported.predict(X[105:]), prediction, rtol=1e-6, atol=1e-6)
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tree_semantics"] == "one independent CTBoost booster per output"
    assert manifest["n_outputs"] == 2


def test_multilabel_classifier_probabilities_persistence_and_exports(tmp_path):
    X, y = _multilabel_data()
    # Keep this wrapper/persistence test's historical deterministic quality
    # contract independent of the objective-aware intercept default.
    estimator = _classifier().set_params(boost_from_average=False)
    model = ctboost.CTBoostMultiLabelClassifier(estimator, n_jobs=1).fit(
        X[:110],
        y[:110],
        eval_set=(X[110:], y[110:]),
    )

    prediction = model.predict(X[110:])
    probabilities = model.predict_proba(X[110:])
    assert prediction.shape == (40, 2)
    assert np.mean(prediction == y[110:]) > 0.8
    assert len(probabilities) == 2
    assert all(values.shape == (40, 2) for values in probabilities)
    np.testing.assert_allclose(
        model.predict_positive_proba(X[110:]),
        np.column_stack([values[:, 1] for values in probabilities]),
    )
    assert model.decision_function(X[110:]).shape == (40, 2)
    assert all(classes.tolist() == [0, 1] for classes in model.classes_)

    model_path = tmp_path / "multilabel.pkl"
    model.save_model(model_path)
    restored = ctboost.CTBoostMultiLabelClassifier.load_model(model_path)
    np.testing.assert_array_equal(restored.predict(X[110:]), prediction)

    bundle_path = tmp_path / "multilabel_bundle"
    model.export_model(bundle_path)
    exported = ctboost.CTBoostMultiLabelClassifier.load_exported_model(bundle_path)
    np.testing.assert_array_equal(exported.predict(X[110:]), prediction)
    np.testing.assert_allclose(
        exported.predict_positive_proba(X[110:]),
        model.predict_positive_proba(X[110:]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert all(values.shape == (1, 2) for values in exported.predict_proba(X[110]))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (2.0, 2.0),
        (0.0, 2.0),
        (-np.inf, 2.0),
        (2.0, np.inf),
        (1.0, 3.0),
    ],
)
def test_aft_gradients_and_hessians_match_finite_differences(lower, upper):
    scale = 0.8
    location = 0.35
    epsilon = 1e-4

    def loss(value):
        return _aft_grad_hess_nll(
            np.asarray([value]),
            np.asarray([lower]),
            np.asarray([upper]),
            scale,
        )[2][0]

    gradient, hessian, _ = _aft_grad_hess_nll(
        np.asarray([location]),
        np.asarray([lower]),
        np.asarray([upper]),
        scale,
    )
    numeric_gradient = (loss(location + epsilon) - loss(location - epsilon)) / (2 * epsilon)
    numeric_hessian = (
        loss(location + epsilon) - 2 * loss(location) + loss(location - epsilon)
    ) / (epsilon * epsilon)
    assert gradient[0] == pytest.approx(numeric_gradient, rel=2e-4, abs=2e-5)
    assert hessian[0] == pytest.approx(numeric_hessian, rel=2e-4, abs=2e-5)


def test_aft_mixed_censoring_eval_persistence_and_bundle(tmp_path):
    X, bounds, location = _aft_data()
    model = ctboost.CTBoostAFTRegressor(
        _regressor(iterations=30),
        scale=0.45,
        prediction_type="time",
    ).fit(
        sparse.csr_matrix(X[:120]),
        bounds[:120],
        sample_weight=np.linspace(0.5, 1.5, 120),
        eval_set=(X[120:], bounds[120:]),
        eval_names="survival_holdout",
    )

    log_time = model.predict_log_time(X[120:])
    median = model.predict_time(X[120:], kind="median")
    mean = model.predict_time(X[120:], kind="mean")
    assert log_time.shape == (40,)
    np.testing.assert_allclose(median, np.exp(log_time), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mean, median * np.exp(0.5 * 0.45**2), rtol=1e-12)
    np.testing.assert_allclose(model.predict(X[120:]), median)
    assert np.corrcoef(log_time, location[120:])[0, 1] > 0.35
    assert np.isfinite(model.training_nll_)
    assert np.isfinite(model.negative_log_likelihood(X[120:], bounds[120:]))
    assert "AFTNLL" in model.evals_result_["survival_holdout"]

    model_path = tmp_path / "aft.pkl"
    model.save_model(model_path)
    restored = ctboost.CTBoostAFTRegressor.load_model(model_path)
    np.testing.assert_allclose(restored.predict(X[120:]), median)

    bundle_path = tmp_path / "aft_bundle"
    model.export_model(bundle_path)
    exported = ctboost.CTBoostAFTRegressor.load_exported_model(bundle_path)
    np.testing.assert_allclose(exported.predict(X[120:]), median, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(exported.predict_log_time(X[120:]), log_time, rtol=1e-6, atol=1e-6)
    manifest = exported.manifest
    assert manifest["objective_semantics"] == "log-normal AFT negative log-likelihood"
    assert "serialization only" in manifest["native_objective_surrogate"]


def test_aft_exact_times_log_prediction_and_pool_weight():
    X, _, location = _aft_data(rows=90)
    exact_time = np.exp(location)
    pool_weight = np.linspace(0.25, 1.25, X.shape[0]).astype(np.float32)
    pool = ctboost.Pool(X, exact_time, weight=pool_weight)
    model = ctboost.CTBoostAFTRegressor(
        _regressor(iterations=8),
        scale=0.7,
        prediction_type="log_time",
    ).fit(pool, exact_time)

    np.testing.assert_allclose(model.predict(X[:5]), model.predict_log_time(X[:5]))
    expected = model.negative_log_likelihood(X, exact_time, sample_weight=pool_weight)
    assert model.training_nll_ == pytest.approx(expected)


def test_multi_target_and_aft_validation_and_sklearn_clone():
    X, y = _regression_data(rows=20)
    assert clone(ctboost.CTBoostMultiOutputRegressor()).get_params() == {
        "estimator": None,
        "n_jobs": None,
    }
    cloned_aft = clone(ctboost.CTBoostAFTRegressor(scale=0.9, prediction_type="log_time"))
    assert cloned_aft.scale == 0.9
    assert cloned_aft.prediction_type == "log_time"

    with pytest.raises(ValueError, match="2D"):
        ctboost.CTBoostMultiOutputRegressor(_regressor(1)).fit(X, y[:, 0])
    with pytest.raises(ValueError, match="exactly two classes"):
        ctboost.CTBoostMultiLabelClassifier(_classifier(1)).fit(
            X,
            np.column_stack([np.zeros(X.shape[0]), y[:, 0] > 0]),
        )
    with pytest.raises(ValueError, match="GPU.*n_jobs=1"):
        ctboost.CTBoostMultiOutputRegressor(
            ctboost.CTBoostRegressor(task_type="GPU"),
            n_jobs=2,
        ).fit(X, y)
    with pytest.raises(ValueError, match="distributed_world_size"):
        ctboost.CTBoostMultiOutputRegressor(
            ctboost.CTBoostRegressor(distributed_world_size=2),
        ).fit(X, y)
    with pytest.raises(ValueError, match="distributed_world_size"):
        ctboost.CTBoostAFTRegressor(
            ctboost.CTBoostRegressor(distributed_world_size=2)
        ).fit(X, np.ones(X.shape[0]))
    with pytest.raises(ValueError, match="target-aware categorical/text/embedding"):
        ctboost.CTBoostAFTRegressor(
            ctboost.CTBoostRegressor(cat_features=[0])
        ).fit(X, np.ones(X.shape[0]))
    with pytest.raises(ValueError, match="1D array"):
        ctboost.CTBoostAFTRegressor(_regressor(1)).fit(
            X,
            np.ones(X.shape[0]),
            sample_weight=np.ones((X.shape[0], 1)),
        )

    with pytest.raises(ValueError, match="lower bounds must not exceed"):
        _aft_bounds(np.asarray([[2.0, 1.0]]))
    with pytest.raises(ValueError, match="censored on both sides"):
        _aft_bounds(np.asarray([[0.0, np.inf]]))

    tuple_lower, tuple_upper = _aft_bounds((1.0, 2.0))
    np.testing.assert_array_equal(tuple_lower, np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(tuple_upper, tuple_lower)

    vector_lower, vector_upper = _aft_bounds(
        (np.asarray([1.0, 2.0]), np.asarray([1.5, np.inf]))
    )
    np.testing.assert_array_equal(vector_lower, np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(vector_upper, np.asarray([1.5, np.inf]))

    with pytest.raises(ValueError, match="two array-like vectors"):
        _aft_bounds((np.asarray([1.0, 2.0]), 3.0))


def test_parallel_non_picklable_configuration_fails_early():
    class NonPicklableObjective:
        def __call__(self, prediction, label):
            return prediction - label, np.ones_like(prediction)

        def __reduce__(self):
            raise TypeError("cannot serialize")

    X, y = _regression_data(rows=20)
    base = _regressor(1).set_params(loss_function=NonPicklableObjective())
    with pytest.raises(ValueError, match="non-picklable custom loss"):
        ctboost.CTBoostMultiOutputRegressor(base, n_jobs=2).fit(X, y)


def test_sequential_non_picklable_objective_fits_and_persists_for_inference(tmp_path):
    class NonPicklableObjective:
        def __call__(self, prediction, label):
            return prediction - label, np.ones_like(prediction)

        def __reduce__(self):
            raise TypeError("cannot serialize")

    X, y = _regression_data(rows=30)
    base = _regressor(3).set_params(loss_function=NonPicklableObjective())
    model = ctboost.CTBoostMultiOutputRegressor(base, n_jobs=1).fit(X, y)
    expected = model.predict(X)

    model_path = tmp_path / "non_picklable_multioutput.pkl"
    model.save_model(model_path)
    restored = ctboost.CTBoostMultiOutputRegressor.load_model(model_path)

    np.testing.assert_allclose(restored.predict(X), expected, rtol=0.0, atol=0.0)
    assert (
        restored.estimator.loss_function
        == model.estimators_[0].get_booster().native_objective_name
    )
    with pytest.raises(ValueError, match="pass that callable objective again"):
        ctboost.CTBoostMultiOutputRegressor(
            restored.estimator,
            n_jobs=1,
        ).fit(X, y, init_model=restored)


def test_aft_ignores_non_picklable_base_objective_and_persists(tmp_path):
    class NonPicklableObjective:
        def __call__(self, prediction, label):
            return prediction - label, np.ones_like(prediction)

        def __reduce__(self):
            raise TypeError("cannot serialize")

    X, bounds, _ = _aft_data(rows=30)
    base = _regressor(3).set_params(loss_function=NonPicklableObjective())
    model = ctboost.CTBoostAFTRegressor(base, scale=0.7).fit(X, bounds)
    expected = model.predict(X)

    model_path = tmp_path / "non_picklable_aft.pkl"
    model.save_model(model_path)
    restored = ctboost.CTBoostAFTRegressor.load_model(model_path)

    np.testing.assert_allclose(restored.predict(X), expected, rtol=0.0, atol=0.0)
    assert restored.estimator.loss_function == "RMSE"
