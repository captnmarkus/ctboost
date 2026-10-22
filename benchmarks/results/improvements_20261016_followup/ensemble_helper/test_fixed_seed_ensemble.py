"""Synthetic math/metadata checks only; no fits or real benchmark data reads."""
import copy
import pickle
import sys
import types

import numpy as np
import pytest

from benchmarks import fixed_seed_ensemble as ensemble


class FixedModel:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.params = {"sigmoid": 1.}
        self.frames = []
        self.inplace_delta = 0.

    def predict(self, frame, **kwargs):
        self.frames.append(frame)
        return self.values.copy()

    def inplace_predict(self, frame, **kwargs):
        self.frames.append(frame)
        return self.values + self.inplace_delta

    def dump_model(self):
        return {"objective": f"binary sigmoid:{self.params.get('sigmoid', 1.)}"}


def records(problem="binary", library="ctboost", values=None, schema=None):
    values = values if values is not None else ([1., -2., .5], [-1., 3., .2])
    return [{"seed": seed, "model_id": f"toy-{seed}", "model_sha256": str(index) * 64,
             "payload": {"library": library, "problem": problem,
                         "schema": copy.deepcopy({} if schema is None else schema),
                         "model": FixedModel(value)}}
            for index, (seed, value) in enumerate(zip((47, 947), values), 1)]


def expit(values):
    return np.exp(-np.logaddexp(0., -np.asarray(values, dtype=np.float64)))


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    return np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)


def test_regression_average_and_squared_error_identity():
    first, second, target = np.array([1., -3., 7.]), np.array([4., 5., -8.]), np.array([3., 1., 0.])
    actual = ensemble.average_outputs(first, second, "regression")
    np.testing.assert_array_equal(actual, .5 * first + .5 * second)
    mean_component_mse = .5 * (np.mean((first-target)**2) + np.mean((second-target)**2))
    assert np.mean((actual-target)**2) == pytest.approx(mean_component_mse - .25*np.mean((first-second)**2))


def test_regression_average_avoids_overflow_in_component_sum():
    values = np.array([1.7e308, -1.7e308])
    np.testing.assert_array_equal(ensemble.average_outputs(values, values, "regression"), values)


def test_binary_is_probability_mean_not_logit_mean():
    first, second = np.array([4., -2., .5]), np.array([-1., 2., 2.])
    actual = expit(ensemble.average_outputs(first, second, "binary"))
    np.testing.assert_allclose(actual, .5 * (expit(first) + expit(second)), rtol=1e-15, atol=1e-15)
    assert not np.allclose(actual, expit(.5 * (first + second)))


def test_binary_extreme_tails_retain_ranking_and_symmetry():
    first = np.array([1000., 999., -1000., -999., 1000.])
    second = np.array([1000., 999., -1000., -999., -1000.])
    actual = ensemble.average_outputs(first, second, "binary")
    assert np.isfinite(actual).all()
    assert actual[0] > actual[1] > 0 > actual[3] > actual[2]
    np.testing.assert_allclose(actual[:4], first[:4], rtol=0, atol=1e-12)
    assert actual[4] == 0


def test_binary_average_has_no_guarantee_against_best_component_auc():
    # Positive row first: seed1 ranks correctly; the fixed mixture can reverse it.
    first, second = np.array([2., 1.]), np.array([-100., 100.])
    mixed = ensemble.average_outputs(first, second, "binary")
    assert first[0] > first[1] and mixed[0] < mixed[1]


def test_multiclass_log_probability_mean_and_logloss_bound():
    first = np.array([[3., 1., -2.], [-5., 0., 2.], [2., -4., 1.]])
    second = np.array([[-1., 2., 1.], [2., -2., 0.], [0., 1., 2.]])
    actual = ensemble.average_outputs(first, second, "multiclass")
    expected = .5 * (softmax(first) + softmax(second))
    np.testing.assert_allclose(np.exp(actual), expected, rtol=2e-15, atol=1e-16)
    np.testing.assert_allclose(np.exp(actual).sum(axis=1), 1., rtol=1e-15)
    labels = np.array([0, 1, 2])
    rows = np.arange(3)
    mean_component_loss = -.5 * (np.log(softmax(first)[rows, labels]) + np.log(softmax(second)[rows, labels]))
    assert np.all(-actual[rows, labels] <= mean_component_loss + 1e-15)


def test_multiclass_extreme_logits_keep_finite_rare_class_log_probability():
    first = np.array([[10000., 0., -10000.]])
    actual = ensemble.average_outputs(first, first, "multiclass")
    np.testing.assert_array_equal(actual, np.array([[0., -10000., -20000.]]))


def test_finite_multiclass_inputs_with_unrepresentable_differences_rejected():
    extreme = np.array([[1.7e308, -1.7e308]])
    with pytest.raises(ValueError, match="finite float64 range"):
        ensemble.average_outputs(extreme, extreme, "multiclass")


@pytest.mark.parametrize("problem,shape", [("binary", (0,)), ("regression", (0,)), ("multiclass", (0, 3))])
def test_empty_batches(problem, shape):
    assert ensemble.average_outputs(np.empty(shape), np.empty(shape), problem).shape == shape


@pytest.mark.parametrize("first,second,problem", [
    ([0., np.nan], [0., 1.], "binary"), ([0.], [np.inf], "regression"),
    ([0., 1.], [0.], "binary"), ([[0., 1.]], [[0., 1.]], "regression"),
    ([0., 1.], [0., 1.], "multiclass"), ([[0.]], [[0.]], "multiclass"),
    ([0.], [0.], "ranking"),
])
def test_invalid_prediction_shapes_and_values_rejected(first, second, problem):
    with pytest.raises(ValueError):
        ensemble.average_outputs(first, second, problem)


@pytest.mark.parametrize("change", ["version", "bool_version", "rule", "weights", "component_count", "same_seed",
                                    "same_id", "bad_hash", "library", "problem", "schema_order", "categories_order",
                                    "schema_type", "output_kind", "extra_record_field"])
def test_strict_versioned_metadata_rejects_changes(change):
    payload = ensemble.make_payload(records(schema={"a": ["x", "y"], "b": ["z"]}))
    second = payload["components"][1]
    if change == "version": payload["format_version"] = 2
    elif change == "bool_version": payload["format_version"] = True
    elif change == "rule": payload["rule"] = "selected_weights"
    elif change == "weights": payload["weights"] = [.25, .75]
    elif change == "component_count": payload["components"].pop()
    elif change == "same_seed": second["seed"] = 47
    elif change == "same_id": second["model_id"] = payload["components"][0]["model_id"]
    elif change == "bad_hash": second["model_sha256"] = "unverified"
    elif change == "library": second["payload"]["library"] = "lightgbm"
    elif change == "problem": second["payload"]["problem"] = "regression"
    elif change == "schema_order": second["payload"]["schema"] = {"b": ["z"], "a": ["x", "y"]}
    elif change == "categories_order": second["payload"]["schema"]["a"].reverse()
    elif change == "schema_type": second["payload"]["schema"]["a"] = ("x", "y")
    elif change == "output_kind": payload["output_kind"] = "raw_logits"
    else: second["weight"] = .8
    with pytest.raises(ValueError):
        ensemble.validate(payload)


def test_identical_component_models_are_allowed_without_seed_reselection():
    payload = ensemble.make_payload(records(values=([1., 2.], [1., 2.])))
    payload["components"][1]["model_sha256"] = payload["components"][0]["model_sha256"]
    ensemble.validate(payload)
    np.testing.assert_allclose(ensemble.predict_normalized(payload, np.zeros((2, 1))), [1., 2.], atol=1e-15)


def test_temperature_applied_once_per_component_before_probability_mean(monkeypatch):
    values = ([[2., 0., -1.]], [[-1., 3., 0.]])
    components = records("multiclass", values=values)
    components[0]["payload"]["temperature"] = 2.
    components[1]["payload"]["temperature"] = .5
    payload = ensemble.make_payload(components)
    calls = []
    original = ensemble.fixed_temperature.apply_temperature
    def counted(component, raw):
        calls.append(component["temperature"])
        return original(component, raw)
    monkeypatch.setattr(ensemble.fixed_temperature, "apply_temperature", counted)
    actual = ensemble.predict_normalized(payload, np.zeros((1, 1)))
    assert calls == [2., .5]
    expected = .5 * (softmax(np.array(values[0]) / 2.) + softmax(np.array(values[1]) / .5))
    np.testing.assert_allclose(np.exp(actual), expected, rtol=1e-15)


def test_shared_raw_normalization_runs_once_and_roundtrip_is_exact(monkeypatch):
    pd = pytest.importorskip("pandas")
    from benchmarks import fair_accuracy_scout as accuracy
    payload = ensemble.make_payload(records(schema={"a": ["x", "y"]}))
    raw = pd.DataFrame({"a": ["x", "y", "missing"]})
    calls = []
    original = accuracy.normalize
    def counted(frame, schema):
        calls.append(schema)
        return original(frame, schema)
    monkeypatch.setattr(accuracy, "normalize", counted)
    expected = ensemble.predict(payload, raw)
    assert len(calls) == 1
    first, second = [record["payload"]["model"] for record in payload["components"]]
    assert first.frames[-1] is second.frames[-1]
    assert list(first.frames[-1]["a"].cat.categories) == ["x", "y"]
    assert pd.isna(first.frames[-1]["a"].iloc[2])
    restored = pickle.loads(pickle.dumps(payload, protocol=5))
    assert type(restored) is dict
    assert [r["model_id"] for r in restored["components"]] == ["toy-47", "toy-947"]
    np.testing.assert_array_equal(ensemble.predict(restored, raw).view(np.uint64), expected.view(np.uint64))


def test_lightgbm_nonunit_binary_link_is_rejected():
    components = records(library="lightgbm")
    components[1]["payload"]["model"].params["sigmoid"] = 2.
    with pytest.raises(ValueError, match="sigmoid"):
        ensemble.make_payload(components)


def test_native_objective_check_is_one_time_and_frozen(monkeypatch):
    payload = ensemble.make_payload(records(library="lightgbm"))
    assert payload["lightgbm_binary_objectives"] == ["binary sigmoid:1.0"] * 2
    def forbidden():
        raise AssertionError("Native model must not be dumped while predicting")
    for record in payload["components"]:
        monkeypatch.setattr(record["payload"]["model"], "dump_model", forbidden)
    ensemble.predict_normalized(payload, np.zeros((3, 1)))
    payload["lightgbm_binary_objectives"][1] = "binary sigmoid:2"
    with pytest.raises(ValueError, match="sigmoid"):
        ensemble.validate(payload)


@pytest.mark.parametrize("sigmoid", [1., 2.])
def test_real_lightgbm_text_reload_uses_native_link_not_empty_params(sigmoid):
    lgb = pytest.importorskip("lightgbm")
    features = np.arange(16, dtype=np.float64).reshape(-1, 1)
    labels = (features[:, 0] >= 8).astype(int)
    native = lgb.train({"objective": "binary", "sigmoid": sigmoid, "verbosity": -1,
                        "num_threads": 1, "num_leaves": 2, "min_data_in_leaf": 1},
                       lgb.Dataset(features, label=labels), num_boost_round=2)
    restored = lgb.Booster(model_str=native.model_to_string())
    # LightGBM4.7 reconstructs params here; older load paths can leave it empty.
    # Clear only Python metadata to prove the intact native link is authoritative.
    restored.params.clear()
    components = records(library="lightgbm")
    for record in components:
        record["payload"]["model"] = restored
    if sigmoid != 1:
        with pytest.raises(ValueError, match="sigmoid"):
            ensemble.make_payload(components)
    else:
        payload = ensemble.make_payload(components)
        ensemble.validate(pickle.loads(pickle.dumps(payload, protocol=5)))


def test_xgb_inplace_guard_is_exact_and_never_persisted(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(DMatrix=lambda frame, **kwargs: frame))
    payload = ensemble.make_payload(records(library="xgboost"))
    before = set(payload)
    frame = np.zeros((3, 1))
    check = ensemble.verify_xgb_inplace(payload, frame)
    assert check["bit_exact"] and set(payload) == before
    np.testing.assert_array_equal(ensemble.predict_normalized(payload, frame).view(np.uint64),
                                  ensemble.predict_normalized(payload, frame, xgb_inplace=True).view(np.uint64))
    payload["components"][1]["payload"]["model"].inplace_delta = 1e-12
    with pytest.raises(AssertionError):
        ensemble.verify_xgb_inplace(payload, frame)
