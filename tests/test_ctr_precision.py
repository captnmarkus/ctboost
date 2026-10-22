"""Mean CTR statistics use double; saved legacy pipelines keep their arithmetic."""
import itertools
import json
import pickle

import numpy as np
import pytest

import ctboost


def pipeline(**kwargs):
    return ctboost.FeaturePipeline(cat_features=[0], ordered_ctr=True,
                                   random_seed=47, **kwargs)


def bits(value):
    return np.asarray(value, dtype=np.float32).view(np.uint32)


@pytest.mark.parametrize("labels", list(itertools.permutations([1.0e8, 1.0, -1.0e8])))
@pytest.mark.parametrize("strength", [0.0, 0.2, 1.0])
def test_cancellation_preserves_prior_totals_and_ordered_prefixes(labels, strength):
    y = np.asarray(labels, dtype=np.float32)
    data = np.asarray([["a"]] * 3, dtype=object)
    model = pipeline(ctr_prior_strength=strength)
    actual = model.fit_transform_array(data, y)[0][:, 1]
    state = model.to_state()
    assert state["feature_pipeline_format_version"] == 5
    assert state["ctr_states"][0]["prior_values"] == [1.0 / 3.0]
    assert state["ctr_states"][0]["total_sums"] == {"a": [1.0]}
    # Enumerate all tiny permutations independently of the native RNG. Each
    # row must use only earlier labels in one valid order, with the same prior.
    possible = []
    for order in itertools.permutations(range(3)):
        result = np.empty(3, dtype=np.float32)
        total = 0.0
        for count, row in enumerate(order):
            denominator = count + strength
            result[row] = (total + strength / 3.0) / (denominator or 1.0)
            total += float(y[row])
        possible.append(bits(result))
    assert any(np.array_equal(bits(actual), expected) for expected in possible)
    queried = model.transform_array(np.asarray([["a"], ["unseen"]], dtype=object))[0][:, 1]
    expected = [1.0 / 3.0, 1.0 / 3.0 if strength else 0.0]
    np.testing.assert_array_equal(bits(queried), bits(expected))
    replay = ctboost.FeaturePipeline.from_state(state)
    np.testing.assert_array_equal(bits(replay.fit_transform_array(data, y)[0][:, 1]), bits(actual))
    assert replay.to_state() == state


def test_finite_average_does_not_require_float_representable_total():
    y = np.full(4, 3.0e38, dtype=np.float32)
    data = np.asarray([["a"]] * 4, dtype=object)
    model = pipeline()
    transformed = model.fit_transform_array(data, y)[0]
    np.testing.assert_array_equal(bits(transformed[:, 1]), bits(y))
    total = model.to_state()["ctr_states"][0]["total_sums"]["a"][0]
    assert total == float(y[0]) * 4 and total > float(np.finfo(np.float32).max)
    query = np.asarray([["a"], ["unseen"]], dtype=object)
    for restored in [ctboost.FeaturePipeline.from_state(json.loads(json.dumps(model.to_state()))),
                     pickle.loads(pickle.dumps(model))]:
        np.testing.assert_array_equal(bits(restored.transform_array(query)[0][:, 1]), bits(y[:2]))
    legacy = model.to_state()
    legacy["feature_pipeline_format_version"] = 4
    with pytest.raises(ValueError, match="finite"):
        ctboost.FeaturePipeline.from_state(legacy)


def test_positive_integer_target_drift_matches_double_reference():
    # This reproduces accumulation drift without requiring mixed-sign labels.
    y = (3000 + np.arange(10001) % 11003).astype(np.float32)
    data = np.asarray([["a"]] * len(y), dtype=object)
    model = pipeline().fit(data, y)
    state = model.to_state()["ctr_states"][0]
    assert state["total_sums"]["a"][0] == int(y.astype(np.int64).sum())
    assert state["prior_values"][0] == y.astype(np.float64).mean()
    expected = np.float32(y.astype(np.float64).mean())
    assert model.transform_array(data[:1])[0][0, 1] == expected


@pytest.mark.parametrize("labels", [[1.0e8, 1.0, -1.0e8], [3.0e38] * 3])
def test_raw_export_retains_double_statistics(tmp_path, labels):
    data = np.asarray([["a"]] * len(labels), dtype=object)
    fitted = pipeline().fit(data, np.asarray(labels, dtype=np.float32))
    query = np.asarray([["a"], ["unseen"]], dtype=object)
    transformed, categorical, names = fitted.transform_array(query)
    # Fit a small root-only booster separately from these deliberately extreme
    # preprocessing targets; export must retain their stored statistics exactly.
    booster = ctboost.train(ctboost.Pool(np.zeros((3, 2), dtype=np.float32),
        [1., 2., 3.], cat_features=categorical, feature_names=names),
        {"objective":"RMSE", "max_depth":1}, num_boost_round=1)
    booster._feature_pipeline = fitted
    path = tmp_path / "double-ctr.json"
    booster.export_model(path)
    restored = ctboost.load_exported_predictor(path)
    assert restored._feature_pipeline.to_state() == fitted.to_state()
    np.testing.assert_array_equal(bits(restored._feature_pipeline.transform_array(query)[0]),
                                  bits(transformed))
    np.testing.assert_array_equal(bits(restored.predict_raw(query)), bits(booster.predict(query)))


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_old_load_rounding_and_refit_keep_legacy_accumulation(version):
    data = np.asarray([["a"], ["a"], ["a"]], dtype=object)
    state = pipeline().fit(data, [1.0e8, 1.0, -1.0e8]).to_state()
    state["feature_pipeline_format_version"] = version
    if version < 3:
        state["categorical_key_encoding_version"] = 1
    state["ctr_states"][0]["prior_values"] = [1.0 + 2.0**-25]
    state["ctr_states"][0]["total_sums"]["a"] = [3.0 + 2.0**-24]
    restored = ctboost.FeaturePipeline.from_state(state)
    stored = restored.to_state()
    assert stored["ctr_states"][0]["prior_values"] == [1.0]
    assert stored["ctr_states"][0]["total_sums"]["a"] == [3.0]
    assert stored["feature_pipeline_format_version"] == (3 if version < 4 else 4)
    np.testing.assert_array_equal(restored.transform_array(data)[0][:, 1], np.ones(3))
    transformed = restored.fit_transform_array(data, [1.0e8, 1.0, -1.0e8])[0]
    replay_state = restored.to_state()
    assert replay_state["feature_pipeline_format_version"] == 4
    assert replay_state["ctr_states"][0]["prior_values"] == [0.0]
    assert replay_state["ctr_states"][0]["total_sums"] == {"a": [0.0]}
    replay = ctboost.FeaturePipeline.from_state(replay_state)
    np.testing.assert_array_equal(bits(replay.fit_transform_array(data, [1.0e8, 1.0, -1.0e8])[0]),
                                  bits(transformed))
    assert replay.to_state() == replay_state


@pytest.mark.parametrize("field", ["prior_values", "total_sums"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_new_statistics_reject_nonfinite_state(field, invalid):
    state = pipeline().fit([["a"]], [2.5]).to_state()
    if field == "prior_values":
        state["ctr_states"][0][field][0] = invalid
    else:
        state["ctr_states"][0][field]["a"][0] = invalid
    with pytest.raises(ValueError, match="finite"):
        ctboost.FeaturePipeline.from_state(state)


def test_new_statistics_still_check_final_float_output_range():
    state = pipeline().fit([["a"]], [2.5]).to_state()
    state["ctr_states"][0]["total_sums"]["a"][0] = 1.0e300
    model = ctboost.FeaturePipeline.from_state(state)
    with pytest.raises(ValueError, match="finite float range"):
        model.transform_array([["a"]])


@pytest.mark.parametrize("version", [4, 5])
def test_wide_dictionary_path_matches_scalar_new_and_legacy_statistics(version):
    pd = pytest.importorskip("pandas")
    width, rows = 32, 1000
    train = pd.DataFrame({f"c{i}": pd.Categorical(["a", "b", "a"] * 4) for i in range(width)})
    y = np.tile([1.0e8, 1.0, -1.0e8], 4).astype(np.float32)
    model = ctboost.FeaturePipeline(cat_features=list(range(width)), ordered_ctr=True).fit(train, y)
    state = model.to_state()
    state["feature_pipeline_format_version"] = version
    model = ctboost.FeaturePipeline.from_state(state)
    query = pd.DataFrame({f"c{i}": pd.Categorical(["a", "b", None, "unseen"] * 250)
                          for i in range(width)})
    expected = model._native.transform_array(query.to_numpy(dtype=object), list(query.columns))[0]
    actual = model.transform_array(query)[0]
    np.testing.assert_array_equal(bits(actual), bits(expected))
    assert actual.shape == (rows, 2 * width)


@pytest.mark.parametrize("version", [3, 4, 5])
def test_frequency_arithmetic_remains_float_compatible(version):
    data = np.asarray([["a"], ["a"], ["b"], ["a"], ["b"]], dtype=object)
    fitted = ctboost.FeaturePipeline(cat_features=[0], simple_ctr=["Frequency"],
                                     ctr_prior_strength=0.2).fit(data, [1, 4, 7, 10, 13])
    state = fitted.to_state()
    state["feature_pipeline_format_version"] = version
    model = ctboost.FeaturePipeline.from_state(state)
    actual = model.transform_array([["a"], ["b"], ["unseen"]])[0][:, 1]
    expected = [(n + 0.2 * float(np.float32(n) / np.float32(5))) / 5.2 for n in [3, 2, 0]]
    np.testing.assert_array_equal(bits(actual), bits(expected))
