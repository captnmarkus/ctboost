"""CTR priors are pseudo-counts; old fitted pipelines retain their old arithmetic."""

import json
import pickle

import numpy as np
import pytest

import ctboost


def _target(task):
    if task == "regression":
        labels = np.asarray([10.5, 20.5, 30.5, 40.5], dtype=np.float32)
    elif task == "binary":
        labels = np.asarray([0, 1, 0, 1], dtype=np.float32)
    else:
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.float32)
    targets = np.eye(3, dtype=np.float32)[labels.astype(int)] if task == "multiclass" else labels[:, None]
    return labels, targets, targets.mean(axis=0)


@pytest.mark.parametrize("strength", [0.0, 0.2, 1.0, 2.0])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_mean_ctr_first_occurrence_and_unseen_preserve_prior(strength, task):
    labels, targets, prior = _target(task)
    # Every row is a first occurrence, regardless of the ordered permutation.
    data = np.asarray([[f"category-{i}"] for i in range(len(labels))], dtype=object)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0], ordered_ctr=True, ctr_prior_strength=strength
    )
    training = pipeline.fit_transform_array(data, labels)[0][:, 1:]
    cold_prior = prior if strength > 0.0 else np.zeros_like(prior)
    np.testing.assert_allclose(training, np.tile(cold_prior, (len(labels), 1)))
    unseen = pipeline.transform_array(np.asarray([["unseen"]], dtype=object))[0][:, 1:]
    np.testing.assert_allclose(unseen, cold_prior[None, :])
    known = pipeline.transform_array(data)[0][:, 1:]
    expected_known = (
        (targets.astype(np.float64) + strength * prior.astype(np.float64)) / (1.0 + strength)
    ).astype(np.float32)
    np.testing.assert_array_equal(known, expected_known)
    assert np.isfinite(training).all() and np.isfinite(unseen).all()
    if task != "regression":
        assert np.all((training >= 0.0) & (training <= 1.0))
    if task == "multiclass" and strength > 0.0:
        np.testing.assert_allclose(training.sum(axis=1), 1.0)
        np.testing.assert_allclose(unseen.sum(axis=1), 1.0)


@pytest.mark.parametrize("strength", [0.0, 0.2, 1.0, 2.0])
def test_frequency_ctr_uses_the_prior_at_the_first_ordered_row(strength):
    data = np.asarray([["a"], ["b"], ["c"], ["d"]], dtype=object)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0], simple_ctr=["Frequency"], ctr_prior_strength=strength
    )
    training = pipeline.fit_transform_array(data, np.arange(4, dtype=np.float32))[0][:, 1]
    expected = [
        strength * 0.25 / (seen + strength) if seen + strength > 0.0 else 0.0
        for seen in range(4)
    ]
    np.testing.assert_allclose(np.sort(training), np.sort(expected))
    np.testing.assert_allclose(pipeline.transform_array(data)[0][:, 1], 0.25)
    assert pipeline.transform_array(np.asarray([["unseen"]], dtype=object))[0][0, 1] == 0.0


@pytest.mark.parametrize("strength", [0.0, 0.2, 1.0, 2.0])
def test_combination_ctr_preserves_cold_start_prior(strength):
    data = np.asarray([["a", "x"], ["a", "y"], ["b", "x"], ["b", "y"]], dtype=object)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0, 1],
        categorical_combinations=[[0, 1]],
        simple_ctr=[],
        combinations_ctr=["Mean"],
        ctr_prior_strength=strength,
    )
    training = pipeline.fit_transform_array(data, [10.5, 20.5, 30.5, 40.5])[0]
    expected = 25.5 if strength > 0.0 else 0.0
    np.testing.assert_allclose(training[:, -1], expected)
    query = np.asarray([["a", "unseen"]], dtype=object)
    np.testing.assert_allclose(pipeline.transform_array(query)[0][:, -1], expected)


@pytest.mark.parametrize("strength", [1.0e-12, 0.2, 1.0, 2.0])
def test_positive_mean_prior_is_invariant_to_regression_label_translation(strength):
    data = np.asarray([["a"], ["b"], ["a"], ["b"]], dtype=object)
    labels = np.asarray([10.5, 20.5, 30.5, 40.5], dtype=np.float32)
    values = []
    for shift in [0.0, 100.0]:
        pipeline = ctboost.FeaturePipeline(
            cat_features=[0], ordered_ctr=True, ctr_prior_strength=strength
        )
        training = pipeline.fit_transform_array(data, labels + shift)[0][:, 1]
        query = np.asarray([["a"], ["unseen"]], dtype=object)
        values.append(np.concatenate([training, pipeline.transform_array(query)[0][:, 1]]))
    np.testing.assert_allclose(values[1] - values[0], 100.0, atol=1e-5)


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_corrected_ctr_pipeline_state_and_pickle_roundtrip(task):
    labels, _, prior = _target(task)
    data = np.asarray([[f"category-{i}"] for i in range(len(labels))], dtype=object)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0], ordered_ctr=True, ctr_prior_strength=0.2
    ).fit(data, labels)
    query = np.asarray([["category-0"], ["unseen"]], dtype=object)
    expected = pipeline.transform_array(query)[0]
    assert pipeline.to_state()["feature_pipeline_format_version"] == 4
    np.testing.assert_allclose(expected[1, 1:], prior)
    restored = [
        ctboost.FeaturePipeline.from_state(json.loads(json.dumps(pipeline.to_state()))),
        pickle.loads(pickle.dumps(pipeline)),
    ]
    for copy in restored:
        assert copy.to_state()["feature_pipeline_format_version"] == 4
        np.testing.assert_array_equal(copy.transform_array(query)[0], expected)


@pytest.mark.parametrize("format_version", [None, 1, 2, 3])
def test_legacy_ctr_transform_is_preserved_until_explicit_refit(format_version):
    data = np.asarray([["a"], ["b"], ["a"], ["b"]], dtype=object)
    labels = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0], ordered_ctr=True, ctr_prior_strength=0.2
    ).fit(data, labels)
    state = pipeline.to_state()
    if format_version is None:
        state.pop("feature_pipeline_format_version")
    else:
        state["feature_pipeline_format_version"] = format_version
    if format_version != 3:
        state.pop("categorical_key_encoding_version")
    legacy = ctboost.FeaturePipeline.from_state(state)
    query = np.asarray([["a"], ["unseen"]], dtype=object)
    expected = np.asarray([45.0 / 2.2, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(legacy.transform_array(query)[0][:, 1], expected)
    assert legacy.to_state()["feature_pipeline_format_version"] == 3
    for restored in [
        ctboost.FeaturePipeline.from_state(json.loads(json.dumps(legacy.to_state()))),
        pickle.loads(pickle.dumps(legacy)),
    ]:
        np.testing.assert_array_equal(restored.transform_array(query)[0][:, 1], expected)
        assert restored.to_state()["feature_pipeline_format_version"] == 3
    legacy.fit(data, labels)
    assert legacy.to_state()["feature_pipeline_format_version"] == 4
    np.testing.assert_allclose(legacy.transform_array(query)[0][:, 1], [45.0 / 2.2, 25.0])


@pytest.mark.parametrize("num_classes", [2, 3])
def test_fractional_ctr_classifier_probabilities_and_saved_model(num_classes, tmp_path):
    data = np.asarray([[f"category-{i % 9}"] for i in range(54)], dtype=object)
    labels = np.arange(54) % num_classes
    model = ctboost.CTBoostClassifier(
        iterations=4,
        max_depth=2,
        alpha=1.0,
        cat_features=[0],
        ordered_ctr=True,
        ctr_prior_strength=0.2,
        random_seed=3,
    ).fit(data, labels)
    query = np.asarray([["category-0"], ["unseen"], [None]], dtype=object)
    expected = model.predict_proba(query)
    assert expected.shape == (3, num_classes)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(expected.sum(axis=1), 1.0)
    path = tmp_path / "fractional-ctr.ctb"
    model.save_model(path)
    restored = ctboost.CTBoostClassifier.load_model(path)
    assert restored._feature_pipeline.to_state()["feature_pipeline_format_version"] == 4
    np.testing.assert_array_equal(restored.predict_proba(query), expected)
