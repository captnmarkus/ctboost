"""Raw exports preserve the smoothing semantics of their fitted pipeline."""

import copy
import json
import pickle

import numpy as np
import pytest

import ctboost
from ctboost.export_payload import _standalone_python_payload
from ctboost.export_runtime import ExportedPredictor
from ctboost.inference_manifest import _fingerprint


def _ctr_booster(pipeline_version):
    data = np.asarray([["a"], ["b"], ["c"]] * 12, dtype=object)
    labels = np.asarray([0.0, 10.0, 50.0] * 12, dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0], ordered_ctr=True, ctr_prior_strength=0.2
    ).fit(data, labels)
    state = pipeline.to_state()
    # Versions 3 and 4 store the same sufficient statistics. The version fixes
    # how those statistics are smoothed, including for unseen categories.
    state["feature_pipeline_format_version"] = pipeline_version
    pipeline = ctboost.FeaturePipeline.from_state(state)
    prepared, categorical, names = pipeline.transform_array(data)
    booster = ctboost.train(
        ctboost.Pool(prepared, labels, cat_features=categorical, feature_names=names),
        {
            "objective": "RMSE", "max_depth": 2, "alpha": 1.0,
            "feature_weights": [0.0, 1.0], "random_seed": 3,
        },
        num_boost_round=2,
    )
    booster._feature_pipeline = pipeline
    return booster


@pytest.mark.parametrize("pipeline_version, unseen_ctr", [(3, 4.0), (4, 20.0)])
def test_fractional_ctr_raw_export_and_saved_model_keep_fitted_semantics(
    tmp_path, pipeline_version, unseen_ctr
):
    booster = _ctr_booster(pipeline_version)
    query = np.asarray([["a"], ["b"], ["c"], ["unseen"], [None]], dtype=object)
    state = booster._feature_pipeline.to_state()
    expected_features = booster._feature_pipeline.transform_array(query)[0]
    np.testing.assert_allclose(expected_features[-2:, 1], unseen_ctr)
    expected = booster.predict(query)

    path = tmp_path / "fractional-ctr.ctb"
    booster.save_model(path)
    restored_models = [
        booster,
        ctboost.load_model(path),
        pickle.loads(pickle.dumps(booster)),
    ]
    for index, restored in enumerate(restored_models):
        assert restored._feature_pipeline.to_state() == state
        np.testing.assert_array_equal(restored.predict(query), expected)
        export_path = tmp_path / f"fractional-ctr-{index}.json"
        restored.export_model(export_path)
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        assert payload["format_version"] == 2
        assert payload["feature_pipeline_state"] == json.loads(json.dumps(state))
        predictor = ctboost.load_exported_predictor(export_path)
        assert predictor._feature_pipeline.to_state() == state
        np.testing.assert_array_equal(
            predictor._feature_pipeline.transform_array(query)[0], expected_features
        )
        np.testing.assert_allclose(predictor.predict_raw(query), expected, atol=1e-6)


def test_legacy_and_corrected_ctr_versions_can_change_unseen_predictions():
    legacy = _ctr_booster(3)
    corrected = _ctr_booster(4)
    query = np.asarray([["unseen"]], dtype=object)
    assert legacy.predict(query)[0] != corrected.predict(query)[0]


def test_raw_export_rejects_unknown_pipeline_version_before_model_access():
    with pytest.raises(ValueError, match="format 3 or 4"):
        _standalone_python_payload(
            object(),
            expects_prepared_features=False,
            feature_pipeline_state={
                "feature_pipeline_format_version": 5,
                "categorical_key_encoding_version": 2,
            },
        )


def test_raw_reader_rejects_unknown_pipeline_version_before_native_load(tmp_path, monkeypatch):
    booster = _ctr_booster(4)
    path = tmp_path / "valid.json"
    booster.export_model(path)
    payload = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    payload["feature_pipeline_state"]["feature_pipeline_format_version"] = 5
    payload["inference_manifest"]["input"]["preprocessing"]["fingerprint"] = (
        _fingerprint(payload["feature_pipeline_state"])
    )

    def unexpected_native_load(*args, **kwargs):
        pytest.fail("unknown pipeline version reached native construction")

    monkeypatch.setattr(ctboost.FeaturePipeline, "from_state", unexpected_native_load)
    with pytest.raises(ValueError, match="format version 3 or 4"):
        ExportedPredictor(payload)
