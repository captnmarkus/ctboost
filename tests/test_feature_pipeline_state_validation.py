import copy
import json
from pathlib import Path

import numpy as np
import pytest

import ctboost
from ctboost.inference_manifest import _fingerprint


def _fitted_pipeline_state():
    pd = pytest.importorskip("pandas")
    rows = 18
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "c"] * 6,
            "numeric": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
            "text": ["alpha beta", "beta gamma", "alpha gamma"] * 6,
            "embedding": [
                np.asarray([index, index % 3, 0.5 * index], dtype=np.float32)
                for index in range(rows)
            ],
        }
    )
    target = np.linspace(0.0, 2.0, rows, dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        cat_features=["category"],
        ordered_ctr=True,
        text_features=["text"],
        text_hash_dim=8,
        text_feature_calcer="tfidf",
        embedding_features=["embedding"],
        embedding_stats=("mean", "l2"),
        embedding_target_features=True,
        embedding_target_mode="regression",
    )
    pipeline.fit(frame, target, feature_names=list(frame.columns))
    return pipeline.to_state()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda state: state.update(n_features_in_=-2),
        lambda state: state["numeric_indices"].append(state["numeric_indices"][0]),
        lambda state: state["categorical_states"][0].update(source_index=10_000),
        lambda state: state["categorical_states"][0]["mapping"].update(a=float("nan")),
        lambda state: state["cat_feature_indices_"].clear(),
        lambda state: state["output_feature_names_"].pop(),
        lambda state: state["ctr_states"][0]["prior_values"].clear(),
        lambda state: state["ctr_states"][0]["total_sums"]["a"].clear(),
        lambda state: state["text_states"][0].update(output_dim=-1),
        lambda state: state["text_states"][0]["idf_values"].pop(),
        lambda state: state["embedding_states"][0]["target_projection_weights"][0].pop(),
        lambda state: state.update(ctr_prior_strength=1.0e308),
        lambda state: state.update(n_features_in_=2_147_483_647),
    ],
)
def test_feature_pipeline_rejects_structurally_unsafe_state(mutate):
    state = copy.deepcopy(_fitted_pipeline_state())
    mutate(state)
    with pytest.raises(ValueError):
        ctboost.FeaturePipeline.from_state(state)


def test_feature_pipeline_large_ngram_bound_is_limited_by_token_count():
    state = copy.deepcopy(_fitted_pipeline_state())
    state["text_ngram_range"] = [1, 2_147_483_647]
    pipeline = ctboost.FeaturePipeline.from_state(state)
    transformed, _, _ = pipeline.transform_array(
        np.asarray([["a", 0.0, "alpha beta", [1.0, 2.0, 3.0]]], dtype=object)
    )
    assert transformed.shape == (1, len(state["output_feature_names_"]))
    assert np.isfinite(transformed).all()


def _legacy_codec_one_pipeline_with_colliding_output_names():
    data = np.asarray([["a-b"], ["a_b"], ["a-b"], ["a_b"]], dtype=object)
    labels = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    fitted = ctboost.FeaturePipeline(cat_features=[0], one_hot_max_size=8)
    fitted.fit(data, labels)
    state = fitted.to_state()
    state["feature_pipeline_format_version"] = 2
    state["categorical_key_encoding_version"] = 1
    state["one_hot_states"][0]["output_names"][1] = state["one_hot_states"][0][
        "output_names"
    ][0]
    state["output_feature_names_"][1] = state["output_feature_names_"][0]
    return ctboost.FeaturePipeline.from_state(state)


def test_legacy_codec_one_colliding_output_names_still_load():
    pipeline = _legacy_codec_one_pipeline_with_colliding_output_names()
    assert pipeline.to_state()["categorical_key_encoding_version"] == 1


def test_raw_export_rejects_legacy_pipeline_before_writing(tmp_path: Path):
    data = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    labels = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    booster = ctboost.train(
        data,
        {"objective": "RMSE", "max_depth": 1, "alpha": 1.0},
        label=labels,
        num_boost_round=2,
    )
    destination = tmp_path / "legacy-raw.json"
    from ctboost._export import export_model

    with pytest.raises(ValueError, match="format 3"):
        export_model(
            destination,
            booster._handle,
            export_format="json_predictor",
            feature_pipeline=_legacy_codec_one_pipeline_with_colliding_output_names(),
        )
    assert not destination.exists()


def test_raw_json_export_validates_self_consistent_pipeline_tampering(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "a", "c", "b", "c"],
            "value": np.linspace(-1.0, 1.0, 6, dtype=np.float32),
        }
    )
    target = np.asarray([0.0, 1.0, 0.2, 1.4, 0.8, 1.7], dtype=np.float32)
    model = ctboost.CTBoostRegressor(
        iterations=3,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
    ).fit(frame, target)
    export_path = tmp_path / "raw-predictor.json"
    model.export_model(export_path, export_format="json_predictor")

    payload = json.loads(export_path.read_text(encoding="utf-8"))
    state = payload["feature_pipeline_state"]
    state["categorical_states"][0]["source_index"] = 1000
    payload["inference_manifest"]["input"]["preprocessing"]["fingerprint"] = (
        _fingerprint(state)
    )
    export_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        ctboost.load_exported_predictor(export_path)


def test_raw_json_export_rejects_legacy_pipeline_before_native_construction(tmp_path: Path):
    state = _fitted_pipeline_state()
    state["feature_pipeline_format_version"] = 2
    state.pop("categorical_key_encoding_version")
    fixture = Path("tests/export_conformance/prepared_binary_v2.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    payload.update(
        expects_prepared_features=False,
        feature_pipeline_state=state,
    )
    path = tmp_path / "legacy-raw.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="format version 3"):
        ctboost.load_exported_predictor(path)
