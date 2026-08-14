import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import ctboost
from ctboost.inference_manifest import _output_contract


def _load_generated_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_booster_inference_manifest_captures_schema_and_stable_fingerprints(tmp_path: Path):
    X, y = make_regression(n_samples=80, n_features=3, random_state=42)
    pool = ctboost.Pool(
        X.astype(np.float32),
        y.astype(np.float32),
        feature_names=["score", "city_code", "ratio"],
        cat_features=[1],
        column_roles=["numeric", "categorical", "numeric"],
        feature_metadata={"score": {"unit": "z-score"}},
        categorical_schema={"city_code": {"categories": [0, 1, 2]}},
    )
    booster = ctboost.train(
        pool,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
        },
        num_boost_round=6,
    )

    manifest = booster.get_inference_manifest()
    assert manifest["format"] == "ctboost-inference-manifest"
    assert manifest["schema_version"] == 1
    assert manifest["producer"]["version"] == ctboost.__version__
    assert manifest["producer"]["build_fingerprint"].startswith("sha256:")
    assert manifest["model"]["fingerprint"].startswith("sha256:")
    assert manifest["model"]["tree_count"] == booster.num_iterations_trained
    assert manifest["output"]["task"] == "regression"
    assert manifest["output"]["default_method"] == "predict_raw"
    assert manifest["input"]["num_features"] == 3
    assert [item["name"] for item in manifest["input"]["features"]] == [
        "score",
        "city_code",
        "ratio",
    ]
    assert manifest["input"]["features"][0]["metadata"] == {"unit": "z-score"}
    assert manifest["input"]["features"][1]["categorical"] is True
    assert manifest["input"]["features"][1]["categorical_schema"] == {
        "categories": [0, 1, 2]
    }

    second_manifest = booster.get_inference_manifest()
    assert second_manifest["producer"]["build_fingerprint"] == manifest["producer"]["build_fingerprint"]
    assert second_manifest["model"]["fingerprint"] == manifest["model"]["fingerprint"]

    manifest_path = tmp_path / "inference-manifest.json"
    booster.export_inference_manifest(manifest_path)
    assert ctboost.load_inference_manifest(manifest_path) == manifest

    predictor_path = tmp_path / "predictor.json"
    booster.export_model(predictor_path, export_format="json_predictor")
    embedded = ctboost.load_inference_manifest(predictor_path)
    assert embedded["artifact"]["kind"] == "json_predictor"
    assert embedded["artifact"]["ctboost_runtime_required"] is False
    assert embedded["model"]["fingerprint"] == manifest["model"]["fingerprint"]
    assert ctboost.load_exported_predictor(predictor_path).get_inference_manifest() == embedded

    tampered_document = json.loads(predictor_path.read_text(encoding="utf-8"))
    tampered_document["trees"][0]["nodes"][-1]["leaf_weight"] += 1.0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered_document), encoding="utf-8")
    with pytest.raises(ValueError, match="model fingerprint mismatch"):
        ctboost.load_exported_predictor(tampered_path)
    with pytest.raises(ValueError, match="model fingerprint mismatch"):
        ctboost.load_inference_manifest(tampered_path)


def test_standalone_classifier_preserves_labels_and_embeds_output_contract(tmp_path: Path):
    X, numeric_y = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    labels = np.where(numeric_y == 0, "decline", "approve")
    classifier = ctboost.CTBoostClassifier(
        iterations=8,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
    ).fit(X.astype(np.float32), labels)

    python_path = tmp_path / "classifier.py"
    classifier.export_model(python_path)
    generated = _load_generated_module(python_path, "standalone_classifier_with_manifest")
    np.testing.assert_array_equal(generated.predict_class(X[:12]), classifier.predict(X[:12]))
    generated_manifest = generated.get_inference_manifest()
    assert generated_manifest["artifact"]["kind"] == "standalone_python"
    assert generated_manifest["artifact"]["ctboost_runtime_required"] is False
    assert generated_manifest["artifact"]["estimator"] == "CTBoostClassifier"
    assert generated_manifest["output"]["default_method"] == "predict_raw"
    assert generated_manifest["output"]["methods"]["predict_proba"]["link"] == "sigmoid"
    assert generated_manifest["output"]["methods"]["predict_class"]["labels"] == [
        "approve",
        "decline",
    ]

    manifest_path = tmp_path / "classifier-manifest.json"
    classifier.export_inference_manifest(manifest_path)
    native_manifest = ctboost.load_inference_manifest(manifest_path)
    assert native_manifest["artifact"]["kind"] == "ctboost_model"
    assert native_manifest["output"]["default_method"] == "predict"
    assert native_manifest["output"]["methods"]["predict_class"]["labels"] == [
        "approve",
        "decline",
    ]

    json_path = tmp_path / "classifier.json"
    classifier.export_model(json_path, export_format="json_predictor")
    predictor = ctboost.load_exported_predictor(json_path)
    np.testing.assert_array_equal(predictor.predict_class(X[:12]), classifier.predict(X[:12]))


def test_feature_pipeline_manifest_distinguishes_raw_and_prepared_inputs(tmp_path: Path):
    X = np.asarray(
        [[0.1, "red"], [0.2, "blue"], [0.4, "red"], [0.8, "green"]] * 8,
        dtype=object,
    )
    y = np.asarray([0.0, 1.0, 0.0, 1.0] * 8, dtype=np.float32)
    model = ctboost.CTBoostRegressor(
        iterations=4,
        max_depth=1,
        alpha=1.0,
        cat_features=[1],
    ).fit(
        X,
        y,
    )
    # Array input has no names, so fit a low-level model to exercise named pipeline metadata.
    named = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "cat_features": ["color"],
            "max_depth": 1,
            "alpha": 1.0,
        },
        label=y,
        feature_names=["value", "color"],
        num_boost_round=4,
    )

    raw_manifest = named.get_inference_manifest()
    assert raw_manifest["input"]["representation"] == "raw_features"
    assert raw_manifest["input"]["num_features"] == 2
    assert raw_manifest["input"]["categorical_feature_indices"] == [1]
    assert raw_manifest["input"]["preprocessing"]["external_preprocessing_required"] is False
    assert raw_manifest["input"]["preprocessing"]["fingerprint"].startswith("sha256:")

    with pytest.raises(ValueError, match="prepared_features=True"):
        named.export_model(tmp_path / "unsupported.py")

    raw_path = tmp_path / "raw.json"
    named.export_model(raw_path, export_format="json_predictor")
    raw_predictor = ctboost.load_exported_predictor(raw_path)
    exported_raw_manifest = raw_predictor.get_inference_manifest()
    assert exported_raw_manifest["input"]["representation"] == "raw_features"
    assert exported_raw_manifest["input"]["categorical_feature_indices"] == [1]
    assert exported_raw_manifest["artifact"]["ctboost_runtime_required"] is True
    np.testing.assert_allclose(raw_predictor.predict(X), named.predict(X), rtol=1e-6, atol=1e-6)

    prepared_path = tmp_path / "prepared.json"
    named.export_model(
        prepared_path,
        export_format="json_predictor",
        prepared_features=True,
    )
    prepared_manifest = ctboost.load_inference_manifest(prepared_path)
    assert prepared_manifest["input"]["representation"] == "prepared_numeric_features"
    assert prepared_manifest["input"]["preprocessing"]["external_preprocessing_required"] is True
    prepared_document = json.loads(prepared_path.read_text(encoding="utf-8"))
    assert prepared_document["feature_pipeline_state"] is None

    assert model.get_inference_manifest()["artifact"]["estimator"] == "CTBoostRegressor"


def test_manifest_loader_and_validator_reject_incompatible_documents(tmp_path: Path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a CTBoost inference manifest"):
        ctboost.load_inference_manifest(invalid)

    with pytest.raises(ValueError, match="schema version"):
        ctboost.validate_inference_manifest(
            {"format": "ctboost-inference-manifest", "schema_version": 99}
        )


def test_manifest_validator_rejects_inconsistent_deployment_contract():
    X, y = make_regression(n_samples=32, n_features=2, random_state=91)
    model = ctboost.train(
        X.astype(np.float32),
        {"objective": "RMSE", "alpha": 1.0, "max_depth": 1},
        label=y.astype(np.float32),
        num_boost_round=2,
    )
    manifest = model.get_inference_manifest()

    mutations = (
        lambda value: value["model"].update(prediction_dimension=0, base_score=[]),
        lambda value: value["model"].update(tree_count=0),
        lambda value: value["model"].update(iteration_count=999),
        lambda value: value["input"].update(model_feature_count=True),
        lambda value: value["output"].update(prediction_dimension=True),
        lambda value: value["output"].update(objective="Logloss"),
        lambda value: value["input"].update(representation="raw_features"),
        lambda value: value["input"].update(training_schema={"invalid": float("nan")}),
    )
    for mutate in mutations:
        invalid = copy.deepcopy(manifest)
        mutate(invalid)
        with pytest.raises(ValueError):
            ctboost.validate_inference_manifest(invalid)


@pytest.mark.parametrize("objective", ["LambdaMART", "LambdaRank", "rank:ndcg"])
def test_manifest_recognizes_ndcg_ranking_objectives(objective):
    contract = _output_contract(
        objective,
        1,
        None,
        artifact_kind="ctboost_model",
        estimator_name="CTBoostRanker",
    )
    assert contract["task"] == "ranking"


@pytest.mark.parametrize(
    "objective", ["SurvivalExponential", "survival:exponential", "survival:aft"]
)
def test_manifest_recognizes_survival_objectives(objective):
    contract = _output_contract(
        objective,
        1,
        None,
        artifact_kind="ctboost_model",
        estimator_name=None,
    )
    assert contract["task"] == "survival"
