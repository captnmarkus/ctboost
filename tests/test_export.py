import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

import ctboost
from ctboost.inference_manifest import _model_fingerprint


def test_booster_export_model_generates_standalone_python_predictor(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=29,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    booster = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        label=y,
        num_boost_round=8,
    )

    export_path = tmp_path / "standalone_predictor.py"
    booster.export_model(export_path)

    spec = importlib.util.spec_from_file_location("ctboost_standalone_predictor", export_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    standalone_pred = np.asarray(module.predict(X), dtype=np.float32)
    np.testing.assert_allclose(standalone_pred, booster.predict(X), rtol=1e-6, atol=1e-6)
    single_prediction = float(module.predict(X[0]))
    np.testing.assert_allclose(single_prediction, booster.predict(X[:1])[0], rtol=1e-6, atol=1e-6)

def test_booster_export_model_generates_json_predictor(tmp_path: Path):
    X, y = make_regression(
        n_samples=96,
        n_features=5,
        n_informative=4,
        noise=0.1,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    booster = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        label=y,
        num_boost_round=8,
    )

    export_path = tmp_path / "predictor.json"
    booster.export_model(export_path, export_format="json_predictor")

    predictor = ctboost.load_exported_predictor(export_path)
    exported_pred = np.asarray(predictor.predict(X), dtype=np.float32)
    np.testing.assert_allclose(exported_pred, booster.predict(X), rtol=1e-6, atol=1e-6)


def test_json_predictor_loader_rejects_duplicate_nonfinite_and_oversize_artifacts(
    tmp_path: Path,
):
    fixture = (
        Path(__file__).parent
        / "export_conformance"
        / "prepared_regression_v1.json"
    )
    source = fixture.read_text(encoding="utf-8")
    duplicate = source.replace(
        '"format": "ctboost-json-predictor",',
        '"format": "ctboost-json-predictor",\n  "format": "duplicate",',
        1,
    )
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        ctboost.load_exported_predictor(duplicate_path)

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text(
        source.replace('"learning_rate": 0.5', '"learning_rate": NaN', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        ctboost.load_exported_predictor(nonfinite_path)

    size = fixture.stat().st_size
    assert ctboost.load_exported_predictor(
        fixture, max_artifact_bytes=size
    ).predict([0.0, 20.0]) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="size limit"):
        ctboost.load_exported_predictor(fixture, max_artifact_bytes=size - 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["quantization_schema"]["categorical_mask"].__setitem__(0, "false"),
        lambda payload: payload["quantization_schema"]["cut_values"].__setitem__(0, float("nan")),
        lambda payload: payload["trees"][0]["nodes"][0].update(split_feature_id=999),
        lambda payload: payload["trees"][0]["nodes"][0].update(left_child=0),
        lambda payload: payload.update(trees=[]),
        lambda payload: payload.update(prediction_dimension=2, base_score=[0.0, 0.0]),
    ],
)
def test_json_predictor_rejects_malformed_quantization_and_tree_state(mutate):
    from ctboost.export_runtime import ExportedPredictor

    fixture = (
        Path(__file__).parent
        / "export_conformance"
        / "prepared_regression_v1.json"
    )
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    mutate(payload)
    with pytest.raises(ValueError):
        ExportedPredictor(payload)


def test_json_predictor_embeds_fitted_categorical_text_and_embedding_pipeline(
    tmp_path: Path,
):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(20260814)
    rows = 72
    category = np.where(np.arange(rows) % 3 == 0, "alpha", "beta")
    embeddings = [
        np.asarray([index % 3, index % 5, index % 7], dtype=np.float32)
        for index in range(rows)
    ]
    frame = pd.DataFrame(
        {
            "category": category,
            "text": np.where(category == "alpha", "red quick fox", "blue slow fox"),
            "embedding": embeddings,
            "numeric": rng.normal(size=rows).astype(np.float32),
        }
    )
    target = (
        (category == "alpha").astype(np.float32)
        + 0.35 * frame["numeric"].to_numpy(dtype=np.float32)
        + np.asarray([value[0] - 0.1 * value[1] for value in embeddings], dtype=np.float32)
    )
    model = ctboost.CTBoostRegressor(
        iterations=8,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
        text_features=["text"],
        text_hash_dim=12,
        embedding_features=["embedding"],
        embedding_stats=("mean", "std", "l2"),
    ).fit(frame, target)

    export_path = tmp_path / "pipeline_predictor.json"
    model.export_model(export_path, export_format="json_predictor")
    document = json.loads(export_path.read_text(encoding="utf-8"))
    assert document["format_version"] == 2
    assert document["expects_prepared_features"] is False
    assert document["feature_pipeline_state"]["feature_names_in_"] == list(frame.columns)
    manifest = document["inference_manifest"]
    assert manifest["artifact"]["ctboost_runtime_required"] is True
    assert manifest["input"]["representation"] == "raw_features"
    assert manifest["input"]["preprocessing"]["external_preprocessing_required"] is False

    predictor = ctboost.load_exported_predictor(export_path)
    np.testing.assert_allclose(
        predictor.predict(frame),
        model.predict(frame),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        predictor.predict(frame.iloc[0].to_numpy(dtype=object)),
        model.predict(frame.iloc[[0]])[0],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        predictor.predict(frame.iloc[0].tolist()),
        model.predict(frame.iloc[[0]])[0],
        rtol=1e-6,
        atol=1e-6,
    )
    assert predictor.predict([]) == []

    for mutate in (
        lambda payload: payload.update(feature_pipeline_state=None),
        lambda payload: payload.update(expects_prepared_features=True),
        lambda payload: payload["inference_manifest"]["input"].update(
            representation="prepared_numeric_features"
        ),
        lambda payload: payload["inference_manifest"]["input"][
            "preprocessing"
        ].update(raw_feature_count=999),
        lambda payload: payload["inference_manifest"]["input"][
            "preprocessing"
        ].update(transformed_feature_count=999),
        lambda payload: payload["inference_manifest"]["input"].update(
            model_feature_count=999
        ),
        lambda payload: payload["inference_manifest"]["artifact"].update(
            ctboost_runtime_required=False
        ),
        lambda payload: payload["inference_manifest"]["model"].update(
            objective="Logloss"
        ),
    ):
        invalid = copy.deepcopy(document)
        mutate(invalid)
        invalid_path = tmp_path / f"invalid-envelope-{id(mutate)}.json"
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(ValueError):
            ctboost.load_exported_predictor(invalid_path)

    invalid_layout = copy.deepcopy(document)
    invalid_layout["quantization_schema"]["categorical_mask"] = [
        0
        for _ in invalid_layout["quantization_schema"]["categorical_mask"]
    ]
    invalid_layout["inference_manifest"]["model"]["fingerprint"] = (
        _model_fingerprint(invalid_layout, invalid_layout.get("class_labels"))
    )
    invalid_layout_path = tmp_path / "invalid-categorical-layout.json"
    invalid_layout_path.write_text(json.dumps(invalid_layout), encoding="utf-8")
    with pytest.raises(ValueError):
        ctboost.load_exported_predictor(invalid_layout_path)

    document["feature_pipeline_state"]["text_hash_dim"] += 1
    tampered_path = tmp_path / "tampered_pipeline_predictor.json"
    tampered_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="feature-pipeline fingerprint mismatch"):
        ctboost.load_exported_predictor(tampered_path)


def test_booster_export_model_generates_cpp17_c_api(tmp_path: Path):
    X, y = make_regression(
        n_samples=64,
        n_features=4,
        n_informative=3,
        random_state=33,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    booster = ctboost.train(
        X,
        {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
        label=y,
        num_boost_round=4,
    )

    export_path = tmp_path / "ctboost_model.cpp"
    booster.export_model(export_path)
    source = export_path.read_text(encoding="utf-8")

    assert "ctboost_predict(" in source
    assert "ctboost_inference_manifest_json" in source
    assert "static constexpr Node kNodes[]" in source
    assert '"kind":"standalone_cpp"' in source
    assert "rows > output_size / kPredictionDimension" in source
    assert "rows > std::numeric_limits<std::size_t>::max() / columns" in source


def test_onnx_export_matches_native_regression_and_binary_probabilities(tmp_path: Path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    X, y = make_classification(
        n_samples=96,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=35,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    model = ctboost.CTBoostClassifier(
        iterations=7,
        learning_rate=0.2,
        max_depth=3,
        alpha=1.0,
        random_seed=5,
    ).fit(X, y)

    export_path = tmp_path / "classifier.onnx"
    model.export_model(export_path)
    session = onnxruntime.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    raw, probabilities = session.run(None, {"features": X})

    np.testing.assert_allclose(raw, model.get_booster().predict(X), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(probabilities, model.predict_proba(X), rtol=2e-5, atol=2e-5)


def test_onnx_export_matches_native_missing_value_routes(tmp_path: Path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(39)
    X = rng.normal(size=(80, 4)).astype(np.float32)
    X[::7, 1] = np.nan
    y = (np.nan_to_num(X[:, 1]) + X[:, 2]).astype(np.float32)
    booster = ctboost.train(
        X,
        {"objective": "RMSE", "max_depth": 3, "alpha": 1.0, "nan_mode": "Max"},
        label=y,
        num_boost_round=6,
    )

    export_path = tmp_path / "regressor.onnx"
    booster.export_model(export_path)
    session = onnxruntime.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    (raw,) = session.run(None, {"features": X})

    np.testing.assert_allclose(raw, booster.predict(X), rtol=2e-5, atol=2e-5)


def test_onnx_export_matches_native_categorical_bin_routes(tmp_path: Path):
    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    X = np.asarray(
        [[0.0, -1.0], [2.0, 0.0], [5.0, 1.0], [0.0, 2.0], [2.0, 3.0], [5.0, 4.0]]
        * 10,
        dtype=np.float32,
    )
    y = (X[:, 0] == 5.0).astype(np.float32)
    pool = ctboost.Pool(X, y, cat_features=[0])
    booster = ctboost.train(
        pool,
        {"objective": "RMSE", "max_depth": 2, "alpha": 1.0},
        num_boost_round=5,
    )
    prediction_data = np.asarray(
        [[0.0, 0.5], [2.0, 0.5], [5.0, 0.5], [3.0, 0.5]], dtype=np.float32
    )
    prediction_pool = ctboost.Pool(prediction_data, cat_features=[0])

    export_path = tmp_path / "categorical.onnx"
    booster.export_model(export_path)
    session = onnxruntime.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
    (raw,) = session.run(None, {"features": prediction_data})

    np.testing.assert_allclose(raw, booster.predict(prediction_pool), rtol=2e-5, atol=2e-5)

def test_estimator_export_model_matches_predict_proba(tmp_path: Path):
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=37,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    export_path = tmp_path / "standalone_classifier.py"
    clf.export_model(export_path)

    spec = importlib.util.spec_from_file_location("ctboost_standalone_classifier", export_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    standalone_proba = np.asarray(module.predict_proba(X), dtype=np.float32)
    np.testing.assert_allclose(standalone_proba, clf.predict_proba(X), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(module.predict_class(X), dtype=np.int32), clf.predict(X))

    with pytest.raises(ValueError, match="class_labels size"):
        clf.get_booster().export_model(
            tmp_path / "invalid-labels.json",
            export_format="json_predictor",
            class_labels=["only-one-label"],
        )
