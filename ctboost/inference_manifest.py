"""Versioned deployment metadata for CTBoost inference artifacts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ._version import __version__

PathLike = Union[str, Path]

INFERENCE_MANIFEST_FORMAT = "ctboost-inference-manifest"
INFERENCE_MANIFEST_SCHEMA_VERSION = 2
_FINGERPRINT_PREFIX = "sha256:"
_MODEL_IDENTITY_KEYS = (
    "objective_name",
    "learning_rate",
    "tree_learning_rates",
    "base_score",
    "prediction_dimension",
    "multi_strategy",
    "num_features",
    "quantization_schema",
    "trees",
)


def _json_ready(value: Any) -> Any:
    """Return a deterministic, JSON-compatible representation."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return _json_ready(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{_FINGERPRINT_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def _model_fingerprint(
    scoring_payload: Mapping[str, Any],
    class_labels: Optional[list[Any]] = None,
) -> str:
    identity = {
        key: scoring_payload[key]
        for key in _MODEL_IDENTITY_KEYS
        if key in scoring_payload
    }
    if class_labels is not None:
        identity["class_labels"] = _json_ready(class_labels)
    return _fingerprint(identity)


def _build_details() -> dict[str, Any]:
    from . import _core

    details = dict(_core.build_info())
    details["version"] = __version__
    return _json_ready(details)


def _feature_value(metadata: Any, *, index: int, name: str) -> Any:
    if not isinstance(metadata, Mapping):
        return None
    for key in (name, str(index), index):
        if key in metadata:
            return _json_ready(metadata[key])
    return None


def _normalized_feature_names(value: Any, num_features: int) -> list[str]:
    if value is None:
        return [f"f{index}" for index in range(num_features)]
    names = [str(name) for name in value]
    if len(names) != num_features:
        return [f"f{index}" for index in range(num_features)]
    return names


def _input_contract(
    *,
    scoring_payload: Mapping[str, Any],
    data_schema: Optional[Mapping[str, Any]],
    feature_pipeline_state: Optional[Mapping[str, Any]],
    expects_prepared_features: bool,
) -> dict[str, Any]:
    model_feature_count = int(scoring_payload["num_features"])
    schema = {} if data_schema is None else dict(data_schema)
    pipeline = None if feature_pipeline_state is None else dict(feature_pipeline_state)

    if pipeline is not None and not expects_prepared_features:
        num_features = int(pipeline.get("n_features_in_", model_feature_count))
        names = _normalized_feature_names(pipeline.get("feature_names_in_"), num_features)
        categorical_indices = [int(value) for value in pipeline.get("cat_feature_indices_", [])]
        representation = "raw_features"
    else:
        num_features = model_feature_count
        output_names = None if pipeline is None else pipeline.get("output_feature_names_")
        names = _normalized_feature_names(output_names or schema.get("feature_names"), num_features)
        categorical_mask = list(
            scoring_payload["quantization_schema"].get("categorical_mask", [])
        )
        categorical_indices = [
            index for index, is_categorical in enumerate(categorical_mask) if is_categorical
        ]
        representation = (
            "prepared_numeric_features"
            if pipeline is not None
            else "numeric_or_preencoded_categorical_features"
        )

    roles = schema.get("column_roles")
    if roles is None or len(roles) != num_features:
        roles = [None] * num_features
    else:
        roles = [None if role is None else str(role) for role in roles]
    feature_metadata = schema.get("feature_metadata")
    categorical_schema = schema.get("categorical_schema")
    categorical_set = set(categorical_indices)
    text_set: set[int] = set()
    embedding_set: set[int] = set()
    if pipeline is not None and not expects_prepared_features:
        text_set = {
            int(state["source_index"])
            for state in pipeline.get("text_states", [])
            if "source_index" in state
        }
        embedding_set = {
            int(state["source_index"])
            for state in pipeline.get("embedding_states", [])
            if "source_index" in state
        }
    features = []
    for index, name in enumerate(names):
        role = roles[index]
        if role is None:
            role = (
                "categorical"
                if index in categorical_set
                else "text"
                if index in text_set
                else "embedding"
                if index in embedding_set
                else "numeric"
            )
        feature = {
            "index": index,
            "name": name,
            "role": role,
            "categorical": index in categorical_set,
        }
        metadata = _feature_value(feature_metadata, index=index, name=name)
        if metadata is not None:
            feature["metadata"] = metadata
        category_contract = _feature_value(categorical_schema, index=index, name=name)
        if category_contract is not None:
            feature["categorical_schema"] = category_contract
        features.append(feature)

    preprocessing: dict[str, Any]
    if pipeline is None:
        preprocessing = {
            "kind": "none",
            "external_preprocessing_required": False,
        }
    else:
        preprocessing = {
            "kind": "ctboost_feature_pipeline",
            "external_preprocessing_required": bool(expects_prepared_features),
            "raw_feature_count": int(pipeline.get("n_features_in_", num_features)),
            "transformed_feature_count": model_feature_count,
            "fingerprint": _fingerprint(pipeline),
        }

    return {
        "representation": representation,
        "num_features": num_features,
        "model_feature_count": model_feature_count,
        "features": features,
        "categorical_feature_indices": categorical_indices,
        "missing_values": {
            "accepted": True,
            "representations": ["NaN", "None"],
            "policy": "per-feature quantization schema",
        },
        "preprocessing": preprocessing,
        "training_schema": _json_ready(schema),
    }


def _output_contract(
    objective_name: str,
    prediction_dimension: int,
    class_labels: Optional[list[Any]],
    *,
    artifact_kind: str,
    estimator_name: Optional[str],
) -> dict[str, Any]:
    normalized = objective_name.strip().lower()
    binary = normalized in {"logloss", "binary_logloss", "binary:logistic"}
    multiclass = normalized in {"multiclass", "softmax", "softmaxloss"}
    ranking = normalized in {
        "pairlogit",
        "pairwise",
        "rank:pairwise",
        "lambdamart",
        "lambdarank",
        "rank:ndcg",
    }
    survival = normalized in {
        "cox",
        "coxph",
        "survival:cox",
        "exponential",
        "survivalexponential",
        "survival:exponential",
        "aft",
        "survival:aft",
    }
    task = (
        "classification"
        if binary or multiclass
        else "ranking"
        if ranking
        else "survival"
        if survival
        else "regression"
    )
    shape = (
        {"single_row": "scalar", "batch": ["n_rows"]}
        if prediction_dimension == 1
        else {
            "single_row": [prediction_dimension],
            "batch": ["n_rows", prediction_dimension],
        }
    )
    methods: dict[str, Any] = {
        "predict_raw": {
            "semantics": "additive raw score (margin)",
            "dtype": "floating-point",
            "shape": shape,
        },
        "predict": {
            "semantics": "alias of predict_raw for standalone predictors",
            "dtype": "floating-point",
            "shape": shape,
        },
    }
    default_method = "predict_raw"
    if binary or multiclass:
        probability_dimension = 2 if binary else prediction_dimension
        methods["predict_proba"] = {
            "semantics": "class probabilities",
            "link": "sigmoid" if binary else "softmax",
            "class_order": (
                list(range(probability_dimension))
                if class_labels is None
                else _json_ready(class_labels)
            ),
            "shape": {
                "single_row": [probability_dimension],
                "batch": ["n_rows", probability_dimension],
            },
        }
        class_contract = {
            "semantics": "maximum-probability class label",
            "labels": (
                list(range(probability_dimension))
                if class_labels is None
                else _json_ready(class_labels)
            ),
            "shape": {"single_row": "scalar", "batch": ["n_rows"]},
        }
        methods["predict_class"] = class_contract
        if (
            artifact_kind == "ctboost_model"
            and estimator_name is not None
            and estimator_name.endswith("Classifier")
        ):
            methods["predict"] = dict(class_contract)
            default_method = "predict"

    return {
        "task": task,
        "objective": objective_name,
        "prediction_dimension": prediction_dimension,
        "default_method": default_method,
        "methods": methods,
    }


def build_inference_manifest(
    scoring_payload: Mapping[str, Any],
    *,
    data_schema: Optional[Mapping[str, Any]] = None,
    feature_pipeline_state: Optional[Mapping[str, Any]] = None,
    expects_prepared_features: bool,
    artifact_kind: str,
    class_labels: Optional[list[Any]] = None,
    estimator_name: Optional[str] = None,
) -> dict[str, Any]:
    """Build a stable, JSON-serializable inference contract for a trained model."""
    objective_name = str(scoring_payload["objective_name"])
    prediction_dimension = int(scoring_payload["prediction_dimension"])
    vector_leaves = scoring_payload.get("multi_strategy") == "multi_output_tree"
    trees_per_iteration = 1 if vector_leaves else prediction_dimension
    build = _build_details()
    manifest = {
        "format": INFERENCE_MANIFEST_FORMAT,
        "schema_version": INFERENCE_MANIFEST_SCHEMA_VERSION if vector_leaves else 1,
        "producer": {
            "name": "ctboost",
            "version": __version__,
            "build": build,
            "build_fingerprint": _fingerprint(build),
        },
        "artifact": {
            "kind": str(artifact_kind),
            "ctboost_runtime_required": artifact_kind == "ctboost_model",
            "estimator": estimator_name,
        },
        "model": {
            "fingerprint": _model_fingerprint(scoring_payload, class_labels),
            "fingerprint_algorithm": "sha256",
            "objective": objective_name,
            "tree_count": len(scoring_payload["trees"]),
            "iteration_count": len(scoring_payload.get("tree_learning_rates", []))
            or len(scoring_payload["trees"]) // trees_per_iteration,
            "prediction_dimension": prediction_dimension,
            "base_score": [float(value) for value in scoring_payload.get("base_score", ())],
        },
        "input": _input_contract(
            scoring_payload=scoring_payload,
            data_schema=data_schema,
            feature_pipeline_state=feature_pipeline_state,
            expects_prepared_features=expects_prepared_features,
        ),
        "output": _output_contract(
            objective_name,
            prediction_dimension,
            class_labels,
            artifact_kind=artifact_kind,
            estimator_name=estimator_name,
        ),
    }
    if vector_leaves:
        expanded = artifact_kind in {"standalone_cpp", "onnx"}
        manifest["model"].update({
            "multi_strategy": "multi_output_tree",
            "tree_representation": "shared_topology_vector_leaves",
            "trees_per_iteration": 1,
        })
        manifest["artifact"].update({
            "tree_representation": "expanded_scalar_trees" if expanded else "shared_topology_vector_leaves",
            "exported_tree_count": len(scoring_payload["trees"]) * (prediction_dimension if expanded else 1),
        })
    return validate_inference_manifest(manifest)


def validate_inference_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a defensive copy of a supported manifest."""
    document = _json_ready(manifest)
    if document.get("format") != INFERENCE_MANIFEST_FORMAT:
        raise ValueError("not a CTBoost inference manifest")
    if document.get("schema_version") not in (1, INFERENCE_MANIFEST_SCHEMA_VERSION):
        raise ValueError(
            "unsupported CTBoost inference manifest schema version: "
            f"{document.get('schema_version')!r}"
        )
    for section in ("producer", "artifact", "model", "input", "output"):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"inference manifest is missing the {section!r} section")
    if document["schema_version"] == 2:
        model = document["model"]
        if (
            model.get("multi_strategy") != "multi_output_tree"
            or model.get("tree_representation") != "shared_topology_vector_leaves"
            or model.get("trees_per_iteration") != 1
            or model.get("tree_count") != model.get("iteration_count")
            or not isinstance(model.get("tree_count"), int)
            or model["tree_count"] < 0
            or not isinstance(model.get("prediction_dimension"), int)
            or model["prediction_dimension"] <= 1
        ):
            raise ValueError("inference manifest vector tree layout is inconsistent")
        artifact = document["artifact"]
        expanded = artifact.get("tree_representation") == "expanded_scalar_trees"
        if artifact.get("tree_representation") not in {
            "shared_topology_vector_leaves", "expanded_scalar_trees"
        }:
            raise ValueError("inference manifest vector artifact representation is invalid")
        expected_count = model["tree_count"] * (model["prediction_dimension"] if expanded else 1)
        if artifact.get("exported_tree_count") != expected_count:
            raise ValueError("inference manifest vector exported tree count is inconsistent")
    for path, value in (
        ("producer.build_fingerprint", document["producer"].get("build_fingerprint")),
        ("model.fingerprint", document["model"].get("fingerprint")),
    ):
        digest = (
            ""
            if not isinstance(value, str)
            else value[len(_FINGERPRINT_PREFIX) :]
            if value.startswith(_FINGERPRINT_PREFIX)
            else value
        )
        if (
            not isinstance(value, str)
            or not value.startswith(_FINGERPRINT_PREFIX)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"inference manifest {path} must be a SHA-256 fingerprint")
    expected_build_fingerprint = _fingerprint(document["producer"].get("build"))
    if document["producer"].get("build_fingerprint") != expected_build_fingerprint:
        raise ValueError("inference manifest producer build fingerprint mismatch")
    num_features = document["input"].get("num_features")
    features = document["input"].get("features")
    if not isinstance(num_features, int) or num_features < 0:
        raise ValueError("inference manifest input.num_features must be a non-negative integer")
    if not isinstance(features, list) or len(features) != num_features:
        raise ValueError("inference manifest features do not match input.num_features")
    return deepcopy(document)


def load_inference_manifest(path: PathLike) -> dict[str, Any]:
    """Load a standalone manifest or the manifest embedded in a JSON predictor."""
    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    manifest = document.get("inference_manifest", document)
    if not isinstance(manifest, Mapping):
        raise ValueError("file does not contain a CTBoost inference manifest")
    validated = validate_inference_manifest(manifest)
    if "inference_manifest" in document:
        expected = _model_fingerprint(document, document.get("class_labels"))
        if validated["model"]["fingerprint"] != expected:
            raise ValueError("inference manifest model fingerprint mismatch")
    return validated


def save_inference_manifest(path: PathLike, manifest: Mapping[str, Any]) -> None:
    """Validate and write an inference manifest as portable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = validate_inference_manifest(manifest)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "INFERENCE_MANIFEST_FORMAT",
    "INFERENCE_MANIFEST_SCHEMA_VERSION",
    "build_inference_manifest",
    "load_inference_manifest",
    "save_inference_manifest",
    "validate_inference_manifest",
]
