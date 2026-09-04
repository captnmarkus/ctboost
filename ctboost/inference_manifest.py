"""Versioned deployment metadata for CTBoost inference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from ._version import __version__

PathLike = Union[str, Path]

INFERENCE_MANIFEST_FORMAT = "ctboost-inference-manifest"
INFERENCE_MANIFEST_SCHEMA_VERSION = 2
_FINGERPRINT_PREFIX = "sha256:"
_DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
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
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("inference manifest values must be finite JSON values")
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
    # Predictor formats v2/v3 make the input representation part of model
    # identity. Keep the historical v1/v2 fingerprints byte-for-byte compatible.
    if scoring_payload.get("format_version") in (2, 3):
        identity["expects_prepared_features"] = scoring_payload.get(
            "expects_prepared_features"
        )
    if class_labels is not None:
        identity["class_labels"] = _json_ready(class_labels)
    return _fingerprint(identity)


def _build_details() -> dict[str, Any]:
    from . import _core

    details = dict(_core.build_info())
    details["package_version"] = __version__
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
        # cat_feature_indices_ describes the transformed matrix.  A raw-input
        # manifest must instead identify the source columns consumed by the
        # fitted scalar/one-hot categorical encoders.
        categorical_indices = sorted(
            {
                int(state["source_index"])
                for key in ("one_hot_states", "categorical_states")
                for state in pipeline.get(key, [])
                if "source_index" in state
            }
        )
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
            "categorical_key_encoding_version": int(
                pipeline.get("categorical_key_encoding_version", 1)
            ),
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
            "ctboost_runtime_required": (
                artifact_kind == "ctboost_model"
                or feature_pipeline_state is not None
                and not expects_prepared_features
            ),
            "estimator": estimator_name,
        },
        "model": {
            "fingerprint": _model_fingerprint(scoring_payload, class_labels),
            "fingerprint_algorithm": "sha256",
            "objective": objective_name,
            "tree_count": len(scoring_payload["trees"]),
            "iteration_count": len(scoring_payload["trees"]) // trees_per_iteration,
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
    if not isinstance(manifest, Mapping):
        raise ValueError("inference manifest must be a JSON object")
    document = _json_ready(manifest)
    if document.get("format") != INFERENCE_MANIFEST_FORMAT:
        raise ValueError("not a CTBoost inference manifest")
    if type(document.get("schema_version")) is not int or document.get("schema_version") not in (
        1, INFERENCE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported CTBoost inference manifest schema version: "
            f"{document.get('schema_version')!r}"
        )
    for section in ("producer", "artifact", "model", "input", "output"):
        if not isinstance(document.get(section), dict):
            raise ValueError(f"inference manifest is missing the {section!r} section")
    vector_leaves = document["schema_version"] == 2
    if not vector_leaves and document["model"].get("multi_strategy") == "multi_output_tree":
        raise ValueError("vector inference manifests require schema version 2")
    if vector_leaves:
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
    if type(num_features) is not int or num_features < 0:
        raise ValueError("inference manifest input.num_features must be a non-negative integer")
    if not isinstance(features, list) or len(features) != num_features:
        raise ValueError("inference manifest features do not match input.num_features")
    categorical_indices = document["input"].get("categorical_feature_indices")
    if not isinstance(categorical_indices, list) or any(
        type(value) is not int for value in categorical_indices
    ):
        raise ValueError(
            "inference manifest categorical_feature_indices must be an integer array"
        )
    categorical_set = set(categorical_indices)
    if len(categorical_set) != len(categorical_indices) or any(
        value < 0 or value >= num_features for value in categorical_indices
    ):
        raise ValueError("inference manifest categorical feature indices are invalid")
    for index, feature in enumerate(features):
        if (
            not isinstance(feature, dict)
            or type(feature.get("index")) is not int
            or feature.get("index") != index
            or type(feature.get("categorical")) is not bool
            or feature.get("categorical") is not (index in categorical_set)
        ):
            raise ValueError("inference manifest feature metadata is inconsistent")
    model_feature_count = document["input"].get("model_feature_count")
    if type(model_feature_count) is not int or model_feature_count < 0:
        raise ValueError(
            "inference manifest input.model_feature_count must be a non-negative "
            "integer"
        )
    tree_count = document["model"].get("tree_count")
    if type(tree_count) is not int or tree_count <= 0:
        raise ValueError("inference manifest model.tree_count must be a positive integer")
    prediction_dimension = document["model"].get("prediction_dimension")
    if type(prediction_dimension) is not int or prediction_dimension <= 0:
        raise ValueError(
            "inference manifest model.prediction_dimension must be a positive integer"
        )
    trees_per_iteration = 1 if vector_leaves else prediction_dimension
    if tree_count % trees_per_iteration != 0:
        raise ValueError(
            "inference manifest model.tree_count must be divisible by "
            "model.prediction_dimension"
        )
    iteration_count = document["model"].get("iteration_count")
    expected_iteration_count = tree_count // trees_per_iteration
    if (
        type(iteration_count) is not int
        or iteration_count != expected_iteration_count
    ):
        raise ValueError(
            "inference manifest model.iteration_count does not match the tree layout"
        )
    base_score = document["model"].get("base_score")
    if not isinstance(base_score, list) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in base_score
    ):
        raise ValueError("inference manifest model.base_score must be finite")
    if len(base_score) != prediction_dimension:
        raise ValueError(
            "inference manifest model.base_score must match prediction_dimension"
        )
    output_dimension = document["output"].get("prediction_dimension")
    if type(output_dimension) is not int or output_dimension <= 0:
        raise ValueError(
            "inference manifest output.prediction_dimension must be a positive integer"
        )
    if output_dimension != prediction_dimension:
        raise ValueError(
            "inference manifest output.prediction_dimension must match the model"
        )
    model_objective = document["model"].get("objective")
    output_objective = document["output"].get("objective")
    if (
        not isinstance(model_objective, str)
        or not model_objective
        or output_objective != model_objective
    ):
        raise ValueError("inference manifest model/output objective mismatch")
    runtime_required = document["artifact"].get("ctboost_runtime_required")
    if type(runtime_required) is not bool:
        raise ValueError(
            "inference manifest artifact.ctboost_runtime_required must be boolean"
        )
    representation = document["input"].get("representation")
    allowed_representations = {
        "raw_features",
        "prepared_numeric_features",
        "numeric_or_preencoded_categorical_features",
    }
    if representation not in allowed_representations:
        raise ValueError("inference manifest input.representation is unsupported")
    preprocessing = document["input"].get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("inference manifest input.preprocessing must be an object")
    preprocessing_kind = preprocessing.get("kind")
    external_preprocessing = preprocessing.get("external_preprocessing_required")
    if type(external_preprocessing) is not bool:
        raise ValueError(
            "inference manifest preprocessing.external_preprocessing_required must "
            "be boolean"
        )
    if representation == "raw_features":
        if (
            preprocessing_kind != "ctboost_feature_pipeline"
            or external_preprocessing is not False
            or runtime_required is not True
        ):
            raise ValueError(
                "raw_features requires embedded CTBoost preprocessing and runtime"
            )
    elif representation == "prepared_numeric_features":
        if (
            preprocessing_kind != "ctboost_feature_pipeline"
            or external_preprocessing is not True
            or num_features != model_feature_count
        ):
            raise ValueError(
                "prepared_numeric_features requires external CTBoost preprocessing"
            )
    elif (
        preprocessing_kind != "none"
        or external_preprocessing is not False
        or num_features != model_feature_count
    ):
        raise ValueError(
            "numeric_or_preencoded_categorical_features has inconsistent preprocessing"
        )
    if preprocessing_kind == "ctboost_feature_pipeline":
        raw_feature_count = preprocessing.get("raw_feature_count")
        transformed_feature_count = preprocessing.get("transformed_feature_count")
        if type(raw_feature_count) is not int or raw_feature_count < 0:
            raise ValueError(
                "inference manifest preprocessing.raw_feature_count must be a "
                "non-negative integer"
            )
        if (
            type(transformed_feature_count) is not int
            or transformed_feature_count < 0
        ):
            raise ValueError(
                "inference manifest preprocessing.transformed_feature_count must be "
                "a non-negative integer"
            )
        if transformed_feature_count != model_feature_count:
            raise ValueError(
                "inference manifest transformed feature count does not match the model"
            )
        if representation == "raw_features" and raw_feature_count != num_features:
            raise ValueError(
                "inference manifest raw feature count does not match input.num_features"
            )
    return deepcopy(document)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"inference manifest JSON contains non-finite value {value!r}")


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"inference manifest JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def load_inference_manifest(
    path: PathLike,
    *,
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Load a standalone manifest or the manifest embedded in a JSON predictor."""
    if type(max_artifact_bytes) is not int or max_artifact_bytes <= 0:
        raise ValueError("max_artifact_bytes must be a positive integer")
    with Path(path).open("rb") as stream:
        encoded = stream.read(max_artifact_bytes + 1)
    if len(encoded) > max_artifact_bytes:
        raise ValueError("inference manifest exceeds the configured size limit")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("inference manifest must be valid UTF-8 JSON") from error
    document = json.loads(
        text,
        parse_constant=_reject_nonfinite_json,
        object_pairs_hook=_reject_duplicate_object,
    )
    if not isinstance(document, Mapping):
        raise ValueError("inference manifest file must contain a JSON object")
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
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
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
