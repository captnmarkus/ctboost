"""Experimental fixed two-seed averaging; no fitting, selection, or native edits.

The payload is an ordinary versioned dictionary. Its two component records keep
the frozen source model IDs/hashes; the caller verifies those files before
construction and freezes the serialized ensemble hash before assessment.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np

from benchmarks import fixed_temperature


FORMAT_VERSION = 1
RULE = "equal_probability_mean_two_seeds_v1"
OUTPUT_KINDS = {"regression": "regression", "binary": "binary_log_odds",
                "multiclass": "log_probabilities"}
_TOP_KEYS = {"format_version", "rule", "library", "problem", "schema", "output_kind", "components",
             "lightgbm_binary_objectives"}
_COMPONENT_KEYS = {"seed", "model_id", "model_sha256", "payload"}


def _same_schema(first, second):
    if type(first) is not dict or type(second) is not dict or list(first) != list(second):
        return False
    for name in first:
        left, right = first[name], second[name]
        if type(name) is not str or type(left) is not list or type(right) is not list or len(left) != len(right):
            return False
        for a, b in zip(left, right):
            if type(a) is not type(b):
                return False
            equal = a == b
            if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
                return False
    return True


def _unit_binary_objective(objective):
    if type(objective) is not str:
        return False
    tokens = objective.split()
    if not tokens or tokens[0] != "binary":
        return False
    sigmoid = [token.split(":", 1)[1] for token in tokens if token.startswith("sigmoid:")]
    if len(sigmoid) != 1:
        return False
    try:
        return float(sigmoid[0]) == 1.
    except ValueError:
        return False


def validate(payload):
    """Reject changed rules, incompatible components, or ambiguous metadata."""
    if type(payload) is not dict or set(payload) != _TOP_KEYS:
        raise ValueError("Expected the exact versioned ensemble metadata fields")
    if type(payload["format_version"]) is not int or payload["format_version"] != FORMAT_VERSION or payload["rule"] != RULE:
        raise ValueError("Unsupported ensemble version or averaging rule")
    library, problem = payload["library"], payload["problem"]
    if library not in ("ctboost", "xgboost", "lightgbm") or problem not in OUTPUT_KINDS:
        raise ValueError("Unsupported library or problem")
    if payload["output_kind"] != OUTPUT_KINDS[problem]:
        raise ValueError("Prediction representation does not match the problem")
    objectives = payload["lightgbm_binary_objectives"]
    if library == "lightgbm" and problem == "binary":
        if type(objectives) is not list or len(objectives) != 2 or not all(map(_unit_binary_objective, objectives)):
            raise ValueError("Frozen native LightGBM objectives must verify sigmoid=1")
    elif objectives is not None:
        raise ValueError("Native LightGBM link metadata is only valid for binary LightGBM")
    components = payload["components"]
    if type(components) not in (list, tuple) or len(components) != 2:
        raise ValueError("Exactly two components are required")
    seeds, identities = set(), set()
    for record in components:
        if type(record) is not dict or set(record) != _COMPONENT_KEYS:
            raise ValueError("Expected seed, model ID, source hash, and native payload")
        seed, identity, digest = record["seed"], record["model_id"], record["model_sha256"]
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError("Two distinct nonnegative integer seeds are required")
        if type(identity) is not str or not identity or identity in identities:
            raise ValueError("Two distinct component IDs are required")
        if type(digest) is not str or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Expected a lowercase SHA256 component hash")
        seeds.add(seed)
        identities.add(identity)
        component = record["payload"]
        if (type(component) is not dict or "model" not in component
                or component.get("library") != library or component.get("problem") != problem
                or not _same_schema(component.get("schema"), payload["schema"])):
            raise ValueError("Components must share library, problem, and ordered categorical schema")
        # Native objective strings were verified once during construction and
        # persist in the frozen payload. Params alone can be empty after a
        # model-string reload; do not dump every model on every prediction.
        if library == "lightgbm" and problem == "binary":
            sigmoid = float(getattr(component["model"], "params", {}).get("sigmoid", 1.0))
            if not np.isfinite(sigmoid) or sigmoid != 1.0:
                raise ValueError("Binary LightGBM components require sigmoid=1")
    return components


def make_payload(components):
    """Build fixed metadata; never choose seeds, components, or weights."""
    if type(components) not in (list, tuple) or len(components) != 2:
        raise ValueError("Exactly two components are required")
    first = components[0]["payload"]
    problem = first["problem"]
    result = {"format_version": FORMAT_VERSION, "rule": RULE,
              "library": first["library"], "problem": problem,
              "schema": copy.deepcopy(first["schema"]),
              "output_kind": OUTPUT_KINDS.get(problem),
              "lightgbm_binary_objectives": None,
              "components": [dict(record) for record in components]}
    if first["library"] == "lightgbm" and problem == "binary":
        result["lightgbm_binary_objectives"] = [record["payload"]["model"].dump_model().get("objective")
                                                  for record in components]
    validate(result)
    return result


def average_outputs(first, second, problem):
    """Combine already calibrated raw margins using exactly equal weights.

    Binary output is log-odds of the arithmetic probability mean, preserving
    extreme-tail ranking without converting probabilities to rounded 0 or 1.
    Multiclass output is normalized log probability, not averaged logits.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Components must return equal-shaped finite predictions")
    if problem in ("regression", "binary"):
        if first.ndim != 1:
            raise ValueError("Scalar problems require one-dimensional predictions")
    elif problem == "multiclass":
        if first.ndim != 2 or first.shape[1] < 2:
            raise ValueError("Multiclass predictions require at least two aligned classes")
    else:
        raise ValueError("Unsupported problem")
    if problem == "regression":
        return .5 * first + .5 * second
    log_two = np.log(2.)
    if problem == "binary":
        positive = np.logaddexp(-np.logaddexp(0., -first), -np.logaddexp(0., -second)) - log_two
        negative = np.logaddexp(-np.logaddexp(0., first), -np.logaddexp(0., second)) - log_two
        return positive - negative
    def log_softmax(margins):
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = margins - margins.max(axis=1, keepdims=True)
        if not np.isfinite(shifted).all():
            raise ValueError("Multiclass margin differences exceed finite float64 range")
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    result = np.logaddexp(log_softmax(first), log_softmax(second)) - log_two
    if not np.isfinite(result).all():
        raise ValueError("Non-finite ensemble log probabilities")
    return result


def _component_prediction(component, frame, xgb_inplace):
    library, model = component["library"], component["model"]
    if library == "ctboost":
        margins = model.predict(frame)
    elif library == "lightgbm":
        margins = model.predict(frame, raw_score=True, num_threads=1)
    elif xgb_inplace:
        margins = model.inplace_predict(frame, predict_type="margin")
    else:
        import xgboost as xgb
        margins = model.predict(xgb.DMatrix(frame, enable_categorical=True), output_margin=True)
    # Apply the existing frozen temperature exactly once per component, before
    # probability averaging. A temperature is never applied to the mixture.
    return fixed_temperature.apply_temperature(component, margins)


def _predict_checked(payload, components, frame, xgb_inplace):
    values = [_component_prediction(record["payload"], frame, xgb_inplace) for record in components]
    return average_outputs(values[0], values[1], payload["problem"])


def predict_normalized(payload, frame, *, xgb_inplace=False):
    """Predict a shared-schema frame; the caller must first verify XGB inplace.

    The opt-in is intentionally not persisted. Fair latency calls
    verify_xgb_inplace on every frozen batch before timing this path.
    """
    return _predict_checked(payload, validate(payload), frame, xgb_inplace)


def predict(payload, raw_frame, *, xgb_inplace=False):
    """Include one shared vocabulary normalization and both full model calls."""
    from benchmarks.fair_accuracy_scout import normalize
    components = validate(payload)
    normalized, _ = normalize(raw_frame, payload["schema"])
    return _predict_checked(payload, components, normalized, xgb_inplace)


def verify_xgb_inplace(payload, normalized_frame):
    """Require bit-exact component and ensemble outputs; never mutate payload."""
    components = validate(payload)
    if payload["library"] != "xgboost":
        raise ValueError("The inplace parity guard is XGBoost-specific")
    reference, candidate = [], []
    for record in components:
        component = record["payload"]
        first = np.asarray(_component_prediction(component, normalized_frame, False))
        second = np.asarray(_component_prediction(component, normalized_frame, True))
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError("Inplace output shape or dtype differs")
        np.testing.assert_array_equal(np.ascontiguousarray(first).view(np.uint8),
                                      np.ascontiguousarray(second).view(np.uint8))
        reference.append(first)
        candidate.append(second)
    first = average_outputs(*reference, payload["problem"])
    second = average_outputs(*candidate, payload["problem"])
    np.testing.assert_array_equal(np.ascontiguousarray(first).view(np.uint8),
                                  np.ascontiguousarray(second).view(np.uint8))
    return {"component_ids": [record["model_id"] for record in components],
            "rows": len(normalized_frame), "columns": normalized_frame.shape[1],
            "bit_exact": True, "output_sha256": hashlib.sha256(np.ascontiguousarray(first).tobytes()).hexdigest()}
