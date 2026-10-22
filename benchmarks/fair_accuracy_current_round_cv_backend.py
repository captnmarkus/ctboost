"""Inactive native adapter for the prospective current-policy round CV.

Modules and the named-array budget are explicit inputs. This does not authorize
real data, admit a parent policy, freeze resources or select a solver. A future
runner must bind these modules and their transitive source/runtime hashes after
the complete parent17b quality gate. Numerical certification remains per solve.
"""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

CORRECTION_FIELDS = ("leaf_refit", "binary_refit", "multiclass_refit")
RECEIPT_FIELDS = ("fixed_leaf_refit", "fixed_binary_leaf_refit", "fixed_multiclass_leaf_refit")


def _fresh_payload(payload):
    if any(name in payload for name in (*RECEIPT_FIELDS, *CORRECTION_FIELDS)):
        raise ValueError("Expected a fresh payload without inherited correction metadata")
    if payload["problem"] not in ("regression", "binary", "multiclass"):
        raise ValueError("Unsupported fixed-policy problem")


def _control_hash(controls):
    return hashlib.sha256(json.dumps(controls, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def make_backend(*, scalar, multiclass, max_working_bytes):
    """Bind verified17b scalar and16 multiclass callbacks without hidden defaults.

    max_working_bytes overrides only the same field in a copied correction dict;
    lambda, shrinkage, solver and temperature are inherited unchanged. Native fit
    provenance is enforced by calling union.fit_fixed here. Absence of metadata
    alone cannot prove an arbitrary externally supplied model was never edited.
    """
    if type(max_working_bytes) is not int or max_working_bytes <= 0:
        raise ValueError("An explicit positive named-array resource cap is required")

    def fit_clean(job, case, X, labels, schema, seed):
        if any(job.get(field) is not None for field in (*CORRECTION_FIELDS, "temperature")):
            raise ValueError("Disable every fixed correction and temperature before the fresh fit")
        if any(field in job for field in RECEIPT_FIELDS):
            raise ValueError("A clean fit cannot inherit a correction receipt")
        payload, details = scalar.union.fit_fixed(job, case, X, labels, schema, seed)
        _fresh_payload(payload)
        if payload.get("temperature") is not None:
            raise ValueError("The fresh native fit must not apply temperature")
        details = {**details, "native_cv_fresh_fit": True,
                   "fixed_corrections_disabled_before_fit": True,
                   "fit_controls_sha256": _control_hash({name: job.get(name) for name in
                       ("dataset", "library", "parameters", "rounds", *CORRECTION_FIELDS, "temperature")})}
        return payload, details

    def apply_fixed_correction(payload, problem, controls, X, labels, verification_frames):
        _fresh_payload(payload)
        if payload["problem"] != problem:
            raise ValueError("Correction problem does not match the fresh payload")
        original = copy.deepcopy(controls)
        effective = copy.deepcopy(controls)
        effective["max_working_bytes"] = max_working_bytes
        frozen_algorithm = {key: value for key, value in original.items() if key != "max_working_bytes"}
        temperature = payload.get("temperature")
        callback = multiclass.corrected if problem == "multiclass" else scalar.corrected
        candidate, evidence = callback(payload, X, labels, effective, verification_frames)
        if (controls != original or {key: value for key, value in effective.items() if key != "max_working_bytes"} != frozen_algorithm
                or effective["max_working_bytes"] != max_working_bytes):
            raise ValueError("The correction callback changed frozen algorithm controls")
        if candidate.get("temperature") != temperature:
            raise ValueError("The correction callback changed the fixed temperature")
        expected_receipt = "fixed_multiclass_leaf_refit" if problem == "multiclass" else "fixed_leaf_refit"
        if expected_receipt not in candidate:
            raise ValueError("Native correction did not retain its folding evidence")
        return candidate, {**evidence, "native_cv_adapter": {
            "algorithm_controls_sha256": _control_hash(frozen_algorithm),
            "inherited_named_array_limit": original.get("max_working_bytes"),
            "effective_named_array_limit": max_working_bytes,
            "resource_override_only": True, "temperature_unchanged": True,
            "fresh_payload_metadata_guard": True}}

    return SimpleNamespace(normalize=scalar.base.normalize, fit_clean=fit_clean,
        apply_fixed_correction=apply_fixed_correction, predict=scalar.predict, score=scalar.base.error)
