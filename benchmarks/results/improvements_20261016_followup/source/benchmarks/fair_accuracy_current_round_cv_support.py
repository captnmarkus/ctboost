"""Inactive support prototype for full-current-data, fixed-policy round CV.

There is no CLI, protocol freeze, data acquisition, native adapter or default
solver here. A future runner must first verify a completed, quality-failed 17b
policy and its entire source/runtime/data chain. This module must not activate
from the numerically failed wave17 or substitute the older wave16 controls.

Reuse wave13's make_folds/index_hash, common normalize/error, union.fit_fixed,
native correction oracles and owned-process watchdog without editing them.
Dispatch one fit per process (one worker), preserving every receipt. Freeze all
540 fold results/36 choices before 36 mandatory union fits. Resource limits,
split seed, conditional replication and certified regression adapter remain
prospective decisions, not defaults supplied by this prototype.
"""
from __future__ import annotations

import copy
import hashlib
import math
import pickle

LIBRARIES = ("ctboost", "xgboost", "lightgbm")
CORRECTION_FIELDS = ("leaf_refit", "binary_refit", "multiclass_refit")
CORRECTION_FIELD = dict(regression="leaf_refit", binary="binary_refit", multiclass="multiclass_refit")
EXPANDED_DATASETS = {"GiveMeSomeCredit", "diamonds", "SDSS17", "houses"}


def index_hash(values):
    import numpy as np
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _rows(frame, labels, ids):
    import numpy as np
    labels, ids = np.asarray(labels), np.asarray(ids)
    if (labels.ndim != 1 or ids.ndim != 1 or ids.dtype.kind not in "iu"
            or len(frame) != len(labels) or len(labels) != len(ids) or not len(ids)
            or (ids < 0).any() or (ids > np.iinfo(np.int64).max).any()
            or len(np.unique(ids)) != len(ids) or not frame.columns.is_unique):
        raise ValueError("Invalid fitting rows, labels, columns or row identities")
    return labels, ids.astype(np.int64, copy=False)


def reconstruct_raw_union(case, job, *, load_role, load_expanded, digest):
    """Load only already-authorized TRAIN/STOP and optional expanded pickle.

    Callbacks perform actual I/O only in the eventual activated runner. digest
    verifies bytes before either loader is called. No development/confirmation/
    official-test role or original ARFF is opened. Exclusion is inherited from
    the frozen authorized-source receipt and checked ordered-ID hash; this is
    not a fresh reconstruction of official split metadata.
    """
    import numpy as np
    import pandas as pd
    if job["dataset"] != case["dataset_name"]:
        raise ValueError("Policy and case disagree")
    roles = []
    for name in ("train", "stop"):
        record = case["roles"][name]
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("Historical fitting role bytes changed")
        role = load_role(case, name)
        labels, ids = _rows(role["X"], role["y"], role["outer_train_positions"])
        if len(ids) != record["rows"] or index_hash(ids) != record["index_sha256"]:
            raise ValueError("Historical fitting role identities changed")
        roles.append((role["X"], labels, ids))
    train, stop = roles
    prefix_ids = np.concatenate([train[2], stop[2]])
    if len(np.unique(prefix_ids)) != len(prefix_ids):
        raise ValueError("Historical TRAIN and STOP overlap")
    if list(train[0].columns) != list(stop[0].columns):
        raise ValueError("Historical feature order differs")
    source = job.get("expanded_source")
    if source is None and job["dataset"] in EXPANDED_DATASETS:
        raise ValueError("A current expanded task cannot fall back to the old capped union")
    if source is None:
        raw = pd.concat([train[0], stop[0]], ignore_index=True)
        labels, ids = np.concatenate([train[1], stop[1]]), prefix_ids
    else:
        if (job["dataset"] not in EXPANDED_DATASETS or source["dataset"] != job["dataset"]
                or digest(source["path"]) != source["sha256"]):
            raise ValueError("Unauthorized or changed expanded fitting source")
        expanded = load_expanded(source)
        if expanded["receipt"] != source["receipt"]:
            raise ValueError("Expanded authorization receipt changed")
        raw = expanded["X"]
        labels, ids = _rows(raw, expanded["y"], expanded["outer_train_positions"])
        receipt = source["receipt"]
        hash_field = "authorized_rows_sha256" if job["dataset"] == "houses" else "expanded_positions_sha256"
        if (len(ids) != receipt["rows"] or index_hash(ids) != receipt[hash_field]
                or not np.array_equal(ids[:len(prefix_ids)], prefix_ids)
                or len(ids) - len(prefix_ids) != receipt["added_rows"]):
            raise ValueError("Expanded authorized row order or count changed")
        extra = ids[len(prefix_ids):]
        if len(extra) > 1 and not np.all(extra[1:] > extra[:-1]):
            raise ValueError("Outside-cap rows are not ascending")
        if ("additional_positions_sha256" in receipt
                and index_hash(extra) != receipt["additional_positions_sha256"]):
            raise ValueError("Outside-cap identities changed")
        offset = 0
        for frame, expected_labels, expected_ids in roles:
            end = offset + len(expected_ids)
            pd.testing.assert_frame_equal(raw.iloc[offset:end].reset_index(drop=True),
                frame.reset_index(drop=True), check_dtype=False, check_categorical=False, check_exact=True)
            np.testing.assert_array_equal(labels[offset:end], expected_labels)
            offset = end
    labels, ids = _rows(raw, labels, ids)
    if case["problem_type"] != "regression":
        np.testing.assert_array_equal(np.unique(labels), np.arange(case["num_classes"]))
    return raw.reset_index(drop=True), labels, ids, {
        "rows": len(ids), "original_train_rows": len(train[2]), "original_stop_rows": len(stop[2]),
        "outside_cap_rows": len(ids) - len(prefix_ids), "ordered_ids_sha256": index_hash(ids),
        "source_sha256": source["sha256"] if source else None,
        "source_roles": copy.deepcopy({name: case["roles"][name] for name in ("train", "stop")}),
        "row_order": "original TRAIN, original STOP, then authorized outside-cap IDs ascending",
        "normalization_performed": False,
        "exclusion_basis": "hash-verified historical roles and authorized expanded-source ordered IDs"}


def round_candidates(rounds):
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Requested rounds must be a positive integer")
    # Retain duplicate profiles (e.g. n=1); never reduce the equal search budget.
    return ((rounds + 1) // 2, rounds, 2 * rounds)


def enumerate_fit_jobs(jobs, *, seed=47):
    """Return 540 identities, each intended for one separate owned process."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Expected an explicit nonnegative integer seed")
    pairs = [(job["dataset"], job["library"]) for job in jobs]
    datasets = {dataset for dataset, _ in pairs}
    if (len(jobs) != 36 or len(datasets) != 12 or len(set(pairs)) != 36
            or set(pairs) != {(dataset, library) for dataset in datasets for library in LIBRARIES}):
        raise ValueError("Expected all36 unique task/library policies")
    return [{"dataset": job["dataset"], "library": job["library"], "profile_index": profile,
             "fold": fold, "requested_rounds": rounds, "seed": seed}
            for job in jobs for profile, rounds in enumerate(round_candidates(job["rounds"])) for fold in range(5)]


def choose_rounds(jobs, records, *, seed=47):
    """All15 identities per policy are mandatory; mean fold error, not pooled AUC.

    The future worker persists prediction archives and adds their file hashes.
    This reducer checks complete receipt identity and successful pickle checks;
    the orchestrator separately verifies those bytes and resource receipts.
    """
    expected = enumerate_fit_jobs(jobs, seed=seed)
    key = lambda row: (row["dataset"], row["library"], row["profile_index"], row["fold"])
    lookup = {key(record): record for record in records}
    if len(records) != 540 or len(lookup) != 540 or set(lookup) != {key(row) for row in expected}:
        raise ValueError("All540 unique fold receipts are required")
    for identity in expected:
        record = lookup[key(identity)]
        if (any(record.get(name) != value for name, value in identity.items())
                or record.get("status") != "ok" or record.get("reload_bit_exact") is not True
                or not math.isfinite(record["validation_error"])):
            raise ValueError("A fold failed or its frozen identity changed")
        retained = record["retained_rounds"]
        if (type(retained) is not int or not 0 < retained <= identity["requested_rounds"]
                or (identity["library"] != "lightgbm" and retained != identity["requested_rounds"])):
            raise ValueError("Unexpected native retained round count")
        for name in ("model_sha256", "prediction_sha256"):
            value = record[name]
            if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing model/prediction content hash")
    choices = []
    for job in jobs:
        scores = [{"profile_index": profile, "requested_rounds": rounds,
                   "mean_fold_error": math.fsum(lookup[job["dataset"], job["library"], profile, fold]["validation_error"]
                                                 for fold in range(5)) / 5}
                  for profile, rounds in enumerate(round_candidates(job["rounds"]))]
        best = min(scores, key=lambda row: (row["mean_fold_error"], row["requested_rounds"], row["profile_index"]))
        choices.append({"dataset": job["dataset"], "library": job["library"], "scores": scores, "best": best,
                        "job": {**copy.deepcopy(job), "rounds": best["requested_rounds"]}})
    return choices


def fit_fold(job, case, raw_train, labels, raw_validation, seed, backend):
    """One fresh fit, with no validation-label argument or default solver.

    backend.normalize and fit_clean should bind the frozen common normalizer and
    union.fit_fixed. apply_fixed_correction(payload, problem, controls, X, y,
    verification_frames) MUST preflight memory, replay CT's fold-training ordered
    representation, use the prospectively certified solver, and perform native
    routing/topology/zero/folding/raw+probability oracle checks. It receives only
    training labels. Its callback/solver source hashes belong in the future plan.

    backend.predict must implement the frozen raw-frame prediction rule with T
    exactly once. This prototype checks pickle prediction bytes independently.
    The returned payload is transient: archive its hash, then release it and all
    designs at process exit. Union fits can use the same function with an input
    frame for reload checks; only the future runner may make that mandatory stage.
    """
    import numpy as np
    if job["dataset"] != case["dataset_name"] or job["library"] not in LIBRARIES:
        raise ValueError("Policy and case disagree")
    if type(seed) is not int or seed < 0:
        raise ValueError("Expected an explicit nonnegative integer seed")
    round_candidates(job["rounds"])
    problem = case["problem_type"]
    field = CORRECTION_FIELD[problem]
    if any(job.get(other) is not None for other in CORRECTION_FIELDS if other != field):
        raise ValueError("Correction controls do not match the problem")
    temperature = job.get("temperature")
    if temperature is not None and (problem != "multiclass" or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Only fixed positive multiclass temperature is supported")
    labels = np.asarray(labels)
    if labels.shape != (len(raw_train),) or not len(labels) or not np.isfinite(labels).all():
        raise ValueError("Invalid fold-training labels")
    if problem != "regression":
        np.testing.assert_array_equal(np.unique(labels), np.arange(case["num_classes"]))
    if not len(raw_validation):
        raise ValueError("A held-out fold must contain rows")
    if list(raw_train.columns) != list(raw_validation.columns):
        raise ValueError("Fold feature order changed")
    for name in ("early_stopping_rounds", "early_stopping", "callbacks", "eval_set", "evals", "valid_sets", "use_best_model"):
        if job["parameters"].get(name) not in (None, False, [], ()):
            raise ValueError("No validation-based stopping is permitted")
    X, schema = backend.normalize(raw_train)
    Xv, validation_schema = backend.normalize(raw_validation, schema)
    if validation_schema != schema:
        raise ValueError("Validation changed the fitted vocabulary")
    for name, values in schema.items():
        observed = sorted(raw_train[name].dropna().unique().tolist(), key=lambda value: (type(value).__name__, str(value)))
        if values != observed:
            raise ValueError("Fold vocabulary contains values absent from fold TRAIN")
    clean = copy.deepcopy(job)
    clean.update({name: None for name in CORRECTION_FIELDS})
    clean["temperature"] = None
    payload, details = backend.fit_clean(clean, case, X, labels, schema, seed)
    if (payload["problem"] != problem or payload["library"] != job["library"]
            or payload["schema"] != schema or payload.get("temperature") is not None):
        raise ValueError("Fresh native fit changed the schema/problem or applied temperature")
    if temperature is not None:
        payload = {**payload, "temperature": temperature}
    correction = None
    if job.get(field) is not None:
        payload, correction = backend.apply_fixed_correction(payload, problem, copy.deepcopy(job[field]), X, labels,
                                                             {"validation": Xv})
    if (payload["schema"] != schema or payload.get("temperature") != temperature
            or payload["problem"] != problem or payload["library"] != job["library"]):
        raise ValueError("Correction changed schema, task, library or fixed temperature")
    predictions = np.asarray(backend.predict(payload, raw_validation))
    expected_shape = ((len(raw_validation), case["num_classes"]) if problem == "multiclass" else (len(raw_validation),))
    if predictions.shape != expected_shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid fold predictions")
    encoded = pickle.dumps(payload, protocol=5)
    restored = np.asarray(backend.predict(pickle.loads(encoded), raw_validation))
    if (predictions.shape != restored.shape or predictions.dtype != restored.dtype
            or predictions.tobytes() != restored.tobytes()):
        raise ValueError("Pickle reload changed prediction bytes")
    retained = details["retained_rounds"]
    if (details["requested_rounds"] != job["rounds"] or type(retained) is not int
            or not 0 < retained <= job["rounds"] or (job["library"] != "lightgbm" and retained != job["rounds"])
            or details.get("early_stopping") is not False):
        raise ValueError("Unexpected fixed-round native fit")
    return payload, predictions, {**details, "fold_category_schema": copy.deepcopy(schema),
        "feature_names": list(raw_train.columns), "correction": correction, "temperature": temperature,
        "training_rows": len(X), "validation_rows": len(Xv), "seed": seed,
        "model_sha256": hashlib.sha256(encoded).hexdigest(), "model_bytes": len(encoded),
        "model_retained": False, "reload_bit_exact": True,
        "prediction_array_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "prediction_dtype": predictions.dtype.str, "prediction_shape": list(predictions.shape),
        "corrections_disabled_before_fit": True, "validation_labels_received": False}
