"""Draft-only, pure cap reconstruction and expanded fitting-row validation.

No file I/O, fitting, normalization, scoring, or imports of OpenML are performed.
Full official-TRAIN labels are used solely to reconstruct the historical cap and
class vocabulary, including labels of excluded DEV/confirmation rows.
"""
from __future__ import annotations

import hashlib

EXPAND_DATASETS = ("GiveMeSomeCredit", "diamonds", "SDSS17")
CAP_ROWS = 10000
CAP_SEED = 47


def index_hash(values, *, length_prefix=False):
    import numpy as np
    values = np.asarray(values, dtype="<i8")
    result = hashlib.sha256()
    if length_prefix:
        result.update(len(values).to_bytes(8, "little"))
    result.update(values.tobytes())
    return result.hexdigest()


def checked_ids(values, total_rows, label):
    import numpy as np
    result = np.asarray(values)
    if (result.ndim != 1 or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= total_rows)
            or len(np.unique(result)) != len(result)):
        raise ValueError(f"Invalid or duplicate {label} row IDs")
    return result.astype(np.int64, copy=False)


def historical_cap(raw_labels, problem, *, cap_rows=CAP_ROWS, cap_seed=CAP_SEED):
    """Match accuracy_grouped_scout.py's LabelEncoder then seed47 split."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    labels = np.asarray(raw_labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Expected nonempty one-dimensional source labels")
    classes = None
    if problem != "regression":
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)
        classes = encoder.classes_.tolist()
    selected = np.arange(len(labels), dtype=np.int64)
    if len(selected) > cap_rows:
        selected, _ = train_test_split(selected, train_size=cap_rows, random_state=cap_seed,
                                       stratify=labels if problem != "regression" else None)
        selected.sort()
    return labels, selected, classes


def verify_prepared_receipt(case, receipt, *, source_hash, protocol_hash):
    """Validate supplied hashes; caller hashes files before unpickling."""
    if (case["dataset_name"] not in EXPAND_DATASETS
            or receipt.get("contains_outer_test_rows") is not False
            or receipt.get("dataset") != case["dataset_name"]
            or receipt.get("problem_type") != case["problem_type"]
            or receipt.get("task_id") != case["task_id"]
            or receipt.get("data_sha256") != source_hash
            or case["source_sha256"] != source_hash
            or receipt.get("protocol_sha256") != protocol_hash
            or receipt.get("rows_outer_train") != case["original_rows"]
            or receipt.get("features") != case["num_features"]):
        raise ValueError("Source is not the historically verified official-TRAIN-only cache")
    outer_hash = receipt.get("outer_train_indices_sha256")
    if not isinstance(outer_hash, str) or len(outer_hash) != 64 or any(c not in "0123456789abcdef" for c in outer_hash):
        raise ValueError("Missing source outer-TRAIN split identity")
    return {"outer_train_indices_sha256": outer_hash,
            "outer_train_index_hash_encoding": "SHA256(length_uint64_le || int64_le positions)",
            "role_index_hash_encoding": "SHA256(int64_le positions), no length prefix"}


def expand_rows(case, original, roles, historical_roles, *, cap_rows=CAP_ROWS, cap_seed=CAP_SEED):
    """Return TRAIN + STOP + ascending outside-cap rows; never use held-out rows.

    `roles` must contain only train/stop/development payloads. Development is used
    only for already-frozen row IDs. The sorted capped remainder must match the
    historical confirmation count/hash; no confirmation payload is accepted.
    """
    import numpy as np
    import pandas as pd
    if case["dataset_name"] not in EXPAND_DATASETS or set(roles) != {"train", "stop", "development"}:
        raise ValueError("Only the three approved tasks and unsealed role IDs are accepted")
    frame = original["X"]
    if (len(frame) != len(original["y"]) or len(frame) != case["original_rows"]
            or frame.shape[1] != case["num_features"] or len(frame) <= cap_rows):
        raise ValueError("Source dimensions changed or no cap is released")
    labels, selected, classes = historical_cap(original["y"], case["problem_type"],
                                               cap_rows=cap_rows, cap_seed=cap_seed)
    if len(selected) != case["capped_rows"]:
        raise ValueError("Historical cap size changed")
    if case["problem_type"] != "regression" and len(classes) != case["num_classes"]:
        raise ValueError("Historical encoded class vocabulary changed")
    if not np.isfinite(np.asarray(labels, dtype=np.float64)).all():
        raise ValueError("Source labels must be finite")
    ids = {}
    for name in ("train", "stop", "development"):
        role = roles[name]
        ids[name] = checked_ids(role["outer_train_positions"], len(frame), name)
        metadata = historical_roles[name]
        if (len(ids[name]) != metadata["rows"] or index_hash(ids[name]) != metadata["index_sha256"]
                or not np.isin(ids[name], selected).all()):
            raise ValueError(f"Historical {name} row membership changed")
        current = case["roles"][name]
        if any(current[key] != metadata[key] for key in ("rows", "index_sha256", "sha256")):
            raise ValueError(f"Current {name} role does not match historical metadata")
        if name in ("train", "stop"):
            if len(role["X"]) != len(ids[name]) or len(role["y"]) != len(ids[name]):
                raise ValueError("Original fitting-role dimensions changed")
            pd.testing.assert_frame_equal(role["X"].reset_index(drop=True),
                                          frame.iloc[ids[name]].reset_index(drop=True),
                                          check_exact=True, check_dtype=True, check_categorical=True)
            np.testing.assert_array_equal(np.asarray(role["y"]), labels[ids[name]])
    observed = np.concatenate([ids[name] for name in ("train", "stop", "development")])
    if len(np.unique(observed)) != len(observed):
        raise ValueError("Historical TRAIN/STOP/DEV overlap")
    confirmation = np.setdiff1d(selected, observed, assume_unique=True)
    sealed = historical_roles["confirmation"]
    if len(confirmation) != sealed["rows"] or index_hash(confirmation) != sealed["index_sha256"]:
        raise ValueError("Capped remainder does not match sealed confirmation metadata")
    additional = np.setdiff1d(np.arange(len(frame), dtype=np.int64), selected, assume_unique=True)
    expanded = np.concatenate([ids["train"], ids["stop"], additional])
    excluded = np.concatenate([ids["development"], confirmation])
    if (len(np.unique(expanded)) != len(expanded) or np.intersect1d(expanded, excluded).size
            or len(expanded) + len(excluded) != len(frame)):
        raise ValueError("Expanded fitting rows overlap held-out roles or omit source rows")
    # Slice before any normalization or correction. Excluded rows are never
    # supplied to these operations, even though the verified cache contains them.
    expanded_frame = frame.iloc[expanded].copy().reset_index(drop=True)
    expanded_labels = labels[expanded]
    receipt = {"dataset": case["dataset_name"], "rows": len(expanded), "added_rows": len(additional),
        "original_train_rows": len(ids["train"]), "original_stop_rows": len(ids["stop"]),
        "development_rows_excluded": len(ids["development"]), "confirmation_rows_excluded": len(confirmation),
        "cap_seed": cap_seed, "cap_rows": cap_rows,
        "row_order": "original TRAIN, original STOP, then outside-cap source positions ascending",
        "cap_positions_sha256": index_hash(selected), "expanded_positions_sha256": index_hash(expanded),
        "additional_positions_sha256": index_hash(additional),
        "development_positions_sha256": index_hash(ids["development"]),
        "confirmation_positions_sha256": index_hash(confirmation),
        "class_vocabulary": classes, "feature_names": list(frame.columns),
        "raw_dtypes": [str(dtype) for dtype in frame.dtypes],
        "label_access": "Full official-TRAIN labels only for historical cap stratification/class vocabulary; excluded labels never enter fitting, normalization, correction, or confirmation scoring"}
    return {"X": expanded_frame, "y": expanded_labels, "outer_train_positions": expanded,
            "verification_frame": roles["stop"]["X"], "receipt": receipt}
