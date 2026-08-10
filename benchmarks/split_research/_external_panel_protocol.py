"""Frozen constants, scheduling, and identity rules for the external panel."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

RESULT_SCHEMA_VERSION = 1
PROTOCOL_NAME = "ctboost-grouped-split-external-panel-v1"
REPEAT = 0
FOLDS = (0, 1, 2)
INNER_VALIDATION_FRACTION = 0.20
BASE_SEED = 20260815
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
EARLY_STOPPING_ROUNDS = 50
HISTOGRAM_THREADS = 8
TIE_THRESHOLD = 0.001
MINIMUM_MEDIAN_IMPROVEMENT = 0.0025
MAX_DATASETS_WORSE_THAN_ONE_PERCENT = 3
MAX_FIT_TIME_RATIO = 1.15


DATASETS: Tuple[Dict[str, Any], ...] = (
    {"task_id": 15, "name": "breast-w", "problem": "binary", "stress_only": False},
    {
        "task_id": 29,
        "name": "credit-approval",
        "problem": "binary",
        "stress_only": False,
    },
    {"task_id": 9952, "name": "phoneme", "problem": "binary", "stress_only": False},
    {"task_id": 53, "name": "vehicle", "problem": "multiclass", "stress_only": False},
    {
        "task_id": 2074,
        "name": "satimage",
        "problem": "multiclass",
        "stress_only": False,
    },
    {
        "task_id": 2079,
        "name": "eucalyptus",
        "problem": "multiclass",
        "stress_only": False,
    },
    {
        "task_id": 361234,
        "name": "abalone",
        "problem": "regression",
        "stress_only": False,
    },
    {
        "task_id": 361236,
        "name": "auction_verification",
        "problem": "regression",
        "stress_only": False,
    },
    {
        "task_id": 361243,
        "name": "geographical_origin_of_music",
        "problem": "regression",
        "stress_only": False,
    },
    {
        "task_id": 361249,
        "name": "white_wine",
        "problem": "regression",
        "stress_only": False,
    },
    {
        "task_id": 361258,
        "name": "kin8nm",
        "problem": "regression",
        "stress_only": False,
    },
    {
        "task_id": 361617,
        "name": "energy_efficiency",
        "problem": "regression",
        "stress_only": False,
    },
    {"task_id": 7592, "name": "adult", "problem": "binary", "stress_only": True},
    {
        "task_id": 361255,
        "name": "california_housing",
        "problem": "regression",
        "stress_only": True,
    },
)


PROFILES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "depthwise-default",
        "iterations": 600,
        "learning_rate": 0.05,
        "max_depth": 6,
        "grow_policy": "DepthWise",
        "lambda_l2": 1.0,
    },
    {
        "name": "depthwise-regularized",
        "iterations": 800,
        "learning_rate": 0.03,
        "max_depth": 4,
        "grow_policy": "DepthWise",
        "lambda_l2": 3.0,
    },
    {
        "name": "leafwise",
        "iterations": 600,
        "learning_rate": 0.05,
        "max_depth": 8,
        "grow_policy": "LeafWise",
        "max_leaves": 31,
        "lambda_l2": 1.0,
    },
)


TREATMENTS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "control",
        "params": {
            "feature_test": "quadratic",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
    },
    {
        "name": "candidate",
        "params": {
            "feature_test": "grouped",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
    },
)


COMMON_PARAMS: Dict[str, Any] = {
    "max_bins": 256,
    "alpha": 0.05,
    "boost_from_average": True,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "bootstrap_type": "No",
    "random_strength": 0.0,
    "task_type": "CPU",
    "verbose": False,
}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("value is not JSON serializable: {!r}".format(type(value)))


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=json_default,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def array_digest(values: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def identity_digest(source: Any, data: Any, config: Any) -> str:
    return sha256_json({"source": source, "data": data, "config": config})


_WINDOWS_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])[^\s\"']*"
)
_POSIX_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_:/\\])/(?:[^/\s\"']+/)*[^/\s\"']+")
_FILE_URI = re.compile(r"file://[^\s\"']+")


def redact_absolute_paths(value: str) -> str:
    redacted = _FILE_URI.sub("<path>", str(value))
    redacted = _WINDOWS_ABSOLUTE.sub("<path>", redacted)
    redacted = _POSIX_ABSOLUTE.sub("<path>", redacted)
    return redacted


def assert_no_absolute_paths(value: Any, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_absolute_paths(item, "{}.{}".format(location, key))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_absolute_paths(item, "{}[{}]".format(location, index))
        return
    if not isinstance(value, str):
        return
    if (
        "file://" in value
        or _WINDOWS_ABSOLUTE.search(value)
        or _POSIX_ABSOLUTE.search(value)
    ):
        raise ValueError(
            "absolute path leaked into public output at {}".format(location)
        )


def seal_record(value: Mapping[str, Any], digest_key: str) -> Dict[str, Any]:
    sealed = dict(value)
    sealed.pop(digest_key, None)
    sealed[digest_key] = sha256_json(sealed)
    return sealed


def validate_record_seal(value: Mapping[str, Any], digest_key: str) -> None:
    claimed = value.get(digest_key)
    if not isinstance(claimed, str):
        raise ValueError("record is missing {}".format(digest_key))
    body = dict(value)
    body.pop(digest_key, None)
    if sha256_json(body) != claimed:
        raise ValueError("record {} mismatch".format(digest_key))


def validate_self_hashed_identity(identity: Mapping[str, Any], digest_key: str) -> None:
    validate_record_seal(identity, digest_key)


def validate_full_job_identity(job: Mapping[str, Any]) -> None:
    source = job.get("source_identity")
    data = job.get("data_identity")
    config = job.get("config")
    if not isinstance(source, Mapping):
        raise ValueError("job source identity is missing")
    if not isinstance(data, Mapping):
        raise ValueError("job data identity is missing")
    if not isinstance(config, Mapping):
        raise ValueError("job config is missing")
    validate_self_hashed_identity(source, "source_identity_sha256")
    validate_self_hashed_identity(data, "data_identity_sha256")
    expected_key = identity_digest(source, data, config)
    if job.get("job_key") != expected_key:
        raise ValueError("full source/data/config job identity mismatch")


def _protocol_core() -> Dict[str, Any]:
    return {
        "name": PROTOCOL_NAME,
        "repeat": REPEAT,
        "folds": list(FOLDS),
        "datasets": [dict(dataset) for dataset in DATASETS],
        "profiles": [dict(profile) for profile in PROFILES],
        "treatments": [dict(treatment) for treatment in TREATMENTS],
        "implicit_control_check": {
            "fold": 0,
            "profiles": "all",
            "datasets": "all",
            "omitted_params": [
                "feature_test",
                "feature_test_bins",
                "feature_test_adjustment",
            ],
            "exact_fields": [
                "raw_prediction_sha256",
                "canonical_tree_sha256",
                "best_iteration",
            ],
        },
        "common_params": dict(COMMON_PARAMS),
        "inner_validation": {
            "fraction": INNER_VALIDATION_FRACTION,
            "classification": "stratified shuffled split",
            "regression": "non-stratified shuffled split",
            "seed_formula": "20260815 + task_id + 100 * fold",
            "outer_test_used_for_early_stopping": False,
        },
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "histogram_threads": HISTOGRAM_THREADS,
        "fit_process": "one isolated subprocess per fit",
        "fit_timer": "ctboost.train call only",
        "treatment_order": "alternating by frozen dataset/fold/profile ordinal",
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "unit": "task",
            "statistic": "median task-relative primary-loss improvement",
        },
        "thresholds": {
            "tie_absolute_relative_change_below": TIE_THRESHOLD,
            "minimum_wins": 7,
            "minimum_median_improvement": MINIMUM_MEDIAN_IMPROVEMENT,
            "maximum_datasets_worse_than_one_percent": (
                MAX_DATASETS_WORSE_THAN_ONE_PERCENT
            ),
            "maximum_median_fit_time_ratio": MAX_FIT_TIME_RATIO,
        },
    }


def protocol_manifest() -> Dict[str, Any]:
    core = _protocol_core()
    decision_dataset_count = sum(not item["stress_only"] for item in DATASETS)
    stress_dataset_count = len(DATASETS) - decision_dataset_count
    paired_fit_count = len(DATASETS) * len(FOLDS) * len(PROFILES) * len(TREATMENTS)
    control_check_fit_count = len(DATASETS) * len(PROFILES)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol": core,
        "protocol_sha256": sha256_json(core),
        "expected_counts": {
            "decision_datasets": decision_dataset_count,
            "stress_datasets": stress_dataset_count,
            "paired_treatment_fits": paired_fit_count,
            "implicit_control_check_fits": control_check_fit_count,
            "total_subprocess_fits": paired_fit_count + control_check_fit_count,
        },
    }


def fit_seed(task_id: int, fold: int) -> int:
    if fold not in FOLDS:
        raise ValueError("fold must be one of {}".format(FOLDS))
    return BASE_SEED + int(task_id) + 100 * int(fold)


def treatment_order(
    dataset_index: int, fold: int, profile_index: int
) -> Tuple[str, str]:
    pair_ordinal = (
        int(dataset_index) * len(FOLDS) * len(PROFILES)
        + int(fold) * len(PROFILES)
        + int(profile_index)
    )
    return (
        ("control", "candidate") if pair_ordinal % 2 == 0 else ("candidate", "control")
    )


def planned_fit_descriptors() -> List[Dict[str, Any]]:
    schedule: List[Dict[str, Any]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        for fold in FOLDS:
            for profile_index, profile in enumerate(PROFILES):
                order = treatment_order(dataset_index, fold, profile_index)
                for order_index, treatment_name in enumerate(order):
                    schedule.append(
                        {
                            "task_id": dataset["task_id"],
                            "fold": fold,
                            "profile": profile["name"],
                            "treatment": treatment_name,
                            "role": "treatment",
                            "order_in_pair": order_index,
                        }
                    )
                if fold == 0:
                    schedule.append(
                        {
                            "task_id": dataset["task_id"],
                            "fold": fold,
                            "profile": profile["name"],
                            "treatment": "implicit-control",
                            "role": "implicit_control_check",
                            "order_in_pair": 2,
                        }
                    )
    return schedule


def make_inner_validation_split(
    outer_train_indices: Sequence[int], target: Any, problem: str, seed: int
) -> Tuple[Any, Any]:
    import numpy as np
    from sklearn.model_selection import train_test_split

    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    if outer_train.ndim != 1 or outer_train.size < 2:
        raise ValueError("outer training indices must be a non-trivial 1D array")
    if np.unique(outer_train).size != outer_train.size:
        raise ValueError("outer training indices must be unique")
    outer_target = np.asarray(target)[outer_train]
    stratify = outer_target if problem in {"binary", "multiclass"} else None
    inner_train, inner_valid = train_test_split(
        outer_train,
        test_size=INNER_VALIDATION_FRACTION,
        random_state=int(seed),
        shuffle=True,
        stratify=stratify,
    )
    return np.asarray(inner_train, dtype=np.int64), np.asarray(
        inner_valid, dtype=np.int64
    )


def validate_inner_split(
    outer_train: Any, outer_test: Any, inner_train: Any, inner_valid: Any
) -> None:
    import numpy as np

    outer_train_values = np.asarray(outer_train, dtype=np.int64)
    outer_test_values = np.asarray(outer_test, dtype=np.int64)
    inner_train_values = np.asarray(inner_train, dtype=np.int64)
    inner_valid_values = np.asarray(inner_valid, dtype=np.int64)
    if np.intersect1d(inner_train_values, inner_valid_values).size:
        raise ValueError("inner train and validation sets overlap")
    if np.intersect1d(inner_train_values, outer_test_values).size:
        raise ValueError("inner training set overlaps the outer test fold")
    if np.intersect1d(inner_valid_values, outer_test_values).size:
        raise ValueError("inner validation set overlaps the outer test fold")
    if not np.array_equal(
        np.sort(np.concatenate([inner_train_values, inner_valid_values])),
        np.sort(outer_train_values),
    ):
        raise ValueError(
            "inner train and validation must partition outer train exactly"
        )


def objective_and_metric(problem: str, class_count: Optional[int]) -> Tuple[str, str]:
    if problem == "binary":
        if class_count != 2:
            raise ValueError("binary task must contain exactly two target classes")
        return "Logloss", "AUC"
    if problem == "multiclass":
        if class_count is None or class_count < 3:
            raise ValueError("multiclass task must contain at least three classes")
        return "MultiClass", "Logloss"
    if problem == "regression":
        return "RMSE", "RMSE"
    raise ValueError("unsupported problem type: {}".format(problem))


_FROZEN_JOB_CONFIG_KEYS = frozenset(
    {
        "protocol_sha256",
        "task_id",
        "dataset_name",
        "problem",
        "stress_only",
        "repeat",
        "fold",
        "profile",
        "treatment",
        "role",
        "order_in_pair",
        "params",
        "iterations",
        "early_stopping_rounds",
        "histogram_threads",
        "outer_train_indices_sha256",
        "outer_test_indices_sha256",
        "inner_train_indices_sha256",
        "inner_validation_indices_sha256",
        "inner_validation_fraction",
        "outer_test_used_for_early_stopping",
    }
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise ValueError("{} must be a lowercase SHA-256 hex digest".format(name))
    return value


def validate_frozen_job_config(
    config: Mapping[str, Any], data_identity: Mapping[str, Any]
) -> None:
    """Require one manifest config to match the frozen protocol exactly."""
    if not isinstance(config, Mapping):
        raise TypeError("frozen job config must be an object")
    if set(config) != _FROZEN_JOB_CONFIG_KEYS:
        raise ValueError("frozen job config key set mismatch")

    task_id = config["task_id"]
    fold = config["fold"]
    profile_name = config["profile"]
    treatment_name = config["treatment"]
    if type(task_id) is not int or type(fold) is not int:
        raise TypeError("frozen job task and fold must be integers")
    if type(profile_name) is not str or type(treatment_name) is not str:
        raise TypeError("frozen job profile and treatment must be strings")

    dataset_matches = [item for item in DATASETS if item["task_id"] == task_id]
    profile_matches = [item for item in PROFILES if item["name"] == profile_name]
    if len(dataset_matches) != 1:
        raise ValueError("frozen job refers to an unknown dataset")
    if len(profile_matches) != 1:
        raise ValueError("frozen job refers to an unknown profile")
    if fold not in FOLDS:
        raise ValueError("frozen job refers to an unknown fold")
    dataset = dataset_matches[0]
    profile = profile_matches[0]
    dataset_index = next(
        index for index, item in enumerate(DATASETS) if item["task_id"] == task_id
    )
    profile_index = next(
        index for index, item in enumerate(PROFILES) if item["name"] == profile_name
    )

    if treatment_name == "implicit-control":
        if fold != 0:
            raise ValueError("implicit control is frozen to fold zero")
        expected_role = "implicit_control_check"
        expected_order = 2
    else:
        ordered_treatments = treatment_order(dataset_index, fold, profile_index)
        if treatment_name not in ordered_treatments:
            raise ValueError("frozen job refers to an unknown treatment")
        expected_role = "treatment"
        expected_order = ordered_treatments.index(treatment_name)

    openml_identity = data_identity.get("openml")
    if not isinstance(openml_identity, Mapping):
        raise ValueError("data identity is missing OpenML task metadata")
    if type(openml_identity.get("task_id")) is not int:
        raise TypeError("data identity OpenML task ID must be an integer")
    if openml_identity["task_id"] != task_id:
        raise ValueError("job config and data identity task IDs differ")
    published_splits = data_identity.get("published_splits")
    if not isinstance(published_splits, list) or len(published_splits) != len(FOLDS):
        raise ValueError(
            "data identity must contain exactly the frozen published folds"
        )
    split_by_fold = {}
    for split in published_splits:
        if not isinstance(split, Mapping):
            raise TypeError("published split identities must be objects")
        identifiers = (split.get("repeat"), split.get("fold"), split.get("sample"))
        if any(type(value) is not int for value in identifiers):
            raise TypeError("published split identifiers must be integers")
        repeat, split_fold, sample = identifiers
        if repeat != REPEAT or split_fold not in FOLDS or sample != 0:
            raise ValueError("data identity contains a non-frozen published split")
        if split_fold in split_by_fold:
            raise ValueError("data identity contains duplicate published folds")
        _require_sha256_hex(
            split.get("train_indices_sha256"), "published train split identity"
        )
        _require_sha256_hex(
            split.get("test_indices_sha256"), "published test split identity"
        )
        split_by_fold[split_fold] = split
    if set(split_by_fold) != set(FOLDS):
        raise ValueError("data identity does not cover every frozen published fold")
    published_split = split_by_fold[fold]

    inner_train_sha256 = _require_sha256_hex(
        config["inner_train_indices_sha256"], "inner train split identity"
    )
    inner_validation_sha256 = _require_sha256_hex(
        config["inner_validation_indices_sha256"],
        "inner validation split identity",
    )
    outer_train_sha256 = _require_sha256_hex(
        config["outer_train_indices_sha256"], "outer train split identity"
    )
    outer_test_sha256 = _require_sha256_hex(
        config["outer_test_indices_sha256"], "outer test split identity"
    )
    if outer_train_sha256 != published_split["train_indices_sha256"]:
        raise ValueError("job outer train hash differs from its published split")
    if outer_test_sha256 != published_split["test_indices_sha256"]:
        raise ValueError("job outer test hash differs from its published split")

    params = dict(COMMON_PARAMS)
    params.update({key: value for key, value in profile.items() if key != "name"})
    if treatment_name != "implicit-control":
        treatment = next(item for item in TREATMENTS if item["name"] == treatment_name)
        params.update(treatment["params"])
    objective, eval_metric = {
        "binary": ("Logloss", "AUC"),
        "multiclass": ("MultiClass", "Logloss"),
        "regression": ("RMSE", "RMSE"),
    }[str(dataset["problem"])]
    params.update(
        {
            "objective": objective,
            "eval_metric": eval_metric,
            "random_seed": fit_seed(task_id, fold),
        }
    )
    expected = {
        "protocol_sha256": protocol_manifest()["protocol_sha256"],
        "task_id": int(dataset["task_id"]),
        "dataset_name": str(dataset["name"]),
        "problem": str(dataset["problem"]),
        "stress_only": bool(dataset["stress_only"]),
        "repeat": REPEAT,
        "fold": fold,
        "profile": str(profile["name"]),
        "treatment": treatment_name,
        "role": expected_role,
        "order_in_pair": expected_order,
        "params": params,
        "iterations": int(profile["iterations"]),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "histogram_threads": HISTOGRAM_THREADS,
        "outer_train_indices_sha256": outer_train_sha256,
        "outer_test_indices_sha256": outer_test_sha256,
        "inner_train_indices_sha256": inner_train_sha256,
        "inner_validation_indices_sha256": inner_validation_sha256,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "outer_test_used_for_early_stopping": False,
    }
    if canonical_json(config) != canonical_json(expected):
        raise ValueError("job config differs from the exact frozen semantics")


def encode_target(target: Any, problem: str) -> Tuple[Any, Optional[List[str]]]:
    import numpy as np
    import pandas as pd

    if problem == "regression":
        values = np.asarray(pd.to_numeric(target, errors="raise"), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("regression target contains non-finite values")
        return values, None
    categorical = pd.Categorical(target)
    codes = categorical.codes.astype(np.int64, copy=False)
    if np.any(codes < 0):
        raise ValueError("classification target contains missing values")
    return codes, [str(value) for value in categorical.categories.tolist()]


def _job_config(
    dataset: Mapping[str, Any],
    fold: int,
    profile: Mapping[str, Any],
    treatment_name: str,
    role: str,
    order_in_pair: int,
    outer_split: Mapping[str, Any],
    inner_train: Any,
    inner_valid: Any,
) -> Dict[str, Any]:
    import numpy as np

    profile_params = {key: value for key, value in profile.items() if key != "name"}
    params = dict(COMMON_PARAMS)
    params.update(profile_params)
    if treatment_name != "implicit-control":
        treatment = next(item for item in TREATMENTS if item["name"] == treatment_name)
        params.update(treatment["params"])
    objective, eval_metric = objective_and_metric(
        str(dataset["problem"]),
        None if dataset["problem"] == "regression" else int(dataset["class_count"]),
    )
    params.update(
        {
            "objective": objective,
            "eval_metric": eval_metric,
            "random_seed": fit_seed(int(dataset["task_id"]), fold),
        }
    )
    outer_train = np.asarray(outer_split["train"], dtype=np.int64)
    outer_test = np.asarray(outer_split["test"], dtype=np.int64)
    validate_inner_split(outer_train, outer_test, inner_train, inner_valid)
    return {
        "protocol_sha256": protocol_manifest()["protocol_sha256"],
        "task_id": int(dataset["task_id"]),
        "dataset_name": str(dataset["name"]),
        "problem": str(dataset["problem"]),
        "stress_only": bool(dataset["stress_only"]),
        "repeat": REPEAT,
        "fold": int(fold),
        "profile": str(profile["name"]),
        "treatment": treatment_name,
        "role": role,
        "order_in_pair": int(order_in_pair),
        "params": params,
        "iterations": int(profile["iterations"]),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "histogram_threads": HISTOGRAM_THREADS,
        "outer_train_indices_sha256": array_digest(
            outer_train.astype("<i8", copy=False)
        ),
        "outer_test_indices_sha256": array_digest(outer_test.astype("<i8", copy=False)),
        "inner_train_indices_sha256": array_digest(
            np.asarray(inner_train, dtype="<i8")
        ),
        "inner_validation_indices_sha256": array_digest(
            np.asarray(inner_valid, dtype="<i8")
        ),
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "outer_test_used_for_early_stopping": False,
    }


def build_jobs(
    staged_payloads: Sequence[Mapping[str, Any]], source_identity: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    if len(staged_payloads) != len(DATASETS):
        raise ValueError("one staged payload is required for every frozen task")
    for dataset_index, (dataset, payload) in enumerate(zip(DATASETS, staged_payloads)):
        if int(payload["metadata"]["task_id"]) != int(dataset["task_id"]):
            raise ValueError("staged task order does not match the frozen task ledger")
        target, class_labels = encode_target(payload["y"], str(dataset["problem"]))
        dataset_for_job = dict(dataset)
        dataset_for_job["class_count"] = (
            None if class_labels is None else len(class_labels)
        )
        for fold in FOLDS:
            outer_split = payload["outer_splits"][fold]
            inner_train, inner_valid = make_inner_validation_split(
                outer_split["train"],
                target,
                str(dataset["problem"]),
                fit_seed(int(dataset["task_id"]), fold),
            )
            validate_inner_split(
                outer_split["train"], outer_split["test"], inner_train, inner_valid
            )
            for profile_index, profile in enumerate(PROFILES):
                order = treatment_order(dataset_index, fold, profile_index)
                entries: List[Tuple[str, str, int]] = [
                    (name, "treatment", order_index)
                    for order_index, name in enumerate(order)
                ]
                if fold == 0:
                    entries.append(("implicit-control", "implicit_control_check", 2))
                for treatment_name, role, order_index in entries:
                    config = _job_config(
                        dataset_for_job,
                        fold,
                        profile,
                        treatment_name,
                        role,
                        order_index,
                        outer_split,
                        inner_train,
                        inner_valid,
                    )
                    data_identity = payload["data_identity"]
                    job_key = identity_digest(source_identity, data_identity, config)
                    jobs.append(
                        {
                            "job_key": job_key,
                            "source_identity": dict(source_identity),
                            "data_identity": dict(data_identity),
                            "config": config,
                            "stage_file": str(payload["_stage_path"]),
                            "stage_file_sha256": str(payload["_stage_file_sha256"]),
                            "inner_train_indices": inner_train.tolist(),
                            "inner_validation_indices": inner_valid.tolist(),
                            "class_labels": class_labels,
                        }
                    )
    expected = protocol_manifest()["expected_counts"]["total_subprocess_fits"]
    if len(jobs) != expected:
        raise AssertionError("generated job count differs from frozen protocol")
    if len({job["job_key"] for job in jobs}) != len(jobs):
        raise AssertionError("generated duplicate full-identity job keys")
    return jobs


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
