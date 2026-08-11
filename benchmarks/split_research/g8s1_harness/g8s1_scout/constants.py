"""Immutable identities and resource contracts from TABARENA_GROUPED_SCOUT.md."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_SHA256 = "fa93282597fba4b7758afb787589e9ad18bbc7ffea22894b21233d8c867d0aff"
PROTOCOL_LF_NORMALIZED_SHA256 = (
    "3d761e5fdf78c319bb99f1da64e9440402c23b84aa086b234658a67e9ba2dc9a"
)
P50_SHA256 = "edf9ec119040cf687220737a8410c3406979ced4b6c47b969ca6f4f5a8e8b6fe"
P200_RANDOM_SHA256 = "bd1b81b98a89ab33ac4cea35cb4b7dd7727b3bcfa3bee1b044fd3fb44f965c72"
P201_SHA256 = "73df0b36b7db41f66adf0dcebd77805e612b579ade1ac8c07ac0399e3ae445b5"
EXPECTED_SCHEDULE_SHA256 = (
    "209035b50ff73c49e330dfad4b76629a8b874ade4cd0e6d8a95e72295368d7eb"
)
PROTOCOL_TABARENA_COMMIT = "50f8ab1bbc6e7f7e5dd9b19d8b643ac284ae9b3c"

RUNTIME_MODULE_FILES = (
    "__init__.py",
    "__main__.py",
    "constants.py",
    "identity.py",
    "loader.py",
    "models.py",
    "schedule.py",
    "summary.py",
)
RUNTIME_DATA_FILES = ("p200.json",)
RUNTIME_FILES = (*RUNTIME_MODULE_FILES, *RUNTIME_DATA_FILES)
CANONICAL_P200_FILE = RUNTIME_DATA_FILES[0]
BOOTSTRAP_RELATIVE = "benchmarks/split_research/g8s1_scout_bootstrap.py"
RUNBOOK_RELATIVE = "benchmarks/split_research/G8S1_SCOUT_RUNBOOK.md"
RUNBOOK_LF_NORMALIZED_SHA256 = (
    "d97454c737df423504291d39352d8f07df0d355de7af8962a8140dafb0551d3d"
)

TASKS = (
    (363614, "anneal", "multiclass", "log_loss"),
    (363621, "blood-transfusion-service-center", "binary", "roc_auc"),
    (363698, "QSAR_fish_toxicity", "regression", "rmse"),
)
DATASETS = tuple(task[1] for task in TASKS)

TREATMENTS = {
    "quadratic": {
        "ag_key": "CTBQS1",
        "ag_name": "CTBoostQuadraticScoutV1",
        "model_class": "CTBoostQuadraticScoutV1Model",
    },
    "grouped": {
        "ag_key": "CTBG8S1",
        "ag_name": "CTBoostGrouped8ScoutV1",
        "model_class": "CTBoostGrouped8ScoutV1Model",
    },
}

NUM_RANDOM_CONFIGS = 50
NUM_CONFIGS = 51
NUM_METHODS = 2
EXPECTED_ARTIFACTS = 306
EXPECTED_CHILD_FITS = 2448
EXPECTED_CHUNKS = (102, 102, 102)
NUM_CPUS = 8
NUM_GPUS = 0
MEMORY_LIMIT_GB = 32
TIME_LIMIT_SECONDS = 3600
JOB_BATCH_SIZE = 8
HISTOGRAM_THREADS = "8"

TREATMENT_COMMON = {
    "feature_test_bins": 8,
    "feature_test_adjustment": "none",
}
FORBIDDEN_BASE_FIELDS = frozenset(
    {"feature_test", "feature_test_bins", "feature_test_adjustment"}
)


def source_root() -> Path:
    """Return the CTBoost checkout that owns this tracked harness."""
    return Path(__file__).resolve().parents[4]


def harness_package_root() -> Path:
    return (
        source_root() / "benchmarks" / "split_research" / "g8s1_harness" / "g8s1_scout"
    )


def manifest_path() -> Path:
    return source_root() / "benchmarks" / "split_research" / "G8S1_SCOUT_MANIFEST.json"


def canonical_p200_path() -> Path:
    return harness_package_root() / CANONICAL_P200_FILE


def bootstrap_path() -> Path:
    return source_root() / Path(BOOTSTRAP_RELATIVE)


def runbook_path() -> Path:
    return source_root() / Path(RUNBOOK_RELATIVE)


def protocol_path() -> Path:
    return source_root() / "benchmarks" / "split_research" / "TABARENA_GROUPED_SCOUT.md"


def config_id(index: int) -> str:
    if index == 0:
        return "c1"
    if 1 <= index <= NUM_RANDOM_CONFIGS:
        return f"r{index}"
    raise ValueError(f"configuration index out of range: {index}")


def experiment_name(treatment: str, index: int) -> str:
    return f"{TREATMENTS[treatment]['ag_name']}_{config_id(index)}_default_BAG_L1"


def namespace_for_commit(ctboost_commit: str) -> str:
    if len(ctboost_commit) != 40:
        raise ValueError("CTBoost commit must be a full 40-character SHA-1")
    return f"g8s1-p50-cpu8-ct{ctboost_commit[:12]}-ta{PROTOCOL_TABARENA_COMMIT[:8]}"
