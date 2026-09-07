"""Gated local 0.1.59 HPO planning and one official TabArena parent per process.

Use the pinned environment's ``python -I``. Both commands require the completed
pilot decision and verify its underlying records. There is no override gate,
controller, automatic fit retry, or publication path in this module.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.59"
PORTFOLIO_ID = "ctboost_0159_learning_hpo25_v1"
TABARENA_COMMIT = "31026f7d758390994353eba79fbfa6747616f365"
PROTOCOL_FILE = "benchmarks/tabarena/pilot_0159_v1.json"
PILOT_SOURCE_FILES = (
    "benchmarks/tabarena/local_pilot.py",
    "benchmarks/tabarena/ctboost_model.py",
    "benchmarks/tabarena/learning_options.py",
    "benchmarks/tabarena/local_resources.py",
    "benchmarks/tabarena/pilot_evaluation.py",
)
HPO_SOURCE_FILES = (
    *PILOT_SOURCE_FILES,
    "benchmarks/tabarena/local_hpo.py",
    "benchmarks/tabarena/kaggle_hpo_worker.py",
)
PINS = {
    "ctboost": VERSION,
    "autogluon.common": "1.6.2b20260821",
    "autogluon.core": "1.6.2b20260821",
    "autogluon.tabular": "1.6.2b20260821",
    "numpy": "2.0.2",
    "pandas": "2.3.3",
    "scipy": "1.16.3",
    "scikit-learn": "1.6.1",
    "openml": "0.15.1",
    "pyarrow": "24.0.0",
}
CONTROL_NAMES = (
    "leaf_estimation_iterations",
    "leaf_estimation_backtracking",
    "multiclass_leaf_solver",
    "multiclass_feature_test",
)
GIB = 1024**3


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Windows status readers can briefly prevent atomic destination replacement.
    deadline = time.monotonic() + 5.0
    while True:
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if time.monotonic() >= deadline:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.01)


def now():
    return datetime.now(timezone.utc).isoformat()


def bootstrap_imports():
    # Loading benchmark modules must never make a developer extension shadow
    # the public wheel. This happens before ROOT is added to sys.path.
    import ctboost
    from ctboost import _core

    if (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents:
        raise RuntimeError("Use python -I with the installed public CTBoost wheel")
    if ctboost.__version__ != VERSION or ctboost.build_info().get("version") != VERSION:
        raise RuntimeError("The local HPO worker requires CTBoost 0.1.59")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ctboost, _core


def _pilot_metric_error(labels, predictions, dataset):
    from autogluon.core.metrics import get_metric

    metric = get_metric(dataset["eval_metric"], problem_type=dataset["problem_type"])
    return float(metric.error(labels, predictions))


def _verify_prediction_archive(path, record, dataset, prepared):
    import numpy as np

    from benchmarks.tabarena.local_pilot import indices_hash

    if not path.is_file() or file_hash(path) != record.get("prediction_sha256"):
        raise ValueError(f"Pilot prediction archive changed or is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"predictions", "labels", "validation_indices"}:
            raise ValueError("Unexpected pilot prediction archive fields")
        predictions = archive["predictions"]
        labels = archive["labels"]
        indices = archive["validation_indices"]
        if (
            labels.ndim != 1
            or not len(labels)
            or indices.shape != labels.shape
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or not np.isfinite(predictions).all()
            or not np.isfinite(labels).all()
        ):
            raise ValueError("Invalid pilot prediction archive")
        outer_rows = prepared["rows_outer_train"]
        split = prepared["splits"][str(record["inner_fold"])]
        if np.any(indices >= outer_rows) or len(indices) != split["rows_validation"]:
            raise ValueError(
                "Pilot validation indices exceed or differ from the frozen split"
            )
        training_indices = np.setdiff1d(
            np.arange(outer_rows), indices, assume_unique=True
        )
        if (
            len(training_indices) != split["rows_train"]
            or indices_hash(training_indices, indices) != record["split_sha256"]
        ):
            raise ValueError(
                "Pilot validation indices do not match the frozen inner split"
            )
        shape = (
            (len(labels), dataset["num_classes"])
            if dataset["problem_type"] == "multiclass"
            else labels.shape
        )
        if predictions.shape != shape:
            raise ValueError("Pilot prediction shape differs from its declared task")
        error = _pilot_metric_error(labels, predictions, dataset)
        if not math.isclose(
            error, record["metric_error"], rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Pilot validation metric disagrees with saved predictions")


def verify_pilot_gate(decision_path, protocol_path):
    """Recompute the gate and bind it to pre-fit provenance and saved evidence."""
    from benchmarks.tabarena.pilot_evaluation import evaluate_pilot

    decision_path, protocol_path = (
        Path(decision_path).resolve(),
        Path(protocol_path).resolve(),
    )
    protocol = read_json(protocol_path)
    protocol_hash = file_hash(protocol_path)
    if (
        protocol_hash != file_hash(ROOT / PROTOCOL_FILE)
        or protocol.get("ctboost_version") != VERSION
        or protocol.get("protocol_id") != "ctboost_0159_learning_pilot_v1"
        or protocol.get("conditional_hpo25", {}).get("new_portfolio_id") != PORTFOLIO_ID
    ):
        raise ValueError("The canonical frozen 0.1.59 pilot protocol is required")
    decision = read_json(decision_path)
    pilot_root = decision_path.parent
    record_paths = sorted((pilot_root / "fits").glob("*/inner*/*.json"))
    records = [read_json(path) for path in record_paths]
    recomputed = evaluate_pilot(protocol, records, protocol_sha256=protocol_hash)
    if decision != recomputed:
        raise ValueError(
            "Pilot decision is stale or differs from its underlying records"
        )
    if (
        recomputed.get("full_hpo_justified") is not True
        or recomputed.get("global_integrity_passed") is not True
        or recomputed.get("expected_fit_count") != 88
        or recomputed.get("observed_record_count") != 88
        or recomputed.get("missing_job_ids")
        or recomputed.get("duplicate_job_ids")
        or not any(
            mode != "baseline"
            for mode in recomputed["approved_modes_by_problem_type"].values()
        )
    ):
        raise ValueError(
            "Full HPO requires a complete, globally valid, passing pilot decision"
        )
    execution_path = pilot_root / "execution.json"
    execution = read_json(execution_path)
    if execution["protocol_sha256"] != protocol_hash:
        raise ValueError("Pilot execution protocol mismatch")
    runtime = execution["runtime"]
    if (
        set(runtime["source_sha256"]) != set(PILOT_SOURCE_FILES)
        or runtime["environment_sha256"] != json_hash(runtime["packages"])
        or execution["resource_contract_sha256"] != json_hash(execution["resources"])
    ):
        raise ValueError("Invalid pilot source, environment, or resource provenance")
    for name in PILOT_SOURCE_FILES:
        if file_hash(ROOT / name) != runtime["source_sha256"][name]:
            raise ValueError(f"Pilot source changed after execution: {name}")
    prepared = execution["prepared"]
    if (
        prepared.get("complete") is not True
        or prepared["protocol_sha256"] != protocol_hash
    ):
        raise ValueError("Pilot prepared-data provenance is incomplete")
    prepared_by_name = {item["dataset"]: item for item in prepared["datasets"]}
    datasets = {item["dataset_name"]: item for item in protocol["datasets"]}
    if set(prepared_by_name) != set(datasets) or len(prepared["datasets"]) != len(
        datasets
    ):
        raise ValueError("Pilot prepared dataset identities differ from the protocol")
    evidence = {}
    for name, item in prepared_by_name.items():
        directory = pilot_root / "data" / name
        if (
            item.get("contains_outer_test_rows") is not False
            or item.get("protocol_sha256") != protocol_hash
            or item.get("task_id") != datasets[name]["task_id"]
            or item.get("problem_type") != datasets[name]["problem_type"]
        ):
            raise ValueError(f"Pilot training-only task provenance mismatch: {name}")
        if read_json(directory / "prepared.json") != item:
            raise ValueError(f"Pilot prepared manifest changed: {name}")
        if file_hash(directory / "outer_train.pkl") != item["data_sha256"]:
            raise ValueError(f"Pilot outer-training artifact changed: {name}")
        evidence[(directory / "prepared.json").relative_to(pilot_root).as_posix()] = (
            file_hash(directory / "prepared.json")
        )
        evidence[(directory / "outer_train.pkl").relative_to(pilot_root).as_posix()] = (
            item["data_sha256"]
        )
    for path, record in zip(record_paths, records):
        item = prepared_by_name[record["dataset"]]
        split_hash = item["splits"][str(record["inner_fold"])]["sha256"]
        preprocessing_hash = json_hash(
            {
                "data_sha256": item["data_sha256"],
                "split_sha256": split_hash,
                "adapter_sha256": runtime["source_sha256"][
                    "benchmarks/tabarena/ctboost_model.py"
                ],
            }
        )
        if (
            record.get("execution_runtime_sha256") != json_hash(runtime)
            or record["resource_contract_sha256"]
            != execution["resource_contract_sha256"]
            or record["split_sha256"] != split_hash
            or record["preprocessing_sha256"] != preprocessing_hash
        ):
            raise ValueError(f"Pilot fit provenance mismatch: {record['job_id']}")
        expected_path = (
            pilot_root
            / "fits"
            / record["dataset"]
            / f"inner{record['inner_fold']}"
            / f"{record['variant']}.json"
        )
        if path != expected_path:
            raise ValueError("Pilot record path does not match its identity")
        evidence[path.relative_to(pilot_root).as_posix()] = file_hash(path)
        if record["status"] == "ok":
            prediction_path = path.with_suffix(".npz")
            _verify_prediction_archive(
                prediction_path, record, datasets[record["dataset"]], item
            )
            evidence[prediction_path.relative_to(pilot_root).as_posix()] = record[
                "prediction_sha256"
            ]
    return {
        "protocol": protocol,
        "decision": decision,
        "pilot_runtime": runtime,
        "binding": {
            "protocol_sha256": protocol_hash,
            "decision_sha256": file_hash(decision_path),
            "execution_sha256": file_hash(execution_path),
            "pilot_evidence_sha256": json_hash(evidence),
            "pilot_evidence": evidence,
        },
    }


def runtime_provenance(gate):
    ctboost, core = bootstrap_imports()
    import tabarena

    packages = {name: importlib.metadata.version(name) for name in PINS}
    if packages != PINS:
        raise RuntimeError(f"Pinned HPO runtime mismatch: {packages}")
    environment = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    source_roots = [
        path
        for path in Path(tabarena.__file__).resolve().parents
        if (path / ".git").exists()
    ]
    if not source_roots:
        raise RuntimeError("The pinned editable TabArena source checkout is required")
    source = source_roots[0]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        commit != TABARENA_COMMIT
        or subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "HEAD", "--", "packages"],
            check=False,
        ).returncode
    ):
        raise RuntimeError("Pinned TabArena source changed")
    metadata = source / gate["protocol"]["provenance"]["metadata_path"]
    if file_hash(metadata) != gate["protocol"]["provenance"]["metadata_sha256"]:
        raise RuntimeError("Pinned TabArena metadata changed")
    native_hash = file_hash(core.__file__)
    if native_hash != gate["pilot_runtime"]["native_extension_sha256"]:
        raise RuntimeError("The HPO native wheel differs from the verified pilot wheel")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "environment": environment,
        "environment_sha256": json_hash(environment),
        "ctboost_build_info": ctboost.build_info(),
        "native_extension_sha256": native_hash,
        "tabarena_commit": commit,
        "metadata_sha256": file_hash(metadata),
        "source_sha256": {name: file_hash(ROOT / name) for name in HPO_SOURCE_FILES},
    }


def resource_contract(num_cpus, memory_limit_gb):
    if (
        isinstance(num_cpus, bool)
        or not isinstance(num_cpus, int)
        or not 1 <= num_cpus <= 16
    ):
        raise ValueError("num_cpus must be an integer from 1 to 16")
    if (
        isinstance(memory_limit_gb, bool)
        or not isinstance(memory_limit_gb, (int, float))
        or not math.isfinite(memory_limit_gb)
        or not 1 <= memory_limit_gb <= 8
    ):
        raise ValueError("memory_limit_gb must be finite and between 1 and 8")
    return {
        "num_cpus": num_cpus,
        "num_gpus": 0,
        "memory_limit_gb": float(memory_limit_gb),
        "time_limit_seconds": 3600,
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "fold_fitting_strategy": "sequential_local",
        "blas_openmp_threads": 1,
    }


def build_experiments(modes, resources):
    from tabarena.utils.config_utils import CustomAGConfigGenerator

    from benchmarks.tabarena.ctboost_model import (
        CTBoostTabArenaModel,
        generate_configs_ctboost_learning_options,
    )

    generate = partial(
        generate_configs_ctboost_learning_options, approved_modes_by_problem_type=modes
    )
    generator = CustomAGConfigGenerator(
        model_cls=CTBoostTabArenaModel, search_space_func=generate, manual_configs=[{}]
    )
    experiments = generator.generate_all_bag_experiments(
        num_random_configs=25,
        name_id_suffix="_default",
        add_seed="fold-config-wise",
        num_bag_folds=8,
        num_bag_sets=1,
        fold_fitting_strategy="sequential_local",
        time_limit=3600,
        time_limit_with_preprocessing=False,
        preprocessing_pipeline="default",
        dynamic_tabarena_validation_protocol=False,
        method_kwargs={
            "init_kwargs": {"verbosity": 2},
            "fit_kwargs": {
                "num_cpus": resources["num_cpus"],
                "num_gpus": 0,
                "memory_limit": resources["memory_limit_gb"],
            },
            "shuffle_features": False,
        },
    )
    expected_names = [
        f"CTBoost_{'c1' if index == 0 else f'r{index}'}_default_BAG_L1"
        for index in range(26)
    ]
    if [experiment.name for experiment in experiments] != expected_names:
        raise RuntimeError("Unexpected globally generated HPO configuration identities")
    for index, experiment in enumerate(experiments):
        ensemble = experiment.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]
        if (
            ensemble["model_random_seed"] != index * 8
            or ensemble["vary_seed_across_folds"] is not True
            or ensemble["fold_fitting_strategy"] != "sequential_local"
        ):
            raise RuntimeError("HPO child seed or sequential-fold contract changed")
    return experiments, [{}] + generate(25)


def _plan_payload(gate, runtime, resources, experiments, configs):
    from benchmarks.tabarena.ctboost_model import generate_configs_ctboost

    protocol = gate["protocol"]
    if (
        json_hash(generate_configs_ctboost(200))
        != protocol["provenance"]["old_portfolio_200_sha256"]
    ):
        raise RuntimeError("The frozen 200-config base portfolio changed")
    population = protocol["selection"]["population_metadata"]
    if (
        len(population) != 51
        or len({item["dataset_name"] for item in population}) != 51
        or len({item["task_id"] for item in population}) != 51
    ):
        raise ValueError(
            "The full HPO population must contain 51 unique official Lite tasks"
        )
    parents = [
        {
            "dataset": dataset["dataset_name"],
            "task_id": dataset["task_id"],
            "problem_type": dataset["problem_type"],
            "config_index": index,
            "config_name": experiment.name,
            "repeat": 0,
            "fold": 0,
            "child_seeds": list(range(index * 8, index * 8 + 8)),
        }
        for index, experiment in enumerate(experiments)
        for dataset in population
    ]
    return {
        "schema_version": 1,
        "portfolio_id": PORTFOLIO_ID,
        "ctboost_version": VERSION,
        "pilot": gate["binding"],
        "approved_modes_by_problem_type": gate["decision"][
            "approved_modes_by_problem_type"
        ],
        "runtime": runtime,
        "resources": resources,
        "configurations": configs,
        "configurations_sha256": json_hash(configs),
        "experiments": [experiment.to_yaml_dict() for experiment in experiments],
        "parents": parents,
        "expected_parent_count": 1326,
        "expected_child_count": 10608,
        "timing_disclosure": "Author-run local CPU timings; not canonical TabArena hardware. "
        "Pilot selection reused some datasets' outer-training rows.",
    }


def build_plan(*, decision_path, protocol_path, output, num_cpus=2, memory_limit_gb=8):
    gate = verify_pilot_gate(decision_path, protocol_path)
    runtime = runtime_provenance(gate)
    resources = resource_contract(num_cpus, memory_limit_gb)
    experiments, configs = build_experiments(
        gate["decision"]["approved_modes_by_problem_type"], resources
    )
    plan = _plan_payload(gate, runtime, resources, experiments, configs)
    output = Path(output).resolve()
    path = output / "plan.json"
    if path.exists():
        if read_json(path) != plan:
            raise ValueError("Existing HPO plan differs; use a new results directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("A new HPO plan requires an empty results directory")
        write_json(path, plan)
    return plan


def audit_children(wrapper, parent):
    """Inspect the actual fitted native controls before TabArena cleans up."""
    from benchmarks.tabarena.learning_options import (
        LEARNING_VARIANT_PARAM,
        resolve_cpu_learning_options,
    )

    bag = wrapper._load_model()
    names = [entry if isinstance(entry, str) else entry.name for entry in bag.models]
    expected_names = [f"S1F{fold}" for fold in range(1, 9)]
    if names != expected_names:
        raise ValueError("Fitted child order differs from the eight-fold protocol")
    audits = {}
    for fold, entry in enumerate(bag.models):
        child = bag.load_child(entry)
        params = child.model.get_params(deep=False)
        handle = child.model.get_booster()._handle
        actual = {name: getattr(handle, name)() for name in CONTROL_NAMES}
        num_classes = getattr(child.model, "n_classes_", None)
        marked = LEARNING_VARIANT_PARAM in child.params
        _, expected = resolve_cpu_learning_options(
            child.params,
            problem_type=child.problem_type,
            num_classes=num_classes,
            variant=None if marked else "baseline",
        )
        if actual != expected["resolved_controls"]:
            raise ValueError(
                f"Native learning controls do not match the approved variant: {child.name}"
            )
        metadata = getattr(child, "_ctboost_learning_options", None)
        if marked and metadata != expected:
            raise ValueError(
                f"Recorded learning applicability differs from fitted task: {child.name}"
            )
        seed = parent["config_index"] * 8 + fold
        if (
            params.get("task_type") != "CPU"
            or params.get("random_seed") != seed
            or child.params.get("random_seed") != seed
            or params.get("feature_test")
            != child.params.get("feature_test", "quadratic")
        ):
            raise ValueError(
                "Fitted child resource, seed, or statistical-test identity changed"
            )
        audits[child.name] = {
            "random_seed": seed,
            "native_controls": actual,
            "applicability": expected,
            "feature_test": params["feature_test"],
            "task_type": "CPU",
        }
    return audits


def _audited_runner(parent, plan_sha256):
    from tabarena.benchmark.experiment import OOFExperimentRunner

    class AuditedLocalHpoRunner(OOFExperimentRunner):
        def post_evaluate(self, out):
            out = super().post_evaluate(out)
            out["local_hpo"] = {
                "ctboost_version": VERSION,
                "portfolio_id": PORTFOLIO_ID,
                "plan_sha256": plan_sha256,
                "children": audit_children(self.model, parent),
            }
            return out

        def run(self):
            try:
                return super().run()
            except Exception:
                if getattr(self, "model", None) is not None:
                    try:
                        self._cleanup()
                    except Exception:
                        pass  # Preserve the original fit/audit failure.
                raise

    return AuditedLocalHpoRunner


def validate_result(path, parent, plan_sha256):
    """Reuse the version-independent raw validator, then require 0.1.59 evidence."""
    import numpy as np

    from benchmarks.tabarena.kaggle_hpo_worker import validate_result_file

    spec = {**parent, "datasets": [parent["dataset"]]}
    validated = validate_result_file(Path(path), spec)
    if str(validated["task_id"]) != str(parent["task_id"]):
        raise ValueError("Raw result uses the wrong official task ID")
    with gzip.open(path, "rb") as stream:
        result = pickle.load(stream)
    audit = result.get("local_hpo", {})
    if (
        audit.get("ctboost_version") != VERSION
        or audit.get("portfolio_id") != PORTFOLIO_ID
        or audit.get("plan_sha256") != plan_sha256
        or set(audit.get("children", {})) != {f"S1F{fold}" for fold in range(1, 9)}
    ):
        raise ValueError(
            "Raw result lacks matching 0.1.59 plan and child-control provenance"
        )
    if result["problem_type"] != parent["problem_type"]:
        raise ValueError("Raw result problem type differs from the official metadata")
    for fold in range(8):
        child = audit["children"][f"S1F{fold + 1}"]
        if (
            child.get("task_type") != "CPU"
            or child.get("random_seed") != parent["config_index"] * 8 + fold
            or child.get("native_controls")
            != child.get("applicability", {}).get("resolved_controls")
        ):
            raise ValueError("Invalid native child-control or seed evidence")
    simulation = result["simulation_artifacts"]
    children = np.asarray(simulation["bag_info"]["pred_proba_test_per_child"])
    ensemble = np.asarray(simulation["pred_proba_dict_test"][parent["config_name"]])
    if not np.allclose(np.mean(children, axis=0), ensemble, rtol=1e-5, atol=1e-6):
        raise ValueError(
            "Ensemble test predictions differ from the eight-child average"
        )
    if parent["problem_type"] != "regression":
        if np.any(children < -1e-6) or np.any(children > 1 + 1e-6):
            raise ValueError("Invalid child test probabilities")
        if parent["problem_type"] == "multiclass" and not np.allclose(
            children.sum(axis=2),
            1.0,
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Child test probability rows do not sum to one")
    return {
        **validated,
        "raw_sha256": file_hash(path),
        "size_bytes": Path(path).stat().st_size,
        "native_children": audit["children"],
    }


@contextmanager
def _parent_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _configure_process(num_cpus, affinity):
    import psutil

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["CTBOOST_HIST_THREADS"] = str(num_cpus)
    process = psutil.Process()
    allowed = process.cpu_affinity()
    if affinity is not None:
        if (
            len(affinity) != num_cpus
            or len(set(affinity)) != num_cpus
            or not set(affinity) <= set(allowed)
        ):
            raise ValueError(
                "CPU affinity must contain num_cpus distinct available logical CPUs"
            )
        process.cpu_affinity(affinity)
    return process.cpu_affinity()


@contextmanager
def _memory_guard(limit_bytes, failure_path):
    """Terminate only this owned worker on RSS overflow, keeping an atomic cause."""
    import psutil

    process = psutil.Process()
    stopped = threading.Event()
    state = {"peak_rss_bytes": 0}

    def sample():
        while not stopped.is_set():
            rss = process.memory_info().rss
            state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss)
            if rss > limit_bytes:
                _terminate_memory_overflow(failure_path, rss, limit_bytes)
            stopped.wait(0.05)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        stopped.set()
        thread.join(timeout=1)


def _terminate_memory_overflow(failure_path, rss, limit_bytes):
    try:
        write_json(
            failure_path,
            {
                "status": "resource_limit",
                "rss_bytes": rss,
                "limit_bytes": limit_bytes,
                "pid": os.getpid(),
                "recorded_at": now(),
            },
        )
    finally:
        # A failed checkpoint must not disable enforcement while training runs.
        os._exit(3)


def _load_task(parent, output):
    import openml
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.benchmark.task.spec import task_spec_from_task_id_str

    openml.config.set_root_cache_directory(str(output / "openml_cache"))
    collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
    metadata = collection.task_metadata_by_dataset()[parent["dataset"]]
    spec = task_spec_from_task_id_str(metadata.task_id_str).with_task_metadata(metadata)
    if (
        str(spec.cache_key) != str(parent["task_id"])
        or metadata.problem_type != parent["problem_type"]
    ):
        raise ValueError(
            "Official task metadata differs from the frozen HPO population"
        )
    return spec, spec.load()


def run_parent(
    *,
    dataset,
    config_index,
    num_cpus,
    memory_limit_gb,
    output,
    decision_path,
    protocol_path,
    affinity=None,
):
    """Run one fresh parent or validate an existing checkpoint of this exact plan."""
    resources = resource_contract(num_cpus, memory_limit_gb)
    affinity = _configure_process(num_cpus, affinity)
    bootstrap_imports()
    gate = verify_pilot_gate(decision_path, protocol_path)
    runtime = runtime_provenance(gate)
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    experiments, configs = build_experiments(
        gate["decision"]["approved_modes_by_problem_type"], resources
    )
    if plan != _plan_payload(gate, runtime, resources, experiments, configs):
        raise ValueError(
            "Worker inputs or provenance differ from the immutable HPO plan"
        )
    if isinstance(config_index, bool) or not isinstance(config_index, int):
        raise ValueError("config_index must be an integer from 0 to 25")
    matches = [
        parent
        for parent in plan["parents"]
        if parent["dataset"] == dataset and parent["config_index"] == config_index
    ]
    if len(matches) != 1:
        raise ValueError("Requested parent is not part of the frozen HPO plan")
    parent = matches[0]
    plan_hash = file_hash(output / "plan.json")
    directory = (
        output / "artifacts" / parent["config_name"] / str(parent["task_id"]) / "0_0"
    )
    raw = (
        output
        / "data"
        / parent["config_name"]
        / str(parent["task_id"])
        / "0_0"
        / "results.pkl"
    )
    manifest_path = directory / "manifest.json"
    with _parent_lock(directory / "worker.lock"):
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if (
                previous.get("plan_sha256") != plan_hash
                or previous.get("parent") != parent
            ):
                raise ValueError(
                    "Existing parent checkpoint belongs to different provenance"
                )
            if not raw.is_file() or (directory / "resource_failure.json").exists():
                raise ValueError(
                    "Incomplete parent requires controller reconciliation; automatic fit retry is disabled"
                )
            if previous.get("validation", {}).get("raw_sha256") not in (
                None,
                file_hash(raw),
            ):
                raise ValueError("Completed raw result changed since its checkpoint")
            validated = validate_result(raw, parent, plan_hash)
            previous.update(status="complete", validation=validated, validated_at=now())
            write_json(manifest_path, previous)
            return previous
        if raw.exists() or any(raw.parent.glob("*")):
            raise ValueError(
                "Existing raw artifacts without matching worker provenance are refused"
            )
        manifest = {
            "schema_version": 1,
            "status": "preparing",
            "started_at": now(),
            "plan_sha256": plan_hash,
            "parent": parent,
            "resources": resources,
            "affinity": affinity,
            "pid": os.getpid(),
            "ctboost_version": VERSION,
            "raw_path": raw.relative_to(output).as_posix(),
        }
        write_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            from tabarena.utils.cache import CacheFunctionPickle
            from threadpoolctl import threadpool_limits

            with _memory_guard(
                int(memory_limit_gb * GIB), directory / "resource_failure.json"
            ) as memory:
                spec, task = _load_task(parent, output)
                experiment = copy.deepcopy(experiments[config_index])
                experiment.experiment_cls = _audited_runner(parent, plan_hash)
                cacher = CacheFunctionPickle(
                    cache_name="results",
                    cache_path=raw.parent,
                    include_self_in_call=True,
                )
                manifest.update(status="running", fit_started_at=now())
                write_json(manifest_path, manifest)
                with threadpool_limits(limits=1):
                    experiment.run(
                        task=task,
                        fold=0,
                        repeat=0,
                        sample=0,
                        task_name=spec.resolve_task_name(task),
                        cache_task_key=spec.cache_key,
                        cacher=cacher,
                        ignore_cache=False,
                        raise_on_failure=True,
                        debug_mode=False,
                        eval_metric_name=task.eval_metric,
                    )
                validation = validate_result(raw, parent, plan_hash)
            manifest.update(status="complete", validation=validation, **memory)
        except Exception as exc:
            manifest.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            manifest.update(
                finished_at=now(), elapsed_seconds=time.monotonic() - started
            )
            write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "worker"):
        command = commands.add_parser(name)
        command.add_argument("--decision", required=True, type=Path)
        command.add_argument("--protocol", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--num-cpus", type=int, default=2)
        command.add_argument("--memory-limit-gb", type=float, default=8)
        if name == "worker":
            command.add_argument("--dataset", required=True)
            command.add_argument("--config-index", required=True, type=int)
            command.add_argument(
                "--affinity",
                help="Comma-separated logical CPU IDs assigned by the controller",
            )
    args = parser.parse_args(argv)
    # Set native/BLAS thread environment before importing the installed wheel.
    if args.command == "worker":
        _configure_process(args.num_cpus, None)
    bootstrap_imports()
    common = {
        "decision_path": args.decision,
        "protocol_path": args.protocol,
        "output": args.output,
        "num_cpus": args.num_cpus,
        "memory_limit_gb": args.memory_limit_gb,
    }
    if args.command == "plan":
        plan = build_plan(**common)
        print(
            json.dumps(
                {
                    "status": "planned",
                    "parents": len(plan["parents"]),
                    "plan_sha256": file_hash(args.output / "plan.json"),
                }
            )
        )
    else:
        affinity = (
            None
            if args.affinity is None
            else [int(value) for value in args.affinity.split(",")]
        )
        result = run_parent(
            **common,
            dataset=args.dataset,
            config_index=args.config_index,
            affinity=affinity,
        )
        print(json.dumps({"status": result["status"], "parent": result["parent"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
