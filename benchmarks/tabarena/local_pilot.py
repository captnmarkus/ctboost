"""Run the frozen 0.1.59 validation pilot in resumable local CPU processes.

Invoke with the isolated benchmark environment's ``python -I``. Preparation
stores only official outer-training rows. Each worker evaluates all variants
for one inner fold, with the same CPU affinity and an atomic record per fit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import random
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def bootstrap_imports():
    # Import the installed wheel before exposing the checkout's benchmark code.
    # The checkout may contain an older developer extension with the same ABI.
    import ctboost
    from ctboost import _core

    if (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents:
        raise RuntimeError("Use python -I and the installed public CTBoost wheel")
    if ctboost.__version__ != "0.1.59" or ctboost.build_info()["version"] != "0.1.59":
        raise RuntimeError("This pilot requires CTBoost 0.1.59")
    sys.path.insert(0, str(ROOT))
    return ctboost, _core


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers briefly hold the destination without delete sharing.
    # Keep atomic replacement and retry that transient lock, instead of turning
    # the controller's progress polling into a failed model fit.
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


def read_protocol(path):
    path = Path(path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["ctboost_version"] != "0.1.59":
        raise ValueError("Unexpected pilot version")
    return protocol, file_hash(path)


def indices_hash(*arrays):
    import numpy as np

    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype="<i8")
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def prepare_dataset(dataset, protocol, protocol_hash, output, openml_cache):
    import numpy as np
    import openml
    from autogluon.common.utils.cv_splitter import CVSplitter

    name = dataset["dataset_name"]
    destination = output / "data" / name
    manifest_path = destination / "prepared.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["protocol_sha256"] != protocol_hash:
            raise ValueError(f"Prepared protocol mismatch: {name}")
        if file_hash(destination / "outer_train.pkl") != existing["data_sha256"]:
            raise ValueError(f"Prepared data hash mismatch: {name}")
        return existing
    if dataset.get("group_on") or dataset.get("time_on") or dataset["task_type"] != "random":
        raise ValueError(f"Pilot requires the registered random split: {name}")
    openml.config.set_root_cache_directory(str(openml_cache))
    task = openml.tasks.get_task(int(dataset["task_id"]), download_data=True,
                                 download_splits=True)
    train_indices, test_indices = task.get_train_test_split_indices(fold=0, repeat=0, sample=0)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if len(np.intersect1d(train_indices, test_indices)):
        raise ValueError("Official outer train/test indices overlap")
    raw_X, raw_y, _, _ = task.get_dataset().get_data(target=task.target_name,
                                                   dataset_format="dataframe")
    X = raw_X.iloc[train_indices].copy().reset_index(drop=True)
    y = raw_y.iloc[train_indices].copy().reset_index(drop=True)
    del raw_X, raw_y, test_indices
    if len(X) != len(y) or not len(X) or y.isna().any():
        raise ValueError(f"Invalid outer-training data: {name}")
    problem_type = dataset["problem_type"]
    if problem_type != "regression" and y.nunique() != dataset["num_classes"]:
        raise ValueError(f"Class-count metadata changed: {name}")
    split_spec = protocol["splits"]
    splitter = CVSplitter(**split_spec["inner_splitter_kwargs"],
                          stratify=split_spec["stratify_by_problem_type"][problem_type],
                          bin=split_spec["bin_by_problem_type"][problem_type])
    all_splits = splitter.split(X, y)
    selected = {}
    for fold in split_spec["inner_fold_indices"]:
        fit_idx, val_idx = [np.asarray(idx, dtype=np.int64) for idx in all_splits[fold]]
        if len(np.intersect1d(fit_idx, val_idx)) or len(fit_idx) + len(val_idx) != len(X):
            raise ValueError("Invalid inner partition")
        selected[str(fold)] = {"train": fit_idx, "validation": val_idx,
                               "sha256": indices_hash(fit_idx, val_idx)}
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / "outer_train.pkl"
    temporary = data_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump({"X": X, "y": y, "splits": selected}, stream, protocol=5)
    temporary.replace(data_path)
    manifest = {"dataset": name, "task_id": dataset["task_id"],
                "dataset_id": task.dataset_id, "problem_type": problem_type,
                "rows_outer_train": len(X), "features": X.shape[1],
                "outer_train_indices_sha256": indices_hash(train_indices),
                "data_sha256": file_hash(data_path), "protocol_sha256": protocol_hash,
                "splits": {key: {"sha256": value["sha256"],
                                  "rows_train": len(value["train"]),
                                  "rows_validation": len(value["validation"])}
                           for key, value in selected.items()},
                "contains_outer_test_rows": False, "created_at": now()}
    write_json(manifest_path, manifest)
    return manifest


def prepare(args):
    bootstrap_imports()
    protocol, protocol_hash = read_protocol(args.protocol)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset in protocol["datasets"]:
        print(f"Preparing {dataset['dataset_name']}", flush=True)
        records.append(prepare_dataset(dataset, protocol, protocol_hash, output,
                                       args.openml_cache))
        write_json(output / "preparation.json", {"protocol_sha256": protocol_hash,
                   "datasets": records, "complete": len(records) == len(protocol["datasets"])})
    print(f"Prepared {len(records)} outer-training datasets", flush=True)


def record_path(output, dataset, fold, variant):
    return output / "fits" / dataset / f"inner{fold}" / f"{variant}.json"


def runtime_provenance():
    ctboost, core = bootstrap_imports()
    packages = {dist.metadata["Name"]: dist.version
                for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    files = ["benchmarks/tabarena/local_pilot.py", "benchmarks/tabarena/ctboost_model.py",
             "benchmarks/tabarena/learning_options.py", "benchmarks/tabarena/local_resources.py",
             "benchmarks/tabarena/pilot_evaluation.py"]
    return {"python": sys.version, "python_executable": sys.executable,
            "packages": packages, "environment_sha256": json_hash(packages),
            "ctboost_build_info": ctboost.build_info(),
            "native_extension_sha256": file_hash(core.__file__),
            "source_sha256": {name: file_hash(ROOT / name) for name in files}}


def build_execution_manifest(protocol, protocol_hash, calibration, output):
    import psutil

    selected = calibration["selected"]
    if selected is None:
        raise ValueError("No successful synthetic resource calibration")
    workers, threads = selected["workers"], selected["threads_per_worker"]
    allowed = protocol["compute"]["cpu_resource_calibration"]["allowed_worker_thread_layouts"]
    if [workers, threads] not in allowed:
        raise ValueError("Calibration selected an unregistered worker layout")
    spec = protocol["compute"]
    free_reserve = int(spec["minimum_free_physical_memory_gib"] * GIB)
    memory_budget = min(int(spec["max_worker_process_tree_rss_gib"] * GIB),
                        psutil.virtual_memory().available - free_reserve)
    per_worker = min(int(spec["max_process_rss_gib"] * GIB), memory_budget // workers)
    if per_worker < GIB:
        raise RuntimeError("Insufficient available memory for the selected worker layout")
    resources = {"workers": workers, "threads_per_worker": threads,
                 "memory_per_worker_bytes": per_worker,
                 "memory_budget_bytes": memory_budget,
                 "minimum_free_bytes": free_reserve,
                 "fit_time_limit_seconds": spec["fit_time_limit_seconds"],
                 "hard_fit_grace_seconds": spec["hard_fit_grace_seconds"],
                 "blas_openmp_threads": 1}
    prepared = json.loads((output / "preparation.json").read_text())
    if not prepared["complete"] or prepared["protocol_sha256"] != protocol_hash:
        raise ValueError("Complete, matching prepared training data is required")
    runtime = runtime_provenance()
    if importlib.metadata.version("autogluon.tabular") != protocol["provenance"]["autogluon_version"]:
        raise ValueError("AutoGluon version differs from the registered protocol")
    wheel = public_wheel_provenance(output, runtime["native_extension_sha256"])
    return {"created_at": now(), "protocol_sha256": protocol_hash,
            "resources": resources, "resource_contract_sha256": json_hash(resources),
            "affinity": selected["affinity"], "calibration_sha256": json_hash(calibration),
            "prepared": prepared, "runtime": runtime, "public_wheel": wheel}


def public_wheel_provenance(output, installed_native_sha256):
    """Bind the cached PyPI wheel to the exact installed native extension."""
    matches = []
    for wheel in sorted((output.parent / "wheels").glob("ctboost-0.1.59-*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            native = [name for name in archive.namelist()
                      if name.startswith("ctboost/_core") and name.endswith((".pyd", ".so"))]
            if len(native) != 1:
                continue
            native_hash = hashlib.sha256(archive.read(native[0])).hexdigest()
            if native_hash == installed_native_sha256:
                matches.append({"filename": wheel.name, "sha256": file_hash(wheel),
                                "native_entry": native[0], "native_sha256": native_hash,
                                "source": "PyPI", "project_url": "https://pypi.org/project/ctboost/0.1.59/"})
    if len(matches) != 1:
        raise ValueError("Require exactly one cached public 0.1.59 wheel matching the installed native binary")
    return matches[0]


class MemorySampler:
    def __init__(self):
        self.peak = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        import psutil

        process = psutil.Process()
        while not self.done.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.done.wait(0.02)

    def __enter__(self):
        import psutil

        self.peak = psutil.Process().memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *_args):
        import psutil

        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.done.set()
        self.thread.join()


def worker_identity():
    import psutil

    return {"pid": os.getpid(), "process_create_time": psutil.Process().create_time()}


def verify_prepared_inputs(execution, protocol_hash, prepared, data_path, dataset):
    """Bind current inputs to the pre-fit manifest, not a mutable sidecar alone."""
    if execution["protocol_sha256"] != protocol_hash or prepared["protocol_sha256"] != protocol_hash:
        raise ValueError("Prepared/execution protocol mismatch")
    frozen = [row for row in execution["prepared"]["datasets"] if row["dataset"] == dataset]
    if len(frozen) != 1 or frozen[0] != prepared:
        raise ValueError("Prepared manifest changed after execution was frozen")
    if file_hash(data_path) != frozen[0]["data_sha256"]:
        raise ValueError("Prepared data changed after execution was frozen")


def run_block(args):
    import psutil

    output = args.output.resolve()
    execution = json.loads((output / "execution.json").read_text())
    resources = execution["resources"]
    affinity = execution["affinity"][args.slot]
    psutil.Process().cpu_affinity(affinity)
    write_json(output / "active" / f"slot{args.slot}.json", {
        "phase": "preparing", **worker_identity(), "dataset": args.dataset,
        "inner_fold": args.fold,
    })
    ctboost, _ = bootstrap_imports()
    if runtime_provenance() != execution["runtime"]:
        raise ValueError("Worker source, native extension or environment changed after freeze")
    import numpy as np
    from autogluon.core.metrics import get_metric
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, log_loss
    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame, _ctboost_eval_metric
    from benchmarks.tabarena.learning_options import resolve_cpu_learning_options
    from benchmarks.tabarena.pilot_evaluation import expected_pilot_jobs

    protocol, protocol_hash = read_protocol(args.protocol)
    if protocol_hash != execution["protocol_sha256"]:
        raise ValueError("Protocol changed after execution was frozen")
    dataset = next(row for row in protocol["datasets"] if row["dataset_name"] == args.dataset)
    directory = output / "data" / args.dataset
    prepared = json.loads((directory / "prepared.json").read_text())
    verify_prepared_inputs(execution, protocol_hash, prepared, directory / "outer_train.pkl", args.dataset)
    with (directory / "outer_train.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    split = payload["splits"][str(args.fold)]
    if indices_hash(split["train"], split["validation"]) != prepared["splits"][str(args.fold)]["sha256"]:
        raise ValueError("Inner split indices differ from the frozen manifest")
    raw_X_train = payload["X"].iloc[split["train"]].copy()
    raw_X_val = payload["X"].iloc[split["validation"]].copy()
    raw_y_train = payload["y"].iloc[split["train"]].copy()
    raw_y_val = payload["y"].iloc[split["validation"]].copy()
    del payload
    problem_type = dataset["problem_type"]
    if problem_type == "regression":
        y_train, y_val = np.asarray(raw_y_train, dtype=np.float64), np.asarray(raw_y_val, dtype=np.float64)
    else:
        encoder = LabelEncoder().fit(raw_y_train)
        y_train, y_val = encoder.transform(raw_y_train), encoder.transform(raw_y_val)
        if len(encoder.classes_) != dataset["num_classes"]:
            raise ValueError("Inner training fold lacks a declared class")
    jobs = [job for job in expected_pilot_jobs(protocol)
            if job["dataset"] == args.dataset and job["inner_fold"] == args.fold]
    shuffle_seed = int(json_hash({"seed": 159, "dataset": args.dataset, "fold": args.fold})[:16], 16)
    random.Random(shuffle_seed).shuffle(jobs)
    preprocessing_hash = json_hash({"data_sha256": prepared["data_sha256"],
                                   "split_sha256": split["sha256"],
                                   "adapter_sha256": execution["runtime"]["source_sha256"][
                                       "benchmarks/tabarena/ctboost_model.py"]})
    for job in jobs:
        destination = record_path(output, args.dataset, args.fold, job["variant"])
        if destination.exists():
            continue
        if args.controller_pid is not None:
            try:
                parent_controller = psutil.Process(args.controller_pid)
                if parent_controller.create_time() != args.controller_create_time:
                    raise RuntimeError("Original controller exited; no new fit will start")
            except psutil.NoSuchProcess as exc:
                raise RuntimeError("Original controller exited; no new fit will start") from exc
        record = {**job, "protocol_sha256": protocol_hash,
                  "split_sha256": split["sha256"], "preprocessing_sha256": preprocessing_hash,
                  "resource_contract_sha256": execution["resource_contract_sha256"],
                  "execution_runtime_sha256": json_hash(execution["runtime"]),
                  "affinity": affinity, "threads": resources["threads_per_worker"],
                  **worker_identity(), "started_at": now(), "status": "failed",
                  "deadline_stopped": False}
        active_path = output / "active" / f"slot{args.slot}.json"
        model = None
        write_json(active_path, {**record, "phase": "fit", "fit_started_epoch": time.time(),
                                "fit_started_monotonic": time.monotonic(),
                                "result_path": str(destination)})
        fit_start = time.perf_counter()
        deadline = time.monotonic() + resources["fit_time_limit_seconds"]
        memory = MemorySampler()
        memory.__enter__()
        try:
            X_train, categoricals = normalize_tabarena_frame(raw_X_train)
            X_val, _ = normalize_tabarena_frame(raw_X_val, categorical_columns=categoricals)
            params = dict(protocol["base_params"])
            patience = params.pop("early_stopping_rounds")
            params.update(random_seed=job["seed"], cat_features=categoricals or None,
                          eval_metric=_ctboost_eval_metric(problem_type, dataset["eval_metric"]))
            params, applicability = resolve_cpu_learning_options(
                params, problem_type=problem_type, labels=y_train,
                num_classes=None if problem_type == "regression" else dataset["num_classes"],
                variant=job["variant"], on_inapplicable="skip")
            if params is None:
                raise ValueError(f"Registered pilot arm is not applicable: {applicability}")
            record["parameters"] = params
            record["applicability"] = applicability
            estimator = ctboost.CTBoostRegressor if problem_type == "regression" else ctboost.CTBoostClassifier
            model = estimator(**params)
            training_start = time.monotonic()

            def stop_callback(env):
                current = time.monotonic()
                iterations = max(1, int(env.iteration) - int(env.begin_iteration) + 1)
                stop = current + 2 * (current - training_start) / iterations >= deadline
                if stop:
                    record["deadline_stopped"] = True
                return stop

            model.fit(X_train, y_train, eval_set=(X_val, y_val),
                      early_stopping_rounds=patience, callbacks=[stop_callback])
            record["fit_seconds"] = time.perf_counter() - fit_start
            memory.__exit__()
            record["peak_rss_bytes"] = memory.peak
            # Stop the fit watchdog before doing final validation prediction.
            write_json(active_path, {**record, "phase": "predict", "result_path": str(destination)})
            prediction_start = time.perf_counter()
            predictions = model.predict(X_val) if problem_type == "regression" else model.predict_proba(X_val)
            if problem_type == "binary":
                predictions = predictions[:, 1]
            record["predict_seconds"] = time.perf_counter() - prediction_start
            metric = get_metric(dataset["eval_metric"], problem_type=problem_type)
            error = float(metric.error(y_val, predictions))
            if not math.isfinite(error) or not np.all(np.isfinite(predictions)):
                raise ValueError("Nonfinite validation predictions or metric")
            prediction_path = destination.with_suffix(".npz")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(prediction_path, predictions=predictions, labels=y_val,
                                validation_indices=split["validation"])
            secondary = {}
            if problem_type == "binary":
                secondary["binary_log_loss"] = float(log_loss(y_val, predictions, labels=[0, 1]))
            elif problem_type == "multiclass":
                secondary["multiclass_accuracy"] = float(accuracy_score(y_val, predictions.argmax(axis=1)))
            booster = model.get_booster()
            rounds = int(booster.num_iterations_trained)
            trees = rounds * (1 if booster.multi_strategy == "multi_output_tree" else int(booster.prediction_dimension))
            record.update(status="ok", metric_error=error, metric=dataset["eval_metric"],
                          prediction_sha256=file_hash(prediction_path),
                          best_iteration=int(model.best_iteration_), rows_train=len(y_train),
                          rows_validation=len(y_val), retained_boosting_rounds=rounds,
                          retained_tree_count=trees, secondary_metrics=secondary)
        except Exception as exc:
            record.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                          fit_seconds=time.perf_counter() - fit_start)
        finally:
            memory.__exit__()
            record.setdefault("peak_rss_bytes", memory.peak)
        record["finished_at"] = now()
        write_json(destination, record)
        write_json(active_path, {"phase": "between_fits", **worker_identity(),
                                 "dataset": args.dataset, "inner_fold": args.fold})
        print(json.dumps({key: record.get(key) for key in ("job_id", "status", "metric_error", "fit_seconds")}), flush=True)
        del model
        gc.collect()
    return 0


@contextmanager
def controller_lock(output):
    path = output / "controller.lock"
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
        yield


def load_records(output):
    return [json.loads(path.read_text()) for path in sorted((output / "fits").glob("*/inner*/*.json"))]


def process_tree_rss(process):
    import psutil

    try:
        parent = psutil.Process(process.pid)
        members = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    rss = 0
    for member in members:
        try:
            rss += member.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return rss


def kill_worker(process):
    """Terminate only this owned worker tree, including Windows venv redirects."""
    import psutil

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
        _, alive = psutil.wait_procs([*children, parent], timeout=15)
        if alive:
            raise RuntimeError("Owned worker processes did not terminate; refusing further dispatch")
    except psutil.NoSuchProcess:
        pass
    if isinstance(process, subprocess.Popen):
        process.wait(timeout=15)


def matching_owned_worker(state, output):
    """Return only an exact journaled process; never kill a recycled user PID."""
    import psutil

    pid, created = state.get("pid"), state.get("process_create_time")
    if not isinstance(pid, int):
        return None
    if not isinstance(created, (int, float)):
        if psutil.pid_exists(pid):
            raise RuntimeError("Live journal PID lacks creation-time provenance; no process was killed")
        return None
    try:
        process = psutil.Process(pid)
        if process.create_time() != created:
            return None
        command = process.cmdline()
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as exc:
        raise RuntimeError("Cannot verify journaled worker ownership; no process was killed") from exc
    script = os.path.normcase(str(Path(__file__).resolve()))
    output_path = os.path.normcase(str(Path(output).resolve()))
    script_indices = [i for i, value in enumerate(command)
                      if os.path.normcase(str(Path(value).resolve())) == script]
    owned = False
    for index in script_indices:
        arguments = command[index + 1:]
        if not arguments or arguments[0] != "worker" or "--output" not in arguments:
            continue
        location = arguments.index("--output") + 1
        if location < len(arguments):
            owned = os.path.normcase(str(Path(arguments[location]).resolve())) == output_path
        if owned:
            break
    if not owned:
        raise RuntimeError("Journal PID matches but command/output ownership does not; no process was killed")
    return process


def preserve_interrupted_fit(state, output, protocol, protocol_hash, reason):
    """Account for a started attempt once; completed results are immutable."""
    from benchmarks.tabarena.pilot_evaluation import expected_pilot_jobs

    if not state.get("result_path"):
        return None
    identity = (state.get("dataset"), state.get("variant"), state.get("inner_fold"))
    expected = {(job["dataset"], job["variant"], job["inner_fold"]): job
                for job in expected_pilot_jobs(protocol)}
    if identity not in expected or state.get("protocol_sha256") != protocol_hash:
        raise ValueError("Interrupted attempt has a foreign job/protocol identity")
    job = expected[identity]
    destination = record_path(output, job["dataset"], job["inner_fold"], job["variant"]).resolve()
    if destination != Path(state["result_path"]).resolve() or not destination.is_relative_to((output / "fits").resolve()):
        raise ValueError("Interrupted result path is outside its expected job location")
    if destination.exists():
        return destination
    excluded = {"result_path", "fit_started_epoch", "fit_started_monotonic", "phase"}
    failed = {key: value for key, value in state.items() if key not in excluded}
    failed.update(status="failed", error=reason, finished_at=now(), interrupted=True)
    write_json(destination, failed)
    return destination


def reconcile_active_journals(output, protocol, protocol_hash, reason="interrupted_controller"):
    """Recover attempts before dispatch, preserving evidence and live-PID checks."""
    for path in sorted((output / "active").glob("slot*.json")):
        state = json.loads(path.read_text())
        if state.get("phase") in {"idle", "reconciled"}:
            continue
        process = matching_owned_worker(state, output)
        if process is not None:
            kill_worker(process)
            # The process may have advanced from startup/between-fits to a fit
            # while its identity was checked. Read its last atomic journal.
            state = json.loads(path.read_text())
        archive = output / "journal" / f"{path.stem}-{time.time_ns()}.json"
        write_json(archive, {"reconciled_at": now(), "reason": reason, "last_state": state})
        preserve_interrupted_fit(state, output, protocol, protocol_hash, reason)
        write_json(path, {"phase": "reconciled", "archive": str(archive), "reconciled_at": now()})


def run(args):
    bootstrap_imports()
    import psutil
    from benchmarks.tabarena.local_resources import worker_environment
    from benchmarks.tabarena.pilot_evaluation import expected_pilot_jobs, evaluate_pilot

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol, protocol_hash = read_protocol(args.protocol)
    calibration = json.loads(args.resources.read_text())
    with controller_lock(output):
        execution_path = output / "execution.json"
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            if execution["protocol_sha256"] != protocol_hash or execution["runtime"] != runtime_provenance():
                raise ValueError("Protocol or runner environment changed; use a new registered run")
        else:
            execution = build_execution_manifest(protocol, protocol_hash, calibration, output)
            write_json(execution_path, execution)
        resources = execution["resources"]
        jobs = expected_pilot_jobs(protocol)
        reconcile_active_journals(output, protocol, protocol_hash)
        existing_report = evaluate_pilot(protocol, load_records(output), protocol_sha256=protocol_hash)
        existing_errors = [error for error in existing_report["global_errors"] if error != "missing_expected_jobs"]
        if existing_errors:
            raise ValueError(f"Existing pilot records fail integrity checks: {existing_errors}")
        blocks = list(dict.fromkeys((job["dataset"], job["inner_fold"]) for job in jobs))
        random.Random(159).shuffle(blocks)
        queue = [(dataset, fold) for dataset, fold in blocks if any(
            not record_path(output, dataset, fold, job["variant"]).exists()
            for job in jobs if job["dataset"] == dataset and job["inner_fold"] == fold)]
        active = {}
        (output / "logs").mkdir(exist_ok=True)
        (output / "active").mkdir(exist_ok=True)
        for slot in range(resources["workers"]):
            write_json(output / "active" / f"slot{slot}.json", {"phase": "idle"})
        print(f"Running {len(queue)} fold blocks with {resources['workers']} workers x "
              f"{resources['threads_per_worker']} threads", flush=True)
        try:
            while queue or active:
                for slot in range(resources["workers"]):
                    if slot in active or not queue or psutil.virtual_memory().available < resources["minimum_free_bytes"] + GIB:
                        continue
                    dataset, fold = queue.pop(0)
                    write_json(output / "active" / f"slot{slot}.json", {"phase": "idle"})
                    log = (output / "logs" / f"{dataset}-inner{fold}.log").open("a", encoding="utf-8")
                    command = [sys.executable, "-I", str(Path(__file__).resolve()), "worker",
                               "--protocol", str(args.protocol.resolve()), "--output", str(output),
                               "--dataset", dataset, "--fold", str(fold), "--slot", str(slot),
                               "--controller-pid", str(os.getpid()), "--controller-create-time",
                               str(psutil.Process().create_time())]
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=worker_environment(resources["threads_per_worker"]),
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    active[slot] = {"process": process, "log": log, "dataset": dataset, "fold": fold,
                                    "started": time.monotonic()}
                    state_path = output / "active" / f"slot{slot}.json"
                    if json.loads(state_path.read_text()).get("phase") == "idle":
                        try:
                            write_json(state_path, {"phase": "startup", "pid": process.pid,
                                "process_create_time": psutil.Process(process.pid).create_time(),
                                "dataset": dataset, "inner_fold": fold})
                        except psutil.NoSuchProcess:
                            pass
                for slot, item in list(active.items()):
                    process = item["process"]
                    state_path = output / "active" / f"slot{slot}.json"
                    state = json.loads(state_path.read_text())
                    reason = None
                    if process.poll() is None:
                        rss = process_tree_rss(process)
                        if rss > resources["memory_per_worker_bytes"]:
                            reason = "resource_limit"
                        elif state.get("fit_started_monotonic") and time.monotonic() - state["fit_started_monotonic"] > (
                            resources["fit_time_limit_seconds"] + resources["hard_fit_grace_seconds"]):
                            reason = "timeout"
                        elif state.get("phase") in {"idle", "startup", "preparing"} and time.monotonic() - item["started"] > 120:
                            reason = "startup_failure"
                        if reason:
                            kill_worker(process)
                    if process.poll() is not None:
                        item["log"].close()
                        if reason and state.get("result_path") and not Path(state["result_path"]).exists():
                            failed = {key: value for key, value in state.items()
                                      if key not in {"result_path", "fit_started_epoch", "fit_started_monotonic", "phase"}}
                            failed.update(status=reason, error=reason, finished_at=now())
                            write_json(Path(state["result_path"]), failed)
                        remaining = [job for job in jobs if job["dataset"] == item["dataset"]
                                     and job["inner_fold"] == item["fold"] and not record_path(
                                         output, job["dataset"], job["inner_fold"], job["variant"]).exists()]
                        if remaining:
                            # Do not selectively retry model failures. Account for every unrun arm.
                            prepared = json.loads((output / "data" / item["dataset"] / "prepared.json").read_text())
                            for job in remaining:
                                failed = {**job, "status": "failed", "error": "worker exited before this fit",
                                          "protocol_sha256": protocol_hash,
                                          "resource_contract_sha256": execution["resource_contract_sha256"],
                                          "split_sha256": prepared["splits"][str(item["fold"])]["sha256"],
                                          "preprocessing_sha256": json_hash({"data_sha256": prepared["data_sha256"],
                                              "split_sha256": prepared["splits"][str(item["fold"])]["sha256"],
                                              "adapter_sha256": execution["runtime"]["source_sha256"][
                                                  "benchmarks/tabarena/ctboost_model.py"]}),
                                          "deadline_stopped": False, "finished_at": now()}
                                write_json(record_path(output, job["dataset"], job["inner_fold"], job["variant"]), failed)
                        del active[slot]
                records = load_records(output)
                write_json(output / "progress.json", {"updated_at": now(), "pid": os.getpid(),
                    "expected_fits": len(jobs), "recorded_fits": len(records),
                    "successful_fits": sum(row["status"] == "ok" for row in records),
                    "active_blocks": [{"slot": slot, "pid": item["process"].pid,
                                       "dataset": item["dataset"], "inner_fold": item["fold"]}
                                      for slot, item in active.items()],
                    "queued_blocks": len(queue), "free_memory_bytes": psutil.virtual_memory().available})
                time.sleep(0.25)
        finally:
            for item in active.values():
                if item["process"].poll() is None:
                    kill_worker(item["process"])
                item["log"].close()
            reconcile_active_journals(output, protocol, protocol_hash, reason="controller_stopped")
        report = evaluate_pilot(protocol, load_records(output), protocol_sha256=protocol_hash)
        write_json(output / "decision.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/tabarena/pilot_0159_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--openml-cache", type=Path, default=Path.home() / ".openml")
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--controller-create-time", type=float)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "worker":
        run_block(args)
    elif args.stage == "run":
        if args.resources is None:
            parser.error("--resources is required for run")
        run(args)
    else:
        bootstrap_imports()
        from benchmarks.tabarena.pilot_evaluation import evaluate_pilot
        protocol, protocol_hash = read_protocol(args.protocol)
        report = evaluate_pilot(protocol, load_records(args.output), protocol_sha256=protocol_hash)
        write_json(args.output / "decision.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
