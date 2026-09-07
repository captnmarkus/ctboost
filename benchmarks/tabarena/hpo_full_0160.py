"""Public CTBoost 0.1.60: default plus 25 HPO configurations on all 816 outer splits.

Uses author-run CPU resources. Each parent is fresh, publicly preregistered,
and owns one exact official repeat/fold; started failures are never retried.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import os
import pickle
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.60"
PORTFOLIO_ID = "ctboost_0160_full_baseline_hpo25_v1"
PORTFOLIO_25_SHA256 = "210f35c95c458f888f83dbb785ff56d25fa56a1ed554299b43fe0a95c41fc4c0"
TABARENA_COMMIT = "31026f7d758390994353eba79fbfa6747616f365"
METADATA_PATH = "packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/TabArena-v0.1_tasks_metadata.csv"
METADATA_SHA256 = "02f35e19dead7e3795f65e91eb00fdb7fa255896ed6bde0f29b2ce0af7a46296"
FULL_SPLITS_SHA256 = "3434075982bc6dd829704a30958de2c6583afa214976d3882d0276146bc7487c"
RESOURCES = {
    "num_cpus": 2,
    "num_gpus": 0,
    "memory_limit_gb": 8.0,
    "time_limit_seconds": 3600,
    "parent_wall_limit_seconds": 4500,
    "num_bag_folds": 8,
    "num_bag_sets": 1,
    "fold_fitting_strategy": "sequential_local",
    "blas_openmp_threads": 1,
}
SOURCE_FILES = (
    "benchmarks/__init__.py",
    "benchmarks/tabarena/__init__.py",
    "benchmarks/tabarena/hpo_0160.py",
    "benchmarks/tabarena/hpo_full_0160.py",
    "benchmarks/tabarena/local_hpo.py",
    "benchmarks/tabarena/kaggle_hpo_worker.py",
    "benchmarks/tabarena/ctboost_model.py",
    "benchmarks/tabarena/learning_options.py",
)


def _base():
    from benchmarks.tabarena import hpo_0160

    return hpo_0160


def _shared():
    return _base()._shared()


def read_json(path):
    return _base().read_json(path)


def plan_hash(value):
    return _base().plan_hash(value)


def source_hash(path):
    return _base().source_hash(path)


def verify_registration(plan, registration_path, output):
    return _base().verify_registration(plan, registration_path, output)


def build_experiments():
    # The original full 200 generator defines the identity/seeds of this prefix.
    experiments, configs = _base().build_experiments()
    if (
        RESOURCES != _base().RESOURCES
        or plan_hash(configs[1:26]) != PORTFOLIO_25_SHA256
    ):
        raise ValueError("Full-run resources or frozen 25-config prefix changed")
    return experiments[:26], configs[:26]


def load_population(path):
    """All actual metadata rows; repeat/fold membership is never extrapolated."""
    if source_hash(path) != METADATA_SHA256:
        raise ValueError("Pinned official full metadata changed")
    inventory = {row["dataset"]: row for row in _base().load_population(path)}
    rows = csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8")))
    population = []
    for row in rows:
        entry = dict(inventory[row["dataset_name"]])
        entry.update(
            repeat=int(row["repeat"]),
            fold=int(row["fold"]),
            num_instances_train=float(row["num_instances_train"]),
            num_instances_test=float(row["num_instances_test"]),
        )
        if (
            entry["task_id"] != int(row["task_id_str"])
            or entry["problem_type"] != row["problem_type"]
        ):
            raise ValueError("Task identity changed across outer splits")
        population.append(entry)
    population.sort(key=lambda row: (row["dataset"], row["repeat"], row["fold"]))
    if (
        len(population) != 816
        or len({(p["task_id"], p["repeat"], p["fold"]) for p in population}) != 816
    ):
        raise ValueError("Expected exactly the 816 unique official outer splits")
    return population


def make_parents(population, names):
    parents = []
    for index, name in enumerate(names):
        for row in sorted(
            population, key=lambda p: (p["repeat"], p["fold"], p["dataset"])
        ):
            ordinal = len(parents)
            parents.append(
                {
                    "parent_id": f"openml-{row['task_id']}-r{row['repeat']}-f{row['fold']}-c{index:03d}",
                    "ordinal": ordinal,
                    "owner": "local" if ordinal % 9 < 4 else "kaggle",
                    **{
                        key: row[key]
                        for key in (
                            "dataset",
                            "task_id",
                            "problem_type",
                            "repeat",
                            "fold",
                            "num_instances_train",
                            "num_instances_test",
                        )
                    },
                    "config_index": index,
                    "config_name": name,
                    "child_seeds": list(range(index * 8, index * 8 + 8)),
                }
            )
    return parents


def validate_plan(plan, *, verify_sources=True):
    base = _base()
    required = {
        "schema_version": 1,
        "portfolio_id": PORTFOLIO_ID,
        "ctboost_version": VERSION,
        "tabarena_commit": TABARENA_COMMIT,
        "metadata_sha256": METADATA_SHA256,
        "portfolio_25_sha256": PORTFOLIO_25_SHA256,
        "package_pins": base.PINS,
        "resources": RESOURCES,
        "baseline_controls": base.BASELINE_CONTROLS,
        "expected_outer_split_count": 816,
        "expected_parent_count": 21216,
        "expected_child_count": 169728,
        "uses_test_outcomes_for_selection": False,
        "authorization": "explicit_user_request_for_full_tabarena_default_plus_25",
    }
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError("Full plan identity, resources, or baseline protocol changed")
    population = plan.get("population", [])
    splits = plan.get("outer_splits", [])
    if len(population) != 51 or len({p["task_id"] for p in population}) != 51:
        raise ValueError("Missing full suite dataset inventory")
    if (
        len(splits) != 816
        or len({(p["task_id"], p["repeat"], p["fold"]) for p in splits}) != 816
    ):
        raise ValueError("Missing or duplicated full outer splits")
    if (
        plan.get("outer_splits_sha256") != FULL_SPLITS_SHA256
        or plan_hash(splits) != FULL_SPLITS_SHA256
    ):
        raise ValueError("Full outer-split metadata digest mismatch")
    configs = plan.get("configurations", [])
    if (
        len(configs) != 26
        or configs[0] != {}
        or plan_hash(configs[1:]) != PORTFOLIO_25_SHA256
    ):
        raise ValueError(
            "Full plan must contain exactly the frozen default plus25 prefix"
        )
    if plan.get("configurations_sha256") != plan_hash(configs):
        raise ValueError("Full configuration digest mismatch")
    names = [f"CTBoost_{'c1' if i == 0 else f'r{i}'}_default_BAG_L1" for i in range(26)]
    if plan.get("parents") != make_parents(splits, names):
        raise ValueError(
            "Full parent identities, outer splits, ownership, or seeds changed"
        )
    if len(plan.get("experiments", [])) != 26:
        raise ValueError("Missing full-run experiment definitions")
    if set(plan.get("source_sha256", {})) != set(SOURCE_FILES):
        raise ValueError("Incomplete full-worker source provenance")
    if verify_sources and plan["source_sha256"] != {
        name: source_hash(ROOT / name) for name in SOURCE_FILES
    }:
        raise ValueError("Full worker source differs from the public plan")
    catalog = plan.get("public_wheels", [])
    release = {
        "info": {"version": VERSION},
        "urls": [
            {
                **entry,
                "digests": {"sha256": entry["sha256"]},
                "packagetype": "bdist_wheel",
                "yanked": False,
            }
            for entry in catalog
        ],
    }
    if (
        base.public_wheels(release) != catalog
        or plan.get("public_release_url") != base.PYPI_URL
    ):
        raise ValueError("Full plan public-wheel catalog changed")
    return plan


def build_plan(*, output, metadata_path, release_manifest=None):
    base = _base()
    release = base._fetch_json(base.PYPI_URL)
    if release_manifest is not None and base.public_wheels(
        read_json(release_manifest)
    ) != base.public_wheels(release):
        raise ValueError("Supplied manifest differs from public0.1.60 wheels")
    experiments, configs = build_experiments()
    splits = load_population(metadata_path)
    path = Path(output).resolve() / "plan.json"
    previous = read_json(path) if path.exists() else None
    plan = {
        "schema_version": 1,
        "portfolio_id": PORTFOLIO_ID,
        "ctboost_version": VERSION,
        "created_at_utc": previous["created_at_utc"] if previous else _shared().now(),
        "authorization": "explicit_user_request_for_full_tabarena_default_plus_25",
        "uses_test_outcomes_for_selection": False,
        "selection_policy": "Fixed default plus25 configurations on every official outer split. OOF validation alone selects configurations/ensemble weights; outer tests are final evidence only.",
        "tabarena_repository": "https://github.com/captnmarkus/tabarena",
        "tabarena_commit": TABARENA_COMMIT,
        "metadata_path": METADATA_PATH,
        "metadata_sha256": METADATA_SHA256,
        "portfolio_25_sha256": PORTFOLIO_25_SHA256,
        "package_pins": dict(base.PINS),
        "python_major_minor": "3.12",
        "resources": dict(RESOURCES),
        "baseline_controls": dict(base.BASELINE_CONTROLS),
        "public_release_url": base.PYPI_URL,
        "public_wheels": base.public_wheels(release),
        "population": base.load_population(metadata_path),
        "outer_splits": splits,
        "outer_splits_sha256": plan_hash(splits),
        "configurations": configs,
        "configurations_sha256": plan_hash(configs),
        "experiments": [item.to_yaml_dict() for item in experiments],
        "parents": make_parents(splits, [item.name for item in experiments]),
        "source_sha256": {name: source_hash(ROOT / name) for name in SOURCE_FILES},
        "expected_outer_split_count": 816,
        "expected_parent_count": 21216,
        "expected_child_count": 169728,
        "seed_policy": "config_index*8+inner_bag_fold; no outer-repeat/fold offset; feature shuffle disabled",
        "timing_disclosure": "Author-run heterogeneous2CPU/8GiB timings, not canonical TabArena8CPU/32GB timings.",
        "failure_policy": "Fresh parents only. No implicit Lite artifact reuse or automatic retries of started failures. Resource revisions require separate public protocols.",
    }
    validate_plan(plan)
    if previous is not None and previous != plan:
        raise ValueError("Existing immutable full plan differs")
    if not path.exists():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("Full plan requires a fresh empty directory")
        _shared().write_json(path, plan)
    return plan


create_plan = build_plan


def runtime_provenance(plan, wheel_path):
    import tabarena

    # The51-dataset inventory and all native/package/wheel/source checks are retained.
    runtime = _base().runtime_provenance(plan, wheel_path)
    source = next(
        path
        for path in Path(tabarena.__file__).resolve().parents
        if (path / ".git").exists()
    )
    if load_population(source / METADATA_PATH) != plan["outer_splits"]:
        raise ValueError(
            "Actual official full outer-split metadata differs from the plan"
        )
    return {**runtime, "outer_splits_sha256": plan["outer_splits_sha256"]}


def _checked_experiments(plan):
    experiments, configs = build_experiments()
    if (
        configs != plan["configurations"]
        or [e.to_yaml_dict() for e in experiments] != plan["experiments"]
    ):
        raise ValueError("Actual pinned full experiment definitions changed")
    return experiments


def preflight(*, plan_path, output, host, wheel_path, registration_path):
    if host not in {"local", "kaggle"}:
        raise ValueError("Unknown full execution host")
    plan = validate_plan(read_json(plan_path))
    registration = verify_registration(plan, registration_path, output)
    runtime = runtime_provenance(plan, wheel_path)
    experiments = _checked_experiments(plan)
    evidence = {
        "schema_version": 1,
        "status": "verified",
        "fit_started": False,
        "host": host,
        "plan_sha256": plan_hash(plan),
        "registration": registration,
        "runtime": runtime,
        "runtime_sha256": plan_hash(runtime),
        "experiment_count": len(experiments),
        "outer_split_count": 816,
        "verified_at_utc": _shared().now(),
    }
    _shared().write_json(Path(output) / f"preflight-{host}.json", evidence)
    return evidence


def _outer_split_receipt(task, parent):
    """Exact ordered official indices; CSV row counts are only dataset averages."""
    import numpy as np

    train, test = task.get_split_indices(
        fold=parent["fold"], repeat=parent["repeat"], sample=0
    )
    arrays = [np.asarray(part) for part in (train, test)]
    if any(
        a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer)
        for a in arrays
    ):
        raise ValueError("Official split must contain nonempty integer index arrays")
    total = sum(a.size for a in arrays)
    if (
        not np.array_equal(np.sort(np.concatenate(arrays)), np.arange(total))
        or getattr(task, "_n_rows", total) != total
    ):
        raise ValueError(
            "Official train/test indices must partition the dataset exactly once"
        )
    receipt = {
        "task_id": parent["task_id"],
        "repeat": parent["repeat"],
        "fold": parent["fold"],
        "sample": 0,
        "train_rows": int(arrays[0].size),
        "test_rows": int(arrays[1].size),
        "train_indices_sha256": hashlib.sha256(
            np.asarray(arrays[0], dtype="<i8").tobytes()
        ).hexdigest(),
        "test_indices_sha256": hashlib.sha256(
            np.asarray(arrays[1], dtype="<i8").tobytes()
        ).hexdigest(),
    }
    return {**receipt, "sha256": plan_hash(receipt)}


def _validate_split_receipt(receipt, parent):
    fields = {
        "task_id",
        "repeat",
        "fold",
        "sample",
        "train_rows",
        "test_rows",
        "train_indices_sha256",
        "test_indices_sha256",
        "sha256",
    }
    if (
        set(receipt) != fields
        or any(receipt.get(k) != parent[k] for k in ("task_id", "repeat", "fold"))
        or receipt["sample"] != 0
    ):
        raise ValueError("Outer-split receipt belongs to different official indices")
    if any(
        type(receipt[k]) is not int or receipt[k] <= 0
        for k in ("train_rows", "test_rows")
    ):
        raise ValueError("Missing exact official outer-split counts")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", receipt[k])
        for k in ("train_indices_sha256", "test_indices_sha256")
    ) or receipt["sha256"] != plan_hash(
        {k: v for k, v in receipt.items() if k != "sha256"}
    ):
        raise ValueError("Outer-split index receipt digest changed")
    return receipt


def _audited_runner(parent, digest, runtime_digest, outer_split):
    from tabarena.benchmark.experiment import OOFExperimentRunner

    class FullRunner(OOFExperimentRunner):
        def post_evaluate(self, out):
            out = super().post_evaluate(out)
            if (self.repeat, self.fold, self.sample) != (
                parent["repeat"],
                parent["fold"],
                0,
            ):
                raise ValueError("Actual runner used a different official outer split")
            if _outer_split_receipt(self.task, parent) != outer_split:
                raise ValueError(
                    "Actual runner indices differ from its pre-fit official receipt"
                )
            out["hpo_full_0160"] = {
                "ctboost_version": VERSION,
                "portfolio_id": PORTFOLIO_ID,
                "plan_sha256": digest,
                "runtime_sha256": runtime_digest,
                "parent_id": parent["parent_id"],
                "owner": parent["owner"],
                "repeat": self.repeat,
                "fold": self.fold,
                "sample": self.sample,
                "outer_split": outer_split,
                "children": _shared().audit_children(self.model, parent),
            }
            return out

        def run(self):
            try:
                return super().run()
            except Exception:
                if getattr(self, "model", None) is not None:
                    try:
                        self._cleanup()
                    except Exception:  # noqa: BLE001, S110 -- preserve original fit error
                        pass
                raise

    return FullRunner


def validate_parent_result(path, parent, plan_sha256, *, outer_split=None):
    """Validate this worker's own pickle, including true outer repeat/fold identity."""
    import numpy as np

    path = Path(path)
    with gzip.open(path, "rb") as stream:
        result = pickle.load(stream)
    for record in (result, result.get("experiment_metadata", {})):
        if (
            not isinstance(record, dict)
            or record.get("success") is False
            or any(
                record.get(k)
                for k in (
                    "exception",
                    "error",
                    "traceback",
                    "failure",
                    "failure_artifact",
                )
            )
        ):
            raise ValueError("Full result contains failure evidence")
    if (
        result.get("framework") != parent["config_name"]
        or result.get("problem_type") != parent["problem_type"]
    ):
        raise ValueError("Full result configuration/problem identity changed")
    task = result.get("task_metadata", {})
    if any(
        task.get(k) != parent[p]
        for k, p in (("name", "dataset"), ("repeat", "repeat"), ("fold", "fold"))
    ) or str(task.get("tid")) != str(parent["task_id"]):
        raise ValueError("Full result task or outer repeat/fold changed")
    if (
        path.parent.name != f"{parent['repeat']}_{parent['fold']}"
        or path.parent.parent.name != str(parent["task_id"])
        or path.parent.parent.parent.name != parent["config_name"]
    ):
        raise ValueError("Full result path differs from its actual outer split")
    for key in ("metric_error", "metric_error_val", "time_train_s", "time_infer_s"):
        value = np.asarray(result.get(key))
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value)
        ):
            raise ValueError("Full result has missing/nonfinite metrics or timings")
        if key.startswith("time_") and value < 0:
            raise ValueError("Full result has negative timings")
    audit = result.get("hpo_full_0160", {})
    expected = {
        "ctboost_version": VERSION,
        "portfolio_id": PORTFOLIO_ID,
        "plan_sha256": plan_sha256,
        "parent_id": parent["parent_id"],
        "owner": parent["owner"],
        "repeat": parent["repeat"],
        "fold": parent["fold"],
        "sample": 0,
    }
    if any(audit.get(k) != v for k, v in expected.items()) or not re.fullmatch(
        r"[0-9a-f]{64}", audit.get("runtime_sha256", "")
    ):
        raise ValueError("Missing matching full-run runtime/outer-split provenance")
    receipt = _validate_split_receipt(audit.get("outer_split", {}), parent)
    if outer_split is not None and receipt != outer_split:
        raise ValueError("Raw outer-split receipt differs from the pre-fit manifest")
    info = result.get("method_metadata", {}).get("info", {})
    bag = info.get("bagged_info", {})
    children = info.get("children_info", {})
    names = {f"S1F{i}" for i in range(1, 9)}
    if (
        bag.get("num_child_models") != 8
        or set(children) != names
        or set(bag.get("child_model_names", [])) != names
        or set(audit.get("children", {})) != names
    ):
        raise ValueError("Full result must retain all eight bag children")
    for child in [info, *children.values()]:
        if not all(
            child.get(flag) is True for flag in ("is_fit", "is_valid", "can_infer")
        ):
            raise ValueError(
                "A full-run parent/child is not fitted, valid, and inferable"
            )
    for fold in range(8):
        name = f"S1F{fold + 1}"
        actual = audit["children"][name]
        seed = parent["config_index"] * 8 + fold
        if (
            children[name].get("hyperparameters", {}).get("random_seed") != seed
            or actual.get("random_seed") != seed
            or actual.get("native_controls") != _base().BASELINE_CONTROLS
            or actual.get("task_type") != "CPU"
            or actual.get("feature_test") != "quadratic"
        ):
            raise ValueError(
                "Full child seed, baseline learning controls, or CPU identity changed"
            )
    simulation = result.get("simulation_artifacts", {})
    problem = parent["problem_type"]
    for split, rows in (
        ("val", receipt["train_rows"]),
        ("test", receipt["test_rows"]),
    ):
        predictions = simulation.get(f"pred_proba_dict_{split}", {})
        if set(predictions) != {parent["config_name"]}:
            raise ValueError("Missing or additional full parent predictions")
        target = np.asarray(simulation.get(f"y_{split}"))
        pred = np.asarray(predictions[parent["config_name"]])
        shape = (
            (rows, int(simulation["num_classes"]))
            if problem == "multiclass"
            else (rows,)
        )
        if (
            target.shape != (rows,)
            or not np.isfinite(target).all()
            or pred.shape != shape
            or not np.isfinite(pred).all()
        ):
            raise ValueError(
                "Full target/prediction rows differ from official outer-split metadata"
            )
        if problem != "regression" and (
            np.any(pred < -1e-6) or np.any(pred > 1 + 1e-6)
        ):
            raise ValueError("Invalid full probabilities")
        if problem == "multiclass" and not np.allclose(
            pred.sum(axis=1), 1, rtol=0, atol=1e-5
        ):
            raise ValueError("Full multiclass probability rows do not sum to one")
    bag = simulation.get("bag_info", {})
    indices = bag.get("val_idx_per_child", [])
    predictions = bag.get("pred_proba_test_per_child", [])
    if len(indices) != 8 or len(predictions) != 8:
        raise ValueError("Missing full bag-fold simulation artifacts")
    arrays = [np.asarray(index) for index in indices]
    if any(
        a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) for a in arrays
    ) or not np.array_equal(
        np.sort(np.concatenate(arrays)), np.arange(receipt["train_rows"])
    ):
        raise ValueError("Full OOF folds must cover outer-training rows exactly once")
    ensemble = np.asarray(simulation["pred_proba_dict_test"][parent["config_name"]])
    if any(
        np.asarray(p).shape != ensemble.shape or not np.isfinite(p).all()
        for p in predictions
    ):
        raise ValueError("Invalid full child test predictions")
    predictions = np.asarray(predictions)
    if not np.allclose(predictions.mean(axis=0), ensemble, rtol=1e-5, atol=1e-6):
        raise ValueError("Full parent predictions are not the eight-child mean")
    if problem != "regression" and (
        np.any(predictions < -1e-6) or np.any(predictions > 1 + 1e-6)
    ):
        raise ValueError("Invalid full child probabilities")
    if problem == "multiclass" and not np.allclose(
        predictions.sum(axis=2), 1, rtol=0, atol=1e-5
    ):
        raise ValueError("Full child probability rows do not sum to one")
    return {
        "dataset": parent["dataset"],
        "task_id": str(parent["task_id"]),
        "repeat": parent["repeat"],
        "fold": parent["fold"],
        "bag_children": 8,
        "raw_sha256": _shared().file_hash(path),
        "size_bytes": path.stat().st_size,
        "runtime_sha256": audit["runtime_sha256"],
        "native_children": audit["children"],
        "outer_split": receipt,
    }


def _load_task(parent, output):
    # This existing loader selects the complete task, not a particular outer split.
    spec, task = _shared()._load_task(parent, output)
    repeats, folds, samples = task.get_split_dimensions()
    if not (
        0 <= parent["repeat"] < repeats and 0 <= parent["fold"] < folds and samples >= 1
    ):
        raise ValueError(
            "Official task does not provide the requested full outer split"
        )
    return spec, task


def run_parent(
    *, plan_path, output, parent_id, host, wheel_path, registration_path, affinity=None
):
    import psutil

    plan = validate_plan(read_json(plan_path))
    digest = plan_hash(plan)
    matches = [p for p in plan["parents"] if p["parent_id"] == parent_id]
    if len(matches) != 1 or host != matches[0]["owner"]:
        raise ValueError("Full parent does not belong to this host")
    parent = matches[0]
    affinity = _shared()._configure_process(
        2, affinity if affinity is not None else psutil.Process().cpu_affinity()[:2]
    )
    output = Path(output).resolve()
    registration = verify_registration(plan, registration_path, output)
    runtime = runtime_provenance(plan, wheel_path)
    experiments = _checked_experiments(plan)
    key = (
        Path(parent["config_name"])
        / str(parent["task_id"])
        / f"{parent['repeat']}_{parent['fold']}"
    )
    directory, raw = output / "artifacts" / key, output / "data" / key / "results.pkl"
    manifest_path = directory / "manifest.json"
    with _shared()._parent_lock(directory / "worker.lock"):
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if (
                previous.get("parent") != parent
                or previous.get("plan_sha256") != digest
                or previous.get("host") != host
                or previous.get("resources") != RESOURCES
                or previous.get("registration") != registration
                or previous.get("runtime") != runtime
            ):
                raise ValueError("Existing full parent belongs to different provenance")
            if (
                previous.get("status") != "complete"
                or (directory / "resource_failure.json").exists()
            ):
                raise ValueError(
                    "Started full parent is terminal; automatic retries are disabled"
                )
            previous_split = _validate_split_receipt(
                previous.get("outer_split", {}), parent
            )
            validation = validate_parent_result(
                raw, parent, digest, outer_split=previous_split
            )
            if validation["raw_sha256"] != previous.get("validation", {}).get(
                "raw_sha256"
            ) or validation["runtime_sha256"] != plan_hash(runtime):
                raise ValueError("Completed full parent artifact changed")
            return previous
        if raw.exists() or any(raw.parent.glob("*")):
            raise ValueError(
                "Existing raw artifacts without full provenance are refused"
            )
        manifest = {
            "schema_version": 1,
            "status": "preparing",
            "started_at": _shared().now(),
            "plan_sha256": digest,
            "parent": parent,
            "resources": dict(RESOURCES),
            "affinity": affinity,
            "pid": os.getpid(),
            "host": host,
            "runtime": runtime,
            "registration": registration,
            "raw_path": raw.relative_to(output).as_posix(),
        }
        _shared().write_json(manifest_path, manifest)
        started = time.monotonic()
        try:
            from tabarena.utils.cache import CacheFunctionPickle
            from threadpoolctl import threadpool_limits

            with _base()._limits(directory) as memory:
                spec, task = _load_task(parent, output)
                outer_split = _outer_split_receipt(task, parent)
                experiment = copy.deepcopy(experiments[parent["config_index"]])
                experiment.experiment_cls = _audited_runner(
                    parent, digest, plan_hash(runtime), outer_split
                )
                cacher = CacheFunctionPickle(
                    cache_name="results",
                    cache_path=raw.parent,
                    include_self_in_call=True,
                )
                manifest.update(
                    status="running",
                    fit_started_at=_shared().now(),
                    outer_split=outer_split,
                )
                _shared().write_json(manifest_path, manifest)
                with threadpool_limits(limits=1):
                    experiment.run(
                        task=task,
                        fold=parent["fold"],
                        repeat=parent["repeat"],
                        sample=0,
                        task_name=spec.resolve_task_name(task),
                        cache_task_key=spec.cache_key,
                        cacher=cacher,
                        ignore_cache=False,
                        raise_on_failure=True,
                        debug_mode=False,
                        eval_metric_name=task.eval_metric,
                    )
                validation = validate_parent_result(
                    raw, parent, digest, outer_split=outer_split
                )
                if validation["runtime_sha256"] != plan_hash(runtime):
                    raise ValueError(
                        "Full artifact runtime differs from attested runtime"
                    )
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
                finished_at=_shared().now(), elapsed_seconds=time.monotonic() - started
            )
            _shared().write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("plan")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--metadata", type=Path, required=True)
    build.add_argument("--release-manifest", type=Path)
    for command in ("run-parent", "preflight"):
        sub = commands.add_parser(command)
        for name in ("plan", "output", "wheel", "registration"):
            sub.add_argument(f"--{name}", type=Path, required=True)
        sub.add_argument("--host", choices=["local", "kaggle"], required=True)
        if command == "run-parent":
            sub.add_argument("--parent-id", required=True)
            sub.add_argument("--affinity")
    args = parser.parse_args(argv)
    import ctboost

    if (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents:
        raise RuntimeError("Use python -I with the installed public0.1.60 wheel")
    sys.path.insert(0, str(ROOT))
    if args.command == "plan":
        plan = build_plan(
            output=args.output,
            metadata_path=args.metadata,
            release_manifest=args.release_manifest,
        )
        print(
            f"Frozen full plan {plan_hash(plan)}:21216 parents/169728 children",
            flush=True,
        )
    else:
        kwargs = {
            "plan_path": args.plan,
            "output": args.output,
            "host": args.host,
            "wheel_path": args.wheel,
            "registration_path": args.registration,
        }
        if args.command == "preflight":
            preflight(**kwargs)
        else:
            run_parent(
                **kwargs,
                parent_id=args.parent_id,
                affinity=None
                if args.affinity is None
                else [int(cpu) for cpu in args.affinity.split(",")],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
