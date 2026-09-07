"""Frozen default-plus-200 CPU evaluation for the public CTBoost 0.1.60 wheel.

This independently authorized baseline portfolio does not promote the optional
learning arms from the older pilot. Each process owns exactly one TabArena
parent. Plans must be publicly registered before fits; failures are terminal.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import importlib.metadata
import io
import os
import pickle
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.1.60"
PORTFOLIO_ID = "ctboost_0160_lite_baseline_hpo200_v1"
TABARENA_COMMIT = "31026f7d758390994353eba79fbfa6747616f365"
METADATA_PATH = "packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/TabArena-v0.1_tasks_metadata.csv"
METADATA_SHA256 = "02f35e19dead7e3795f65e91eb00fdb7fa255896ed6bde0f29b2ce0af7a46296"
PORTFOLIO_SHA256 = "bd1b81b98a89ab33ac4cea35cb4b7dd7727b3bcfa3bee1b044fd3fb44f965c72"
PYPI_URL = f"https://pypi.org/pypi/ctboost/{VERSION}/json"
SOURCE_FILES = (
    "benchmarks/__init__.py",
    "benchmarks/tabarena/__init__.py",
    "benchmarks/tabarena/hpo_0160.py",
    "benchmarks/tabarena/local_hpo.py",
    "benchmarks/tabarena/kaggle_hpo_worker.py",
    "benchmarks/tabarena/ctboost_model.py",
    "benchmarks/tabarena/learning_options.py",
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
    "ray": "2.55.1",
    "pydantic": "2.12.3",
    "psutil": "7.2.2",
    "threadpoolctl": "3.6.0",
}
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
BASELINE_CONTROLS = {
    "leaf_estimation_iterations": 1,
    "leaf_estimation_backtracking": False,
    "multiclass_leaf_solver": "diagonal",
    "multiclass_feature_test": "single",
}


def _shared():
    from benchmarks.tabarena import local_hpo

    return local_hpo


def plan_hash(value):
    return _shared().json_hash(value)


def source_hash(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_json(path):
    return _shared().read_json(path)


def _fetch_json(url):
    import json

    request = urllib.request.Request(url, headers={"User-Agent": "ctboost-hpo0160/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def public_wheels(release):
    if release.get("info", {}).get("version") != VERSION:
        raise ValueError("Release manifest is not public CTBoost 0.1.60")
    wheels = []
    for entry in release.get("urls", []):
        name = entry.get("filename", "")
        if not name.startswith(f"ctboost-{VERSION}-cp312-cp312-") or not name.endswith(
            ".whl"
        ):
            continue
        if not (
            name.endswith("win_amd64.whl")
            or ("manylinux" in name and name.endswith("x86_64.whl"))
        ):
            continue
        url = urllib.parse.urlparse(entry.get("url", ""))
        digest = entry.get("digests", {}).get("sha256", "")
        if entry.get("packagetype") != "bdist_wheel" or entry.get("yanked"):
            raise ValueError("Approved release wheel is missing or yanked")
        if (
            url.scheme != "https"
            or url.hostname != "files.pythonhosted.org"
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("Invalid public wheel URL or digest")
        wheels.append(
            {
                "filename": name,
                "url": entry["url"],
                "sha256": digest,
                "upload_time_iso_8601": entry["upload_time_iso_8601"],
            }
        )
    if (
        len(wheels) != 2
        or sum(w["filename"].endswith("win_amd64.whl") for w in wheels) != 1
    ):
        raise ValueError(
            "Exactly one public cp312 Windows and Linux x86_64 wheel is required"
        )
    return sorted(wheels, key=lambda item: item["filename"])


def load_population(path):
    if source_hash(path) != METADATA_SHA256:
        raise ValueError("Pinned TabArena task metadata changed")
    rows = csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8")))
    population = [
        {
            "dataset": row["dataset_name"],
            "task_id": int(row["task_id_str"]),
            "problem_type": row["problem_type"],
            "eval_metric": row["eval_metric"],
            "num_instances_train": float(row["num_instances_train"]),
            "num_features": int(row["num_features"]),
            "num_classes": int(row["num_classes"]),
            "task_type": row["task_type"],
            "stratify_on": row["stratify_on"] or None,
            "group_on": row["group_on"] or None,
            "time_on": row["time_on"] or None,
        }
        for row in rows
        if row["repeat"] == "0" and row["fold"] == "0"
    ]
    if (
        len(population) != 51
        or len({row["task_id"] for row in population}) != 51
        or len({row["dataset"] for row in population}) != 51
    ):
        raise ValueError("Expected all 51 distinct official Lite r0f0 tasks")
    return sorted(population, key=lambda row: row["dataset"])


def build_experiments():
    from tabarena.utils.config_utils import CustomAGConfigGenerator

    from benchmarks.tabarena.ctboost_model import (
        CTBoostTabArenaModel,
        generate_configs_ctboost,
    )

    configs = generate_configs_ctboost(200)
    if plan_hash(configs) != PORTFOLIO_SHA256 or configs[
        :25
    ] != generate_configs_ctboost(25):
        raise ValueError("The frozen 200-config portfolio or its first 25 changed")
    generator = CustomAGConfigGenerator(
        model_cls=CTBoostTabArenaModel,
        search_space_func=generate_configs_ctboost,
        manual_configs=[{}],
    )
    experiments = generator.generate_all_bag_experiments(
        num_random_configs=200,
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
                "num_cpus": 2,
                "num_gpus": 0,
                "memory_limit": 8.0,
            },
            "shuffle_features": False,
        },
    )
    if len(experiments) != 201:
        raise ValueError("Expected a default and 200 globally generated configurations")
    for index, experiment in enumerate(experiments):
        name = f"CTBoost_{'c1' if index == 0 else f'r{index}'}_default_BAG_L1"
        ensemble = experiment.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]
        if (
            experiment.name != name
            or ensemble["model_random_seed"] != index * 8
            or ensemble["vary_seed_across_folds"] is not True
            or ensemble["fold_fitting_strategy"] != "sequential_local"
        ):
            raise ValueError(
                "Configuration identity, seed, or sequential-fold contract changed"
            )
    return experiments, [{}] + configs


def make_parents(population, names):
    parents = []
    for config_index, name in enumerate(names):
        for dataset in sorted(population, key=lambda row: row["dataset"]):
            ordinal = len(parents)
            parents.append(
                {
                    "parent_id": f"openml-{dataset['task_id']}-r0-f0-c{config_index:03d}",
                    "ordinal": ordinal,
                    "owner": "local" if ordinal % 9 < 4 else "kaggle",
                    "dataset": dataset["dataset"],
                    "task_id": dataset["task_id"],
                    "problem_type": dataset["problem_type"],
                    "config_index": config_index,
                    "config_name": name,
                    "repeat": 0,
                    "fold": 0,
                    "child_seeds": list(range(config_index * 8, config_index * 8 + 8)),
                }
            )
    return parents


def validate_plan(plan, *, verify_sources=True):
    required = {
        "schema_version": 1,
        "portfolio_id": PORTFOLIO_ID,
        "ctboost_version": VERSION,
        "tabarena_commit": TABARENA_COMMIT,
        "metadata_sha256": METADATA_SHA256,
        "portfolio_200_sha256": PORTFOLIO_SHA256,
        "package_pins": PINS,
        "resources": RESOURCES,
        "baseline_controls": BASELINE_CONTROLS,
        "expected_parent_count": 10251,
        "expected_child_count": 82008,
        "uses_test_outcomes_for_selection": False,
        "authorization": "explicit_user_request_for_independent_baseline_hpo200",
    }
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError("Plan identity, resources, pins, or baseline contract changed")
    population = plan.get("population", [])
    if (
        len(population) != 51
        or len({row["task_id"] for row in population}) != 51
        or len({row["dataset"] for row in population}) != 51
    ):
        raise ValueError("Plan population must contain exactly 51 unique tasks")
    configs = plan.get("configurations", [])
    if (
        len(configs) != 201
        or configs[0] != {}
        or plan_hash(configs[1:]) != PORTFOLIO_SHA256
    ):
        raise ValueError(
            "Plan configurations differ from the frozen baseline portfolio"
        )
    names = [
        f"CTBoost_{'c1' if i == 0 else f'r{i}'}_default_BAG_L1" for i in range(201)
    ]
    if plan.get("parents") != make_parents(population, names):
        raise ValueError("Parent membership, seeds, order, or host ownership changed")
    if set(plan.get("source_sha256", {})) != set(SOURCE_FILES):
        raise ValueError("Incomplete worker source provenance")
    if verify_sources and plan["source_sha256"] != {
        name: source_hash(ROOT / name) for name in SOURCE_FILES
    }:
        raise ValueError("Worker source differs from the frozen public plan")
    if plan.get("configurations_sha256") != plan_hash(configs):
        raise ValueError("Configuration digest mismatch")
    catalog = plan.get("public_wheels", [])
    release = {
        "info": {"version": VERSION},
        "urls": [
            {
                **entry,
                "digests": {"sha256": entry.get("sha256", "")},
                "packagetype": "bdist_wheel",
                "yanked": False,
            }
            for entry in catalog
        ],
    }
    if public_wheels(release) != catalog or plan.get("public_release_url") != PYPI_URL:
        raise ValueError("Public release wheel catalog changed")
    if not isinstance(plan.get("experiments"), list) or len(plan["experiments"]) != 201:
        raise ValueError("Missing official TabArena experiment definitions")
    return plan


def build_plan(*, output, metadata_path, release_manifest=None):
    release = _fetch_json(PYPI_URL)
    if release_manifest is not None and public_wheels(
        read_json(release_manifest)
    ) != public_wheels(release):
        raise ValueError(
            "Supplied release manifest differs from current public PyPI wheels"
        )
    experiments, configs = build_experiments()
    population = load_population(metadata_path)
    path = Path(output).resolve() / "plan.json"
    existing = read_json(path) if path.exists() else None
    plan = {
        "schema_version": 1,
        "portfolio_id": PORTFOLIO_ID,
        "ctboost_version": VERSION,
        "created_at_utc": existing["created_at_utc"] if existing else _shared().now(),
        "authorization": "explicit_user_request_for_independent_baseline_hpo200",
        "uses_test_outcomes_for_selection": False,
        "selection_policy": "No optional learning-arm promotion; configurations and task ownership are fixed before outcomes. Outer test predictions are final evidence only.",
        "tabarena_repository": "https://github.com/captnmarkus/tabarena",
        "tabarena_commit": TABARENA_COMMIT,
        "metadata_path": METADATA_PATH,
        "metadata_sha256": METADATA_SHA256,
        "portfolio_200_sha256": PORTFOLIO_SHA256,
        "package_pins": dict(PINS),
        "python_major_minor": "3.12",
        "resources": dict(RESOURCES),
        "baseline_controls": dict(BASELINE_CONTROLS),
        "public_release_url": PYPI_URL,
        "public_wheels": public_wheels(release),
        "population": population,
        "configurations": configs,
        "configurations_sha256": plan_hash(configs),
        "experiments": [experiment.to_yaml_dict() for experiment in experiments],
        "parents": make_parents(
            population, [experiment.name for experiment in experiments]
        ),
        "source_sha256": {name: source_hash(ROOT / name) for name in SOURCE_FILES},
        "expected_parent_count": 10251,
        "expected_child_count": 82008,
        "timing_disclosure": "Author-run CPU timings on heterogeneous hardware, not canonical TabArena timings. Host identity and hardware are recorded per parent.",
        "failure_policy": "Started-parent and resource failures are terminal. Any resource revision requires a separate public protocol; no silent fallback or retry.",
    }
    validate_plan(plan)
    if existing is not None and existing != plan:
        raise ValueError("Existing immutable plan differs; use a new output directory")
    if not path.exists():
        if path.parent.exists() and any(path.parent.iterdir()):
            raise ValueError("A new plan requires an empty output directory")
        _shared().write_json(path, plan)
    return plan


def verify_registration(plan, registration_path, output):
    receipt = read_json(registration_path)
    commit = receipt.get("commit", "")
    name = receipt.get("plan_path", "")
    if receipt.get("repository") != "captnmarkus/ctboost" or not re.fullmatch(
        r"[0-9a-f]{40}", commit
    ):
        raise ValueError("Public registration must name an immutable CTBoost commit")
    if not name or name.startswith("/") or ".." in Path(name).parts or "\\" in name:
        raise ValueError("Invalid public plan path")
    if receipt.get("plan_sha256") != plan_hash(plan):
        raise ValueError("Registration belongs to a different plan")
    identity = {
        "registration_sha256": plan_hash(receipt),
        "plan_sha256": plan_hash(plan),
    }
    cache = Path(output) / "registration_verified.json"
    if cache.exists():
        previous = read_json(cache)
        if all(previous.get(key) == value for key, value in identity.items()):
            return previous
        raise ValueError("Output contains a different public registration")
    url = f"https://raw.githubusercontent.com/{receipt['repository']}/{commit}/{name}"
    if plan_hash(_fetch_json(url)) != plan_hash(plan):
        raise ValueError("Public preregistered plan differs from worker inputs")
    verified = {
        **identity,
        "receipt": receipt,
        "public_url": url,
        "verified_at_utc": _shared().now(),
    }
    _shared().write_json(cache, verified)
    return verified


def verify_installed_wheel(plan, wheel_path, *, package_root, native_file):
    """Verify native and Python package bytes against the preregistered wheel."""
    wheel = Path(wheel_path).resolve()
    matches = [
        entry for entry in plan["public_wheels"] if entry["filename"] == wheel.name
    ]
    if len(matches) != 1 or _shared().file_hash(wheel) != matches[0]["sha256"]:
        raise ValueError("Worker wheel is not an approved public release artifact")
    import hashlib

    package_hashes = {}
    site_packages = Path(package_root).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.endswith("/") or not member.startswith(
                ("ctboost/", "ctboost.libs/")
            ):
                continue
            relative = Path(member)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe wheel member")
            digest = hashlib.sha256(archive.read(member)).hexdigest()
            if _shared().file_hash(site_packages / relative) != digest:
                raise ValueError(
                    f"Installed package differs from its public wheel: {member}"
                )
            package_hashes[member] = digest
    native_member = Path(native_file).resolve().relative_to(site_packages).as_posix()
    if native_member not in package_hashes:
        raise ValueError("Public wheel does not contain the loaded native binary")
    return matches[0], package_hashes


def runtime_provenance(plan, wheel_path):
    import psutil
    import tabarena

    import ctboost
    from ctboost import _core

    if (
        sys.version_info[:2] != (3, 12)
        or (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents
    ):
        raise ValueError("Use Python 3.12 -I with the installed public wheel")
    packages = {name: importlib.metadata.version(name) for name in PINS}
    if (
        packages != PINS
        or ctboost.__version__ != VERSION
        or ctboost.build_info().get("version") != VERSION
    ):
        raise ValueError(
            f"Installed public wheel or pinned packages differ: {packages}"
        )
    public_wheel, package_hashes = verify_installed_wheel(
        plan,
        wheel_path,
        package_root=Path(ctboost.__file__).resolve().parent,
        native_file=_core.__file__,
    )
    roots = [
        path
        for path in Path(tabarena.__file__).resolve().parents
        if (path / ".git").exists()
    ]
    if not roots:
        raise ValueError("Pinned editable TabArena checkout is required")
    source = roots[0]
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
        raise ValueError("Pinned TabArena source changed")
    if load_population(source / METADATA_PATH) != plan["population"]:
        raise ValueError("Actual official task population differs from the plan")
    environment = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    processor = platform.processor()
    if sys.platform.startswith("linux") and Path("/proc/cpuinfo").exists():
        processor = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            processor,
        )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": processor,
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "total_memory_bytes": int(psutil.virtual_memory().total),
        "packages": packages,
        "environment": environment,
        "environment_sha256": plan_hash(environment),
        "public_wheel": public_wheel,
        "package_files_sha256": package_hashes,
        "native_extension_sha256": _shared().file_hash(_core.__file__),
        "ctboost_build_info": ctboost.build_info(),
        "tabarena_commit": commit,
        "source_sha256": plan["source_sha256"],
    }


def preflight(*, plan_path, output, host, wheel_path, registration_path):
    """Attest the public plan, installed runtime and actual API without task loading."""
    if host not in {"local", "kaggle"}:
        raise ValueError("Unknown execution host")
    plan = validate_plan(read_json(plan_path))
    output = Path(output).resolve()
    registration = verify_registration(plan, registration_path, output)
    runtime = runtime_provenance(plan, wheel_path)
    experiments, configs = build_experiments()
    if (
        configs != plan["configurations"]
        or [item.to_yaml_dict() for item in experiments] != plan["experiments"]
    ):
        raise ValueError("Actual pinned TabArena experiment definitions changed")
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
        "verified_at_utc": _shared().now(),
    }
    _shared().write_json(output / f"preflight-{host}.json", evidence)
    return evidence


def _audited_runner(parent, digest, runtime_digest):
    from tabarena.benchmark.experiment import OOFExperimentRunner

    class AuditedHpo0160Runner(OOFExperimentRunner):
        def post_evaluate(self, out):
            out = super().post_evaluate(out)
            out["hpo_0160"] = {
                "ctboost_version": VERSION,
                "portfolio_id": PORTFOLIO_ID,
                "plan_sha256": digest,
                "parent_id": parent["parent_id"],
                "owner": parent["owner"],
                "runtime_sha256": runtime_digest,
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
                    except Exception:  # noqa: BLE001, S110 -- preserve the original fit failure
                        pass
                raise

    return AuditedHpo0160Runner


def validate_parent_result(path, parent, plan_sha256):
    import numpy as np

    from benchmarks.tabarena.kaggle_hpo_worker import validate_result_file

    validated = validate_result_file(
        Path(path), {**parent, "datasets": [parent["dataset"]]}
    )
    with gzip.open(path, "rb") as stream:
        result = pickle.load(stream)
    audit = result.get("hpo_0160", {})
    expected = {
        "ctboost_version": VERSION,
        "portfolio_id": PORTFOLIO_ID,
        "plan_sha256": plan_sha256,
        "parent_id": parent["parent_id"],
        "owner": parent["owner"],
    }
    if any(
        audit.get(key) != value for key, value in expected.items()
    ) or not re.fullmatch(r"[0-9a-f]{64}", audit.get("runtime_sha256", "")):
        raise ValueError("Raw result lacks this plan's parent/runtime provenance")
    if (
        str(validated["task_id"]) != str(parent["task_id"])
        or result["problem_type"] != parent["problem_type"]
    ):
        raise ValueError("Official task identity or problem type mismatch")
    children = audit.get("children", {})
    if set(children) != {f"S1F{i}" for i in range(1, 9)}:
        raise ValueError("Missing native child audits")
    for fold in range(8):
        child = children[f"S1F{fold + 1}"]
        if (
            child.get("random_seed") != parent["config_index"] * 8 + fold
            or child.get("task_type") != "CPU"
            or child.get("feature_test") != "quadratic"
            or child.get("native_controls") != BASELINE_CONTROLS
        ):
            raise ValueError(
                "Fitted child changed the baseline controls, seed, or CPU contract"
            )
    simulation = result["simulation_artifacts"]
    predictions = np.asarray(simulation["bag_info"]["pred_proba_test_per_child"])
    ensemble = np.asarray(simulation["pred_proba_dict_test"][parent["config_name"]])
    if not np.allclose(predictions.mean(axis=0), ensemble, rtol=1e-5, atol=1e-6):
        raise ValueError("Parent predictions are not the eight-child mean")
    if parent["problem_type"] != "regression" and (
        np.any(predictions < -1e-6) or np.any(predictions > 1 + 1e-6)
    ):
        raise ValueError("Invalid child probabilities")
    if parent["problem_type"] == "multiclass" and not np.allclose(
        predictions.sum(axis=2), 1.0, rtol=0, atol=1e-5
    ):
        raise ValueError("Child probabilities do not sum to one")
    return {
        **validated,
        "raw_sha256": _shared().file_hash(path),
        "size_bytes": Path(path).stat().st_size,
        "runtime_sha256": audit["runtime_sha256"],
        "native_children": children,
    }


@contextmanager
def _limits(directory):
    stopped = threading.Event()

    def wall_guard():
        if not stopped.wait(RESOURCES["parent_wall_limit_seconds"]):
            try:
                _shared().write_json(
                    directory / "resource_failure.json",
                    {
                        "status": "timeout",
                        "limit_seconds": RESOURCES["parent_wall_limit_seconds"],
                        "pid": os.getpid(),
                        "recorded_at": _shared().now(),
                    },
                )
            finally:
                os._exit(4)

    thread = threading.Thread(target=wall_guard, daemon=True)
    thread.start()
    try:
        with _shared()._memory_guard(
            8 * 1024**3, directory / "resource_failure.json"
        ) as memory:
            yield memory
    finally:
        stopped.set()
        thread.join(timeout=1)


def run_parent(
    *, plan_path, output, parent_id, host, wheel_path, registration_path, affinity=None
):
    plan = validate_plan(read_json(plan_path))
    digest = plan_hash(plan)
    matches = [parent for parent in plan["parents"] if parent["parent_id"] == parent_id]
    if len(matches) != 1 or host != matches[0]["owner"]:
        raise ValueError("Parent does not belong to this host in the frozen plan")
    parent = matches[0]
    if affinity is None:
        import psutil

        affinity = psutil.Process().cpu_affinity()[:2]
    affinity = _shared()._configure_process(2, affinity)
    output = Path(output).resolve()
    registration = verify_registration(plan, registration_path, output)
    runtime = runtime_provenance(plan, wheel_path)
    experiments, configs = build_experiments()
    if (
        configs != plan["configurations"]
        or [item.to_yaml_dict() for item in experiments] != plan["experiments"]
    ):
        raise ValueError("Actual pinned TabArena experiment definitions changed")
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
    with _shared()._parent_lock(directory / "worker.lock"):
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if (
                previous.get("plan_sha256") != digest
                or previous.get("parent") != parent
            ):
                raise ValueError("Existing parent belongs to different provenance")
            if (
                previous.get("status") != "complete"
                or (directory / "resource_failure.json").exists()
            ):
                raise ValueError(
                    "Started-parent failures require deliberate reconciliation; no retries"
                )
            validation = validate_parent_result(raw, parent, digest)
            if previous.get("validation", {}).get("raw_sha256") != validation[
                "raw_sha256"
            ] or validation["runtime_sha256"] != plan_hash(previous["runtime"]):
                raise ValueError(
                    "Completed parent artifact or runtime evidence changed"
                )
            return previous
        if raw.exists() or any(raw.parent.glob("*")):
            raise ValueError(
                "Existing raw artifacts without parent provenance are refused"
            )
        manifest = {
            "schema_version": 1,
            "status": "preparing",
            "started_at": _shared().now(),
            "plan_sha256": digest,
            "parent": parent,
            "resources": RESOURCES,
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

            with _limits(directory) as memory:
                spec, task = _shared()._load_task(parent, output)
                experiment = copy.deepcopy(experiments[parent["config_index"]])
                experiment.experiment_cls = _audited_runner(
                    parent, digest, plan_hash(runtime)
                )
                cacher = CacheFunctionPickle(
                    cache_name="results",
                    cache_path=raw.parent,
                    include_self_in_call=True,
                )
                manifest.update(status="running", fit_started_at=_shared().now())
                _shared().write_json(manifest_path, manifest)
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
                validation = validate_parent_result(raw, parent, digest)
                if validation["runtime_sha256"] != plan_hash(runtime):
                    raise ValueError("Result runtime differs from the attested worker")
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
    run = commands.add_parser("run-parent")
    for name in ("plan", "output", "wheel", "registration"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--parent-id", required=True)
    run.add_argument("--host", choices=["local", "kaggle"], required=True)
    run.add_argument("--affinity")
    check = commands.add_parser("preflight")
    for name in ("plan", "output", "wheel", "registration"):
        check.add_argument(f"--{name}", type=Path, required=True)
    check.add_argument("--host", choices=["local", "kaggle"], required=True)
    args = parser.parse_args(argv)
    # Preload the installed package before adding benchmark-only source imports.
    import ctboost

    if (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents:
        raise RuntimeError("Run with python -I using an installed public wheel")
    sys.path.insert(0, str(ROOT))
    if args.command == "plan":
        plan = build_plan(
            output=args.output,
            metadata_path=args.metadata,
            release_manifest=args.release_manifest,
        )
        print(
            f"Frozen plan {plan_hash(plan)}: 10251 parents / 82008 children", flush=True
        )
    elif args.command == "preflight":
        evidence = preflight(
            plan_path=args.plan,
            output=args.output,
            host=args.host,
            wheel_path=args.wheel,
            registration_path=args.registration,
        )
        print(
            f"Verified {evidence['experiment_count']} experiment definitions; no fits started",
            flush=True,
        )
    else:
        run_parent(
            plan_path=args.plan,
            output=args.output,
            parent_id=args.parent_id,
            host=args.host,
            wheel_path=args.wheel,
            registration_path=args.registration,
            affinity=None
            if args.affinity is None
            else [int(cpu) for cpu in args.affinity.split(",")],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
