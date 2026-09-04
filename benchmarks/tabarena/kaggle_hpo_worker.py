#!/usr/bin/env python3
"""One reproducible CPU shard of CTBoost 0.1.58's TabArena Lite HPO run.

The 156 shards cover the manual default and the first 25 frozen configurations
on all 51 r0f0 tasks, with eight bag children per parent. No search parameters
are selected from benchmark outcomes. Kaggle timings are non-canonical.

The controller embeds SHARD_INDEX into each standalone Kaggle script. Local
``--plan-only`` validates the experiment identities without installing packages,
downloading datasets, or fitting. Actual workers install immutable TabArena
source, the public CTBoost wheel, and the same AutoGluon version as the 0.1.56
default run. Partial results and their archive are checkpointed after each task.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TABARENA_REPOSITORY = "https://github.com/captnmarkus/tabarena.git"
TABARENA_COMMIT = "31026f7d758390994353eba79fbfa6747616f365"
CTBOOST_VERSION = "0.1.58"
AUTOGLUON_VERSION = "1.6.2b20260821"
RUNTIME_PACKAGE_PINS = {
    "autogluon.tabular": AUTOGLUON_VERSION,
    "numpy": "2.0.2",
    "pandas": "2.3.3",
    "scipy": "1.16.3",
    "scikit-learn": "1.6.1",
    "openml": "0.15.1",
    "pyarrow": "24.0.0",
    "ray": "2.55.1",
    "pydantic": "2.12.3",
}
BENCHMARK_NAME = "ctboost_0158_lite_hpo25_20260904"
PORTFOLIO_200_SHA256 = (
    "bd1b81b98a89ab33ac4cea35cb4b7dd7727b3bcfa3bee1b044fd3fb44f965c72"
)
HPO_CONFIGS = 25
CONFIGURATION_COUNT = HPO_CONFIGS + 1
SHARD_INDEX = 0
BOOTSTRAP_ENV = "CTBOOST_TABARENA_HPO_0158_BOOTSTRAPPED"
DATASET_GROUPS = (
    (
        "APSFailure",
        "airfoil_self_noise",
        "maternal_health_risk",
        "diabetes",
        "QSAR_fish_toxicity",
        "blood-transfusion-service-center",
    ),
    (
        "kddcup09_appetency",
        "anneal",
        "Is-this-a-good-customer",
        "credit-g",
        "website_phishing",
        "Another-Dataset-on-used-Fiat-500",
        "Fitness_Club",
        "healthcare_insurance_expenses",
        "concrete_compressive_strength",
    ),
    (
        "Bioresponse",
        "coil2000_insurance_policies",
        "NATICUSdroid",
        "physiochemical_protein",
        "in_vehicle_coupon_recommendation",
        "online_shoppers_intention",
        "houses",
        "churn",
        "Marketing_Campaign",
    ),
    (
        "hiva_agnostic",
        "GiveMeSomeCredit",
        "bank-marketing",
        "Amazon_employee_access",
        "HR_Analytics_Job_Change_of_Data_Scientists",
        "splice",
        "E-CommereShippingData",
        "hazelnut-spread-contaminant-detection",
        "seismic-bumps",
    ),
    (
        "QSAR-TID-11",
        "superconductivity",
        "taiwanese_bankruptcy_prediction",
        "Food_Delivery_Time",
        "heloc",
        "miami_housing",
        "students_dropout_and_academic_success",
        "wine_quality",
        "qsar-biodeg",
    ),
    (
        "Diabetes130US",
        "customer_satisfaction_in_airline",
        "SDSS17",
        "credit_card_clients_default",
        "diamonds",
        "polish_companies_bankruptcy",
        "jm1",
        "MIC",
        "Bank_Customer_Churn",
    ),
)
SHARD_COUNT = CONFIGURATION_COUNT * len(DATASET_GROUPS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def shard_spec(index: int) -> dict[str, Any]:
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < SHARD_COUNT
    ):
        raise ValueError(f"shard_index must be an integer in [0, {SHARD_COUNT})")
    config_index, dataset_group = divmod(index, len(DATASET_GROUPS))
    suffix = "c1" if config_index == 0 else f"r{config_index}"
    datasets = list(DATASET_GROUPS[dataset_group])
    return {
        "shard_index": index,
        "shard_count": SHARD_COUNT,
        "config_index": config_index,
        "dataset_group": dataset_group,
        "config_name": f"CTBoost_{suffix}_default_BAG_L1",
        "datasets": datasets,
        "expected_parent_results_in_shard": len(datasets),
        "expected_child_fits_in_shard": len(datasets) * 8,
    }


def run_command(
    args: list[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    print("\n>>> " + shlex.join(args), flush=True)
    return subprocess.run(args, cwd=cwd, check=check)


def install_exact_source(source_dir: Path, artifacts: Path) -> dict[str, Any]:
    """Fetch a commit rather than a moving branch, and require the public wheel."""
    source_dir.mkdir(parents=True, exist_ok=True)
    if not (source_dir / ".git").exists():
        run_command(["git", "init", str(source_dir)])
    run_command(
        [
            "git",
            "-C",
            str(source_dir),
            "fetch",
            "--depth",
            "1",
            TABARENA_REPOSITORY,
            TABARENA_COMMIT,
        ]
    )
    run_command(["git", "-C", str(source_dir), "checkout", "--detach", TABARENA_COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != TABARENA_COMMIT:
        raise RuntimeError(f"Expected TabArena commit {TABARENA_COMMIT}, got {head}")
    wheel_dir = artifacts / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--disable-pip-version-check",
            "--index-url",
            "https://pypi.org/simple",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(wheel_dir),
            f"ctboost=={CTBOOST_VERSION}",
        ]
    )
    wheels = list(wheel_dir.glob(f"ctboost-{CTBOOST_VERSION}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one CTBoost wheel, got {len(wheels)}")
    uv = shutil.which("uv")
    if uv is None:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "uv",
            ]
        )
        uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv was installed but is not on PATH")
    run_command(
        [
            uv,
            "pip",
            "install",
            "--system",
            "--python",
            sys.executable,
            "--prerelease=allow",
            str(wheels[0]),
            *[f"{name}=={version}" for name, version in RUNTIME_PACKAGE_PINS.items()],
            str(source_dir / "packages" / "tabarena"),
            str(source_dir / "packages" / "tabflow_slurm"),
        ]
    )
    return {
        "tabarena_commit": head,
        "ctboost_wheel": {
            "filename": wheels[0].name,
            "size_bytes": wheels[0].stat().st_size,
            "sha256": sha256_file(wheels[0]),
            "source": "https://pypi.org/simple/ctboost/",
        },
    }


def build_experiments(num_cpus: int = 4) -> tuple[list[Any], dict[str, Any]]:
    """Generate before sharding so model names and fold/config seeds are stable."""
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.models.ctboost.hpo import generate_configs_ctboost

    portfolio = generate_configs_ctboost(200)
    if len(portfolio) != 200 or json_hash(portfolio) != PORTFOLIO_200_SHA256:
        raise RuntimeError(
            "Frozen 200-configuration portfolio does not match the audited source"
        )
    prefix = generate_configs_ctboost(HPO_CONFIGS)
    if prefix != portfolio[:HPO_CONFIGS]:
        raise RuntimeError("The 25 configurations must be the frozen portfolio prefix")
    bundle = TabArenaV0pt1ExperimentBundle(
        models=[("CTBoost", HPO_CONFIGS)], model_verbosity=2
    )
    experiments = bundle.build_experiments(
        time_limit=3600, num_cpus=num_cpus, num_gpus=0, memory_limit=28
    )
    expected = [
        shard_spec(i * len(DATASET_GROUPS))["config_name"]
        for i in range(CONFIGURATION_COUNT)
    ]
    if [experiment.name for experiment in experiments] != expected:
        raise RuntimeError("TabArena generated unexpected configuration names or order")
    records = [experiment.to_yaml_dict() for experiment in experiments]
    return experiments, {
        "portfolio_25_sha256": json_hash(prefix),
        "experiments_sha256": json_hash(records),
        "experiments": records,
    }


def build_plan(
    workspace: Path, spec: dict[str, Any], experiments: list[Any], num_cpus: int
) -> tuple[list[str], Path | None]:
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.benchmark.task.metadata import TaskSubset
    from tabflow_slurm import (
        LocalSequentialSetup,
        ModelJob,
        PathSetup,
        TabArenaV0pt1BenchmarkPlan,
        TabArenaV0pt1ResourcesSetup,
    )

    # Passing the already-built experiment leaves all full-portfolio seed assignments intact.
    plan = TabArenaV0pt1BenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[ModelJob(models=[experiments[spec["config_index"]]], name="cpu")],
        task_subset=TaskSubset(subset="lite", dataset_names=spec["datasets"]),
        path_setup=PathSetup(workspace=str(workspace), python_path=sys.executable),
        experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
        resources_setup=TabArenaV0pt1ResourcesSetup(
            time_limit=3600, num_cpus=num_cpus, memory_limit=28
        ),
        scheduler_setup=LocalSequentialSetup(
            continue_on_error=True, execution_mode="subprocess"
        ),
        prefetch_model_weights=False,
    )
    commands = plan.setup_jobs(num_ray_cpus=num_cpus)
    if not commands:
        return commands, None
    if len(commands) != 1:
        raise RuntimeError(f"Expected one local run command, got {commands!r}")
    paths = [
        Path(token)
        for token in shlex.split(commands[0], posix=os.name != "nt")
        if token.endswith(".json")
    ]
    if len(paths) != 1 or not paths[0].is_file():
        raise RuntimeError("Could not resolve generated job JSON")
    jobs = json.loads(paths[0].read_text(encoding="utf-8"))
    items = [item for job in jobs["jobs"] for item in job["items"]]
    actual = {
        (item["experiment"], item["dataset"], item["repeat"], item["fold"])
        for item in items
    }
    expected = {(spec["config_name"], dataset, 0, 0) for dataset in spec["datasets"]}
    if len(actual) != len(items) or not actual <= expected:
        raise RuntimeError(
            "Generated jobs contain duplicates, unexpected configurations, tasks, or splits"
        )
    return commands, paths[0]


def validate_result_file(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Validate only this worker's own generated pickle, never untrusted uploads."""
    import numpy as np

    with gzip.open(path, "rb") as stream:
        result = pickle.load(stream)
    if not isinstance(result, dict):
        raise TypeError("result is not a dictionary")
    for record in (result, result.get("experiment_metadata", {})):
        if not isinstance(record, dict):
            raise TypeError("invalid experiment metadata")
        for key in ("exception", "error", "traceback", "failure", "failure_artifact"):
            if record.get(key):
                raise ValueError(f"result records {key}")
        if record.get("success") is False:
            raise ValueError("result explicitly records failure")
    for metric in ("metric_error", "metric_error_val"):
        value = np.asarray(result.get(metric))
        if (
            value.ndim != 0
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{metric} is missing or non-finite")
    if result.get("framework") != spec["config_name"]:
        raise ValueError("result configuration does not match its shard")
    task = result.get("task_metadata", {})
    if (
        task.get("name") not in spec["datasets"]
        or task.get("fold") != 0
        or task.get("repeat") != 0
    ):
        raise ValueError("result dataset/split does not match its shard")
    if path.parent.name != "0_0" or path.parent.parent.name != str(task.get("tid")):
        raise ValueError("result path does not match its task id/split")
    if path.parent.parent.parent.name != spec["config_name"]:
        raise ValueError("result path does not match its configuration")
    method = result.get("method_metadata", {})
    info = method.get("info", {})
    bagged = info.get("bagged_info", {})
    children = info.get("children_info", {})
    if bagged.get("num_child_models") != 8 or len(children) != 8:
        raise ValueError("result does not contain all eight fitted bag children")
    expected_names = {f"S1F{fold}" for fold in range(1, 9)}
    if (
        set(children) != expected_names
        or set(bagged.get("child_model_names", [])) != expected_names
    ):
        raise ValueError("bag child identities do not match the eight-fold protocol")
    for name, child in [("parent", info), *children.items()]:
        if not all(
            child.get(flag) is True for flag in ("is_fit", "is_valid", "can_infer")
        ):
            raise ValueError(f"{name} is not a fitted, valid, inferable model")
    for fold in range(8):
        seed = children[f"S1F{fold + 1}"]["hyperparameters"].get("random_seed")
        if seed != spec["config_index"] * 8 + fold:
            raise ValueError(
                "bag child seeds differ from the frozen full-portfolio assignment"
            )
    simulation = result.get("simulation_artifacts", {})
    problem = result.get("problem_type")
    if problem not in ("binary", "multiclass", "regression"):
        raise ValueError("unsupported or missing problem type")
    for split in ("val", "test"):
        predictions = simulation.get(f"pred_proba_dict_{split}", {})
        if set(predictions) != {spec["config_name"]}:
            raise ValueError(f"missing or unexpected {split} predictions")
        pred = np.asarray(predictions[spec["config_name"]])
        target = np.asarray(simulation.get(f"y_{split}"))
        if target.ndim != 1 or target.size == 0 or not np.isfinite(target).all():
            raise ValueError(f"invalid {split} targets")
        expected_shape = (
            (len(target), int(simulation["num_classes"]))
            if problem == "multiclass"
            else target.shape
        )
        if pred.shape != expected_shape or not np.isfinite(pred).all():
            raise ValueError(f"invalid/non-finite {split} prediction shape or values")
        if problem != "regression" and (
            np.any(pred < -1e-6) or np.any(pred > 1 + 1e-6)
        ):
            raise ValueError(f"invalid {split} probabilities")
        if problem == "multiclass" and not np.allclose(
            pred.sum(axis=1), 1.0, atol=1e-5, rtol=0
        ):
            raise ValueError(f"{split} probability rows do not sum to one")
    bag = simulation.get("bag_info", {})
    fold_indices = bag.get("val_idx_per_child", [])
    child_predictions = bag.get("pred_proba_test_per_child", [])
    if len(fold_indices) != 8 or len(child_predictions) != 8:
        raise ValueError("simulation artifacts do not contain all eight bag folds")
    all_indices = np.concatenate([np.asarray(indices) for indices in fold_indices])
    if not np.array_equal(np.sort(all_indices), np.arange(len(simulation["y_val"]))):
        raise ValueError(
            "validation folds do not cover the out-of-fold target exactly once"
        )
    test_shape = np.asarray(
        simulation["pred_proba_dict_test"][spec["config_name"]]
    ).shape
    if any(
        np.asarray(pred).shape != test_shape or not np.isfinite(pred).all()
        for pred in child_predictions
    ):
        raise ValueError("invalid/non-finite child test predictions")
    return {"dataset": task["name"], "task_id": str(task["tid"]), "bag_children": 8}


def checkpoint(
    workspace: Path,
    artifacts: Path,
    manifest: dict[str, Any],
    *,
    validate: bool = False,
) -> None:
    paths = sorted(
        (workspace / "output" / BENCHMARK_NAME / "data").rglob("results.pkl")
    )
    manifest["result_files"] = [
        {
            "path": path.relative_to(workspace).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    manifest["result_file_count"] = len(paths)
    if validate:
        verified = []
        invalid = []
        for path in paths:
            try:
                verified.append(validate_result_file(path, manifest))
            except Exception as exc:  # noqa: BLE001 -- preserve every invalid artifact for diagnosis
                invalid.append(
                    {
                        "path": path.relative_to(workspace).as_posix(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        datasets = [record["dataset"] for record in verified]
        if len(set(datasets)) != len(datasets):
            invalid.append({"error": "duplicate dataset results in shard"})
        manifest["validation"] = {
            "valid_result_count": len(verified),
            "invalid_results": invalid,
            "bag_children_verified": sum(record["bag_children"] for record in verified),
            "missing_datasets": sorted(set(manifest["datasets"]) - set(datasets)),
        }
    manifest["last_checkpoint_at_utc"] = utc_now()
    if workspace.exists():
        archive = (
            artifacts
            / f"ctboost_tabarena_lite_hpo_s{manifest['shard_index']:03d}_raw.tar.gz"
        )
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as stream:
            # Store results and reproducibility records; omit downloaded OpenML datasets/cache.
            for dirname in ("output", "setup_out"):
                path = workspace / dirname
                if path.exists():
                    stream.add(path, arcname=f"workspace/{dirname}")
        temporary.replace(archive)
        manifest["workspace_archive"] = {
            "path": archive.relative_to(artifacts.parent).as_posix(),
            "size_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
        }
    write_json(artifacts / "manifest.json", manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=SHARD_INDEX)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    spec = shard_spec(args.shard_index)
    root = (
        args.output_root
        or Path("/kaggle/working")
        / f"ctboost_tabarena_lite_hpo_s{args.shard_index:03d}"
    )
    root = root.resolve()
    artifacts = root / "artifacts"
    workspace = root / "workspace"
    artifacts.mkdir(parents=True, exist_ok=True)
    num_cpus = min(4, os.cpu_count() or 1)
    manifest: dict[str, Any] = {
        **spec,
        "schema_version": 1,
        "status": "initializing",
        "started_at_utc": utc_now(),
        "benchmark_name": BENCHMARK_NAME,
        "protocol": "TabArena-v0.1 Lite (r0f0), CTBoost default plus 25 frozen HPO configurations",
        "official_tabarena_lite_dataset_count": 51,
        "configuration_count": CONFIGURATION_COUNT,
        "hpo_configs_run": HPO_CONFIGS,
        "expected_parent_results_total": 51 * CONFIGURATION_COUNT,
        "expected_child_fits_total": 51 * CONFIGURATION_COUNT * 8,
        "ctboost_version": CTBOOST_VERSION,
        "runtime_package_pins": RUNTIME_PACKAGE_PINS,
        "tabarena_repository": TABARENA_REPOSITORY,
        "tabarena_commit": TABARENA_COMMIT,
        "portfolio_200_sha256": PORTFOLIO_200_SHA256,
        "canonical_resources": {
            "num_cpus": 8,
            "memory_limit_gb": 32,
            "time_limit_seconds": 3600,
        },
        "requested_resources": {
            "num_cpus": num_cpus,
            "memory_limit_gb": 28,
            "time_limit_seconds": 3600,
        },
        "resource_disclosure": "Kaggle CPU hardware is non-canonical; do not quote its timings as official.",
        "platform": platform.platform(),
        "python": sys.version,
        "os_cpu_count": os.cpu_count(),
        "worker_sha256": sha256_file(Path(__file__)),
    }
    started = time.time()
    if args.plan_only:
        _, provenance = build_experiments(num_cpus)
        manifest.update(provenance, status="planned")
        write_json(artifacts / "plan.json", manifest)
        print(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "experiments"},
                indent=2,
            )
        )
        return 0

    write_json(artifacts / "manifest.json", manifest)
    failures: list[dict[str, Any]] = []
    fatal_error = None
    benchmark_exit_code = None
    try:
        install_path = artifacts / "installation.json"
        if os.environ.get(BOOTSTRAP_ENV) != "1":
            source = (
                Path(tempfile.gettempdir())
                / f"ctboost-tabarena-hpo0158-source-s{args.shard_index:03d}"
            )
            installation = install_exact_source(source, artifacts)
            installation.update(
                started_at_utc=manifest["started_at_utc"], started_epoch=started
            )
            write_json(install_path, installation)
            environment = os.environ.copy()
            environment[BOOTSTRAP_ENV] = "1"
            os.execve(
                sys.executable,
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                environment,
            )
        installation = json.loads(install_path.read_text(encoding="utf-8"))
        started = installation["started_epoch"]
        manifest.update(installation)
        import ctboost

        installed = {
            name: importlib.metadata.version(name)
            for name in ("ctboost", "tabarena", "tabflow_slurm", *RUNTIME_PACKAGE_PINS)
        }
        expected_versions = {"ctboost": CTBOOST_VERSION, **RUNTIME_PACKAGE_PINS}
        if any(
            installed[name] != version for name, version in expected_versions.items()
        ):
            raise RuntimeError(f"Installed package version mismatch: {installed}")
        manifest["installed_packages"] = installed
        manifest["ctboost_build_info"] = ctboost.build_info()
        if manifest["ctboost_build_info"].get("package_version") != CTBOOST_VERSION:
            raise RuntimeError(
                "Imported CTBoost does not match the installed public release"
            )
        with (artifacts / "pip-freeze.txt").open("w", encoding="utf-8") as stream:
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--all"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        experiments, provenance = build_experiments(num_cpus)
        write_json(artifacts / "experiments.json", provenance)
        manifest.update(
            {key: value for key, value in provenance.items() if key != "experiments"}
        )
        commands, job_json = build_plan(workspace, spec, experiments, num_cpus)
        manifest.update(status="running", run_commands=commands)
        if job_json is not None:
            manifest["generated_job_json"] = job_json.relative_to(root).as_posix()
            jobs = json.loads(job_json.read_text(encoding="utf-8"))
            # Pinned TabArena's own command construction, one fresh process per parent.
            from tabflow_slurm.run_local import _build_item_command

            for item in [item for job in jobs["jobs"] for item in job["items"]]:
                result = run_command(
                    _build_item_command(jobs["defaults"], item), check=False
                )
                if result.returncode:
                    failures.append({"item": item, "exit_code": result.returncode})
                manifest["failures"] = failures
                checkpoint(workspace, artifacts, manifest, validate=True)
                if manifest["validation"]["invalid_results"]:
                    raise RuntimeError(
                        "Generated result validation failed; see manifest validation.invalid_results"
                    )
        benchmark_exit_code = 1 if failures else 0
    except Exception as exc:  # noqa: BLE001 -- publish partial Kaggle output after any worker failure
        fatal_error = f"{type(exc).__name__}: {exc}"
        print(f"FATAL: {fatal_error}", file=sys.stderr, flush=True)
    manifest.update(
        benchmark_exit_code=benchmark_exit_code,
        fatal_error=fatal_error,
        failures=failures,
        finished_at_utc=utc_now(),
        elapsed_seconds=time.time() - started,
    )
    checkpoint(workspace, artifacts, manifest, validate=True)
    complete = (
        fatal_error is None
        and benchmark_exit_code == 0
        and manifest["validation"]["valid_result_count"] == len(spec["datasets"])
        and not manifest["validation"]["invalid_results"]
        and not manifest["validation"]["missing_datasets"]
    )
    manifest["status"] = "complete" if complete else "incomplete"
    write_json(artifacts / "manifest.json", manifest)
    (artifacts / ("SUCCESS.txt" if complete else "INCOMPLETE.txt")).write_text(
        f"status={manifest['status']}\nresults={manifest['result_file_count']}/{len(spec['datasets'])}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    # Kaggle must publish partial artifacts. Manifest status, not kernel exit, certifies success.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
