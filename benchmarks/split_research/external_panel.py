"""CLI and orchestration for the frozen OpenML split-statistics panel.

``metadata`` and ``preflight`` never access the network or fit a model.
``run`` downloads frozen OpenML task IDs on first use and then launches one
isolated process per fit; completed full-identity jobs resume atomically.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from ._external_panel_data import (
    atomic_write_json,
    collect_source_identity,
    load_json,
    probe_native_feature_test_api,
    stage_openml_payload,
)
from ._external_panel_protocol import (
    DATASETS,
    HISTOGRAM_THREADS,
    RESULT_SCHEMA_VERSION,
    assert_no_absolute_paths,
    build_jobs,
    fit_seed,
    identity_digest,
    make_inner_validation_split,
    planned_fit_descriptors,
    protocol_manifest,
    redact_absolute_paths,
    seal_record,
    sha256_json,
    treatment_order,
    validate_frozen_job_config,
    validate_full_job_identity,
    validate_record_seal,
    validate_self_hashed_identity,
)
from ._external_panel_results import (
    load_ledger,
    store_result,
    summarize_results,
    validate_ledger_entries,
)
from ._external_panel_worker import failure_result, worker_main

__all__ = [
    "DATASETS",
    "build_jobs",
    "build_preflight_report",
    "fit_seed",
    "main",
    "make_inner_validation_split",
    "planned_fit_descriptors",
    "protocol_manifest",
    "run_panel",
    "summarize_cached_results",
    "treatment_order",
]


def build_preflight_report(
    source_collector: Callable[[], Dict[str, Any]] = collect_source_identity,
    feature_probe: Callable[[], Dict[str, Any]] = probe_native_feature_test_api,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
) -> Dict[str, Any]:
    """Validate local readiness without touching OpenML tasks or fitting."""
    dependencies = {
        name: find_spec(name) is not None
        for name in ("numpy", "pandas", "sklearn", "openml", "ctboost")
    }
    errors = []
    if not all(dependencies.values()):
        missing = sorted(name for name, present in dependencies.items() if not present)
        errors.append("missing dependencies: {}".format(", ".join(missing)))
    source_identity = None
    native_probe = None
    if dependencies.get("ctboost"):
        try:
            source_identity = source_collector()
            native_probe = feature_probe()
        except Exception as exc:
            errors.append(
                redact_absolute_paths(
                    "native/source probe failed: {}: {}".format(type(exc).__name__, exc)
                )
            )
    schedule = planned_fit_descriptors()
    manifest = protocol_manifest()
    if len(schedule) != manifest["expected_counts"]["total_subprocess_fits"]:
        errors.append("schedule count does not match frozen protocol")
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": "preflight",
        "network_accessed": False,
        "model_fit_started": False,
        "ready": not errors,
        "errors": errors,
        "dependencies": dependencies,
        "source_identity": source_identity,
        "native_feature_test_api": native_probe,
        "protocol_sha256": manifest["protocol_sha256"],
        "planned_fit_count": len(schedule),
        "histogram_threads_per_fit": HISTOGRAM_THREADS,
    }
    assert_no_absolute_paths(report)
    return report


def _public_run_manifest(
    jobs: Sequence[Mapping[str, Any]],
    source_identity: Mapping[str, Any],
    staged_payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    for job in jobs:
        validate_full_job_identity(job)
    manifest = seal_record(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol": protocol_manifest(),
            "source_identity": dict(source_identity),
            "data_identities": [
                dict(payload["data_identity"]) for payload in staged_payloads
            ],
            "jobs": [
                {
                    "job_key": job["job_key"],
                    "source_identity_sha256": job["source_identity"][
                        "source_identity_sha256"
                    ],
                    "data_identity_sha256": job["data_identity"][
                        "data_identity_sha256"
                    ],
                    "config_identity_sha256": sha256_json(job["config"]),
                    "config": job["config"],
                }
                for job in jobs
            ],
        },
        "manifest_sha256",
    )
    assert_no_absolute_paths(manifest)
    return manifest


def _validate_public_run_manifest(manifest: Mapping[str, Any]) -> None:
    validate_record_seal(manifest, "manifest_sha256")
    if manifest.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("run manifest has an unsupported schema version")
    if manifest.get("protocol") != protocol_manifest():
        raise ValueError("run manifest does not match the frozen protocol")
    source_identity = manifest.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("run manifest source identity is missing")
    validate_self_hashed_identity(source_identity, "source_identity_sha256")
    data_identities = manifest.get("data_identities")
    if not isinstance(data_identities, list):
        raise ValueError("run manifest data identities must be a list")
    if len(data_identities) != len(DATASETS):
        raise ValueError("run manifest does not cover every frozen OpenML task")
    expected_task_ids = [int(dataset["task_id"]) for dataset in DATASETS]
    actual_task_ids = []
    data_by_task = {}
    for data_identity in data_identities:
        if not isinstance(data_identity, Mapping):
            raise ValueError("run manifest data identity must be an object")
        validate_self_hashed_identity(data_identity, "data_identity_sha256")
        task_id = int(data_identity.get("openml", {}).get("task_id", -1))
        actual_task_ids.append(task_id)
        data_by_task[task_id] = data_identity
    if actual_task_ids != expected_task_ids or len(data_by_task) != len(DATASETS):
        raise ValueError("run manifest data identities differ from frozen task order")

    entries = manifest.get("jobs")
    if not isinstance(entries, list):
        raise ValueError("run manifest jobs must be a list")
    expected_schedule = planned_fit_descriptors()
    actual_schedule = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("run manifest job entries must be objects")
        config = entry.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("run manifest job config must be an object")
        config_sha256 = sha256_json(config)
        if entry.get("config_identity_sha256") != config_sha256:
            raise ValueError("run manifest config identity mismatch")
        task_id = int(config.get("task_id", -1))
        data_identity = data_by_task.get(task_id)
        if data_identity is None:
            raise ValueError("run manifest job refers to an unknown task")
        validate_frozen_job_config(config, data_identity)
        if (
            entry.get("source_identity_sha256")
            != source_identity["source_identity_sha256"]
        ):
            raise ValueError("run manifest job source identity mismatch")
        if entry.get("data_identity_sha256") != data_identity["data_identity_sha256"]:
            raise ValueError("run manifest job data identity mismatch")
        expected_job_key = identity_digest(source_identity, data_identity, config)
        if entry.get("job_key") != expected_job_key:
            raise ValueError("run manifest full job identity mismatch")
        actual_schedule.append(
            {
                "task_id": config.get("task_id"),
                "fold": config.get("fold"),
                "profile": config.get("profile"),
                "treatment": config.get("treatment"),
                "role": config.get("role"),
                "order_in_pair": config.get("order_in_pair"),
            }
        )
    if actual_schedule != expected_schedule:
        raise ValueError("run manifest job schedule differs from the frozen protocol")
    keys = [entry.get("job_key") for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("run manifest contains duplicate job identities")
    assert_no_absolute_paths(manifest)


def _jobs_from_public_manifest(manifest: Mapping[str, Any]) -> Sequence[Dict[str, Any]]:
    source_identity = manifest["source_identity"]
    data_by_task = {
        int(value["openml"]["task_id"]): value for value in manifest["data_identities"]
    }
    return [
        {
            "job_key": entry["job_key"],
            "source_identity": source_identity,
            "data_identity": data_by_task[int(entry["config"]["task_id"])],
            "config": entry["config"],
        }
        for entry in manifest["jobs"]
    ]


def run_panel(
    results_dir: Path,
    cache_dir: Path,
    *,
    rerun_failures: bool = False,
    openml_module: Any = None,
    subprocess_runner: Callable[..., Any] = subprocess.run,
    source_collector: Callable[[], Dict[str, Any]] = collect_source_identity,
    feature_probe: Callable[[], Dict[str, Any]] = probe_native_feature_test_api,
) -> Dict[str, Any]:
    """Stage data, then atomically resume the fixed schedule in isolated workers."""
    results_dir = Path(results_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if openml_module is None:
        openml_module = importlib.import_module("openml")
    if hasattr(openml_module, "config"):
        openml_module.config.cache_directory = str(cache_dir / "openml")

    source_identity = source_collector()
    feature_probe()
    staged_payloads = [
        stage_openml_payload(cache_dir, dataset, openml_module) for dataset in DATASETS
    ]
    jobs = build_jobs(staged_payloads, source_identity)
    manifest = _public_run_manifest(jobs, source_identity, staged_payloads)
    manifest_path = results_dir / "run_manifest.json"
    _validate_public_run_manifest(manifest)
    if manifest_path.exists():
        existing_manifest = load_json(manifest_path)
        _validate_public_run_manifest(existing_manifest)
        if existing_manifest != manifest:
            raise ValueError(
                "existing run manifest belongs to a different full run identity"
            )
    else:
        atomic_write_json(manifest_path, manifest)
    ledger_path = results_dir / "results.json"
    protocol_sha256 = manifest["protocol"]["protocol_sha256"]
    ledger = load_ledger(ledger_path, protocol_sha256)
    validate_ledger_entries(ledger, jobs)
    private_dir = results_dir / ".private"
    private_dir.mkdir(parents=True, exist_ok=True)

    for ordinal, job in enumerate(jobs, start=1):
        previous = ledger["jobs"].get(job["job_key"])
        if previous is not None and (
            previous.get("status") == "success" or not rerun_failures
        ):
            continue
        job_path = private_dir / "job-{}.json".format(job["job_key"])
        output_path = private_dir / "worker-{}.json".format(job["job_key"])
        atomic_write_json(job_path, job)
        if output_path.exists():
            output_path.unlink()
        environment = os.environ.copy()
        environment["CTBOOST_HIST_THREADS"] = str(HISTOGRAM_THREADS)
        command = [
            sys.executable,
            "-m",
            "benchmarks.split_research.external_panel",
            "_worker",
            "--job",
            str(job_path),
            "--output",
            str(output_path),
        ]
        print(
            "[{}/{}] task={} fold={} profile={} treatment={}".format(
                ordinal,
                len(jobs),
                job["config"]["task_id"],
                job["config"]["fold"],
                job["config"]["profile"],
                job["config"]["treatment"],
            ),
            flush=True,
        )
        completed_process = subprocess_runner(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=environment,
            check=False,
        )
        if output_path.exists():
            result = load_json(output_path)
        else:
            result = failure_result(
                job,
                RuntimeError(
                    "worker exited with code {} without an output record".format(
                        completed_process.returncode
                    )
                ),
                record_process_rss=False,
            )
        if result.get("job_key") != job["job_key"]:
            raise RuntimeError("worker output job key mismatch")
        store_result(ledger_path, ledger, result, expected_job=job)
        if output_path.exists():
            output_path.unlink()
        if job_path.exists():
            job_path.unlink()

    summary = summarize_results(
        jobs,
        ledger,
        source_identity,
        [payload["data_identity"] for payload in staged_payloads],
    )
    atomic_write_json(results_dir / "summary.json", summary)
    return summary


def summarize_cached_results(results_dir: Path) -> Dict[str, Any]:
    """Summarize existing artifacts without importing or contacting OpenML."""
    results_dir = Path(results_dir).expanduser().resolve()
    manifest = load_json(results_dir / "run_manifest.json")
    protocol_sha256 = manifest.get("protocol", {}).get("protocol_sha256")
    if protocol_sha256 != protocol_manifest()["protocol_sha256"]:
        raise ValueError("run manifest does not match the frozen protocol")
    _validate_public_run_manifest(manifest)
    jobs = _jobs_from_public_manifest(manifest)
    ledger = load_ledger(results_dir / "results.json", protocol_sha256)
    summary = summarize_results(
        jobs,
        ledger,
        manifest["source_identity"],
        manifest["data_identities"],
    )
    atomic_write_json(results_dir / "summary.json", summary)
    return summary


def _write_or_print(value: Mapping[str, Any], output: Optional[Path]) -> None:
    assert_no_absolute_paths(value)
    if output is None:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    else:
        atomic_write_json(Path(output), value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    metadata = subparsers.add_parser(
        "metadata",
        help="emit the protocol without importing CTBoost/OpenML or accessing data",
    )
    metadata.add_argument("--output", type=Path)
    preflight = subparsers.add_parser(
        "preflight",
        help="validate dependencies/native API without network access or fitting",
    )
    preflight.add_argument("--output", type=Path)
    run = subparsers.add_parser(
        "run",
        help="cache frozen OpenML tasks as needed and resume isolated fits",
    )
    run.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark-results/split_research/external-panel"),
    )
    run.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("benchmark-results/split_research/openml-cache"),
    )
    run.add_argument(
        "--rerun-failures",
        action="store_true",
        help="retry full-identity jobs already recorded as failures",
    )
    summarize = subparsers.add_parser(
        "summarize",
        help="regenerate a public summary from an existing ledger without OpenML",
    )
    summarize.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark-results/split_research/external-panel"),
    )
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "metadata":
        value = protocol_manifest()
        value["schedule_sha256"] = sha256_json(planned_fit_descriptors())
        _write_or_print(value, args.output)
        return 0
    if args.command == "preflight":
        report = build_preflight_report()
        _write_or_print(report, args.output)
        return 0 if report["ready"] else 2
    if args.command == "run":
        summary = run_panel(
            args.results_dir,
            args.cache_dir,
            rerun_failures=bool(args.rerun_failures),
        )
        print(json.dumps(summary["coverage"], indent=2, sort_keys=True))
        return 0 if summary["coverage"]["failed_jobs"] == 0 else 1
    if args.command == "summarize":
        summary = summarize_cached_results(args.results_dir)
        print(json.dumps(summary["decision_aggregation"], indent=2, sort_keys=True))
        return 0
    if args.command == "_worker":
        return worker_main(args.job, args.output)
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
