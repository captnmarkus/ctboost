"""CLI for the frozen grouped-8 TabArena scout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .constants import HISTOGRAM_THREADS, namespace_for_commit, source_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Prerequisite: install the candidate wheel in a fresh environment with "
            "PYTHONDONTWRITEBYTECODE=1 and pip install --no-compile. Invoke every "
            "scout command only as python -I -B "
            "benchmarks/split_research/g8s1_scout_bootstrap.py. See "
            "benchmarks/split_research/G8S1_SCOUT_RUNBOOK.md."
        ),
    )
    parser.add_argument("command", choices=("preflight", "run", "summarize"))
    parser.add_argument("--tabarena-root", type=Path, required=True)
    parser.add_argument("--expected-ctboost-commit", required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=source_root() / "benchmark-results" / "tabarena",
    )
    return parser.parse_args()


def _preflight(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Any]:
    from .identity import collect_provenance, validate_loaded_tabarena_modules

    ctboost_root = source_root().resolve()
    expected_commit = args.expected_ctboost_commit.lower()
    namespace = namespace_for_commit(expected_commit)
    namespace_root = args.results_root.resolve() / namespace
    raw_dir = namespace_root / "raw"
    report_dir = namespace_root / "report"
    provenance = collect_provenance(
        ctboost_root=ctboost_root,
        tabarena_root=args.tabarena_root,
        expected_ctboost_commit=expected_commit,
        expected_native_sha256=args.expected_native_sha256.lower(),
    )
    from .schedule import build_and_validate_schedule, fixed_args

    fixed = fixed_args(stage="run", results_dir=raw_dir, output_dir=report_dir)
    runner, _experiments, _chunks, schedule = build_and_validate_schedule(fixed)
    validate_loaded_tabarena_modules(args.tabarena_root)
    sealed = {**provenance, "schedule": schedule, "namespace": namespace}
    return sealed, schedule, raw_dir, report_dir, runner


def _prepare_namespace(
    raw_dir: Path,
    report_dir: Path,
    provenance: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    from .identity import write_or_validate_provenance
    from .summary import namespace_inventory, validate_namespace

    manifest = report_dir / "scout_provenance.json"
    preliminary = namespace_inventory(
        raw_dir=raw_dir, report_dir=report_dir, phase=phase
    )
    if preliminary["stale_or_unexpected"]:
        raise RuntimeError(
            "sealed scout namespace contains stale, unexpected, or linked files"
        )
    if not manifest.exists():
        if phase != "run_input":
            return validate_namespace(
                raw_dir=raw_dir, report_dir=report_dir, phase=phase
            )
        if preliminary["raw_file_count"] or preliminary["report_file_count"]:
            raise RuntimeError(
                "new scout namespace is not empty and has no matching provenance seal"
            )
    write_or_validate_provenance(manifest, provenance)
    return validate_namespace(raw_dir=raw_dir, report_dir=report_dir, phase=phase)


def _run(args: argparse.Namespace) -> int:
    provenance, _schedule, raw_dir, report_dir, runner = _preflight(args)
    _prepare_namespace(raw_dir, report_dir, provenance, phase="run_input")
    os.environ["CTBOOST_HIST_THREADS"] = HISTOGRAM_THREADS

    from .models import build_generators
    from .schedule import (
        build_frozen_experiments,
        experiment_models,
        fixed_args,
        validate_job_chunks,
    )

    fixed = fixed_args(stage="run", results_dir=raw_dir, output_dir=report_dir)
    original_run_job_shard = runner._run_job_shard

    def guarded_run_job_shard(
        context: Any, job_chunks: Any, **kwargs: Any
    ) -> dict[str, int]:
        chunks = [list(chunk) for chunk in job_chunks]
        validate_job_chunks(context, chunks)
        if kwargs != {
            "results_dir": raw_dir,
            "shard_count": 1,
            "shard_index": 0,
            "job_batch_size": 8,
            "use_ray": False,
        }:
            raise RuntimeError(
                "runner dispatch arguments differ from the frozen resource contract"
            )
        return original_run_job_shard(context, chunks, **kwargs)

    runner._parse_args = lambda: fixed
    runner._experiment_models = experiment_models
    runner._build_experiments = build_frozen_experiments
    runner.gen_ctboost_cpu = build_generators()[0]
    runner._run_job_shard = guarded_run_job_shard
    status = int(runner.main())
    if status == 0:
        from .summary import validate_namespace

        validate_namespace(raw_dir=raw_dir, report_dir=report_dir, phase="run_output")
    return status


def _summarize(args: argparse.Namespace) -> int:
    provenance, schedule, raw_dir, report_dir, _runner = _preflight(args)
    namespace_state = _prepare_namespace(
        raw_dir, report_dir, provenance, phase="summarize_input"
    )
    # The pinned Windows environment has no g++; tell TabArena to use its
    # deterministic Python fallbacks instead of attempting optional compilation.
    os.environ.setdefault("TABARENA_SKIP_FAST_RMSE", "1")
    os.environ.setdefault("TABARENA_SKIP_FAST_ROC_AUC", "1")
    from .summary import summarize, validate_namespace, write_failure_summary

    sanitized = report_dir / "sanitized"
    try:
        summary = summarize(
            raw_dir=raw_dir,
            output_dir=sanitized,
            provenance=provenance,
            schedule=schedule,
            namespace_state=namespace_state,
        )
    except Exception as error:
        write_failure_summary(
            output_dir=sanitized,
            provenance=provenance,
            schedule=schedule,
            error=error,
        )
        validate_namespace(
            raw_dir=raw_dir, report_dir=report_dir, phase="summary_failure"
        )
        raise
    validate_namespace(raw_dir=raw_dir, report_dir=report_dir, phase="summary_success")
    print(json.dumps(summary["decision_evaluation"], indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        return _run(args)
    if args.command == "summarize":
        return _summarize(args)
    provenance, schedule, raw_dir, report_dir, _runner = _preflight(args)
    namespace_state = _prepare_namespace(
        raw_dir, report_dir, provenance, phase="preflight"
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "namespace": provenance["namespace"],
                "identity_sha256": provenance["identity_sha256"],
                "schedule": schedule,
                "raw_artifacts_present": namespace_state["observed_outer_artifacts"],
                "report_exists": report_dir.exists(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
