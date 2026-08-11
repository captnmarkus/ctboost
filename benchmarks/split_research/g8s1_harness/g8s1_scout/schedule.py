"""Construct and validate the exact 306-job frozen TabArena schedule."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    DATASETS,
    EXPECTED_ARTIFACTS,
    EXPECTED_CHUNKS,
    EXPECTED_SCHEDULE_SHA256,
    JOB_BATCH_SIZE,
    MEMORY_LIMIT_GB,
    NUM_CONFIGS,
    NUM_CPUS,
    NUM_GPUS,
    NUM_RANDOM_CONFIGS,
    TASKS,
    TIME_LIMIT_SECONDS,
    TREATMENTS,
    config_id,
    experiment_name,
)
from .loader import load_benchmark_module
from .models import build_generators, canonical_json_bytes, paired_configs


def fixed_args(
    *, stage: str, results_dir: Path, output_dir: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        stage=stage,
        subset="lite",
        datasets=list(DATASETS),
        n_configs=NUM_RANDOM_CONFIGS,
        results_dir=results_dir,
        output_dir=output_dir,
        ray=False,
        shard_count=1,
        shard_index=0,
        job_batch_size=JOB_BATCH_SIZE,
        allow_incomplete=False,
        device="cpu",
        rerun_competitors=False,
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        memory_limit_gb=MEMORY_LIMIT_GB,
        time_limit=TIME_LIMIT_SECONDS,
    )


def experiment_models(
    _args: argparse.Namespace, *, device: str | None = None
) -> list[tuple[Any, int]]:
    if device not in (None, "cpu"):
        raise RuntimeError("frozen scout supports CPU experiments only")
    quadratic, grouped = build_generators()
    return [(quadratic, NUM_RANDOM_CONFIGS), (grouped, NUM_RANDOM_CONFIGS)]


def build_frozen_experiments(
    args: argparse.Namespace, bundle_cls: Any
) -> tuple[list[Any], int]:
    """Build the CPU bundle with sequential folds so no Ray worker can be used."""
    bundle = bundle_cls(
        models=experiment_models(args, device="cpu"),
        sequential_local_fold_fitting=True,
    )
    effective_time_limit = TIME_LIMIT_SECONDS
    experiments = bundle.build_experiments(
        time_limit=effective_time_limit,
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        memory_limit=MEMORY_LIMIT_GB,
    )
    return experiments, effective_time_limit


def _strip_runtime_keys(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"ag_args", "ag_args_ensemble"} and not key.startswith("ag.")
    }


def validate_experiments(experiments: list[Any]) -> None:
    expected_configs = paired_configs()
    expected_names = [
        experiment_name(treatment, index)
        for treatment in ("quadratic", "grouped")
        for index in range(NUM_CONFIGS)
    ]
    observed_names = [str(experiment.name) for experiment in experiments]
    if observed_names != expected_names:
        raise RuntimeError(
            "experiment names/order differ from the frozen paired P50 schedule"
        )

    for position, experiment in enumerate(experiments):
        treatment = "quadratic" if position < NUM_CONFIGS else "grouped"
        index = position % NUM_CONFIGS
        local = dict(experiment._locals)
        expected_identity = TREATMENTS[treatment]
        model_cls = local.get("model_cls")
        if getattr(model_cls, "ag_key", None) != expected_identity["ag_key"]:
            raise RuntimeError(f"{experiment.name}: model ag_key drift")
        if getattr(model_cls, "ag_name", None) != expected_identity["ag_name"]:
            raise RuntimeError(f"{experiment.name}: model ag_name drift")
        if getattr(model_cls, "__module__", None) != "g8s1_scout.models":
            raise RuntimeError(
                f"{experiment.name}: model class is not subprocess-importable"
            )
        if getattr(model_cls, "__name__", None) != expected_identity["model_class"]:
            raise RuntimeError(f"{experiment.name}: model class identity drift")

        config = dict(local.get("model_hyperparameters", {}))
        if _strip_runtime_keys(config) != expected_configs[treatment][index]:
            raise RuntimeError(f"{experiment.name}: effective treatment config drift")
        ag_args = config.get("ag_args", {})
        expected_suffix = f"_{config_id(index)}_default"
        if ag_args != {"name_suffix": expected_suffix}:
            raise RuntimeError(f"{experiment.name}: config suffix drift")
        ag_ensemble = config.get("ag_args_ensemble", {})
        if ag_ensemble != {
            "model_random_seed": index * 8,
            "vary_seed_across_folds": True,
            "fold_fitting_strategy": "sequential_local",
        }:
            raise RuntimeError(
                f"{experiment.name}: fold/config-wise seed contract drift"
            )
        if "random_seed" in _strip_runtime_keys(config):
            raise RuntimeError(f"{experiment.name}: model seed was added to P50")

        if (
            int(local.get("num_bag_folds", -1)) != 8
            or int(local.get("num_bag_sets", -1)) != 1
        ):
            raise RuntimeError(f"{experiment.name}: eight-fold bag contract drift")
        if int(local.get("time_limit", -1)) != TIME_LIMIT_SECONDS:
            raise RuntimeError(f"{experiment.name}: time-limit contract drift")
        if local.get("preprocessing_pipeline") is not None:
            raise RuntimeError(f"{experiment.name}: TabArena-v0.1 preprocessing drift")
        if bool(local.get("dynamic_tabarena_validation_protocol", True)):
            raise RuntimeError(
                f"{experiment.name}: TabArena-v0.1 validation protocol drift"
            )

        method_kwargs = dict(local.get("method_kwargs", {}))
        if bool(method_kwargs.get("shuffle_features", True)):
            raise RuntimeError(
                f"{experiment.name}: TabArena-v0.1 feature-order contract drift"
            )
        fit_kwargs = dict(method_kwargs.get("fit_kwargs", {}))
        expected_resources = {
            "num_cpus": NUM_CPUS,
            "num_gpus": NUM_GPUS,
            "memory_limit": MEMORY_LIMIT_GB,
        }
        if {
            key: fit_kwargs.get(key) for key in expected_resources
        } != expected_resources:
            raise RuntimeError(f"{experiment.name}: fit-resource contract drift")


def _pinned_task_triple(task: Any) -> tuple[str, int, int]:
    try:
        state = vars(task)
    except TypeError as error:
        raise RuntimeError("pinned TabArena Task API shape drift") from error
    if set(state) != {"dataset", "fold", "repeat"} or hasattr(task, "sample"):
        raise RuntimeError("pinned TabArena Task API shape drift")
    as_triple = getattr(task, "as_triple", None)
    if not callable(as_triple):
        raise TypeError("pinned TabArena Task API is missing as_triple")
    coordinates = as_triple()
    expected = (state["dataset"], state["fold"], state["repeat"])
    if type(coordinates) is not tuple or coordinates != expected:
        raise RuntimeError("pinned TabArena Task triple drift")
    dataset, fold, repeat = coordinates
    if type(dataset) is not str or type(fold) is not int or type(repeat) is not int:
        raise TypeError("pinned TabArena Task coordinate type drift")
    return dataset, fold, repeat


def validate_job_chunks(context: Any, chunks: Iterable[list[Any]]) -> dict[str, Any]:
    materialized = [list(chunk) for chunk in chunks]
    chunk_sizes = tuple(len(chunk) for chunk in materialized)
    if chunk_sizes != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"expected frozen job chunks {EXPECTED_CHUNKS}, observed {chunk_sizes}"
        )
    jobs = [job for chunk in materialized for job in chunk]
    if len(jobs) != EXPECTED_ARTIFACTS:
        raise RuntimeError(f"expected {EXPECTED_ARTIFACTS} jobs, observed {len(jobs)}")
    dataset_to_tid = {
        str(dataset): int(tid)
        for dataset, tid in context.task_metadata_collection.dataset_to_tid().items()
    }
    task_lookup = {dataset: task_id for task_id, dataset, _problem, _metric in TASKS}
    expected = {
        (experiment_name(treatment, index), task_id, 0, 0, 0)
        for treatment in ("quadratic", "grouped")
        for index in range(NUM_CONFIGS)
        for task_id, _dataset, _problem, _metric in TASKS
    }
    observed: Counter[tuple[str, int, int, int, int]] = Counter()
    schedule_rows: list[dict[str, Any]] = []
    for job in jobs:
        dataset, fold, repeat = _pinned_task_triple(job.task)
        if (
            dataset not in task_lookup
            or dataset_to_tid.get(dataset) != task_lookup[dataset]
        ):
            raise RuntimeError(f"unexpected TabArena task in schedule: {dataset}")
        sample = 0
        if repeat != 0 or fold != 0 or sample != 0:
            raise RuntimeError("scout schedule contains a nonzero repeat/fold/sample")
        method = str(job.experiment.name)
        key = (method, task_lookup[dataset], repeat, fold, sample)
        observed[key] += 1
        schedule_rows.append(
            {
                "method": method,
                "task_id": task_lookup[dataset],
                "dataset": dataset,
                "repeat": repeat,
                "fold": fold,
                "sample": sample,
            }
        )
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        raise RuntimeError(
            "scout schedule identities differ from the frozen 306-job graph"
        )
    schedule_rows.sort(
        key=lambda row: (
            row["method"],
            row["task_id"],
            row["repeat"],
            row["fold"],
            row["sample"],
        )
    )
    schedule = {
        "jobs": EXPECTED_ARTIFACTS,
        "schedule_sha256": hashlib.sha256(
            canonical_json_bytes(schedule_rows)
        ).hexdigest(),
        "chunks": list(chunk_sizes),
    }
    if schedule["schedule_sha256"] != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            "scout schedule hash drift: "
            f"expected {EXPECTED_SCHEDULE_SHA256}, observed {schedule['schedule_sha256']}"
        )
    return schedule


def build_and_validate_schedule(
    args: argparse.Namespace,
) -> tuple[Any, list[Any], list[list[Any]], dict[str, Any]]:
    runner = load_benchmark_module("run")
    runner._experiment_models = experiment_models
    runner._build_experiments = build_frozen_experiments
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    experiments, effective_time_limit = build_frozen_experiments(
        args, TabArenaV0pt1ExperimentBundle
    )
    if effective_time_limit != TIME_LIMIT_SECONDS:
        raise RuntimeError("effective TabArena time limit drift")
    validate_experiments(experiments)
    context = TabArenaContext()
    scoped = runner._scoped_dataset_names(
        context, subset="lite", datasets=list(DATASETS)
    )
    if set(scoped) != set(DATASETS) or len(scoped) != len(DATASETS):
        raise RuntimeError(f"resolved TabArena task scope drift: {scoped}")
    chunks = list(
        runner._iter_dataset_job_chunks(
            context,
            experiments,
            subset="lite",
            dataset_names=scoped,
        )
    )
    schedule = validate_job_chunks(context, chunks)
    return runner, experiments, chunks, schedule
