"""Run CTBoost on official TabArena folds and compare with cached baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import unquote, urlparse

from .ctboost_model import (
    TABARENA_SEARCH_PORTFOLIO_SIZE,
    gen_ctboost_cpu,
    gen_ctboost_gpu,
)

DEFAULT_LITE_DATASETS = [
    "blood-transfusion-service-center",
    "QSAR_fish_toxicity",
    "anneal",
]

_RESOURCE_FIELDS = (
    "method",
    "dataset",
    "task_id",
    "fold",
    "repeat",
    "sample",
    "problem_type",
    "metric",
    "metric_error",
    "time_train_s",
    "time_infer_s",
    "peak_mem_cpu_bytes",
    "incremental_peak_mem_cpu_bytes",
    "peak_mem_gpu_bytes",
    "incremental_peak_mem_gpu_bytes",
    "gpu_tracking_enabled",
    "num_cpus",
    "num_gpus",
    "disk_usage_bytes",
    "artifact",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "run", "evaluate"),
        help=(
            "Run jobs and evaluate them, run one resumable job shard only, or "
            "evaluate already-cached raw artifacts."
        ),
    )
    parser.add_argument(
        "--subset",
        default="lite",
        choices=("lite", "all"),
        help="Use three quick datasets or the complete TabArena-v0.1 task set.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Optional dataset filter; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--n-configs",
        type=int,
        default=0,
        help="Number of frozen HPO portfolio configurations in addition to the default.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark-results/tabarena/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/tabarena/report"))
    parser.add_argument(
        "--ray",
        action="store_true",
        help=(
            "Use TabArena's Ray fold/evaluation backend. Outer benchmark jobs are "
            "distributed with --shard-count/--shard-index, not by this flag."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of disjoint outer-job shards (use with --stage run).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based outer-job shard to execute (use with --stage run).",
    )
    parser.add_argument(
        "--job-batch-size",
        type=int,
        default=32,
        help="Maximum raw result objects retained by the run driver at once.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Evaluate partial raw coverage instead of rejecting it (not leaderboard-valid).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "gpu", "both"),
        help="Benchmark the CPU model, the CUDA model, or both as distinct methods.",
    )
    parser.add_argument(
        "--rerun-competitors",
        action="store_true",
        help="Rerun CatBoost and XGBoost on the same folds instead of relying only on cached baselines.",
    )
    parser.add_argument("--num-cpus", type=int, help="CPU limit supplied to every TabArena fit.")
    parser.add_argument("--num-gpus", type=int, help="GPU limit supplied to the experiment bundle.")
    parser.add_argument("--memory-limit-gb", type=int, help="RAM limit supplied to TabArena.")
    parser.add_argument("--time-limit", type=int, help="Per-fit time limit in seconds.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_configs < 0:
        raise SystemExit("--n-configs must be non-negative")
    if args.n_configs > TABARENA_SEARCH_PORTFOLIO_SIZE:
        raise SystemExit(
            "--n-configs cannot exceed the frozen "
            f"{TABARENA_SEARCH_PORTFOLIO_SIZE}-configuration portfolio"
        )
    if args.shard_count <= 0:
        raise SystemExit("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= index < --shard-count")
    if args.job_batch_size <= 0:
        raise SystemExit("--job-batch-size must be positive")
    if args.stage == "all" and args.shard_count != 1:
        raise SystemExit(
            "--stage all requires --shard-count 1; run every shard with --stage run, "
            "then invoke --stage evaluate"
        )
    if args.stage == "evaluate" and args.shard_index != 0:
        raise SystemExit("--shard-index is only meaningful with --stage run")
    for name in ("num_cpus", "num_gpus", "memory_limit_gb", "time_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0 and name != "num_gpus":
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if name == "num_gpus" and value is not None and value < 0:
            raise SystemExit("--num-gpus must be non-negative")


def _resolve_effective_time_limit(requested: Optional[int], experiment_bundle: Any) -> int:
    """Match TabArena's bundle default while retaining the CLI value separately."""
    if requested is not None:
        return int(requested)
    return int(experiment_bundle.DEFAULT_TIME_LIMIT)


def _format_table(frame: Any) -> str:
    """Render a leaderboard without making pandas' ``tabulate`` extra mandatory."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)


def _write_console_table(value: str, stream: Optional[Any] = None) -> None:
    """Write a leaderboard even when the host console has a legacy encoding."""
    output = sys.stdout if stream is None else stream
    encoding = getattr(output, "encoding", None)
    if encoding:
        value = value.encode(encoding, errors="replace").decode(encoding)
    output.write(value)
    if not value.endswith("\n"):
        output.write("\n")


def _distribution_source(name: str) -> Optional[Dict[str, Any]]:
    try:
        distribution = importlib.metadata.distribution(name)
        direct_url = distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    try:
        return json.loads(direct_url)
    except json.JSONDecodeError:
        return {"raw": direct_url}


def _local_distribution_checkout(name: str) -> Optional[Path]:
    """Resolve a PEP 610 local-source install back to its checkout directory."""
    source = _distribution_source(name)
    if not isinstance(source, dict):
        return None
    raw_url = source.get("url")
    if not isinstance(raw_url, str):
        return None
    parsed = urlparse(raw_url)
    if parsed.scheme.lower() != "file":
        return None
    path_text = unquote(parsed.path)
    if parsed.netloc:
        path_text = f"//{parsed.netloc}{path_text}"
    if os.name == "nt" and len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
        path_text = path_text[1:]
    checkout = Path(path_text).resolve()
    return checkout if checkout.is_dir() else None


def _ctboost_install_fingerprint() -> str:
    import ctboost

    package_root = Path(ctboost.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in package_root.rglob("*")
        if item.is_file() and (item.suffix == ".py" or item.name.startswith("_core."))
    ):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_source_identity(distribution_name: str = "ctboost") -> Optional[Dict[str, Any]]:
    """Describe an installed distribution's source checkout, when available."""
    candidates: list[tuple[Path, bool]] = []
    local_checkout = _local_distribution_checkout(distribution_name)
    if local_checkout is not None:
        # PEP 610 identifies the runtime package source. Editable packages such
        # as TabArena live below their repository root, so parent ascent is safe
        # only for this explicit source path.
        candidates.append((local_checkout, True))
    if distribution_name == "ctboost":
        adapter_root = Path(__file__).resolve().parents[2]
        if adapter_root != local_checkout and (adapter_root / ".git").exists():
            # Source-tree fallback. Never ascend from site-packages: a virtual
            # environment can itself live inside an unrelated Git checkout.
            candidates.append((adapter_root, False))

    repository = None
    for candidate, allow_parent_ascent in candidates:
        possible_roots = (candidate, *candidate.parents) if allow_parent_ascent else (candidate,)
        for possible_root in possible_roots:
            if (possible_root / ".git").exists():
                repository = possible_root
                break
        if repository is not None:
            break
    if repository is None:
        return None

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    revision = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if revision.returncode != 0 or status.returncode != 0:
        return None
    status_text = status.stdout
    diff_fingerprint = None
    if status_text:
        diff = git("diff", "--binary", "HEAD")
        untracked = git("ls-files", "--others", "--exclude-standard")
        if diff.returncode == 0 and untracked.returncode == 0:
            digest = hashlib.sha256()
            digest.update(diff.stdout.encode("utf-8"))
            for relative_name in sorted(untracked.stdout.splitlines()):
                path = repository / relative_name
                if path.is_file():
                    digest.update(relative_name.encode("utf-8"))
                    digest.update(path.read_bytes())
            diff_fingerprint = digest.hexdigest()
    return {
        "commit": revision.stdout.strip(),
        "dirty": bool(status_text),
        "dirty_fingerprint_sha256": diff_fingerprint,
        "status": status_text.splitlines(),
    }


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    *,
    status: str,
    run_stats: Optional[Dict[str, Any]] = None,
    coverage: Optional[Dict[str, Any]] = None,
) -> None:
    import ctboost

    package_versions = {}
    for name in ("ctboost", "tabarena", "autogluon.tabular", "numpy", "pandas"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    manifest = {
        "schema_version": 1,
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "TabArena-v0.1",
        "stage": getattr(args, "stage", "all"),
        "subset": args.subset,
        "datasets": args.datasets,
        "n_random_configs": args.n_configs,
        "ray_backend": bool(args.ray),
        "sharding": {
            "count": getattr(args, "shard_count", 1),
            "index": getattr(args, "shard_index", 0),
            "job_batch_size": getattr(args, "job_batch_size", None),
        },
        "allow_incomplete": bool(getattr(args, "allow_incomplete", False)),
        "device": args.device,
        "rerun_competitors": bool(args.rerun_competitors),
        "resources": {
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
            "memory_limit_gb": args.memory_limit_gb,
            "requested_time_limit_seconds": args.time_limit,
            "time_limit_seconds": getattr(args, "effective_time_limit", args.time_limit),
        },
        "results_dir": str(args.results_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "package_versions": package_versions,
        "ctboost_build": ctboost.build_info(),
        "ctboost_install_sha256": _ctboost_install_fingerprint(),
        "ctboost_git": _git_source_identity(),
        "tabarena_git": _git_source_identity("tabarena"),
        "ctboost_source": _distribution_source("ctboost"),
        "tabarena_source": _distribution_source("tabarena"),
    }
    if run_stats is not None:
        manifest["run_stats"] = run_stats
    if coverage is not None:
        manifest["raw_coverage"] = coverage
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shard_manifest_path(output_dir: Path, *, shard_count: int, shard_index: int) -> Path:
    return output_dir / f"run_manifest.shard-{shard_index:05d}-of-{shard_count:05d}.json"


def _write_resource_summary(
    results_dir: Path,
    output_dir: Path,
    *,
    _raw_loading: Optional[Any] = None,
    file_paths: Optional[Iterable[Path]] = None,
    collect_rows: bool = True,
) -> list[Dict[str, Any]] | int:
    """Export per-split resource records without retaining full-run rows by default."""
    if _raw_loading is None:
        from tabarena.benchmark.result import raw_loading as _raw_loading

    def json_scalar(value: Any) -> Any:
        item = getattr(value, "item", None)
        return item() if callable(item) else value

    paths = (
        list(_raw_loading.fetch_raw_result_paths(results_dir))
        if file_paths is None
        else list(file_paths)
    )
    paths.sort(key=str)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "resources_per_split.json"
    csv_path = output_dir / "resources_per_split.csv"
    rows: list[Dict[str, Any]] = []
    row_count = 0
    with csv_path.open("w", encoding="utf-8", newline="") as csv_stream, json_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as json_stream:
        writer = csv.DictWriter(csv_stream, fieldnames=list(_RESOURCE_FIELDS))
        writer.writeheader()
        json_stream.write("[\n")
        for path in paths:
            artifact = _raw_loading.load_and_align(path)
            result = artifact.result
            task = dict(result.get("task_metadata", {}))
            expected_identity = _raw_result_key(path)
            observed_identity = (
                str(result.get("framework", "")),
                str(json_scalar(task.get("tid"))),
                f"{json_scalar(task.get('repeat'))}_{json_scalar(task.get('fold'))}",
            )
            if observed_identity != expected_identity:
                raise RuntimeError(
                    "Raw TabArena artifact identity does not match its cache path: "
                    f"path={path}, expected={expected_identity}, observed={observed_identity}"
                )
            memory = dict(result.get("memory_usage", {}))
            peak_cpu = json_scalar(memory.get("peak_mem_cpu"))
            minimum_cpu = json_scalar(memory.get("min_mem_cpu"))
            peak_gpu = json_scalar(memory.get("peak_mem_gpu"))
            minimum_gpu = json_scalar(memory.get("min_mem_gpu"))
            method_metadata = dict(result.get("method_metadata", {}))
            row = {
                "method": str(result.get("framework", "")),
                "dataset": str(task.get("name", "")),
                "task_id": json_scalar(task.get("tid")),
                "fold": json_scalar(task.get("fold")),
                "repeat": json_scalar(task.get("repeat")),
                "sample": json_scalar(task.get("sample")),
                "problem_type": str(result.get("problem_type", "")),
                "metric": str(result.get("metric", "")),
                "metric_error": float(result["metric_error"]),
                "time_train_s": float(result["time_train_s"]),
                "time_infer_s": float(result["time_infer_s"]),
                "peak_mem_cpu_bytes": peak_cpu,
                "incremental_peak_mem_cpu_bytes": (
                    None
                    if peak_cpu is None or minimum_cpu is None
                    else int(peak_cpu) - int(minimum_cpu)
                ),
                "peak_mem_gpu_bytes": peak_gpu,
                "incremental_peak_mem_gpu_bytes": (
                    None
                    if peak_gpu is None or minimum_gpu is None
                    else int(peak_gpu) - int(minimum_gpu)
                ),
                "gpu_tracking_enabled": bool(memory.get("gpu_tracking_enabled", False)),
                "num_cpus": json_scalar(method_metadata.get("num_cpus")),
                "num_gpus": json_scalar(method_metadata.get("num_gpus")),
                "disk_usage_bytes": json_scalar(method_metadata.get("disk_usage")),
                "artifact": str(path),
            }
            writer.writerow(row)
            if row_count:
                json_stream.write(",\n")
            json_stream.write("  " + json.dumps(row, sort_keys=True))
            row_count += 1
            if collect_rows:
                rows.append(row)
        json_stream.write("\n]\n")
    return rows if collect_rows else row_count


def _run_job_shard(
    context: Any,
    job_chunks: Iterable[list[Any]],
    *,
    results_dir: Path,
    shard_count: int,
    shard_index: int,
    job_batch_size: int,
    use_ray: bool,
) -> Dict[str, int]:
    """Run one deterministic shard while discarding each bounded result batch."""
    total_jobs = 0
    selected_jobs = 0
    completed_results = 0
    batches = 0
    batch: list[Any] = []

    def flush() -> None:
        nonlocal completed_results, batches
        if not batch:
            return
        results = context.run_jobs(
            list(batch),
            expname=str(results_dir),
            register=False,
            debug_mode=not use_ray,
            cache_mode="default",
        )
        if len(results) != len(batch):
            raise RuntimeError(
                f"TabArena returned {len(results)} results for a {len(batch)}-job batch"
            )
        completed_results += len(results)
        batches += 1
        batch.clear()

    for jobs in job_chunks:
        for job in jobs:
            index = total_jobs
            total_jobs += 1
            if index % shard_count != shard_index:
                continue
            selected_jobs += 1
            batch.append(job)
            if len(batch) >= job_batch_size:
                flush()
    flush()
    return {
        "total_jobs": total_jobs,
        "selected_jobs": selected_jobs,
        "completed_results": completed_results,
        "batches": batches,
    }


def _raw_result_key(path: Path) -> tuple[str, str, str]:
    """Return ``(method, task, split)`` from TabArena's raw cache layout."""
    path = Path(path)
    if path.name != "results.pkl" or len(path.parents) < 3:
        raise ValueError(f"not a TabArena raw result path: {path}")
    return path.parents[2].name, path.parents[1].name, path.parents[0].name


def _select_expected_raw_paths(
    expected: Counter[tuple[str, str, str]],
    file_paths: Iterable[Path],
    *,
    allow_incomplete: bool,
) -> tuple[list[Path], Dict[str, Any]]:
    """Filter unrelated artifacts and require every exact scheduled job key."""
    paths = [Path(path) for path in file_paths]
    keyed_paths = [(path, _raw_result_key(path)) for path in paths]
    selected = [path for path, key in keyed_paths if key in expected]
    observed = Counter(key for _path, key in keyed_paths if key in expected)
    mismatches = [
        {
            "method": key[0],
            "task": key[1],
            "split": key[2],
            "expected": expected[key],
            "observed": observed.get(key, 0),
        }
        for key in sorted(expected)
        if observed.get(key, 0) != expected[key]
    ]
    coverage: Dict[str, Any] = {
        "expected_results": sum(expected.values()),
        "observed_results": len(selected),
        "ignored_stale_or_other_results": len(paths) - len(selected),
        "complete": not mismatches,
        "mismatches": mismatches,
    }
    if mismatches and not allow_incomplete:
        preview = mismatches[:10]
        raise RuntimeError(
            "Raw TabArena coverage is incomplete or duplicated; refusing leaderboard evaluation. "
            f"First mismatches: {preview}. Finish all run shards or pass --allow-incomplete."
        )
    if not selected:
        raise RuntimeError("No raw artifacts match the requested benchmark configuration")
    return sorted(selected, key=str), coverage


def _expected_raw_keys(
    context: Any,
    job_chunks: Iterable[list[Any]],
) -> Counter[tuple[str, str, str]]:
    """Count exact scheduled cache keys without retaining the complete job grid."""
    dataset_to_tid = {
        str(dataset): str(tid)
        for dataset, tid in context.task_metadata_collection.dataset_to_tid().items()
    }
    counts: Counter[tuple[str, str, str]] = Counter()
    for jobs in job_chunks:
        for job in jobs:
            dataset = str(job.task.dataset)
            try:
                task_id = dataset_to_tid[dataset]
            except KeyError as exc:
                raise RuntimeError(
                    f"Scheduled dataset {dataset!r} is missing from TabArena task metadata"
                ) from exc
            key = (
                str(job.experiment.name),
                task_id,
                f"{int(job.task.repeat)}_{int(job.task.fold)}",
            )
            counts[key] += 1
    return counts


def _scoped_dataset_names(
    context: Any,
    *,
    subset: str,
    datasets: Optional[list[str]],
) -> list[str]:
    """Resolve the benchmark scope using metadata only (no dataset materialization)."""
    filters: Dict[str, Any] = {"subset": subset}
    if datasets is not None:
        filters["dataset_names"] = datasets
    collection = context.task_metadata_collection.subset_tasks(
        predicates=context.subset_predicates,
        **filters,
    )
    return collection.dataset_names()


def _iter_dataset_job_chunks(
    context: Any,
    experiments: list[Any],
    *,
    subset: str,
    dataset_names: Iterable[str],
) -> Iterable[list[Any]]:
    """Build at most one dataset's jobs at a time instead of a 164k-job list."""
    for dataset_name in dataset_names:
        yield context.build_jobs(
            experiments,
            subset=subset,
            dataset_names=[dataset_name],
        )


_GPU_COMPETITOR_GENERATORS: Optional[list[Any]] = None


def _gpu_competitor_generators() -> list[Any]:
    global _GPU_COMPETITOR_GENERATORS
    if _GPU_COMPETITOR_GENERATORS is not None:
        return list(_GPU_COMPETITOR_GENERATORS)
    from autogluon.tabular.models import CatBoostModel, XGBoostModel
    from tabarena.models.catboost.hpo import generate_configs_catboost
    from tabarena.models.xgboost.hpo import generate_configs_xgboost
    from tabarena.utils.config_utils import CustomAGConfigGenerator

    class CatBoostGPUModel(CatBoostModel):
        ag_key = "CAT_GPU"
        ag_name = "CatBoostGPU"
        default_num_gpus = 1
        minimum_num_gpus = 1
        gpu_required = True

    class XGBoostGPUModel(XGBoostModel):
        ag_key = "XGB_GPU"
        ag_name = "XGBoostGPU"
        default_num_gpus = 1
        minimum_num_gpus = 1
        gpu_required = True

    # AutoGluon/Ray serializes model classes by module-qualified name. Expose
    # these dynamically imported optional-dependency subclasses at module scope.
    CatBoostGPUModel.__name__ = "TabArenaCatBoostGPUModel"
    CatBoostGPUModel.__qualname__ = CatBoostGPUModel.__name__
    CatBoostGPUModel.__module__ = __name__
    XGBoostGPUModel.__name__ = "TabArenaXGBoostGPUModel"
    XGBoostGPUModel.__qualname__ = XGBoostGPUModel.__name__
    XGBoostGPUModel.__module__ = __name__
    globals()[CatBoostGPUModel.__name__] = CatBoostGPUModel
    globals()[XGBoostGPUModel.__name__] = XGBoostGPUModel

    _GPU_COMPETITOR_GENERATORS = [
        CustomAGConfigGenerator(
            model_cls=CatBoostGPUModel,
            search_space_func=generate_configs_catboost,
            manual_configs=[{}],
        ),
        CustomAGConfigGenerator(
            model_cls=XGBoostGPUModel,
            search_space_func=generate_configs_xgboost,
            manual_configs=[{}],
        ),
    ]
    return list(_GPU_COMPETITOR_GENERATORS)


def _experiment_models(
    args: argparse.Namespace,
    *,
    device: Optional[str] = None,
) -> list[tuple[Any, int]]:
    selected_device = args.device if device is None else device
    models: list[tuple[Any, int]] = []
    if selected_device in {"cpu", "both"}:
        models.append((gen_ctboost_cpu, args.n_configs))
    if selected_device in {"gpu", "both"}:
        models.append((gen_ctboost_gpu, args.n_configs))
    if args.rerun_competitors:
        from tabarena.models.catboost import gen_catboost
        from tabarena.models.xgboost import gen_xgboost

        if selected_device in {"cpu", "both"}:
            models.extend([(gen_catboost, args.n_configs), (gen_xgboost, args.n_configs)])
        if selected_device in {"gpu", "both"}:
            models.extend(
                (generator, args.n_configs)
                for generator in _gpu_competitor_generators()
            )
    return models


def _build_experiments(
    args: argparse.Namespace,
    bundle_cls: Any,
) -> tuple[list[Any], int]:
    """Build CPU and GPU methods with device-specific resource contracts."""
    resource_groups: list[tuple[str, int]] = []
    if args.device in {"cpu", "both"}:
        resource_groups.append(("cpu", 0))
    if args.device in {"gpu", "both"}:
        # Evaluation still needs the same GPU experiment identity even when it
        # consumes already-produced artifacts and no local GPU was requested.
        gpu_count = args.num_gpus if args.num_gpus is not None and args.num_gpus > 0 else 1
        resource_groups.append(("gpu", gpu_count))

    bundles = [
        (bundle_cls(models=_experiment_models(args, device=device)), num_gpus)
        for device, num_gpus in resource_groups
    ]
    effective_time_limit = _resolve_effective_time_limit(args.time_limit, bundles[0][0])
    experiments: list[Any] = []
    for experiment_bundle, num_gpus in bundles:
        experiments.extend(
            experiment_bundle.build_experiments(
                time_limit=effective_time_limit,
                num_cpus=args.num_cpus,
                num_gpus=num_gpus,
                memory_limit=args.memory_limit_gb,
            )
        )
    return experiments, effective_time_limit


def main() -> int:
    if gen_ctboost_cpu is None:
        raise SystemExit(
            "TabArena is not installed. Follow benchmarks/tabarena/README.md before running this module."
        )
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    args = _parse_args()
    _validate_args(args)
    datasets = args.datasets
    if datasets is None and args.subset == "lite":
        datasets = DEFAULT_LITE_DATASETS
    args.datasets = datasets

    if args.stage in {"all", "run"} and args.device in {"gpu", "both"}:
        import ctboost

        if not bool(ctboost.build_info().get("cuda_enabled", False)):
            raise SystemExit(
                "GPU benchmarking requires a CUDA-enabled CTBoost wheel; "
                "install or upgrade the unified wheel with "
                "`python -m pip install --upgrade --only-binary=:all: \"ctboost>=0.1.54\"`"
            )
        if args.num_gpus == 0:
            raise SystemExit("--device gpu/both cannot be combined with --num-gpus 0")

    experiments, args.effective_time_limit = _build_experiments(
        args,
        TabArenaV0pt1ExperimentBundle,
    )

    context = TabArenaContext()
    scoped_dataset_names = _scoped_dataset_names(
        context,
        subset=args.subset,
        datasets=datasets,
    )

    def job_chunks() -> Iterable[list[Any]]:
        return _iter_dataset_job_chunks(
            context,
            experiments,
            subset=args.subset,
            dataset_names=scoped_dataset_names,
        )

    if args.stage in {"all", "run"}:
        shard_manifest = _shard_manifest_path(
            args.output_dir,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        _write_manifest(shard_manifest, args, status="run_started")
        try:
            run_stats = _run_job_shard(
                context,
                job_chunks(),
                results_dir=args.results_dir,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                job_batch_size=args.job_batch_size,
                use_ray=args.ray,
            )
        except Exception:
            _write_manifest(shard_manifest, args, status="run_failed")
            raise
        _write_manifest(
            shard_manifest,
            args,
            status="run_completed",
            run_stats=run_stats,
        )
        if args.stage == "run":
            _write_console_table(json.dumps(run_stats, indent=2, sort_keys=True))
            return 0

    from tabarena.benchmark.result import raw_loading
    from tabarena.end_to_end import EndToEnd

    all_raw_paths = raw_loading.fetch_raw_result_paths(args.results_dir)
    expected_raw_keys = _expected_raw_keys(context, job_chunks())
    selected_raw_paths, coverage = _select_expected_raw_paths(
        expected_raw_keys,
        all_raw_paths,
        allow_incomplete=args.allow_incomplete,
    )
    _write_resource_summary(
        args.results_dir,
        args.output_dir,
        _raw_loading=raw_loading,
        file_paths=selected_raw_paths,
        collect_rows=False,
    )
    processed = EndToEnd.from_path_raw(
        path_raw=args.results_dir,
        file_paths=selected_raw_paths,
        cache=True,
        cache_processed=True,
        backend="ray" if args.ray else "native",
        num_cpus=args.num_cpus,
    )
    new_methods = processed.to_method_metadata_lst(new_result_prefix="[New] ")
    evaluation_context = TabArenaContext(extra_methods=new_methods, only_valid_tasks=True)
    leaderboard = evaluation_context.compare(output_dir=args.output_dir)
    website = evaluation_context.leaderboard_to_website_format(leaderboard=leaderboard)
    _write_manifest(
        args.output_dir / "run_manifest.json",
        args,
        status="completed" if coverage["complete"] else "completed_incomplete",
        coverage=coverage,
    )
    _write_console_table(_format_table(website))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
