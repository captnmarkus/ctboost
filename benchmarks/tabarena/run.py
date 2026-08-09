"""Run CTBoost on official TabArena folds and compare with cached baselines."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from .ctboost_model import gen_ctboost_cpu, gen_ctboost_gpu


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
        help="Number of sampled HPO configurations in addition to the default.",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("benchmark-results/tabarena/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/tabarena/report"))
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Use TabArena's Ray backend; the default runs in-process for easier debugging.",
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
    return Path(path_text).resolve()


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


def _git_source_identity() -> Optional[Dict[str, Any]]:
    """Describe a source checkout without assuming the benchmark runs in one."""
    candidates = [Path(__file__).resolve().parents[2]]
    local_checkout = _local_distribution_checkout("ctboost")
    if local_checkout is not None and local_checkout not in candidates:
        candidates.append(local_checkout)
    repository = next(
        (candidate for candidate in candidates if (candidate / ".git").exists()),
        None,
    )
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


def _write_manifest(path: Path, args: argparse.Namespace, *, status: str) -> None:
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
        "subset": args.subset,
        "datasets": args.datasets,
        "n_random_configs": args.n_configs,
        "ray_backend": bool(args.ray),
        "device": args.device,
        "rerun_competitors": bool(args.rerun_competitors),
        "resources": {
            "num_cpus": args.num_cpus,
            "num_gpus": args.num_gpus,
            "memory_limit_gb": args.memory_limit_gb,
            "time_limit_seconds": args.time_limit,
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
        "ctboost_source": _distribution_source("ctboost"),
        "tabarena_source": _distribution_source("tabarena"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_resource_summary(
    results_dir: Path,
    output_dir: Path,
    *,
    _raw_loading: Optional[Any] = None,
) -> list[Dict[str, Any]]:
    """Export TabArena's per-split timing and CPU/GPU peak-memory records."""
    if _raw_loading is None:
        from tabarena.benchmark.result import raw_loading as _raw_loading

    def json_scalar(value: Any) -> Any:
        item = getattr(value, "item", None)
        return item() if callable(item) else value

    rows: list[Dict[str, Any]] = []
    for path in _raw_loading.fetch_raw_result_paths(results_dir):
        artifact = _raw_loading.load_and_align(path)
        result = artifact.result
        task = dict(result.get("task_metadata", {}))
        memory = dict(result.get("memory_usage", {}))
        peak_cpu = json_scalar(memory.get("peak_mem_cpu"))
        minimum_cpu = json_scalar(memory.get("min_mem_cpu"))
        peak_gpu = json_scalar(memory.get("peak_mem_gpu"))
        minimum_gpu = json_scalar(memory.get("min_mem_gpu"))
        method_metadata = dict(result.get("method_metadata", {}))
        rows.append(
            {
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
        )
    rows.sort(
        key=lambda row: (
            row["method"],
            row["dataset"],
            int(row["repeat"] or 0),
            int(row["fold"] or 0),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "resources_per_split.json"
    json_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "resources_per_split.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_RESOURCE_FIELDS))
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    return rows


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


def _experiment_models(args: argparse.Namespace) -> list[tuple[Any, int]]:
    models: list[tuple[Any, int]] = []
    if args.device in {"cpu", "both"}:
        models.append((gen_ctboost_cpu, args.n_configs))
    if args.device in {"gpu", "both"}:
        models.append((gen_ctboost_gpu, args.n_configs))
    if args.rerun_competitors:
        from tabarena.models.catboost import gen_catboost
        from tabarena.models.xgboost import gen_xgboost

        if args.device in {"cpu", "both"}:
            models.extend([(gen_catboost, args.n_configs), (gen_xgboost, args.n_configs)])
        if args.device in {"gpu", "both"}:
            models.extend(
                (generator, args.n_configs)
                for generator in _gpu_competitor_generators()
            )
    return models


def main() -> int:
    if gen_ctboost_cpu is None:
        raise SystemExit(
            "TabArena is not installed. Follow benchmarks/tabarena/README.md before running this module."
        )
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    args = _parse_args()
    datasets = args.datasets
    if datasets is None and args.subset == "lite":
        datasets = DEFAULT_LITE_DATASETS
    args.datasets = datasets

    if args.device in {"gpu", "both"}:
        import ctboost

        if not bool(ctboost.build_info().get("cuda_enabled", False)):
            raise SystemExit(
                "GPU benchmarking requires a CUDA-enabled CTBoost wheel; "
                "run ctboost-install-gpu first"
            )
        if args.num_gpus == 0:
            raise SystemExit("--device gpu/both cannot be combined with --num-gpus 0")

    models = _experiment_models(args)
    experiments = TabArenaV0pt1ExperimentBundle(models=models).build_experiments(
        time_limit=args.time_limit,
        num_cpus=args.num_cpus,
        num_gpus=(0 if args.device == "cpu" and args.num_gpus is None else args.num_gpus),
        memory_limit=args.memory_limit_gb,
    )

    context = TabArenaContext()
    manifest_path = args.output_dir / "run_manifest.json"
    _write_manifest(manifest_path, args, status="started")
    build_kwargs = {} if datasets is None else {"dataset_names": datasets}
    context.build_and_run_jobs(
        experiments,
        expname=str(args.results_dir),
        subset=args.subset,
        build_kwargs=build_kwargs,
        new_result_prefix="[New] ",
        debug_mode=not args.ray,
    )
    _write_resource_summary(args.results_dir, args.output_dir)
    leaderboard = context.compare(output_dir=args.output_dir)
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    _write_manifest(manifest_path, args, status="completed")
    _write_console_table(_format_table(website))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
