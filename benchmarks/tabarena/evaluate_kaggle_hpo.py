"""Validate and evaluate this repository's own downloaded Kaggle HPO outputs.

Run with the pinned TabArena/AutoGluon environment. Only pass outputs downloaded
from your own trusted workers: results.pkl is executable pickle data. Checksums
detect corruption and mismatches; they do not authenticate a foreign producer.
No results are scored until all 156 shards and 1,326 config/task pairs validate.
Raw predictions and targets stay under private-raw, outside the public bundle.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.tabarena import kaggle_hpo_worker as protocol

PORTFOLIO_25_SHA256 = "210f35c95c458f888f83dbb785ff56d25fa56a1ed554299b43fe0a95c41fc4c0"
METADATA_RELATIVE = (
    "packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/"
    "TabArena-v0.1_tasks_metadata.csv"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_relative(value: str) -> PurePosixPath:
    require(isinstance(value, str) and bool(value), "Empty relative path")
    require("\\" not in value and ":" not in value, f"Unsafe relative path: {value}")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and ".." not in path.parts,
        f"Unsafe relative path: {value}",
    )
    require(bool(path.parts), f"Empty relative path: {value}")
    return path


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def inside(root: Path, relative: str) -> Path:
    path = root.joinpath(*safe_relative(relative).parts)
    require(
        is_within(path.resolve(), root.resolve()),
        f"Path escapes input root: {relative}",
    )
    return path


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": protocol.sha256_file(path),
    }


def verify_file(path: Path, record: dict[str, Any]) -> None:
    require(path.is_file(), f"Missing artifact: {path}")
    require(path.stat().st_size == record.get("size_bytes"), f"Size mismatch: {path}")
    require(
        protocol.sha256_file(path) == record.get("sha256"), f"SHA-256 mismatch: {path}"
    )


def load_task_ids(source_repo: Path) -> dict[str, str]:
    with (source_repo / METADATA_RELATIVE).open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split_index"] == "r0f0"]
    tasks = {row["dataset_name"]: row["task_id_str"] for row in rows}
    expected = {name for group in protocol.DATASET_GROUPS for name in group}
    require(
        len(rows) == len(tasks) == 51 and set(tasks) == expected,
        "Pinned task metadata must contain exactly the frozen 51 Lite tasks",
    )
    require(len(set(tasks.values())) == 51, "Duplicate task IDs in pinned metadata")
    return tasks


def expected_worker_hashes(index: int, worker_template: Path | None = None) -> set[str]:
    source = (worker_template or Path(protocol.__file__)).read_text(encoding="utf-8")
    source, count = re.subn(r"(?m)^SHARD_INDEX = 0$", f"SHARD_INDEX = {index}", source)
    require(count == 1, "Worker template does not contain one shard identity")
    # The controller normalizes frozen templates and uploaded scripts to LF on every platform.
    return {hashlib.sha256(source.encode("utf-8")).hexdigest()}


def validate_manifest(
    manifest: dict[str, Any],
    task_ids: dict[str, str],
    worker_template: Path | None = None,
) -> dict[str, tuple[str, str]]:
    index = manifest.get("shard_index")
    spec = protocol.shard_spec(index)
    for key, value in spec.items():
        require(manifest.get(key) == value, f"Shard {index}: mismatched {key}")
    expected = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_exit_code": 0,
        "fatal_error": None,
        "benchmark_name": protocol.BENCHMARK_NAME,
        "ctboost_version": protocol.CTBOOST_VERSION,
        "tabarena_commit": protocol.TABARENA_COMMIT,
        "configuration_count": 26,
        "hpo_configs_run": 25,
        "expected_parent_results_total": 1326,
        "expected_child_fits_total": 10608,
        "official_tabarena_lite_dataset_count": 51,
        "portfolio_200_sha256": protocol.PORTFOLIO_200_SHA256,
        "portfolio_25_sha256": PORTFOLIO_25_SHA256,
        "result_file_count": len(spec["datasets"]),
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Shard {index}: mismatched {key}")
    require(not manifest.get("failures"), f"Shard {index}: recorded task failures")
    require(
        manifest.get("worker_sha256") in expected_worker_hashes(index, worker_template),
        f"Shard {index}: worker source hash mismatch",
    )
    packages = manifest.get("installed_packages", {})
    require(
        packages.get("ctboost") == protocol.CTBOOST_VERSION
        and packages.get("autogluon.tabular") == protocol.AUTOGLUON_VERSION,
        f"Shard {index}: installed package versions mismatch",
    )
    require(
        manifest.get("ctboost_build_info", {}).get("package_version")
        == protocol.CTBOOST_VERSION,
        f"Shard {index}: native CTBoost build version mismatch",
    )
    wheel = manifest.get("ctboost_wheel", {})
    require(
        wheel.get("filename", "").startswith(f"ctboost-{protocol.CTBOOST_VERSION}-")
        and re.fullmatch(r"[a-f0-9]{64}", wheel.get("sha256", "")) is not None,
        f"Shard {index}: missing public-wheel provenance",
    )
    resources = manifest.get("requested_resources", {})
    require(
        resources == {"num_cpus": 4, "memory_limit_gb": 28, "time_limit_seconds": 3600},
        f"Shard {index}: resource protocol mismatch",
    )
    require(
        re.fullmatch(r"[a-f0-9]{64}", manifest.get("experiments_sha256", ""))
        is not None,
        f"Shard {index}: missing frozen-experiment hash",
    )
    expected_paths = {
        f"output/{protocol.BENCHMARK_NAME}/data/{spec['config_name']}/{task_ids[name]}/0_0/results.pkl": (
            spec["config_name"],
            name,
        )
        for name in spec["datasets"]
    }
    records = manifest.get("result_files", [])
    names = [safe_relative(record["path"]).as_posix() for record in records]
    require(
        len(names) == len(set(names)) and set(names) == set(expected_paths),
        f"Shard {index}: missing, duplicate, or unexpected config/task results",
    )
    for record in records:
        require(
            isinstance(record.get("size_bytes"), int)
            and record["size_bytes"] > 0
            and re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")) is not None,
            f"Shard {index}: invalid result hash or size",
        )
    return expected_paths


def discover_shards(
    shards_root: Path, task_ids: dict[str, str], worker_template: Path | None = None
) -> list[tuple[Path, dict[str, Any]]]:
    manifests = sorted(shards_root.rglob("artifacts/manifest.json"))
    require(
        len(manifests) == protocol.SHARD_COUNT,
        f"Incomplete run: expected 156 shard manifests, found {len(manifests)}; no scores produced",
    )
    found: dict[int, tuple[Path, dict[str, Any]]] = {}
    keys = set()
    experiment_hashes = set()
    wheel_hashes = set()
    for path in manifests:
        require(
            is_within(path.resolve(), shards_root.resolve()),
            "Manifest escapes shards root",
        )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected_paths = validate_manifest(manifest, task_ids, worker_template)
        index = manifest["shard_index"]
        require(index not in found, f"Duplicate shard {index}")
        require(
            not keys.intersection(expected_paths.values()), "Duplicate config/task pair"
        )
        keys.update(expected_paths.values())
        archive = inside(path.parent.parent, manifest["workspace_archive"]["path"])
        verify_file(archive, manifest["workspace_archive"])
        experiment_hashes.add(manifest["experiments_sha256"])
        wheel_hashes.add(manifest["ctboost_wheel"]["sha256"])
        found[index] = (path, manifest)
    require(
        set(found) == set(range(156)) and len(keys) == 1326,
        "Incomplete config/task coverage",
    )
    require(
        len(experiment_hashes) == 1 and len(wheel_hashes) == 1,
        "Worker experiment or public-wheel hashes differ across shards",
    )
    return [found[index] for index in range(156)]


def extract_results(
    archive: Path, records: list[dict[str, Any]], raw_root: Path
) -> list[Path]:
    """Stream only declared raw results; never use tar.extract or extractall."""
    selected = {
        "workspace/" + safe_relative(record["path"]).as_posix(): record
        for record in records
    }
    require(len(selected) == len(records), "Duplicate declared archive result")
    found = set()
    paths = []
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = safe_relative(member.name).as_posix()
            require(
                member.isdir() or member.isfile(), f"Unsafe archive member type: {name}"
            )
            if name not in selected:
                require(
                    not name.endswith("/results.pkl"),
                    f"Undeclared result in archive: {name}",
                )
                continue
            require(
                name not in found and member.isfile(),
                f"Duplicate archive result: {name}",
            )
            record = selected[name]
            require(
                member.size == record["size_bytes"],
                f"Archive result size mismatch: {name}",
            )
            parts = safe_relative(record["path"]).parts
            require(
                len(parts) == 7
                and parts[:3] == ("output", protocol.BENCHMARK_NAME, "data")
                and parts[-2:] == ("0_0", "results.pkl"),
                f"Invalid result path: {name}",
            )
            target = inside(raw_root, PurePosixPath(*parts[3:]).as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), f"Refusing to overwrite raw result: {target}")
            source = stream.extractfile(member)
            require(source is not None, f"Missing archive file body: {name}")
            with source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            verify_file(target, record)
            found.add(name)
            paths.append(target)
    require(found == set(selected), "Archive is missing declared result files")
    return paths


def validate_raw_result(path: Path, task_ids: dict[str, str]) -> dict[str, Any]:
    """Inspect a checksum-validated pickle from our own worker, one file at a time."""
    import numpy as np

    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    with gzip.open(path, "rb") if compressed else path.open("rb") as stream:
        result = pickle.load(stream)
    config, tid = path.parts[-4:-2]
    metadata = result["task_metadata"]
    dataset = metadata["name"]
    require(
        task_ids.get(dataset) == tid == str(metadata["tid"])
        and metadata["repeat"] == metadata["fold"] == 0
        and result["framework"] == config,
        f"Raw config/task identity mismatch: {path}",
    )
    for key in ("metric_error", "metric_error_val", "time_train_s", "time_infer_s"):
        require(math.isfinite(float(result[key])), f"Non-finite {key}: {path}")
    require(
        result["time_train_s"] >= 0 and result["time_infer_s"] >= 0,
        f"Negative runtime: {path}",
    )
    info = result["method_metadata"]["info"]
    require(
        info["bagged_info"]["num_child_models"] == 8
        and len(info["children_info"]) == 8,
        f"Expected eight fitted bag children: {path}",
    )
    require(
        all(info.get(flag) is True for flag in ("is_fit", "is_valid", "can_infer")),
        f"Parent model is not a valid fitted predictor: {path}",
    )
    match = re.fullmatch(r"CTBoost_(c1|r([1-9][0-9]*))_default_BAG_L1", config)
    require(match is not None, f"Unexpected configuration name: {path}")
    config_index = 0 if match.group(1) == "c1" else int(match.group(2))
    children = info["children_info"]
    require(
        set(children) == {f"S1F{i + 1}" for i in range(8)},
        f"Unexpected bag child names: {path}",
    )
    for fold in range(8):
        child = children[f"S1F{fold + 1}"]
        require(
            all(child.get(flag) is True for flag in ("is_fit", "is_valid", "can_infer"))
            and child.get("hyperparameters", {}).get("random_seed")
            == 8 * config_index + fold,
            f"Invalid child predictor or frozen fold seed: {path}",
        )
    simulation = result["simulation_artifacts"]
    bag = simulation["bag_info"]
    require(
        len(bag["val_idx_per_child"]) == len(bag["pred_proba_test_per_child"]) == 8,
        f"Expected eight bag-fold artifacts: {path}",
    )
    predictions = {}
    target_hashes = {}
    for split in ("val", "test"):
        mapping = simulation[f"pred_proba_dict_{split}"]
        require(set(mapping) == {config}, f"Unexpected prediction config: {path}")
        pred = np.asarray(mapping[config])
        target = np.asarray(simulation[f"y_{split}"])
        require(
            len(target) > 0
            and pred.shape[0] == len(target)
            and np.isfinite(pred).all()
            and np.isfinite(target).all(),
            f"Invalid prediction/target array: {path}",
        )
        predictions[split] = pred
        target_hashes[split] = hashlib.sha256(
            np.ascontiguousarray(target).tobytes()
        ).hexdigest()
        require(pred.ndim in (1, 2), f"Invalid prediction dimension: {path}")
        if result["problem_type"] in ("binary", "multiclass"):
            require(
                np.all(pred >= -1e-6) and np.all(pred <= 1 + 1e-6),
                f"Invalid classification probability: {path}",
            )
            if pred.ndim == 2:
                require(
                    np.allclose(pred.sum(axis=1), 1, atol=1e-5),
                    f"Probabilities do not sum to one: {path}",
                )
    indices = np.concatenate([np.asarray(idx) for idx in bag["val_idx_per_child"]])
    require(
        np.array_equal(np.sort(indices), np.arange(len(predictions["val"]))),
        f"Bag folds do not partition validation rows exactly once: {path}",
    )
    for child in bag["pred_proba_test_per_child"]:
        require(
            np.asarray(child).shape == predictions["test"].shape
            and np.isfinite(child).all(),
            f"Invalid child prediction array: {path}",
        )
    return {
        "config": config,
        "dataset": dataset,
        "tid": tid,
        "child_fits": 8,
        "target_hashes": target_hashes,
    }


def evaluate(
    raw_root: Path, output_root: Path, source_repo: Path, task_ids: dict[str, str]
) -> dict[str, Any]:
    from tabarena.benchmark.task.metadata import TaskMetadataCollection
    from tabarena.contexts import TabArenaContext
    from tabarena.end_to_end import EndToEnd, EndToEndResults
    from tabarena.models._method_metadata import MethodMetadata

    task_metadata = TaskMetadataCollection.from_source(
        source_repo / METADATA_RELATIVE
    ).subset_tasks(split_indices="lite")
    canonical = output_root / "canonical-results"
    # Task-by-task processing keeps prediction arrays out of the driver result set.
    EndToEnd.from_path_raw(
        path_raw=raw_root,
        name_prefix_raw="CTBoost",
        method="CTBoost",
        suite=protocol.BENCHMARK_NAME,
        artifact_dir=canonical,
        task_metadata=task_metadata,
        backend="native",
        num_cpus=1,
        cache=True,
        cache_raw=False,
        cache_processed=False,
    )
    metadata = MethodMetadata.from_yaml(path=canonical / "metadata.yaml")
    require(
        metadata.can_hpo is True,
        "Canonical metadata did not recognize the HPO portfolio",
    )
    results = EndToEndResults.from_cache(methods=[metadata])
    per_config = results.get_results(use_model_results=True)
    configs = {protocol.shard_spec(i * 6)["config_name"] for i in range(26)}
    keys = list(zip(per_config["method"], per_config["dataset"], per_config["fold"]))
    require(
        len(keys) == len(set(keys)) == 1326
        and set(keys) == {(c, d, 0) for c in configs for d in task_ids},
        "Canonical results lost or added config/task pairs",
    )
    hpo = results.get_results()
    require(
        not hpo["imputed"].any(), "Canonical HPO evaluation imputed candidate results"
    )
    require(
        set(hpo["method_subtype"]) == {"default", "tuned", "tuned_ensemble"},
        "Expected default, tuned, and tuned-ensemble result rows",
    )
    context = TabArenaContext(
        extra_methods=results.to_method_metadata_lst(),
        only_valid_tasks=True,
        backend="native",
    )
    leaderboard = context.leaderboard(subset="lite")
    candidate = leaderboard.loc[leaderboard["ta_suite"].eq(protocol.BENCHMARK_NAME)]
    require(
        len(candidate) == 3 and not candidate["imputed"].any(),
        "Missing or imputed candidate leaderboard rows",
    )
    reports = output_root / "reports"
    reports.mkdir()
    leaderboard.to_csv(reports / "leaderboard_lite.csv", index=False)
    summary = {
        "protocol": "TabArena-v0.1 Lite (r0f0), default plus 25 frozen HPO configurations",
        "benchmark_name": protocol.BENCHMARK_NAME,
        "ctboost_version": protocol.CTBOOST_VERSION,
        "configuration_count": 26,
        "hpo_configs_run": 25,
        "dataset_count": 51,
        "raw_result_files": 1326,
        "child_fits": 10608,
        "timing_comparable": False,
        "official_leaderboard_entry": False,
        "leaderboard_rows": len(leaderboard),
        "candidate_rows": json.loads(candidate.to_json(orient="records")),
    }
    protocol.write_json(reports / "ctboost_lite_hpo_summary.json", summary)
    return summary


def public_bundle(output_root: Path, manifest: dict[str, Any]) -> Path:
    bundle = output_root / "public-bundle"
    bundle.mkdir()
    for relative in (
        "canonical-results/metadata.yaml",
        "canonical-results/results/hpo_results.parquet",
        "canonical-results/results/model_results.parquet",
        "reports/leaderboard_lite.csv",
        "reports/ctboost_lite_hpo_summary.json",
    ):
        source = output_root / relative
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    protocol.write_json(bundle / "validation/aggregate_manifest.json", manifest)
    readme = (
        "---\nlicense: other\nviewer: false\ntags: [tabarena, ctboost, benchmark]\n---\n\n"
        "# CTBoost 0.1.58: TabArena-v0.1 Lite, default plus 25 HPO configurations\n\n"
        "This benchmark-artifact bundle covers 51 datasets at r0f0, 26 configurations "
        "(the manual default plus the first 25 configurations of the frozen 200-config "
        "portfolio), and eight-fold bagging: 1,326 parent results and 10,608 child fits. "
        "TabArena selects configurations and ensemble weights using validation predictions. "
        "The test outcomes do not change the frozen search space.\n\n"
        "See reports/ctboost_lite_hpo_summary.json for the measured default, tuned, and "
        "tuned-ensemble scores and reports/leaderboard_lite.csv for their comparison roster. "
        "Elo is roster-dependent. This is a local Lite HPO25 result, not a full 200-config "
        "run, TabArena-Full result, or official leaderboard entry. Kaggle used 4 CPUs, "
        "28 GB RAM and a 3,600-second limit; its timing is not comparable to canonical "
        "8-CPU, 32-GB TabArena runtimes.\n\n"
        f"Fitting and evaluation use TabArena commit `{protocol.TABARENA_COMMIT}`. "
        "The validation manifest records source, wheel, experiment, task, and archive "
        "hashes. Run-level metadata records can_hpo: true.\n\n"
        "Only prediction-free result tables, metadata, and validation records are included. "
        "Raw pickles, predictions, targets, and processed repositories are excluded. "
        "Underlying OpenML datasets retain their respective licenses.\n\n"
        "Protocol: [TabArena](https://github.com/autogluon/tabarena); "
        "integration: [PR #479](https://github.com/autogluon/tabarena/pull/479).\n"
    )
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    checksums = [
        f"{protocol.sha256_file(path)}  {path.relative_to(bundle).as_posix()}"
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    ]
    (bundle / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-root",
        type=Path,
        required=True,
        help="Downloaded outputs of your own trusted workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New directory for validated outputs",
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        required=True,
        help="Checkout of the pinned TabArena commit",
    )
    parser.add_argument(
        "--worker-template",
        type=Path,
        help="Frozen controller worker_template.py; automatically found beside shards/",
    )
    args = parser.parse_args(argv)
    source = args.source_repo.resolve()
    output = args.output_root.resolve()
    shards_root = args.shards_root.resolve()
    require(not output.exists(), "Output root already exists; choose a new directory")
    require(
        not is_within(output, shards_root),
        "Output root must be outside downloaded shards",
    )
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    require(
        head == protocol.TABARENA_COMMIT,
        "Evaluation checkout does not match the frozen TabArena commit",
    )
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--", "packages/tabarena"], text=True
    )
    require(not tracked_diff, "Evaluation package has uncommitted changes")
    task_ids = load_task_ids(source)
    template = args.worker_template or shards_root.parent / "worker_template.py"
    require(
        template.is_file(),
        "Missing frozen worker_template.py; specify --worker-template",
    )
    shards = discover_shards(shards_root, task_ids, template)
    # Import the audited source, not a potentially different installed TabArena revision.
    sys.path.insert(0, str(source / "packages/tabarena/src"))
    _, experiments = protocol.build_experiments(num_cpus=4)
    require(
        experiments["experiments_sha256"] == shards[0][1]["experiments_sha256"],
        "Evaluation experiment hash differs from the workers",
    )
    output.mkdir(parents=True)
    raw = output / "private-raw"
    raw.mkdir()
    records = []
    shard_records = []
    target_hashes_by_task = {}
    for path, manifest in shards:
        archive = inside(path.parent.parent, manifest["workspace_archive"]["path"])
        for result_path in extract_results(archive, manifest["result_files"], raw):
            validated = validate_raw_result(result_path, task_ids)
            tid, hashes = validated["tid"], validated["target_hashes"]
            require(
                target_hashes_by_task.setdefault(tid, hashes) == hashes,
                f"Target rows differ across configurations for task {tid}",
            )
            records.append({**validated, **file_record(result_path, raw)})
        shard_records.append(
            {
                "shard_index": manifest["shard_index"],
                "manifest_sha256": protocol.sha256_file(path),
                "archive": manifest["workspace_archive"],
                "worker_sha256": manifest["worker_sha256"],
            }
        )
        print(f"Validated shard {manifest['shard_index'] + 1}/156", flush=True)
    summary = evaluate(raw, output, source, task_ids)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "benchmark_name": protocol.BENCHMARK_NAME,
        "scope": {
            "datasets": 51,
            "split": "r0f0",
            "configurations": 26,
            "hpo_configs_run": 25,
            "parent_results": 1326,
            "bag_children": 8,
            "child_fits": 10608,
        },
        "provenance": {
            "ctboost_version": protocol.CTBOOST_VERSION,
            "tabarena_commit": head,
            "ctboost_wheel": shards[0][1]["ctboost_wheel"],
            "installed_packages": shards[0][1]["installed_packages"],
            "portfolio_200_sha256": protocol.PORTFOLIO_200_SHA256,
            "portfolio_25_sha256": PORTFOLIO_25_SHA256,
            "experiments_sha256": experiments["experiments_sha256"],
            "worker_template_sha256": protocol.sha256_file(template),
            "task_metadata_sha256": protocol.sha256_file(source / METADATA_RELATIVE),
            "evaluator_sha256": protocol.sha256_file(Path(__file__)),
        },
        "resources": shards[0][1]["requested_resources"],
        "timing_comparable": False,
        "summary": summary,
        "shards": shard_records,
        "results": records,
    }
    protocol.write_json(output / "validation/aggregate_manifest.json", manifest)
    bundle = public_bundle(output, manifest)
    print(f"Validated 1,326 results; public artifact bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
