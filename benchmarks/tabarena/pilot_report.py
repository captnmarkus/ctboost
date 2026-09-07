"""Stage a deterministic, complete validation-pilot report without publishing.

Only terminal runs are accepted. The frozen evaluator owns the decision; this
helper exports every arm, including failures, and never reads outer-test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

from benchmarks.tabarena.pilot_evaluation import evaluate_pilot, expected_pilot_jobs

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def dataset_arm_rows(protocol, records):
    """Keep incomplete/failed arm means undefined instead of dropping failures."""
    rows = []
    expected_folds = len(protocol["splits"]["inner_fold_indices"])
    for dataset in protocol["datasets"]:
        family = dataset["problem_type"]
        for variant in protocol["variants_by_problem_type"][family]:
            arm = [r for r in records if r["dataset"] == dataset["dataset_name"] and r["variant"] == variant]
            successful = [r for r in arm if r["status"] == "ok"]
            complete = len(successful) == expected_folds
            times = [r["fit_seconds"] for r in arm if finite(r.get("fit_seconds"))]
            memory = [r["peak_rss_bytes"] for r in arm if finite(r.get("peak_rss_bytes"))]
            secondary = {}
            for name in ("binary_log_loss", "multiclass_accuracy"):
                values = [r.get("secondary_metrics", {}).get(name) for r in successful]
                secondary[name] = statistics.mean(values) if complete and all(finite(v) for v in values) else None
            rounds = [r.get("retained_boosting_rounds") for r in successful]
            rows.append({
                "dataset": dataset["dataset_name"], "problem_type": family,
                "variant": variant, "successful_fits": len(successful), "expected_fits": expected_folds,
                "statuses": "; ".join(f"inner{r['inner_fold']}:{r['status']}" for r in sorted(arm, key=lambda r: r["inner_fold"])),
                "mean_validation_error": statistics.mean(r["metric_error"] for r in successful) if complete else None,
                "mean_fit_seconds": statistics.mean(r["fit_seconds"] for r in successful) if complete else None,
                "recorded_fit_seconds_total": sum(times),
                "peak_rss_bytes": max(memory) if memory else None,
                "deadline_stopped_fits": sum(r.get("deadline_stopped") is True for r in arm),
                "mean_retained_boosting_rounds": statistics.mean(rounds) if complete and all(finite(v) for v in rounds) else None,
                **secondary,
            })
    for row in rows:
        baseline = next(r for r in rows if r["dataset"] == row["dataset"] and r["variant"] == "baseline")
        a, b = baseline["mean_validation_error"], row["mean_validation_error"]
        row["relative_error_improvement"] = (a - b) / max(a, 1e-12) if a is not None and b is not None else None
    return rows


def verify_prediction(path, record, prepared):
    """Verify a saved inner-validation artifact without accessing raw features."""
    import numpy as np

    if sha256(path) != record.get("prediction_sha256"):
        raise ValueError(f"Prediction checksum mismatch: {record['job_id']}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"predictions", "labels", "validation_indices"}:
            raise ValueError("Prediction archive must contain only inner predictions, labels and indices")
        prediction, labels, validation = (archive[key] for key in ("predictions", "labels", "validation_indices"))
        count = prepared["rows_outer_train"]
        if validation.ndim != 1 or not np.issubdtype(validation.dtype, np.integer):
            raise ValueError("Validation indices must be a one-dimensional integer array")
        if len(np.unique(validation)) != len(validation) or np.any(validation < 0) or np.any(validation >= count):
            raise ValueError("Validation indices lie outside the prepared outer-training table")
        if labels.ndim != 1 or len(labels) != len(validation) or prediction.shape[0] != len(validation):
            raise ValueError("Prediction/label/validation row counts disagree")
        if len(validation) != record["rows_validation"] or not np.isfinite(prediction).all() or not np.isfinite(labels).all():
            raise ValueError("Validation prediction artifact has invalid values or row count")
        training = np.setdiff1d(np.arange(count, dtype=np.int64), validation)
        digest = hashlib.sha256()
        for indices in (training, validation):
            values = np.asarray(indices, dtype="<i8")
            digest.update(len(values).to_bytes(8, "little"))
            digest.update(values.tobytes())
        if digest.hexdigest() != record["split_sha256"]:
            raise ValueError("Prediction indices do not match the frozen inner split")


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def findings_markdown(protocol, decision, records, execution, calibration, rows):
    outcome = "At least one task family met the frozen gate for a new HPO portfolio." if decision["full_hpo_justified"] else "No task family met the frozen gate; this pilot does not justify the proposed full HPO rerun."
    lines = [
        "# CTBoost 0.1.59 validation-only pilot", "", outcome, "",
        f"All **{decision['expected_fit_count']} predeclared fits** are accounted for across "
        f"{len(protocol['datasets'])} datasets. Status counts: " + ", ".join(f"{name}: {count}" for name, count in sorted(decision["status_counts"].items())) + ".", "",
        "Only official outer-training rows were used. Two fixed inner folds supplied validation "
        "scores and early stopping. Outer-test scores and Elo were neither computed nor used for "
        "selection. This is an author-run development pilot on noncanonical hardware, not a "
        "TabArena leaderboard result, an HPO25 run, or evidence of a library-default promotion.", "",
        "## Decisions by task family", "",
        "| Family | Approved mode | Candidate | Passed | Median error reduction | Dataset wins | Geometric mean fit ratio | P90 fit ratio | Reasons |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for family, report in decision["families"].items():
        for name, arm in report["variants"].items():
            m = arm["metrics"]
            gain = m.get("median_relative_error_improvement")
            lines.append(f"| {family} | {report['winner']} | {name} | {arm['passed']} | "
                         f"{_number(None if gain is None else gain * 100, 2)}% | "
                         f"{m.get('dataset_wins', '—')}/{m.get('dataset_count', '—')} | "
                         f"{_number(m.get('geometric_mean_fit_time_ratio'), 2)} | {_number(m.get('p90_fit_time_ratio'), 2)} | "
                         f"{'; '.join(arm['reasons']) or 'All frozen checks passed'} |")
    gate = protocol["decision_gate"]
    lines += ["", f"Admission required at least {gate['minimum_median_relative_error_improvement'] * 100:g}% "
              "median relative validation-error reduction, a strict majority of dataset wins, "
              f"no dataset regression above {gate['maximum_any_dataset_relative_error_increase'] * 100:g}%, "
              f"geometric mean fit ratio ≤{gate['maximum_geometric_mean_fit_time_ratio']:g}× and "
              f"90th percentile ≤{gate['maximum_p90_fit_time_ratio']:g}×, with complete successful "
              "paired fits and intact provenance. Failures remain in the report and disqualify "
              "their affected comparison. These are pragmatic development thresholds, not statistical significance tests.", "",
              "## Every dataset and arm", "",
              "Errors are 1 − ROC AUC for binary, RMSE for regression and log loss for multiclass. "
              "Each mean requires both successful inner folds. A dash preserves an undefined "
              "comparison after failure; no task is imputed or omitted. Positive error reduction favors the candidate.", "",
              "| Dataset | Arm | Successful folds | Mean error | Error reduction | Mean fit seconds | Peak RSS GiB | Deadline stops |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        gain, rss = row["relative_error_improvement"], row["peak_rss_bytes"]
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['successful_fits']}/{row['expected_fits']} | "
                     f"{_number(row['mean_validation_error'], 6)} | {_number(None if gain is None else gain * 100, 2)}% | "
                     f"{_number(row['mean_fit_seconds'], 2)} | {_number(None if rss is None else rss / 1024**3, 3)} | {row['deadline_stopped_fits']} |")
    resources, hardware = execution["resources"], calibration.get("hardware", {})
    total_seconds = sum(r.get("fit_seconds", 0) for r in records if finite(r.get("fit_seconds")))
    lines += ["", "## Runtime and reproducibility", "",
              f"Recorded fit time totals {total_seconds:.2f} seconds across concurrent workers; "
              "this sum is not elapsed wall time. Prediction times, secondary binary log loss / "
              "multiclass accuracy, retained rounds, per-fit status and all recorded failures are "
              "in the JSON/CSV tables and original fit records.", "",
              f"Hardware: {protocol['compute']['cpu_resource_calibration']['machine']}. "
              f"Detected platform: {hardware.get('platform', 'recorded in resources.json')}; "
              f"physical/logical CPUs: {hardware.get('physical_cores', '—')}/{hardware.get('logical_cpus', '—')}. "
              f"Selected layout: {resources['workers']} workers × {resources['threads_per_worker']} histogram threads, "
              f"with {resources['blas_openmp_threads']} BLAS/OpenMP thread per process. "
              f"Each fit receives {resources['fit_time_limit_seconds']} seconds plus "
              f"{resources['hard_fit_grace_seconds']} seconds of watchdog grace. "
              f"Worker memory cap: {resources['memory_per_worker_bytes'] / 1024**3:.2f} GiB; "
              f"aggregate budget: {resources['memory_budget_bytes'] / 1024**3:.2f} GiB. GPU use is disabled.", "",
              "| Evidence | SHA256 |", "| --- | --- |",
              f"| Frozen protocol | `{decision['protocol_sha256']}` |",
              f"| Installed native extension | `{execution['runtime']['native_extension_sha256']}` |",
              f"| Public wheel | `{execution['public_wheel']['sha256']}` |",
              f"| Resource contract | `{execution['resource_contract_sha256']}` |",
              f"| Package environment | `{execution['runtime']['environment_sha256']}` |", "",
              "The complete package versions, CPU affinity, public-wheel identity, preparation "
              "hashes and frozen source files are under `provenance/`. `SHA256SUMS` covers every "
              "staged artifact. Original JSON records are copied byte-for-byte. Inner-validation "
              "prediction archives contain only predictions, encoded labels and indices relative "
              "to the outer-training table; raw feature tables and outer-test labels are excluded.", "",
              "## Limits and next action", "", decision["interpretation"], "",
              "Any new 25-configuration portfolio must follow the frozen admission mapping and "
              "be committed with its exact configurations and resource contract before final "
              "outer-test evaluation. The historical 0.1.58 results remain separate. This report "
              "does not claim official leaderboard acceptance.", "",
              "Underlying datasets and validation labels retain their original terms; no blanket "
              "license is asserted over them. Benchmark code follows the "
              "[CTBoost repository license](https://github.com/captnmarkus/ctboost/blob/v0.1.59/LICENSE). "
              "This package is evaluation evidence and does not redistribute raw feature tables.", "",
              "## Source tasks", "",
              "| Dataset | OpenML task | OpenML dataset |", "| --- | --- | --- |"]
    prepared_by_name = {row["dataset"]: row for row in execution["prepared"]["datasets"]}
    for dataset in protocol["datasets"]:
        task_id = dataset["task_id"]
        dataset_id = prepared_by_name[dataset["dataset_name"]].get("dataset_id")
        dataset_link = f"[{dataset_id}](https://www.openml.org/d/{dataset_id})" if dataset_id is not None else "—"
        lines.append(f"| {dataset['dataset_name']} | [{task_id}](https://www.openml.org/t/{task_id}) | {dataset_link} |")
    lines.append("")
    return "\n".join(lines)


def export_pilot_report(run_root, output, *, protocol_path, calibration_path, source_root=ROOT, preregistration_path=None):
    """Validate a terminal run, then write a new isolated publication directory."""
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    source_root = Path(source_root).resolve()
    if output.exists() or output == run_root or output.is_relative_to(run_root) or run_root.is_relative_to(output):
        raise ValueError("Publication output must be a new directory outside the run")
    if not (run_root / "decision.json").is_file():
        raise ValueError("A final decision is required; partial pilots are never evaluated here")
    progress = read_json(run_root / "progress.json")
    if progress.get("active_blocks") or progress.get("queued_blocks") != 0:
        raise ValueError("The pilot is still active or incomplete")
    protocol, protocol_hash = read_json(protocol_path), sha256(protocol_path)
    execution, calibration = read_json(run_root / "execution.json"), read_json(calibration_path)
    if execution["protocol_sha256"] != protocol_hash or canonical_hash(calibration) != execution["calibration_sha256"]:
        raise ValueError("Protocol/calibration provenance mismatch")
    if canonical_hash(execution["resources"]) != execution["resource_contract_sha256"]:
        raise ValueError("Resource contract hash mismatch")
    source_files = []
    for relative, expected_hash in execution["runtime"]["source_sha256"].items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or sha256(path) != expected_hash:
            raise ValueError(f"Frozen source mismatch: {relative}")
        source_files.append((path, Path("provenance/source") / relative))
    record_paths = sorted((run_root / "fits").glob("*/inner*/*.json"))
    records = [read_json(path) for path in record_paths]
    expected = {(r["dataset"], r["variant"], r["inner_fold"]) for r in expected_pilot_jobs(protocol)}
    identities = [(r["dataset"], r["variant"], r["inner_fold"]) for r in records]
    if len(records) != len(expected) or len(set(identities)) != len(expected) or set(identities) != expected:
        raise ValueError("Require every unique predeclared terminal record before evaluating")
    decision = evaluate_pilot(protocol, records, protocol_sha256=protocol_hash)
    if not decision["global_integrity_passed"] or decision != read_json(run_root / "decision.json"):
        raise ValueError("Final decision or record integrity mismatch")
    runtime_hash = canonical_hash(execution["runtime"])
    predictions = []
    prepared = {r["dataset"]: r for r in execution["prepared"]["datasets"]}
    for path, record in zip(record_paths, records):
        if record["resource_contract_sha256"] != execution["resource_contract_sha256"]:
            raise ValueError("Record does not match the frozen resource contract")
        if record.get("execution_runtime_sha256", runtime_hash) != runtime_hash:
            raise ValueError("Record runtime provenance mismatch")
        if record["status"] == "ok":
            if record.get("execution_runtime_sha256") != runtime_hash:
                raise ValueError("Successful record lacks runtime provenance")
            prediction_path = path.with_suffix(".npz")
            verify_prediction(prediction_path, record, prepared[record["dataset"]])
            predictions.append((prediction_path, Path("fits") / prediction_path.relative_to(run_root / "fits")))
    rows = dataset_arm_rows(protocol, records)
    markdown = findings_markdown(protocol, decision, records, execution, calibration, rows)
    copies = [(Path(protocol_path), Path("protocol.json")),
              (Path(calibration_path), Path("provenance/resources.json")),
              (run_root / "execution.json", Path("provenance/execution.json")),
              (run_root / "decision.json", Path("decision.json")),
              (run_root / "progress.json", Path("provenance/final_progress.json")),
              *source_files, *predictions]
    receipt_path = Path(preregistration_path) if preregistration_path is not None else run_root.parent / "preregistration.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        normalized_hash = hashlib.sha256(Path(protocol_path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if receipt.get("protocol_sha256") != protocol_hash or receipt.get("git_blob_sha256") != normalized_hash:
            raise ValueError("Public preregistration receipt does not match the frozen protocol")
        copies.append((receipt_path, Path("provenance/preregistration.json")))
        markdown += ("\nThe [public preregistration](" + receipt["url"] + ") preceded the first pilot fit. "
                     "`provenance/preregistration.json` records both the exact frozen Windows JSON hash "
                     "and the public Git blob hash; their only difference is CRLF versus LF line endings. "
                     "The frozen protocol is copied byte-for-byte here.\n")
    for directory in ("fits", "journal", "logs"):
        pattern = "*/inner*/*.json" if directory == "fits" else "**/*"
        for path in sorted((run_root / directory).glob(pattern)):
            if path.is_file():
                copies.append((path, path.relative_to(run_root)))
    output.mkdir(parents=True)
    for source, relative in copies:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (output / "README.md").write_text("---\npretty_name: CTBoost 0.1.59 validation-only learning-option pilot\ntags:\n- tabular\n- benchmarking\n- ctboost\n---\n\n" + markdown, encoding="utf-8")
    (output / "GITHUB_FINDINGS.md").write_text(markdown, encoding="utf-8")
    write_json(output / "dataset_arm_results.json", rows)
    with (output / "dataset_arm_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "publication_manifest.json", {
        "schema_version": 1, "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "expected_fits": len(expected), "prediction_archives": len(predictions),
        "all_original_records_preserved": True, "outer_test_data_included": False,
        "raw_feature_tables_included": False, "exporter_sha256": sha256(__file__),
        "source_record_sha256": {str(path.relative_to(run_root)).replace("\\", "/"): sha256(path) for path in record_paths},
    })
    lines = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "records": len(records), "prediction_archives": len(predictions), "full_hpo_justified": decision["full_hpo_justified"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks/tabarena/pilot_0159_v1.json")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_pilot_report(args.run_root, args.output, protocol_path=args.protocol,
                                        calibration_path=args.calibration, source_root=args.source_root,
                                        preregistration_path=args.preregistration), indent=2))


if __name__ == "__main__":
    main()
