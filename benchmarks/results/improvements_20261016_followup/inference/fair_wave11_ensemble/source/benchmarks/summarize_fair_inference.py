"""Summarize three balanced immutable latency passes without mixing policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


LIBRARIES = ("ctboost", "xgboost", "lightgbm")
DETAILS = ("development_error", "retained_rounds", "model_bytes", "model_sha256", "problem_type",
           "components", "native_model_calls", "seed_pair", "total_retained_rounds", "component_source_bytes",
           "output_kind", "development_pair")


def summarize(paths):
    paths = [Path(path) for path in paths]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != 3:
        raise ValueError("Exactly three balanced library-order passes are required")
    if any(set(run["library_order"]) != set(LIBRARIES) for run in runs):
        raise ValueError("Each pass must cover every library")
    if any({run["library_order"][position] for run in runs} != set(LIBRARIES) for position in range(3)):
        raise ValueError("Each library must appear once in every timing-order position")
    for field in ("selection_sha256", "plan_sha256", "development_quality_sha256", "runner_sha256",
                  "preprocessing_runner_sha256", "runtime", "temperature_helper_sha256",
                  "ensemble_helper_sha256", "reference_sha256"):
        if any(run.get(field) != runs[0].get(field) for run in runs):
            raise ValueError(f"Cannot combine differing frozen receipts: {field}")
    indexed = [{(row["dataset"], row["rows"], row["library"]): row for row in run["records"]} for run in runs]
    if any(len(index) != len(run["records"]) or set(index) != set(indexed[0]) for index, run in zip(indexed, runs)):
        raise ValueError("Cases changed or duplicate records appeared between passes")
    rows = []
    for dataset, count in dict.fromkeys((row["dataset"], row["rows"]) for row in runs[0]["records"]):
        row = {"dataset": dataset, "rows": count, "libraries": {}}
        for library in LIBRARIES:
            members = [index[dataset, count, library] for index in indexed]
            for field in DETAILS + ("input_sha256", "row_selection_sha256"):
                if any(member.get(field) != members[0].get(field) for member in members):
                    raise ValueError(f"Case metadata changed: {dataset}/{library}/{field}")
            row["libraries"][library] = {
                **{key: members[0][key] for key in DETAILS if key in members[0]},
                **{path: {"median_ms": statistics.geometric_mean(member[path]["median_ms"] for member in members),
                          "order_medians_ms": [member[path]["median_ms"] for member in members]}
                   for path in ("raw", "normalized_frame")}}
        row["ctboost_speedup"] = {
            other: {path: row["libraries"][other][path]["median_ms"] / row["libraries"]["ctboost"][path]["median_ms"]
                    for path in ("raw", "normalized_frame")} for other in ("xgboost", "lightgbm")}
        rows.append(row)
    aggregate = {}
    for batch in (None, *sorted({row["rows"] for row in rows})):
        subset = [row for row in rows if batch is None or row["rows"] == batch]
        aggregate["all" if batch is None else str(batch)] = {
            other: {path: {"geometric_speedup": statistics.geometric_mean(row["ctboost_speedup"][other][path] for row in subset),
                           "faster_cases": sum(row["ctboost_speedup"][other][path] > 1 for row in subset), "cases": len(subset)}
                    for path in ("raw", "normalized_frame")} for other in ("xgboost", "lightgbm")}
    return {"scope": "Three balanced passes of one frozen model policy/runtime. Speedup is competitor latency/CTBoost latency; greater than1 means CTBoost faster. Quality/capacity differ; no equal-quality or leaderboard claim.",
            "source_receipts": [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
            "runtime": runs[0]["runtime"], "selection_sha256": runs[0]["selection_sha256"],
            "plan_sha256": runs[0]["plan_sha256"], "aggregate": aggregate, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing summaries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.runs), indent=2)+"\n", encoding="utf-8")
