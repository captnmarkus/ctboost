"""Archive a completed four-pass timing panel without source data or predictions."""
from pathlib import Path
import hashlib
import json
import shutil
import statistics

ROOT = Path(r"C:\apps\ctboost")
SOURCE = ROOT / ".tmp/release-readiness-20260908/timing-phase3"
TARGET = ROOT / "benchmarks/results/release_readiness_20260908/inference_timing_phase3"
ADAPTER = ROOT / "benchmarks/inference_release_timing.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


plan = read(SOURCE / "plan.json")
orchestration = read(SOURCE / "orchestration.json")
assert orchestration["complete"] and all(row["exit_code"] == 0 for row in orchestration["passes"])
assert digest(ADAPTER) == plan["adapter_sha256"]
passes = [read(SOURCE / f"pass-{index}-{arm}/summary.json")
          for index, arm in enumerate(plan["order"], start=1)]
assert len(passes) == 4
for item in passes:
    assert item["all14_complete"] and len(item["rows"]) == 14
    assert item["plan_sha256"] == digest(SOURCE / "plan.json")
    assert item["adapter_sha256"] == plan["adapter_sha256"]
    assert item["runtime"]["cpu_affinity"] == [6] and item["runtime"]["threads"] == 1
    for row in item["rows"]:
        assert row["input_sha256"] == plan["input_sha256"][row["dataset"]]
        assert row["timed_rows"] == 1000
        for model in row["models"].values():
            assert model["public_prediction_bitwise"] and len(model["block_ms"]) == 7
for a, b in ((0, 3), (1, 2)):
    assert passes[a]["runtime"] == passes[b]["runtime"]
compatibility = read(ROOT / ".tmp/release-readiness-20260908/public-0160-model-compatibility-portable.json")
expected_python = {name.removeprefix("installed/ctboost/"): value
                   for name, value in compatibility["source_sha256"].items()
                   if name.startswith("installed/ctboost/")}
assert passes[1]["runtime"]["ctboost_python_sha256"] == expected_python
assert passes[1]["runtime"]["native_sha256"] == compatibility["native_sha256"]
assert plan["final_wheel_sha256"] == compatibility["wheel_sha256"]
assert all(item["runtime"]["distribution_versions"] == plan["original_runtime"]["versions"] for item in passes)

rows = []
for index, base_row in enumerate(passes[0]["rows"]):
    name = base_row["dataset"]
    current = [item["rows"][index] for item in passes]
    assert all(row["dataset"] == name for row in current)
    for model_arm in ("ctboost_default", "catboost_ag_default"):
        for field in ("model_sha256", "rounds", "trees", "prediction_values_sha256", "prediction_dtype", "prediction_shape"):
            assert all(row["models"][model_arm][field] == current[0]["models"][model_arm][field] for row in current)
    ct = [row["models"]["ctboost_default"]["median_ms"] for row in current]
    cb = [row["models"]["catboost_ag_default"]["median_ms"] for row in current]
    base_ct, final_ct = statistics.median((ct[0], ct[3])), statistics.median((ct[1], ct[2]))
    base_cb, final_cb = statistics.median((cb[0], cb[3])), statistics.median((cb[1], cb[2]))
    rows.append({"dataset": name, "problem_type": base_row["problem_type"],
                 "ctboost_pass_median_ms": ct, "catboost_pass_median_ms": cb,
                 "baseline_ctboost_ms": base_ct, "final_ctboost_ms": final_ct,
                 "baseline_catboost_ms": base_cb, "final_catboost_ms": final_cb,
                 "ctboost_speedup": base_ct / final_ct,
                 "ctboost_speedup_baseline_then_final": ct[0] / ct[1],
                 "ctboost_speedup_final_then_baseline": ct[3] / ct[2],
                 "baseline_ctboost_last_to_first": ct[3] / ct[0],
                 "final_ctboost_second_to_first": ct[2] / ct[1],
                 "catboost_baseline_period_to_final_period": base_cb / final_cb,
                 "catboost_last_to_first": cb[3] / cb[0],
                 "control_adjusted_ctboost_speedup": (base_ct / final_ct) / (base_cb / final_cb),
                 "final_ctboost_to_catboost": final_ct / final_cb})

report = {"scope": "Contemporaneous saved-model local inference diagnostic, not accuracy evidence",
          "pass_order": plan["order"], "models_per_pass": 28, "datasets": 14,
          "public_prediction_bitwise_all_models_all_passes": True,
          "final_matches_compatibility_receipt_python_and_native": True,
          "source_plan_sha256": digest(SOURCE / "plan.json"), "rows": rows,
          "faster_datasets": sum(row["ctboost_speedup"] > 1 for row in rows),
          "median_ctboost_speedup": statistics.median(row["ctboost_speedup"] for row in rows),
          "median_catboost_period_ratio": statistics.median(row["catboost_baseline_period_to_final_period"] for row in rows),
          "median_control_adjusted_ctboost_speedup": statistics.median(row["control_adjusted_ctboost_speedup"] for row in rows),
          "median_final_ctboost_to_catboost": statistics.median(row["final_ctboost_to_catboost"] for row in rows),
          "median_baseline_ctboost_last_to_first": statistics.median(row["baseline_ctboost_last_to_first"] for row in rows),
          "median_catboost_last_to_first": statistics.median(row["catboost_last_to_first"] for row in rows),
          "pass_wall_seconds": [item["pass_wall_seconds"] for item in passes]}

TARGET.mkdir(parents=True, exist_ok=False)
for path in sorted(SOURCE.rglob("*")):
    if path.is_file():
        target = TARGET / "raw" / path.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert target.read_bytes() == path.read_bytes()
for original, relative in ((ADAPTER, "measure.py"),
                           (ROOT / ".tmp/release-readiness-20260908/run-timing-phase3.py", "orchestrate.py"),
                           (Path(__file__), "archive.py")):
    shutil.copyfile(original, TARGET / relative)
    assert (TARGET / relative).read_bytes() == original.read_bytes()
write(TARGET / "comparison.json", report)
(TARGET / ".gitattributes").write_text("* -text\n", encoding="utf-8", newline="\n")
table = ["| Dataset | Public CTBoost ms | Final CTBoost ms | Speedup | CatBoost control ratio | Final / CatBoost |",
         "| --- | ---: | ---: | ---: | ---: | ---: |"]
for row in rows:
    table.append(f"| {row['dataset']} | {row['baseline_ctboost_ms']:.3f} | {row['final_ctboost_ms']:.3f} | {row['ctboost_speedup']:.2f}x | {row['catboost_baseline_period_to_final_period']:.2f}x | {row['final_ctboost_to_catboost']:.2f}x |")
readme = f'''# Final-source saved-model inference timing

Four passes used baseline-final-final-baseline order on the same saved 14 CTBoost models and 14 AutoGluon CatBoost models. All predictions matched both archived public reference arrays bit for bit in all passes. No models were fitted; only the same 1,000 cyclic development-fold-6 rows were used. Confirmation fold 7 and outer-test rows remained unopened.

The final CTBoost candidate was faster on {report['faster_datasets']}/14 datasets; the median per-dataset speedup was {report['median_ctboost_speedup']:.2f}x. The unchanged CatBoost control had a median baseline-period/final-period ratio of {report['median_catboost_period_ratio']:.2f}x. The median CTBoost speedup divided by that task's control ratio was {report['median_control_adjusted_ctboost_speedup']:.2f}x. These describe this selected local panel, not general population speed or statistical significance.

{'\n'.join(table)}

Each cell uses the median of its two pass medians. Speedup is public/final CTBoost time; values above one favor the final candidate. The CatBoost control ratio is its baseline-period time divided by its final-period time, despite using the same CatBoost model and package in every pass. Final/CatBoost compares the two libraries during final passes; values below one favor CTBoost. The fitted tree counts differ across libraries and are recorded per model in the raw summaries. No equal-quality or equal-tree-count comparison is claimed.

The final wheel is `{plan['final_wheel_sha256']}`, native `{plan['final_native_sha256']}`, from production commit `{plan['source_commit']}`. Its installed Python file hashes and native bytes match the separate phase-three 14-model compatibility receipt. The public CTBoost native is `{plan['original_runtime']['native_sha256']}`. The explicit import loads only the final CTBoost package into the unchanged public baseline environment. Distribution metadata intentionally still reports ctboost 0.1.60 while the actual candidate module reports 0.1.61; every runtime receipt records both, with the actual package path and hashes. NumPy, pandas, scikit-learn, AutoGluon and CatBoost dependencies remain the same across arms.

CPU affinity was logical CPU 6, with all declared native/BLAS/OpenMP thread budgets set to one. The original frozen `measure_call` function performs three warmups, ten calibration calls and seven timed blocks. Normalization and model preprocessing are included; model loading and row selection are excluded. Cold latency means first prediction after each model is loaded, and its raw value is retained. RSS is the whole process after warmup, not an isolated model allocation or peak-memory measurement. Pass wall times, all seven block values, adaptive call counts, model/data/reference hashes and prediction-value hashes are preserved under `raw/`. Source data, labels, models and prediction arrays are excluded.

Repeated controls reveal remaining system variation. The median final/initial baseline CTBoost time was {report['median_baseline_ctboost_last_to_first']:.2f}x, and the same CatBoost last/first ratio was {report['median_catboost_last_to_first']:.2f}x. `comparison.json` retains every dataset's two ordering-specific speedups, both CTBoost repeat ratios, control ratios and raw pass medians. ABBA reduces simple ordering bias but does not eliminate shared-system noise. The panel was chosen earlier for development work and is not an independent holdout or leaderboard benchmark.

The plan and source hashes were frozen before pass 1. The adapter has no fitting entry point. Exact adapter bytes are archived as `measure.py`, with the orchestration and archiving scripts. The temporary untracked `benchmarks/inference_release_timing.py` is removed after archiving so this research helper is not shipped in the wheel. To reproduce, restore that exact source to the original recorded location in a separate research checkout, provide the existing hashed fixtures and matching environments, and choose a fresh output directory; existing evidence is never overwritten. `manifest.json` hashes all evidence files; `SHA256SUMS` additionally covers the manifest. Local Git attributes preserve exact bytes.
'''
(TARGET / "README.md").write_text(readme, encoding="utf-8", newline="\n")
files = {p.relative_to(TARGET).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
         for p in sorted(TARGET.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "SHA256SUMS")}
write(TARGET / "manifest.json", {"schema_version": 1, "files": files,
                                 "original_output": str(SOURCE), "original_adapter": str(ADAPTER)})
(TARGET / "SHA256SUMS").write_text("\n".join(digest(p) + "  " + p.relative_to(TARGET).as_posix()
                                             for p in sorted(TARGET.rglob("*")) if p.is_file() and p.name != "SHA256SUMS") + "\n",
                                  encoding="utf-8", newline="\n")
print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
