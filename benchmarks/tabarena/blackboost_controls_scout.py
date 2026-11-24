"""Compare the manual CTBoost default with tree-size and sampling controls inspired by mboost::blackboost."""

import argparse
import hashlib
import json
import os
import platform
import sys
from functools import partial
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_model_development import run_task
from quantile_ties_scout import STUDIES, fit, verified_bytes

VARIANTS = {
    "baseline": {"max_depth": 6, "min_samples_split": 2, "min_data_in_leaf": 0, "subsample": 0.8},
    "shallow": {"max_depth": 2, "min_samples_split": 10, "min_data_in_leaf": 4, "subsample": 0.8},
    "all_rows": {"max_depth": 6, "min_samples_split": 2, "min_data_in_leaf": 0, "subsample": 1.0},
}
REFERENCE = Path(__file__).resolve().parents[1] / "results/four_model_development_20261117.json"


def summarize(rows):
    group = {"tasks": len(rows)}
    for variant in VARIANTS:
        values = [row[variant] for row in rows]
        summary = {
            "mean_rank": mean(arm["four_model_rank"] for arm in values),
            "at_round_cap": sum(arm["rounds"] == 200 for arm in values),
        }
        if variant != "baseline":
            paired = [(row["baseline"]["error"], row[variant]["error"]) for row in rows]
            reductions = [(base - trial) / base * 100 for base, trial in paired if base > 0]
            summary.update(
                wins=sum(trial < base for base, trial in paired),
                ties=sum(trial == base for base, trial in paired),
                losses=sum(trial > base for base, trial in paired),
                positive_baseline_error_tasks=len(reductions),
                mean_relative_error_reduction_pct=mean(reductions) if reductions else None,
                median_relative_error_reduction_pct=median(reductions) if reductions else None,
            )
        group[variant] = summary
    return group


def run(cache):
    import ctboost

    reference_text = REFERENCE.read_text(encoding="utf-8")
    published = json.loads(reference_text)
    previous = {task["dataset"]: task for task in published["tasks"]}
    rows = []
    for study, digest in STUDIES.items():
        directory = cache / study
        plan = json.loads(verified_bytes(directory / "plan.json", digest))
        for case in plan["datasets"]:
            name = case["dataset_name"]
            row = {"dataset": name, "problem": case["problem_type"]}
            models = previous[name]["models"]
            rivals = [model["error"] for library, model in models.items() if library != "ctboost"]
            for variant, settings in VARIANTS.items():
                error, rounds, fingerprint = run_task(
                    "ctboost", directory, case, ctboost_fitter=partial(fit, **settings)
                )
                if variant == "baseline":
                    old = models["ctboost"]
                    expected = (old["error"], old["selected_rounds"], old["prediction_sha256"])
                    if (error, rounds, fingerprint) != expected:
                        raise ValueError(f"Baseline disagrees with frozen predictions on {name}")
                rank = 1 + sum(r < error for r in rivals) + 0.5 * sum(r == error for r in rivals)
                row[variant] = {
                    "error": error,
                    "rounds": rounds,
                    "prediction_sha256": fingerprint,
                    "four_model_rank": rank,
                }
            rows.append(row)
    if len(rows) != 51 or {r["dataset"] for r in rows} != set(previous):
        raise ValueError("Incomplete or mismatched task roster")
    return {
        "protocol": {
            **published["protocol"],
            "variants": VARIANTS,
            "reference_file": REFERENCE.name,
            "reference_sha256_lf": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            "ctboost": ctboost.__version__,
            "python": platform.python_version(),
            "native_sha256": hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
        },
        "groups": {
            group: summarize([row for row in rows if group == "all" or row["problem"] == group])
            for group in ("all", "binary", "multiclass", "regression")
        },
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for variable in (
        "CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    result = run(args.cache)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
