"""Export reproducible CTBoost controls only after both development gates pass."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(study, seed_check, output):
    plan, frozen, development = (read(study / name) for name in
        ("plan.json", "frozen_selections.json", "development.json"))
    seed_plan, seed_frozen, seed_development = (read(seed_check / name) for name in
        ("plan.json", "frozen_selections.json", "development.json"))
    for directory, state, report in ((study, frozen, development), (seed_check, seed_frozen, seed_development)):
        if (not state["all_fits_successful"] or not report["development_target_met"]
                or state["plan_sha256"] != digest(directory / "plan.json")
                or report["plan_sha256"] != digest(directory / "plan.json")
                or report["frozen_selections_sha256"] != digest(directory / "frozen_selections.json")):
            raise ValueError("Both complete frozen development gates must pass")
    if Path(seed_plan["previous"]).resolve() != study.resolve() or plan["seed"] != 47:
        raise ValueError("Export the primary seed 47 policy linked to its fixed seed check")
    models = {s["dataset"]: s for s in frozen["selections"] if s["library"] == "ctboost"}
    jobs = [j for j in frozen["jobs"] if j["library"] == "ctboost"]
    if len(jobs) != 12 or len(models) != 12 or {j["dataset"] for j in jobs} != set(models):
        raise ValueError("Expected the complete twelve-dataset CTBoost policy")
    configurations = []
    for job in jobs:
        model = models[job["dataset"]]
        if digest(model["model_path"]) != model["model_sha256"]:
            raise ValueError("Frozen selected model changed")
        parameters = copy.deepcopy(job["parameters"])
        parameters.update(iterations=job["rounds"], random_seed=47)
        configurations.append({"dataset": job["dataset"], "parameters": parameters,
            "requested_rounds": job["rounds"],
            "fixed_postfit_controls": {key: job.get(key) for key in
                ("leaf_refit", "binary_refit", "multiclass_refit", "temperature")},
            "native_model_path": model["model_path"], "native_model_sha256": model["model_sha256"],
            "model_bytes": model["model_bytes"], "expanded_source": job.get("expanded_source")})
    result = {"primary_seed": 47, "protocol": plan["protocol"], "runtime": plan["runtime"],
        "source_study": str(study.resolve()), "source_seed_check": str(seed_check.resolve()),
        "source_sha256": {str(directory / name): digest(directory / name)
            for directory in (study, seed_check) for name in ("plan.json", "frozen_selections.json", "development.json")},
        "controls": configurations,
        "usage": "Dataset-specific exploratory policies, not new library defaults. The fitting runner defines raw data normalization and ordered categorical fitting. Non-null postfit controls require the frozen helper and fitting policy; they are not constructor parameters. Use the native model and its saved schema for inference.",
        "limits": "Passing development gates does not establish an official-test match or a universal ranking. Official assessment is reported separately. No model or seed is selected using latency."}
    if output.exists():
        raise ValueError("Preserve an existing export")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "seed-check", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    export(args.study, args.seed_check, args.output)
