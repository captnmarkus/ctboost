"""Frozen wave8: grouped-test granularity and matched comparator regularization."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

BASE_PATH = Path(__file__).with_name("fair_accuracy_scout.py")
TEMPERATURE_PATH = Path(__file__).with_name("fixed_temperature.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("power_base", BASE_PATH)
temperature = load_module("power_temperature", TEMPERATURE_PATH)
raw_predict = base.predict


def predict(payload, frame):
    return temperature.apply_temperature(payload, raw_predict(payload, frame))


def configurations(library):
    if library == "ctboost":
        return [{"learning_rate": .05, "max_depth": 8, "lambda_l2": 1., "subsample": 1.,
                 "bootstrap_type": "Bernoulli", "alpha": alpha, "min_data_in_leaf": 5,
                 "feature_test": "grouped", "feature_test_bins": bins,
                 "leaf_estimation_iterations": 3, "leaf_estimation_backtracking": True,
                 "multi_strategy": "one_output_per_tree"}
                for bins in (2, 4, 16, 32) for alpha in (.2, .5)]
    if library == "xgboost":
        return [{"eta": .05, "max_depth": 6, "min_child_weight": 1., "lambda": 1.,
                 "alpha": l1, "colsample_bytree": fraction, "subsample": 1.,
                 "multi_strategy": "one_output_per_tree"}
                for l1 in (0., 1., 10., 100.) for fraction in (.8, 1.)]
    return [{"learning_rate": .05, "num_leaves": 31, "min_data_in_leaf": 20,
             "lambda_l2": 1., "lambda_l1": l1, "feature_fraction": fraction,
             "bagging_fraction": 1., "bagging_freq": 1}
            for l1 in (0., 1., 10., 100.) for fraction in (.8, 1.)]


def duplicate_audit(previous, cases):
    """Check all earlier successful fit parameter receipts, without new fits."""
    dependencies, matches = {}, []
    defaults = {"ctboost": {"leaf_estimation_iterations": 1, "leaf_estimation_backtracking": True,
                            "multi_strategy": "one_output_per_tree"},
                "xgboost": {"alpha": 0., "colsample_bytree": 1., "min_child_weight": 1.,
                            "multi_strategy": "one_output_per_tree"},
                "lightgbm": {"lambda_l1": 0., "feature_fraction": 1., "min_data_in_leaf": 20}}
    directory = previous
    while directory:
        plan = base.read(directory / "plan.json")
        dependencies[str(directory / "plan.json")] = base.digest(directory / "plan.json")
        if plan["wave"] <= 5:
            for case in cases:
                for library in base.LIBRARIES:
                    for path in sorted((directory / "fits" / case["dataset_name"] / library).glob("config_*.json")):
                        record = base.read(path)
                        if record["status"] != "ok":
                            continue
                        parameters = {**defaults[library], **record["parameters"]}
                        for index, controls in enumerate(configurations(library)):
                            if all(parameters.get(key) == value for key, value in controls.items()):
                                matches.append({"dataset": case["dataset_name"], "library": library,
                                                "new_config_index": index, "prior_receipt": str(path),
                                                "prior_receipt_sha256": base.digest(path)})
        directory = Path(plan["previous"]) if plan["previous"] else None
    return dependencies, {"matching_declared_controls": matches,
        "meaning": "A match is a potential duplicate under the new declared controls; other old options may differ. No matches rules out exact full-config duplicates under these controls. Defaults filled only for listed standard omitted L1/column sampling/minimum leaf/leaf solver controls; no prediction-equivalence claim."}


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts; choose fresh paths")
    previous = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (previous["wave"] != 7 or previous["runtime"] != runtime or not frozen["all_fits_successful"]
            or development["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Expected intact failed wave7 development gate")
    cases = previous["cases"]
    expected = {(case["dataset_name"], library) for case in cases for library in base.LIBRARIES}
    if len(cases) != 12 or len(expected) != 36 or len(frozen["selections"]) != 36 or {
            (s["dataset"], s["library"]) for s in frozen["selections"]} != expected:
        raise ValueError("Study population changed")
    for selected in frozen["selections"]:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Parent model changed")
    active = [case for case in cases if case["problem_type"] != "multiclass"]
    if len(active) != 8:
        raise ValueError("Expected eight binary/regression tasks")
    dependencies, audit = duplicate_audit(args.previous, active)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("frozen_selections.json", "development.json")})
    plan = {"protocol": "grouped_granularity_and_comparator_regularization_20261016", "wave": 8,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "base_script_sha256": base.digest(BASE_PATH),
        "temperature_helper_sha256": base.digest(TEMPERATURE_PATH), "runtime": runtime,
        "previous": str(args.previous), "dependencies": dependencies, "cases": cases,
        "active_datasets": [case["dataset_name"] for case in active], "seed": 47, "round_cap": 2000, "patience": 50,
        "candidate_grids": {library: {problem: configurations(library) for problem in ("binary", "regression")}
                            for library in base.LIBRARIES}, "duplicate_audit": audit,
        "resources": {"workers": 2, "threads_per_fit": 1, "cpus": [4, 6],
                      "maximum_job_seconds": 3600, "maximum_job_rss_bytes": 8 * 1024 ** 3},
        "target": previous["target"], "official_test_scored": False,
        "selection": "Eight added tree fits per library on each of eight binary/regression tasks. Within-wave STOP ties prefer fewer rounds then config index. Compare best new model to frozen incumbent by STOP error; incumbent wins exact ties. Carry all twelve calibrated multiclass models unchanged. Freeze all36models before DEV.",
        "hypothesis": "Group granularity changes association-test degrees of freedom and local power;2/4/32 are additional granularities,16 was explored earlier with different controls. Three scalar leaf steps are fixed as requested; for unconstrained squared loss they are not a new optimization mechanism. Comparators receive matched-count L1/feature-sampling portfolios.",
        "limitations": "Adaptive historically reused development study; equal192 added fits, not equal compute or capacity. CT categorical defaults remain those of the frozen base runner. No new correction or temperature search for newly chosen tree models; retained incumbent policies may already contain a frozen correction. Four multiclass policies stay fixed and use the shared temperature helper."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fits": 192, "potential_duplicates": len(audit["matching_declared_controls"])})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if args.previous != Path(plan["previous"]):
        raise ValueError("Command-line parent differs from frozen parent")
    if (plan["script_sha256"] != base.digest(__file__) or plan["base_script_sha256"] != base.digest(BASE_PATH)
            or plan["temperature_helper_sha256"] != base.digest(TEMPERATURE_PATH) or plan["runtime"] != base.bootstrap(args)):
        raise ValueError("Frozen implementation/runtime changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Parent evidence changed")
    return plan


def fit(args):
    plan = checked(args)
    if args.dataset not in plan["active_datasets"]:
        raise ValueError("Multiclass models must be carried unchanged")
    base.checked = checked
    base.fit_job(args)


def run(args):
    import psutil
    plan = checked(args)
    if (args.output / "fits").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "logs").mkdir()
    jobs = [(dataset, library) for dataset in plan["active_datasets"] for library in base.LIBRARIES]

    def worker(slot):
        for dataset, library in jobs[slot::2]:
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(args.output),
                "--artifacts", str(args.artifacts), "--package-root", str(args.package_root), "--dependencies", str(args.dependencies),
                "--previous", str(args.previous), "--dataset", dataset, "--library", library, "--slot", str(slot)]
            started, peak, failure = time.monotonic(), 0, None
            with (args.output / "logs" / f"{dataset}-{library}.txt").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                owner = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        peak = max(peak, owner.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic() - started > plan["resources"]["maximum_job_seconds"] or peak > plan["resources"]["maximum_job_rss_bytes"]:
                        failure = "job_resource_limit"
                        owner.kill()
                        process.wait(timeout=15)
                        break
                    time.sleep(.5)
            base.write(args.output / "logs" / f"{dataset}-{library}-resource.json", {"exit_code": process.returncode,
                "resource_failure": failure, "peak_rss_bytes": peak, "wall_seconds": time.monotonic() - started})
            print({"completed": dataset, "library": library, "exit_code": process.returncode}, flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, range(2)))
    parent = base.read(Path(plan["previous"]) / "frozen_selections.json")
    selections, successful = [], parent["all_fits_successful"]
    for original in parent["selections"]:
        if original["dataset"] not in plan["active_datasets"]:
            selections.append(original)
            continue
        current = base.read(args.output / "fits" / original["dataset"] / original["library"] / "selection.json")
        chosen = current if current["best"] is not None and current["best"]["stop_error"] < original["best"]["stop_error"] else original
        successful = successful and current["all_eight_successful"]
        selections.append({**chosen, "current_wave_all_eight_successful": current["all_eight_successful"],
            "cumulative_search_fit_seconds": current["search_fit_seconds"] + original.get("cumulative_search_fit_seconds", original.get("search_fit_seconds", 0.))})
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "selections": selections, "all_fits_successful": successful,
        "development_scored": False, "official_test_scored": False})


def development(args):
    if not base.read(args.output / "frozen_selections.json")["all_fits_successful"]:
        raise ValueError("All192 additional fits must succeed before development assessment")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "fit", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=base.LIBRARIES)
    parser.add_argument("--slot", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "fit": fit, "development": development}[args.stage](args)
