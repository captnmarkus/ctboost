"""Preregistered, isolated multiclass joint-cut development study.

Uses the previous eight multiclass tasks and immutable train/stop/development
roles. Confirmation fold seven is neither copied nor loaded. All three arms
are refitted; archived public-default and XGBoost outputs are audit references.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("ctboost_default", "joint_feature_scalar_cut", "joint_feature_joint_cut")
ROLES = ("train", "stop", "development")


def bootstrap():
    import ctboost

    if Path(ctboost.__file__).resolve().is_relative_to(ROOT / "ctboost"):
        raise RuntimeError("Use the isolated prototype Python with -I")
    sys.path.insert(0, str(ROOT))
    from benchmarks import accuracy_grouped_scout as base

    return base, ctboost


def package_hashes(base, ctboost):
    directory = Path(ctboost.__file__).parent
    return {str(path.relative_to(directory)): base.digest(path)
            for path in sorted(directory.rglob("*.py"))}


def checked_plan(output):
    base, ctboost = bootstrap()
    plan = base.read(output / "plan.json")
    if base.digest(__file__) != plan["runner_sha256"]:
        raise RuntimeError("Runner changed after registration")
    for name, digest in plan["dependency_sha256"].items():
        if base.digest(ROOT / name) != digest:
            raise RuntimeError(f"Frozen adapter changed: {name}")
    if base.digest(ctboost._core.__file__) != plan["native_sha256"]:
        raise RuntimeError("Prototype native runtime changed")
    if package_hashes(base, ctboost) != plan["installed_python_sha256"]:
        raise RuntimeError("Prototype Python package changed")
    if base.digest(plan["wheel"]) != plan["wheel_sha256"]:
        raise RuntimeError("Frozen prototype wheel changed")
    if str(Path(sys.executable).resolve()) != plan["python"]:
        raise RuntimeError("Wrong execution interpreter")
    return base, ctboost, plan


def load_role(base, output, case, role):
    if role not in ROLES:
        raise ValueError("This development runner cannot access confirmation")
    return base.load_role(output, case, role)


def prepare(output, predecessor, manifest):
    base, ctboost = bootstrap()
    import numpy as np

    if output.exists():
        raise ValueError("Preserve previous runs; use a new output directory")
    old = base.checked_plan(predecessor)
    build = base.read(manifest)
    if build["native_sha256"] != base.digest(ctboost._core.__file__) or build["wheel_sha256"] != base.digest(build["wheel"]):
        raise ValueError("Finalized wheel does not match the installed native runtime")
    if build["validation"]["focused_plus_relevant_existing_suites"]["passed"] != 208:
        raise ValueError("Finalized prototype validation receipt is missing")
    equivalence = Path(build["validation"]["public_default_equivalence_file"])
    probes = base.read(equivalence)
    if len(probes["checks"]) != 8 or not all(row["full_state_and_prediction_bits_equal"] for row in probes["checks"]):
        raise ValueError("Default prototype equivalence probes did not pass")
    common = {**old["ctboost_params"], "eval_metric": "MultiClass",
              "feature_test_adjustment": "none", "feature_test_bins": 8,
              "multiclass_leaf_solver": "diagonal", "leaf_estimation_iterations": 1,
              "leaf_estimation_backtracking": False, "multi_strategy": "one_output_per_tree"}
    parameters = {
        ARMS[0]: {**common, "feature_test": "quadratic", "multiclass_feature_test": "single", "multiclass_split_score": "scalar"},
        ARMS[1]: {**common, "feature_test": "grouped", "multiclass_feature_test": "joint", "multiclass_split_score": "scalar"},
        ARMS[2]: {**common, "feature_test": "grouped", "multiclass_feature_test": "joint", "multiclass_split_score": "joint"},
    }
    if [key for key in parameters[ARMS[1]] if parameters[ARMS[1]][key] != parameters[ARMS[2]][key]] != ["multiclass_split_score"]:
        raise ValueError("The paired experimental arms must differ only in cut scoring")
    cases = []
    for original_case in old["datasets"]:
        if original_case["problem_type"] != "multiclass":
            continue
        case = {**original_case, "roles": {role: original_case["roles"][role] for role in ROLES}, "archived_references": {}}
        all_indices = []
        for role in ROLES:
            payload = load_role(base, predecessor, case, role)
            all_indices.append(payload["outer_train_positions"])
            if role == "train" and len(np.unique(payload["y"])) != case["num_classes"]:
                raise ValueError("Training role lacks a class")
            target = output / "data" / case["dataset_name"] / (role + ".pkl")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(predecessor / "data" / case["dataset_name"] / (role + ".pkl"), target)
            if base.digest(target) != case["roles"][role]["sha256"]:
                raise ValueError("Copied immutable role changed")
        joined = np.concatenate(all_indices)
        if len(np.unique(joined)) != len(joined):
            raise ValueError("Fit, stopping, and development roles overlap")
        for arm in ("ctboost_default", "xgboost_ag_default"):
            directory = predecessor / "fits" / case["dataset_name"] / arm
            record = base.read(directory / "result.json")
            if record["status"] != "ok":
                raise ValueError("Expected a valid archived reference")
            case["archived_references"][arm] = {"record_sha256": base.digest(directory / "result.json"),
                "model_sha256": record["model_sha256"], "prediction_sha256": record["development_sha256"]}
            if arm == "ctboost_default":
                # Compare effective constructor controls without reading scores.
                previous = ctboost.CTBoostClassifier(**record["parameters"]).get_params()
                current = ctboost.CTBoostClassifier(**parameters[ARMS[0]], cat_features=record["parameters"].get("cat_features")).get_params()
                if previous != current:
                    raise ValueError("New default controls differ from the archived default")
        cases.append(case)
    if len(cases) != 8:
        raise ValueError("Include all eight multiclass tasks, without filtering")
    dependencies = ["benchmarks/accuracy_grouped_scout.py", "benchmarks/tabarena/ctboost_model.py", "benchmarks/tabarena/learning_options.py"]
    plan = {"protocol_id": "ctboost_joint_cut_multiclass_development_20260908_v2", "created_at": base.now(),
        "prelaunch_registration": "Replaces preparation300dd07a06b8f757178363f6b24c1a9c4eda0d2e004bd6b6465013457bc63dc3 before any fits or new scores; adds physical-tree counts and import-format lint correction only. Original plan, runner, and withdrawal preserved.",
        "runner_sha256": base.digest(__file__), "dependency_sha256": {p: base.digest(ROOT / p) for p in dependencies},
        "python": str(Path(sys.executable).resolve()), "installed_python_sha256": package_hashes(base, ctboost),
        "native_sha256": build["native_sha256"], "wheel": build["wheel"], "wheel_sha256": build["wheel_sha256"],
        "source_sha256": build["source_sha256"], "prototype_manifest_sha256": base.digest(manifest),
        "prototype_default_equivalence_sha256": base.digest(equivalence),
        "versions": {name: importlib.metadata.version(name) for name in ("ctboost", "numpy", "pandas", "scikit-learn", "autogluon.tabular", "xgboost", "psutil")},
        "predecessor": str(predecessor), "predecessor_plan_sha256": base.digest(predecessor / "plan.json"),
        "selection": "All eight multiclass tasks from the original14 metadata population; no task or arm selection from development outcomes.",
        "roles": "Reuse immutable original trainfolds0-4, stopping5, development6. Original cap10000 and CV8seed47 retained. Fold7 is neither copied nor loaded, regardless of this result.",
        "reuse_disclosure": "New preregistered hypothesis after the failed grouped and temperature studies on these same development tasks/rows. This is staged exploratory development, not independent confirmation or official outer-test/Elo evidence. Archived default predictions audit new-build equivalence; archived XGBoost is a secondary comparison only. All24 CTBoost arms are refitted in the new isolated build.",
        "arms": list(ARMS), "parameters": parameters, "datasets": cases, "expected_fits": 24,
        "resources": {"fit_seconds": 300, "hard_wall_seconds": 390, "rss_bytes": 8 * 1024 ** 3,
                      "threads": 2, "max_workers": 2, "affinities": [[0, 8], [1, 9]], "early_stopping_rounds": 50},
        "gate": {"comparator": ARMS[1], "candidate": ARMS[2], "minimum_median_relative_error_gain": .01,
                 "minimum_dataset_wins": 5, "maximum_relative_regression": .05,
                 "maximum_geometric_fit_ratio": 3, "maximum_p90_fit_ratio": 5,
                 "all24_finite_successful": True, "all8_default_prediction_and_iteration_audits_exact": True},
        "retry_policy": "Each started arm is terminal; preserve errors and resource failures without tuning or retries.",
        "promotion_policy": "Root reviews the complete report before any promotion. No global default changes or confirmation unsealing are authorized by this gate.",
    }
    base.write(output / "plan.json", plan)
    shutil.copyfile(manifest, output / "prototype-manifest.json")
    shutil.copyfile(equivalence, output / "prototype-default-equivalence.json")
    print(json.dumps({"ready": True, "fits": 24, "plan_sha256": base.digest(output / "plan.json")}))


def fit(output, dataset, arm, slot):
    import psutil

    psutil.Process().cpu_affinity([[0, 8], [1, 9]][slot])
    base, ctboost, plan = checked_plan(output)
    import numpy as np

    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

    case = next(case for case in plan["datasets"] if case["dataset_name"] == dataset)
    directory = output / "fits" / dataset / arm
    if directory.exists():
        raise ValueError("Started arms are terminal; do not retry")
    directory.mkdir(parents=True)
    record = {"dataset": dataset, "arm": arm, "started_at": base.now(), "status": "failed",
              "plan_sha256": base.digest(output / "plan.json"), "affinity": [[0, 8], [1, 9]][slot], "soft_deadline_stopped": False}
    base.write(directory / "started.json", record)
    try:
        training = load_role(base, output, case, "train")
        stopping = load_role(base, output, case, "stop")
        X, categoricals = normalize_tabarena_frame(training["X"])
        X_stop, _ = normalize_tabarena_frame(stopping["X"], categorical_columns=categoricals)
        parameters = {**plan["parameters"][arm], "cat_features": categoricals or None}
        model = ctboost.CTBoostClassifier(**parameters)
        started = time.perf_counter()

        def deadline(env):
            elapsed = time.perf_counter() - started
            stop = elapsed + 2 * elapsed / (env.iteration + 1) >= plan["resources"]["fit_seconds"]
            record["soft_deadline_stopped"] |= stop
            return stop

        model.fit(X, training["y"], eval_set=(X_stop, stopping["y"]),
                  early_stopping_rounds=50, callbacks=[deadline])
        record.update(fit_seconds=time.perf_counter() - started,
                      rounds=model.get_booster().num_iterations_trained,
                      physical_trees=int(model.get_booster()._handle.num_trees()), parameters=parameters)
        payload = {"model": model, "arm": arm, "categorical_columns": categoricals}
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(payload, stream, protocol=5)
        development = load_role(base, output, case, "development")
        prediction = base.predict_saved(payload, development["X"], case)
        error = base.score_predictions(development["y"], prediction, case)
        with (directory / "model.pkl").open("rb") as stream:
            loaded = pickle.load(stream)
        np.testing.assert_array_equal(base.predict_saved(loaded, development["X"], case), prediction)
        np.savez_compressed(directory / "development.npz", predictions=prediction, labels=development["y"],
                            outer_train_positions=development["outer_train_positions"])
        record.update(status="ok", development_error=error, model_sha256=base.digest(directory / "model.pkl"),
                      development_sha256=base.digest(directory / "development.npz"), prediction_reload_exact=True)
        if arm == ARMS[0]:
            previous = Path(plan["predecessor"]) / "fits" / dataset / "ctboost_default"
            previous_record = base.read(previous / "result.json")
            reference = case["archived_references"]["ctboost_default"]
            if base.digest(previous / "result.json") != reference["record_sha256"] or base.digest(previous / "development.npz") != reference["prediction_sha256"]:
                raise ValueError("Archived default changed")
            with np.load(previous / "development.npz") as archived:
                record["default_predictions_exact"] = bool(np.array_equal(prediction, archived["predictions"]))
                record["default_prediction_max_abs_difference"] = float(np.max(np.abs(prediction - archived["predictions"])))
            record["default_rounds_exact"] = record["rounds"] == previous_record["rounds"]
    except Exception:  # noqa: BLE001 - keep every predeclared failed arm visible
        record.update(status="failed", error=traceback.format_exc())
    record["finished_at"] = base.now()
    base.write(directory / "result.json", record)


def run(output):
    base, _, plan = checked_plan(output)
    import psutil

    jobs = [(case["dataset_name"], arm) for case in plan["datasets"] for arm in ARMS]
    random.Random(47).shuffle(jobs)
    if any((output / "fits" / dataset / arm).exists() for dataset, arm in jobs):
        raise ValueError("A started run is terminal; do not resume or retry this protocol")
    env = dict(os.environ, CTBOOST_HIST_THREADS="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def worker(slot):
        for dataset, arm in jobs[slot::2]:
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(output),
                       "--dataset", dataset, "--arm", arm, "--slot", str(slot)]
            log = output / "logs" / (dataset + "_" + arm + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            peak, reason, started = 0, None, time.monotonic()
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                owner = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        peak = max(peak, sum(p.memory_info().rss for p in [owner] + owner.children(recursive=True)))
                    except psutil.NoSuchProcess:
                        pass
                    if peak > plan["resources"]["rss_bytes"]:
                        reason = "memory_limit"
                    elif time.monotonic() - started > plan["resources"]["hard_wall_seconds"]:
                        reason = "hard_timeout"
                    if reason:
                        for child in owner.children(recursive=True):
                            child.kill()
                        owner.kill()
                        process.wait(timeout=15)
                        break
                    time.sleep(.25)
            result_path = output / "fits" / dataset / arm / "result.json"
            record = base.read(result_path) if result_path.exists() else {"dataset": dataset, "arm": arm,
                "status": "failed", "plan_sha256": base.digest(output / "plan.json")}
            record.update(peak_rss_bytes=peak, exit_code=process.returncode, resource_failure=reason)
            if reason or process.returncode != 0:
                record["status"] = "failed"
            base.write(result_path, record)
            print(json.dumps({"finished": dataset + "/" + arm, "status": record["status"]}), flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, range(2)))


def report(output):
    base, _, plan = checked_plan(output)
    import numpy as np

    rows, records, ratios, audit = [], [], [], []
    for case in plan["datasets"]:
        development = load_role(base, output, case, "development")
        errors, paired = {}, {}
        for arm in ARMS:
            directory = output / "fits" / case["dataset_name"] / arm
            record = base.read(directory / "result.json")
            records.append(record)
            paired[arm] = record
            if record["status"] != "ok" or record.get("resource_failure"):
                continue
            if record["plan_sha256"] != base.digest(output / "plan.json") or base.digest(directory / "model.pkl") != record["model_sha256"] or base.digest(directory / "development.npz") != record["development_sha256"]:
                raise ValueError("Recorded artifact changed")
            with np.load(directory / "development.npz") as saved:
                np.testing.assert_array_equal(saved["labels"], development["y"])
                np.testing.assert_array_equal(saved["outer_train_positions"], development["outer_train_positions"])
                error = base.score_predictions(saved["labels"], saved["predictions"], case)
                if abs(error - record["development_error"]) > 1e-12:
                    raise ValueError("Metric differs from saved predictions")
            errors[arm] = error
        reference_dir = Path(plan["predecessor"]) / "fits" / case["dataset_name"] / "xgboost_ag_default"
        reference = case["archived_references"]["xgboost_ag_default"]
        if base.digest(reference_dir / "result.json") != reference["record_sha256"] or base.digest(reference_dir / "development.npz") != reference["prediction_sha256"]:
            raise ValueError("Archived XGBoost reference changed")
        with np.load(reference_dir / "development.npz") as saved:
            np.testing.assert_array_equal(saved["labels"], development["y"])
            np.testing.assert_array_equal(saved["outer_train_positions"], development["outer_train_positions"])
            errors["archived_xgboost_ag_default"] = base.score_predictions(saved["labels"], saved["predictions"], case)
        row = {"dataset": case["dataset_name"], "errors": errors}
        if all(arm in errors for arm in ARMS):
            row["candidate_relative_gain"] = (errors[ARMS[1]] - errors[ARMS[2]]) / max(errors[ARMS[1]], 1e-12)
            row["candidate_vs_default_gain"] = (errors[ARMS[0]] - errors[ARMS[2]]) / max(errors[ARMS[0]], 1e-12)
            row["control_vs_default_gain"] = (errors[ARMS[0]] - errors[ARMS[1]]) / max(errors[ARMS[0]], 1e-12)
            row["candidate_vs_xgboost_gain"] = (errors["archived_xgboost_ag_default"] - errors[ARMS[2]]) / max(errors["archived_xgboost_ag_default"], 1e-12)
            ratios.append(paired[ARMS[2]]["fit_seconds"] / max(paired[ARMS[1]]["fit_seconds"], 1e-12))
        audit.append({"dataset": case["dataset_name"], "default_predictions_exact": paired[ARMS[0]].get("default_predictions_exact", False),
                      "default_rounds_exact": paired[ARMS[0]].get("default_rounds_exact", False)})
        rows.append(row)
    gains = [r["candidate_relative_gain"] for r in rows if "candidate_relative_gain" in r]
    complete = len(records) == 24 and len(gains) == 8 and all(r["status"] == "ok" and not r.get("resource_failure") for r in records)
    defaults_exact = all(a["default_predictions_exact"] and a["default_rounds_exact"] for a in audit)
    stats = {"median_relative_gain": float(np.median(gains)) if gains else None,
             "wins": sum(g > 1e-12 for g in gains), "worst_relative_regression": max([0.] + [-g for g in gains]),
             "geometric_fit_ratio": float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
             "p90_fit_ratio": float(np.quantile(ratios, .9)) if ratios else None}
    gate = plan["gate"]
    passed = complete and defaults_exact and stats["median_relative_gain"] >= gate["minimum_median_relative_error_gain"] and stats["wins"] >= gate["minimum_dataset_wins"] and stats["worst_relative_regression"] <= gate["maximum_relative_regression"] and stats["geometric_fit_ratio"] <= gate["maximum_geometric_fit_ratio"] and stats["p90_fit_ratio"] <= gate["maximum_p90_fit_ratio"]
    result = {"created_at": base.now(), "plan_sha256": base.digest(output / "plan.json"),
              "stage": "reused8task_multiclass_joint_cut_development", "complete_success": complete,
              "all8_defaults_exact": defaults_exact, "gate_passed": bool(passed), "statistics": stats,
              "datasets": rows, "default_equivalence_audit": audit,
              "failures": [r for r in records if r["status"] != "ok" or r.get("resource_failure")],
              "confirmation_scored": False, "reuse_disclosure": plan["reuse_disclosure"]}
    base.write(output / "development_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--slot", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output.resolve(), args.predecessor.resolve(), args.manifest.resolve())
    elif args.stage == "fit":
        fit(args.output.resolve(), args.dataset, args.arm, args.slot)
    elif args.stage == "run":
        run(args.output.resolve())
    else:
        report(args.output.resolve())
