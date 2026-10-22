"""Fixed two-seed ensembles of the mandatory wave10 policy; no new selection."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import sys
import time
import traceback

BINARY_PATH = Path(__file__).with_name("fair_accuracy_binary_refit.py")
HELPER_PATH = Path(__file__).with_name("fixed_seed_ensemble.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_ensemble.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binary = load_module("ensemble_binary", BINARY_PATH)
base, TEMPERATURE_PATH = binary.base, binary.union.TEMPERATURE_PATH


def helper():
    return load_module("frozen_seed_ensemble", HELPER_PATH)


def predict(payload, frame):
    return helper().predict(payload, frame)


def identities(cases):
    expected = {(c["dataset_name"], library) for c in cases for library in base.LIBRARIES}
    if len(cases) != 12 or len(expected) != 36:
        raise ValueError("Expected12 unique tasks")
    return expected


def verify_selections(selections, expected):
    if len(selections) != 36 or {(s["dataset"], s["library"]) for s in selections} != expected:
        raise ValueError("Expected36 unique frozen models")
    for selected in selections:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Frozen model changed")


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts; use fresh paths")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (parent["runtime"] != runtime or parent["scipy_version"] != importlib.metadata.version("scipy")
            or not frozen["all_fits_successful"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Parent evidence/runtime changed")
    if args.phase == "initial":
        if (parent["wave"] != 10 or parent["script_sha256"] != base.digest(BINARY_PATH)
                or development["development_target_met"] or parent["seed"] != 47
                or frozen["binary_controls_sha256"] != base.digest(args.previous / "binary_controls.json")):
            raise ValueError("This new hypothesis must follow the failed wave10 policy")
        pair, new_seeds = [47, 947], [947]
    else:
        if (parent["protocol"] != "fixed_two_seed_ensemble_20261016" or parent["phase"] != "initial"
                or parent["script_sha256"] != base.digest(__file__) or not development["development_target_met"]):
            raise ValueError("The first fixed ensemble pair must pass before replication")
        pair = new_seeds = [137, 1543]
    expected = identities(parent["cases"])
    verify_selections(frozen["selections"], expected)
    dependencies = dict(parent["dependencies"])
    for path, expected_hash in dependencies.items():
        if base.digest(path) != expected_hash:
            raise ValueError("Parent implementation changed")
    for path in (BINARY_PATH, HELPER_PATH, POLICY, Path(__file__)):
        dependencies[str(path)] = base.digest(path)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")})
    if args.phase == "initial":
        dependencies[str(args.previous / "binary_controls.json")] = base.digest(args.previous / "binary_controls.json")
    jobs = []
    for job in parent["jobs"]:
        selected = next(s for s in frozen["selections"] if (s["dataset"], s["library"]) == (job["dataset"], job["library"]))
        jobs.append({**job, "binary_refit": selected.get("binary_refit") if args.phase == "initial" else job.get("binary_refit")})
    if len(jobs) != 36 or {(j["dataset"], j["library"]) for j in jobs} != expected:
        raise ValueError("Expected36 fixed tree policies")
    plan = {"protocol": "fixed_two_seed_ensemble_20261016", "wave": 11, "phase": args.phase,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": parent["cases"], "jobs": jobs, "seed_pair": pair, "new_seeds": new_seeds,
        "base_selections": frozen["selections"], "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4]},
        "official_test_scored": False,
        "selection": "None. Mandatory equal-weight two-seed ensemble for every library/task; fixed union rows, original selected tree parameters/rounds, binary/regression corrections and multiclass temperatures.",
        "limitations": "Seed947 is a component of a new hypothesis after the failed wave10 gate, not a successful earlier confirmation. All adaptive development failures remain preserved. Two ensembles on the same split are not independent datasets. Extra native models increase storage/inference cost."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "new_tree_fits": 36 * len(new_seeds), "seed_pair": pair})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy")
            or args.previous != Path(plan["previous"]) or args.phase != plan["phase"]):
        raise ValueError("Frozen code/runtime/policy changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def persist(payload, path, prediction_function, frame):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".candidate.pkl")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    with temporary.open("rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(prediction_function(payload, frame), prediction_function(restored, frame))
    temporary.replace(path)
    return {"model_path": str(path), "model_sha256": base.digest(path), "model_bytes": path.stat().st_size}


def run(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    if (args.output / "fits").exists():
        raise ValueError("Preserve every started attempt")
    (args.output / "fits").mkdir()
    components, selections, successful = [], [], True
    for job in plan["jobs"]:
        case = next(c for c in plan["cases"] if c["dataset_name"] == job["dataset"])
        member_records = []
        for seed in plan["seed_pair"]:
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            record = {"dataset": job["dataset"], "library": job["library"], "seed": seed, "status": "failed", "started_at": base.now()}
            try:
                if seed == 47:
                    original = next(s for s in plan["base_selections"] if (s["dataset"], s["library"]) == (job["dataset"], job["library"]))
                    if base.digest(original["model_path"]) != original["model_sha256"]:
                        raise ValueError("Seed47 native model changed")
                    record.update({key: original[key] for key in ("model_path", "model_sha256", "model_bytes")}, status="ok", reused=True)
                else:
                    payload, details, frame = binary.fit_policy(job, case, seed)
                    path = args.artifacts / "components" / job["dataset"] / f"{job['library']}-seed{seed}.pkl"
                    record.update(persist(payload, path, binary.predict, frame), status="ok", refit_details=details, reused=False)
            except Exception:
                successful = False
                record["error"] = traceback.format_exc()
            record["finished_at"] = base.now()
            components.append(record)
            member_records.append(record)
            base.write(args.output / "fits" / job["dataset"] / f"{job['library']}-seed{seed}.json", record)
            print({key: record[key] for key in ("dataset", "library", "seed", "status")}, flush=True)
        if not all(m["status"] == "ok" for m in member_records):
            continue
        record = {"dataset": job["dataset"], "library": job["library"], "status": "failed", "components": member_records}
        try:
            entries = []
            for member in member_records:
                if base.digest(member["model_path"]) != member["model_sha256"]:
                    raise ValueError("Component changed before ensemble construction")
                with Path(member["model_path"]).open("rb") as stream:
                    payload = pickle.load(stream)
                entries.append({"seed": member["seed"], "model_id": f"{job['dataset']}/{job['library']}/seed{member['seed']}",
                                "model_sha256": member["model_sha256"], "payload": payload})
            ensemble = helper().make_payload(entries)
            train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
            frame = stop["X"]
            if case["problem_type"] != "regression":
                label_ids = np.unique(np.concatenate([train["y"], stop["y"]]))
                np.testing.assert_array_equal(label_ids, np.arange(case["num_classes"]))
                record["shared_encoded_label_ids"] = label_ids.tolist()
            first = np.asarray(binary.predict(entries[0]["payload"], frame), dtype=np.float64)
            second = np.asarray(binary.predict(entries[1]["payload"], frame), dtype=np.float64)
            record["component_stop_prediction_diagnostic"] = {"exactly_equal": bool(np.array_equal(first, second)),
                "maximum_absolute_difference": float(np.max(np.abs(first - second))),
                "purpose": "Diagnostic only; never replace a component or select seeds"}
            path = args.artifacts / "ensembles" / job["dataset"] / (job["library"] + ".pkl")
            record.update(persist(ensemble, path, predict, frame), status="ok", seed_pair=plan["seed_pair"])
            if plan["phase"] == "replication":
                record["original_selection"] = next(s for s in plan["base_selections"] if (s["dataset"], s["library"]) == (job["dataset"], job["library"]))
            selections.append(record)
        except Exception:
            successful = False
            record["error"] = traceback.format_exc()
        base.write(args.output / "ensembles" / job["dataset"] / (job["library"] + ".json"), record)
    expected = identities(plan["cases"])
    success = (successful and len(components) == 72 and len(selections) == 36 and
               {(s["dataset"], s["library"]) for s in selections} == expected and
               {(s["dataset"], s["library"], s["seed"]) for s in components} ==
               {(dataset, library, seed) for dataset, library in expected for seed in plan["seed_pair"]})
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "components": components, "selections": selections, "all_fits_successful": success,
        "development_scored": False, "official_test_scored": False})


def aggregate(results):
    import numpy as np
    groups = list(results[0]["models"]["ctboost"]["errors"])
    summary = []
    for group in groups:
        for comparator in ("xgboost", "lightgbm"):
            for family in ("all", "binary", "regression", "multiclass"):
                rows = [r for r in results if family == "all" or r["problem_type"] == family]
                gains = [(r["models"][comparator]["errors"][group] - r["models"]["ctboost"]["errors"][group])
                         / max(r["models"][comparator]["errors"][group], 1e-12) for r in rows]
                summary.append({"ensemble_pair": group, "comparator": comparator, "family": family, "datasets": len(rows),
                    "median_relative_error_gain": float(np.median(gains)), "wins": sum(g > 1e-12 for g in gains),
                    "ties": sum(abs(g) <= 1e-12 for g in gains), "worst_relative_regression": max(0, -min(gains))})
    passed = all(s["median_relative_error_gain"] >= 0 and s["wins"] + s["ties"] >= 6
                 for s in summary if s["ensemble_pair"] == "mean" and s["family"] == "all")
    return summary, passed


def development(args):
    import numpy as np
    plan = checked(args)
    frozen = base.read(args.output / "frozen_selections.json")
    if (frozen["plan_sha256"] != base.digest(args.output / "plan.json") or not frozen["all_fits_successful"]
            or (args.output / "development.json").exists()):
        raise ValueError("All36 ensembles must be intact and previously unassessed")
    verify_selections(frozen["selections"], identities(plan["cases"]))
    original = base.read(Path(plan["previous"]) / "development.json") if plan["phase"] == "replication" else None
    results = []
    pair = "+".join(map(str, plan["seed_pair"]))
    for case in plan["cases"]:
        data = base.load_role(case, "development")
        for role in ("train", "stop"):
            if np.intersect1d(data["outer_train_positions"], base.load_role(case, role)["outer_train_positions"]).size:
                raise ValueError("Development overlaps fitting rows")
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "models": {}}
        for library in base.LIBRARIES:
            selected = next(s for s in frozen["selections"] if (s["dataset"], s["library"]) == (case["dataset_name"], library))
            with Path(selected["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            predictions = predict(payload, data["X"])
            errors = {pair: base.error(predictions, data["y"], case["problem_type"])}
            if original:
                prior = next(r for r in original["results"] if r["dataset"] == case["dataset_name"])
                errors["47+947"] = prior["models"][library]["errors"]["47+947"]
            errors["mean"] = sum(errors.values()) / len(errors)
            path = args.artifacts / "development" / case["dataset_name"] / (library + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, predictions=predictions, labels=data["y"], outer_train_positions=data["outer_train_positions"])
            row["models"][library] = {"errors": errors, "selection": selected, "predictions_sha256": base.digest(path)}
        results.append(row)
    summary, passed = aggregate(results)
    base.write(args.output / "development.json", {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json"),
        "frozen_selections_sha256": base.digest(args.output / "frozen_selections.json"), "development_target_met": passed,
        "results": results, "summary": summary, "official_test_scored": False, "limitations": plan["limitations"]})
    print({"development_target_met": passed, "summary": [s for s in summary if s["family"] == "all"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "development"))
    parser.add_argument("--phase", choices=("initial", "replication"), required=True)
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "development": development}[args.stage](args)
