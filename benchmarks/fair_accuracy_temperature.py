"""STOP-selected positive multiclass temperature; native tree models unchanged."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import time
import traceback

BASE_PATH = Path(__file__).with_name("fair_accuracy_scout.py")
HELPER_PATH = Path(__file__).with_name("fixed_temperature.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_temperature.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("temperature_base", BASE_PATH)
helper = load_module("fixed_temperature", HELPER_PATH)
raw_predict = base.predict


def predict(payload, frame):
    return helper.apply_temperature(payload, raw_predict(payload, frame))


def native_digest(payload):
    model = payload["model"]
    if payload["library"] == "ctboost":
        # This is an internal state fingerprint; missing-value sentinels may
        # legitimately be NaN, unlike finite prediction/scoring receipts.
        data = json.dumps(model._handle.export_state(), sort_keys=True, separators=(",", ":"), allow_nan=True).encode()
    elif payload["library"] == "xgboost":
        data = bytes(model.save_raw(raw_format="ubj"))
    else:
        data = model.model_to_string().encode()
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts; use fresh paths")
    previous = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (previous["wave"] != 6 or previous["runtime"] != runtime or not frozen["all_fits_successful"]
            or development["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Expected the intact failed wave6 gate")
    datasets = {c["dataset_name"] for c in previous["cases"]}
    if (len(datasets) != 12 or len(previous["cases"]) != 12 or len(frozen["selections"]) != 36
            or {(s["dataset"], s["library"]) for s in frozen["selections"]} != {
                (dataset, library) for dataset in datasets for library in base.LIBRARIES}):
        raise ValueError("Expected twelve unique cases and36 unique native models")
    for selected in frozen["selections"]:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Parent model changed")
    plan = {"protocol": "positive_multiclass_temperature_20261016", "wave": 7,
        "created_at": base.now(), "script_sha256": base.digest(__file__),
        "base_script_sha256": base.digest(BASE_PATH), "helper_sha256": base.digest(HELPER_PATH),
        "policy_sha256": base.digest(POLICY), "runtime": runtime, "previous": str(args.previous),
        "dependencies": {str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")},
        "cases": previous["cases"], "base_selections": frozen["selections"], "seed": 47,
        "patience": previous["patience"], "temperatures": [.25, .5, .75, 1., 1.25, 1.5, 2., 4.],
        "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4]},
        "selection": "Four multiclass tasks only,8 fixed temperatures including identity for each library. Minimize STOP logloss; exact ties prefer identity then minimum abs(log(T)), then smallerT. Other8tasks unchanged. No native tree or preprocessing change; no fitted continuous optimizer.",
        "prediction": "Shared helper converts multiclass margins to float64 and divides by positiveT; all downstream scoring/latency must include this operation. Positive scaling preserves argmax.",
        "target": previous["target"], "official_test_scored": False,
        "limitations": "Adaptive reused-development study. Equal8temperature evaluations per library/task are inexpensive calibration, not additional tree fits. STOP already selected base models, so repeated reuse can overfit. All36policies freeze before DEV. No official-test access."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "candidate_evaluations": 96})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["base_script_sha256"] != base.digest(BASE_PATH)
            or plan["helper_sha256"] != base.digest(HELPER_PATH) or plan["policy_sha256"] != base.digest(POLICY)
            or plan["runtime"] != base.bootstrap(args)):
        raise ValueError("Frozen implementation/runtime changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Parent evidence changed")
    return plan


def run(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    if (args.output / "fits").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "fits").mkdir()
    selections, successful = [], True
    for original in plan["base_selections"]:
        case = next(c for c in plan["cases"] if c["dataset_name"] == original["dataset"])
        if case["problem_type"] != "multiclass":
            selections.append(original)
            continue
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        record = {"dataset": original["dataset"], "library": original["library"], "status": "failed", "started_at": base.now()}
        try:
            if base.digest(original["model_path"]) != original["model_sha256"]:
                raise ValueError("Parent native model changed")
            with Path(original["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            if "temperature" in payload:
                raise ValueError("This protocol expects uncalibrated parent margins")
            stop = base.load_role(case, "stop")
            margins = raw_predict(payload, stop["X"])
            before = native_digest(payload)
            candidates = []
            for temperature in plan["temperatures"]:
                scaled = helper.apply_temperature({**payload, "temperature": temperature}, margins)
                np.testing.assert_array_equal(np.argmax(scaled, axis=1), np.argmax(margins, axis=1))
                score = base.error(scaled, stop["y"], "multiclass")
                candidates.append({"temperature": temperature, "stop_error": score,
                    "selection_key": [score, int(temperature != 1.), float(abs(np.log(temperature))), temperature]})
            identity = next(candidate for candidate in candidates if candidate["temperature"] == 1.)
            np.testing.assert_allclose(identity["stop_error"], original["best"]["stop_error"], rtol=1e-12, atol=1e-12)
            chosen = min(candidates, key=lambda candidate: candidate["selection_key"])
            candidate = {**payload, "temperature": chosen["temperature"]}
            path = args.artifacts / "models" / original["dataset"] / (original["library"] + ".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".candidate.pkl")
            with temporary.open("wb") as stream:
                pickle.dump(candidate, stream, protocol=5)
            with temporary.open("rb") as stream:
                restored = pickle.load(stream)
            np.testing.assert_array_equal(predict(candidate, stop["X"]), predict(restored, stop["X"]))
            np.testing.assert_array_equal(raw_predict(restored, stop["X"]), margins)
            if native_digest(restored) != before:
                raise ValueError("Native tree serialization changed")
            temporary.replace(path)
            selected = {**original, "base_selection": original, "temperature": chosen["temperature"],
                "source_wave": 7, "plan_sha256": base.digest(args.output / "plan.json"),
                "model_path": str(path), "model_sha256": base.digest(path), "model_bytes": path.stat().st_size,
                "best": {**original["best"], "stop_error": chosen["stop_error"], "selection_key": chosen["selection_key"]}}
            selections.append(selected)
            record.update(status="ok", candidates=candidates, selection=selected, native_model_sha256=before,
                          raw_prediction_roundtrip_exact=True, calibrated_roundtrip_exact=True)
        except Exception:
            successful = False
            record["error"] = traceback.format_exc()
        record["finished_at"] = base.now()
        base.write(args.output / "fits" / original["dataset"] / (original["library"] + ".json"), record)
        print({"dataset": original["dataset"], "library": original["library"], "status": record["status"]}, flush=True)
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "all_fits_successful": successful and len(selections) == 36,
        "selections": selections, "development_scored": False, "official_test_scored": False})


def development(args):
    if not base.read(args.output / "frozen_selections.json")["all_fits_successful"]:
        raise ValueError("All twelve calibration jobs must succeed before assessment")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "development": development}[args.stage](args)
