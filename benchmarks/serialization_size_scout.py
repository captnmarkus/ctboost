"""Measure existing export sizes and prediction parity, without timing claims."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(args.package_root.resolve()))
    import ctboost
    import numpy as np
    import psutil
    psutil.Process().cpu_affinity([4])
    plan = json.loads((args.accuracy / "plan.json").read_text())
    fits = json.loads((args.accuracy / "fits.json").read_text())["records"]
    if digest(ctboost._core.__file__) != plan["runtime"]["native_sha256"]:
        raise ValueError("Use the accuracy study's exact native runtime")
    case = next(case for case in plan["cases"] if case["id"] == "digits_20261016")
    fit = next(fit for fit in fits if fit["case"] == case["id"] and fit["variant"] == "baseline")
    source = Path(fit["model_path"])
    test_role = Path(case["roles"]["test"]["path"])
    if digest(source) != fit["model_sha256"] or digest(test_role) != case["roles"]["test"]["sha256"]:
        raise ValueError("Frozen source model or prediction rows changed")
    args.output.mkdir(parents=True, exist_ok=False)
    args.artifacts.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", {
        "script_sha256": digest(__file__), "accuracy_plan_sha256": digest(args.accuracy / "plan.json"),
        "case": case["id"], "variant": "baseline", "source_model": str(source), "source_sha256": digest(source),
        "native_sha256": digest(ctboost._core.__file__), "test_role_sha256": digest(test_role),
        "exports": ["pickle", "json_predictor"], "pickle_parity": "exact raw scores",
        "json_predictor_parity": {"rtol": 2e-5, "atol": 2e-5, "same_predicted_class": True},
        "purpose": "Post-assessment size/parity follow-up of existing exports on the first fixed digits split; no fitting, model selection, accuracy comparison, or latency claim",
    })
    model = ctboost.Booster.load_model(source)
    with np.load(test_role) as data:
        X = data["X"].copy()
    expected = model.predict(X)
    outputs = []
    for name, suffix in (("pickle", ".pkl"), ("json_predictor", ".json")):
        path = args.artifacts / ("digits_baseline" + suffix)
        if name == "pickle":
            model.save_model(path, model_format="pickle")
            restored = ctboost.Booster.load_model(path)
            actual = restored.predict(X)
            np.testing.assert_array_equal(actual, expected)
        else:
            model.export_model(path, export_format="json_predictor")
            restored = ctboost.load_exported_predictor(path)
            actual = np.asarray(restored.predict_raw(X))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(expected, axis=1))
        outputs.append({"format": name, "bytes": path.stat().st_size, "path": str(path.resolve()), "sha256": digest(path),
                        "fraction_of_native_json_bytes": path.stat().st_size / source.stat().st_size,
                        "raw_scores_exact": bool(np.array_equal(actual, expected)),
                        "maximum_absolute_raw_score_difference": float(np.max(np.abs(actual - expected))),
                        "same_predicted_class": True, "rows": len(X)})
    result = {"plan_sha256": digest(args.output / "plan.json"), "native_json_bytes": source.stat().st_size,
              "exports": outputs, "caveat": "Pickle and JSON predictor are different contracts; compact bytes do not imply faster Python inference. JSON predictor may accumulate in different precision."}
    write(args.output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
