"""Read-only prediction compatibility against already unsealed 0.1.60 fixtures."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
import subprocess
import traceback
import zipfile

if len(sys.argv) != 3:
    raise SystemExit("Usage: verify_public_models_portable.py EXPECTED_NATIVE_SHA256 EXPECTED_WHEEL_SHA256")
EXPECTED_NATIVE_SHA256, EXPECTED_WHEEL_SHA256 = sys.argv[1:]
assert all(len(value) == 64 and all(c in "0123456789abcdef" for c in value)
           for value in (EXPECTED_NATIVE_SHA256, EXPECTED_WHEEL_SHA256))
BUILD_SOURCE_GIT_COMMIT = "5780b41c7321c70e508dccf34e2f76f2503d1175"
ROOT = Path("C:/apps/ctboost")
OUTPUT = ROOT / ".tmp/release-readiness-20260908"
OLD = ROOT / ".tmp/inference-accuracy-20260908"
for key in ("CTBOOST_HIST_THREADS", "CTBOOST_NODE_HIST_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import psutil
psutil.Process().cpu_affinity([4, 12])
import ctboost
import numpy as np
assert ctboost.__version__ == "0.1.61"
assert Path(ctboost.__file__).resolve().is_relative_to(OUTPUT / "release-env")
sys.path.insert(0, str(ROOT))
from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert np.isfinite(actual).all()
    assert actual.tobytes(order="C") == expected.tobytes(order="C"), "prediction bits differ"


def state_hash(model):
    state = {"booster": model.get_booster()._handle.export_state(),
             "pipeline": None if model._feature_pipeline is None else model._feature_pipeline.to_state()}
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def predict(model, data, problem_type):
    result = np.asarray(model.predict(data) if problem_type == "regression" else model.predict_proba(data))
    return result[:, 1] if problem_type == "binary" else result


latency_dir = OLD / "real-latency"
latency_plan = read(latency_dir / "plan.json")
scout_dir = Path(latency_plan["scout"])
scout_plan = read(scout_dir / "plan.json")
assert digest(scout_dir / "plan.json") == latency_plan["scout_plan_sha256"]
assert len(scout_plan["datasets"]) == 14
for name, expected in scout_plan["source_sha256"].items():
    assert digest(ROOT / name) == expected, name
versions = {key: importlib.metadata.version(key) for key in ("ctboost", "numpy", "pandas", "scikit-learn")}
for key in ("numpy", "pandas", "scikit-learn"):
    assert versions[key] == scout_plan["versions"][key], key
native_hash = digest(ctboost._core.__file__)
assert native_hash == EXPECTED_NATIVE_SHA256
wheel = next((OUTPUT / "dist-portable").glob("ctboost-0.1.61-cp312-cp312-win_amd64.whl"))
assert digest(wheel) == EXPECTED_WHEEL_SHA256
with zipfile.ZipFile(wheel) as archive:
    core_name = next(name for name in archive.namelist() if name.startswith("ctboost/_core") and name.endswith(".pyd"))
    assert hashlib.sha256(archive.read(core_name)).hexdigest() == native_hash
reference_summaries = {}
for repeat in (1, 2):
    summary = read(latency_dir / "measurements" / f"baseline-{repeat}" / "summary.json")
    assert summary["plan_sha256"] == digest(latency_dir / "plan.json")
    assert summary["runtime"]["native_sha256"] == scout_plan["ctboost_native_sha256"]
    reference_summaries[repeat] = {row["dataset"]: row for row in summary["rows"]}
receipt = {
    "verification_phase": "phase3_portability",
    "build_source_git_commit": BUILD_SOURCE_GIT_COMMIT,
    "purpose": "Saved public 0.1.60 model compatibility; no fits, score calculations or timing study",
    "roles_opened": ["development (fold 6)"], "confirmation_and_outer_test_opened": False,
    "source_git_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    "working_tree_status_at_verification": subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
    "versions": versions, "native_sha256": native_hash, "wheel_sha256": digest(wheel),
    "public_native_sha256": scout_plan["ctboost_native_sha256"],
    "scout_plan_sha256": digest(scout_dir / "plan.json"),
    "latency_plan_sha256": digest(latency_dir / "plan.json"),
    "verifier_sha256": digest(__file__), "cpu_affinity": psutil.Process().cpu_affinity(),
    "source_sha256": {}, "results": [],
}
package_root = Path(ctboost.__file__).resolve().parent
for path in sorted(package_root.rglob("*.py")):
    receipt["source_sha256"]["installed/ctboost/" + path.relative_to(package_root).as_posix()] = digest(path)
for directory in (ROOT / "src/core", ROOT / "src/bindings", ROOT / "include/ctboost"):
    for path in sorted(directory.iterdir()):
        if path.suffix in (".cpp", ".hpp"):
            receipt["source_sha256"][path.relative_to(ROOT).as_posix()] = digest(path)

for case in scout_plan["datasets"]:
    name = case["dataset_name"]
    row = {"dataset": name, "problem_type": case["problem_type"], "status": "failed"}
    try:
        role_path = scout_dir / "data" / name / "development.pkl"
        assert digest(role_path) == case["roles"]["development"]["sha256"]
        role = pickle.loads(role_path.read_bytes())
        assert len(role["X"]) == case["roles"]["development"]["rows"]
        assert array_hash(np.asarray(role["outer_train_positions"], dtype="<i8")) == case["roles"]["development"]["index_sha256"]
        fit_dir = scout_dir / "fits" / name / "ctboost_default"
        fit_receipt = read(fit_dir / "result.json")
        assert fit_receipt["status"] == "ok" and not fit_receipt["resource_failure"]
        assert fit_receipt["plan_sha256"] == digest(scout_dir / "plan.json")
        assert digest(fit_dir / "model.pkl") == fit_receipt["model_sha256"]
        assert digest(fit_dir / "development.npz") == fit_receipt["development_sha256"]
        payload = pickle.loads((fit_dir / "model.pkl").read_bytes())
        model = payload["model"]
        before_state = state_hash(model)
        frame, _ = normalize_tabarena_frame(role["X"], categorical_columns=payload["categorical_columns"])
        with np.load(fit_dir / "development.npz", allow_pickle=False) as reference:
            expected = reference["predictions"]
            np.testing.assert_array_equal(role["outer_train_positions"], reference["outer_train_positions"])
        raw_prediction = predict(model, frame, case["problem_type"])
        prepared_pool = model._transform_prediction_pool(frame)
        prepared_prediction = predict(model, prepared_pool, case["problem_type"])
        check_bits(raw_prediction, expected)
        check_bits(prepared_prediction, expected)
        row.update(development_rows=len(frame), prediction_dtype=str(expected.dtype),
                   public_prediction_sha256=array_hash(expected), raw_input_bitwise=True,
                   prepared_pool_bitwise=True, model_sha256=fit_receipt["model_sha256"],
                   development_data_sha256=digest(role_path), development_reference_sha256=digest(fit_dir / "development.npz"))
        positions = np.arange(1000) % len(frame)
        batch = frame.iloc[positions].copy().reset_index(drop=True)
        raw_batch = predict(model, batch, case["problem_type"])
        prepared_batch = predict(model, model._transform_prediction_pool(batch), case["problem_type"])
        for repeat in (1, 2):
            old_row = reference_summaries[repeat][name]
            reference_path = latency_dir / "measurements" / f"baseline-{repeat}" / f"{name}.npz"
            assert digest(reference_path) == old_row["prediction_sha256"]
            assert array_hash(positions.astype("<i8")) == old_row["cycle_index_sha256"]
            assert old_row["models"]["ctboost_default"]["model_sha256"] == fit_receipt["model_sha256"]
            with np.load(reference_path, allow_pickle=False) as reference:
                check_bits(raw_batch, reference["ctboost_default"])
                check_bits(prepared_batch, reference["ctboost_default"])
        row["existing_1000_row_batches_bitwise"] = True
        if name == "hiva_agnostic":
            original_matrix_path = OLD / "hiva-profile/public-transformed-37.npz"
            transformed, cat_features, feature_names = model._feature_pipeline.transform_array(frame.iloc[:37])
            with np.load(original_matrix_path, allow_pickle=False) as reference:
                check_bits(transformed, reference["matrix"])
            original_matrix_receipt = read(OLD / "hiva-profile/public-transformed-37.json")
            assert array_hash(transformed) == original_matrix_receipt["matrix_sha256"]
            assert feature_names == original_matrix_receipt["names"]
            assert cat_features == original_matrix_receipt["cat_features"]
            row["existing_hiva_prepared_37_matrix_bitwise"] = True
        assert state_hash(model) == before_state
        assert digest(fit_dir / "model.pkl") == fit_receipt["model_sha256"]
        row.update(status="ok", state_unchanged=True)
    except Exception:
        row["error"] = traceback.format_exc()
    receipt["results"].append(row)
    print(json.dumps({"dataset": name, "status": row["status"]}), flush=True)
    (OUTPUT / "public-0160-model-compatibility-portable.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

receipt["successful_models"] = sum(row["status"] == "ok" for row in receipt["results"])
receipt["all14_passed"] = receipt["successful_models"] == 14
(OUTPUT / "public-0160-model-compatibility-portable.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if receipt["all14_passed"] else 1)
