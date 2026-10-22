"""Exact-c5 inactive runner plus final-parent native toy validation; no real dataset/model files."""
from __future__ import annotations
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS"):
    os.environ[name] = "1"
import psutil
psutil.Process().cpu_affinity([0])
sys.meta_path[:] = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
package = ROOT / ".tmp/improvements-20261016/followup/runtime"
sys.path[:0] = [str(package), str(ROOT / ".tmp/improvements-20261016/followup/deps"), str(ROOT)]
import ctboost
import lightgbm
import xgboost
import pytest

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

native = Path(ctboost._core.__file__).resolve()
assert native.is_relative_to(package) and Path(ctboost.__file__).resolve().is_relative_to(package)
assert digest(native) == "c5ea7e026942c3f1190ca5511fbf74f353bfeb08e7771f76bc5f6c1ca3e4d197"
assert xgboost.__version__ == "3.4.1" and lightgbm.__version__ == "4.7.0"
names = ["fair_accuracy_current_round_cv", "fair_accuracy_current_round_cv_resources",
         "fair_accuracy_current_round_cv_seed_check", "fair_accuracy_current_round_cv_outer_test",
         "test_fair_accuracy_current_round_cv_runner", "fair_accuracy_current_round_cv_support", "fair_accuracy_current_round_cv_backend",
         "test_fair_accuracy_current_round_cv_native", "fair_accuracy_scalar_numerical_retry",
         "fair_accuracy_multiclass_refit", "fixed_multiclass_leaf_refit", "fixed_binary_leaf_refit",
         "fixed_leaf_refit", "fixed_clean_refit_check", "ridge_numerical_diagnostic_extended",
         "fair_accuracy_union_refit", "fair_accuracy_scout", "fixed_temperature"]
sources = {str(ROOT / "benchmarks" / (name + ".py")): digest(ROOT / "benchmarks" / (name + ".py")) for name in names}
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
class Results:
    def __init__(self):
        self.results = []
    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.skipped):
            self.results.append({"nodeid": report.nodeid, "outcome": report.outcome})
results = Results()
with (HERE / "runner-validation-02.log").open("w", encoding="utf-8") as stream:
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        exit_code = pytest.main(["-q", str(ROOT / "benchmarks/test_fair_accuracy_current_round_cv_runner.py")], plugins=[results])
changed = [path for path, expected in sources.items() if digest(Path(path)) != expected]
receipt = {
    "status": "inactive_final_executable_synthetic_integration", "started_at": started,
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": int(exit_code),
    "native_path": str(native), "native_sha256": digest(native), "package_path": ctboost.__file__,
    "python": sys.version, "python_executable": sys.executable,
    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pytest", "xgboost", "lightgbm")},
    "cpu_affinity": psutil.Process().cpu_affinity(), "threads": 1,
    "synthetic_four_round_native_fits_planned": 0, "synthetic_fake_learner_worker_calls_planned": 37,
    "synthetic_native_fit_cases_passed": sum(r["outcome"] == "passed" and "test_native_fold_local" in r["nodeid"] for r in results.results),
    "real_dataset_or_model_payloads_opened": 0, "real_quality_scores": 0, "protocol_frozen": False,
    "prior_final_parent_native_validation_sha256": digest(HERE / "runner-validation-01.json"), "sources": sources, "sources_changed_during_validation": changed, "tests": results.results,
    "runner_sha256": digest(Path(__file__)), "log_sha256": digest(HERE / "runner-validation-02.log"),
    "limits": "This pass uses fake learners/synthetic caches only and no native fits. The preceding runner-validation-01 preserves the30 exact-c5 native checks against final17b source, including22 four-round toy fits. No real-data preparation, scoring, protocol or fold freeze occurred."
}
(HERE / "runner-validation-02.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in receipt.items() if key not in ("sources", "tests")}, indent=2))
raise SystemExit(exit_code or bool(changed))
