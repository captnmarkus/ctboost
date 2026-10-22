"""Exact-c5 synthetic native integration; never opens real dataset/model files."""
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
names = ["fair_accuracy_current_round_cv_support", "fair_accuracy_current_round_cv_backend",
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
with (HERE / "native-validation.log").open("w", encoding="utf-8") as stream:
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        exit_code = pytest.main(["-q", str(ROOT / "benchmarks/test_fair_accuracy_current_round_cv_native.py")], plugins=[results])
changed = [path for path, expected in sources.items() if digest(Path(path)) != expected]
receipt = {
    "status": "inactive_native_backend_toy_integration", "started_at": started,
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": int(exit_code),
    "native_path": str(native), "native_sha256": digest(native), "package_path": ctboost.__file__,
    "python": sys.version, "python_executable": sys.executable,
    "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pytest", "xgboost", "lightgbm")},
    "cpu_affinity": psutil.Process().cpu_affinity(), "threads": 1,
    "synthetic_four_round_native_fits_planned": 22,
    "synthetic_native_fit_cases_passed": sum(r["outcome"] == "passed" and "test_native_fold_local" in r["nodeid"] for r in results.results),
    "real_dataset_or_model_payloads_opened": 0, "real_quality_scores": 0, "protocol_frozen": False,
    "sources": sources, "sources_changed_during_validation": changed, "tests": results.results,
    "runner_sha256": digest(Path(__file__)), "log_sha256": digest(HERE / "native-validation.log"),
    "limits": "64MiB named-array cap is synthetic-test-only. No prospective CV memory/RSS/time limit or parent17b admission is implemented. Module hashes will be rebound to eventual completed frozen17b; any later source change requires review/revalidation of affected callbacks."
}
(HERE / "native-validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in receipt.items() if key not in ("sources", "tests")}, indent=2))
raise SystemExit(exit_code or bool(changed))
