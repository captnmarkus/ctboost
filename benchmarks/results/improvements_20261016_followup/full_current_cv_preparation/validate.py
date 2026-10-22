"""Synthetic-only draft validation, independent from any accuracy protocol."""
from __future__ import annotations
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import psutil
psutil.Process().cpu_affinity([0])
import numpy as np
import pandas as pd
import pytest

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

frozen = {}
for wave, filename in ((13, "fair_accuracy_round_cv.py"), (16, "fair_accuracy_multiclass_refit.py"), (17, "fair_accuracy_scalar_refresh.py")):
    source = ROOT / "benchmarks" / filename
    plan = ROOT / "benchmarks/results/improvements_20261016_followup" / f"accuracy_wave{wave}/plan.json"
    expected = json.loads(plan.read_text())["script_sha256"]
    assert digest(source) == expected, filename
    frozen[str(source)] = expected

test = ROOT / "benchmarks/test_fair_accuracy_current_round_cv_support.py"
source = test.with_name("fair_accuracy_current_round_cv_support.py")
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (HERE / "validation.log").open("w", encoding="utf-8") as stream:
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = pytest.main(["-q", str(test)])
for path, expected in frozen.items():
    assert digest(Path(path)) == expected
receipt = {
    "status": "inactive_prototype_synthetic_only", "started_at": started,
    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": int(result),
    "python": sys.version, "python_executable": sys.executable, "numpy": np.__version__, "pandas": pd.__version__,
    "cpu_affinity": psutil.Process().cpu_affinity(), "threads": 1,
    "native_fits": 0, "real_dataset_or_model_payloads_opened": 0, "real_scores": 0, "protocol_frozen": False,
    "sources": {str(path): digest(path) for path in (source, test, Path(__file__))},
    "frozen_sources_still_match_original_plans": frozen,
    "log_sha256": digest(HERE / "validation.log"),
    "limitations": "Adapter spies establish orchestration contracts only. No native correction backend, solver choice, parent17b activation, shared fold freeze or resource policy is implemented. Add exact-c5 all-library native toy tests only after certification; never claim current tests cover native math."
}
(HERE / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
raise SystemExit(result)
