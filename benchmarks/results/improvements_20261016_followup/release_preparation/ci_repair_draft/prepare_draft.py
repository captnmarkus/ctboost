"""Prepare reviewable CI/test copies only. Never run tests or edit active files."""
from pathlib import Path
from datetime import datetime, timezone
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
assert (ROOT / "pyproject.toml").is_file(), ROOT
FILES = [
    "tests/conftest.py", ".github/workflows/cmake.yml", "pyproject.toml",
    "tests/test_houses_cap_experiment.py", "tests/test_houses_cap_experiment_v2.py",
    "tests/test_multiclass_refit_experiment.py", "tests/test_scalar_refresh_experiment.py",
    "tests/test_scalar_numerical_retry_experiment.py",
]
sha = lambda value: hashlib.sha256(value).hexdigest()
originals = {name: (ROOT / name).read_bytes() for name in FILES}
texts = {name: value.decode("utf-8") for name, value in originals.items()}

def replace(name, before, after):
    assert texts[name].count(before) == 1, (name, before)
    texts[name] = texts[name].replace(before, after)

replace(".github/workflows/cmake.yml", "scikit-learn scipy pytest", "scikit-learn scipy pytest psutil")
replace(".github/workflows/cmake.yml", "actions/checkout@v6", "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6.0.2")
replace(".github/workflows/cmake.yml", "actions/setup-python@v6", "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6.2.0")
replace("pyproject.toml", '  "pytest>=8.0",', '  "pytest>=8.0",\n  "psutil>=7",')
replace("pyproject.toml", '  "/benchmarks/*.cpp",', '''  "/benchmarks/*.cpp",
  "/benchmarks/fair_accuracy_current_round_cv_policy.json",
  # Small source-test policy fixtures; full research archives remain excluded.
  "/benchmarks/results/improvements_20261016_followup/robustness_protocol_multiclass.json",
  "/benchmarks/results/improvements_20261016_followup/robustness_protocol_scalar_refresh.json",
  "/benchmarks/results/improvements_20261016_followup/robustness_protocol_scalar_numerical.json",''')

# Historical fit_fixed imports both comparison libraries, even for CTBoost.
# The fixture applies only to native integration cases, not entire modules.
texts["tests/conftest.py"] += '''


@pytest.fixture
def require_scientific_stack():
    """Optional imports required by the frozen three-library fit helper."""
    def require(*, native_library=None, multiclass=False):
        reason = "Frozen scientific fit helper imports both comparison libraries"
        xgb = pytest.importorskip("xgboost", reason=reason)
        pytest.importorskip("lightgbm", reason=reason)
        if native_library == "xgboost" and multiclass and xgb.__version__ != "3.4.1":
            pytest.skip("Native multiclass serialization tests require XGBoost3.4.1")
    return require


@pytest.fixture
def record_worker_affinity(monkeypatch):
    """Record synthetic protocol placement without restricting the pytest process.

    These tests verify selection/folding/persistence, not OS CPU placement.
    Real process-tree watchdog tests do not request this fixture.
    """
    psutil = pytest.importorskip("psutil")
    calls = []
    def record(process, cpus=None):
        if cpus is None:
            raise AssertionError("Synthetic worker should only request placement")
        calls.append(tuple(cpus))
    monkeypatch.setattr(psutil.Process, "cpu_affinity", record, raising=False)
    return calls
'''

for name in ("tests/test_houses_cap_experiment.py", "tests/test_houses_cap_experiment_v2.py"):
    replace(name,
        "def test_native_houses_policy_roundtrip_and_seed349(tmp_path, library):",
        "def test_native_houses_policy_roundtrip_and_seed349(tmp_path, library, require_scientific_stack):\n    require_scientific_stack(native_library=library)")

name = "tests/test_multiclass_refit_experiment.py"
replace(name,
    "def test_anchor_stop_and_seed_transfer_pipeline(tmp_path, library, vector):",
    "def test_anchor_stop_and_seed_transfer_pipeline(tmp_path, library, vector, require_scientific_stack):\n    require_scientific_stack(native_library=library, multiclass=True)")
replace(name,
    "def test_identity_transfer_keeps_parent_artifact(tmp_path, monkeypatch):",
    "def test_identity_transfer_keeps_parent_artifact(tmp_path, monkeypatch, record_worker_affinity):")
replace(name, "    module.transfer_job(args)",
    "    module.transfer_job(args)\n    assert record_worker_affinity == [(4,)]")

for name in ("tests/test_scalar_refresh_experiment.py", "tests/test_scalar_numerical_retry_experiment.py"):
    replace(name,
        "def test_fresh_anchor_and_fixed_seed_pipeline(tmp_path, monkeypatch, library, problem):",
        "def test_fresh_anchor_and_fixed_seed_pipeline(tmp_path, monkeypatch, library, problem, require_scientific_stack):\n    require_scientific_stack(native_library=library)")
    replace(name,
        "def test_identity_union_is_fresh_and_clean_not_parent(tmp_path, monkeypatch):",
        "def test_identity_union_is_fresh_and_clean_not_parent(tmp_path, monkeypatch, require_scientific_stack, record_worker_affinity):\n    require_scientific_stack(native_library=\"ctboost\")")
    replace(name, "    module.transfer_job(args)",
        "    module.transfer_job(args)\n    assert record_worker_affinity == [(4,)]")

name = "tests/test_scalar_refresh_experiment.py"
replace(name,
    "def test_selection_runs_all_eight_candidates_from_clean_anchor(tmp_path, monkeypatch):",
    "def test_selection_runs_all_eight_candidates_from_clean_anchor(tmp_path, monkeypatch, require_scientific_stack, record_worker_affinity):\n    require_scientific_stack(native_library=\"xgboost\")")
replace(name, "    module.selection_job(args)",
    "    module.selection_job(args)\n    assert record_worker_affinity == [(4,)]")

name = "tests/test_scalar_numerical_retry_experiment.py"
replace(name,
    "def test_penalty_worker_reuses_anchor_and_checks_two_native_candidates(tmp_path, monkeypatch):",
    "def test_penalty_worker_reuses_anchor_and_checks_two_native_candidates(tmp_path, monkeypatch, require_scientific_stack, record_worker_affinity):\n    require_scientific_stack(native_library=\"lightgbm\")")
replace(name, "    module.penalty_job(args)",
    "    module.penalty_job(args)\n    assert record_worker_affinity == [(4,)]")

manifest = []
diffs = []
for name in FILES:
    # Normalize only the draft to LF; original snapshots preserve exact bytes.
    updated = texts[name].replace("\r\n", "\n").encode("utf-8")
    for directory, value in (("originals", originals[name]), ("files", updated)):
        path = HERE / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    assert (ROOT / name).read_bytes() == originals[name]
    manifest.append({"path": name, "original_sha256": sha(originals[name]), "draft_sha256": sha(updated)})
    diffs.extend(difflib.unified_diff(originals[name].decode("utf-8").replace("\r\n", "\n").splitlines(True),
        updated.decode("utf-8").splitlines(True), fromfile="a/"+name, tofile="b/"+name))

workflow_name = ".github/workflows/scientific-tests.yml"
workflow = HERE / "files" / workflow_name
assert workflow.is_file() and not (ROOT / workflow_name).exists()
workflow_bytes = workflow.read_bytes()
manifest.append({"path": workflow_name, "original_sha256": None, "draft_sha256": sha(workflow_bytes)})
diffs.extend(difflib.unified_diff([], workflow_bytes.decode("utf-8").splitlines(True),
    fromfile="/dev/null", tofile="b/"+workflow_name))

(HERE / "changes.diff").write_text("".join(diffs), encoding="utf-8")
receipt = {"recorded_at": datetime.now(timezone.utc).isoformat(), "status": "draft_only_not_applied_or_executed",
    "files": manifest, "active_sources_unchanged": True, "tests_run": False,
    "scientific_environment_metadata": {"python": "3.12.11", "psutil": "7.2.2", "pytest": "9.1.1",
        "numpy": "2.0.2", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3",
        "xgboost": "3.4.1", "lightgbm": "4.7.0"},
    "metadata_source": "baseline-env importlib.metadata; task-local lightgbm4.7.0 METADATA; no native fit/import or test",
    "limitations": ["Draft source review only; test execution and actual hosted CI remain required.",
        "Apply only after all permitted quality and timing work is terminal and frozen test evidence is preserved."]}
(HERE / "manifest.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"draft": str(HERE), "files": len(FILES), "active_sources_unchanged": True, "tests_run": False}))
