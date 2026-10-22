"""One clean current CP310 full-suite sequence; existing dependency policy."""
from __future__ import annotations
import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(r"C:/apps/ctboost")
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / ".tmp/improvements-20261016/ctr-precision"
EXPECTED_NATIVE = "21ed48b6234bf570413fd4336ffdf76279b0ff9e3d9003cbcf0b34620e2aee17"
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "CTBOOST_HIST_THREADS", "CTBOOST_NODE_HIST_THREADS"):
    os.environ[name] = "1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    import psutil
    import pytest
    process = psutil.Process()
    initial_affinity = process.cpu_affinity()
    # Existing controller tests need two worker slots with two CPUs each.
    # CPU4 remains available to the independent accuracy worker.
    validation_affinity = [0, 2, 6, 8]
    assert set(validation_affinity).issubset(initial_affinity)
    process.cpu_affinity(validation_affinity)
    sys.meta_path[:] = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(ROOT))
    import ctboost
    native = Path(ctboost._core.__file__).resolve()
    assert sys.version_info[:2] == (3, 10)
    assert native.parent == ROOT / "ctboost" and digest(native) == EXPECTED_NATIVE
    assert Path(ctboost.__file__).resolve().parent == ROOT / "ctboost"
    assert importlib.util.find_spec("lightgbm") is None, "Do not load an unreviewed CP310 comparison stack"
    skip_document = json.loads((ARCHIVE / "scientific-stack-skiplist.json").read_text())
    skipped = set(skip_document["nodes"])
    assert len(skipped) == skip_document["count"] == 19
    source_paths = sorted(set((ROOT / "tests").rglob("*.py")) | set((ROOT / "ctboost").rglob("*.py"))
                          | set((ROOT / "benchmarks").glob("*.py")) | set((ROOT / "benchmarks").glob("*.json")))
    sources = {str(path): digest(path) for path in source_paths}
    native_sources = json.loads((ARCHIVE / "native-source-hashes.json").read_text())
    # Preserve this manifest verbatim; its representation has already been audited.
    collected, outcomes, reports, policy_skips = [], [], [], []
    class Evidence:
        def pytest_collection_modifyitems(self, items):
            present = {item.nodeid for item in items}
            assert skipped <= present, "The established19-node policy must not silently change"
            for item in items:
                collected.append(item.nodeid)
                if item.nodeid in skipped:
                    item.add_marker(pytest.mark.skip(reason=skip_document["reason"] + "; covered separately by exact CP31234-case validation"))
                    policy_skips.append(item.nodeid)
        def pytest_runtest_logreport(self, report):
            reports.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome})
            if report.when == "call" or report.failed or report.skipped:
                row = {"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome}
                if report.failed or report.skipped:
                    row["detail"] = str(report.longrepr)
                outcomes.append(row)
        def pytest_sessionfinish(self, session, exitstatus):
            print(json.dumps({"complete": True, "exit_status": int(exitstatus), "collected": len(collected)}), flush=True)
    started = now()
    before = {"available_bytes": psutil.virtual_memory().available, "total_bytes": psutil.virtual_memory().total,
              "study_overlap": "Independent17b accuracy union fits may run onCPU4; this is correctness validation, not performance timing"}
    print(json.dumps({"started_at": started, "pid": os.getpid(), "python": sys.executable,
                      "native_sha256": digest(native), "affinity": validation_affinity,
                      "available_GiB": before["available_bytes"] / 2**30}), flush=True)
    with (HERE / "pytest.log").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = pytest.main(["-q", str(ROOT / "tests")], plugins=[Evidence()])
    changed = [path for path, expected in sources.items() if not Path(path).exists() or digest(path) != expected]
    added = [str(path) for path in sorted((ROOT / "tests").rglob("*.py")) if str(path) not in sources]
    receipt = {
        "status": "clean_full_cp310_post_affinity_and_format_expectation_repairs",
        "started_at": started, "finished_at": now(), "exit_code": int(result), "pid": os.getpid(),
        "python": sys.executable, "python_version": sys.version, "package": ctboost.__file__,
        "native_path": str(native), "native_sha256": digest(native), "build_info": ctboost.build_info(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pytest", "xgboost", "psutil")},
        "validation_affinity": validation_affinity, "affinity_after_tests": process.cpu_affinity(), "threads": 1,
        "available_memory_at_start": before, "sources_at_collection_start": sources,
        "sources_changed_during_run": changed, "test_files_added_during_run_not_claimed_collected": added,
        "native_source_manifest_sha256": digest(ARCHIVE / "native-source-hashes.json"),
        "skip_policy_sha256": digest(ARCHIVE / "scientific-stack-skiplist.json"), "explicit_policy_skip_nodes": policy_skips,
        "collected_count": len(collected), "collected": collected, "test_outcomes": outcomes,
        "counts": {status: sum(row["outcome"] == status for row in outcomes) for status in ("passed", "failed", "skipped")},
        "teardown_failures": [row for row in reports if row["when"] == "teardown" and row["outcome"] == "failed"],
        "fixture_sha256": digest(ROOT / "tests/conftest.py"), "runner_sha256": digest(__file__),
        "log_sha256": digest(HERE / "pytest.log"),
        "scope": "One new full tests/ collection, same established19-node optional comparison-stack exclusions; skips are not passes. Prior run and repair chronology are preserved. No installs, builds, production edits, frozen scientific-source edits, accuracy scoring or latency timing."
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    process.cpu_affinity(initial_affinity)
    print(json.dumps({key: receipt[key] for key in ("finished_at", "exit_code", "counts", "sources_changed_during_run",
                                                 "collected_count", "native_sha256", "affinity_after_tests")}), flush=True)
    return int(result) or bool(changed)


if __name__ == "__main__":
    raise SystemExit(main())
