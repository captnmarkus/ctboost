"""Resource selection must not oversubscribe CPUs or bypass memory failures."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "tabarena" / "local_resources.py"
_SPEC = importlib.util.spec_from_file_location("ctboost_local_resource_calibration", _PATH)
resources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resources)


def test_affinity_respects_noncontiguous_allowed_cpus():
    assert resources.partition_affinity([11, 2, 7, 4], 2, 2) == [[2, 4], [7, 11]]
    with pytest.raises(ValueError, match="exceeds"):
        resources.partition_affinity([2, 4, 7, 11], 3, 2)


def test_histogram_and_blas_thread_budgets(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    environment = resources.worker_environment(4)
    assert environment["CTBOOST_HIST_THREADS"] == "4"
    assert all(environment[key] == "1" for key in resources.THREAD_ENV)


def _record(name, seconds, memory=100, status="complete"):
    workers, threads = map(int, name.split("x"))
    return {"layout": name, "workers": workers, "threads_per_worker": threads,
            "affinity": resources.partition_affinity(range(16), workers, threads),
            "status": status, "completed_fits": 8, "fit_wall_seconds": seconds,
            "peak_worker_tree_rss_bytes": memory}


def test_select_rejects_memory_failures_and_incomplete_repeats():
    rows = [_record("2x8", 4), _record("2x8", 6),
            _record("4x4", 1, 1000), _record("4x4", 1, 1000),
            _record("8x2", 1), _record("8x2", 1, status="memory_guard"),
            _record("1x16", 1)]
    selected = resources.select_layout(rows, memory_budget_bytes=500)
    assert selected["layout"] == "2x8"
    assert selected["fits_per_second"] == 1.6
    assert selected["memory_per_worker_bytes"] == 250
    assert resources.select_layout(rows, memory_budget_bytes=50) is None
