"""Keep process-level settings changed by scientific worker tests isolated."""

import pytest


@pytest.fixture(autouse=True)
def restore_process_cpu_affinity():
    try:
        import psutil
    except ImportError:
        yield
        return

    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        yield
        return

    original = process.cpu_affinity()
    try:
        yield
    finally:
        # Do not suppress restoration failures on platforms supporting affinity.
        process.cpu_affinity(original)



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
