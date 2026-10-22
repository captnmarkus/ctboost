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
