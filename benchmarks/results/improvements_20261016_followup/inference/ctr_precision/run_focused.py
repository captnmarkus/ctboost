"""Assert isolated runtime and run bounded correctness checks on CPU0."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "CTBOOST_HIST_THREADS", "CTBOOST_NODE_HIST_THREADS"):
    os.environ[key] = "1"
import psutil
psutil.Process().cpu_affinity([0])
ROOT = Path(r"C:/apps/ctboost")
package = ROOT / ".tmp/improvements-20261016/ctr-precision/runtime-cp312"
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != "_editable_skbc_ctboost"]
sys.path[:0] = [str(package), str(ROOT / ".tmp/improvements-20261016/followup/deps"), str(ROOT)]
import ctboost
import pytest

if __name__ == "__main__":
    assert Path(ctboost.__file__).resolve().is_relative_to(package)
    assert Path(ctboost._core.__file__).resolve().is_relative_to(package)
    print(json.dumps({"native":ctboost._core.__file__,
        "native_sha256":hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
        "versions":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","pytest")}}), flush=True)
    raise SystemExit(pytest.main(sys.argv[1:]))
