import ctypes
import ctypes.util
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import ctboost

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _cuda_runtime_candidates():
    package_parent = pathlib.Path(ctboost.__file__).resolve().parent.parent
    runtime_pattern = "cudart64_*.dll" if os.name == "nt" else "libcudart.so*"
    yield from sorted(package_parent.glob(f"*/{runtime_pattern}"), reverse=True)

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        bin_dir = pathlib.Path(cuda_path) / ("bin" if os.name == "nt" else "lib64")
        pattern = "cudart64_*.dll" if os.name == "nt" else "libcudart.so*"
        yield from sorted(bin_dir.glob(pattern), reverse=True)

    if os.name == "nt":
        for path_entry in os.environ.get("PATH", "").split(os.pathsep):
            if path_entry:
                yield from sorted(
                    pathlib.Path(path_entry).glob("cudart64_*.dll"), reverse=True
                )
    else:
        maps_path = pathlib.Path("/proc/self/maps")
        if maps_path.is_file():
            for line in maps_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                mapped_path = line.rsplit(maxsplit=1)[-1]
                if "libcudart.so" in mapped_path:
                    yield mapped_path
        yield "libcudart.so"
        yield "libcudart.so.12"
        yield "/usr/local/cuda/lib64/libcudart.so"

    discovered = ctypes.util.find_library("cudart")
    if discovered:
        yield discovered


def _load_cuda_runtime():
    errors = []
    seen = set()
    for candidate in _cuda_runtime_candidates():
        candidate = str(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            runtime = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue

        runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDevice.restype = ctypes.c_int
        runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDeviceCount.restype = ctypes.c_int
        runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        runtime.cudaSetDevice.restype = ctypes.c_int
        runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        runtime.cudaGetErrorString.restype = ctypes.c_char_p
        return runtime

    pytest.skip("CUDA runtime library is unavailable: " + "; ".join(errors))


def _cuda_error(runtime, status):
    message = runtime.cudaGetErrorString(status)
    return message.decode("utf-8", errors="replace") if message else f"status {status}"


def _cuda_check(runtime, status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed: {_cuda_error(runtime, status)}")


def _cuda_device_count(runtime):
    count = ctypes.c_int()
    status = runtime.cudaGetDeviceCount(ctypes.byref(count))
    if status != 0:
        pytest.skip(f"CUDA devices are unavailable: {_cuda_error(runtime, status)}")
    return count.value


def _current_device(runtime):
    device = ctypes.c_int()
    _cuda_check(runtime, runtime.cudaGetDevice(ctypes.byref(device)), "cudaGetDevice")
    return device.value


def _set_device(runtime, device):
    _cuda_check(runtime, runtime.cudaSetDevice(device), f"cudaSetDevice({device})")


def _run_dual_device_sequence():
    runtime = _load_cuda_runtime()
    if _cuda_device_count(runtime) < 2:
        raise RuntimeError(
            "dual-device worker requires at least two visible CUDA devices"
        )

    original_device = _current_device(runtime)
    rng = np.random.default_rng(15501)
    X = rng.normal(size=(2048, 8)).astype(np.float32)
    binary_margin = 2.2 * X[:, 0] - 1.5 * X[:, 1] + 0.9 * X[:, 2] * X[:, 3]
    y_binary = (binary_margin + rng.normal(scale=0.25, size=X.shape[0]) > 0.0).astype(
        np.int64
    )
    y_regression = (
        2.0 * X[:, 0]
        - 1.25 * X[:, 1]
        + 0.7 * X[:, 2] * X[:, 2]
        + rng.normal(scale=0.15, size=X.shape[0])
    ).astype(np.float32)
    X_train = X[:1536]
    X_test = X[1536:]
    y_binary_train = y_binary[:1536]
    y_regression_train = y_regression[:1536]

    common = {
        "iterations": 24,
        "learning_rate": 0.15,
        "max_depth": 4,
        "task_type": "GPU",
        "devices": "0,1",
    }

    try:
        _set_device(runtime, 0)
        classifier = ctboost.CTBoostClassifier(**common, random_seed=15501)
        classifier.fit(X_train, y_binary_train)
        if _current_device(runtime) != 0:
            raise RuntimeError("classifier fit did not restore caller CUDA device 0")
        probabilities = classifier.predict_proba(X_test)
        if _current_device(runtime) != 0:
            raise RuntimeError(
                "classifier prediction did not restore caller CUDA device 0"
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("classifier produced non-finite probabilities")

        # Starting a second dual-device fit on device 1 reproduces the former
        # cross-device workspace relocation independently of destruction order.
        _set_device(runtime, 1)
        regressor = ctboost.CTBoostRegressor(**common, random_seed=15502)
        regressor.fit(X_train, y_regression_train)
        if _current_device(runtime) != 1:
            raise RuntimeError("regressor fit did not restore caller CUDA device 1")
        predictions = regressor.predict(X_test)
        if _current_device(runtime) != 1:
            raise RuntimeError(
                "regressor prediction did not restore caller CUDA device 1"
            )
        if not np.isfinite(predictions).all():
            raise RuntimeError("regressor produced non-finite predictions")
    finally:
        _set_device(runtime, original_device)


def test_sequential_dual_device_fits_preserve_workspace_ownership_and_caller_device():
    if not ctboost.build_info()["cuda_enabled"]:
        pytest.skip("CUDA support is not compiled into this build")

    runtime = _load_cuda_runtime()
    if _cuda_device_count(runtime) < 2:
        pytest.skip("at least two visible CUDA devices are required")

    env = os.environ.copy()
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            str(pathlib.Path(__file__).resolve()),
            "--worker",
        ],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, (
        f"dual-device worker exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("expected --worker")
    _run_dual_device_sequence()
