from __future__ import annotations

import argparse
import base64
import json
import os
import textwrap
import time
from pathlib import Path

import kagglehub
from kagglehub import auth as kaggle_auth
from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
    ApiSaveKernelRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType, KernelWorkerStatus


PATCHED_FILES = [
    "CMakeLists.txt",
    "ctboost/core.py",
    "ctboost/sklearn.py",
    "ctboost/training.py",
    "cuda/cuda_backend.cu",
    "cuda/hist_kernels.cu",
    "cuda/hist_kernels.cuh",
    "include/ctboost/booster.hpp",
    "include/ctboost/cuda_backend.hpp",
    "include/ctboost/histogram.hpp",
    "include/ctboost/profiler.hpp",
    "include/ctboost/statistics.hpp",
    "include/ctboost/tree.hpp",
    "src/bindings/module.cpp",
    "src/core/booster.cpp",
    "src/core/cuda_backend_stub.cpp",
    "src/core/histogram.cpp",
    "src/core/profiler.cpp",
    "src/core/statistics.cpp",
    "src/core/tree.cpp",
]


def _load_overlay_payloads(repo_root: Path) -> dict[str, str]:
    payloads: dict[str, str] = {}
    for relative_path in PATCHED_FILES:
        payloads[relative_path] = base64.b64encode((repo_root / relative_path).read_bytes()).decode("ascii")
    return payloads


def _build_kernel_source(repo_root: Path, base_commit: str) -> str:
    overlay_json = json.dumps(_load_overlay_payloads(repo_root), separators=(",", ":"))
    patched_files_json = json.dumps(PATCHED_FILES)
    return textwrap.dedent(
        f"""
        import base64
        import glob
        import json
        import os
        import shutil
        import subprocess
        import sys
        import time
        from pathlib import Path

        import numpy as np
        import pandas as pd

        PATCHED_FILES = {patched_files_json}
        OVERLAY = json.loads({overlay_json!r})
        BASE_COMMIT = {base_commit!r}


        def run(cmd, cwd=None, env=None, check=True):
            print("+", " ".join(cmd))
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                text=True,
                capture_output=True,
            )
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)
            if check and completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    cmd,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            return completed


        def find_competition_file(file_name: str) -> Path:
            direct = Path("/kaggle/input/competitions/playground-series-s6e4") / file_name
            if direct.exists():
                return direct
            matches = glob.glob(f"/kaggle/input/**/{{file_name}}", recursive=True)
            if not matches:
                raise FileNotFoundError(file_name)
            return Path(matches[0])


        def timed_predict(booster, data):
            started = time.perf_counter()
            predictions = booster.predict(data)
            elapsed = time.perf_counter() - started
            return predictions, elapsed


        repo = Path("/kaggle/working/ctboost")
        if repo.exists():
            shutil.rmtree(repo)

        run(["git", "clone", "https://github.com/captnmarkus/ctboost.git", str(repo)])
        run(["git", "checkout", BASE_COMMIT], cwd=repo)

        for relative_path, encoded in OVERLAY.items():
            destination = repo / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(encoded))

        run(["git", "status", "--short"], cwd=repo)
        run(["git", "diff", "--stat"], cwd=repo)
        run(["bash", "-lc", "which nvcc && nvcc --version && nvidia-smi"])

        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "ctboost"], check=False)
        run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "build",
            "cmake",
            "ninja",
            "numpy",
            "pandas",
            "pybind11",
            "scikit-build-core",
        ])

        build_env = os.environ.copy()
        build_env["CMAKE_ARGS"] = "-DCTBOOST_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75"
        build_env["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
        build_env["CMAKE_CUDA_COMPILER"] = "/usr/local/cuda/bin/nvcc"
        build_env["CUDAToolkit_ROOT"] = "/usr/local/cuda"
        build_env["CUDA_HOME"] = "/usr/local/cuda"
        run([sys.executable, "-m", "pip", "install", "-v", "."], cwd=repo, env=build_env)

        import ctboost

        build_info = ctboost.build_info()
        print(json.dumps(build_info, indent=2))
        if not build_info.get("cuda_enabled", False):
            raise RuntimeError(f"CUDA build failed: {{build_info}}")

        train = pd.read_csv(find_competition_file("train.csv"))
        test = pd.read_csv(find_competition_file("test.csv"))

        x_multiclass_train = train.drop(columns=["id", "Irrigation_Need"])
        x_multiclass_test = test.drop(columns=["id"])
        multiclass_labels, multiclass_targets = np.unique(train["Irrigation_Need"], return_inverse=True)

        x_binary_train = x_multiclass_train
        x_binary_test = x_multiclass_test
        binary_targets = (train["Irrigation_Need"] == "High").astype(np.float32).to_numpy()

        x_regression_train = train.drop(columns=["id", "Irrigation_Need", "Previous_Irrigation_mm"])
        x_regression_test = test.drop(columns=["id", "Previous_Irrigation_mm"])
        regression_targets = train["Previous_Irrigation_mm"].astype(np.float32).to_numpy()

        results = {{
            "build_info": build_info,
            "base_commit": BASE_COMMIT,
            "patched_files": PATCHED_FILES,
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
        }}

        multiclass_pool = ctboost.Pool(x_multiclass_train, multiclass_targets.astype(np.float32))
        started = time.perf_counter()
        multiclass_booster = ctboost.train(
            multiclass_pool,
            {{
                "objective": "MultiClass",
                "num_classes": int(len(multiclass_labels)),
                "iterations": 5,
                "learning_rate": 0.1,
                "max_depth": 6,
                "task_type": "GPU",
                "verbose": True,
            }},
        )
        multiclass_fit_seconds = time.perf_counter() - started
        multiclass_raw_scores, multiclass_predict_seconds = timed_predict(multiclass_booster, x_multiclass_test)
        multiclass_raw_scores = np.asarray(multiclass_raw_scores, dtype=np.float32)
        results["multiclass"] = {{
            "fit_seconds": multiclass_fit_seconds,
            "predict_seconds": multiclass_predict_seconds,
            "raw_shape": list(multiclass_raw_scores.shape),
            "checksum": float(multiclass_raw_scores[:2048].sum(dtype=np.float64)),
        }}

        binary_pool = ctboost.Pool(x_binary_train, binary_targets)
        started = time.perf_counter()
        binary_booster = ctboost.train(
            binary_pool,
            {{
                "objective": "Logloss",
                "iterations": 5,
                "learning_rate": 0.1,
                "max_depth": 6,
                "task_type": "GPU",
                "verbose": True,
            }},
        )
        binary_fit_seconds = time.perf_counter() - started
        binary_raw_scores, binary_predict_seconds = timed_predict(binary_booster, x_binary_test)
        binary_raw_scores = np.asarray(binary_raw_scores, dtype=np.float32)
        results["binary"] = {{
            "fit_seconds": binary_fit_seconds,
            "predict_seconds": binary_predict_seconds,
            "raw_shape": list(binary_raw_scores.shape),
            "checksum": float(binary_raw_scores[:4096].sum(dtype=np.float64)),
        }}

        regression_pool = ctboost.Pool(x_regression_train, regression_targets)
        started = time.perf_counter()
        regression_booster = ctboost.train(
            regression_pool,
            {{
                "objective": "SquaredError",
                "iterations": 5,
                "learning_rate": 0.1,
                "max_depth": 6,
                "task_type": "GPU",
                "verbose": True,
            }},
        )
        regression_fit_seconds = time.perf_counter() - started
        regression_predictions, regression_predict_seconds = timed_predict(regression_booster, x_regression_test)
        regression_predictions = np.asarray(regression_predictions, dtype=np.float32)
        results["regression"] = {{
            "fit_seconds": regression_fit_seconds,
            "predict_seconds": regression_predict_seconds,
            "raw_shape": list(regression_predictions.shape),
            "checksum": float(regression_predictions[:4096].sum(dtype=np.float64)),
        }}

        output_path = Path("/kaggle/working/ctboost_gpu_source_validate_results.json")
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        """
    ).strip() + "\n"


def _poll_kernel(client: KaggleClient, username: str, kernel_slug: str, timeout_seconds: int) -> KernelWorkerStatus:
    request = ApiGetKernelSessionStatusRequest()
    request.user_name = username
    request.kernel_slug = kernel_slug

    started = time.monotonic()
    last_status: KernelWorkerStatus | None = None
    while True:
        response = client.kernels.kernels_api_client.get_kernel_session_status(request)
        if response.status != last_status:
            print(f"status={response.status.name}")
            last_status = response.status

        if response.status in {KernelWorkerStatus.COMPLETE, KernelWorkerStatus.ERROR}:
            if response.failure_message:
                print(response.failure_message)
            return response.status

        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Kaggle kernel did not finish within {timeout_seconds} seconds")

        time.sleep(30)


def _download_outputs(
    username: str,
    kernel_slug: str,
    version_number: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    handle = f"{username}/{kernel_slug}/versions/{version_number}"
    try:
        kagglehub.notebook_output_download(handle, output_dir=str(output_dir), force_download=True)
        return
    except Exception as exc:  # pragma: no cover - fallback path for Kaggle Hub transport quirks
        raise RuntimeError(f"kagglehub notebook output download failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", "maiernator"))
    parser.add_argument("--token", default=os.environ.get("KAGGLE_API_TOKEN"))
    parser.add_argument("--slug", default="ctboost-gpu-source-validate-s6e4")
    parser.add_argument("--output-dir", default=".tmp/kaggle-kernel-output-source-validate")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("missing Kaggle API token: pass --token or set KAGGLE_API_TOKEN")

    repo_root = Path(__file__).resolve().parent
    base_commit = (
        os.popen(f'git -C "{repo_root}" rev-parse HEAD').read().strip()
    )
    if not base_commit:
        raise SystemExit("failed to resolve git HEAD")

    kaggle_auth.set_kaggle_api_token(args.token)
    whoami = kaggle_auth.whoami(verbose=False)
    resolved_username = str(whoami.get("username") or args.username)
    print(f"authenticated as {resolved_username}")

    client = KaggleClient(username=resolved_username, api_token=args.token, verbose=False)
    request = ApiSaveKernelRequest()
    request.slug = f"{resolved_username}/{args.slug}"
    request.new_title = "CTBoost GPU Source Validate S6E4"
    request.language = "python"
    request.kernel_type = "script"
    request.is_private = True
    request.enable_internet = True
    request.enable_gpu = True
    request.machine_shape = "NvidiaTeslaT4"
    request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    request.competition_data_sources = ["playground-series-s6e4"]
    request.session_timeout_seconds = args.timeout_seconds
    request.text = _build_kernel_source(repo_root, base_commit)

    response = client.kernels.kernels_api_client.save_kernel(request)
    if response.error:
        raise RuntimeError(response.error)

    print(json.dumps(response.to_dict(), indent=2))
    final_status = _poll_kernel(client, resolved_username, args.slug, args.timeout_seconds)

    output_dir = Path(args.output_dir)
    log_request = ApiListKernelSessionOutputRequest()
    log_request.user_name = resolved_username
    log_request.kernel_slug = args.slug
    log_request.page_size = 100
    log_response = client.kernels.kernels_api_client.list_kernel_session_output(log_request)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "session.log").write_text(log_response.log, encoding="utf-8")
    print(f"downloaded log with {len(log_response.files)} output file(s)")

    _download_outputs(resolved_username, args.slug, response.version_number, output_dir)
    print(f"kernel_url={response.url}")
    print(f"status={final_status.name}")
    print(f"output_dir={output_dir.resolve()}")


if __name__ == "__main__":
    main()
