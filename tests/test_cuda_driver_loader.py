import os
import re
from pathlib import Path

import pytest


def _repository_root() -> Path:
    candidates = [Path(__file__).resolve().parents[1]]
    github_workspace = os.environ.get("GITHUB_WORKSPACE")
    if github_workspace:
        candidates.append(Path(github_workspace).resolve())
    for candidate in candidates:
        if (candidate / "CMakeLists.txt").is_file() and (
            candidate / "cuda" / "cuda_backend.cu"
        ).is_file():
            return candidate
    pytest.skip("CUDA source contract requires a repository checkout")


def test_cuda_backend_resolves_driver_entry_point_through_cudart():
    repository_root = _repository_root()
    cmake_source = (repository_root / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda_source = (repository_root / "cuda" / "cuda_backend.cu").read_text(
        encoding="utf-8"
    )

    assert "CUDA::cuda_driver" not in cmake_source
    assert re.search(r"PUBLIC\s+CUDA::cudart\b", cmake_source)
    assert "#include <cudaTypedefs.h>" in cuda_source
    assert (
        'cudaGetDriverEntryPointByVersion(\n        "cuMemGetAddressRange"'
        in cuda_source
    )
    assert "kCuMemGetAddressRangeAbiVersion = 3020U" in cuda_source
    assert "PFN_cuMemGetAddressRange_v3020" in cuda_source
    assert not re.search(r"\bcuMemGetAddressRange\s*\(", cuda_source)


def test_cuda_driver_entry_point_lookup_fails_closed():
    cuda_source = (_repository_root() / "cuda" / "cuda_backend.cu").read_text(
        encoding="utf-8"
    )

    assert "entry_point.runtime_status != cudaSuccess" in cuda_source
    assert "entry_point.driver_status != cudaDriverEntryPointSuccess" in cuda_source
    assert "entry_point.function == nullptr" in cuda_source
    assert "CUDA driver does not expose an ABI-compatible" in cuda_source
