import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_cuda_backend_resolves_driver_entry_point_through_cudart():
    cmake_source = (_REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda_source = (_REPOSITORY_ROOT / "cuda" / "cuda_backend.cu").read_text(
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
    cuda_source = (_REPOSITORY_ROOT / "cuda" / "cuda_backend.cu").read_text(
        encoding="utf-8"
    )

    assert "entry_point.runtime_status != cudaSuccess" in cuda_source
    assert "entry_point.driver_status != cudaDriverEntryPointSuccess" in cuda_source
    assert "entry_point.function == nullptr" in cuda_source
    assert "CUDA driver does not expose an ABI-compatible" in cuda_source
