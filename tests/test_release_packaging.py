import hashlib
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Set

import pytest

from scripts import prepare_cuda_runtime_license as cuda_license
from scripts import validate_release_artifacts as release_artifacts


def _cuda_license_archive(payload: bytes, *, members: int = 1) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for index in range(members):
            info = tarfile.TarInfo(f"cuda-cudart/archive-{index}/LICENSE")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_extract_cuda_runtime_license_verifies_size_hash_and_uniqueness():
    payload = b"authoritative license bytes\n"
    archive = _cuda_license_archive(payload)

    assert (
        cuda_license.extract_cuda_runtime_license(
            archive,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
        )
        == payload
    )

    with pytest.raises(cuda_license.LicensePreparationError, match="exactly one"):
        cuda_license.extract_cuda_runtime_license(
            _cuda_license_archive(payload, members=2),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
        )

    with pytest.raises(cuda_license.LicensePreparationError, match="license SHA-256"):
        cuda_license.extract_cuda_runtime_license(
            archive,
            expected_sha256="0" * 64,
            expected_size=len(payload),
        )


def test_download_cuda_runtime_license_verifies_archive_before_extraction():
    payload = b"license\n"
    archive = _cuda_license_archive(payload)

    result = cuda_license.download_cuda_runtime_license(
        url="https://example.invalid/cudart.tar.xz",
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        opener=lambda *_args, **_kwargs: io.BytesIO(archive),
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
        expected_license_size=len(payload),
    )
    assert result == payload

    with pytest.raises(cuda_license.LicensePreparationError, match="archive SHA-256"):
        cuda_license.download_cuda_runtime_license(
            url="https://example.invalid/cudart.tar.xz",
            archive_sha256="f" * 64,
            opener=lambda *_args, **_kwargs: io.BytesIO(archive),
            expected_license_sha256=hashlib.sha256(payload).hexdigest(),
            expected_license_size=len(payload),
        )


def test_write_cuda_runtime_license_replaces_destination(tmp_path):
    output = tmp_path / "NVIDIA-CUDA-Toolkit-LICENSE.txt"
    output.write_bytes(b"old")

    cuda_license.write_cuda_runtime_license(output, b"verified")

    assert output.read_bytes() == b"verified"
    assert not list(tmp_path.glob("*.tmp"))


def _platform_tag(platform_family: str) -> str:
    return {
        "linux-x86_64": "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
        "linux-aarch64": "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
        "windows-amd64": "win_amd64",
        "macos-x86_64": "macosx_10_15_x86_64",
    }[platform_family]


def _write_wheel(
    directory: Path,
    *,
    version: str,
    platform_family: str,
    python_tag: str,
    cuda: bool,
    cuda_license_payload: bytes,
    include_cudart: bool = True,
    include_cuda_license: bool = True,
    build_tag: str = "",
) -> Path:
    platform_tag = _platform_tag(platform_family)
    build_component = f"-{build_tag}" if build_tag else ""
    filename = (
        f"ctboost-{version}{build_component}-{python_tag}-{python_tag}-"
        f"{platform_tag}.whl"
    )
    path = directory / filename
    dist_info = f"ctboost-{version}.dist-info"
    wheel_lines = ["Wheel-Version: 1.0", "Root-Is-Purelib: false"]
    if build_tag:
        wheel_lines.append(f"Build: {build_tag}")
    wheel_lines.extend(
        f"Tag: {python_tag}-{python_tag}-{tag}" for tag in platform_tag.split(".")
    )
    with zipfile.ZipFile(path, mode="w") as wheel:
        wheel.writestr(f"{dist_info}/WHEEL", "\n".join(wheel_lines) + "\n")
        wheel.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.2\nName: ctboost\nVersion: {version}\n",
        )
        extension = "_core.pyd" if platform_family == "windows-amd64" else "_core.so"
        wheel.writestr(f"ctboost/{extension}", b"native-extension")
        if cuda and include_cudart:
            runtime = (
                "ctboost.libs/cudart64_12-feedface.dll"
                if platform_family == "windows-amd64"
                else "ctboost.libs/libcudart-feedface.so.12.8.90"
            )
            wheel.writestr(runtime, b"runtime")
        if cuda and include_cuda_license:
            wheel.writestr(
                f"{dist_info}/licenses/NVIDIA-CUDA-Toolkit-LICENSE.txt",
                cuda_license_payload,
            )
    return path


def _write_sdist(
    directory: Path,
    *,
    version: str,
    metadata_name: str = "ctboost",
    metadata_version: Optional[str] = None,
    omitted: Optional[Set[str]] = None,
    extra_members: Optional[Dict[str, bytes]] = None,
) -> Path:
    root = f"ctboost-{version}"
    omitted = set() if omitted is None else omitted
    metadata_version = version if metadata_version is None else metadata_version
    payloads = {
        "PKG-INFO": (
            "Metadata-Version: 2.2\n"
            f"Name: {metadata_name}\n"
            f"Version: {metadata_version}\n"
        ).encode(),
        "CMakeLists.txt": (
            "add_library(ctboost_core src/core/booster.cpp)\n"
            "target_sources(ctboost_core PRIVATE cuda/cuda_backend.cu)\n"
        ).encode(),
        "LICENSE": b"project license\n",
        "README.md": b"# CTBoost\n",
        "pyproject.toml": b"[project]\nname = 'ctboost'\n",
        "ctboost/__init__.py": b"",
        "ctboost/_version.py": f'__version__ = "{version}"\n'.encode(),
        "ctboost/gpu_install.py": b"",
        "scripts/prepare_cuda_runtime_license.py": b"",
        "scripts/validate_release_artifacts.py": b"",
        "src/core/booster.cpp": b"",
        "cuda/cuda_backend.cu": b"",
    }
    payloads.update(extra_members or {})
    path = directory / f"ctboost-{version}.tar.gz"
    with tarfile.open(path, mode="w:gz") as archive:
        for relative_name, payload in sorted(payloads.items()):
            if relative_name in omitted:
                continue
            member_name = (
                relative_name
                if relative_name.startswith("../") or relative_name.startswith("/")
                else f"{root}/{relative_name}"
            )
            info = tarfile.TarInfo(member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _write_complete_release(
    directory: Path,
    *,
    version: str = "0.1.54",
    cuda_license_payload: bytes = b"license",
) -> None:
    directory.mkdir()
    for (platform_family, python_tag), cuda in sorted(
        release_artifacts.expected_release_matrix().items()
    ):
        _write_wheel(
            directory,
            version=version,
            platform_family=platform_family,
            python_tag=python_tag,
            cuda=cuda,
            cuda_license_payload=cuda_license_payload,
        )
    _write_sdist(directory, version=version)


def test_release_matrix_has_one_wheel_per_tag_and_validates(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)

    matrix = release_artifacts.expected_release_matrix()
    assert len(matrix) == 26
    assert sum(matrix.values()) == 10
    assert (
        release_artifacts.validate_release_artifacts(
            release,
            version="0.1.54",
            expected_license_sha256=hashlib.sha256(payload).hexdigest(),
        )
        == []
    )


def test_release_validator_rejects_missing_cuda_runtime_and_license(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    target = release / (
        "ctboost-0.1.54-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    )
    target.unlink()
    _write_wheel(
        release,
        version="0.1.54",
        platform_family="linux-x86_64",
        python_tag="cp312",
        cuda=True,
        cuda_license_payload=payload,
        include_cudart=False,
        include_cuda_license=False,
    )

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any("must bundle exactly one cudart" in error for error in errors)
    assert any("authoritative NVIDIA CUDA license" in error for error in errors)


def test_release_validator_rejects_wheel_build_tags(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    target = release / "ctboost-0.1.54-cp312-cp312-win_amd64.whl"
    target.unlink()
    _write_wheel(
        release,
        version="0.1.54",
        platform_family="windows-amd64",
        python_tag="cp312",
        cuda=True,
        cuda_license_payload=payload,
        build_tag="1gpu",
    )

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any("wheel build tags are forbidden" in error for error in errors)
    assert any("missing wheel for windows-amd64/cp312" in error for error in errors)


def test_release_validator_rejects_unexpected_artifacts_and_bundled_driver(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    (release / "debug.log").write_text("not for release", encoding="utf-8")
    target = release / "ctboost-0.1.54-cp313-cp313-win_amd64.whl"
    with zipfile.ZipFile(target, mode="a") as wheel:
        wheel.writestr("ctboost.libs/nvcuda.dll", b"driver")

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any("unexpected release artifact" in error for error in errors)
    assert any("must never bundle the NVIDIA driver" in error for error in errors)


def test_release_validator_inspects_sdist_metadata_and_rebuild_sources(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    (release / "ctboost-0.1.54.tar.gz").unlink()
    _write_sdist(
        release,
        version="0.1.54",
        metadata_name="not-ctboost",
        metadata_version="9.9.9",
        omitted={
            "scripts/validate_release_artifacts.py",
            "src/core/booster.cpp",
        },
    )

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any("PKG-INFO Name" in error for error in errors)
    assert any("PKG-INFO Version" in error for error in errors)
    assert any(
        "missing required file scripts/validate_release_artifacts.py" in error
        for error in errors
    )
    assert any(
        "references missing source file src/core/booster.cpp" in error
        for error in errors
    )


def test_release_validator_rejects_unsafe_sdist_paths(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    (release / "ctboost-0.1.54.tar.gz").unlink()
    _write_sdist(
        release,
        version="0.1.54",
        extra_members={"../outside.txt": b"escape"},
    )

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any("unsafe or out-of-root sdist member path" in error for error in errors)


def test_release_validator_rejects_unexpected_release_scripts(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    (release / "ctboost-0.1.54.tar.gz").unlink()
    _write_sdist(
        release,
        version="0.1.54",
        extra_members={"scripts/local_release_helper.py": b"local only\n"},
    )

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any(
        "unexpected release script scripts/local_release_helper.py" in error
        for error in errors
    )


def test_release_validator_rejects_unsafe_duplicate_native_wheel(tmp_path):
    payload = b"pinned NVIDIA license"
    release = tmp_path / "dist"
    _write_complete_release(release, cuda_license_payload=payload)
    target = release / "ctboost-0.1.54-cp312-cp312-win_amd64.whl"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(target, mode="a") as wheel:
            wheel.writestr("ctboost/_core.pyd", b"duplicate-native-extension")
            wheel.writestr("../outside.dll", b"unsafe")

    errors = release_artifacts.validate_release_artifacts(
        release,
        version="0.1.54",
        expected_license_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert any("unsafe wheel member path" in error for error in errors)
    assert any("duplicate wheel member path" in error for error in errors)
    assert any("exactly one CTBoost native extension" in error for error in errors)
