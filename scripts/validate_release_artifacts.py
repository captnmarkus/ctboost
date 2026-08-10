"""Validate CTBoost's release matrix before any artifact is published.

The ordinary package intentionally has one compatible wheel for every
supported interpreter/platform pair.  On CPython 3.10+ the Linux x86-64 and
Windows wheels contain CUDA support; publishing an additional CPU wheel for
the same pair would make pip choose between implementation variants using an
opaque wheel build tag.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from prepare_cuda_runtime_license import CUDA_RUNTIME_LICENSE_SHA256
except ImportError:  # Imported as ``scripts.validate_release_artifacts`` in tests.
    from scripts.prepare_cuda_runtime_license import CUDA_RUNTIME_LICENSE_SHA256


_ALL_CPYTHON_TAGS = ("cp38", "cp39", "cp310", "cp311", "cp312", "cp313", "cp314")
_CUDA_CPYTHON_TAGS = ("cp310", "cp311", "cp312", "cp313", "cp314")
_CPU_X86_CPYTHON_TAGS = ("cp38", "cp39")

_LINUX_X86_64 = "linux-x86_64"
_LINUX_AARCH64 = "linux-aarch64"
_WINDOWS_AMD64 = "windows-amd64"
_MACOS_X86_64 = "macos-x86_64"

_MAX_SDIST_MEMBERS = 20_000
_MAX_SDIST_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
_MAX_METADATA_BYTES = 2 * 1024 * 1024
_MAX_CMAKE_BYTES = 2 * 1024 * 1024
_REQUIRED_SDIST_FILES = frozenset(
    {
        "CMakeLists.txt",
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "ctboost/__init__.py",
        "ctboost/_version.py",
        "ctboost/gpu_install.py",
        "scripts/prepare_cuda_runtime_license.py",
        "scripts/validate_release_artifacts.py",
    }
)
_RELEASE_SCRIPT_FILES = frozenset(
    {
        "scripts/prepare_cuda_runtime_license.py",
        "scripts/validate_release_artifacts.py",
    }
)
_CMAKE_SOURCE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:src|cuda)/[A-Za-z0-9_./+-]+\.(?:cpp|hpp|cu|cuh))"
)


@dataclass(frozen=True)
class WheelIdentity:
    filename: str
    version: str
    python_tag: str
    abi_tag: str
    platform_tag: str
    platform_family: str

    @property
    def matrix_key(self) -> Tuple[str, str]:
        return self.platform_family, self.python_tag


def expected_release_matrix() -> Dict[Tuple[str, str], bool]:
    """Return ``(platform, Python tag) -> CUDA expected`` for a release."""

    expected: Dict[Tuple[str, str], bool] = {}
    for python_tag in _ALL_CPYTHON_TAGS:
        expected[(_LINUX_AARCH64, python_tag)] = False
    for python_tag in _CUDA_CPYTHON_TAGS:
        expected[(_MACOS_X86_64, python_tag)] = False
        expected[(_LINUX_X86_64, python_tag)] = True
        expected[(_WINDOWS_AMD64, python_tag)] = True
    for python_tag in _CPU_X86_CPYTHON_TAGS:
        expected[(_LINUX_X86_64, python_tag)] = False
        expected[(_WINDOWS_AMD64, python_tag)] = False
    return expected


def _platform_family(platform_tag: str) -> str:
    component_tags = platform_tag.split(".")
    if component_tags == ["win_amd64"]:
        return _WINDOWS_AMD64
    if component_tags and all(
        tag.startswith("macosx_") and tag.endswith("_x86_64") for tag in component_tags
    ):
        return _MACOS_X86_64
    if component_tags and all(
        tag.startswith("manylinux_") and tag.endswith("_x86_64")
        for tag in component_tags
    ):
        return _LINUX_X86_64
    if component_tags and all(
        tag.startswith("manylinux_") and tag.endswith("_aarch64")
        for tag in component_tags
    ):
        return _LINUX_AARCH64
    raise ValueError(f"unsupported or mixed platform tag: {platform_tag}")


def parse_wheel_identity(filename: str) -> WheelIdentity:
    """Parse the deliberately narrow CTBoost wheel filename contract."""

    if not filename.endswith(".whl"):
        raise ValueError("wheel filename must end in .whl")
    parts = filename[:-4].split("-")
    if len(parts) == 6:
        raise ValueError(
            f"wheel build tags are forbidden in release artifacts (found {parts[2]!r})"
        )
    if len(parts) != 5:
        raise ValueError(
            f"expected an untagged five-part wheel filename, found {len(parts)} parts"
        )
    distribution, version, python_tag, abi_tag, platform_tag = parts
    if distribution != "ctboost":
        raise ValueError(f"expected distribution 'ctboost', found {distribution!r}")
    if python_tag not in _ALL_CPYTHON_TAGS:
        raise ValueError(f"unsupported Python tag: {python_tag}")
    if abi_tag != python_tag:
        raise ValueError(
            f"expected matching CPython ABI tag {python_tag}, found {abi_tag}"
        )
    return WheelIdentity(
        filename=filename,
        version=version,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
        platform_family=_platform_family(platform_tag),
    )


def _single_entry(names: Iterable[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} entry, found {len(matches)}")
    return matches[0]


def _metadata_headers(payload: str) -> Dict[str, List[str]]:
    headers: Dict[str, List[str]] = {}
    for line in payload.splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers.setdefault(name.strip().lower(), []).append(value.strip())
    return headers


def _safe_sdist_member_name(name: str, *, root: str) -> Optional[str]:
    """Return a path relative to the sdist root, or ``None`` when unsafe."""

    if not name or "\\" in name or "\x00" in name:
        return None
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    if not path.parts or path.parts[0] != root:
        return None
    if len(path.parts) == 1:
        return ""
    return PurePosixPath(*path.parts[1:]).as_posix()


def _safe_wheel_member_name(name: str) -> bool:
    """Return whether a ZIP member is a portable, relative wheel path."""

    if not name or "\\" in name or "\x00" in name:
        return False
    normalized = name[:-1] if name.endswith("/") else name
    if not normalized or normalized.startswith("/"):
        return False
    parts = normalized.split("/")
    return not any(part in {"", ".", ".."} or ":" in part for part in parts)


def _read_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    limit: int,
) -> bytes:
    if member.size > limit:
        raise ValueError(f"{member.name} exceeds the {limit}-byte inspection limit")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"could not read {member.name}")
    payload = extracted.read(limit + 1)
    if len(payload) != member.size:
        raise ValueError(
            f"{member.name} declared {member.size} bytes but yielded {len(payload)}"
        )
    return payload


def _sdist_errors(sdist_path: Path, *, expected_version: str) -> List[str]:
    """Inspect an sdist without extracting it and return deterministic errors."""

    errors: List[str] = []
    root = f"ctboost-{expected_version}"
    relative_members: Dict[str, tarfile.TarInfo] = {}
    try:
        with tarfile.open(sdist_path, mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > _MAX_SDIST_MEMBERS:
                errors.append(
                    f"source distribution has {len(members)} members, limit is "
                    f"{_MAX_SDIST_MEMBERS}"
                )
            total_size = sum(member.size for member in members)
            if total_size > _MAX_SDIST_UNCOMPRESSED_BYTES:
                errors.append(
                    f"source distribution expands to {total_size} bytes, limit is "
                    f"{_MAX_SDIST_UNCOMPRESSED_BYTES}"
                )

            for member in members:
                relative_name = _safe_sdist_member_name(member.name, root=root)
                if relative_name is None:
                    errors.append(
                        f"unsafe or out-of-root sdist member path: {member.name!r}"
                    )
                    continue
                if not (member.isfile() or member.isdir()):
                    errors.append(
                        "sdist member must be a regular file or directory: "
                        f"{member.name!r}"
                    )
                    continue
                if relative_name in relative_members:
                    errors.append(f"duplicate sdist member path: {member.name!r}")
                    continue
                relative_members[relative_name] = member

            missing_files = sorted(
                required
                for required in _REQUIRED_SDIST_FILES
                if required not in relative_members
                or not relative_members[required].isfile()
            )
            for missing_file in missing_files:
                errors.append(
                    f"source distribution is missing required file {missing_file}"
                )

            unexpected_release_scripts = sorted(
                relative_name
                for relative_name, member in relative_members.items()
                if relative_name.startswith("scripts/")
                and member.isfile()
                and relative_name not in _RELEASE_SCRIPT_FILES
            )
            for unexpected_release_script in unexpected_release_scripts:
                errors.append(
                    "source distribution contains unexpected release script "
                    f"{unexpected_release_script}"
                )

            pkg_info = relative_members.get("PKG-INFO")
            if pkg_info is None or not pkg_info.isfile():
                errors.append("source distribution must contain one root PKG-INFO file")
            else:
                try:
                    pkg_headers = _metadata_headers(
                        _read_tar_member(
                            archive, pkg_info, limit=_MAX_METADATA_BYTES
                        ).decode("utf-8", errors="strict")
                    )
                except (UnicodeDecodeError, ValueError) as exc:
                    errors.append(f"could not inspect PKG-INFO: {exc}")
                else:
                    if pkg_headers.get("name") != ["ctboost"]:
                        errors.append(
                            "PKG-INFO Name is "
                            f"{pkg_headers.get('name')!r}, expected ['ctboost']"
                        )
                    if pkg_headers.get("version") != [expected_version]:
                        errors.append(
                            "PKG-INFO Version is "
                            f"{pkg_headers.get('version')!r}, "
                            f"expected [{expected_version!r}]"
                        )

            cmake_info = relative_members.get("CMakeLists.txt")
            if cmake_info is not None and cmake_info.isfile():
                try:
                    cmake_payload = _read_tar_member(
                        archive, cmake_info, limit=_MAX_CMAKE_BYTES
                    ).decode("utf-8", errors="strict")
                except (UnicodeDecodeError, ValueError) as exc:
                    errors.append(f"could not inspect CMakeLists.txt: {exc}")
                else:
                    source_references = set(
                        _CMAKE_SOURCE_REFERENCE.findall(cmake_payload)
                    )
                    if not source_references:
                        errors.append("CMakeLists.txt declares no native source files")
                    for source_reference in sorted(source_references):
                        source_member = relative_members.get(source_reference)
                        if source_member is None or not source_member.isfile():
                            errors.append(
                                "CMakeLists.txt references missing source file "
                                f"{source_reference}"
                            )
    except (OSError, EOFError, tarfile.TarError) as exc:
        errors.append(f"could not inspect source distribution: {exc}")
    return errors


def _wheel_errors(
    wheel_path: Path,
    identity: WheelIdentity,
    *,
    expected_version: str,
    expected_cuda: bool,
    expected_license_sha256: str,
) -> List[str]:
    errors: List[str] = []
    if identity.version != expected_version:
        errors.append(
            f"filename version is {identity.version}, expected {expected_version}"
        )
    try:
        with zipfile.ZipFile(wheel_path) as wheel:
            bad_member = wheel.testzip()
            if bad_member is not None:
                errors.append(f"ZIP CRC failed for {bad_member}")
            names = wheel.namelist()
            unsafe_names = sorted(
                name for name in names if not _safe_wheel_member_name(name)
            )
            for unsafe_name in unsafe_names:
                errors.append(f"unsafe wheel member path: {unsafe_name!r}")
            duplicate_names = sorted(
                name for name in set(names) if names.count(name) > 1
            )
            for duplicate_name in duplicate_names:
                errors.append(f"duplicate wheel member path: {duplicate_name!r}")

            try:
                wheel_metadata_name = _single_entry(names, ".dist-info/WHEEL")
                package_metadata_name = _single_entry(names, ".dist-info/METADATA")
            except ValueError as exc:
                errors.append(str(exc))
                return errors

            wheel_headers = _metadata_headers(
                wheel.read(wheel_metadata_name).decode("utf-8", errors="strict")
            )
            if wheel_headers.get("build"):
                errors.append(
                    f"WHEEL metadata contains forbidden Build: {wheel_headers['build']}"
                )
            expected_tags = {
                f"{identity.python_tag}-{identity.abi_tag}-{platform}"
                for platform in identity.platform_tag.split(".")
            }
            actual_tags = set(wheel_headers.get("tag", []))
            if actual_tags != expected_tags:
                errors.append(
                    f"WHEEL tags {sorted(actual_tags)} do not match filename tags "
                    f"{sorted(expected_tags)}"
                )

            package_headers = _metadata_headers(
                wheel.read(package_metadata_name).decode("utf-8", errors="strict")
            )
            if package_headers.get("name") != ["ctboost"]:
                errors.append(
                    f"METADATA Name is {package_headers.get('name')!r}, "
                    "expected ['ctboost']"
                )
            if package_headers.get("version") != [expected_version]:
                errors.append(
                    "METADATA Version is "
                    f"{package_headers.get('version')!r}, expected "
                    f"[{expected_version!r}]"
                )

            native_suffix = (
                ".pyd" if identity.platform_family == _WINDOWS_AMD64 else ".so"
            )
            native_entries = [
                name
                for name in names
                if name.startswith("ctboost/_core") and name.endswith(native_suffix)
            ]
            if len(native_entries) != 1:
                errors.append(
                    "wheel must include exactly one CTBoost native extension "
                    f"ending in {native_suffix}, found {len(native_entries)}"
                )

            basenames = [Path(name).name.lower() for name in names]
            cudart_entries = [
                names[index]
                for index, basename in enumerate(basenames)
                if (basename.startswith("libcudart") and ".so" in basename)
                or (basename.startswith("cudart64_") and basename.endswith(".dll"))
            ]
            driver_entries = [
                names[index]
                for index, basename in enumerate(basenames)
                if re.match(r"^libcuda(?:-[^.]+)?\.so(?:\.|$)", basename)
                or re.match(r"^nvcuda(?:-[^.]+)?\.dll$", basename)
            ]
            if driver_entries:
                errors.append(
                    "wheel must never bundle the NVIDIA driver: "
                    + ", ".join(driver_entries)
                )
            if expected_cuda and len(cudart_entries) != 1:
                errors.append(
                    "CUDA wheel must bundle exactly one cudart library, found "
                    f"{len(cudart_entries)}"
                )
            if not expected_cuda and cudart_entries:
                errors.append(
                    "CPU wheel unexpectedly bundles cudart: "
                    + ", ".join(cudart_entries)
                )

            license_entries = [
                name
                for name in names
                if name.endswith(".dist-info/licenses/NVIDIA-CUDA-Toolkit-LICENSE.txt")
            ]
            if expected_cuda:
                if len(license_entries) != 1:
                    errors.append(
                        "CUDA wheel must include exactly one authoritative NVIDIA "
                        "CUDA license, "
                        f"found {len(license_entries)}"
                    )
                else:
                    actual_license_sha256 = hashlib.sha256(
                        wheel.read(license_entries[0])
                    ).hexdigest()
                    if actual_license_sha256 != expected_license_sha256.lower():
                        errors.append(
                            "NVIDIA CUDA license SHA-256 mismatch: "
                            f"expected {expected_license_sha256}, found "
                            f"{actual_license_sha256}"
                        )
            elif license_entries:
                errors.append(
                    "CPU wheel unexpectedly includes the NVIDIA CUDA runtime license"
                )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        errors.append(f"could not inspect wheel: {exc}")
    return errors


def validate_release_artifacts(
    directory: Path,
    *,
    version: str,
    expected_license_sha256: str = CUDA_RUNTIME_LICENSE_SHA256,
) -> List[str]:
    """Return deterministic validation errors for an assembled release directory."""

    directory = Path(directory)
    if not directory.is_dir():
        return [f"release directory does not exist: {directory}"]

    expected = expected_release_matrix()
    seen: Set[Tuple[str, str]] = set()
    errors: List[str] = []
    expected_sdist = f"ctboost-{version}.tar.gz"
    seen_sdist = False

    for artifact in sorted(path for path in directory.iterdir() if path.is_file()):
        if artifact.name == expected_sdist:
            if seen_sdist:
                errors.append(f"{artifact.name}: duplicate source distribution")
            seen_sdist = True
            for sdist_error in _sdist_errors(artifact, expected_version=version):
                errors.append(f"{artifact.name}: {sdist_error}")
            continue
        if not artifact.name.endswith(".whl"):
            errors.append(f"{artifact.name}: unexpected release artifact")
            continue
        try:
            identity = parse_wheel_identity(artifact.name)
        except ValueError as exc:
            errors.append(f"{artifact.name}: {exc}")
            continue
        if identity.matrix_key not in expected:
            errors.append(
                f"{artifact.name}: unexpected matrix entry "
                f"{identity.platform_family}/{identity.python_tag}"
            )
            continue
        if identity.matrix_key in seen:
            errors.append(
                f"{artifact.name}: duplicate matrix entry "
                f"{identity.platform_family}/{identity.python_tag}"
            )
            continue
        seen.add(identity.matrix_key)
        for wheel_error in _wheel_errors(
            artifact,
            identity,
            expected_version=version,
            expected_cuda=expected[identity.matrix_key],
            expected_license_sha256=expected_license_sha256,
        ):
            errors.append(f"{artifact.name}: {wheel_error}")

    missing = sorted(set(expected).difference(seen))
    for platform_family, python_tag in missing:
        errors.append(f"missing wheel for {platform_family}/{python_tag}")
    if not seen_sdist:
        errors.append(f"missing source distribution {expected_sdist}")
    return sorted(errors)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the complete CTBoost wheel/sdist release matrix."
    )
    parser.add_argument("directory", type=Path, help="assembled release directory")
    parser.add_argument(
        "--version", required=True, help="expected public release version"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    errors = validate_release_artifacts(arguments.directory, version=arguments.version)
    if errors:
        print("CTBoost release artifact validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    matrix = expected_release_matrix()
    cuda_count = sum(matrix.values())
    print(
        f"Validated {len(matrix)} wheels ({cuda_count} CUDA-enabled, "
        f"{len(matrix) - cuda_count} CPU-only) and one source distribution "
        f"for CTBoost {arguments.version}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
