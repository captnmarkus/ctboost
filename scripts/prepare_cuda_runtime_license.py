"""Fetch the exact NVIDIA CUDA Runtime license embedded in release wheels.

CUDA-enabled wheels bundle ``cudart``.  NVIDIA publishes the runtime archive,
its checksum, and its governing license in the CUDA 12.8.1 redistribution
manifest.  Release jobs use this helper rather than copying license text from
an unversioned toolkit installation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

CUDA_REDISTRIBUTION_MANIFEST_URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json"
)
CUDA_RUNTIME_ARCHIVE_URL = (
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/"
    "linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz"
)
CUDA_RUNTIME_ARCHIVE_SHA256 = (
    "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68"
)
CUDA_RUNTIME_LICENSE_SHA256 = (
    "e2c71babfd18a8e69542dd7e9ca018f9caa438094001a58e6bc4d8c999bf0d07"
)
CUDA_RUNTIME_LICENSE_BYTES = 63_021
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024


class LicensePreparationError(RuntimeError):
    """Raised when the authoritative runtime license cannot be verified."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_cuda_runtime_license(
    archive: bytes,
    *,
    expected_sha256: str = CUDA_RUNTIME_LICENSE_SHA256,
    expected_size: int = CUDA_RUNTIME_LICENSE_BYTES,
) -> bytes:
    """Extract and verify the sole ``LICENSE`` file from a cudart archive."""

    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as bundle:
            members = [
                member
                for member in bundle.getmembers()
                if member.isfile() and member.name.rstrip("/").endswith("/LICENSE")
            ]
            if len(members) != 1:
                raise LicensePreparationError(
                    "expected exactly one regular LICENSE file in the CUDA "
                    "runtime archive, "
                    f"found {len(members)}"
                )
            member = members[0]
            if member.size != expected_size:
                raise LicensePreparationError(
                    "CUDA runtime license size mismatch: "
                    f"expected {expected_size} bytes, found {member.size}"
                )
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise LicensePreparationError("could not read the CUDA runtime license")
            payload = extracted.read(expected_size + 1)
    except LicensePreparationError:
        raise
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise LicensePreparationError(f"invalid CUDA runtime archive: {exc}") from exc

    if len(payload) != expected_size:
        raise LicensePreparationError(
            "CUDA runtime license payload size mismatch: "
            f"expected {expected_size} bytes, read {len(payload)}"
        )
    actual_sha256 = _sha256(payload)
    if actual_sha256 != expected_sha256.lower():
        raise LicensePreparationError(
            "CUDA runtime license SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return payload


def download_cuda_runtime_license(
    *,
    url: str = CUDA_RUNTIME_ARCHIVE_URL,
    archive_sha256: str = CUDA_RUNTIME_ARCHIVE_SHA256,
    opener: Callable[..., object] = urllib.request.urlopen,
    expected_license_sha256: str = CUDA_RUNTIME_LICENSE_SHA256,
    expected_license_size: int = CUDA_RUNTIME_LICENSE_BYTES,
) -> bytes:
    """Download the pinned cudart archive and return its verified license."""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ctboost-release-license-fetcher/1"},
    )
    try:
        with opener(request, timeout=60) as response:  # type: ignore[attr-defined]
            archive = response.read(MAX_ARCHIVE_BYTES + 1)
    except Exception as exc:
        raise LicensePreparationError(
            f"could not download the pinned CUDA runtime archive: {exc}"
        ) from exc

    if len(archive) > MAX_ARCHIVE_BYTES:
        raise LicensePreparationError(
            f"CUDA runtime archive exceeded the {MAX_ARCHIVE_BYTES}-byte safety limit"
        )
    actual_archive_sha256 = _sha256(archive)
    if actual_archive_sha256 != archive_sha256.lower():
        raise LicensePreparationError(
            "CUDA runtime archive SHA-256 mismatch: "
            f"expected {archive_sha256}, found {actual_archive_sha256}"
        )
    return extract_cuda_runtime_license(
        archive,
        expected_sha256=expected_license_sha256,
        expected_size=expected_license_size,
    )


def write_cuda_runtime_license(output: Path, payload: bytes) -> None:
    """Atomically write a verified license for the wheel builder."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(output.parent),
            prefix=output.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def prepare_cuda_runtime_license(output: Path) -> Path:
    """Fetch, verify, and write the CUDA license required by GPU wheels."""

    payload = download_cuda_runtime_license()
    write_cuda_runtime_license(output, payload)
    return output.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the pinned NVIDIA cudart license for a CTBoost release wheel."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "NVIDIA-CUDA-Toolkit-LICENSE.txt",
        help="destination file (default: repository root)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        output = prepare_cuda_runtime_license(arguments.output)
    except LicensePreparationError as exc:
        print(f"prepare_cuda_runtime_license: {exc}")
        return 1
    print(f"Verified NVIDIA CUDA Runtime license: {output}")
    print(f"Source manifest: {CUDA_REDISTRIBUTION_MANIFEST_URL}")
    print(f"SHA-256: {CUDA_RUNTIME_LICENSE_SHA256}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
