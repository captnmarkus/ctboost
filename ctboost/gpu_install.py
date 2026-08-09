"""Install the matching CUDA-enabled CTBoost wheel from GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from ._version import __version__


_REPOSITORY = "captnmarkus/ctboost"


def _cpython_tag() -> str:
    if sys.implementation.name != "cpython":
        raise RuntimeError("prebuilt CTBoost GPU wheels currently require CPython")
    return "cp%d%d" % (sys.version_info.major, sys.version_info.minor)


def _platform_suffix() -> str:
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise RuntimeError("prebuilt CTBoost GPU wheels currently require an x86-64 machine")
    if sys.platform == "win32":
        return "win_amd64.whl"
    if sys.platform.startswith("linux"):
        return "x86_64.whl"
    raise RuntimeError("prebuilt CTBoost GPU wheels are currently available for Linux and Windows")


def select_gpu_asset(
    assets: Iterable[Dict[str, Any]],
    *,
    version: str,
    python_tag: Optional[str] = None,
    platform_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """Select exactly one wheel from a GitHub release asset listing."""

    tag = _cpython_tag() if python_tag is None else str(python_tag)
    suffix = _platform_suffix() if platform_suffix is None else str(platform_suffix)
    prefix = "ctboost-%s-1gpu-%s-%s-" % (version, tag, tag)
    matches = [
        dict(asset)
        for asset in assets
        if str(asset.get("name", "")).startswith(prefix)
        and str(asset.get("name", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one CTBoost GPU wheel matching %s*%s, found %d"
            % (prefix, suffix, len(matches))
        )
    asset = matches[0]
    if not asset.get("browser_download_url"):
        raise RuntimeError("the matching GitHub release asset has no download URL")
    return asset


def _release_assets(version: str, *, repository: str = _REPOSITORY) -> Sequence[Dict[str, Any]]:
    url = "https://api.github.com/repos/%s/releases/tags/v%s" % (repository, version)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ctboost-gpu-installer"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:
        raise RuntimeError("could not read CTBoost release metadata from GitHub: %s" % exc) from exc
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("GitHub release metadata did not contain an asset list")
    return assets


def _download_verified(asset: Dict[str, Any], destination: Path) -> str:
    request = urllib.request.Request(
        str(asset["browser_download_url"]),
        headers={"Accept": "application/octet-stream", "User-Agent": "ctboost-gpu-installer"},
    )
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
    except Exception as exc:
        raise RuntimeError("could not download the CTBoost GPU wheel: %s" % exc) from exc

    expected_size = asset.get("size")
    if expected_size is not None and received != int(expected_size):
        raise RuntimeError(
            "downloaded GPU wheel size mismatch: expected %d bytes, received %d"
            % (int(expected_size), received)
        )
    actual_digest = digest.hexdigest()
    expected_digest = str(asset.get("digest") or "")
    if not expected_digest.startswith("sha256:"):
        raise RuntimeError("GitHub did not provide a SHA-256 digest for the GPU wheel")
    if actual_digest.lower() != expected_digest.split(":", 1)[1].lower():
        raise RuntimeError("downloaded GPU wheel failed SHA-256 verification")
    return actual_digest


def install_gpu(
    *,
    version: str = __version__,
    repository: str = _REPOSITORY,
    dry_run: bool = False,
    user: bool = False,
    target: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the GPU build matching this interpreter and return asset details."""

    asset = select_gpu_asset(_release_assets(version, repository=repository), version=version)
    result = {
        "name": str(asset["name"]),
        "url": str(asset["browser_download_url"]),
        "digest": str(asset.get("digest") or ""),
        "version": str(version),
    }
    if dry_run:
        return result
    if user and target is not None:
        raise ValueError("user=True and target cannot be combined")

    with tempfile.TemporaryDirectory(prefix="ctboost-gpu-") as temporary_directory:
        wheel_path = Path(temporary_directory) / str(asset["name"])
        result["sha256"] = _download_verified(asset, wheel_path)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
        ]
        if user:
            command.append("--user")
        if target is not None:
            command.extend(["--target", str(target)])
        command.append(str(wheel_path))
        subprocess.run(command, check=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the CUDA-enabled CTBoost wheel matching this Python and platform."
    )
    parser.add_argument("--version", default=__version__, help="release version (default: installed version)")
    parser.add_argument("--repository", default=_REPOSITORY, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="show the selected verified release asset")
    parser.add_argument("--user", action="store_true", help="pass --user to pip")
    parser.add_argument("--target", help="pass --target DIRECTORY to pip")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = install_gpu(
            version=str(arguments.version),
            repository=str(arguments.repository),
            dry_run=bool(arguments.dry_run),
            user=bool(arguments.user),
            target=arguments.target,
        )
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print("ctboost-install-gpu: %s" % exc, file=sys.stderr)
        return 1
    action = "Selected" if arguments.dry_run else "Installed"
    print("%s %s" % (action, result["name"]))
    print(result["url"])
    if not arguments.dry_run:
        print("Restart Python, then verify with: python -c \"import ctboost; print(ctboost.build_info())\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
