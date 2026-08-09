import hashlib
import io
from pathlib import Path

import pytest

from ctboost import gpu_install


def _asset(name, payload=b"wheel"):
    return {
        "name": name,
        "browser_download_url": "https://example.invalid/" + name,
        "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def test_select_gpu_asset_uses_build_tag_python_abi_and_platform():
    assets = [
        _asset("ctboost-0.1.52-cp312-cp312-win_amd64.whl"),
        _asset("ctboost-0.1.52-1gpu-cp311-cp311-win_amd64.whl"),
        _asset("ctboost-0.1.52-1gpu-cp312-cp312-win_amd64.whl"),
    ]

    selected = gpu_install.select_gpu_asset(
        assets,
        version="0.1.52",
        python_tag="cp312",
        platform_suffix="win_amd64.whl",
    )

    assert selected["name"] == "ctboost-0.1.52-1gpu-cp312-cp312-win_amd64.whl"


def test_select_gpu_asset_rejects_missing_or_ambiguous_assets():
    with pytest.raises(RuntimeError, match="found 0"):
        gpu_install.select_gpu_asset(
            [],
            version="0.1.52",
            python_tag="cp312",
            platform_suffix="win_amd64.whl",
        )


def test_download_verified_checks_digest_and_size(tmp_path, monkeypatch):
    payload = b"verified wheel bytes"
    asset = _asset("ctboost.whl", payload)
    monkeypatch.setattr(gpu_install.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    destination = tmp_path / "ctboost.whl"

    digest = gpu_install._download_verified(asset, destination)

    assert destination.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()


def test_download_verified_rejects_tampering(tmp_path, monkeypatch):
    asset = _asset("ctboost.whl", b"expected")
    monkeypatch.setattr(
        gpu_install.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(b"tampered"),
    )
    asset["size"] = len(b"tampered")

    with pytest.raises(RuntimeError, match="SHA-256"):
        gpu_install._download_verified(asset, tmp_path / "ctboost.whl")


def test_dry_run_never_downloads_or_invokes_pip(monkeypatch):
    asset = _asset("ctboost-0.1.52-1gpu-cp310-cp310-win_amd64.whl")
    monkeypatch.setattr(gpu_install, "_release_assets", lambda *args, **kwargs: [asset])
    monkeypatch.setattr(gpu_install, "_cpython_tag", lambda: "cp310")
    monkeypatch.setattr(gpu_install, "_platform_suffix", lambda: "win_amd64.whl")
    monkeypatch.setattr(
        gpu_install,
        "_download_verified",
        lambda *args, **kwargs: pytest.fail("dry-run downloaded a wheel"),
    )

    result = gpu_install.install_gpu(version="0.1.52", dry_run=True)

    assert result["name"] == asset["name"]
