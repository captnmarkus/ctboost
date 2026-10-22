"""Build only the source distribution and verify the new runtime helper member."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tarfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    import psutil
    from scikit_build_core.build import build_sdist
    psutil.Process().cpu_affinity([0])
    root = Path.cwd().resolve()
    source = root / "ctboost/_categorical_codes.py"
    project = root / "pyproject.toml"
    source_bytes, project_bytes = source.read_bytes(), project.read_bytes()
    output = root / ".tmp/improvements-20261016-followup/packaging/sdist"
    output.mkdir(parents=True, exist_ok=True)
    filename = build_sdist(str(output))
    artifact = output / filename
    with tarfile.open(artifact, "r:gz") as archive:
        members = archive.getmembers()
        matches = [entry for entry in members if entry.name.endswith("/ctboost/_categorical_codes.py")]
        assert len(matches) == 1 and matches[0].isfile()
        packaged = archive.extractfile(matches[0]).read()
        assert packaged == source_bytes, "Packaged helper bytes differ from checked source"
        projects = [entry for entry in members if entry.name.endswith("/pyproject.toml")]
        assert len(projects) == 1
        packaged_project = archive.extractfile(projects[0]).read()
        assert packaged_project == project_bytes
        assert b'"/ctboost/_categorical_codes.py"' in packaged_project
        names = sorted(entry.name for entry in members)
        assert not any("/.tmp/" in name or name.endswith(".pyd") for name in names)
    assert source.read_bytes() == source_bytes and project.read_bytes() == project_bytes
    receipt = {
        "status": "sdist_member_present_and_byte_exact",
        "method": "scikit_build_core.build.build_sdist, installed backend, no build isolation",
        "python": platform.python_version(),
        "scikit_build_core": importlib.metadata.version("scikit-build-core"),
        "cpu_affinity": psutil.Process().cpu_affinity(), "native_threads": 1,
        "native_build": False, "artifact": str(artifact),
        "artifact_sha256": sha256(artifact.read_bytes()), "artifact_bytes": artifact.stat().st_size,
        "archive_member_count": len(members), "member": matches[0].name,
        "member_bytes": len(packaged), "member_sha256": sha256(packaged),
        "working_source_sha256": sha256(source_bytes), "working_pyproject_sha256": sha256(project_bytes),
        "source_unchanged_during_build": True, "packaged_pyproject_exact": True,
        "scratch_and_native_binaries_absent": True,
        "verification_script_sha256": sha256(Path(__file__).read_bytes()),
    }
    destination = Path(__file__).parent
    (destination / "sdist_membership.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (destination / "sdist_members.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
