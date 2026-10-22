"""Verify current source distribution includes all CTR changes byte-for-byte."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tarfile

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


def main():
    import psutil
    from scikit_build_core.build import build_sdist
    psutil.Process().cpu_affinity([0])
    root = Path(r"C:/apps/ctboost")
    work = root / ".tmp/improvements-20261016/ctr-precision"
    sources = set(json.loads((work / "baseline-source-hashes.json").read_text())) | {
        "tests/test_ctr_precision.py","tests/test_feature_pipeline_ctr_inference.py",
        "tests/test_distributed_training_raw_pipeline.py",
        "tests/conftest.py",
        "ctboost/_categorical_codes.py","pyproject.toml"}
    contents = {relative:(root / relative).read_bytes() for relative in sources}
    output = work / "sdist"
    output.mkdir(exist_ok=True)
    artifact = output / build_sdist(str(output))
    matched = {}
    with tarfile.open(artifact,"r:gz") as archive:
        members = archive.getmembers()
        for relative, expected in contents.items():
            entries = [entry for entry in members if entry.name.endswith("/"+relative)]
            assert len(entries) == 1 and entries[0].isfile(), relative
            actual = archive.extractfile(entries[0]).read()
            assert actual == expected, relative
            matched[relative] = {"member":entries[0].name,"sha256":hashlib.sha256(actual).hexdigest(),"bytes":len(actual)}
        names = sorted(entry.name for entry in members)
        assert not any("/.tmp/" in name or name.endswith(".pyd") for name in names)
    assert all((root / relative).read_bytes() == content for relative,content in contents.items())
    receipt = {"status":"all_changed_sources_and_new_test_present_byte_exact",
        "method":"scikit_build_core.build.build_sdist, existing backend; no native or wheel build",
        "artifact":str(artifact),"artifact_sha256":hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "artifact_bytes":artifact.stat().st_size,"archive_member_count":len(members),"checked_members":matched,
        "python":platform.python_version(),"scikit_build_core":importlib.metadata.version("scikit-build-core"),
        "source_unchanged_during_build":True,"cpu_affinity":psutil.Process().cpu_affinity(),
        "verification_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (work / "sdist-membership.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (work / "sdist-members.txt").write_text("\n".join(names)+"\n")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == "__main__":
    main()
