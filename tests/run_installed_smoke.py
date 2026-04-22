import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a wheel smoke test from a temp copy of tests/."
    )
    parser.add_argument(
        "test_file",
        help="Path to the smoke-test file inside the repository checkout.",
    )
    return parser.parse_args()


def _clean_pythonpath(project_root: pathlib.Path) -> None:
    pythonpath = os.environ.get("PYTHONPATH")
    if not pythonpath:
        return

    cleaned = []
    for entry in pythonpath.split(os.pathsep):
        if not entry:
            cleaned.append(entry)
            continue

        resolved = pathlib.Path(entry).resolve()
        if resolved == project_root or project_root in resolved.parents:
            continue
        cleaned.append(entry)

    if cleaned:
        os.environ["PYTHONPATH"] = os.pathsep.join(cleaned)
    else:
        os.environ.pop("PYTHONPATH", None)


def main() -> int:
    args = _parse_args()
    project_root = pathlib.Path(__file__).resolve().parents[1]
    requested = pathlib.Path(args.test_file).resolve()
    tests_root = project_root / "tests"

    try:
        relative_test = requested.relative_to(tests_root)
    except ValueError as exc:
        raise SystemExit(f"expected a test under {tests_root}, got {requested}") from exc

    _clean_pythonpath(project_root)

    with tempfile.TemporaryDirectory(prefix="ctboost-installed-tests-") as tmp_dir:
        tmp_root = pathlib.Path(tmp_dir)
        copied_tests = tmp_root / "tests"
        shutil.copytree(tests_root, copied_tests)
        copied_test = copied_tests / relative_test

        import ctboost

        package_path = pathlib.Path(ctboost.__file__).resolve()
        if package_path == project_root or project_root in package_path.parents:
            raise SystemExit(
                f"expected installed ctboost package outside {project_root}, got {package_path}"
            )

        print(f"ctboost imported from: {package_path}")

        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-m", "pytest", str(copied_test), "-q"],
            cwd=tmp_root,
            check=False,
        )
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
