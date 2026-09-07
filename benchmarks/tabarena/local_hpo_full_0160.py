"""Run frozen full TabArena-v0.1 local parents, preserving every repeat/fold.

Reuses the existing 0.1.60 controller's process ownership, memory admission,
pause and no-retry behavior. This module never resumes a started failed parent.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_LEGACY_PATH = ROOT / "benchmarks/tabarena/local_hpo_0160.py"
_spec = importlib.util.spec_from_file_location(
    "_ctboost_local_hpo_0160_delegate", _LEGACY_PATH
)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
base = legacy.base


class Controller(legacy.Controller):
    def _freeze_execution(self):
        binding = {
            "schema_version": 1,
            "controller": "ctboost_local_hpo_full_0160",
            "plan_sha256": self.plan_hash,
            "plan_path": str(self.plan_path),
            "controller_source_sha256": self.worker.source_hash(__file__),
            "delegated_controller_source_sha256": self.worker.source_hash(_LEGACY_PATH),
            "base_controller_source_sha256": self.worker.source_hash(legacy._BASE_PATH),
            "worker_source_sha256": self.worker.source_hash(self.worker_script),
            "python": self.python,
            "python_sha256": base.file_hash(self.python),
            "base_python": str(Path(sys._base_executable).resolve()),
            "base_python_sha256": base.file_hash(sys._base_executable),
            "psutil_version": base.psutil.__version__,
            "wheel_path": str(self.wheel),
            "wheel_sha256": base.file_hash(self.wheel),
            "runtime_sha256": self.runtime_hash,
            "registration_path": str(self.registration),
            "registration": self.registration_verified,
            "allocation_policy": self.policy,
            "poll_seconds": self.poll_seconds,
        }
        path = self.output / "controller_execution.json"
        if path.exists():
            existing = base.read_json(path)
            if {
                key: value for key, value in existing.items() if key != "created_at"
            } != binding:
                raise ValueError(
                    "Frozen full local controller execution or provenance changed"
                )
        else:
            base.write_json(path, {**binding, "created_at": base.now()})

    def _worker_command(self, parent, slot, claim, permit, token):
        command = super()._worker_command(parent, slot, claim, permit, token)
        command[2] = str(Path(__file__).resolve())
        return command

    def _validate_completed(self, parent, manifest, raw):
        validated = super()._validate_completed(parent, manifest, raw)
        if (
            not isinstance(manifest.get("outer_split"), dict)
            or validated.get("outer_split") != manifest["outer_split"]
        ):
            raise ValueError(
                "Raw outer-split receipt differs from the pre-fit manifest"
            )
        return validated


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return base._worker_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "plan", "wheel", "registration"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    import ctboost

    if (
        not sys.flags.isolated
        or (ROOT / "ctboost") in Path(ctboost.__file__).resolve().parents
    ):
        raise RuntimeError("Use python -I with the installed public 0.1.60 wheel")
    # Keep the existing worker's import protocol: bind the installed package
    # before exposing benchmark sources to its versioned validation helpers.
    sys.path.insert(0, str(ROOT))
    from benchmarks.tabarena import hpo_full_0160 as worker

    with base.controller_lock(args.output):
        controller = Controller(
            output=args.output,
            plan_path=args.plan,
            wheel=args.wheel,
            registration=args.registration,
            worker=worker,
            max_workers=args.max_workers,
            poll_seconds=args.poll_seconds,
        )
        return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
