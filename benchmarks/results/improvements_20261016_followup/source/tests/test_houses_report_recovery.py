"""A reporting-contract fixture; no real data or models."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("houses_report_recovery",
    Path(__file__).resolve().parents[1] / "benchmarks/fair_accuracy_houses_report_recovery.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_only_missing_descriptive_field_is_added_without_mutation():
    plan = {"cases": [{"dataset": "synthetic"}], "seed": 47}
    fixed = recovery.report_plan(plan, {"limitations": "Already frozen caveat"})
    assert fixed == {**plan, "limitations": "Already frozen caveat"}
    assert "limitations" not in plan
    assert fixed["cases"] is plan["cases"]
    with pytest.raises(ValueError, match="Only the missing"):
        recovery.report_plan(fixed, {"limitations": "replacement"})
