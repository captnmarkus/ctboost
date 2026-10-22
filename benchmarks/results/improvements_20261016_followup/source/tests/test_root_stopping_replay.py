"""Check diagnostic classification without fitting or loading quality data."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/root_stopping_replay.py"
spec = importlib.util.spec_from_file_location("root_stopping_replay", SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def root(p="0.2", feature="0", tested="3", valid="0", left="0", right="0"):
    return {"search": {"stopping_p_value": p, "feature": feature, "tested_features": tested,
                       "split_valid": valid, "left_rows": left, "right_rows": right}}


@pytest.mark.parametrize("record,leaf,expected", [
    (root("0.2"), True, "test_rejected"),
    (root("1"), True, "test_rejected_p_rounded_one"),
    (root("0.001"), True, "test_passed_no_feasible_positive_gain"),
    (root("0.05"), True, "rounded_p_boundary_unresolved"),
    (root(feature="-1", tested="0"), True, "no_eligible_feature"),
    ({"search": None}, True, "pre_search_structural_limit"),
    (root(valid="1", left="0", right="10"), True, "empty_partition_after_search"),
    (root(valid="1", left="5", right="5"), False, "split_created"),
])
def test_distinguishes_only_log_observable_causes(record, leaf, expected):
    assert replay.classify_root(record, leaf, .05) == expected


def test_profile_cannot_override_contradictory_topology():
    with pytest.raises(ValueError):
        replay.classify_root(root(valid="1", left="5", right="5"), True, .05)


def test_parser_joins_root_to_following_tree_boundary(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("[ctboost] node_hist depth=0 rows=10\n"
                   "[ctboost] node_search depth=0 rows=10 feature=0 stopping_p_value=0.3\n"
                   "[ctboost] node_hist depth=1 rows=5\n"
                   "[ctboost] tree_build iteration=1/2 class=1/1 elapsed_ms=0.1\n"
                   "[ctboost] node_hist depth=0 rows=10\n"
                   "[ctboost] tree_build iteration=2/2 class=1/1 elapsed_ms=0.1\n")
    roots = replay.parse_roots(log)
    assert [r["iteration"] for r in roots] == [1, 2]
    assert roots[0]["search"]["stopping_p_value"] == "0.3"
    assert roots[1]["search"] is None


def test_incomplete_log_is_rejected(tmp_path):
    log = tmp_path / "native.log"
    log.write_text("[ctboost] node_hist depth=0 rows=10\n")
    with pytest.raises(ValueError):
        replay.parse_roots(log)


def test_development_access_is_rejected_before_file_lookup():
    with pytest.raises(ValueError, match="TRAIN and STOP"):
        replay.training_role_only({}, "development")
