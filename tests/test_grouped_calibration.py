"""Grouped-score equivalence and the frozen, non-training calibration harness."""

import re

import numpy as np
import pytest

from benchmarks.split_research import grouped_calibration as calibration


def test_frequency_case_represents_original_observation_count():
    case = next(case for case in calibration.PROTOCOL["cases"] if case.get("weights"))
    scores, bins, weights, missing_bin = calibration.generate_case(case, 0)
    assert weights.sum() == case["rows"]
    assert np.array_equal(weights, weights.astype(np.int64))
    assert set(scores) <= {0.0, 1.0}
    assert len(set(zip(bins, scores))) == len(scores)
    assert missing_bin == -1


def test_fractional_scale_diagnostics_share_same_observations():
    cases = [case for case in calibration.PROTOCOL["cases"] if "weight_scale" in case]
    generated = [calibration.generate_case(case, 7) for case in cases]
    for case, (scores, bins, weights, _) in zip(cases, generated):
        np.testing.assert_array_equal(scores, generated[0][0])
        np.testing.assert_array_equal(bins, generated[0][1])
        np.testing.assert_allclose(
            weights / case["weight_scale"], generated[0][2], rtol=2e-7
        )


def test_wilson_interval_includes_edge_rates():
    assert calibration.wilson_interval(0, 2000)[0] == 0.0
    assert calibration.wilson_interval(2000, 2000)[1] == 1.0
    low, high = calibration.wilson_interval(100, 2000)
    assert 0.03 < low < 0.05 < high < 0.07


@pytest.mark.parametrize("groups", [2, 8, 16, 64])
@pytest.mark.parametrize("missing_bin", [-1, 0, 255])
@pytest.mark.parametrize("weight_mode", ["unit", "fractional", "sparse", "extreme"])
def test_native_grouped_score_is_exactly_legacy(groups, missing_bin, weight_mode):
    import ctboost._core as core

    rng = np.random.default_rng(20260907)
    bins = rng.integers(0, 256, size=512, dtype=np.int64)
    bins[0] = 255  # Keep the declared Max missing bin inside the debug histogram.
    scores = rng.normal(size=bins.size).astype(np.float32)
    weights = np.ones(bins.size, dtype=np.float32)
    if weight_mode == "fractional":
        weights = rng.lognormal(size=bins.size).astype(np.float32)
    elif weight_mode == "sparse":
        weights[rng.random(bins.size) < 0.9] = 0.0
    elif weight_mode == "extreme":
        # A dominant first bin and tiny final bin exercise the reduced-covariance
        # fallback (the denominator approaches zero), including missing-as-bin.
        # Exact equality also catches FMA contraction of the stored expectation
        # before subtraction on Apple Silicon (the legacy path rounds it first).
        bins = np.array([0, 1, 255], dtype=np.int64)
        scores = np.array([-1.0, 1.0, 3.0], dtype=np.float32)
        weights = np.array([1e12, 1.0, 1.0], dtype=np.float32)
    result = core._debug_compute_grouped_pvalue(
        scores, bins, weights, groups, missing_bin
    )
    for key in ("p_value", "chi_square", "degrees_of_freedom"):
        assert result[key] == result["optimized_" + key]


@pytest.mark.parametrize(
    "rows,weight,missing_bin", [(0, 1.0, -1), (12, 0.0, -1), (12, 1.0, 0)]
)
def test_native_grouped_degenerate_inputs_keep_nonrejection(rows, weight, missing_bin):
    import ctboost._core as core

    result = core._debug_compute_grouped_pvalue(
        np.ones(rows, dtype=np.float32),
        np.zeros(rows, dtype=np.int64),
        np.full(rows, weight, dtype=np.float32),
        8,
        missing_bin,
    )
    assert result["p_value"] == result["optimized_p_value"] == 1.0
    assert result["degrees_of_freedom"] == result["optimized_degrees_of_freedom"] == 0


def test_native_calibration_smoke_reports_scope_without_training():
    report = calibration.run_diagnostics(repetitions=2)
    assert report["registered_repetitions_used"] is False
    assert len(report["scenarios"]) == 10
    assert all(row["legacy_optimized_exact"] for row in report["scenarios"])
    assert all(row["replicates"] == 2 for row in report["scenarios"])
    assert "no full-tree FWER" in report["scope"]


def test_ranked_profiler_identifies_stopping_and_selected_features(capfd):
    import ctboost

    rng = np.random.default_rng(20260907)
    X = rng.normal(size=(512, 64)).astype(np.float32)
    y = (2.0 * X[:, 0] + 0.7 * X[:, 1]).astype(np.float32)
    ctboost.train(
        X,
        {
            "objective": "RMSE",
            "feature_test": "grouped",
            "feature_test_adjustment": "bonferroni",
            "monotone_constraints": [-1] + [0] * 63,
            "max_bins": 256,
            "max_depth": 1,
            "alpha": 1.0,
            "verbose": True,
        },
        label=y,
        num_boost_round=1,
    )
    lines = [
        line for line in capfd.readouterr().err.splitlines() if "node_search" in line
    ]
    assert lines
    fields = dict(re.findall(r"(\w+)=([^ ]+)", lines[0]))
    assert int(fields["feature"]) == 1
    assert int(fields["minimum_p_feature"]) == 0
    assert float(fields["p_value"]) > float(fields["minimum_p_value"])
    assert float(fields["stopping_p_value"]) == pytest.approx(
        min(1.0, int(fields["tested_features"]) * float(fields["minimum_p_value"])),
        rel=2e-5,
    )
    assert int(fields["degrees_of_freedom"]) == 7
    # This is a coarse instrumentation check, not a timing regression threshold:
    # the 64 full-bin scans must appear in split_ms rather than feature_ms alone.
    assert float(fields["split_ms"]) > 0.0
