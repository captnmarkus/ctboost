import itertools
import json

import numpy as np
import pytest

from benchmarks.split_research.multiclass import (
    best_numeric_split_gain,
    full_multiclass_quadratic_test,
    grouped_full_multiclass_quadratic_test,
    legacy_multiclass_quadratic_test,
    multiclass_partition_gain,
    permutation_calibrated_multiclass_test,
    select_legacy_structure_class,
    softmax_diagonal_hessians,
    softmax_score_matrix,
)
from benchmarks.split_research.multiclass_simulation import (
    BIN_PROFILES,
    SCENARIOS,
    _write_json_atomic,
    evaluate_bounded_gate,
    run_class_permutation_screen,
    run_main_screen,
    run_numeric_cut_screen,
    run_reference_screen,
)
from benchmarks.split_research.statistics import nominal_quadratic_test


def _balanced_scores(n_repeats=24):
    labels = np.tile(np.arange(3, dtype=np.int64), n_repeats)
    return softmax_score_matrix(labels, np.asarray([1.0 / 3.0] * 3))


def _passing_bounded_gate_fixture():
    gated_profiles = ("raw_2", "raw_8", "raw_32", "grouped8_from_raw255")
    main_rows = [
        {
            "scenario": scenario,
            "family": "null",
            "profile": profile,
            "full_rejection_rate": 0.05,
        }
        for scenario in ("null_balanced_k3", "null_long_tail_k5")
        for profile in gated_profiles
    ]
    main_rows.extend(
        {
            "scenario": scenario,
            "family": "power",
            "profile": "grouped8_from_raw255",
            "full_minus_legacy": 0.10,
        }
        for scenario in (
            "power_exact_tie_k4",
            "power_imbalanced_hidden_k3",
            "power_diffuse_k5",
        )
    )
    main_rows.append(
        {
            "scenario": "power_aligned_k3",
            "family": "power",
            "profile": "grouped8_from_raw255",
            "full_minus_legacy": -0.10,
        }
    )
    return {
        "main_screen": {"rows": main_rows},
        "class_permutation": {
            # Equal aggregate rates deliberately coexist with the paired
            # mismatch counter so the regression cannot fall back to the v1
            # aggregate-only implementation.
            "rows": [
                {"full_rejection_rate": 0.5},
                {"full_rejection_rate": 0.5},
            ],
            "maximum_paired_full_statistic_difference": 0.0,
            "maximum_paired_full_rejection_mismatch_count": 0,
        },
        "permutation_oracle": {
            "rows": [
                {
                    "method": "full",
                    "asymptotic_rejection_rate": 0.05,
                    "permutation_rejection_rate": 0.05,
                    "absolute_rejection_difference": 0.0,
                }
            ]
        },
        "numeric_cut": {
            "rows": [
                {
                    "scenario": "power_aligned_k3",
                    "legacy_boundary_hit_rate": 0.56,
                    "full_boundary_hit_rate": 0.56,
                }
            ],
            "maximum_class_permuted_gain_difference": 0.0,
            "maximum_class_permuted_threshold_difference": 0.25,
        },
    }


def test_softmax_reference_scores_sum_to_zero_and_hessians_match_diagonal_rule():
    labels = np.asarray([0, 2, 1, 2])
    probabilities = np.asarray([0.5, 0.3, 0.2])

    scores = softmax_score_matrix(labels, probabilities)
    hessians = softmax_diagonal_hessians(probabilities, labels.shape[0])

    np.testing.assert_allclose(np.sum(scores, axis=1), 0.0, atol=1e-15)
    np.testing.assert_allclose(hessians[0], probabilities * (1.0 - probabilities))


def test_legacy_exact_variance_tie_selects_lowest_class_index():
    # Exactly representable values avoid turning this tie-contract check into
    # a floating-point summation-order test.
    scores = np.tile(
        np.asarray([[-1.0, 0.0, 1.0], [0.0, 1.0, -1.0], [1.0, -1.0, 0.0]]),
        (24, 1),
    )
    selected, variances = select_legacy_structure_class(scores)

    assert selected == 0
    np.testing.assert_allclose(variances, np.repeat(variances[0], 3), atol=1e-15)


def test_dyadic_four_class_softmax_margins_are_an_exact_native_style_tie():
    rng = np.random.default_rng(901)
    labels = np.repeat(np.arange(4, dtype=np.int64), 120)
    rng.shuffle(labels)
    scores = softmax_score_matrix(labels, np.repeat(0.25, 4))

    selected, variances = select_legacy_structure_class(scores)

    assert selected == 0
    np.testing.assert_array_equal(variances, np.repeat(variances[0], 4))


def test_binary_rank_one_full_statistic_matches_scalar_legacy_reference():
    labels = np.asarray([0, 1, 0, 1, 0, 1, 1, 1, 0, 0] * 12)
    feature = np.tile(np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 3, 0]), 12)
    scores = softmax_score_matrix(labels, np.asarray([0.5, 0.5]))

    full = full_multiclass_quadratic_test(scores, feature)
    legacy = legacy_multiclass_quadratic_test(scores, feature, epsilon=1e-12)

    assert full.response_rank == 1
    assert full.degrees_of_freedom == legacy.degrees_of_freedom == 3
    assert full.statistic == pytest.approx(legacy.statistic, rel=1e-10, abs=1e-11)
    assert full.p_value == pytest.approx(legacy.p_value, rel=1e-10, abs=1e-11)


@pytest.mark.parametrize("n_bins", [2, 8, 32, 255])
def test_o_b_legacy_evaluation_matches_existing_dense_reference(n_bins):
    rng = np.random.default_rng(4300 + n_bins)
    scores = rng.normal(size=(480, 3))
    feature = np.arange(480) * n_bins // 480
    weights = rng.integers(1, 4, size=480).astype(np.float64)

    linear_time = legacy_multiclass_quadratic_test(
        scores, feature, weights, structure_class=1
    )
    dense = nominal_quadratic_test(scores[:, 1], feature, weights)

    assert linear_time.degrees_of_freedom == dense.degrees_of_freedom
    assert linear_time.statistic == pytest.approx(dense.statistic, rel=1e-11, abs=1e-11)
    assert linear_time.p_value == pytest.approx(dense.p_value, rel=1e-11, abs=1e-13)


def test_o_b_legacy_reference_uses_native_bin_id_order_after_row_shuffle():
    core = pytest.importorskip("ctboost._core")
    rng = np.random.default_rng(771)
    bins = np.tile(np.arange(8, dtype=np.int64), 40)
    gradients = np.linspace(-2e-4, 2e-4, bins.shape[0], dtype=np.float32)
    order = rng.permutation(bins.shape[0])
    bins = bins[order]
    gradients = gradients[order]
    scores = np.column_stack(
        (gradients.astype(np.float64), -gradients.astype(np.float64))
    )

    native = core._debug_compute_pvalue(gradients, bins)
    reference = legacy_multiclass_quadratic_test(
        scores, bins, structure_class=0
    )

    assert reference.degrees_of_freedom == native["degrees_of_freedom"]
    assert reference.statistic == pytest.approx(native["chi_square"], rel=1e-10)
    assert reference.p_value == pytest.approx(native["p_value"], rel=1e-10)


def test_reference_rejects_invalid_epsilon_and_infinite_cut_features():
    scores = _balanced_scores(2)
    feature = np.arange(scores.shape[0], dtype=np.float64)
    hessians = np.full_like(scores, 0.2)

    with pytest.raises(ValueError, match="epsilon"):
        legacy_multiclass_quadratic_test(scores, feature, epsilon=np.nan)
    feature[0] = np.inf
    with pytest.raises(ValueError, match="ordered_feature"):
        best_numeric_split_gain(scores, hessians, feature)


def test_full_statistic_is_invariant_to_every_three_class_label_permutation():
    rng = np.random.default_rng(71)
    labels = rng.integers(0, 3, size=180)
    feature = rng.integers(0, 8, size=180)
    scores = softmax_score_matrix(labels, np.asarray([0.45, 0.35, 0.20]))
    baseline = full_multiclass_quadratic_test(scores, feature)

    for permutation in itertools.permutations(range(3)):
        permuted = full_multiclass_quadratic_test(scores[:, permutation], feature)
        assert permuted.response_rank == baseline.response_rank
        assert permuted.degrees_of_freedom == baseline.degrees_of_freedom
        assert permuted.statistic == pytest.approx(baseline.statistic, abs=1e-12)
        assert permuted.p_value == pytest.approx(baseline.p_value, abs=1e-13)


def test_full_statistic_is_invariant_to_orthonormal_score_contrasts():
    rng = np.random.default_rng(19)
    labels = rng.integers(0, 4, size=240)
    feature = rng.integers(0, 6, size=240)
    scores = softmax_score_matrix(labels, np.asarray([0.4, 0.3, 0.2, 0.1]))
    spanning = np.vstack((np.eye(3), -np.ones(3)))
    contrast, _ = np.linalg.qr(spanning)

    full = full_multiclass_quadratic_test(scores, feature)
    reduced = full_multiclass_quadratic_test(scores @ contrast, feature)

    assert full.response_rank == reduced.response_rank == 3
    assert full.degrees_of_freedom == reduced.degrees_of_freedom
    assert full.statistic == pytest.approx(reduced.statistic, abs=1e-11)
    assert full.p_value == pytest.approx(reduced.p_value, abs=1e-13)


def test_integer_frequency_weights_match_explicit_row_replication():
    labels = np.asarray([0, 1, 2, 0, 2, 1])
    feature = np.asarray([0, 0, 1, 2, 2, 3])
    weights = np.asarray([1, 3, 2, 4, 1, 2])
    scores = softmax_score_matrix(labels, np.asarray([0.4, 0.35, 0.25]))

    weighted = full_multiclass_quadratic_test(scores, feature, weights)
    expanded = full_multiclass_quadratic_test(
        np.repeat(scores, weights, axis=0), np.repeat(feature, weights)
    )

    assert weighted.response_rank == expanded.response_rank
    assert weighted.degrees_of_freedom == expanded.degrees_of_freedom
    assert weighted.statistic == pytest.approx(expanded.statistic, abs=1e-12)
    assert weighted.p_value == pytest.approx(expanded.p_value, abs=1e-13)


def test_closed_form_matches_explicit_kronecker_moore_penrose_quadratic():
    rng = np.random.default_rng(811)
    raw = rng.normal(size=(48, 3))
    scores = raw - np.mean(raw, axis=1, keepdims=True)
    feature = np.repeat(np.arange(4), 12)
    weights = rng.integers(1, 4, size=48).astype(np.float64)

    result = full_multiclass_quadratic_test(scores, feature, weights)
    weight_sum = float(np.sum(weights))
    mean = np.sum(weights[:, None] * scores, axis=0) / weight_sum
    centered = scores - mean
    response_covariance = (centered * weights[:, None]).T @ centered / weight_sum
    bin_weights = np.bincount(feature, weights=weights)
    bin_sums = np.zeros((4, 3))
    np.add.at(bin_sums, feature, weights[:, None] * scores)
    differences = bin_sums - bin_weights[:, None] * mean
    bin_covariance = (
        weight_sum * np.diag(bin_weights) - np.outer(bin_weights, bin_weights)
    ) / (weight_sum - 1.0)
    full_covariance = np.kron(bin_covariance, response_covariance)
    explicit = float(
        differences.reshape(-1)
        @ np.linalg.pinv(full_covariance, hermitian=True)
        @ differences.reshape(-1)
    )

    assert result.statistic == pytest.approx(explicit, rel=1e-11, abs=1e-11)
    assert result.degrees_of_freedom == int(np.linalg.matrix_rank(full_covariance))


def test_zero_weight_rows_and_bin_relabeling_do_not_change_full_statistic():
    scores = _balanced_scores(20)
    feature = np.tile(np.arange(6), 10)
    baseline = full_multiclass_quadratic_test(scores, feature)
    extended_scores = np.vstack((scores, np.asarray([[100.0, -50.0, -50.0]])))
    extended_feature = np.r_[feature, 999]
    weights = np.r_[np.ones(scores.shape[0]), 0.0]
    extended = full_multiclass_quadratic_test(
        extended_scores, extended_feature, weights
    )
    relabeled = full_multiclass_quadratic_test(
        scores, np.asarray([101, 7, 91, 4, 13, 55])[feature]
    )

    assert extended.active_bins == baseline.active_bins
    assert extended.statistic == pytest.approx(baseline.statistic, abs=1e-12)
    assert relabeled.statistic == pytest.approx(baseline.statistic, abs=1e-12)


def test_constant_and_rank_deficient_scores_have_explicit_rank_and_df():
    feature = np.tile(np.arange(3), 20)
    constant = np.ones((feature.shape[0], 4))
    constant_result = full_multiclass_quadratic_test(constant, feature)
    coordinate = np.linspace(-1.0, 1.0, feature.shape[0])
    rank_one = np.column_stack((coordinate, -coordinate, np.zeros_like(coordinate)))
    rank_one_result = full_multiclass_quadratic_test(rank_one, feature)

    assert constant_result.response_rank == 0
    assert constant_result.degrees_of_freedom == 0
    assert constant_result.p_value == 1.0
    assert rank_one_result.response_rank == 1
    assert rank_one_result.degrees_of_freedom == 2


def test_positive_weight_missing_values_are_a_dedicated_selection_bin():
    scores = _balanced_scores(20)
    feature = np.tile(np.asarray([0.0, 1.0, np.nan]), 20)

    production = full_multiclass_quadratic_test(scores, feature)
    omitted = full_multiclass_quadratic_test(
        scores, feature, missing_policy="ctree_omit"
    )
    grouped = grouped_full_multiclass_quadratic_test(scores, feature, n_groups=2)

    assert production.active_bins == 3
    assert omitted.active_bins == 2
    assert production.degrees_of_freedom == 2 * production.response_rank
    assert grouped.active_bins == 3


def test_permutation_oracle_uses_whole_score_rows_and_inclusive_plus_one():
    rng = np.random.default_rng(17)
    labels = rng.integers(0, 3, size=90)
    feature = rng.integers(0, 4, size=90)
    scores = softmax_score_matrix(labels, np.asarray([0.4, 0.35, 0.25]))

    result = permutation_calibrated_multiclass_test(
        scores, feature, n_permutations=39, random_state=23
    )

    assert result.permutation_p_value == (1.0 + result.exceedances) / 40.0
    assert result.permutation_p_value >= 1.0 / 40.0
    assert result.response_rank == 2


@pytest.mark.parametrize("method", ["legacy", "full"])
def test_optimized_permutation_oracle_matches_direct_recomputation(method):
    rng = np.random.default_rng(191)
    labels = rng.integers(0, 3, size=72)
    feature = rng.integers(0, 6, size=72)
    scores = softmax_score_matrix(labels, np.asarray([0.45, 0.35, 0.20]))
    n_permutations = 19
    seed = 818
    optimized = permutation_calibrated_multiclass_test(
        scores,
        feature,
        method=method,
        n_permutations=n_permutations,
        random_state=seed,
    )
    if method == "legacy":
        structure_class, _ = select_legacy_structure_class(scores)
        observed = legacy_multiclass_quadratic_test(
            scores, feature, structure_class=structure_class
        )

        def evaluate(permuted_scores):
            return legacy_multiclass_quadratic_test(
                permuted_scores, feature, structure_class=structure_class
            )

    else:
        observed = full_multiclass_quadratic_test(scores, feature)

        def evaluate(permuted_scores):
            return full_multiclass_quadratic_test(permuted_scores, feature)

    direct_rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = evaluate(scores[direct_rng.permutation(scores.shape[0])])
        exceedances += int(permuted.statistic >= observed.statistic - 1e-12)

    assert optimized.statistic == pytest.approx(observed.statistic, abs=1e-12)
    assert optimized.exceedances == exceedances
    assert optimized.permutation_p_value == (1.0 + exceedances) / (
        n_permutations + 1.0
    )


def test_full_partition_gain_equals_sum_of_scalar_class_gains_and_is_permutation_invariant():
    rng = np.random.default_rng(29)
    raw = rng.normal(size=(80, 4))
    scores = raw - np.mean(raw, axis=1, keepdims=True)
    hessians = rng.uniform(0.05, 0.3, size=scores.shape)
    left = np.arange(scores.shape[0]) < 37

    full = multiclass_partition_gain(scores, hessians, left, lambda_l2=0.7)
    scalar_sum = sum(
        multiclass_partition_gain(
            scores, hessians, left, lambda_l2=0.7, class_index=class_index
        )
        for class_index in range(scores.shape[1])
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = multiclass_partition_gain(
        scores[:, permutation], hessians[:, permutation], left, lambda_l2=0.7
    )

    assert full == pytest.approx(scalar_sum, abs=1e-12)
    assert permuted == pytest.approx(full, abs=1e-12)


def test_best_full_numeric_gain_recovers_boundary_and_is_class_permutation_invariant():
    feature = np.linspace(0.0, 1.0, 200, endpoint=False)
    left = feature < 0.5
    scores = np.zeros((feature.shape[0], 3))
    scores[left] = np.asarray([-0.6, 0.4, 0.2])
    scores[~left] = np.asarray([0.6, -0.4, -0.2])
    hessians = np.full_like(scores, 0.2)

    split = best_numeric_split_gain(scores, hessians, feature, lambda_l2=1.0)
    permutation = np.asarray([1, 2, 0])
    permuted = best_numeric_split_gain(
        scores[:, permutation], hessians[:, permutation], feature, lambda_l2=1.0
    )

    assert split.threshold == pytest.approx(0.4975)
    assert split.gain > 0.0
    assert permuted.threshold == split.threshold
    assert permuted.gain == pytest.approx(split.gain, abs=1e-12)


def test_best_numeric_gain_evaluates_missing_values_on_both_sides():
    feature = np.asarray([0.0, 0.1, 0.2, 0.8, 0.9, 1.0, np.nan, np.nan])
    scores = np.asarray(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    hessians = np.full_like(scores, 0.25)

    split = best_numeric_split_gain(scores, hessians, feature, lambda_l2=1.0)

    assert split.missing_go_left
    assert split.threshold == pytest.approx(0.5)


def test_full_numeric_gain_tie_break_is_class_order_invariant_on_a_frozen_seed():
    # This bounded v1 prefix produced a one-bin threshold flip under reversed
    # class columns even though the maximum gain difference was only roundoff.
    result = run_numeric_cut_screen(
        repetitions=20,
        n_observations=480,
        seed=20261120,
    )

    assert result["maximum_class_permuted_gain_difference"] <= 1e-10
    assert result["maximum_class_permuted_threshold_difference"] == 0.0
    assert result["full_numeric_gain_tie_relative_tolerance"] > 0.0
    assert (
        result["full_numeric_gain_tie_break"]
        == "lowest_threshold_then_missing_right"
    )


def test_class_permutation_screen_records_paired_rejection_mismatches():
    result = run_class_permutation_screen(
        repetitions=2,
        n_observations=480,
        seed=20261020,
    )

    assert result["maximum_paired_full_rejection_mismatch_count"] == 0
    assert result["total_paired_full_rejection_mismatch_count"] == 0
    assert all(
        row["paired_full_rejection_mismatch_count"] == 0
        for row in result["rows"]
    )


def test_v2_gate_does_not_add_threshold_equality_to_frozen_gain_gate():
    result = _passing_bounded_gate_fixture()

    decision = evaluate_bounded_gate(result)

    assert decision["checks"]["numeric_cut_class_permutation_invariance"]
    assert decision["statistical_screen_passed"]
    assert (
        decision["diagnostics_not_gated"][
            "maximum_class_permuted_threshold_difference"
        ]
        == 0.25
    )


def test_v2_gate_rejects_paired_mismatches_hidden_by_equal_aggregate_rates():
    result = _passing_bounded_gate_fixture()
    result["class_permutation"][
        "maximum_paired_full_rejection_mismatch_count"
    ] = 1

    decision = evaluate_bounded_gate(result)

    assert not decision["checks"]["class_permutation_invariance"]
    assert not decision["statistical_screen_passed"]


def test_v2_gate_keeps_both_robust_aligned_guards_unchanged():
    power_result = _passing_bounded_gate_fixture()
    power_result["main_screen"]["rows"][-1]["full_minus_legacy"] = -0.1000001
    power_decision = evaluate_bounded_gate(power_result)

    cut_result = _passing_bounded_gate_fixture()
    cut_result["numeric_cut"]["rows"][0]["full_boundary_hit_rate"] = 0.55
    cut_decision = evaluate_bounded_gate(cut_result)

    assert not power_decision["checks"]["aligned_grouped8_power_guard"]
    assert not cut_decision["checks"]["aligned_numeric_cut_guard"]


def test_bounded_screen_smoke_is_deterministic_and_covers_null_and_hidden_signal():
    selected_scenarios = (SCENARIOS[0], SCENARIOS[2])
    selected_profiles = (BIN_PROFILES[0], BIN_PROFILES[1])
    first = run_main_screen(
        repetitions=4,
        n_observations=480,
        seed=20260820,
        scenarios=selected_scenarios,
        bin_profiles=selected_profiles,
    )
    second = run_main_screen(
        repetitions=4,
        n_observations=480,
        seed=20260820,
        scenarios=selected_scenarios,
        bin_profiles=selected_profiles,
    )

    def stable_rows(result):
        return [
            {key: value for key, value in row.items() if key != "timing_rows"}
            for row in result["rows"]
        ]

    assert stable_rows(first) == stable_rows(second)
    assert {row["family"] for row in first["rows"]} == {"null", "power"}
    assert {row["profile"] for row in first["rows"]} == {"raw_2", "raw_8"}


def test_reference_screen_rejects_invalid_counts_before_running_any_cells():
    with pytest.raises(ValueError, match="counts"):
        run_reference_screen(repetitions=0)
    with pytest.raises(ValueError, match="alpha"):
        run_reference_screen(alpha=1.0)
    with pytest.raises(ValueError, match="multiple of 120"):
        run_reference_screen(n_observations=121)


def test_atomic_ledger_writer_replaces_complete_json_and_rejects_nan(tmp_path):
    output = tmp_path / "ledger.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    _write_json_atomic(output, {"schema_version": 1, "complete": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "complete": True,
    }
    assert not output.with_name(output.name + ".tmp").exists()
    with pytest.raises(ValueError, match="Out of range float values"):
        _write_json_atomic(output, {"bad": float("nan")})
    assert json.loads(output.read_text(encoding="utf-8"))["complete"]
