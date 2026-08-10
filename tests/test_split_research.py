import numpy as np
import pytest

from benchmarks.split_research import (
    binary_category_newton_scores,
    chi_square_survival,
    compare_newton_and_woe_ordering,
    cross_fitted_binary_category_scores,
    equal_weight_ordered_groups,
    global_bonferroni_stop,
    grouped_ordered_quadratic_test,
    nominal_quadratic_test,
    ordered_grouped_hybrid_test,
    ordered_linear_test,
    permutation_maxstat_test,
    smoothed_woe_scores,
    weighted_midranks,
)
from benchmarks.split_research.simulation import (
    DEFAULT_SCENARIOS,
    run_missing_policy_experiment,
    run_numeric_quantization_experiment,
    run_scenario_experiment,
)


def test_chi_square_survival_matches_closed_form_two_df_case():
    assert chi_square_survival(2.0, 2) == pytest.approx(np.exp(-1.0), rel=1e-13)
    assert chi_square_survival(0.0, 2) == 1.0


def test_nominal_reference_matches_current_native_quadratic_statistic():
    core = pytest.importorskip("ctboost._core")
    gradients = np.repeat(np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=np.float32), 64)
    bins = np.repeat(np.arange(4, dtype=np.int64), 64)

    native = core._debug_compute_pvalue(gradients, bins)
    reference = nominal_quadratic_test(gradients, bins)

    assert reference.degrees_of_freedom == native["degrees_of_freedom"]
    assert reference.statistic == pytest.approx(native["chi_square"], rel=1e-12)
    assert reference.p_value == pytest.approx(native["p_value"], rel=1e-12)


def test_weighted_midranks_tie_and_frequency_semantics():
    values = np.asarray([3.0, 1.0, 1.0, 2.0])
    weights = np.asarray([1.0, 2.0, 1.0, 2.0])

    ranks = weighted_midranks(values, weights)

    np.testing.assert_allclose(ranks, [5.5 / 6.0, 1.5 / 6.0, 1.5 / 6.0, 4.0 / 6.0])


def test_frequency_weights_match_explicit_replication_for_all_statistics():
    feature = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    scores = np.asarray([-1.0, -0.25, 0.0, 0.8, 1.4])
    weights = np.asarray([1, 3, 2, 4, 2])
    expanded_feature = np.repeat(feature, weights)
    expanded_scores = np.repeat(scores, weights)

    weighted_nominal = nominal_quadratic_test(scores, feature, weights)
    expanded_nominal = nominal_quadratic_test(expanded_scores, expanded_feature)
    weighted_ordered = ordered_linear_test(scores, feature, weights)
    expanded_ordered = ordered_linear_test(expanded_scores, expanded_feature)
    weighted_maxstat = permutation_maxstat_test(
        scores, feature, weights, n_permutations=99, random_state=91
    )
    expanded_maxstat = permutation_maxstat_test(
        expanded_scores, expanded_feature, n_permutations=99, random_state=91
    )

    assert weighted_nominal.statistic == pytest.approx(expanded_nominal.statistic)
    assert weighted_ordered.statistic == pytest.approx(expanded_ordered.statistic)
    assert weighted_maxstat.statistic == pytest.approx(expanded_maxstat.statistic)
    assert weighted_maxstat.p_value == expanded_maxstat.p_value


def test_missing_values_are_equivalent_to_featurewise_omission():
    feature = np.asarray([0.0, 1.0, np.nan, 2.0, 3.0, np.nan])
    scores = np.asarray([-1.0, -0.3, 50.0, 0.4, 1.2, -50.0])
    keep = np.isfinite(feature)

    nominal_with_missing = nominal_quadratic_test(
        scores, feature, missing_policy="ctree_omit"
    )
    ordered_with_missing = ordered_linear_test(scores, feature)
    maxstat_with_missing = permutation_maxstat_test(
        scores, feature, n_permutations=39, random_state=7
    )

    assert nominal_with_missing.statistic == pytest.approx(
        nominal_quadratic_test(
            scores[keep], feature[keep], missing_policy="ctree_omit"
        ).statistic
    )
    assert ordered_with_missing.statistic == pytest.approx(
        ordered_linear_test(scores[keep], feature[keep]).statistic
    )
    assert (
        maxstat_with_missing.p_value
        == permutation_maxstat_test(
            scores[keep], feature[keep], n_permutations=39, random_state=7
        ).p_value
    )


def test_nominal_production_missing_policy_uses_a_dedicated_level():
    feature = np.asarray([0.0, 0.0, 1.0, 1.0, np.nan, np.nan])
    scores = np.asarray([-1.0, -0.8, 0.0, 0.1, 1.0, 1.2])

    production = nominal_quadratic_test(scores, feature)
    ctree = nominal_quadratic_test(scores, feature, missing_policy="ctree_omit")

    assert production.details["missing_policy"] == "production_bin"
    assert production.details["active_levels"] == 3
    assert production.degrees_of_freedom == 2
    assert ctree.details["missing_policy"] == "ctree_omit"
    assert ctree.details["active_levels"] == 2
    assert ctree.degrees_of_freedom == 1
    assert production.statistic != pytest.approx(ctree.statistic)


def test_maxstat_trims_edges_and_uses_inclusive_plus_one_p_value():
    feature = np.arange(20, dtype=np.float64)
    scores = np.r_[np.zeros(10), np.ones(10)]
    result = permutation_maxstat_test(
        scores,
        feature,
        n_permutations=39,
        min_fraction=0.2,
        random_state=11,
    )

    assert result.details["minimum_count"] == 4
    assert result.details["eligible_cutpoints"] == 13
    assert 4 <= result.details["best_cut_position"] <= 16
    assert result.p_value >= 1.0 / 40.0
    assert result.p_value == (1.0 + result.details["exceedances"]) / 40.0


def test_maxstat_rejects_noninteger_case_weights():
    with pytest.raises(ValueError, match="integer frequency weights"):
        permutation_maxstat_test(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.5, 1.0],
            n_permutations=19,
        )


def test_global_bonferroni_stop_controls_feature_level_multiplicity():
    stopped = global_bonferroni_stop([0.02, 0.4, 0.9], alpha=0.05)
    split = global_bonferroni_stop([0.01, 0.4, 0.9], alpha=0.05)

    assert not stopped.should_split
    assert stopped.selected_index is None
    assert stopped.adjusted_p_value == pytest.approx(0.06)
    assert split.should_split
    assert split.selected_index == 0
    assert split.adjusted_p_value == pytest.approx(0.03)


def test_equal_weight_groups_are_contiguous_and_selection_only():
    feature = np.arange(255, dtype=np.float64)
    scores = np.sin(feature / 20.0)
    groups = equal_weight_ordered_groups(feature, n_groups=8)
    result = grouped_ordered_quadratic_test(scores, feature, n_groups=8)

    assert np.all(np.diff(groups) >= 0)
    assert np.unique(groups).shape[0] == 8
    assert result.details["source_levels"] == 255
    assert result.details["effective_groups"] == 8
    assert result.details["selection_only_grouping"]


def test_hybrid_uses_within_feature_bonferroni_minimum():
    rng = np.random.default_rng(41)
    feature = np.arange(120, dtype=np.float64)
    scores = np.square(feature - 60.0) + rng.normal(scale=10.0, size=120)
    ordered = ordered_linear_test(scores, feature)
    grouped = grouped_ordered_quadratic_test(scores, feature, n_groups=8)
    hybrid = ordered_grouped_hybrid_test(scores, feature, n_groups=8)

    assert hybrid.p_value == pytest.approx(
        min(1.0, 2.0 * min(ordered.p_value, grouped.p_value))
    )
    assert hybrid.details["within_feature_adjustment"] == "bonferroni_2"


def test_grouped_and_hybrid_preserve_a_missingness_only_signal():
    rng = np.random.default_rng(117)
    feature = rng.normal(size=240)
    missing = np.zeros(240, dtype=bool)
    missing[::3] = True
    feature[missing] = np.nan
    scores = missing.astype(np.float64) * 4.0 + rng.normal(scale=0.25, size=240)

    grouped = grouped_ordered_quadratic_test(scores, feature, n_groups=8)
    hybrid = ordered_grouped_hybrid_test(scores, feature, n_groups=8)
    omitted = grouped_ordered_quadratic_test(
        scores, feature, n_groups=8, missing_policy="ctree_omit"
    )

    assert grouped.details["missing_level_present"]
    assert grouped.p_value < 1e-10
    assert hybrid.p_value < 1e-10
    assert omitted.p_value > 0.01


def test_matched_binary_newton_and_woe_scores_have_the_same_order():
    categories = np.repeat(["a", "b", "c", "d"], [7, 19, 41, 83])
    target = np.concatenate(
        [
            np.asarray([0, 0, 0, 0, 0, 1, 1]),
            np.asarray([0] * 12 + [1] * 7),
            np.asarray([0] * 20 + [1] * 21),
            np.asarray([0] * 18 + [1] * 65),
        ]
    )
    comparison = compare_newton_and_woe_ordering(categories, target, l2=3.0)

    assert comparison.pairwise_agreement == 1.0
    assert comparison.newton_order == comparison.woe_order

    prior = float(np.mean(target))
    newton = binary_category_newton_scores(categories, target, prior=prior, l2=3.0)
    woe = smoothed_woe_scores(
        categories,
        target,
        prior=prior,
        smoothing=3.0 / (prior * (1.0 - prior)),
    )
    assert sorted(newton, key=newton.get) == sorted(woe, key=woe.get)


def test_cross_fitting_never_uses_a_rows_own_target():
    categories = np.asarray(["a", "b", "c", "d"] * 20, dtype=object)
    target = np.asarray(([0, 1, 0, 1] * 10) + ([1, 0, 1, 0] * 10), dtype=float)
    encoded, folds = cross_fitted_binary_category_scores(
        categories, target, n_splits=5, random_state=29, smoothing=5.0
    )
    changed_target = target.copy()
    changed_target[7] = 1.0 - changed_target[7]
    changed, changed_folds = cross_fitted_binary_category_scores(
        categories, changed_target, n_splits=5, random_state=29, smoothing=5.0
    )

    np.testing.assert_array_equal(folds, changed_folds)
    assert encoded[7] == changed[7]


def test_cross_fitting_maps_unseen_unique_categories_to_neutral_score():
    categories = np.asarray([f"id_{index}" for index in range(40)], dtype=object)
    target = np.asarray([0.0, 1.0] * 20)
    encoded, _ = cross_fitted_binary_category_scores(
        categories, target, n_splits=5, random_state=3
    )

    np.testing.assert_array_equal(encoded, np.zeros_like(encoded))


def test_repeated_null_simulations_stay_within_calibration_ranges():
    null_scenarios = [
        scenario for scenario in DEFAULT_SCENARIOS if scenario.family == "null"
    ]
    result = run_scenario_experiment(
        repetitions=80,
        n_observations=192,
        n_permutations=99,
        alpha=0.05,
        seed=7321,
        scenarios=null_scenarios,
    )

    for method in ("nominal_quadratic", "ordered_midrank", "permutation_maxstat"):
        rates = np.asarray([row["methods"][method]["rejection_rate"] for row in result])
        assert np.all((0.0 <= rates) & (rates <= 0.15))
        assert 0.01 <= float(np.mean(rates)) <= 0.10


def test_repeated_power_simulations_recover_expected_signal_shapes():
    power_scenarios = [
        scenario for scenario in DEFAULT_SCENARIOS if scenario.family == "power"
    ]
    result = run_scenario_experiment(
        repetitions=50,
        n_observations=192,
        n_permutations=99,
        alpha=0.05,
        effect_size=0.65,
        seed=9234,
        scenarios=power_scenarios,
    )
    by_name = {row["scenario"]["name"]: row["methods"] for row in result}

    assert by_name["power_smooth"]["ordered_midrank"]["rejection_rate"] >= 0.80
    assert by_name["power_abrupt"]["permutation_maxstat"]["rejection_rate"] >= 0.80
    assert by_name["power_u_shaped"]["nominal_quadratic"]["rejection_rate"] >= 0.70
    assert by_name["power_u_shaped"]["permutation_maxstat"]["rejection_rate"] >= 0.60
    assert by_name["power_u_shaped"]["ordered_midrank"]["rejection_rate"] <= 0.25


def test_numeric_quantization_matrix_covers_production_relevant_cardinalities():
    result = run_numeric_quantization_experiment(
        repetitions=40,
        n_observations=160,
        requested_bin_counts=(8, 32, 64, 255),
        n_permutations=79,
        effect_size=0.65,
        seed=5150,
    )

    assert result["requested_bin_counts"] == [8, 32, 64, 160]
    rows = {(row["relationship"], row["requested_bins"]): row for row in result["rows"]}
    assert rows[("null", 8)]["nominal_rejection_rate"] <= 0.15
    assert rows[("null", 160)]["nominal_rejection_rate"] <= 0.15
    assert rows[("smooth", 8)]["nominal_rejection_rate"] >= 0.75
    assert (
        rows[("smooth", 160)]["nominal_rejection_rate"]
        <= rows[("smooth", 8)]["nominal_rejection_rate"]
    )


def test_both_nominal_missing_policies_are_null_calibrated():
    result = run_missing_policy_experiment(
        repetitions=300,
        n_observations=192,
        cardinality=8,
        missing_rate=0.30,
        seed=8844,
    )

    for policy in ("production_bin", "ctree_omit"):
        assert 0.01 <= result["policies"][policy]["rejection_rate"] <= 0.10
