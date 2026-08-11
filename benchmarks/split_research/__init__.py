"""Reference experiments for possible CTBoost split-statistic extensions.

Nothing in this package is wired into the production learner. The explicit
separation makes it possible to reject weak ideas without changing defaults.
"""

from .categorical import (
    CategoryOrderingComparison,
    binary_category_newton_scores,
    compare_newton_and_woe_ordering,
    cross_fitted_binary_category_scores,
    smoothed_woe_scores,
)
from .multiclass import (
    MulticlassTestResult,
    NumericSplitResult,
    PermutationCalibrationResult,
    best_legacy_numeric_split_gain,
    best_numeric_split_gain,
    full_multiclass_quadratic_test,
    grouped_full_multiclass_quadratic_test,
    grouped_legacy_multiclass_quadratic_test,
    grouped_ordered_selection_bins,
    legacy_multiclass_quadratic_test,
    multiclass_partition_gain,
    permutation_calibrated_multiclass_test,
    select_legacy_structure_class,
    softmax_diagonal_hessians,
    softmax_score_matrix,
)
from .statistics import (
    BonferroniDecision,
    TestResult,
    chi_square_survival,
    equal_frequency_bins,
    equal_weight_ordered_groups,
    global_bonferroni_stop,
    grouped_ordered_quadratic_test,
    nominal_quadratic_test,
    ordered_grouped_hybrid_test,
    ordered_linear_test,
    permutation_maxstat_test,
    weighted_midranks,
)

__all__ = [
    "BonferroniDecision",
    "CategoryOrderingComparison",
    "MulticlassTestResult",
    "NumericSplitResult",
    "PermutationCalibrationResult",
    "TestResult",
    "best_legacy_numeric_split_gain",
    "best_numeric_split_gain",
    "binary_category_newton_scores",
    "chi_square_survival",
    "compare_newton_and_woe_ordering",
    "cross_fitted_binary_category_scores",
    "equal_frequency_bins",
    "equal_weight_ordered_groups",
    "full_multiclass_quadratic_test",
    "global_bonferroni_stop",
    "grouped_ordered_quadratic_test",
    "grouped_full_multiclass_quadratic_test",
    "grouped_legacy_multiclass_quadratic_test",
    "grouped_ordered_selection_bins",
    "legacy_multiclass_quadratic_test",
    "multiclass_partition_gain",
    "nominal_quadratic_test",
    "ordered_grouped_hybrid_test",
    "ordered_linear_test",
    "permutation_maxstat_test",
    "permutation_calibrated_multiclass_test",
    "select_legacy_structure_class",
    "smoothed_woe_scores",
    "softmax_diagonal_hessians",
    "softmax_score_matrix",
    "weighted_midranks",
]
