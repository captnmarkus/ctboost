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
    "TestResult",
    "binary_category_newton_scores",
    "chi_square_survival",
    "compare_newton_and_woe_ordering",
    "cross_fitted_binary_category_scores",
    "equal_frequency_bins",
    "equal_weight_ordered_groups",
    "global_bonferroni_stop",
    "grouped_ordered_quadratic_test",
    "nominal_quadratic_test",
    "ordered_grouped_hybrid_test",
    "ordered_linear_test",
    "permutation_maxstat_test",
    "smoothed_woe_scores",
    "weighted_midranks",
]
