#pragma once

#include "ctboost/tree.hpp"

namespace ctboost::booster_detail {

// Fit only the values of an already selected shared multiclass tree. The
// objective is weighted softmax loss plus lambda_l2 / 2 * ||leaf||^2, in the
// sum-zero gauge. Both scalar-per-class and vector-leaf layouts use this solver.
// Each constrained Newton proposal is checked against the actual leaf loss.
void FitFullSoftmaxTreeLeaves(
    std::vector<Tree>& trees,
    const std::vector<std::size_t>& row_indices,
    const std::vector<LeafRowRange>& leaf_row_ranges,
    const std::vector<float>& baseline_logits,
    const std::vector<float>& labels,
    const std::vector<float>& weights,
    int num_classes,
    double lambda_l2,
    double max_leaf_weight,
    int iterations,
    bool vector_leaves);

}  // namespace ctboost::booster_detail
