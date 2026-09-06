#include "booster_leaf_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ctboost::booster_detail {
namespace {

// Use the same float addition as the prediction update, then evaluate the
// softmax and loss in double. Acceptance therefore checks the stored candidate,
// including its rounding and constraints, rather than an unconstrained proxy.
double RowProbabilities(const std::vector<float>& baseline_logits,
                        std::size_t row,
                        const std::vector<float>& values,
                        std::vector<double>& probabilities,
                        int label) {
  const std::size_t dimension = values.size();
  const std::size_t offset = row * dimension;
  double maximum = -std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < dimension; ++c) {
    probabilities[c] = static_cast<double>(baseline_logits[offset + c] + values[c]);
    if (!std::isfinite(probabilities[c])) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(maximum, probabilities[c]);
  }
  const double selected = probabilities[static_cast<std::size_t>(label)];
  double sum = 0.0;
  for (double& probability : probabilities) {
    probability = std::exp(probability - maximum);
    sum += probability;
  }
  for (double& probability : probabilities) {
    probability /= sum;
  }
  return (maximum - selected) + std::log(sum);
}

double LeafLoss(const std::vector<std::size_t>& row_indices,
                const LeafRowRange& range,
                const std::vector<float>& baseline_logits,
                const std::vector<float>& labels,
                const std::vector<float>& weights,
                const std::vector<float>& values,
                double lambda_l2,
                std::vector<double>& probabilities) {
  double loss = 0.0;
  for (std::size_t position = range.begin; position < range.end; ++position) {
    const std::size_t row = row_indices[position];
    const double weight = weights[row];
    if (weight == 0.0) continue;
    loss += weight * RowProbabilities(baseline_logits, row, values, probabilities,
                                      static_cast<int>(labels[row]));
  }
  for (float value : values) {
    loss += 0.5 * lambda_l2 * static_cast<double>(value) * value;
  }
  return loss;
}

void Center(std::vector<double>& values) {
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  for (double& value : values) value -= mean;
}

// Conjugate gradients stays in the sum-zero subspace, so the softmax gauge
// never enters the system. Adding a large 11^T term before a factorization
// loses accuracy when probabilities saturate and the actual Hessian is tiny.
// Relative damping also covers additional null directions at lambda_l2=0.
bool ProjectedNewtonDirection(const std::vector<double>& hessian,
                              const std::vector<double>& gradient,
                              double damping,
                              std::vector<double>& solution) {
  const std::size_t dimension = gradient.size();
  solution.assign(dimension, 0.0);
  std::vector<double> residual = gradient;
  std::vector<double> direction = gradient;
  std::vector<double> product(dimension);
  double residual_norm = std::inner_product(residual.begin(), residual.end(), residual.begin(), 0.0);
  if (residual_norm == 0.0) return true;
  const double tolerance = residual_norm * 1e-20;
  bool converged = false;
  for (std::size_t step = 0; step < 4 * dimension; ++step) {
    for (std::size_t c = 0; c < dimension; ++c) {
      product[c] = damping * direction[c];
      for (std::size_t d = 0; d < dimension; ++d) {
        product[c] += hessian[c * dimension + d] * direction[d];
      }
    }
    Center(product);
    const double curvature = std::inner_product(direction.begin(), direction.end(), product.begin(), 0.0);
    if (!(curvature > 0.0) || !std::isfinite(curvature)) return false;
    const double fraction = residual_norm / curvature;
    for (std::size_t c = 0; c < dimension; ++c) {
      solution[c] += fraction * direction[c];
      residual[c] -= fraction * product[c];
    }
    Center(residual);
    const double next_norm = std::inner_product(residual.begin(), residual.end(), residual.begin(), 0.0);
    if (next_norm <= tolerance) {
      converged = true;
      break;
    }
    const double beta = next_norm / residual_norm;
    for (std::size_t c = 0; c < dimension; ++c) {
      direction[c] = residual[c] + beta * direction[c];
    }
    Center(direction);
    residual_norm = next_norm;
  }
  if (!converged) return false;
  Center(solution);
  for (double value : solution) {
    if (!std::isfinite(value)) return false;
  }
  return std::inner_product(gradient.begin(), gradient.end(), solution.begin(), 0.0) > 0.0;
}

std::vector<float> ConstrainValues(std::vector<double> values, double max_leaf_weight) {
  Center(values);
  // Project onto the intersection of the sum-zero hyperplane and the box.
  // Clipping then centering alone can break the requested box constraint.
  const double bound = max_leaf_weight > 0.0
                           ? std::min(max_leaf_weight,
                                      static_cast<double>(std::numeric_limits<float>::max()))
                           : static_cast<double>(std::numeric_limits<float>::max());
  const auto extrema = std::minmax_element(values.begin(), values.end());
  if (*extrema.first < -bound || *extrema.second > bound) {
    double lower = *extrema.first - bound;
    double upper = *extrema.second + bound;
    for (int step = 0; step < 80; ++step) {
      const double midpoint = lower + 0.5 * (upper - lower);
      double sum = 0.0;
      for (double value : values) sum += std::clamp(value - midpoint, -bound, bound);
      if (sum > 0.0) {
        lower = midpoint;
      } else {
        upper = midpoint;
      }
    }
    const double shift = lower + 0.5 * (upper - lower);
    for (double& value : values) value = std::clamp(value - shift, -bound, bound);
  }
  // If the cap is tiny compared with the Newton direction, subtraction in
  // the projection can lose its last bits. Redistribute any residual equally
  // across coordinates with available slack, preserving class symmetry.
  for (std::size_t pass = 0; pass <= values.size(); ++pass) {
    const double residual = std::accumulate(values.begin(), values.end(), 0.0);
    if (residual == 0.0) break;
    std::size_t movable = 0;
    for (double value : values) {
      if ((residual > 0.0 && value > -bound) || (residual < 0.0 && value < bound)) ++movable;
    }
    if (movable == 0) break;
    const double correction = residual / static_cast<double>(movable);
    for (double& value : values) {
      if ((residual > 0.0 && value > -bound) || (residual < 0.0 && value < bound)) {
        value = std::clamp(value - correction, -bound, bound);
      }
    }
  }
  std::vector<float> result(values.size());
  for (std::size_t c = 0; c < values.size(); ++c) {
    result[c] = static_cast<float>(values[c]);
    // A non-representable cap can round outward when converted to float.
    if (static_cast<double>(result[c]) > bound) {
      result[c] = std::nextafter(result[c], 0.0F);
    } else if (static_cast<double>(result[c]) < -bound) {
      result[c] = std::nextafter(result[c], 0.0F);
    }
  }
  return result;
}

std::vector<float> FitLeaf(const std::vector<std::size_t>& row_indices,
                           const LeafRowRange& range,
                           const std::vector<float>& baseline_logits,
                           const std::vector<float>& labels,
                           const std::vector<float>& weights,
                           std::size_t dimension,
                           double lambda_l2,
                           double max_leaf_weight,
                           int iterations) {
  std::vector<float> values(dimension, 0.0F);
  std::vector<double> probabilities(dimension);
  double total_weight = 0.0;
  for (std::size_t position = range.begin; position < range.end; ++position) {
    total_weight += weights[row_indices[position]];
  }
  if (!(total_weight > 0.0)) return values;
  double previous_loss = LeafLoss(row_indices, range, baseline_logits, labels, weights,
                                  values, lambda_l2, probabilities);
  for (int step = 0; step < iterations; ++step) {
    std::vector<double> gradient(dimension, 0.0);
    std::vector<double> hessian(dimension * dimension, 0.0);
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const double weight = weights[row];
      if (weight == 0.0) continue;
      const int label = static_cast<int>(labels[row]);
      if (!std::isfinite(RowProbabilities(baseline_logits, row, values, probabilities, label))) {
        return values;
      }
      for (std::size_t c = 0; c < dimension; ++c) {
        gradient[c] += weight * (probabilities[c] - (c == static_cast<std::size_t>(label) ? 1.0 : 0.0));
        for (std::size_t d = 0; d <= c; ++d) {
          hessian[c * dimension + d] += weight * probabilities[c] *
                                        ((c == d ? 1.0 : 0.0) - probabilities[d]);
        }
      }
    }
    for (std::size_t c = 0; c < dimension; ++c) {
      gradient[c] += lambda_l2 * values[c];
      hessian[c * dimension + c] += lambda_l2;
      for (std::size_t d = 0; d < c; ++d) {
        hessian[d * dimension + c] = hessian[c * dimension + d];
      }
    }
    Center(gradient);
    const double scale = std::max(total_weight, lambda_l2);
    std::vector<double> direction;
    bool solved = false;
    for (int attempt = 0; attempt < 6 && !solved; ++attempt) {
      // Bound conditioning on the sum-zero subspace even when all class
      // probabilities have saturated. A much smaller floor makes float leaf
      // updates depend on class order through an unstable Newton direction.
      const double damping = scale * (1e-6 * std::pow(100.0, attempt));
      solved = ProjectedNewtonDirection(hessian, gradient, damping, direction);
    }
    if (!solved) break;
    Center(direction);
    bool accepted = false;
    for (int attempt = 0; attempt < 50; ++attempt) {
      const double fraction = std::ldexp(1.0, -attempt);
      std::vector<double> proposal(dimension);
      for (std::size_t c = 0; c < dimension; ++c) {
        proposal[c] = static_cast<double>(values[c]) - fraction * direction[c];
      }
      std::vector<float> candidate = ConstrainValues(std::move(proposal), max_leaf_weight);
      const double loss = LeafLoss(row_indices, range, baseline_logits, labels, weights,
                                    candidate, lambda_l2, probabilities);
      if (std::isfinite(loss) && loss <= previous_loss) {
        accepted = true;
        values = std::move(candidate);
        previous_loss = loss;
        break;
      }
    }
    if (!accepted) break;
  }
  return values;
}

}  // namespace

void FitFullSoftmaxTreeLeaves(std::vector<Tree>& trees,
                              const std::vector<std::size_t>& row_indices,
                              const std::vector<LeafRowRange>& leaf_row_ranges,
                              const std::vector<float>& baseline_logits,
                              const std::vector<float>& labels,
                              const std::vector<float>& weights,
                              int num_classes,
                              double lambda_l2,
                              double max_leaf_weight,
                              int iterations,
                              bool vector_leaves) {
  const std::size_t dimension = static_cast<std::size_t>(num_classes);
  if (num_classes < 2 || num_classes > 32 || iterations < 1 || iterations > 5 ||
      trees.size() != (vector_leaves ? 1 : dimension) ||
      baseline_logits.size() != labels.size() * dimension || weights.size() != labels.size()) {
    throw std::invalid_argument("invalid full softmax leaf solver inputs");
  }
  const auto& nodes = trees.front().nodes();
  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    if (!nodes[node_index].is_leaf || node_index >= leaf_row_ranges.size()) continue;
    std::vector<float> values = FitLeaf(row_indices, leaf_row_ranges[node_index],
                                        baseline_logits, labels, weights, dimension,
                                        lambda_l2, max_leaf_weight, iterations);
    if (vector_leaves) {
      trees.front().SetLeafWeights(node_index, std::move(values));
    } else {
      for (std::size_t c = 0; c < dimension; ++c) {
        trees[c].SetLeafWeight(node_index, values[c]);
      }
    }
  }
}

}  // namespace ctboost::booster_detail
