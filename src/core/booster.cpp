#include "ctboost/booster.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost {
namespace {

std::string NormalizeToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool IsSquaredErrorObjective(const std::string& normalized_objective) {
  return normalized_objective == "rmse" || normalized_objective == "squarederror" ||
         normalized_objective == "squared_error";
}

bool IsAbsoluteErrorObjective(const std::string& normalized_objective) {
  return normalized_objective == "mae" || normalized_objective == "l1" ||
         normalized_objective == "absoluteerror" || normalized_objective == "absolute_error";
}

bool IsHuberObjective(const std::string& normalized_objective) {
  return normalized_objective == "huber" || normalized_objective == "huberloss";
}

bool IsQuantileObjective(const std::string& normalized_objective) {
  return normalized_objective == "quantile" || normalized_objective == "quantileloss";
}

bool IsBinaryObjective(const std::string& normalized_objective) {
  return normalized_objective == "logloss" || normalized_objective == "binary_logloss" ||
         normalized_objective == "binary:logistic";
}

bool IsMulticlassObjective(const std::string& normalized_objective) {
  return normalized_objective == "multiclass" || normalized_objective == "softmax" ||
         normalized_objective == "softmaxloss";
}

bool IsRankingObjective(const std::string& normalized_objective) {
  return normalized_objective == "pairlogit" || normalized_objective == "pairwise" ||
         normalized_objective == "ranknet";
}

bool IsRegressionObjective(const std::string& normalized_objective) {
  return IsSquaredErrorObjective(normalized_objective) ||
         IsAbsoluteErrorObjective(normalized_objective) ||
         IsHuberObjective(normalized_objective) ||
         IsQuantileObjective(normalized_objective);
}

bool SameCategoricalFeatures(const Pool& lhs, const Pool& rhs) {
  const auto& lhs_features = lhs.cat_features();
  const auto& rhs_features = rhs.cat_features();
  return lhs_features.size() == rhs_features.size() &&
         std::is_permutation(lhs_features.begin(), lhs_features.end(), rhs_features.begin());
}

int LabelToClassIndex(float label, int num_classes) {
  const float rounded = std::round(label);
  if (std::fabs(label - rounded) > 1e-6F) {
    throw std::invalid_argument("multiclass labels must be integer encoded");
  }

  const int class_index = static_cast<int>(rounded);
  if (class_index < 0 || class_index >= num_classes) {
    throw std::invalid_argument("multiclass label is out of range");
  }

  return class_index;
}

std::string NormalizeTaskType(std::string task_type) {
  return NormalizeToken(std::move(task_type));
}

HistMatrix BuildPredictionHist(const Pool& pool, const Tree& reference_tree) {
  const std::size_t model_num_features = reference_tree.num_bins_per_feature().size();
  if (pool.num_cols() != model_num_features) {
    throw std::invalid_argument(
        "prediction pool must have the same number of columns as the fitted model");
  }

  HistMatrix hist;
  hist.num_rows = pool.num_rows();
  hist.num_cols = pool.num_cols();
  hist.bin_indices.resize(hist.num_rows * hist.num_cols, 0);
  hist.num_bins_per_feature = reference_tree.num_bins_per_feature();
  hist.cut_offsets = reference_tree.cut_offsets();
  hist.cut_values = reference_tree.cut_values();
  hist.categorical_mask = reference_tree.categorical_mask();
  hist.missing_value_mask = reference_tree.missing_value_mask();
  hist.nan_mode = reference_tree.nan_mode();

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t offset = feature * hist.num_rows;
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      hist.bin_indices[offset + row] = hist.bin_value(feature, pool.feature_value(row, feature));
    }
  }

  return hist;
}

std::vector<GpuTreeNode> FlattenTreesForGpu(const std::vector<Tree>& trees,
                                            std::size_t tree_limit,
                                            std::vector<std::int32_t>& tree_offsets) {
  std::size_t total_nodes = 0;
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    total_nodes += trees[tree_index].nodes().size();
  }

  std::vector<GpuTreeNode> flattened_nodes;
  flattened_nodes.reserve(total_nodes);
  tree_offsets.clear();
  tree_offsets.reserve(tree_limit);

  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const auto& tree_nodes = trees[tree_index].nodes();
    const std::int32_t tree_offset = static_cast<std::int32_t>(flattened_nodes.size());
    tree_offsets.push_back(tree_offset);
    for (const Node& node : tree_nodes) {
      GpuTreeNode gpu_node;
      gpu_node.is_leaf = node.is_leaf ? 1U : 0U;
      gpu_node.is_categorical_split = node.is_categorical_split ? 1U : 0U;
      gpu_node.split_bin_index = node.split_bin_index;
      gpu_node.split_feature_id = static_cast<std::int32_t>(node.split_feature_id);
      gpu_node.left_child =
          node.left_child < 0 ? -1 : tree_offset + static_cast<std::int32_t>(node.left_child);
      gpu_node.right_child =
          node.right_child < 0 ? -1 : tree_offset + static_cast<std::int32_t>(node.right_child);
      gpu_node.leaf_weight = node.leaf_weight;
      std::copy(node.left_categories.begin(), node.left_categories.end(), gpu_node.left_categories);
      flattened_nodes.push_back(std::move(gpu_node));
    }
  }

  return flattened_nodes;
}

void UpdatePredictions(const Tree& tree,
                       const HistMatrix& hist,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      predictions[row] += learning_rate * tree.PredictBinnedRow(hist, row);
    }
    return;
  }

  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictBinnedRow(hist, row);
  }
}

void UpdatePredictions(const Tree& tree,
                       const Pool& pool,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row] += learning_rate * tree.PredictRow(pool, row);
    }
    return;
  }

  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictRow(pool, row);
  }
}

void AccumulateFeatureImportances(const Tree& tree, std::vector<double>& feature_importance_sums) {
  const auto& tree_feature_importances = tree.feature_importances();
  for (std::size_t feature = 0; feature < tree_feature_importances.size(); ++feature) {
    feature_importance_sums[feature] += tree_feature_importances[feature];
  }
}

void RecomputeFeatureImportances(const std::vector<Tree>& trees,
                                 std::size_t num_features,
                                 std::vector<double>& feature_importance_sums) {
  feature_importance_sums.assign(num_features, 0.0);
  for (const Tree& tree : trees) {
    AccumulateFeatureImportances(tree, feature_importance_sums);
  }
}

}  // namespace

GradientBooster::GradientBooster(std::string objective,
                                 int iterations,
                                 double learning_rate,
                                 int max_depth,
                                 double alpha,
                                 double lambda_l2,
                                 int num_classes,
                                 std::size_t max_bins,
                                 std::string nan_mode,
                                 std::string eval_metric,
                                 double quantile_alpha,
                                 double huber_delta,
                                 std::string task_type,
                                 std::string devices,
                                 bool verbose)
    : objective_name_(std::move(objective)),
      eval_metric_name_(std::move(eval_metric)),
      objective_config_{huber_delta, quantile_alpha},
      objective_(CreateObjectiveFunction(objective_name_, objective_config_)),
      objective_metric_(CreateMetricFunctionForObjective(objective_name_, objective_config_)),
      iterations_(iterations),
      learning_rate_(learning_rate),
      max_depth_(max_depth),
      alpha_(alpha),
      lambda_l2_(lambda_l2),
      num_classes_(num_classes),
      max_bins_(max_bins),
      devices_(std::move(devices)),
      verbose_(TrainingProfiler::ResolveEnabled(verbose)),
      hist_builder_(max_bins_, std::move(nan_mode)) {
  if (eval_metric_name_.empty()) {
    eval_metric_name_ = objective_name_;
  }
  eval_metric_ = CreateMetricFunction(eval_metric_name_, objective_config_);
  maximize_eval_metric_ = eval_metric_->HigherIsBetter();

  if (iterations_ <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (learning_rate_ <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive");
  }
  if (max_depth_ < 0) {
    throw std::invalid_argument("max_depth must be non-negative");
  }
  if (lambda_l2_ < 0.0) {
    throw std::invalid_argument("lambda_l2 must be non-negative");
  }
  if (num_classes_ <= 0) {
    throw std::invalid_argument("num_classes must be positive");
  }

  const std::string normalized_objective = NormalizeToken(objective_name_);
  if (IsMulticlassObjective(normalized_objective)) {
    if (num_classes_ <= 2) {
      throw std::invalid_argument("multiclass objective requires num_classes greater than two");
    }
    prediction_dimension_ = num_classes_;
  } else if (IsRankingObjective(normalized_objective)) {
    if (num_classes_ != 1) {
      throw std::invalid_argument("ranking objectives require num_classes equal to one");
    }
    prediction_dimension_ = 1;
  } else if (IsRegressionObjective(normalized_objective)) {
    if (num_classes_ != 1) {
      throw std::invalid_argument("regression objectives require num_classes equal to one");
    }
    prediction_dimension_ = 1;
  } else if (IsBinaryObjective(normalized_objective)) {
    if (num_classes_ != 1 && num_classes_ != 2) {
      throw std::invalid_argument("binary objectives require num_classes equal to one or two");
    }
    prediction_dimension_ = 1;
  }

  const std::string normalized_task_type = NormalizeTaskType(std::move(task_type));
  if (normalized_task_type == "cpu") {
    use_gpu_ = false;
  } else if (normalized_task_type == "gpu") {
    if (!CudaBackendCompiled()) {
      throw std::runtime_error(
          "task_type='GPU' was requested but CTBoost was compiled without CUDA support");
    }
    use_gpu_ = true;
  } else {
    throw std::invalid_argument("task_type must be either 'CPU' or 'GPU'");
  }
}

void GradientBooster::Fit(const Pool& pool,
                          const Pool* eval_pool,
                          int early_stopping_rounds,
                          bool continue_training) {
  const auto fit_start = std::chrono::steady_clock::now();
  const TrainingProfiler profiler(verbose_);
  profiler.LogFitStart(
      pool.num_rows(), pool.num_cols(), iterations_, use_gpu_, prediction_dimension_);
  if (early_stopping_rounds < 0) {
    early_stopping_rounds = 0;
  }
  if (early_stopping_rounds > 0 && eval_pool == nullptr) {
    throw std::invalid_argument("early_stopping_rounds requires eval_pool");
  }
  if (eval_pool != nullptr) {
    if (eval_pool->num_cols() != pool.num_cols()) {
      throw std::invalid_argument(
          "eval_pool must have the same number of columns as the training pool");
    }
    if (!SameCategoricalFeatures(pool, *eval_pool)) {
      throw std::invalid_argument(
          "eval_pool categorical feature indices must match the training pool");
    }
  }

  const bool has_existing_state = continue_training && !trees_.empty();
  if (!continue_training) {
    trees_.clear();
    loss_history_.clear();
    eval_loss_history_.clear();
    best_iteration_ = -1;
    best_score_ = 0.0;
  } else if (!feature_importance_sums_.empty() &&
             feature_importance_sums_.size() != pool.num_cols()) {
    throw std::invalid_argument(
        "warm-start training requires the same number of features as the initial model");
  }

  const auto hist_build_start = std::chrono::steady_clock::now();
  const HistMatrix hist = hist_builder_.Build(pool, &profiler);
  const double hist_build_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_build_start)
          .count();
  const auto& labels = pool.labels();
  const auto& weights = pool.weights();
  GpuHistogramWorkspacePtr gpu_hist_workspace(nullptr, DestroyGpuHistogramWorkspace);
  if (use_gpu_) {
    gpu_hist_workspace = CreateGpuHistogramWorkspace(hist, weights);
  }
  const std::vector<std::int64_t>* group_ids =
      pool.has_group_ids() ? &pool.group_ids() : nullptr;
  const double total_weight = std::accumulate(
      weights.begin(), weights.end(), 0.0,
      [](double acc, float value) { return acc + static_cast<double>(value); });
  if (total_weight <= 0.0) {
    throw std::invalid_argument("training pool must have a positive total sample weight");
  }
  std::vector<float> predictions = has_existing_state
                                       ? Predict(pool, -1)
                                       : std::vector<float>(
                                             pool.num_rows() *
                                                 static_cast<std::size_t>(prediction_dimension_),
                                             0.0F);
  if (!continue_training || feature_importance_sums_.empty()) {
    feature_importance_sums_.assign(pool.num_cols(), 0.0);
  }
  const int initial_completed_iterations = static_cast<int>(num_iterations_trained());
  int completed_iterations = initial_completed_iterations;
  const int target_total_iterations = initial_completed_iterations + iterations_;
  bool early_stopped = false;

  const std::vector<float>* eval_labels = nullptr;
  const std::vector<float>* eval_weights = nullptr;
  const std::vector<std::int64_t>* eval_group_ids = nullptr;
  std::vector<float> eval_predictions;
  if (eval_pool != nullptr) {
    eval_labels = &eval_pool->labels();
    eval_weights = &eval_pool->weights();
    eval_group_ids = eval_pool->has_group_ids() ? &eval_pool->group_ids() : nullptr;
    const double eval_total_weight = std::accumulate(
        eval_weights->begin(), eval_weights->end(), 0.0,
        [](double acc, float value) { return acc + static_cast<double>(value); });
    if (eval_total_weight <= 0.0) {
      throw std::invalid_argument("eval_pool must have a positive total sample weight");
    }
    eval_predictions = has_existing_state
                           ? Predict(*eval_pool, -1)
                           : std::vector<float>(
                                 eval_pool->num_rows() *
                                     static_cast<std::size_t>(prediction_dimension_),
                                 0.0F);
    if (!continue_training || eval_loss_history_.empty()) {
      best_score_ = maximize_eval_metric_ ? -std::numeric_limits<double>::infinity()
                                          : std::numeric_limits<double>::infinity();
      if (!continue_training) {
        best_iteration_ = -1;
      }
    }
  }

  for (int iteration = 0; iteration < iterations_; ++iteration) {
    const auto iteration_start = std::chrono::steady_clock::now();
    const int total_iteration = initial_completed_iterations + iteration;
    std::vector<float> gradients;
    std::vector<float> hessians;
    const auto gradient_start = std::chrono::steady_clock::now();
    objective_->compute_gradients(predictions, labels, gradients, hessians, num_classes_, group_ids);
    const double gradient_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gradient_start)
            .count();
    double tree_ms = 0.0;

    if (prediction_dimension_ == 1) {
      if (use_gpu_) {
        UploadHistogramTargetsGpu(gpu_hist_workspace.get(), gradients, hessians);
      }
      Tree tree;
      const auto tree_start = std::chrono::steady_clock::now();
      tree.Build(
          hist,
          gradients,
          hessians,
          weights,
          alpha_,
          max_depth_,
          lambda_l2_,
          use_gpu_,
          gpu_hist_workspace.get(),
          &profiler);
      const double single_tree_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tree_start)
              .count();
      tree_ms += single_tree_ms;
      profiler.LogTreeBuild(
          total_iteration + 1, target_total_iterations, 0, prediction_dimension_, single_tree_ms);

      UpdatePredictions(tree, hist, learning_rate_, prediction_dimension_, 0, predictions);
      if (eval_pool != nullptr) {
        UpdatePredictions(tree, *eval_pool, learning_rate_, prediction_dimension_, 0, eval_predictions);
      }
      AccumulateFeatureImportances(tree, feature_importance_sums_);

      trees_.push_back(std::move(tree));
    } else {
      std::vector<float> class_gradients(pool.num_rows(), 0.0F);
      std::vector<float> class_hessians(pool.num_rows(), 0.0F);

      for (int class_index = 0; class_index < prediction_dimension_; ++class_index) {
        for (std::size_t row = 0; row < pool.num_rows(); ++row) {
          const std::size_t offset =
              row * static_cast<std::size_t>(prediction_dimension_) + class_index;
          class_gradients[row] = gradients[offset];
          class_hessians[row] = hessians[offset];
        }

        if (use_gpu_) {
          UploadHistogramTargetsGpu(gpu_hist_workspace.get(), class_gradients, class_hessians);
        }
        Tree tree;
        const auto tree_start = std::chrono::steady_clock::now();
        tree.Build(hist,
                   class_gradients,
                   class_hessians,
                   weights,
                   alpha_,
                   max_depth_,
                   lambda_l2_,
                   use_gpu_,
                   gpu_hist_workspace.get(),
                   &profiler);
        const double single_tree_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tree_start)
                .count();
        tree_ms += single_tree_ms;
        profiler.LogTreeBuild(total_iteration + 1,
                              target_total_iterations,
                              class_index,
                              prediction_dimension_,
                              single_tree_ms);

        UpdatePredictions(
            tree, hist, learning_rate_, prediction_dimension_, class_index, predictions);
        if (eval_pool != nullptr) {
          UpdatePredictions(
              tree,
              *eval_pool,
              learning_rate_,
              prediction_dimension_,
              class_index,
              eval_predictions);
        }
        AccumulateFeatureImportances(tree, feature_importance_sums_);

        trees_.push_back(std::move(tree));
      }
    }

    const auto metric_start = std::chrono::steady_clock::now();
    loss_history_.push_back(
        objective_metric_->Evaluate(predictions, labels, weights, num_classes_, group_ids));
    const double metric_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metric_start)
            .count();
    completed_iterations = total_iteration + 1;
    if (eval_pool == nullptr) {
      const double iteration_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start)
              .count();
      profiler.LogIteration(total_iteration + 1,
                            target_total_iterations,
                            gradient_ms,
                            tree_ms,
                            metric_ms,
                            0.0,
                            iteration_ms);
      continue;
    }

    const auto eval_start = std::chrono::steady_clock::now();
    const double eval_score =
        eval_metric_->Evaluate(
            eval_predictions, *eval_labels, *eval_weights, num_classes_, eval_group_ids);
    const double eval_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - eval_start)
            .count();
    eval_loss_history_.push_back(eval_score);
    const double iteration_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start)
            .count();
    profiler.LogIteration(total_iteration + 1,
                          target_total_iterations,
                          gradient_ms,
                          tree_ms,
                          metric_ms,
                          eval_ms,
                          iteration_ms);
    const bool improved =
        best_iteration_ < 0 ||
        (maximize_eval_metric_ ? eval_score > best_score_ : eval_score < best_score_);
    if (improved) {
      best_iteration_ = total_iteration;
      best_score_ = eval_score;
      continue;
    }

    if (early_stopping_rounds > 0 &&
        total_iteration - best_iteration_ >= early_stopping_rounds) {
      early_stopped = true;
      break;
    }
  }

  if (eval_pool == nullptr) {
    best_iteration_ = completed_iterations > 0 ? completed_iterations - 1 : -1;
    const double total_fit_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start)
            .count();
    profiler.LogFitSummary(hist_build_ms, total_fit_ms);
    return;
  }

  if (!early_stopped || best_iteration_ < 0) {
    const double total_fit_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start)
            .count();
    profiler.LogFitSummary(hist_build_ms, total_fit_ms);
    return;
  }

  const std::size_t retained_iterations =
      static_cast<std::size_t>(best_iteration_ + 1);
  trees_.resize(retained_iterations * static_cast<std::size_t>(prediction_dimension_));
  loss_history_.resize(retained_iterations);
  eval_loss_history_.resize(retained_iterations);
  RecomputeFeatureImportances(trees_, pool.num_cols(), feature_importance_sums_);
  const double total_fit_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start)
          .count();
  profiler.LogFitSummary(hist_build_ms, total_fit_ms);
}

std::vector<float> GradientBooster::Predict(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  std::vector<float> predictions(
      pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  if (tree_limit == 0 || pool.num_rows() == 0) {
    return predictions;
  }

  if (use_gpu_ && CudaBackendCompiled()) {
    const HistMatrix hist = BuildPredictionHist(pool, trees_.front());
    std::vector<std::int32_t> tree_offsets;
    const std::vector<GpuTreeNode> flattened_nodes = FlattenTreesForGpu(trees_, tree_limit, tree_offsets);
    PredictRawGpu(
        hist, flattened_nodes, tree_offsets, static_cast<float>(learning_rate_), prediction_dimension_, predictions);
    return predictions;
  }

  if (prediction_dimension_ == 1) {
    for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
      const Tree& tree = trees_[tree_index];
      for (std::size_t row = 0; row < pool.num_rows(); ++row) {
        predictions[row] += learning_rate_ * tree.PredictRow(pool, row);
      }
    }
    return predictions;
  }

  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const int class_index = static_cast<int>(
        tree_index % static_cast<std::size_t>(prediction_dimension_));
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row * static_cast<std::size_t>(prediction_dimension_) + class_index] +=
          learning_rate_ * trees_[tree_index].PredictRow(pool, row);
    }
  }
  return predictions;
}

std::vector<std::int32_t> GradientBooster::PredictLeafIndices(const Pool& pool,
                                                              int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  std::vector<std::int32_t> leaf_indices(pool.num_rows() * tree_limit, -1);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      leaf_indices[row * tree_limit + tree_index] =
          trees_[tree_index].PredictLeafIndex(pool, row);
    }
  }
  return leaf_indices;
}

std::vector<float> GradientBooster::PredictContributions(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  const std::size_t row_width =
      static_cast<std::size_t>(prediction_dimension_) * (pool.num_cols() + 1);
  std::vector<float> contributions(pool.num_rows() * row_width, 0.0F);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const std::size_t class_index = prediction_dimension_ == 1
                                        ? 0
                                        : tree_index % static_cast<std::size_t>(prediction_dimension_);
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      std::vector<float> row_buffer(pool.num_cols() + 1, 0.0F);
      trees_[tree_index].AccumulateContributions(pool, row, static_cast<float>(learning_rate_), row_buffer);
      const std::size_t row_offset = row * row_width + class_index * (pool.num_cols() + 1);
      for (std::size_t feature = 0; feature < row_buffer.size(); ++feature) {
        contributions[row_offset + feature] += row_buffer[feature];
      }
    }
  }
  return contributions;
}

void GradientBooster::LoadState(std::vector<Tree> trees,
                                std::vector<double> loss_history,
                                std::vector<double> eval_loss_history,
                                std::vector<double> feature_importance_sums,
                                int best_iteration,
                                double best_score,
                                bool use_gpu) {
  trees_ = std::move(trees);
  loss_history_ = std::move(loss_history);
  eval_loss_history_ = std::move(eval_loss_history);
  if (feature_importance_sums.empty()) {
    const std::size_t num_features =
        trees_.empty() ? 0 : trees_.front().feature_importances().size();
    RecomputeFeatureImportances(trees_, num_features, feature_importance_sums_);
  } else {
    feature_importance_sums_ = std::move(feature_importance_sums);
  }
  best_iteration_ = best_iteration;
  best_score_ = best_score;
  use_gpu_ = use_gpu;
}

const std::vector<double>& GradientBooster::loss_history() const noexcept {
  return loss_history_;
}

const std::vector<double>& GradientBooster::eval_loss_history() const noexcept {
  return eval_loss_history_;
}

std::size_t GradientBooster::num_trees() const noexcept { return trees_.size(); }

std::size_t GradientBooster::num_iterations_trained() const noexcept {
  if (prediction_dimension_ <= 0) {
    return 0;
  }
  return trees_.size() / static_cast<std::size_t>(prediction_dimension_);
}

int GradientBooster::num_classes() const noexcept { return num_classes_; }

int GradientBooster::prediction_dimension() const noexcept { return prediction_dimension_; }

int GradientBooster::best_iteration() const noexcept { return best_iteration_; }

double GradientBooster::best_score() const noexcept { return best_score_; }

const std::string& GradientBooster::objective_name() const noexcept {
  return objective_name_;
}

int GradientBooster::iterations() const noexcept { return iterations_; }

double GradientBooster::learning_rate() const noexcept { return learning_rate_; }

int GradientBooster::max_depth() const noexcept { return max_depth_; }

double GradientBooster::alpha() const noexcept { return alpha_; }

double GradientBooster::lambda_l2() const noexcept { return lambda_l2_; }

std::size_t GradientBooster::max_bins() const noexcept { return max_bins_; }

const std::string& GradientBooster::nan_mode_name() const noexcept {
  return hist_builder_.nan_mode_name();
}

const std::string& GradientBooster::eval_metric_name() const noexcept {
  return eval_metric_name_;
}

double GradientBooster::quantile_alpha() const noexcept {
  return objective_config_.quantile_alpha;
}

double GradientBooster::huber_delta() const noexcept {
  return objective_config_.huber_delta;
}

bool GradientBooster::use_gpu() const noexcept { return use_gpu_; }

const std::string& GradientBooster::devices() const noexcept { return devices_; }

bool GradientBooster::verbose() const noexcept { return verbose_; }

const std::vector<Tree>& GradientBooster::trees() const noexcept { return trees_; }

std::vector<float> GradientBooster::get_feature_importances() const {
  std::vector<float> importances(feature_importance_sums_.size(), 0.0F);
  const double total_importance = std::accumulate(
      feature_importance_sums_.begin(), feature_importance_sums_.end(), 0.0);
  if (total_importance <= 0.0) {
    return importances;
  }

  const double scale = 100.0 / total_importance;
  for (std::size_t feature = 0; feature < feature_importance_sums_.size(); ++feature) {
    importances[feature] = static_cast<float>(feature_importance_sums_[feature] * scale);
  }
  return importances;
}

}  // namespace ctboost
