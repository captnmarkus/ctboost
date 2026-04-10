#include "ctboost/objective.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ctboost {
namespace {

void ValidatePredictionLabelSizes(const std::vector<float>& preds,
                                  const std::vector<float>& labels) {
  if (preds.size() != labels.size()) {
    throw std::invalid_argument("predictions and labels must have the same size");
  }
}

float Sigmoid(float margin) {
  if (margin >= 0.0F) {
    const float exp_term = std::exp(-margin);
    return 1.0F / (1.0F + exp_term);
  }

  const float exp_term = std::exp(margin);
  return exp_term / (1.0F + exp_term);
}

int LabelToClassIndex(float label, int num_classes) {
  if (num_classes <= 1) {
    throw std::invalid_argument("num_classes must be greater than one");
  }

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

void ValidateMulticlassSizes(const std::vector<float>& preds,
                             const std::vector<float>& labels,
                             int num_classes) {
  if (num_classes <= 2) {
    throw std::invalid_argument("softmax loss requires num_classes greater than two");
  }
  if (preds.size() != labels.size() * static_cast<std::size_t>(num_classes)) {
    throw std::invalid_argument(
        "multiclass predictions must have num_rows * num_classes elements");
  }
}

std::string NormalizeName(std::string_view name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (const char ch : name) {
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return normalized;
}

}  // namespace

void SquaredError::compute_gradients(const std::vector<float>& preds,
                                     const std::vector<float>& labels,
                                     std::vector<float>& out_g,
                                     std::vector<float>& out_h,
                                     int num_classes) const {
  if (num_classes != 1) {
    throw std::invalid_argument("squared error expects num_classes equal to one");
  }
  ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size(), 1.0F);

  for (std::size_t i = 0; i < preds.size(); ++i) {
    out_g[i] = preds[i] - labels[i];
  }
}

void LogLoss::compute_gradients(const std::vector<float>& preds,
                                const std::vector<float>& labels,
                                std::vector<float>& out_g,
                                std::vector<float>& out_h,
                                int num_classes) const {
  if (num_classes != 1 && num_classes != 2) {
    throw std::invalid_argument("logloss expects num_classes equal to one or two");
  }
  ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size());

  for (std::size_t i = 0; i < preds.size(); ++i) {
    const float probability = Sigmoid(preds[i]);
    out_g[i] = probability - labels[i];
    out_h[i] = probability * (1.0F - probability);
  }
}

void SoftmaxLoss::compute_gradients(const std::vector<float>& preds,
                                    const std::vector<float>& labels,
                                    std::vector<float>& out_g,
                                    std::vector<float>& out_h,
                                    int num_classes) const {
  ValidateMulticlassSizes(preds, labels, num_classes);

  out_g.resize(preds.size());
  out_h.resize(preds.size());

  for (std::size_t row = 0; row < labels.size(); ++row) {
    const std::size_t row_offset =
        row * static_cast<std::size_t>(num_classes);
    const auto row_begin = preds.begin() + static_cast<std::ptrdiff_t>(row_offset);
    const auto row_end = row_begin + num_classes;
    const float max_logit = *std::max_element(row_begin, row_end);

    double exp_sum = 0.0;
    for (int class_index = 0; class_index < num_classes; ++class_index) {
      exp_sum += std::exp(static_cast<double>(preds[row_offset + class_index] - max_logit));
    }

    const int target_class = LabelToClassIndex(labels[row], num_classes);
    for (int class_index = 0; class_index < num_classes; ++class_index) {
      const float probability =
          static_cast<float>(std::exp(static_cast<double>(
                                 preds[row_offset + class_index] - max_logit)) /
                             exp_sum);
      out_g[row_offset + class_index] =
          probability - (class_index == target_class ? 1.0F : 0.0F);
      out_h[row_offset + class_index] = probability * (1.0F - probability);
    }
  }
}

std::unique_ptr<ObjectiveFunction> CreateObjectiveFunction(std::string_view name) {
  const std::string normalized = NormalizeName(name);

  if (normalized == "rmse" || normalized == "squarederror" ||
      normalized == "squared_error") {
    return std::make_unique<SquaredError>();
  }
  if (normalized == "logloss" || normalized == "binary_logloss" ||
      normalized == "binary:logistic") {
    return std::make_unique<LogLoss>();
  }
  if (normalized == "multiclass" || normalized == "softmax" ||
      normalized == "softmaxloss") {
    return std::make_unique<SoftmaxLoss>();
  }

  throw std::invalid_argument("unknown objective function: " + std::string(name));
}

}  // namespace ctboost
