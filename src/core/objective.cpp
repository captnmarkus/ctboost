#include "ctboost/objective.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
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
                                     std::vector<float>& out_h) const {
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
                                std::vector<float>& out_h) const {
  ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size());

  for (std::size_t i = 0; i < preds.size(); ++i) {
    const float probability = Sigmoid(preds[i]);
    out_g[i] = probability - labels[i];
    out_h[i] = probability * (1.0F - probability);
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

  throw std::invalid_argument("unknown objective function: " + std::string(name));
}

}  // namespace ctboost
