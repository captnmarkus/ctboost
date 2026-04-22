#include "objective_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ctboost {

void SquaredError::compute_gradients(const std::vector<float>& preds,
                                     const std::vector<float>& labels,
                                     std::vector<float>& out_g,
                                     std::vector<float>& out_h,
                                     int num_classes,
                                     const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("squared error expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size(), 1.0F);
  for (std::size_t i = 0; i < preds.size(); ++i) {
    out_g[i] = preds[i] - labels[i];
  }
}

void AbsoluteError::compute_gradients(const std::vector<float>& preds,
                                      const std::vector<float>& labels,
                                      std::vector<float>& out_g,
                                      std::vector<float>& out_h,
                                      int num_classes,
                                      const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("absolute error expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size(), 1.0F);
  for (std::size_t i = 0; i < preds.size(); ++i) {
    const float residual = preds[i] - labels[i];
    if (residual > 0.0F) {
      out_g[i] = 1.0F;
    } else if (residual < 0.0F) {
      out_g[i] = -1.0F;
    } else {
      out_g[i] = 0.0F;
    }
  }
}

HuberLoss::HuberLoss(double delta) : delta_(delta) {
  if (delta_ <= 0.0) {
    throw std::invalid_argument("huber_delta must be positive");
  }
}

void HuberLoss::compute_gradients(const std::vector<float>& preds,
                                  const std::vector<float>& labels,
                                  std::vector<float>& out_g,
                                  std::vector<float>& out_h,
                                  int num_classes,
                                  const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("huber loss expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size());
  for (std::size_t i = 0; i < preds.size(); ++i) {
    const double residual = static_cast<double>(preds[i]) - labels[i];
    const double absolute_residual = std::fabs(residual);
    if (absolute_residual <= delta_) {
      out_g[i] = static_cast<float>(residual);
      out_h[i] = 1.0F;
    } else {
      out_g[i] = static_cast<float>(delta_ * (residual > 0.0 ? 1.0 : -1.0));
      out_h[i] = 0.0F;
    }
  }
}

QuantileLoss::QuantileLoss(double alpha) : alpha_(alpha) {
  if (!(alpha_ > 0.0 && alpha_ < 1.0)) {
    throw std::invalid_argument("quantile_alpha must be in the open interval (0, 1)");
  }
}

void QuantileLoss::compute_gradients(const std::vector<float>& preds,
                                     const std::vector<float>& labels,
                                     std::vector<float>& out_g,
                                     std::vector<float>& out_h,
                                     int num_classes,
                                     const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("quantile loss expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size(), 1.0F);
  for (std::size_t i = 0; i < preds.size(); ++i) {
    const double residual = static_cast<double>(labels[i]) - preds[i];
    out_g[i] = residual > 0.0 ? static_cast<float>(-alpha_) : static_cast<float>(1.0 - alpha_);
  }
}

void PoissonLoss::compute_gradients(const std::vector<float>& preds,
                                    const std::vector<float>& labels,
                                    std::vector<float>& out_g,
                                    std::vector<float>& out_h,
                                    int num_classes,
                                    const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("poisson loss expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);
  detail::ValidateNonNegativeLabels(labels, "poisson loss");

  out_g.resize(preds.size());
  out_h.resize(preds.size());
  for (std::size_t i = 0; i < preds.size(); ++i) {
    const double mean = std::exp(static_cast<double>(preds[i]));
    out_g[i] = static_cast<float>(mean - static_cast<double>(labels[i]));
    out_h[i] = static_cast<float>(mean);
  }
}

TweedieLoss::TweedieLoss(double variance_power) : variance_power_(variance_power) {
  if (!(variance_power_ > 1.0 && variance_power_ < 2.0)) {
    throw std::invalid_argument("tweedie_variance_power must be in the open interval (1, 2)");
  }
}

void TweedieLoss::compute_gradients(const std::vector<float>& preds,
                                    const std::vector<float>& labels,
                                    std::vector<float>& out_g,
                                    std::vector<float>& out_h,
                                    int num_classes,
                                    const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("tweedie loss expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);
  detail::ValidateNonNegativeLabels(labels, "tweedie loss");

  out_g.resize(preds.size());
  out_h.resize(preds.size());
  const double one_minus_power = 1.0 - variance_power_;
  const double two_minus_power = 2.0 - variance_power_;
  for (std::size_t i = 0; i < preds.size(); ++i) {
    const double prediction = static_cast<double>(preds[i]);
    const double label = static_cast<double>(labels[i]);
    const double exp_one_minus = std::exp(one_minus_power * prediction);
    const double exp_two_minus = std::exp(two_minus_power * prediction);
    const double gradient = -label * exp_one_minus + exp_two_minus;
    const double hessian =
        label * (variance_power_ - 1.0) * exp_one_minus + two_minus_power * exp_two_minus;
    out_g[i] = static_cast<float>(gradient);
    out_h[i] = static_cast<float>(std::max(hessian, 1e-12));
  }
}

}  // namespace ctboost

namespace ctboost::detail {

std::unique_ptr<ObjectiveFunction> CreateRegressionObjective(std::string_view normalized,
                                                             const ObjectiveConfig& config) {
  if (normalized == "rmse" || normalized == "squarederror" ||
      normalized == "squared_error") {
    return std::make_unique<SquaredError>();
  }
  if (normalized == "mae" || normalized == "l1" || normalized == "absoluteerror" ||
      normalized == "absolute_error") {
    return std::make_unique<AbsoluteError>();
  }
  if (normalized == "huber" || normalized == "huberloss") {
    return std::make_unique<HuberLoss>(config.huber_delta);
  }
  if (normalized == "quantile" || normalized == "quantileloss") {
    return std::make_unique<QuantileLoss>(config.quantile_alpha);
  }
  if (normalized == "poisson" || normalized == "poissonregression") {
    return std::make_unique<PoissonLoss>();
  }
  if (normalized == "tweedie" || normalized == "tweedieloss" ||
      normalized == "reg:tweedie") {
    return std::make_unique<TweedieLoss>(config.tweedie_variance_power);
  }
  return nullptr;
}

}  // namespace ctboost::detail
