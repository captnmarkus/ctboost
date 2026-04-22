#include "metric_internal.hpp"

#include <cmath>
#include <stdexcept>

namespace {

class RMSEMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double squared_error_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double residual = static_cast<double>(preds[i]) - labels[i];
      const double sample_weight = weights[i];
      squared_error_sum += sample_weight * residual * residual;
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : std::sqrt(squared_error_sum / weight_sum);
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class MAEMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double absolute_error_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double sample_weight = weights[i];
      absolute_error_sum += sample_weight * std::fabs(static_cast<double>(preds[i]) - labels[i]);
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : absolute_error_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class HuberMetric final : public ctboost::MetricFunction {
 public:
  explicit HuberMetric(double delta) : delta_(delta) {
    if (delta_ <= 0.0) {
      throw std::invalid_argument("huber_delta must be positive");
    }
  }

  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double residual = static_cast<double>(preds[i]) - labels[i];
      const double absolute_residual = std::fabs(residual);
      const double sample_weight = weights[i];
      if (absolute_residual <= delta_) {
        loss_sum += sample_weight * 0.5 * residual * residual;
      } else {
        loss_sum += sample_weight * delta_ * (absolute_residual - 0.5 * delta_);
      }
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }

 private:
  double delta_{1.0};
};

class QuantileMetric final : public ctboost::MetricFunction {
 public:
  explicit QuantileMetric(double alpha) : alpha_(alpha) {
    if (!(alpha_ > 0.0 && alpha_ < 1.0)) {
      throw std::invalid_argument("quantile_alpha must be in the open interval (0, 1)");
    }
  }

  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double residual = static_cast<double>(labels[i]) - preds[i];
      const double sample_weight = weights[i];
      loss_sum += sample_weight *
                  (residual >= 0.0 ? alpha_ * residual : (1.0 - alpha_) * (-residual));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }

 private:
  double alpha_{0.5};
};

class PoissonMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    ctboost::detail::ValidateNonNegativeMetricLabels(labels, "poisson metric");
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double label = static_cast<double>(labels[i]);
      const double mean = std::exp(static_cast<double>(preds[i]));
      const double sample_weight = static_cast<double>(weights[i]);
      loss_sum += sample_weight *
                  (mean - label * static_cast<double>(preds[i]) + std::lgamma(label + 1.0));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class TweedieMetric final : public ctboost::MetricFunction {
 public:
  explicit TweedieMetric(double variance_power) : variance_power_(variance_power) {
    if (!(variance_power_ > 1.0 && variance_power_ < 2.0)) {
      throw std::invalid_argument("tweedie_variance_power must be in the open interval (1, 2)");
    }
  }

  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    ctboost::detail::ValidateNonNegativeMetricLabels(labels, "tweedie metric");
    const double one_minus_power = 1.0 - variance_power_;
    const double two_minus_power = 2.0 - variance_power_;
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double label = static_cast<double>(labels[i]);
      const double prediction = static_cast<double>(preds[i]);
      const double sample_weight = static_cast<double>(weights[i]);
      const double part_one =
          label <= 0.0 ? 0.0 : label * std::exp(one_minus_power * prediction) / one_minus_power;
      const double part_two = std::exp(two_minus_power * prediction) / two_minus_power;
      loss_sum += sample_weight * (-part_one + part_two);
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }

 private:
  double variance_power_{1.5};
};

}  // namespace

namespace ctboost::detail {

std::unique_ptr<MetricFunction> CreateRegressionMetric(std::string_view normalized,
                                                       const ObjectiveConfig& config) {
  if (normalized == "rmse" || normalized == "squarederror" ||
      normalized == "squared_error") {
    return std::make_unique<RMSEMetric>();
  }
  if (normalized == "mae" || normalized == "l1" || normalized == "absoluteerror" ||
      normalized == "absolute_error") {
    return std::make_unique<MAEMetric>();
  }
  if (normalized == "huber" || normalized == "huberloss") {
    return std::make_unique<HuberMetric>(config.huber_delta);
  }
  if (normalized == "quantile" || normalized == "quantileloss") {
    return std::make_unique<QuantileMetric>(config.quantile_alpha);
  }
  if (normalized == "poisson" || normalized == "poissonregression") {
    return std::make_unique<PoissonMetric>();
  }
  if (normalized == "tweedie" || normalized == "tweedieloss" ||
      normalized == "reg:tweedie") {
    return std::make_unique<TweedieMetric>(config.tweedie_variance_power);
  }
  return nullptr;
}

}  // namespace ctboost::detail
