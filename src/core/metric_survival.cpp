#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

class CoxMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    const std::vector<ctboost::detail::MetricSurvivalLabel> survival =
        ctboost::detail::ParseSignedMetricSurvivalLabels(labels, "cox metric");

    std::vector<std::size_t> order(preds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (survival[lhs].time == survival[rhs].time) {
        return survival[lhs].event && !survival[rhs].event;
      }
      return survival[lhs].time > survival[rhs].time;
    });

    std::vector<double> exp_preds(preds.size(), 0.0);
    for (std::size_t index = 0; index < preds.size(); ++index) {
      exp_preds[index] = std::exp(static_cast<double>(preds[index]));
    }

    double loss_sum = 0.0;
    double weight_sum = 0.0;
    double risk_sum = 0.0;
    std::size_t position = 0;
    while (position < order.size()) {
      const double group_time = survival[order[position]].time;
      std::size_t group_end = position;
      double event_weight = 0.0;
      while (group_end < order.size() && survival[order[group_end]].time == group_time) {
        risk_sum += exp_preds[order[group_end]];
        if (survival[order[group_end]].event) {
          event_weight += static_cast<double>(weights[order[group_end]]);
        }
        ++group_end;
      }
      if (event_weight > 0.0) {
        const double log_risk = std::log(std::max(risk_sum, 1e-12));
        for (std::size_t group_position = position; group_position < group_end; ++group_position) {
          const std::size_t row = order[group_position];
          if (!survival[row].event) {
            continue;
          }
          const double sample_weight = static_cast<double>(weights[row]);
          loss_sum += sample_weight * (log_risk - static_cast<double>(preds[row]));
          weight_sum += sample_weight;
        }
      }
      position = group_end;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class SurvivalExponentialMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    const std::vector<ctboost::detail::MetricSurvivalLabel> survival =
        ctboost::detail::ParseSignedMetricSurvivalLabels(labels, "survival exponential metric");

    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < preds.size(); ++index) {
      const double hazard = std::exp(static_cast<double>(preds[index]));
      const double event = survival[index].event ? 1.0 : 0.0;
      const double sample_weight = static_cast<double>(weights[index]);
      loss_sum +=
          sample_weight * (hazard * survival[index].time - event * static_cast<double>(preds[index]));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class ConcordanceIndexMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    const std::vector<ctboost::detail::MetricSurvivalLabel> survival =
        ctboost::detail::ParseSignedMetricSurvivalLabels(labels, "cindex");

    double concordant = 0.0;
    double comparable = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      if (!survival[i].event) {
        continue;
      }
      for (std::size_t j = 0; j < preds.size(); ++j) {
        if (i == j) {
          continue;
        }
        if (survival[j].time <= survival[i].time) {
          continue;
        }
        const double pair_weight =
            static_cast<double>(weights[i]) * static_cast<double>(weights[j]);
        if (pair_weight <= 0.0) {
          continue;
        }
        comparable += pair_weight;
        if (preds[i] > preds[j]) {
          concordant += pair_weight;
        } else if (preds[i] == preds[j]) {
          concordant += 0.5 * pair_weight;
        }
      }
    }
    return comparable <= 0.0 ? 0.0 : concordant / comparable;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

}  // namespace

namespace ctboost::detail {

std::unique_ptr<MetricFunction> CreateSurvivalMetric(std::string_view normalized,
                                                     const ObjectiveConfig&) {
  if (normalized == "cox" || normalized == "coxph" || normalized == "survival:cox") {
    return std::make_unique<CoxMetric>();
  }
  if (normalized == "survivalexponential" || normalized == "survival_exp" ||
      normalized == "survival:exponential") {
    return std::make_unique<SurvivalExponentialMetric>();
  }
  if (normalized == "cindex" || normalized == "concordance_index") {
    return std::make_unique<ConcordanceIndexMetric>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
