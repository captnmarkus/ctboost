#include "objective_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ctboost {

void CoxLoss::compute_gradients(const std::vector<float>& preds,
                                const std::vector<float>& labels,
                                std::vector<float>& out_g,
                                std::vector<float>& out_h,
                                int num_classes,
                                const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("cox loss expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);
  const std::vector<detail::SurvivalLabel> survival =
      detail::ParseSignedSurvivalLabels(labels, "cox loss");

  std::vector<std::size_t> order(preds.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
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

  std::vector<double> suffix_event_scale(preds.size(), 0.0);
  std::vector<double> suffix_event_square_scale(preds.size(), 0.0);
  std::vector<std::size_t> group_begin_for_position(preds.size(), 0);
  std::size_t position = 0;
  double risk_sum = 0.0;
  while (position < order.size()) {
    const double group_time = survival[order[position]].time;
    const std::size_t group_begin = position;
    std::size_t group_end = position;
    std::size_t event_count = 0;
    while (group_end < order.size() && survival[order[group_end]].time == group_time) {
      risk_sum += exp_preds[order[group_end]];
      if (survival[order[group_end]].event) {
        ++event_count;
      }
      ++group_end;
    }
    if (event_count > 0) {
      const double scale = static_cast<double>(event_count) / std::max(risk_sum, 1e-12);
      const double square_scale =
          static_cast<double>(event_count) / std::max(risk_sum * risk_sum, 1e-12);
      suffix_event_scale[group_begin] = scale;
      suffix_event_square_scale[group_begin] = square_scale;
    }
    for (std::size_t group_position = group_begin; group_position < group_end; ++group_position) {
      group_begin_for_position[group_position] = group_begin;
    }
    position = group_end;
  }

  for (std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(order.size()) - 2; offset >= 0; --offset) {
    suffix_event_scale[static_cast<std::size_t>(offset)] +=
        suffix_event_scale[static_cast<std::size_t>(offset + 1)];
    suffix_event_square_scale[static_cast<std::size_t>(offset)] +=
        suffix_event_square_scale[static_cast<std::size_t>(offset + 1)];
  }

  for (std::size_t sorted_position = 0; sorted_position < order.size(); ++sorted_position) {
    const std::size_t group_begin = group_begin_for_position[sorted_position];
    suffix_event_scale[sorted_position] = suffix_event_scale[group_begin];
    suffix_event_square_scale[sorted_position] = suffix_event_square_scale[group_begin];
  }

  out_g.assign(preds.size(), 0.0F);
  out_h.assign(preds.size(), 0.0F);
  for (std::size_t sorted_position = 0; sorted_position < order.size(); ++sorted_position) {
    const std::size_t row = order[sorted_position];
    const double exp_pred = exp_preds[row];
    const double scale = suffix_event_scale[sorted_position];
    const double square_scale = suffix_event_square_scale[sorted_position];
    const double gradient = exp_pred * scale - (survival[row].event ? 1.0 : 0.0);
    const double hessian = std::max(exp_pred * scale - exp_pred * exp_pred * square_scale, 1e-12);
    out_g[row] = static_cast<float>(gradient);
    out_h[row] = static_cast<float>(hessian);
  }
}

void SurvivalExponentialLoss::compute_gradients(const std::vector<float>& preds,
                                                const std::vector<float>& labels,
                                                std::vector<float>& out_g,
                                                std::vector<float>& out_h,
                                                int num_classes,
                                                const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("survival exponential loss expects num_classes equal to one");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);
  const std::vector<detail::SurvivalLabel> survival =
      detail::ParseSignedSurvivalLabels(labels, "survival exponential loss");

  out_g.resize(preds.size());
  out_h.resize(preds.size());
  for (std::size_t index = 0; index < preds.size(); ++index) {
    const double hazard = std::exp(static_cast<double>(preds[index]));
    const double time = survival[index].time;
    const double event = survival[index].event ? 1.0 : 0.0;
    out_g[index] = static_cast<float>(hazard * time - event);
    out_h[index] = static_cast<float>(std::max(hazard * time, 1e-12));
  }
}

}  // namespace ctboost

namespace ctboost::detail {

std::unique_ptr<ObjectiveFunction> CreateSurvivalObjective(std::string_view normalized,
                                                           const ObjectiveConfig&) {
  if (normalized == "cox" || normalized == "coxph" || normalized == "survival:cox") {
    return std::make_unique<CoxLoss>();
  }
  if (normalized == "survivalexponential" || normalized == "survival_exp" ||
      normalized == "survival:exponential") {
    return std::make_unique<SurvivalExponentialLoss>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
