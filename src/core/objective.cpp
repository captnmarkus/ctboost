#include "ctboost/objective.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace ctboost {
namespace {

void ValidatePredictionLabelSizes(const std::vector<float>& preds,
                                  const std::vector<float>& labels) {
  if (preds.size() != labels.size()) {
    throw std::invalid_argument("predictions and labels must have the same size");
  }
}

void ValidateNonNegativeLabels(const std::vector<float>& labels, const char* objective_name) {
  for (const float label : labels) {
    if (!(label >= 0.0F) || !std::isfinite(label)) {
      throw std::invalid_argument(std::string(objective_name) +
                                  " requires finite non-negative labels");
    }
  }
}

struct SurvivalLabel {
  double time{0.0};
  bool event{false};
};

std::vector<SurvivalLabel> ParseSignedSurvivalLabels(const std::vector<float>& labels,
                                                     const char* objective_name) {
  std::vector<SurvivalLabel> parsed(labels.size());
  bool saw_event = false;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double signed_time = static_cast<double>(labels[index]);
    const double absolute_time = std::fabs(signed_time);
    if (!std::isfinite(signed_time) || absolute_time <= 0.0) {
      throw std::invalid_argument(std::string(objective_name) +
                                  " expects signed survival times with non-zero magnitude");
    }
    parsed[index] = SurvivalLabel{absolute_time, signed_time > 0.0};
    saw_event = saw_event || parsed[index].event;
  }
  if (!saw_event) {
    throw std::invalid_argument(std::string(objective_name) + " requires at least one observed event");
  }
  return parsed;
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

const std::vector<std::int64_t>& ValidateRankingInputs(
    const std::vector<float>& preds,
    const std::vector<float>& labels,
    int num_classes,
    const RankingMetadataView* ranking) {
  if (num_classes != 1) {
    throw std::invalid_argument("ranking objectives expect num_classes equal to one");
  }
  ValidatePredictionLabelSizes(preds, labels);
  if (ranking == nullptr || ranking->group_ids == nullptr || ranking->group_ids->empty()) {
    throw std::invalid_argument("ranking objectives require group_id values");
  }
  if (ranking->group_ids->size() != labels.size()) {
    throw std::invalid_argument("group_id size must match the number of labels");
  }
  return *ranking->group_ids;
}

float ResolveGroupWeight(const RankingMetadataView* ranking, std::size_t row) {
  if (ranking == nullptr || ranking->group_weights == nullptr || ranking->group_weights->empty()) {
    return 1.0F;
  }
  return (*ranking->group_weights)[row];
}

bool SameSubgroup(const RankingMetadataView* ranking, std::size_t left, std::size_t right) {
  if (ranking == nullptr || ranking->subgroup_ids == nullptr || ranking->subgroup_ids->empty()) {
    return false;
  }
  return (*ranking->subgroup_ids)[left] == (*ranking->subgroup_ids)[right];
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
                                     int num_classes,
                                     const RankingMetadataView*) const {
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
                                int num_classes,
                                const RankingMetadataView*) const {
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
                                    int num_classes,
                                    const RankingMetadataView*) const {
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

void AbsoluteError::compute_gradients(const std::vector<float>& preds,
                                      const std::vector<float>& labels,
                                      std::vector<float>& out_g,
                                      std::vector<float>& out_h,
                                      int num_classes,
                                      const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("absolute error expects num_classes equal to one");
  }
  ValidatePredictionLabelSizes(preds, labels);

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
  ValidatePredictionLabelSizes(preds, labels);

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
  ValidatePredictionLabelSizes(preds, labels);

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
  ValidatePredictionLabelSizes(preds, labels);
  ValidateNonNegativeLabels(labels, "poisson loss");

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
  ValidatePredictionLabelSizes(preds, labels);
  ValidateNonNegativeLabels(labels, "tweedie loss");

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

void CoxLoss::compute_gradients(const std::vector<float>& preds,
                                const std::vector<float>& labels,
                                std::vector<float>& out_g,
                                std::vector<float>& out_h,
                                int num_classes,
                                const RankingMetadataView*) const {
  if (num_classes != 1) {
    throw std::invalid_argument("cox loss expects num_classes equal to one");
  }
  ValidatePredictionLabelSizes(preds, labels);
  const std::vector<SurvivalLabel> survival = ParseSignedSurvivalLabels(labels, "cox loss");

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
  ValidatePredictionLabelSizes(preds, labels);
  const std::vector<SurvivalLabel> survival =
      ParseSignedSurvivalLabels(labels, "survival exponential loss");

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

void PairLogitLoss::compute_gradients(const std::vector<float>& preds,
                                      const std::vector<float>& labels,
                                      std::vector<float>& out_g,
                                      std::vector<float>& out_h,
                                      int num_classes,
                                      const RankingMetadataView* ranking) const {
  const auto& resolved_group_ids =
      ValidateRankingInputs(preds, labels, num_classes, ranking);

  out_g.assign(preds.size(), 0.0F);
  out_h.assign(preds.size(), 0.0F);

  std::size_t pair_count = 0;
  auto apply_pair = [&](std::size_t winner, std::size_t loser, float pair_weight) {
    if (!(pair_weight > 0.0F)) {
      return;
    }
    const float margin = preds[winner] - preds[loser];
    const float probability = Sigmoid(margin);
    const float gradient = (probability - 1.0F) * pair_weight;
    const float hessian = probability * (1.0F - probability) * pair_weight;

    out_g[winner] += gradient;
    out_g[loser] -= gradient;
    out_h[winner] += hessian;
    out_h[loser] += hessian;
    ++pair_count;
  };

  if (ranking != nullptr && ranking->pairs != nullptr && !ranking->pairs->empty()) {
    for (const RankingPair& pair : *ranking->pairs) {
      const std::size_t winner = static_cast<std::size_t>(pair.winner);
      const std::size_t loser = static_cast<std::size_t>(pair.loser);
      apply_pair(winner, loser, pair.weight * ResolveGroupWeight(ranking, winner));
    }
  } else {
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    for (const auto& entry : group_rows) {
      const auto& rows = entry.second;
      for (std::size_t left = 0; left < rows.size(); ++left) {
        for (std::size_t right = left + 1; right < rows.size(); ++right) {
          const std::size_t i = rows[left];
          const std::size_t j = rows[right];
          if (labels[i] == labels[j] || SameSubgroup(ranking, i, j)) {
            continue;
          }

          const std::size_t winner = labels[i] > labels[j] ? i : j;
          const std::size_t loser = labels[i] > labels[j] ? j : i;
          apply_pair(winner, loser, ResolveGroupWeight(ranking, winner));
        }
      }
    }
  }

  if (pair_count == 0) {
    throw std::invalid_argument("ranking objective requires at least one comparable pair");
  }
}

std::unique_ptr<ObjectiveFunction> CreateObjectiveFunction(std::string_view name,
                                                           const ObjectiveConfig& config) {
  const std::string normalized = NormalizeName(name);

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
  if (normalized == "cox" || normalized == "coxph" || normalized == "survival:cox") {
    return std::make_unique<CoxLoss>();
  }
  if (normalized == "survivalexponential" || normalized == "survival_exp" ||
      normalized == "survival:exponential") {
    return std::make_unique<SurvivalExponentialLoss>();
  }
  if (normalized == "logloss" || normalized == "binary_logloss" ||
      normalized == "binary:logistic") {
    return std::make_unique<LogLoss>();
  }
  if (normalized == "multiclass" || normalized == "softmax" ||
      normalized == "softmaxloss") {
    return std::make_unique<SoftmaxLoss>();
  }
  if (normalized == "pairlogit" || normalized == "pairwise" ||
      normalized == "ranknet") {
    return std::make_unique<PairLogitLoss>();
  }

  throw std::invalid_argument("unknown objective function: " + std::string(name));
}

}  // namespace ctboost
