#include "objective_internal.hpp"

#include <cctype>
#include <cmath>
#include <stdexcept>

namespace ctboost::detail {

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

std::string NormalizeObjectiveName(std::string_view name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (const char ch : name) {
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return normalized;
}

}  // namespace ctboost::detail
