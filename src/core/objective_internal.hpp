#pragma once

#include "ctboost/objective.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace ctboost::detail {

void ValidatePredictionLabelSizes(const std::vector<float>& preds,
                                  const std::vector<float>& labels);
void ValidateNonNegativeLabels(const std::vector<float>& labels, const char* objective_name);

struct SurvivalLabel {
  double time{0.0};
  bool event{false};
};

std::vector<SurvivalLabel> ParseSignedSurvivalLabels(const std::vector<float>& labels,
                                                     const char* objective_name);
float Sigmoid(float margin);
int LabelToClassIndex(float label, int num_classes);
void ValidateMulticlassSizes(const std::vector<float>& preds,
                             const std::vector<float>& labels,
                             int num_classes);
const std::vector<std::int64_t>& ValidateRankingInputs(
    const std::vector<float>& preds,
    const std::vector<float>& labels,
    int num_classes,
    const RankingMetadataView* ranking);
float ResolveGroupWeight(const RankingMetadataView* ranking, std::size_t row);
bool SameSubgroup(const RankingMetadataView* ranking, std::size_t left, std::size_t right);
std::string NormalizeObjectiveName(std::string_view name);

std::unique_ptr<ObjectiveFunction> CreateRegressionObjective(std::string_view normalized,
                                                             const ObjectiveConfig& config);
std::unique_ptr<ObjectiveFunction> CreateClassificationObjective(std::string_view normalized,
                                                                 const ObjectiveConfig& config);
std::unique_ptr<ObjectiveFunction> CreateSurvivalObjective(std::string_view normalized,
                                                           const ObjectiveConfig& config);
std::unique_ptr<ObjectiveFunction> CreateRankingObjective(std::string_view normalized,
                                                          const ObjectiveConfig& config);

}  // namespace ctboost::detail
