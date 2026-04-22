#include "objective_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ctboost {

void LogLoss::compute_gradients(const std::vector<float>& preds,
                                const std::vector<float>& labels,
                                std::vector<float>& out_g,
                                std::vector<float>& out_h,
                                int num_classes,
                                const RankingMetadataView*) const {
  if (num_classes != 1 && num_classes != 2) {
    throw std::invalid_argument("logloss expects num_classes equal to one or two");
  }
  detail::ValidatePredictionLabelSizes(preds, labels);

  out_g.resize(preds.size());
  out_h.resize(preds.size());
  for (std::size_t i = 0; i < preds.size(); ++i) {
    const float probability = detail::Sigmoid(preds[i]);
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
  detail::ValidateMulticlassSizes(preds, labels, num_classes);

  out_g.resize(preds.size());
  out_h.resize(preds.size());
  for (std::size_t row = 0; row < labels.size(); ++row) {
    const std::size_t row_offset = row * static_cast<std::size_t>(num_classes);
    const auto row_begin = preds.begin() + static_cast<std::ptrdiff_t>(row_offset);
    const auto row_end = row_begin + num_classes;
    const float max_logit = *std::max_element(row_begin, row_end);

    double exp_sum = 0.0;
    for (int class_index = 0; class_index < num_classes; ++class_index) {
      exp_sum += std::exp(static_cast<double>(preds[row_offset + class_index] - max_logit));
    }

    const int target_class = detail::LabelToClassIndex(labels[row], num_classes);
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

}  // namespace ctboost

namespace ctboost::detail {

std::unique_ptr<ObjectiveFunction> CreateClassificationObjective(std::string_view normalized,
                                                                 const ObjectiveConfig&) {
  if (normalized == "logloss" || normalized == "binary_logloss" ||
      normalized == "binary:logistic") {
    return std::make_unique<LogLoss>();
  }
  if (normalized == "multiclass" || normalized == "softmax" ||
      normalized == "softmaxloss") {
    return std::make_unique<SoftmaxLoss>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
