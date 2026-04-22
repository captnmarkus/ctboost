#include "objective_internal.hpp"

#include <stdexcept>
#include <unordered_map>

namespace ctboost {

void PairLogitLoss::compute_gradients(const std::vector<float>& preds,
                                      const std::vector<float>& labels,
                                      std::vector<float>& out_g,
                                      std::vector<float>& out_h,
                                      int num_classes,
                                      const RankingMetadataView* ranking) const {
  const auto& resolved_group_ids =
      detail::ValidateRankingInputs(preds, labels, num_classes, ranking);

  out_g.assign(preds.size(), 0.0F);
  out_h.assign(preds.size(), 0.0F);

  std::size_t pair_count = 0;
  auto apply_pair = [&](std::size_t winner, std::size_t loser, float pair_weight) {
    if (!(pair_weight > 0.0F)) {
      return;
    }
    const float margin = preds[winner] - preds[loser];
    const float probability = detail::Sigmoid(margin);
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
      apply_pair(winner, loser, pair.weight * detail::ResolveGroupWeight(ranking, winner));
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
          if (labels[i] == labels[j] || detail::SameSubgroup(ranking, i, j)) {
            continue;
          }

          const std::size_t winner = labels[i] > labels[j] ? i : j;
          const std::size_t loser = labels[i] > labels[j] ? j : i;
          apply_pair(winner, loser, detail::ResolveGroupWeight(ranking, winner));
        }
      }
    }
  }

  if (pair_count == 0) {
    throw std::invalid_argument("ranking objective requires at least one comparable pair");
  }
}

}  // namespace ctboost

namespace ctboost::detail {

std::unique_ptr<ObjectiveFunction> CreateRankingObjective(std::string_view normalized,
                                                          const ObjectiveConfig&) {
  if (normalized == "pairlogit" || normalized == "pairwise" ||
      normalized == "ranknet") {
    return std::make_unique<PairLogitLoss>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
