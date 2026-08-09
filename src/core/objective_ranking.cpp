#include "objective_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

void LambdaMARTLoss::compute_gradients(const std::vector<float>& preds,
                                       const std::vector<float>& labels,
                                       std::vector<float>& out_g,
                                       std::vector<float>& out_h,
                                       int num_classes,
                                       const RankingMetadataView* ranking) const {
  const auto& resolved_group_ids =
      detail::ValidateRankingInputs(preds, labels, num_classes, ranking);
  for (const float label : labels) {
    if (!(label >= 0.0F) || !std::isfinite(label)) {
      throw std::invalid_argument("LambdaMART requires finite non-negative relevance labels");
    }
  }

  out_g.assign(preds.size(), 0.0F);
  out_h.assign(preds.size(), 0.0F);

  struct QueryState {
    std::vector<std::size_t> rows;
    std::unordered_map<std::size_t, std::size_t> predicted_rank;
    double ideal_dcg{0.0};
  };
  std::unordered_map<std::int64_t, QueryState> queries;
  queries.reserve(resolved_group_ids.size());
  for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
    queries[resolved_group_ids[row]].rows.push_back(row);
  }

  auto gain = [&labels](std::size_t row) {
    const double result = std::exp2(static_cast<double>(labels[row])) - 1.0;
    if (!std::isfinite(result)) {
      throw std::invalid_argument("LambdaMART relevance labels produce non-finite NDCG gains");
    }
    return result;
  };
  auto discount = [](std::size_t rank) {
    return 1.0 / std::log2(static_cast<double>(rank) + 2.0);
  };

  for (auto& entry : queries) {
    QueryState& query = entry.second;
    std::vector<std::size_t> predicted_order = query.rows;
    std::sort(predicted_order.begin(), predicted_order.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (preds[lhs] == preds[rhs]) {
        return lhs < rhs;
      }
      return preds[lhs] > preds[rhs];
    });
    query.predicted_rank.reserve(predicted_order.size());
    for (std::size_t rank = 0; rank < predicted_order.size(); ++rank) {
      query.predicted_rank.emplace(predicted_order[rank], rank);
    }

    std::vector<std::size_t> ideal_order = query.rows;
    std::sort(ideal_order.begin(), ideal_order.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (labels[lhs] == labels[rhs]) {
        return lhs < rhs;
      }
      return labels[lhs] > labels[rhs];
    });
    for (std::size_t rank = 0; rank < ideal_order.size(); ++rank) {
      query.ideal_dcg += gain(ideal_order[rank]) * discount(rank);
    }
  }

  std::size_t pair_count = 0;
  auto apply_pair = [&](std::size_t winner, std::size_t loser, double base_weight) {
    if (!(base_weight > 0.0) || labels[winner] <= labels[loser] ||
        detail::SameSubgroup(ranking, winner, loser)) {
      return;
    }
    QueryState& query = queries.at(resolved_group_ids[winner]);
    if (!(query.ideal_dcg > 0.0)) {
      return;
    }
    const std::size_t winner_rank = query.predicted_rank.at(winner);
    const std::size_t loser_rank = query.predicted_rank.at(loser);
    const double delta_ndcg =
        std::fabs((gain(winner) - gain(loser)) *
                  (discount(winner_rank) - discount(loser_rank))) /
        query.ideal_dcg;
    const double pair_weight = base_weight * delta_ndcg;
    if (!(pair_weight > 0.0) || !std::isfinite(pair_weight)) {
      return;
    }

    const float probability = detail::Sigmoid(preds[winner] - preds[loser]);
    const float gradient = static_cast<float>((static_cast<double>(probability) - 1.0) * pair_weight);
    const float hessian = static_cast<float>(
        static_cast<double>(probability) * (1.0 - static_cast<double>(probability)) * pair_weight);
    out_g[winner] += gradient;
    out_g[loser] -= gradient;
    out_h[winner] += hessian;
    out_h[loser] += hessian;
    ++pair_count;
  };

  if (ranking != nullptr && ranking->pairs != nullptr && !ranking->pairs->empty()) {
    for (const RankingPair& pair : *ranking->pairs) {
      const std::size_t first = static_cast<std::size_t>(pair.winner);
      const std::size_t second = static_cast<std::size_t>(pair.loser);
      if (labels[first] == labels[second]) {
        continue;
      }
      const std::size_t winner = labels[first] > labels[second] ? first : second;
      const std::size_t loser = labels[first] > labels[second] ? second : first;
      apply_pair(
          winner,
          loser,
          static_cast<double>(pair.weight) * detail::ResolveGroupWeight(ranking, winner));
    }
  } else {
    for (const auto& entry : queries) {
      const auto& rows = entry.second.rows;
      for (std::size_t left = 0; left < rows.size(); ++left) {
        for (std::size_t right = left + 1; right < rows.size(); ++right) {
          const std::size_t i = rows[left];
          const std::size_t j = rows[right];
          if (labels[i] == labels[j]) {
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
    throw std::invalid_argument("LambdaMART requires at least one relevance-changing pair");
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
  if (normalized == "lambdamart" || normalized == "lambdarank" ||
      normalized == "rank:ndcg") {
    return std::make_unique<LambdaMARTLoss>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
