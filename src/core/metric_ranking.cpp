#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace {

class PairLogitMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView* ranking) const override {
    const auto& resolved_group_ids =
        ctboost::detail::ValidateRankingMetricInputs(preds, labels, weights, num_classes, ranking);
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    auto accumulate_pair = [&](std::size_t winner, std::size_t loser, double pair_weight) {
      if (!(pair_weight > 0.0)) {
        return;
      }
      const double margin = static_cast<double>(preds[winner]) - preds[loser];
      loss_sum += pair_weight * std::log1p(std::exp(-margin));
      weight_sum += pair_weight;
    };

    if (ranking != nullptr && ranking->pairs != nullptr && !ranking->pairs->empty()) {
      for (const ctboost::RankingPair& pair : *ranking->pairs) {
        const std::size_t winner = static_cast<std::size_t>(pair.winner);
        const std::size_t loser = static_cast<std::size_t>(pair.loser);
        accumulate_pair(
            winner,
            loser,
            static_cast<double>(pair.weight) *
                ctboost::detail::ResolveMetricGroupWeight(ranking, winner) *
                static_cast<double>(weights[winner]) * static_cast<double>(weights[loser]));
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
            if (labels[i] == labels[j] || ctboost::detail::SameMetricSubgroup(ranking, i, j)) {
              continue;
            }

            const std::size_t winner = labels[i] > labels[j] ? i : j;
            const std::size_t loser = labels[i] > labels[j] ? j : i;
            accumulate_pair(
                winner,
                loser,
                ctboost::detail::ResolveMetricGroupWeight(ranking, winner) *
                    static_cast<double>(weights[winner]) * static_cast<double>(weights[loser]));
          }
        }
      }
    }

    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class NDCGMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView* ranking) const override {
    const auto& resolved_group_ids =
        ctboost::detail::ValidateRankingMetricInputs(preds, labels, weights, num_classes, ranking);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double ndcg_sum = 0.0;
    double group_weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      const auto& rows = entry.second;
      if (rows.size() <= 1) {
        continue;
      }

      auto gain = [&](std::size_t row_index) {
        return static_cast<double>(weights[row_index]) *
               (std::pow(2.0, static_cast<double>(labels[row_index])) - 1.0);
      };

      std::vector<std::size_t> prediction_order = rows;
      std::sort(prediction_order.begin(), prediction_order.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (preds[lhs] == preds[rhs]) {
          return lhs < rhs;
        }
        return preds[lhs] > preds[rhs];
      });

      std::vector<std::size_t> ideal_order = rows;
      std::sort(ideal_order.begin(), ideal_order.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (labels[lhs] == labels[rhs]) {
          return lhs < rhs;
        }
        return labels[lhs] > labels[rhs];
      });

      double dcg = 0.0;
      double ideal_dcg = 0.0;
      const double group_weight = ctboost::detail::ResolveMetricGroupWeight(ranking, rows.front());
      for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        const double discount = 1.0 / std::log2(static_cast<double>(rank) + 2.0);
        dcg += gain(prediction_order[rank]) * discount;
        ideal_dcg += gain(ideal_order[rank]) * discount;
      }
      if (ideal_dcg <= 0.0 || group_weight <= 0.0) {
        continue;
      }

      ndcg_sum += group_weight * (dcg / ideal_dcg);
      group_weight_sum += group_weight;
    }

    return group_weight_sum <= 0.0 ? 0.0 : ndcg_sum / group_weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class MAPMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView* ranking) const override {
    const auto& resolved_group_ids =
        ctboost::detail::ValidateRankingMetricInputs(preds, labels, weights, num_classes, ranking);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double map_sum = 0.0;
    double group_weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      std::vector<std::size_t> rows = entry.second;
      std::sort(rows.begin(), rows.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (preds[lhs] == preds[rhs]) {
          return lhs < rhs;
        }
        return preds[lhs] > preds[rhs];
      });

      double precision_sum = 0.0;
      double hit_count = 0.0;
      double relevant_weight = 0.0;
      const double group_weight = ctboost::detail::ResolveMetricGroupWeight(ranking, rows.front());
      for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        const std::size_t row = rows[rank];
        const double sample_weight = static_cast<double>(weights[row]);
        if (labels[row] > 0.0F) {
          hit_count += 1.0;
          precision_sum += hit_count / static_cast<double>(rank + 1);
          relevant_weight += sample_weight;
        }
      }
      if (relevant_weight <= 0.0 || group_weight <= 0.0) {
        continue;
      }
      map_sum += group_weight * (precision_sum / hit_count);
      group_weight_sum += group_weight;
    }
    return group_weight_sum <= 0.0 ? 0.0 : map_sum / group_weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class MRRMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView* ranking) const override {
    const auto& resolved_group_ids =
        ctboost::detail::ValidateRankingMetricInputs(preds, labels, weights, num_classes, ranking);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double reciprocal_rank_sum = 0.0;
    double group_weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      std::vector<std::size_t> rows = entry.second;
      std::sort(rows.begin(), rows.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (preds[lhs] == preds[rhs]) {
          return lhs < rhs;
        }
        return preds[lhs] > preds[rhs];
      });

      const double group_weight = ctboost::detail::ResolveMetricGroupWeight(ranking, rows.front());
      if (group_weight <= 0.0) {
        continue;
      }

      for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        if (labels[rows[rank]] > 0.0F) {
          reciprocal_rank_sum += group_weight / static_cast<double>(rank + 1);
          group_weight_sum += group_weight;
          break;
        }
      }
    }

    return group_weight_sum <= 0.0 ? 0.0 : reciprocal_rank_sum / group_weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

}  // namespace

namespace ctboost::detail {

std::unique_ptr<MetricFunction> CreateRankingMetric(std::string_view normalized,
                                                    const ObjectiveConfig&) {
  if (normalized == "pairlogit" || normalized == "pairwise" ||
      normalized == "ranknet") {
    return std::make_unique<PairLogitMetric>();
  }
  if (normalized == "ndcg") {
    return std::make_unique<NDCGMetric>();
  }
  if (normalized == "map" || normalized == "map@all") {
    return std::make_unique<MAPMetric>();
  }
  if (normalized == "mrr" || normalized == "mrr@all") {
    return std::make_unique<MRRMetric>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
