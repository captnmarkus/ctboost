#pragma once

#include <cstdint>
#include <vector>

namespace ctboost {

struct RankingPair {
  std::int64_t winner{0};
  std::int64_t loser{0};
  float weight{1.0F};
};

struct RankingMetadataView {
  const std::vector<std::int64_t>* group_ids{nullptr};
  const std::vector<std::int64_t>* subgroup_ids{nullptr};
  const std::vector<float>* group_weights{nullptr};
  const std::vector<RankingPair>* pairs{nullptr};
};

}  // namespace ctboost
