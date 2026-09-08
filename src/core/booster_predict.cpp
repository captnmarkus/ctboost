#include "booster_internal.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace ctboost::booster_detail {

// Keep training's Node representation and serialized node indices intact. The
// hot traversal needs neither its vector object nor 256 categorical route bytes.
struct PredictionNode {
  std::size_t payload_index{0};
  int split_feature_id{-1};
  int left_child{-1};
  int right_child{-1};
  float leaf_update{0.0F};
  std::uint16_t split_bin_index{0};
  bool is_categorical_split{false};
};
static_assert(sizeof(PredictionNode) <= 32, "prediction nodes must remain compact");

struct PredictionTree {
  std::vector<PredictionNode> nodes;
  std::vector<std::array<std::uint64_t, 4>> categorical_routes;
  std::vector<float> vector_updates;
  bool vector_leaves{false};
  std::size_t consumed_trees{1};
};

struct PredictionCache {
  std::vector<PredictionTree> trees;
  // Prefix prediction must not quantize a feature used only by later trees (in
  // particular, nan_mode=Forbidden should retain its existing prefix behavior).
  std::vector<std::size_t> first_feature_tree;

  std::vector<std::uint8_t> ActiveFeatures(std::size_t tree_limit) const {
    std::vector<std::uint8_t> active(first_feature_tree.size());
    for (std::size_t feature = 0; feature < active.size(); ++feature) {
      active[feature] = first_feature_tree[feature] < tree_limit ? 1U : 0U;
    }
    return active;
  }
};

namespace {

// GNU can contract a multiply and a subsequent += across statements when the
// target has FMA. A preweighted cache forces an extra rounding in that case.
// Retain the original arithmetic there; do not alter training or global flags.
#if defined(__FAST_MATH__) || defined(_M_FP_FAST) || \
    (defined(__GNUC__) && !defined(__clang__) && \
     (defined(__FMA__) || defined(__FMA4__) || defined(__ARM_FEATURE_FMA) || \
      defined(__aarch64__) || defined(__FP_FAST_FMAF)))
constexpr bool kUseLegacyPredictionArithmetic = true;
#else
constexpr bool kUseLegacyPredictionArithmetic = false;
#endif

bool SamePredictionTopology(const Tree& left, const Tree& right) {
  if (left.nodes().size() != right.nodes().size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.nodes().size(); ++index) {
    const Node& a = left.nodes()[index];
    const Node& b = right.nodes()[index];
    if (a.is_leaf != b.is_leaf || a.is_categorical_split != b.is_categorical_split ||
        a.split_feature_id != b.split_feature_id || a.split_bin_index != b.split_bin_index ||
        a.left_child != b.left_child || a.right_child != b.right_child ||
        a.left_categories != b.left_categories) {
      return false;
    }
  }
  return true;
}

template <typename BinType>
int CachedLeafIndex(const PredictionTree& tree,
                    const BinType* bins,
                    std::size_t num_rows,
                    std::size_t row) noexcept {
  int node_index = 0;
  while (tree.nodes[static_cast<std::size_t>(node_index)].split_feature_id >= 0) {
    const PredictionNode& node = tree.nodes[static_cast<std::size_t>(node_index)];
    const std::uint16_t bin = bins[static_cast<std::size_t>(node.split_feature_id) * num_rows + row];
    const bool go_left = node.is_categorical_split
        ? ((tree.categorical_routes[node.payload_index][bin / 64U] >> (bin % 64U)) & 1U) != 0U
        : bin <= node.split_bin_index;
    node_index = go_left ? node.left_child : node.right_child;
  }
  return node_index;
}

template <typename BinType>
void PredictCached(const PredictionCache& cache,
                   const BinType* bins,
                   std::size_t num_rows,
                   std::size_t tree_limit,
                   std::size_t dimension,
                   std::vector<float>& predictions) {
  for (std::size_t tree_index = 0; tree_index < tree_limit;) {
    const PredictionTree& tree = cache.trees[tree_index];
    if (tree.nodes.empty()) {
      ++tree_index;
      continue;
    }
    const bool vector_updates = tree.vector_leaves && tree.consumed_trees <= tree_limit - tree_index;
    const std::size_t class_index = tree_index % dimension;
    const PredictionNode& root = tree.nodes.front();
    if (root.split_feature_id < 0) {
      // Conditional inference can stop at the root. Keep every tree's float
      // addition (including zero) while avoiding traversal and row dispatch.
      if (vector_updates) {
        const float* updates = tree.vector_updates.data() + root.payload_index;
        for (std::size_t offset = 0; offset < predictions.size(); offset += dimension) {
          for (std::size_t output = 0; output < dimension; ++output) {
            predictions[offset + output] += updates[output];
          }
        }
      } else if (dimension == 1U) {
        const float update = root.leaf_update;
        for (float& prediction : predictions) {
          prediction += update;
        }
      } else {
        const float update = root.leaf_update;
        for (std::size_t offset = class_index; offset < predictions.size(); offset += dimension) {
          predictions[offset] += update;
        }
      }
      tree_index += vector_updates ? tree.consumed_trees : 1U;
      continue;
    }
    const auto add_leaf = [&](int leaf_index, std::size_t row) {
      const PredictionNode& leaf = tree.nodes[static_cast<std::size_t>(leaf_index)];
      if (vector_updates) {
        for (std::size_t output = 0; output < dimension; ++output) {
          predictions[row * dimension + output] += tree.vector_updates[leaf.payload_index + output];
        }
      } else {
        predictions[row * dimension + class_index] += leaf.leaf_update;
      }
    };
    for (std::size_t row = 0; row < num_rows; ++row) {
      add_leaf(CachedLeafIndex(tree, bins, num_rows, row), row);
    }
    tree_index += vector_updates ? tree.consumed_trees : 1U;
  }
}

template <typename BinType>
void PredictCachedLeaves(const PredictionCache& cache,
                         const BinType* bins,
                         std::size_t num_rows,
                         std::size_t tree_limit,
                         std::vector<std::int32_t>& leaf_indices) {
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const PredictionTree& tree = cache.trees[tree_index];
    if (tree.nodes.empty()) {
      continue;
    }
    for (std::size_t row = 0; row < num_rows; ++row) {
      leaf_indices[row * tree_limit + tree_index] = CachedLeafIndex(tree, bins, num_rows, row);
    }
  }
}

}  // namespace
}  // namespace ctboost::booster_detail

namespace ctboost {

void GradientBooster::InvalidatePredictionCache() noexcept {
  std::atomic_store(&prediction_cache_, std::shared_ptr<const booster_detail::PredictionCache>{});
}

std::shared_ptr<const booster_detail::PredictionCache> GradientBooster::GetPredictionCache() const {
  if (prediction_cache_enabled_) {
    const auto cached = std::atomic_load(&prediction_cache_);
    if (cached != nullptr) {
      return cached;
    }
  }
  auto cache = std::make_shared<booster_detail::PredictionCache>();
  cache->first_feature_tree.assign(
      booster_detail::RequireQuantizationSchema(quantization_schema_).num_cols(),
      std::numeric_limits<std::size_t>::max());
  cache->trees.reserve(trees_.size());
  for (std::size_t tree_index = 0; tree_index < trees_.size(); ++tree_index) {
    const Tree& tree = trees_[tree_index];
    booster_detail::PredictionTree cached_tree;
    cached_tree.vector_leaves = tree.is_vector_leaf();
    cached_tree.nodes.reserve(tree.nodes().size());
    const float learning_rate = static_cast<float>(booster_detail::ResolveIterationLearningRate(
        tree_learning_rates_, tree_index, trees_per_iteration(), learning_rate_));
    for (const Node& node : tree.nodes()) {
      booster_detail::PredictionNode cached_node;
      if (node.is_leaf) {
        // The existing CPU path rounds the float product before adding it to
        // each float prediction. Storing that product preserves this order.
        cached_node.leaf_update = learning_rate * node.leaf_weight;
        if (cached_tree.vector_leaves) {
          cached_node.payload_index = cached_tree.vector_updates.size();
          for (const float weight : node.leaf_weights) {
            cached_tree.vector_updates.push_back(learning_rate * weight);
          }
        }
      } else {
        cached_node.split_feature_id = node.split_feature_id;
        cached_node.split_bin_index = node.split_bin_index;
        cached_node.left_child = node.left_child;
        cached_node.right_child = node.right_child;
        cached_node.is_categorical_split = node.is_categorical_split;
        if (node.split_feature_id >= 0 &&
            static_cast<std::size_t>(node.split_feature_id) < cache->first_feature_tree.size()) {
          auto& first_tree = cache->first_feature_tree[static_cast<std::size_t>(node.split_feature_id)];
          first_tree = std::min(first_tree, tree_index);
        }
        if (node.is_categorical_split) {
          std::array<std::uint64_t, 4> routes{};
          for (std::size_t bin = 0; bin < node.left_categories.size(); ++bin) {
            if (node.left_categories[bin] != 0U) {
              routes[bin / 64U] |= std::uint64_t{1} << (bin % 64U);
            }
          }
          cached_node.payload_index = cached_tree.categorical_routes.size();
          cached_tree.categorical_routes.push_back(routes);
        }
      }
      cached_tree.nodes.push_back(cached_node);
    }
    cache->trees.push_back(std::move(cached_tree));
  }
  if (multi_strategy_ == "one_output_per_tree" && prediction_dimension_ > 1) {
    const std::size_t dimension = static_cast<std::size_t>(prediction_dimension_);
    // Scalar multiclass training can store K copies of the same topology.
    // Fuse only complete, exactly matching iterations in the inference cache;
    // physical trees and their leaf-index columns stay unchanged.
    for (std::size_t first = 0; first + dimension <= trees_.size(); first += dimension) {
      bool same_topology = !trees_[first].nodes().empty();
      for (std::size_t output = 1; same_topology && output < dimension; ++output) {
        same_topology = booster_detail::SamePredictionTopology(trees_[first], trees_[first + output]);
      }
      if (!same_topology) {
        continue;
      }
      auto& fused = cache->trees[first];
      fused.vector_leaves = true;
      fused.consumed_trees = dimension;
      for (std::size_t index = 0; index < fused.nodes.size(); ++index) {
        auto& node = fused.nodes[index];
        if (node.split_feature_id < 0) {
          node.payload_index = fused.vector_updates.size();
          for (std::size_t output = 0; output < dimension; ++output) {
            fused.vector_updates.push_back(cache->trees[first + output].nodes[index].leaf_update);
          }
        }
      }
    }
  }
  std::shared_ptr<const booster_detail::PredictionCache> result = std::move(cache);
  if (prediction_cache_enabled_) {
    // Concurrent read-only C++ predictions may build equivalent local caches;
    // publish an immutable snapshot without a shared mutable traversal buffer.
    std::atomic_store(&prediction_cache_, result);
  }
  return result;
}

std::vector<float> GradientBooster::Predict(const Pool& pool, int num_iteration) const {
  return PredictImpl(pool, num_iteration, true);
}

std::vector<float> GradientBooster::PredictUncached(const Pool& pool, int num_iteration) const {
  return PredictImpl(pool, num_iteration, false);
}

std::vector<float> GradientBooster::PredictImpl(const Pool& pool,
                                               int num_iteration,
                                               bool use_cache) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(trees_per_iteration()));
  }

  std::vector<float> predictions(
      pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  if (tree_limit == 0) {
    booster_detail::AddBaseScoreToPredictions(base_score_, prediction_dimension_, predictions);
    booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }
  const auto& quantization_schema =
      booster_detail::RequireQuantizationSchema(quantization_schema_);
  const bool use_cached_prediction =
      use_cache && !use_gpu_ && !booster_detail::kUseLegacyPredictionArithmetic;
  const auto cache = use_cached_prediction
      ? GetPredictionCache() : std::shared_ptr<const booster_detail::PredictionCache>{};
  auto active_features = cache != nullptr
      ? cache->ActiveFeatures(tree_limit) : std::vector<std::uint8_t>(quantization_schema.num_cols(), 0U);
  if (cache == nullptr) {
    for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
      booster_detail::MarkUsedFeatures(trees_[tree_index], active_features);
    }
  }
  const HistMatrix hist = booster_detail::BuildPredictionHist(
      pool,
      quantization_schema,
      &active_features);
  if (!use_cached_prediction) {
    predictions = booster_detail::PredictFromHist(trees_,
                                                hist,
                                                tree_limit,
                                                tree_learning_rates_,
                                                learning_rate_,
                                                use_gpu_,
                                                prediction_dimension_,
                                                devices_,
                                                base_score_);
  } else {
    booster_detail::AddBaseScoreToPredictions(base_score_, prediction_dimension_, predictions);
    if (hist.bin_storage_bytes() == 1) {
      booster_detail::PredictCached(*cache, hist.compact_bin_indices.data(), hist.num_rows,
                                    tree_limit, prediction_dimension_, predictions);
    } else {
      booster_detail::PredictCached(*cache, hist.bin_indices.data(), hist.num_rows,
                                    tree_limit, prediction_dimension_, predictions);
    }
  }
  booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
  return predictions;
}

std::vector<std::int32_t> GradientBooster::PredictLeafIndices(const Pool& pool,
                                                              int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(trees_per_iteration()));
  }
  std::vector<std::int32_t> leaf_indices(pool.num_rows() * tree_limit, -1);
  if (tree_limit == 0) {
    return leaf_indices;
  }
  const auto& quantization_schema =
      booster_detail::RequireQuantizationSchema(quantization_schema_);
  // GPU leaf indices use the existing CPU traversal, without building a full
  // CPU score cache that GPU prediction itself never consumes.
  const auto cache = !use_gpu_
      ? GetPredictionCache() : std::shared_ptr<const booster_detail::PredictionCache>{};
  auto active_features = cache != nullptr
      ? cache->ActiveFeatures(tree_limit) : std::vector<std::uint8_t>(quantization_schema.num_cols(), 0U);
  if (cache == nullptr) {
    for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
      booster_detail::MarkUsedFeatures(trees_[tree_index], active_features);
    }
  }
  const HistMatrix hist = booster_detail::BuildPredictionHist(
      pool,
      quantization_schema,
      &active_features);
  if (cache == nullptr) {
    for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
      for (std::size_t row = 0; row < hist.num_rows; ++row) {
        leaf_indices[row * tree_limit + tree_index] =
            trees_[tree_index].PredictBinnedLeafIndex(hist, row);
      }
    }
  } else if (hist.bin_storage_bytes() == 1) {
    booster_detail::PredictCachedLeaves(*cache, hist.compact_bin_indices.data(), hist.num_rows,
                                        tree_limit, leaf_indices);
  } else {
    booster_detail::PredictCachedLeaves(*cache, hist.bin_indices.data(), hist.num_rows,
                                        tree_limit, leaf_indices);
  }
  return leaf_indices;
}

std::vector<float> GradientBooster::PredictContributions(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(trees_per_iteration()));
  }
  const std::size_t row_width = static_cast<std::size_t>(prediction_dimension_) * (pool.num_cols() + 1);
  std::vector<float> contributions(pool.num_rows() * row_width, 0.0F);
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    for (int output = 0; output < prediction_dimension_; ++output) {
      const std::size_t bias_index =
          row * row_width + static_cast<std::size_t>(output) * (pool.num_cols() + 1) +
          pool.num_cols();
      contributions[bias_index] = base_score_[static_cast<std::size_t>(output)];
    }
  }
  if (tree_limit == 0) {
    return contributions;
  }
  const auto& quantization_schema =
      booster_detail::RequireQuantizationSchema(quantization_schema_);
  std::vector<std::uint8_t> active_features(quantization_schema.num_cols(), 0U);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    booster_detail::MarkUsedFeatures(trees_[tree_index], active_features);
  }
  const HistMatrix hist = booster_detail::BuildPredictionHist(
      pool,
      quantization_schema,
      &active_features);
  std::vector<float> row_buffer(pool.num_cols() + 1, 0.0F);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const bool vector_leaves = trees_[tree_index].is_vector_leaf();
    const std::size_t first_output = vector_leaves ? 0U
        : tree_index % static_cast<std::size_t>(prediction_dimension_);
    const std::size_t output_end = vector_leaves
        ? static_cast<std::size_t>(prediction_dimension_) : first_output + 1U;
    const float tree_learning_rate = static_cast<float>(booster_detail::ResolveIterationLearningRate(
        tree_learning_rates_, tree_index, trees_per_iteration(), learning_rate_));
    for (std::size_t output = first_output; output < output_end; ++output) {
      for (std::size_t row = 0; row < pool.num_rows(); ++row) {
        std::fill(row_buffer.begin(), row_buffer.end(), 0.0F);
        trees_[tree_index].AccumulateBinnedContributions(
            hist, row, tree_learning_rate, row_buffer,
            vector_leaves ? static_cast<int>(output) : -1);
        const std::size_t row_offset = row * row_width + output * (pool.num_cols() + 1);
        for (std::size_t feature = 0; feature < row_buffer.size(); ++feature) {
          contributions[row_offset + feature] += row_buffer[feature];
        }
      }
    }
  }
  return contributions;
}

}  // namespace ctboost
