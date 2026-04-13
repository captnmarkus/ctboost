#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace ctboost {

class NativeFeaturePipeline {
 public:
  NativeFeaturePipeline(pybind11::object cat_features = pybind11::none(),
                        bool ordered_ctr = false,
                        int one_hot_max_size = 0,
                        int max_cat_threshold = 0,
                        pybind11::object categorical_combinations = pybind11::none(),
                        bool pairwise_categorical_combinations = false,
                        pybind11::object simple_ctr = pybind11::none(),
                        pybind11::object combinations_ctr = pybind11::none(),
                        pybind11::object per_feature_ctr = pybind11::none(),
                        pybind11::object text_features = pybind11::none(),
                        int text_hash_dim = 64,
                        pybind11::object embedding_features = pybind11::none(),
                        pybind11::object embedding_stats = pybind11::none(),
                        double ctr_prior_strength = 1.0,
                        int random_seed = 0);

  void fit_array(pybind11::array raw_matrix,
                 pybind11::array_t<float, pybind11::array::forcecast> labels,
                 pybind11::object feature_names = pybind11::none());
  pybind11::tuple fit_transform_array(
      pybind11::array raw_matrix,
      pybind11::array_t<float, pybind11::array::forcecast> labels,
      pybind11::object feature_names = pybind11::none());
  pybind11::tuple transform_array(pybind11::array raw_matrix,
                                  pybind11::object feature_names = pybind11::none()) const;

  pybind11::dict to_state() const;
  static NativeFeaturePipeline FromState(const pybind11::dict& state);

 private:
  struct CategoricalEncoderState {
    int source_index{-1};
    std::string output_name;
    std::unordered_map<std::string, float> mapping;
    std::uint8_t has_other_bucket{0};
    float other_value{0.0F};
  };

  struct OneHotEncoderState {
    int source_index{-1};
    std::string prefix;
    std::vector<std::string> category_keys;
    std::vector<std::string> output_names;
    std::uint8_t has_other_bucket{0};
  };

  struct CtrState {
    std::vector<int> source_indices;
    std::vector<std::string> output_names;
    std::string ctr_type;
    std::vector<float> prior_values;
    std::unordered_map<std::string, int> total_counts;
    std::unordered_map<std::string, std::vector<float>> total_sums;
    float global_frequency_prior{0.0F};
    std::size_t total_rows{0};
  };

  struct TextState {
    int source_index{-1};
    std::string prefix;
  };

  struct EmbeddingState {
    int source_index{-1};
    std::string prefix;
    std::vector<std::string> stats;
  };

  void FitInternal(pybind11::array raw_matrix,
                   pybind11::array_t<float, pybind11::array::forcecast> labels,
                   pybind11::object feature_names);
  pybind11::tuple TransformInternal(pybind11::array raw_matrix,
                                    pybind11::object feature_names,
                                    bool use_training_ctr_columns) const;
  std::vector<int> ResolveIndices(const pybind11::object& selectors) const;
  void RefreshCombinationSourceIndices();
  void LoadState(const pybind11::dict& state);

  pybind11::object cat_features_;
  bool ordered_ctr_{false};
  int one_hot_max_size_{0};
  int max_cat_threshold_{0};
  pybind11::object categorical_combinations_;
  bool pairwise_categorical_combinations_{false};
  pybind11::object simple_ctr_;
  pybind11::object combinations_ctr_;
  pybind11::object per_feature_ctr_;
  pybind11::object text_features_;
  int text_hash_dim_{64};
  pybind11::object embedding_features_;
  pybind11::object embedding_stats_;
  double ctr_prior_strength_{1.0};
  int random_seed_{0};
  std::optional<std::vector<std::string>> feature_names_in_;
  int n_features_in_{-1};
  std::vector<int> cat_feature_indices_;
  std::vector<std::string> output_feature_names_;
  std::vector<int> numeric_indices_;
  std::vector<OneHotEncoderState> one_hot_states_;
  std::vector<CategoricalEncoderState> categorical_states_;
  std::vector<CategoricalEncoderState> combination_states_;
  std::vector<std::vector<int>> combination_source_indices_;
  std::vector<CtrState> ctr_states_;
  std::vector<TextState> text_states_;
  std::vector<EmbeddingState> embedding_states_;
  std::vector<std::vector<float>> training_ctr_columns_;
  mutable std::unordered_map<std::string, std::uint64_t> text_hash_cache_;
};

}  // namespace ctboost
