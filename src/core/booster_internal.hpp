#pragma once

#include "ctboost/booster.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/distributed_client.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost::booster_detail {

enum class BootstrapType {
  kNone,
  kBernoulli,
  kPoisson,
  kBayesian,
};

enum class BoostingType {
  kGradientBoosting,
  kRandomForest,
  kDart,
};
struct DistributedMetricControl {
  double train_loss{0.0};
  double eval_score{0.0};
  double best_score{0.0};
  int best_iteration{-1};
  std::uint8_t has_eval{0};
  std::uint8_t should_stop{0};
};

struct DistributedMetricInputs {
  std::vector<float> predictions;
  std::vector<float> labels;
  std::vector<float> weights;
  std::vector<std::int64_t> group_ids;
  bool has_group_ids{false};
};

std::string NormalizeToken(std::string value);
bool IsSquaredErrorObjective(const std::string& normalized_objective);
bool IsAbsoluteErrorObjective(const std::string& normalized_objective);
bool IsHuberObjective(const std::string& normalized_objective); bool IsQuantileObjective(const std::string& normalized_objective);
bool IsPoissonObjective(const std::string& normalized_objective); bool IsTweedieObjective(const std::string& normalized_objective);
bool IsSurvivalObjective(const std::string& normalized_objective); bool IsBinaryObjective(const std::string& normalized_objective);
bool IsMulticlassObjective(const std::string& normalized_objective); bool IsRankingObjective(const std::string& normalized_objective);
bool IsRegressionObjective(const std::string& normalized_objective);
std::uint64_t NormalizeRngState(std::uint64_t seed);
std::uint64_t NextRandom(std::uint64_t& state);
std::size_t UniformIndex(std::uint64_t& state, std::size_t limit);
bool SameCategoricalFeatures(const Pool& lhs, const Pool& rhs);
void AddPoolBaselineToPredictions(const Pool& pool,
                                  int prediction_dimension,
                                  std::vector<float>& predictions);
int LabelToClassIndex(float label, int num_classes);
std::string NormalizeTaskType(std::string task_type);
std::string CanonicalBootstrapType(std::string bootstrap_type);
BootstrapType ParseBootstrapType(const std::string& bootstrap_type);
std::string CanonicalBoostingType(std::string boosting_type);
BoostingType ParseBoostingType(const std::string& boosting_type); std::string CanonicalGrowPolicy(std::string grow_policy);
GrowPolicy ParseGrowPolicy(const std::string& grow_policy);
const QuantizationSchema& RequireQuantizationSchema(const QuantizationSchemaPtr& quantization_schema);

std::vector<std::uint8_t> SerializeDistributedMetricControl(const DistributedMetricControl& control);
DistributedMetricControl DeserializeDistributedMetricControl(
    const std::vector<std::uint8_t>& buffer);
DistributedMetricControl BroadcastDistributedMetricControl(
    const DistributedCoordinator* coordinator,
    const char* label,
    const DistributedMetricControl* root_control);
DistributedMetricInputs AllGatherDistributedMetricInputs(
    const DistributedCoordinator* coordinator,
    const char* label,
    const DistributedMetricInputs& local_inputs);

double UniformUnit(std::uint64_t& state);
std::uint32_t SamplePoisson(double lambda, std::uint64_t& state);
float SampleBayesianBootstrapWeight(float base_weight,
                                    double bagging_temperature,
                                    std::uint64_t& state);
std::vector<float> SampleRowWeights(const std::vector<float>& base_weights,
                                    double subsample,
                                    BootstrapType bootstrap_type,
                                    double bagging_temperature,
                                    std::uint64_t& rng_state);
void ScaleTreeLeafWeights(Tree& tree, double scale);
double ResolveIterationLearningRate(const std::vector<double>& tree_learning_rates,
                                    std::size_t tree_index,
                                    int prediction_dimension,
                                    double default_learning_rate);
std::vector<std::size_t> SampleDroppedTreeGroups(std::size_t completed_iterations,
                                                 double drop_rate,
                                                 double skip_drop,
                                                 int max_drop,
                                                 std::uint64_t& rng_state);
InteractionConstraintSet BuildInteractionConstraintSet(
    const std::vector<std::vector<int>>& raw_constraints,
    std::size_t num_features);
std::vector<int> SampleFeatureSubset(std::size_t num_features,
                                     double colsample_bytree,
                                     const std::vector<double>* feature_weights,
                                     std::uint64_t& rng_state);
void AccumulateFeatureImportances(const Tree& tree, std::vector<double>& feature_importance_sums);
void RecomputeFeatureImportances(const std::vector<Tree>& trees,
                                 std::size_t num_features,
                                 std::vector<double>& feature_importance_sums);
void MarkUsedFeatures(const Tree& tree, std::vector<std::uint8_t>& feature_used_mask);
HistMatrix BuildPredictionHist(const Pool& pool, const QuantizationSchema& quantization_schema);
HistMatrix BuildPredictionHist(const Pool& pool, const Tree& reference_tree);
std::vector<GpuTreeNode> FlattenTreesForGpu(const std::vector<Tree>& trees,
                                            std::size_t tree_limit,
                                            const std::vector<double>& tree_learning_rates,
                                            double default_learning_rate,
                                            int prediction_dimension,
                                            std::vector<std::int32_t>& tree_offsets);
void UpdatePredictions(const Tree& tree,
                       const HistMatrix& hist,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions);
void AccumulateIterationPredictions(const std::vector<Tree>& trees,
                                    std::size_t iteration_index,
                                    const HistMatrix& hist,
                                    const std::vector<double>& tree_learning_rates,
                                    double default_learning_rate,
                                    int prediction_dimension,
                                    std::vector<float>& predictions);
std::vector<float> PredictFromHist(const std::vector<Tree>& trees,
                                   const HistMatrix& hist,
                                   std::size_t tree_limit,
                                   const std::vector<double>& tree_learning_rates,
                                   double default_learning_rate,
                                   bool use_gpu,
                                   int prediction_dimension,
                                   const std::string& devices = "0");
void UpdatePredictionsFromLeafRanges(const Tree& tree,
                                     const std::vector<std::size_t>& row_indices,
                                     const std::vector<LeafRowRange>& leaf_row_ranges,
                                     double learning_rate,
                                     int prediction_dimension,
                                     int class_index,
                                     std::vector<float>& predictions);
float ComputeLeafWeightFromSums(double gradient_sum, double hessian_sum, double lambda_l2);
void BuildSharedMulticlassTargets(const std::vector<float>& gradients,
                                  const std::vector<float>& hessians,
                                  const std::vector<float>& weights,
                                  std::size_t num_rows,
                                  int prediction_dimension,
                                  std::vector<float>& structure_gradients,
                                  std::vector<float>& structure_hessians);
std::vector<int> PredictLeafIndicesFromHist(const Tree& tree, const HistMatrix& hist);
void UpdatePredictionsFromLeafIndices(const Tree& tree,
                                      const std::vector<int>& leaf_indices,
                                      double learning_rate,
                                      int prediction_dimension,
                                      int class_index,
                                      std::vector<float>& predictions);
std::vector<Tree> MaterializeMulticlassTreesFromStructure(
    const Tree& structure_tree,
    const std::vector<std::size_t>& row_indices,
    const std::vector<LeafRowRange>& leaf_row_ranges,
    const std::vector<float>& gradients,
    const std::vector<float>& hessians,
    const std::vector<float>& weights,
    int prediction_dimension,
    double lambda_l2);
void UpdatePredictions(const Tree& tree,
                       const Pool& pool,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions);

}  // namespace ctboost::booster_detail
