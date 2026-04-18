#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "ctboost/ranking.hpp"

namespace ctboost {

struct ObjectiveConfig {
  double huber_delta{1.0};
  double quantile_alpha{0.5};
  double tweedie_variance_power{1.5};
};

class ObjectiveFunction {
 public:
  virtual ~ObjectiveFunction() = default;

  virtual void compute_gradients(const std::vector<float>& preds,
                                 const std::vector<float>& labels,
                                 std::vector<float>& out_g,
                                 std::vector<float>& out_h,
                                 int num_classes = 1,
                                 const RankingMetadataView* ranking = nullptr) const = 0;
};

class SquaredError final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class LogLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class SoftmaxLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class AbsoluteError final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class HuberLoss final : public ObjectiveFunction {
 public:
  explicit HuberLoss(double delta);

  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;

 private:
  double delta_{1.0};
};

class QuantileLoss final : public ObjectiveFunction {
 public:
  explicit QuantileLoss(double alpha);

  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;

 private:
  double alpha_{0.5};
};

class PoissonLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class TweedieLoss final : public ObjectiveFunction {
 public:
  explicit TweedieLoss(double variance_power);

  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;

 private:
  double variance_power_{1.5};
};

class CoxLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class SurvivalExponentialLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

class PairLogitLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1,
                         const RankingMetadataView* ranking = nullptr) const override;
};

std::unique_ptr<ObjectiveFunction> CreateObjectiveFunction(
    std::string_view name,
    const ObjectiveConfig& config = {});

}  // namespace ctboost
