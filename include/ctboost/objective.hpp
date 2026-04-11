#pragma once

#include <memory>
#include <string_view>
#include <vector>

namespace ctboost {

struct ObjectiveConfig {
  double huber_delta{1.0};
  double quantile_alpha{0.5};
};

class ObjectiveFunction {
 public:
  virtual ~ObjectiveFunction() = default;

  virtual void compute_gradients(const std::vector<float>& preds,
                                 const std::vector<float>& labels,
                                 std::vector<float>& out_g,
                                 std::vector<float>& out_h,
                                 int num_classes = 1) const = 0;
};

class SquaredError final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1) const override;
};

class LogLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1) const override;
};

class SoftmaxLoss final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1) const override;
};

class AbsoluteError final : public ObjectiveFunction {
 public:
  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1) const override;
};

class HuberLoss final : public ObjectiveFunction {
 public:
  explicit HuberLoss(double delta);

  void compute_gradients(const std::vector<float>& preds,
                         const std::vector<float>& labels,
                         std::vector<float>& out_g,
                         std::vector<float>& out_h,
                         int num_classes = 1) const override;

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
                         int num_classes = 1) const override;

 private:
  double alpha_{0.5};
};

std::unique_ptr<ObjectiveFunction> CreateObjectiveFunction(
    std::string_view name,
    const ObjectiveConfig& config = {});

}  // namespace ctboost
