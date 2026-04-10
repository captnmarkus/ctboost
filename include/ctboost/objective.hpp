#pragma once

#include <memory>
#include <string_view>
#include <vector>

namespace ctboost {

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

std::unique_ptr<ObjectiveFunction> CreateObjectiveFunction(std::string_view name);

}  // namespace ctboost
