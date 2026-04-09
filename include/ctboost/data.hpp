#pragma once

#include <cstddef>
#include <vector>

#include <pybind11/numpy.h>

namespace ctboost {

class Pool {
 public:
  Pool(pybind11::array_t<float, pybind11::array::forcecast> data,
       pybind11::array_t<float, pybind11::array::forcecast> label,
       std::vector<int> cat_features = {});

  std::size_t num_rows() const noexcept;
  std::size_t num_cols() const noexcept;
  const std::vector<float>& feature_data() const noexcept;
  const std::vector<float>& labels() const noexcept;
  const std::vector<int>& cat_features() const noexcept;
  float feature_value(std::size_t row, std::size_t col) const;

 private:
  std::size_t num_rows_{0};
  std::size_t num_cols_{0};
  std::vector<float> feature_data_;
  std::vector<float> labels_;
  std::vector<int> cat_features_;
};

}  // namespace ctboost
