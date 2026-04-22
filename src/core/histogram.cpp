#include "histogram_internal.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace ctboost {
namespace {

std::string NormalizeToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

}  // namespace

NanMode ParseNanMode(std::string_view name) {
  const std::string normalized = NormalizeToken(std::string(name));
  if (normalized == "forbidden") {
    return NanMode::Forbidden;
  }
  if (normalized == "min") {
    return NanMode::Min;
  }
  if (normalized == "max") {
    return NanMode::Max;
  }
  throw std::invalid_argument("nan_mode must be one of 'Forbidden', 'Min', or 'Max'");
}

const char* NanModeName(NanMode nan_mode) noexcept {
  switch (nan_mode) {
    case NanMode::Forbidden:
      return "Forbidden";
    case NanMode::Min:
      return "Min";
    case NanMode::Max:
      return "Max";
  }
  return "Min";
}

BorderSelectionMethod ParseBorderSelectionMethod(std::string_view name) {
  const std::string normalized = NormalizeToken(std::string(name));
  if (normalized.empty() || normalized == "quantile" || normalized == "median") {
    return BorderSelectionMethod::Quantile;
  }
  if (normalized == "uniform") {
    return BorderSelectionMethod::Uniform;
  }
  throw std::invalid_argument(
      "border_selection_method must be one of 'Quantile' or 'Uniform'");
}

const char* BorderSelectionMethodName(BorderSelectionMethod method) noexcept {
  switch (method) {
    case BorderSelectionMethod::Quantile:
      return "Quantile";
    case BorderSelectionMethod::Uniform:
      return "Uniform";
  }
  return "Quantile";
}

std::size_t HistBuilder::max_bins() const noexcept { return max_bins_; }

const std::vector<std::uint16_t>& HistBuilder::max_bins_by_feature() const noexcept {
  return max_bins_by_feature_;
}

NanMode HistBuilder::nan_mode() const noexcept { return nan_mode_; }

const std::string& HistBuilder::nan_mode_name() const noexcept { return nan_mode_name_; }

BorderSelectionMethod HistBuilder::border_selection_method() const noexcept {
  return border_selection_method_;
}

const std::string& HistBuilder::border_selection_method_name() const noexcept {
  return border_selection_method_name_;
}

const std::vector<std::string>& HistBuilder::nan_mode_by_feature_names() const noexcept {
  return nan_mode_by_feature_names_;
}

const std::vector<std::vector<float>>& HistBuilder::feature_borders() const noexcept {
  return feature_borders_;
}

bool HistBuilder::external_memory() const noexcept { return external_memory_; }

const std::string& HistBuilder::external_memory_dir() const noexcept {
  return external_memory_dir_;
}

}  // namespace ctboost
