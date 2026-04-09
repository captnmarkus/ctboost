#pragma once

#include <string>

namespace ctboost {

struct BuildInfo {
  std::string version;
  bool cuda_enabled{false};
  std::string cuda_runtime;
  std::string compiler;
  int cxx_standard{17};
};

BuildInfo GetBuildInfo();

}  // namespace ctboost
