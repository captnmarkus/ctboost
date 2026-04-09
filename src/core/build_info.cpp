#include "ctboost/build_info.hpp"
#include "ctboost/cuda_backend.hpp"

#include <string>

namespace ctboost {
namespace {

std::string CompilerString() {
#if defined(__clang__)
  return "Clang " + std::to_string(__clang_major__) + "." +
         std::to_string(__clang_minor__) + "." + std::to_string(__clang_patchlevel__);
#elif defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
  return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "Unknown";
#endif
}

}  // namespace

BuildInfo GetBuildInfo() {
  BuildInfo info;
  info.version = CTBOOST_VERSION;
  info.cuda_enabled = CudaBackendCompiled();
  info.cuda_runtime = CudaRuntimeVersionString();
  info.compiler = CompilerString();
  info.cxx_standard = 17;
  return info;
}

}  // namespace ctboost
