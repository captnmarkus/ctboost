#include "ctboost/cuda_backend.hpp"

#include <string>

namespace ctboost {

bool CudaBackendCompiled() noexcept { return false; }

std::string CudaRuntimeVersionString() { return "not compiled"; }

}  // namespace ctboost
