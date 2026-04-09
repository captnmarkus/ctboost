#pragma once

#include <string>

namespace ctboost {

bool CudaBackendCompiled() noexcept;
std::string CudaRuntimeVersionString();

}  // namespace ctboost
