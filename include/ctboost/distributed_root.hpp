#pragma once

#include <string>

namespace ctboost {

inline bool DistributedRootUsesTcp(const std::string& root) {
  return root.rfind("tcp://", 0) == 0;
}

inline std::string RedactDistributedTcpRoot(const std::string& root) {
  if (!DistributedRootUsesTcp(root)) {
    return root;
  }
  const std::size_t slash = root.find('/', 6U);
  return slash == std::string::npos ? root : root.substr(0, slash);
}

}  // namespace ctboost
