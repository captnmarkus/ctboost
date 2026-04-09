#include <pybind11/pybind11.h>

#include "ctboost/build_info.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native backend scaffolding for CTBoost";

  m.def("build_info", []() {
    const ctboost::BuildInfo info = ctboost::GetBuildInfo();
    py::dict result;
    result["version"] = info.version;
    result["cuda_enabled"] = info.cuda_enabled;
    result["cuda_runtime"] = info.cuda_runtime;
    result["compiler"] = info.compiler;
    result["cxx_standard"] = info.cxx_standard;
    return result;
  });
}
