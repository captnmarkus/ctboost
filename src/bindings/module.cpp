#include "module_internal.hpp"

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native backend scaffolding for CTBoost";
  ctboost::bindings::BindModuleFunctions(m);
  ctboost::bindings::BindPool(m);
  ctboost::bindings::BindNativeFeaturePipeline(m);
  ctboost::bindings::BindGradientBooster(m);
}
