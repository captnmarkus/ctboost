#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ctboost/build_info.hpp"
#include "ctboost/booster.hpp"
#include "ctboost/cuda_backend.hpp"
#include "ctboost/data.hpp"
#include "ctboost/feature_pipeline.hpp"
#include "ctboost/histogram.hpp"
#include "ctboost/objective.hpp"
#include "ctboost/statistics.hpp"

namespace py = pybind11;

namespace ctboost::bindings {

py::array_t<float> VectorToArray(const std::vector<float>& values);
py::array_t<std::int32_t> IntVectorToArray(const std::vector<std::int32_t>& values);
std::vector<float> ArrayToVector(py::array_t<float, py::array::forcecast> values, const char* name);
std::vector<float> ArrayToFlatFloatVector(py::array_t<float, py::array::forcecast> values,
                                          const char* name);
std::vector<std::int64_t> ArrayToInt64Vector(
    py::array_t<std::int64_t, py::array::forcecast> values,
    const char* name);
std::vector<ctboost::RankingPair> ArrayToRankingPairs(
    py::array_t<std::int64_t, py::array::forcecast> pairs,
    py::object pairs_weight,
    std::size_t num_rows);
std::vector<std::uint16_t> ArrayToBinVector(
    py::array_t<std::int64_t, py::array::forcecast> values,
    const char* name);

py::dict QuantizationSchemaToStateDict(const ctboost::QuantizationSchema& quantization_schema);
ctboost::QuantizationSchemaPtr QuantizationSchemaFromStateDict(const py::handle& handle);
ctboost::QuantizationSchemaPtr QuantizationSchemaFromTreeStateDict(const py::dict& state);
py::tuple NodeToState(const ctboost::Node& node);
ctboost::Node NodeFromState(const py::handle& handle);
py::dict NodeToStateDict(const ctboost::Node& node);
ctboost::Node NodeFromStateDict(const py::handle& handle);
py::tuple TreeToState(const ctboost::Tree& tree);
ctboost::Tree TreeFromState(const py::handle& handle);
py::dict TreeToStateDict(const ctboost::Tree& tree);
ctboost::Tree TreeFromStateDict(
    const py::handle& handle,
    const ctboost::QuantizationSchemaPtr& shared_quantization_schema = nullptr);
py::dict BoosterToStateDict(const ctboost::GradientBooster& booster);
ctboost::GradientBooster BoosterFromStateDict(const py::dict& state);

void BindModuleFunctions(py::module_& m);
void BindPool(py::module_& m);
void BindNativeFeaturePipeline(py::module_& m);
void BindGradientBooster(py::module_& m);
void BindGradientBoosterAccessors(py::class_<ctboost::GradientBooster>& booster_class);
void BindGradientBoosterStateMethods(py::class_<ctboost::GradientBooster>& booster_class);

}  // namespace ctboost::bindings
