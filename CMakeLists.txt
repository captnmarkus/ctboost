cmake_minimum_required(VERSION 3.24...4.3)

project(CTBoost VERSION 0.1.43 LANGUAGES CXX)

if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  set(CTBOOST_PACKAGE_VERSION "${SKBUILD_PROJECT_VERSION_FULL}")
else()
  set(CTBOOST_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

option(CTBOOST_ENABLE_CUDA "Build CUDA backends when a CUDA toolkit is available" ON)
option(CTBOOST_REQUIRE_CUDA
       "Fail the build if CUDA was requested but a working toolkit is unavailable"
       OFF)

include(CheckLanguage)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

if(pybind11_VERSION VERSION_GREATER_EQUAL "3.0.0")
  message(FATAL_ERROR
          "CTBoost currently requires pybind11 < 3.0. "
          "Install a 2.x release such as pybind11==2.13.6 before building.")
endif()

set(CTBOOST_WITH_CUDA OFF)
if(CTBOOST_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build")
      endif()
      set(CTBOOST_WITH_CUDA ON)
      message(STATUS "CTBoost: CUDA toolkit detected, enabling GPU backend")
      message(STATUS "CTBoost: CUDA compiler: ${CMAKE_CUDA_COMPILER}")
      message(STATUS "CTBoost: CUDA toolkit root: ${CUDAToolkit_ROOT}")
      message(STATUS "CTBoost: CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
      if(CTBOOST_REQUIRE_CUDA)
        message(FATAL_ERROR
                "CTBoost: CUDA was required but CUDAToolkit could not be found. "
                "Install the full CUDA toolkit and set CUDAToolkit_ROOT if needed.")
      endif()
      message(STATUS "CTBoost: CUDA compiler found but toolkit missing, building CPU-only")
    endif()
  else()
    if(CTBOOST_REQUIRE_CUDA)
      message(FATAL_ERROR
              "CTBoost: CUDA was required but no CUDA compiler was found. "
              "Set CUDACXX/CMAKE_CUDA_COMPILER and ensure the CUDA bin directory is on PATH.")
    endif()
    message(STATUS "CTBoost: CUDA compiler not found, building CPU-only")
  endif()
else()
  message(STATUS "CTBoost: CUDA support disabled explicitly, building CPU-only")
endif()

add_library(ctboost_core STATIC
  src/core/booster.cpp
  src/core/booster_accessors.cpp
  src/core/booster_constructor.cpp
  src/core/booster_distributed.cpp
  src/core/booster_fit.cpp
  src/core/booster_fit_iteration.cpp
  src/core/booster_fit_loop.cpp
  src/core/booster_fit_metrics.cpp
  src/core/booster_fit_multiclass.cpp
  src/core/booster_fit_single_output.cpp
  src/core/booster_fit_support.cpp
  src/core/booster_predict.cpp
  src/core/booster_prediction.cpp
  src/core/booster_sampling.cpp
  src/core/booster_state.cpp
  src/core/booster_tokens.cpp
  src/core/booster_tree_updates.cpp
  src/core/build_info.cpp
  src/core/data.cpp
  src/core/data_dense.cpp
  src/core/data_metadata.cpp
  src/core/data_validation.cpp
  src/core/feature_pipeline.cpp
  src/core/feature_pipeline_fit.cpp
  src/core/feature_pipeline_fit_core.cpp
  src/core/feature_pipeline_fit_ctr.cpp
  src/core/feature_pipeline_normalize.cpp
  src/core/feature_pipeline_state.cpp
  src/core/feature_pipeline_state_load.cpp
  src/core/feature_pipeline_state_support.cpp
  src/core/feature_pipeline_transform.cpp
  src/core/histogram.cpp
  src/core/histogram_builder.cpp
  src/core/histogram_builder_support.cpp
  src/core/histogram_quantization.cpp
  src/core/histogram_quantization_support.cpp
  src/core/histogram_schema.cpp
  src/core/histogram_storage.cpp
  src/core/metric.cpp
  src/core/metric_classification.cpp
  src/core/metric_classification_scores.cpp
  src/core/metric_ranking.cpp
  src/core/metric_regression.cpp
  src/core/metric_support.cpp
  src/core/metric_survival.cpp
  src/core/objective.cpp
  src/core/objective_classification.cpp
  src/core/objective_ranking.cpp
  src/core/objective_regression.cpp
  src/core/objective_support.cpp
  src/core/objective_survival.cpp
  src/core/profiler.cpp
  src/core/statistics.cpp
  src/core/statistics_gamma.cpp
  src/core/tree.cpp
  src/core/tree_build.cpp
  src/core/tree_build_cpu.cpp
  src/core/tree_build_gpu.cpp
  src/core/tree_build_children.cpp
  src/core/tree_build_gpu_support.cpp
  src/core/tree_build_support.cpp
  src/core/tree_candidates.cpp
  src/core/tree_distributed.cpp
  src/core/tree_distributed_payloads.cpp
  src/core/tree_predict.cpp
  src/core/tree_split_scoring.cpp
)

target_include_directories(ctboost_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(ctboost_core PUBLIC cxx_std_17)
target_link_libraries(ctboost_core PUBLIC pybind11::headers Python::Module)
target_compile_definitions(ctboost_core
  PUBLIC
    CTBOOST_VERSION="${CTBOOST_PACKAGE_VERSION}"
)

if(WIN32)
  target_link_libraries(ctboost_core PUBLIC ws2_32)
endif()

if(CTBOOST_WITH_CUDA)
  add_library(ctboost_cuda_backend STATIC
    cuda/cuda_backend.cu
    cuda/hist_kernels.cu
  )

  set_target_properties(ctboost_cuda_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )

  target_include_directories(ctboost_cuda_backend
    PUBLIC
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/cuda
  )

  target_compile_definitions(ctboost_core PUBLIC CTBOOST_WITH_CUDA=1)
  target_link_libraries(ctboost_cuda_backend PUBLIC CUDA::cudart pybind11::headers Python::Module)
  target_link_libraries(ctboost_core PUBLIC ctboost_cuda_backend)
else()
  target_sources(ctboost_core PRIVATE src/core/cuda_backend_stub.cpp)
  target_compile_definitions(ctboost_core PUBLIC CTBOOST_WITH_CUDA=0)
endif()

pybind11_add_module(_core MODULE
  src/bindings/module.cpp
  src/bindings/module_arrays.cpp
  src/bindings/module_registration.cpp
  src/bindings/module_registration_booster.cpp
  src/bindings/module_registration_booster_accessors.cpp
  src/bindings/module_registration_booster_state.cpp
  src/bindings/module_registration_pipeline.cpp
  src/bindings/module_registration_pool.cpp
  src/bindings/module_state.cpp
  src/bindings/module_state_booster.cpp
)

target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(_core PRIVATE ctboost_core)

if(CTBOOST_WITH_CUDA)
  target_link_libraries(_core PRIVATE ctboost_cuda_backend)
  set_target_properties(_core PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

install(TARGETS _core
  LIBRARY DESTINATION ctboost
  RUNTIME DESTINATION ctboost
  ARCHIVE DESTINATION ctboost
)
