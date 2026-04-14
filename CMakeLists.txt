cmake_minimum_required(VERSION 3.24...4.3)

project(CTBoost VERSION 0.1.32 LANGUAGES CXX)

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
  src/core/build_info.cpp
  src/core/data.cpp
  src/core/feature_pipeline.cpp
  src/core/histogram.cpp
  src/core/metric.cpp
  src/core/objective.cpp
  src/core/profiler.cpp
  src/core/statistics.cpp
  src/core/tree.cpp
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
