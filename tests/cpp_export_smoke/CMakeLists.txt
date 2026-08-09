cmake_minimum_required(VERSION 3.20)
project(ctboost_generated_export_smoke LANGUAGES CXX)

if(NOT DEFINED CTBOOST_GENERATED_SOURCE OR NOT EXISTS "${CTBOOST_GENERATED_SOURCE}")
  message(FATAL_ERROR "Set CTBOOST_GENERATED_SOURCE to a generated CTBoost .cpp file")
endif()

add_library(ctboost_generated_export SHARED "${CTBOOST_GENERATED_SOURCE}")
set_target_properties(
  ctboost_generated_export
  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
)
