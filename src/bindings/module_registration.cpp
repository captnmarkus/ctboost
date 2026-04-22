#include "module_internal.hpp"

#include <algorithm>
#include <chrono>
#include <memory>

namespace ctboost::bindings {

void BindModuleFunctions(py::module_& m) {
  m.def("_evaluate_metric",
        [](py::array_t<float, py::array::forcecast> predictions,
           py::array_t<float, py::array::forcecast> labels,
           py::array_t<float, py::array::forcecast> weights,
           std::string metric_name,
           int num_classes,
           py::object group_ids,
           py::object group_weights,
           py::object subgroup_ids,
           py::object pairs,
           py::object pairs_weight,
           double quantile_alpha,
           double huber_delta,
           double tweedie_variance_power) {
          ctboost::ObjectiveConfig config;
          config.quantile_alpha = quantile_alpha;
          config.huber_delta = huber_delta;
          config.tweedie_variance_power = tweedie_variance_power;
          auto metric = ctboost::CreateMetricFunction(metric_name, config);
          const std::vector<std::int64_t> resolved_group_ids =
              group_ids.is_none()
                  ? std::vector<std::int64_t>{}
                  : ArrayToInt64Vector(
                        group_ids.cast<py::array_t<std::int64_t, py::array::forcecast>>(),
                        "group_ids");
          const std::vector<float> resolved_group_weights =
              group_weights.is_none()
                  ? std::vector<float>{}
                  : ArrayToVector(
                        group_weights.cast<py::array_t<float, py::array::forcecast>>(),
                        "group_weight");
          const std::vector<std::int64_t> resolved_subgroup_ids =
              subgroup_ids.is_none()
                  ? std::vector<std::int64_t>{}
                  : ArrayToInt64Vector(
                        subgroup_ids.cast<py::array_t<std::int64_t, py::array::forcecast>>(),
                        "subgroup_id");
          const std::vector<ctboost::RankingPair> resolved_pairs =
              pairs.is_none()
                  ? std::vector<ctboost::RankingPair>{}
                  : ArrayToRankingPairs(
                        pairs.cast<py::array_t<std::int64_t, py::array::forcecast>>(),
                        pairs_weight,
                        static_cast<std::size_t>(labels.request().shape[0]));
          const ctboost::RankingMetadataView ranking{
              group_ids.is_none() ? nullptr : &resolved_group_ids,
              subgroup_ids.is_none() ? nullptr : &resolved_subgroup_ids,
              group_weights.is_none() ? nullptr : &resolved_group_weights,
              pairs.is_none() ? nullptr : &resolved_pairs,
          };
          return metric->Evaluate(ArrayToFlatFloatVector(predictions, "predictions"),
                                  ArrayToVector(labels, "labels"),
                                  ArrayToVector(weights, "weights"),
                                  num_classes,
                                  group_ids.is_none() && group_weights.is_none() &&
                                          subgroup_ids.is_none() && pairs.is_none()
                                      ? nullptr
                                      : &ranking);
        },
        py::arg("predictions"),
        py::arg("labels"),
        py::arg("weights"),
        py::arg("metric_name"),
        py::arg("num_classes") = 1,
        py::arg("group_ids") = py::none(),
        py::arg("group_weight") = py::none(),
        py::arg("subgroup_id") = py::none(),
        py::arg("pairs") = py::none(),
        py::arg("pairs_weight") = py::none(),
        py::arg("quantile_alpha") = 0.5,
        py::arg("huber_delta") = 1.0,
        py::arg("tweedie_variance_power") = 1.5);

  m.def("_metric_higher_is_better",
        [](std::string metric_name,
           double quantile_alpha,
           double huber_delta,
           double tweedie_variance_power) {
          ctboost::ObjectiveConfig config;
          config.quantile_alpha = quantile_alpha;
          config.huber_delta = huber_delta;
          config.tweedie_variance_power = tweedie_variance_power;
          return ctboost::CreateMetricFunction(metric_name, config)->HigherIsBetter();
        },
        py::arg("metric_name"),
        py::arg("quantile_alpha") = 0.5,
        py::arg("huber_delta") = 1.0,
        py::arg("tweedie_variance_power") = 1.5);

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

  m.def("_booster_from_state", [](const py::dict& state) {
    return BoosterFromStateDict(state);
  });

  m.def("_debug_compute_objective",
        [](const std::string& objective_name,
           py::array_t<float, py::array::forcecast> preds,
           py::array_t<float, py::array::forcecast> labels,
           int num_classes,
           double quantile_alpha,
           double huber_delta,
           double tweedie_variance_power) {
          const auto pred_values = ArrayToVector(preds, "preds");
          const auto label_values = ArrayToVector(labels, "labels");
          std::unique_ptr<ctboost::ObjectiveFunction> objective =
              ctboost::CreateObjectiveFunction(
                  objective_name,
                  ctboost::ObjectiveConfig{
                      huber_delta,
                      quantile_alpha,
                      tweedie_variance_power,
                  });

          std::vector<float> gradients;
          std::vector<float> hessians;
          objective->compute_gradients(
              pred_values, label_values, gradients, hessians, num_classes, nullptr);
          return py::make_tuple(VectorToArray(gradients), VectorToArray(hessians));
        },
        py::arg("objective_name"),
        py::arg("preds"),
        py::arg("labels"),
        py::arg("num_classes") = 1,
        py::arg("quantile_alpha") = 0.5,
        py::arg("huber_delta") = 1.0,
        py::arg("tweedie_variance_power") = 1.5);

  m.def("_debug_compute_pvalue",
        [](py::array_t<float, py::array::forcecast> gradients,
           py::array_t<std::int64_t, py::array::forcecast> binned_feature) {
          const auto gradient_values = ArrayToVector(gradients, "gradients");
          const auto bin_values = ArrayToBinVector(binned_feature, "binned_feature");
          const std::uint16_t max_bin =
              bin_values.empty() ? 0 : *std::max_element(bin_values.begin(), bin_values.end());
          std::vector<float> hessians(gradient_values.size(), 1.0F);
          std::vector<float> weights(gradient_values.size(), 1.0F);
          const ctboost::LinearStatistic statistic;
          const ctboost::LinearStatisticResult result =
              statistic.Evaluate(gradient_values,
                                 hessians,
                                 weights,
                                 bin_values,
                                 static_cast<std::size_t>(max_bin) + 1);

          py::dict out;
          out["p_value"] = result.p_value;
          out["chi_square"] = result.chi_square;
          out["degrees_of_freedom"] = result.degrees_of_freedom;
          out["statistic"] = result.statistic;
          out["expectation"] = result.expectation;
          out["covariance"] = result.covariance;
          return out;
        },
        py::arg("gradients"),
        py::arg("binned_feature"));

  m.def("_debug_build_histogram",
        [](const ctboost::Pool& pool,
           std::size_t max_bins,
           const std::string& nan_mode,
           const std::vector<std::uint16_t>& max_bin_by_feature,
           const std::string& border_selection_method,
           const std::vector<std::string>& nan_mode_by_feature,
           const std::vector<std::vector<float>>& feature_borders,
           bool external_memory,
           const std::string& external_memory_dir) {
          const auto started = std::chrono::steady_clock::now();
          const ctboost::HistBuilder builder(max_bins,
                                             nan_mode,
                                             max_bin_by_feature,
                                             border_selection_method,
                                             nan_mode_by_feature,
                                             feature_borders,
                                             external_memory,
                                             external_memory_dir);
          ctboost::HistMatrix hist = builder.Build(pool, nullptr);
          const double elapsed_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                  .count();

          py::dict out;
          out["elapsed_ms"] = elapsed_ms;
          out["num_rows"] = hist.num_rows;
          out["num_cols"] = hist.num_cols;
          out["num_bins_per_feature"] = hist.num_bins_per_feature;
          out["cut_offsets"] = hist.cut_offsets;
          out["cut_values"] = hist.cut_values;
          out["categorical_mask"] = hist.categorical_mask;
          out["missing_value_mask"] = hist.missing_value_mask;
          out["nan_mode"] = hist.nan_mode;
          out["nan_modes"] = hist.nan_modes;
          out["cut_values_count"] = hist.cut_values.size();
          out["uses_external_bin_storage"] = hist.uses_external_bin_storage();
          out["external_bin_storage_dir"] = hist.external_bin_storage_dir;
          out["storage_bytes"] = hist.storage_bytes();
          hist.ReleaseStorage();
          return out;
        },
        py::arg("pool"),
        py::arg("max_bins") = 256,
        py::arg("nan_mode") = "Min",
        py::arg("max_bin_by_feature") = std::vector<std::uint16_t>{},
        py::arg("border_selection_method") = "Quantile",
        py::arg("nan_mode_by_feature") = std::vector<std::string>{},
        py::arg("feature_borders") = std::vector<std::vector<float>>{},
        py::arg("external_memory") = false,
        py::arg("external_memory_dir") = "");
}

}  // namespace ctboost::bindings
