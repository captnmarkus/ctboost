#include "module_internal.hpp"

#include <cmath>
#include <cstring>

namespace ctboost::bindings {

namespace {

class PythonObjective final : public ctboost::ObjectiveFunction {
 public:
  explicit PythonObjective(py::object callable) : callable_(std::move(callable)) {
    if (!PyCallable_Check(callable_.ptr())) {
      throw py::type_error("custom_objective must be callable");
    }
  }

  void compute_gradients(const std::vector<float>& predictions,
                         const std::vector<float>& labels,
                         std::vector<float>& gradients,
                         std::vector<float>& hessians,
                         int /*num_classes*/,
                         const ctboost::RankingMetadataView* /*ranking*/) const override {
    if (labels.empty() || predictions.size() % labels.size() != 0U) {
      throw std::invalid_argument(
          "custom objective predictions must contain one or more values per training row");
    }
    const py::ssize_t row_count = static_cast<py::ssize_t>(labels.size());
    const py::ssize_t prediction_dimension =
        static_cast<py::ssize_t>(predictions.size() / labels.size());

    py::array_t<float> prediction_array = prediction_dimension == 1
                                              ? py::array_t<float>({row_count})
                                              : py::array_t<float>({row_count, prediction_dimension});
    py::array_t<float> label_array({row_count});
    std::memcpy(prediction_array.mutable_data(),
                predictions.data(),
                predictions.size() * sizeof(float));
    std::memcpy(label_array.mutable_data(), labels.data(), labels.size() * sizeof(float));

    py::object result = callable_(prediction_array, label_array);
    if (!py::isinstance<py::tuple>(result) && !py::isinstance<py::list>(result)) {
      throw py::type_error(
          "custom objective must return a (gradients, hessians) tuple");
    }
    const py::sequence derivative_pair = result.cast<py::sequence>();
    if (derivative_pair.size() != 2) {
      throw py::type_error(
          "custom objective must return exactly two values: gradients and hessians");
    }

    using ContiguousFloatArray =
        py::array_t<float, py::array::c_style | py::array::forcecast>;
    ContiguousFloatArray gradient_array =
        ContiguousFloatArray::ensure(derivative_pair[0]);
    ContiguousFloatArray hessian_array =
        ContiguousFloatArray::ensure(derivative_pair[1]);
    if (!gradient_array || !hessian_array) {
      throw py::type_error(
          "custom objective gradients and hessians must be numeric arrays");
    }

    const py::buffer_info prediction_info = prediction_array.request();
    const py::buffer_info gradient_info = gradient_array.request();
    const py::buffer_info hessian_info = hessian_array.request();
    const auto has_expected_shape = [&prediction_info](const py::buffer_info& candidate) {
      if (candidate.ndim != prediction_info.ndim) {
        return false;
      }
      for (py::ssize_t axis = 0; axis < prediction_info.ndim; ++axis) {
        if (candidate.shape[static_cast<std::size_t>(axis)] !=
            prediction_info.shape[static_cast<std::size_t>(axis)]) {
          return false;
        }
      }
      return true;
    };
    if (!has_expected_shape(gradient_info)) {
      throw py::value_error(
          "custom objective gradients must have exactly the same shape as predictions");
    }
    if (!has_expected_shape(hessian_info)) {
      throw py::value_error(
          "custom objective hessians must have exactly the same shape as predictions");
    }

    const std::size_t value_count = predictions.size();
    const auto* gradient_data = static_cast<const float*>(gradient_info.ptr);
    const auto* hessian_data = static_cast<const float*>(hessian_info.ptr);
    gradients.assign(gradient_data, gradient_data + value_count);
    hessians.assign(hessian_data, hessian_data + value_count);
    for (std::size_t index = 0; index < value_count; ++index) {
      if (!std::isfinite(gradients[index])) {
        throw py::value_error("custom objective gradients must contain only finite values");
      }
      if (!std::isfinite(hessians[index])) {
        throw py::value_error("custom objective hessians must contain only finite values");
      }
      if (hessians[index] < 0.0F) {
        throw py::value_error("custom objective hessians must be non-negative");
      }
    }
  }

 private:
  py::object callable_;
};

}  // namespace

void BindGradientBooster(py::module_& m) {
  py::class_<ctboost::GradientBooster> booster_class(m, "GradientBooster");
  booster_class
      .def(py::init<std::string,
                    int,
                    double,
                    int,
                    double,
                    double,
                    double,
                    std::string,
                    double,
                    std::string,
                    double,
                    double,
                    int,
                    std::vector<int>,
                    std::vector<std::vector<int>>,
                    double,
                    std::vector<double>,
                    std::vector<double>,
                    double,
                    std::string,
                    int,
                    int,
                    int,
                    double,
                    double,
                    double,
                    int,
                    std::size_t,
                    std::string,
                    std::vector<std::uint16_t>,
                    std::string,
                    std::vector<std::string>,
                    std::vector<std::vector<float>>,
                    bool,
                    std::string,
                    std::string,
                    double,
                    double,
                    double,
                    std::string,
                    std::string,
                    int,
                    int,
                    std::string,
                    std::string,
                    double,
                    std::uint64_t,
                    bool>(),
           py::arg("objective") = "RMSE",
           py::arg("iterations") = 100,
           py::arg("learning_rate") = 0.1,
           py::arg("max_depth") = 6,
           py::arg("alpha") = 0.05,
           py::arg("lambda_l2") = 1.0,
           py::arg("subsample") = 1.0,
           py::arg("bootstrap_type") = "No",
           py::arg("bagging_temperature") = 0.0,
           py::arg("boosting_type") = "GradientBoosting",
           py::arg("drop_rate") = 0.1,
           py::arg("skip_drop") = 0.5,
           py::arg("max_drop") = 0,
           py::arg("monotone_constraints") = std::vector<int>{},
           py::arg("interaction_constraints") = std::vector<std::vector<int>>{},
           py::arg("colsample_bytree") = 1.0,
           py::arg("feature_weights") = std::vector<double>{},
           py::arg("first_feature_use_penalties") = std::vector<double>{},
           py::arg("random_strength") = 0.0,
           py::arg("grow_policy") = "DepthWise",
           py::arg("max_leaves") = 0,
           py::arg("min_samples_split") = 2,
           py::arg("min_data_in_leaf") = 0,
           py::arg("min_child_weight") = 0.0,
           py::arg("gamma") = 0.0,
           py::arg("max_leaf_weight") = 0.0,
           py::arg("num_classes") = 1,
           py::arg("max_bins") = 256,
           py::arg("nan_mode") = "Min",
           py::arg("max_bin_by_feature") = std::vector<std::uint16_t>{},
           py::arg("border_selection_method") = "Quantile",
           py::arg("nan_mode_by_feature") = std::vector<std::string>{},
           py::arg("feature_borders") = std::vector<std::vector<float>>{},
           py::arg("external_memory") = false,
           py::arg("external_memory_dir") = "",
           py::arg("eval_metric") = "",
           py::arg("quantile_alpha") = 0.5,
           py::arg("huber_delta") = 1.0,
           py::arg("tweedie_variance_power") = 1.5,
           py::arg("task_type") = "CPU",
           py::arg("devices") = "0",
           py::arg("distributed_world_size") = 1,
           py::arg("distributed_rank") = 0,
           py::arg("distributed_root") = "",
           py::arg("distributed_run_id") = "default",
           py::arg("distributed_timeout") = 600.0,
           py::arg("random_seed") = 0,
           py::arg("verbose") = false)
      .def("fit",
           [](ctboost::GradientBooster& booster,
              py::object pool_obj,
              py::object eval_pool,
              int early_stopping_rounds,
              bool continue_training)
               -> ctboost::GradientBooster& {
             auto& pool = pool_obj.cast<ctboost::Pool&>();
             if (eval_pool.is_none()) {
               booster.Fit(pool, nullptr, early_stopping_rounds, continue_training);
             } else {
               auto& eval_pool_ref = eval_pool.cast<ctboost::Pool&>();
               booster.Fit(pool, &eval_pool_ref, early_stopping_rounds, continue_training);
             }
             return booster;
           },
           py::arg("pool"),
           py::arg("eval_pool") = py::none(),
           py::arg("early_stopping_rounds") = 0,
           py::arg("continue_training") = false,
           py::return_value_policy::reference_internal)
      .def("fit_custom_objective",
           [](ctboost::GradientBooster& booster,
              py::object pool_obj,
              py::object custom_objective,
              py::object eval_pool,
              int early_stopping_rounds,
              bool continue_training)
               -> ctboost::GradientBooster& {
             auto& pool = pool_obj.cast<ctboost::Pool&>();
             const PythonObjective objective(std::move(custom_objective));
             if (eval_pool.is_none()) {
               booster.FitWithObjective(
                   pool, objective, nullptr, early_stopping_rounds, continue_training);
             } else {
               auto& eval_pool_ref = eval_pool.cast<ctboost::Pool&>();
               booster.FitWithObjective(
                   pool, objective, &eval_pool_ref, early_stopping_rounds, continue_training);
             }
             return booster;
           },
           py::arg("pool"),
           py::arg("custom_objective"),
           py::arg("eval_pool") = py::none(),
           py::arg("early_stopping_rounds") = 0,
           py::arg("continue_training") = false,
           py::return_value_policy::reference_internal)
      .def("predict",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return VectorToArray(booster.Predict(pool, num_iteration));
           },
           py::arg("pool"),
           py::arg("num_iteration") = -1)
      .def("predict_leaf_indices",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return IntVectorToArray(booster.PredictLeafIndices(pool, num_iteration));
           },
           py::arg("pool"),
           py::arg("num_iteration") = -1)
      .def("predict_contributions",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return VectorToArray(booster.PredictContributions(pool, num_iteration));
           },
           py::arg("pool"),
           py::arg("num_iteration") = -1);

  BindGradientBoosterAccessors(booster_class);
  BindGradientBoosterStateMethods(booster_class);
}

}  // namespace ctboost::bindings
