#include "module_internal.hpp"

namespace ctboost::bindings {

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
