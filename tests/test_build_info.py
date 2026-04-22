import ctboost
import numpy as np

def test_build_info_matches_package_version():
    info = ctboost.build_info()
    assert info["version"] == ctboost.__version__
    assert info["cxx_standard"] == 17
    assert isinstance(info["cuda_enabled"], bool)
    assert isinstance(info["cuda_runtime"], str)
    assert isinstance(info["compiler"], str)

def test_pool_shell_preserves_constructor_shape():
    pool = ctboost.Pool(
        data=[[0.0, 1.0], [1.0, 0.0]],
        label=[0, 1],
        cat_features=[1],
        feature_names=["a", "b"],
    )
    assert pool.num_rows == 2
    assert pool.num_cols == 2
    assert pool.cat_features == [1]
    assert pool.feature_names == ["a", "b"]
    np.testing.assert_array_equal(pool.label, np.array([0.0, 1.0], dtype=np.float32))

def test_estimators_expose_sklearn_params():
    clf = ctboost.CTBoostClassifier(
        iterations=32,
        learning_rate=0.05,
        task_type="GPU",
        verbose=True,
        subsample=0.7,
        bootstrap_type="Bernoulli",
        bagging_temperature=0.7,
        boosting_type="DART",
        drop_rate=0.2,
        skip_drop=0.0,
        max_drop=3,
        one_hot_max_size=4,
        max_cat_threshold=8,
        cat_features=[0],
        ordered_ctr=True,
        categorical_combinations=[[0, 1]],
        simple_ctr=["Mean", "Frequency"],
        combinations_ctr=["Mean"],
        per_feature_ctr={0: ["Frequency"]},
        text_features=[2],
        embedding_features=[3],
        colsample_bytree=0.5,
        feature_weights={0: 2.0, 1: 0.5},
        first_feature_use_penalties={2: 1.25},
        random_strength=0.3,
        grow_policy="LeafWise",
        max_leaves=8,
        min_samples_split=5,
        min_data_in_leaf=3,
        min_child_weight=0.2,
        gamma=0.1,
        max_leaf_weight=1.5,
        max_bins=192,
        max_bin_by_feature={0: 7, 2: 31},
        border_selection_method="Uniform",
        nan_mode_by_feature={1: "Max"},
        feature_borders={3: [-0.25, 0.5]},
        random_seed=13,
        distributed_world_size=2,
        distributed_rank=1,
        distributed_root="tcp://127.0.0.1:19091",
        distributed_run_id="sk-unit",
        distributed_timeout=75.0,
    )
    reg = ctboost.CTBoostRegressor(iterations=12)

    clf_params = clf.get_params()
    reg_params = reg.get_params()

    assert clf_params["iterations"] == 32
    assert clf_params["learning_rate"] == 0.05
    assert clf_params["task_type"] == "GPU"
    assert clf_params["verbose"] is True
    assert clf_params["subsample"] == 0.7
    assert clf_params["bootstrap_type"] == "Bernoulli"
    assert clf_params["bagging_temperature"] == 0.7
    assert clf_params["boosting_type"] == "DART"
    assert clf_params["drop_rate"] == 0.2
    assert clf_params["skip_drop"] == 0.0
    assert clf_params["max_drop"] == 3
    assert clf_params["one_hot_max_size"] == 4
    assert clf_params["max_cat_threshold"] == 8
    assert clf_params["cat_features"] == [0]
    assert clf_params["ordered_ctr"] is True
    assert clf_params["categorical_combinations"] == [[0, 1]]
    assert clf_params["simple_ctr"] == ["Mean", "Frequency"]
    assert clf_params["combinations_ctr"] == ["Mean"]
    assert clf_params["per_feature_ctr"] == {0: ["Frequency"]}
    assert clf_params["text_features"] == [2]
    assert clf_params["embedding_features"] == [3]
    assert clf_params["colsample_bytree"] == 0.5
    assert clf_params["feature_weights"] == {0: 2.0, 1: 0.5}
    assert clf_params["first_feature_use_penalties"] == {2: 1.25}
    assert clf_params["random_strength"] == 0.3
    assert clf_params["grow_policy"] == "LeafWise"
    assert clf_params["max_leaves"] == 8
    assert clf_params["min_samples_split"] == 5
    assert clf_params["min_data_in_leaf"] == 3
    assert clf_params["min_child_weight"] == 0.2
    assert clf_params["gamma"] == 0.1
    assert clf_params["max_leaf_weight"] == 1.5
    assert clf_params["max_bins"] == 192
    assert clf_params["max_bin_by_feature"] == {0: 7, 2: 31}
    assert clf_params["border_selection_method"] == "Uniform"
    assert clf_params["nan_mode_by_feature"] == {1: "Max"}
    assert clf_params["feature_borders"] == {3: [-0.25, 0.5]}
    assert clf_params["random_seed"] == 13
    assert clf_params["distributed_world_size"] == 2
    assert clf_params["distributed_rank"] == 1
    assert clf_params["distributed_root"] == "tcp://127.0.0.1:19091"
    assert clf_params["distributed_run_id"] == "sk-unit"
    assert clf_params["distributed_timeout"] == 75.0
    assert clf_params["loss_function"] == "Logloss"
    assert reg_params["iterations"] == 12
    assert reg_params["loss_function"] == "RMSE"

def test_native_booster_exports_verbose_flag():
    handle = ctboost._core.GradientBooster(verbose=True)
    state = handle.export_state()
    assert state["verbose"] is True

def test_native_booster_exports_tree_control_state():
    handle = ctboost._core.GradientBooster(
        subsample=0.8,
        bootstrap_type="Poisson",
        bagging_temperature=0.4,
        boosting_type="DART",
        drop_rate=0.25,
        skip_drop=0.1,
        max_drop=2,
        monotone_constraints=[1, 0, -1],
        interaction_constraints=[[0, 1], [2]],
        colsample_bytree=0.6,
        feature_weights=[1.0, 0.5, 2.0],
        first_feature_use_penalties=[0.0, 0.0, 1.0],
        random_strength=0.15,
        grow_policy="LeafWise",
        max_leaves=7,
        min_samples_split=6,
        min_data_in_leaf=4,
        min_child_weight=0.5,
        gamma=0.2,
        max_leaf_weight=2.0,
        max_bins=300,
        max_bin_by_feature=[5, 0, 12],
        border_selection_method="Uniform",
        nan_mode_by_feature=["Min", "Max", "Forbidden"],
        feature_borders=[[0.0], [], [-1.0, 1.0]],
        external_memory=True,
        external_memory_dir="ctboost-native-spill",
        distributed_world_size=2,
        distributed_rank=1,
        distributed_root="ctboost-dist",
        distributed_run_id="unit-case",
        distributed_timeout=42.0,
        random_seed=17,
    )
    state = handle.export_state()

    assert state["subsample"] == 0.8
    assert state["bootstrap_type"] == "Poisson"
    assert state["bagging_temperature"] == 0.4
    assert state["boosting_type"] == "DART"
    assert state["drop_rate"] == 0.25
    assert state["skip_drop"] == 0.1
    assert state["max_drop"] == 2
    assert state["monotone_constraints"] == [1, 0, -1]
    assert state["interaction_constraints"] == [[0, 1], [2]]
    assert state["colsample_bytree"] == 0.6
    assert state["feature_weights"] == [1.0, 0.5, 2.0]
    assert state["first_feature_use_penalties"] == [0.0, 0.0, 1.0]
    assert state["random_strength"] == 0.15
    assert state["grow_policy"] == "LeafWise"
    assert state["max_leaves"] == 7
    assert state["min_samples_split"] == 6
    assert state["min_data_in_leaf"] == 4
    assert state["min_child_weight"] == 0.5
    assert state["gamma"] == 0.2
    assert state["max_leaf_weight"] == 2.0
    assert state["max_bins"] == 300
    assert state["max_bin_by_feature"] == [5, 0, 12]
    assert state["border_selection_method"] == "Uniform"
    assert state["nan_mode_by_feature"] == ["Min", "Max", "Forbidden"]
    assert state["feature_borders"] == [[0.0], [], [-1.0, 1.0]]
    assert state["external_memory"] is True
    assert state["external_memory_dir"] == "ctboost-native-spill"
    assert state["distributed_world_size"] == 2
    assert state["distributed_rank"] == 1
    assert state["distributed_root"] == "ctboost-dist"
    assert state["distributed_run_id"] == "unit-case"
    assert state["distributed_timeout"] == 42.0
    assert state["random_seed"] == 17
