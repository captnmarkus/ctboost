import ctboost


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
    assert pool.cat_features == [1]
    assert pool.feature_names == ["a", "b"]


def test_estimators_expose_sklearn_params():
    clf = ctboost.CTBoostClassifier(iterations=32, learning_rate=0.05, task_type="GPU")
    reg = ctboost.CTBoostRegressor(iterations=12)

    clf_params = clf.get_params()
    reg_params = reg.get_params()

    assert clf_params["iterations"] == 32
    assert clf_params["learning_rate"] == 0.05
    assert clf_params["task_type"] == "GPU"
    assert clf_params["loss_function"] == "Logloss"
    assert reg_params["iterations"] == 12
    assert reg_params["loss_function"] == "RMSE"
