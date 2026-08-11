import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import ctboost
from ctboost._serialization import _deserialize_json_value, _serialize_json_value


MISSING_SENTINEL = "__ctboost_missing__"
OTHER_SENTINEL = "__ctboost_other__"


def _legacy_unfitted_pipeline(**kwargs):
    state = ctboost.FeaturePipeline(**kwargs).to_state()
    state["feature_pipeline_format_version"] = 2
    state.pop("categorical_key_encoding_version")
    return ctboost.FeaturePipeline.from_state(state)


def test_codec2_scalar_keys_separate_missing_literals_and_escape_markers():
    values = np.asarray(
        [
            [None],
            [MISSING_SENTINEL],
            [r"\m"],
            [OTHER_SENTINEL],
            [r"\o"],
            ["plain"],
        ],
        dtype=object,
    )
    labels = np.arange(values.shape[0], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(cat_features=[0], one_hot_max_size=16)

    transformed, categorical, names = pipeline.fit_transform_array(values, labels)
    state = pipeline.to_state()

    assert state["feature_pipeline_format_version"] == 3
    assert state["categorical_key_encoding_version"] == 2
    assert categorical == []
    assert transformed.shape == (6, 6)
    assert np.unique(transformed, axis=0).shape[0] == values.shape[0]
    one_hot = state["one_hot_states"][0]
    assert set(one_hot["category_keys"]) == {
        MISSING_SENTINEL,
        r"\m",
        r"\\m",
        OTHER_SENTINEL,
        r"\\o",
        "plain",
    }
    assert len(names) == len(set(names))
    assert "f0_is_missing" in names
    assert "f0_is_literal_missing" in names
    assert "f0_is_literal_other" in names


def test_codec2_synthetic_other_bucket_does_not_alias_literal_other_or_escape_value():
    values = np.asarray(
        [[OTHER_SENTINEL]] * 4 + [["rare-a"], ["rare-b"], [r"\o"], [None]],
        dtype=object,
    )
    labels = np.arange(values.shape[0], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0],
        one_hot_max_size=2,
        max_cat_threshold=2,
    )
    pipeline.fit(values, labels)

    state = pipeline.to_state()
    one_hot = state["one_hot_states"][0]
    assert one_hot["category_keys"] == [OTHER_SENTINEL, r"\o"]
    assert one_hot["has_other_bucket"] is True
    assert one_hot["output_names"] == ["f0_is_literal_other", "f0_is_other"]

    probe = np.asarray(
        [[OTHER_SENTINEL], ["rare-a"], [r"\o"], [None]], dtype=object
    )
    transformed, _, _ = pipeline.transform_array(probe)
    np.testing.assert_array_equal(transformed[1], transformed[2])
    np.testing.assert_array_equal(transformed[1], transformed[3])
    assert not np.array_equal(transformed[0], transformed[1])


def test_codec2_composite_keys_escape_delimiters_backslashes_and_missing_literals():
    values = np.asarray(
        [
            ["a||b", "c"],
            ["a", "b||c"],
            ["a\\", "|b"],
            [None, MISSING_SENTINEL],
            [MISSING_SENTINEL, None],
        ],
        dtype=object,
    )
    labels = np.arange(values.shape[0], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0, 1],
        categorical_combinations=[[0, 1]],
    )

    transformed, categorical, _ = pipeline.fit_transform_array(values, labels)
    state = pipeline.to_state()
    mapping = state["combination_states"][0]["mapping"]

    assert len(mapping) == values.shape[0]
    combination_column = transformed[:, categorical[-1]]
    assert np.unique(combination_column).size == values.shape[0]
    assert r"a\|\|b||c" in mapping
    assert r"a||b\|\|c" in mapping
    assert r"a\\||\|b" in mapping


def test_codec2_one_hot_names_remain_unique_when_sanitized_tokens_collide():
    values = np.asarray([["a-b"], ["a_b"], ["a b"], ["a/b"]], dtype=object)
    pipeline = ctboost.FeaturePipeline(cat_features=[0], one_hot_max_size=8)
    pipeline.fit(values, np.arange(4, dtype=np.float32))

    names = pipeline.to_state()["one_hot_states"][0]["output_names"]
    assert len(names) == len(set(names)) == 4
    assert names == ["f0_is_a_b", "f0_is_a_b_2", "f0_is_a_b_3", "f0_is_a_b_4"]


def test_codec2_allocates_globally_unique_names_across_all_transform_families():
    feature_names = [
        "city",
        "city_ctr",
        "flag",
        "flag_is_x",
        "segment",
        "city_x_segment",
        "text",
        "text_hash0",
        "embedding",
        "embedding_mean",
    ]
    rows = []
    for index in range(6):
        rows.append(
            [
                ["a", "b", "c"][index % 3],
                float(index),
                ["x", "y"][index % 2],
                float(index + 10),
                ["s", "t"][index % 2],
                float(index + 20),
                ["red", "blue"][index % 2],
                float(index + 30),
                np.asarray([float(index), float(index + 1)], dtype=np.float32),
                float(index + 40),
            ]
        )
    values = np.asarray(rows, dtype=object)
    labels = np.linspace(0.25, 2.75, values.shape[0], dtype=np.float32)
    kwargs = {
        "cat_features": ["city", "flag", "segment"],
        "one_hot_max_size": 3,
        "categorical_combinations": [["city", "segment"]],
        "per_feature_ctr": {"city": ["Mean"]},
        "combinations_ctr": ["Frequency"],
        "text_features": ["text"],
        "text_hash_dim": 2,
        "embedding_features": ["embedding"],
        "embedding_stats": ("mean",),
    }

    pipeline = ctboost.FeaturePipeline(**kwargs)
    _, _, names = pipeline.fit_transform_array(
        values, labels, feature_names=feature_names
    )
    repeated = ctboost.FeaturePipeline(**kwargs)
    _, _, repeated_names = repeated.fit_transform_array(
        values, labels, feature_names=feature_names
    )

    assert names == repeated_names
    assert len(names) == len(set(names))
    assert {
        "city_ctr_2",
        "flag_is_x_2",
        "city_x_segment_2",
        "text_hash0_2",
        "embedding_mean_2",
    }.issubset(names)

    duplicate_state = copy.deepcopy(pipeline.to_state())
    duplicate_state["output_feature_names_"][-1] = duplicate_state[
        "output_feature_names_"
    ][0]
    with pytest.raises(ValueError, match="globally unique output names"):
        ctboost.FeaturePipeline.from_state(duplicate_state)

    codec1 = _legacy_unfitted_pipeline(**kwargs)
    _, _, codec1_names = codec1.fit_transform_array(
        values, labels, feature_names=feature_names
    )
    assert len(codec1_names) > len(set(codec1_names))
    assert ctboost.FeaturePipeline.from_state(codec1.to_state()).to_state() == codec1.to_state()


def test_ordinary_data_is_bit_exact_between_legacy_codec1_and_codec2():
    values = np.asarray(
        [
            ["alpha", "north", 0.0],
            ["beta", "south", 1.0],
            ["alpha", "south", 2.0],
            ["gamma", "north", 3.0],
            ["beta", "north", 4.0],
            ["gamma", "south", 5.0],
        ],
        dtype=object,
    )
    labels = np.asarray([0.0, 1.0, 0.5, 2.0, 1.5, 2.5], dtype=np.float32)
    kwargs = {
        "cat_features": [0, 1],
        "ordered_ctr": True,
        "categorical_combinations": [[0, 1]],
        "simple_ctr": ["Mean"],
        "combinations_ctr": ["Frequency"],
        "random_seed": 19,
    }
    codec2 = ctboost.FeaturePipeline(**kwargs)
    codec1 = _legacy_unfitted_pipeline(**kwargs)

    current_values, current_categorical, current_names = codec2.fit_transform_array(
        values, labels
    )
    legacy_values, legacy_categorical, legacy_names = codec1.fit_transform_array(
        values, labels
    )

    np.testing.assert_array_equal(current_values, legacy_values)
    assert current_categorical == legacy_categorical
    assert current_names == legacy_names
    assert codec2.to_state()["categorical_key_encoding_version"] == 2
    assert codec1.to_state()["categorical_key_encoding_version"] == 1

    current_state = codec2.to_state()
    legacy_state = codec1.to_state()
    for state in (current_state, legacy_state):
        state.pop("feature_pipeline_format_version")
        state.pop("categorical_key_encoding_version")
    assert current_state == legacy_state


def test_golden_codec1_state_load_preserves_historical_missing_sentinel_behavior():
    golden_state = {
        "feature_pipeline_format_version": 2,
        "cat_features": [0],
        "feature_names_in_": ["category"],
        "n_features_in_": 1,
        "cat_feature_indices_": [0],
        "output_feature_names_": ["category"],
        "numeric_indices": [0],
        "categorical_states": [
            {
                "source_index": 0,
                "output_name": "category",
                "has_other_bucket": False,
                "other_value": 0.0,
                "mapping": {MISSING_SENTINEL: 0.0, "alpha": 1.0},
            }
        ],
    }
    pipeline = ctboost.FeaturePipeline.from_state(golden_state)
    probe = np.asarray(
        [[None], [MISSING_SENTINEL], ["alpha"], ["unseen"]], dtype=object
    )

    transformed, categorical, names = pipeline.transform_array(
        probe, feature_names=["category"]
    )

    np.testing.assert_array_equal(
        transformed[:3, 0], np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    )
    assert np.isnan(transformed[3, 0])
    assert categorical == [0]
    assert names == ["category"]
    reserialized = pipeline.to_state()
    assert reserialized["feature_pipeline_format_version"] == 3
    assert reserialized["categorical_key_encoding_version"] == 1

    pipeline.fit(
        probe[:3],
        np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        feature_names=["category"],
    )
    refitted = pipeline.to_state()
    assert refitted["categorical_key_encoding_version"] == 1
    assert set(refitted["categorical_states"][0]["mapping"]) == {
        MISSING_SENTINEL,
        "alpha",
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"feature_pipeline_format_version": 4}, "format version"),
        (
            {
                "feature_pipeline_format_version": 3,
                "categorical_key_encoding_version": 99,
            },
            "key encoding version",
        ),
        (
            {
                "feature_pipeline_format_version": 2,
                "categorical_key_encoding_version": 2,
            },
            "formats before version 3",
        ),
    ],
)
def test_feature_pipeline_rejects_unknown_or_inconsistent_codec_versions(updates, message):
    state = ctboost.FeaturePipeline(cat_features=[0]).to_state()
    state.update(updates)
    with pytest.raises(ValueError, match=message):
        ctboost.FeaturePipeline.from_state(state)


def test_codec2_state_validation_rejects_ambiguous_one_hot_and_other_bucket_state():
    values = np.asarray([["keep"]] * 3 + [["rare-a"], ["rare-b"]], dtype=object)
    pipeline = ctboost.FeaturePipeline(
        cat_features=[0], one_hot_max_size=2, max_cat_threshold=2
    )
    pipeline.fit(values, np.arange(values.shape[0], dtype=np.float32))
    state = pipeline.to_state()

    duplicate_names = copy.deepcopy(state)
    duplicate_names["one_hot_states"][0]["output_names"][1] = duplicate_names[
        "one_hot_states"
    ][0]["output_names"][0]
    with pytest.raises(ValueError, match="unique one-hot output names"):
        ctboost.FeaturePipeline.from_state(duplicate_names)

    wrong_other_value = copy.deepcopy(state)
    wrong_other_value["one_hot_states"] = []
    wrong_other_value["output_feature_names_"] = ["f0"]
    wrong_other_value["cat_feature_indices_"] = [0]
    wrong_other_value["categorical_states"] = [
        {
            "source_index": 0,
            "output_name": "f0",
            "has_other_bucket": True,
            "other_value": 99.0,
            "mapping": {"keep": 0.0, r"\o": 1.0},
        }
    ]
    with pytest.raises(ValueError, match="other-bucket value is inconsistent"):
        ctboost.FeaturePipeline.from_state(wrong_other_value)


def test_codec2_survives_pickle_json_sklearn_warm_start_and_manifest(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    categories = np.asarray(
        [None, MISSING_SENTINEL, r"\m", OTHER_SENTINEL, r"\o", "plain"] * 4,
        dtype=object,
    )
    frame = pd.DataFrame(
        {
            "category": categories,
            "value": np.linspace(-1.0, 1.0, categories.size, dtype=np.float32),
        }
    )
    target = (
        frame["value"].to_numpy(dtype=np.float32)
        + 0.5 * (frame["category"] == MISSING_SENTINEL).to_numpy(dtype=np.float32)
    )
    model = ctboost.CTBoostRegressor(
        iterations=4,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
        ordered_ctr=True,
        random_seed=7,
    ).fit(frame, target)
    expected = model.predict(frame)

    pickle_model = pickle.loads(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    np.testing.assert_array_equal(pickle_model.predict(frame), expected)
    assert pickle_model._feature_pipeline.to_state()["categorical_key_encoding_version"] == 2

    model_path = tmp_path / "codec2-estimator.ctb"
    model.save_model(model_path)
    document = json.loads(model_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    np.testing.assert_array_equal(restored.predict(frame), expected)
    assert restored._feature_pipeline.to_state()["categorical_key_encoding_version"] == 2

    legacy_outer_path = tmp_path / "codec2-estimator-schema-v1.ctb"
    document["schema_version"] = 1
    legacy_outer_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version 1 cannot contain"):
        ctboost.CTBoostRegressor.load_model(legacy_outer_path)

    unsupported_path = tmp_path / "codec2-estimator-schema-unknown.ctb"
    document["schema_version"] = 99
    unsupported_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported CTBoost model schema version"):
        ctboost.CTBoostRegressor.load_model(unsupported_path)

    manifest = model.get_inference_manifest()
    preprocessing = manifest["input"]["preprocessing"]
    assert preprocessing["categorical_key_encoding_version"] == 2

    warm = ctboost.CTBoostRegressor(
        iterations=2,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
        ordered_ctr=True,
        random_seed=7,
    ).fit(frame, target, init_model=restored)
    assert warm._feature_pipeline.to_state()["categorical_key_encoding_version"] == 2


def test_model_schema_outer_inner_consistency_and_codec1_forward_compatibility(
    tmp_path: Path,
):
    values = np.asarray(
        [["alpha", 0.0], ["beta", 1.0], ["alpha", 2.0], ["gamma", 3.0]],
        dtype=object,
    )
    labels = np.asarray([0.0, 1.0, 1.5, 3.0], dtype=np.float32)
    pipeline = _legacy_unfitted_pipeline(cat_features=[0], random_seed=13)
    transformed, categorical, names = pipeline.fit_transform_array(values, labels)
    booster = ctboost.train(
        ctboost.Pool(
            transformed,
            labels,
            cat_features=categorical,
            feature_names=names,
        ),
        {
            "objective": "RMSE",
            "max_depth": 1,
            "alpha": 1.0,
            "random_seed": 13,
        },
        num_boost_round=2,
    )
    booster._feature_pipeline = pipeline

    codec1_schema2_path = tmp_path / "codec1-schema2.ctb"
    booster.save_model(codec1_schema2_path)
    document = json.loads(codec1_schema2_path.read_text(encoding="utf-8"))
    inner = _deserialize_json_value(document["feature_pipeline_state"])
    assert document["schema_version"] == 2
    assert inner["feature_pipeline_format_version"] == 3
    assert inner["categorical_key_encoding_version"] == 1
    restored = ctboost.load_model(codec1_schema2_path)
    assert restored._feature_pipeline.to_state()["categorical_key_encoding_version"] == 1
    np.testing.assert_array_equal(restored.predict(values), booster.predict(values))

    # Relabeling a schema-2/format-3 document as schema 1 must fail even when
    # the embedded codec is legacy codec 1.
    document["schema_version"] = 1
    reverse_relabel_path = tmp_path / "codec1-format3-schema1.ctb"
    reverse_relabel_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version 1 cannot contain"):
        ctboost.load_model(reverse_relabel_path)

    historical_inner = copy.deepcopy(inner)
    historical_inner["feature_pipeline_format_version"] = 2
    historical_inner.pop("categorical_key_encoding_version")
    document["feature_pipeline_state"] = _serialize_json_value(historical_inner)
    historical_path = tmp_path / "codec1-format2-schema1.ctb"
    historical_path.write_text(json.dumps(document), encoding="utf-8")
    historical = ctboost.load_model(historical_path)
    assert historical._feature_pipeline.to_state()["categorical_key_encoding_version"] == 1
    np.testing.assert_array_equal(historical.predict(values), booster.predict(values))

    invalid_inner = copy.deepcopy(historical_inner)
    invalid_inner["categorical_key_encoding_version"] = 2
    document["feature_pipeline_state"] = _serialize_json_value(invalid_inner)
    invalid_codec_path = tmp_path / "codec2-format2-schema1.ctb"
    invalid_codec_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema version 1 cannot contain"):
        ctboost.load_model(invalid_codec_path)


def test_sklearn_warm_start_preserves_loaded_legacy_codec1_keyspace():
    pd = pytest.importorskip("pandas")
    seed_frame = pd.DataFrame(
        {
            "category": [None, MISSING_SENTINEL, "alpha", "beta"] * 6,
            "value": np.linspace(-1.0, 1.0, 24, dtype=np.float32),
        }
    )
    seed_target = seed_frame["value"].to_numpy(dtype=np.float32)
    pipeline = _legacy_unfitted_pipeline(
        cat_features=["category"], ordered_ctr=True, random_seed=5
    )
    transformed, categorical, names = pipeline.fit_transform_array(
        seed_frame, seed_target
    )
    fitted_pipeline_state = copy.deepcopy(pipeline.to_state())
    seed = ctboost.train(
        ctboost.Pool(
            transformed,
            seed_target,
            cat_features=categorical,
            feature_names=names,
        ),
        {
            "objective": "RMSE",
            "max_depth": 2,
            "alpha": 1.0,
            "random_seed": 5,
        },
        num_boost_round=2,
    )
    seed._feature_pipeline = pipeline
    seed_state = dict(seed._handle.export_state())

    adversarial_values = [
        None,
        MISSING_SENTINEL,
        r"\m",
        OTHER_SENTINEL,
        r"\o",
        "new||value",
        "alpha",
        "beta",
    ] * 3
    continuation_frame = pd.DataFrame(
        {
            "category": adversarial_values,
            "value": np.linspace(-0.75, 1.25, len(adversarial_values), dtype=np.float32),
        }
    )
    continuation_target = continuation_frame["value"].to_numpy(dtype=np.float32)

    continued = ctboost.CTBoostRegressor(
        iterations=2,
        max_depth=2,
        alpha=1.0,
        cat_features=["category"],
        ordered_ctr=True,
        random_seed=5,
    ).fit(continuation_frame, continuation_target, init_model=seed)

    continued_state = dict(continued._booster._handle.export_state())
    assert pipeline.to_state() == fitted_pipeline_state
    assert continued._feature_pipeline is not pipeline
    assert continued._feature_pipeline.to_state() == fitted_pipeline_state
    assert continued_state["trees"][: len(seed_state["trees"])] == seed_state["trees"]
    assert continued_state["tree_learning_rates"][: len(seed_state["tree_learning_rates"])] == seed_state[
        "tree_learning_rates"
    ]
    assert np.all(np.isfinite(continued.predict(continuation_frame)))


def test_codec2_snapshot_refit_is_exact_with_colliding_values(tmp_path: Path):
    values = np.asarray(
        [[value, float(index)] for index, value in enumerate(
            [None, MISSING_SENTINEL, r"\m", OTHER_SENTINEL, r"\o", "plain"] * 5
        )],
        dtype=object,
    )
    labels = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float32)
    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 2,
        "alpha": 1.0,
        "cat_features": [0],
        "ordered_ctr": True,
        "random_seed": 13,
    }
    reference = ctboost.train(values, params, label=labels, num_boost_round=5)
    snapshot = tmp_path / "codec2-snapshot.ctb"
    ctboost.train(
        values,
        params,
        label=labels,
        num_boost_round=2,
        snapshot_path=snapshot,
        snapshot_interval=1,
    )
    resumed = ctboost.train(
        values,
        params,
        label=labels,
        num_boost_round=5,
        snapshot_path=snapshot,
        snapshot_interval=1,
        resume_from_snapshot=True,
    )

    np.testing.assert_array_equal(resumed.predict(values), reference.predict(values))
    resumed_state = dict(resumed._handle.export_state())
    reference_state = dict(reference._handle.export_state())
    assert resumed.loss_history == reference.loss_history
    assert resumed_state["trees"] == reference_state["trees"]
    assert resumed_state["tree_learning_rates"] == reference_state["tree_learning_rates"]
    assert resumed_state["rng_state"] == reference_state["rng_state"]
    assert resumed._feature_pipeline.to_state() == reference._feature_pipeline.to_state()
    assert resumed._feature_pipeline.to_state()["categorical_key_encoding_version"] == 2
