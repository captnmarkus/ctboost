import copy
import pickle

import numpy as np
import pytest
from sklearn.base import clone

import ctboost


def test_learned_text_dictionary_tfidf_ngrams_are_persisted():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "text": ["Alpha beta", "alpha beta", "ALPHA gamma", "delta"],
            "numeric": np.arange(4, dtype=np.float32),
        }
    )
    label = np.asarray([0.0, 0.2, 0.7, 1.0], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        text_features=["text"],
        text_tokenizer="word",
        text_ngram_range=(1, 2),
        text_lowercase=True,
        text_min_token_count=2,
        text_max_dictionary_size=8,
        text_feature_calcer="tfidf",
    )

    transformed, cat_features, names = pipeline.fit_transform_array(frame, label)
    state = pipeline.to_state()
    text_state = state["text_states"][0]

    assert cat_features == []
    assert text_state["vocabulary"] == ["alpha", "alpha_beta", "beta"]
    assert text_state["uses_dictionary"] is True
    assert state["text_ngram_range"] == (1, 2)
    assert state["text_feature_calcer"] == "tfidf"
    assert names[1:] == [
        "text_token_alpha_0",
        "text_token_alpha_beta_1",
        "text_token_beta_2",
    ]
    # Alpha appears in three documents; the bigram and beta appear in two.
    np.testing.assert_allclose(
        text_state["idf_values"],
        [np.log(5.0 / 4.0) + 1.0, np.log(5.0 / 3.0) + 1.0, np.log(5.0 / 3.0) + 1.0],
        rtol=1e-6,
    )
    assert transformed[3, 1:].tolist() == [0.0, 0.0, 0.0]

    restored = ctboost.FeaturePipeline.from_state(state)
    restored_values, restored_cat_features, restored_names = restored.transform_array(frame)
    np.testing.assert_allclose(transformed, restored_values, rtol=1e-7, atol=1e-7)
    assert restored_cat_features == cat_features
    assert restored_names == names


def test_character_tokenizer_and_binary_calcer_are_configurable():
    data = np.asarray([["ABA"], ["aca"]], dtype=object)
    label = np.asarray([0.0, 1.0], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        text_features=[0],
        text_tokenizer="character",
        text_ngram_range=(2, 2),
        text_max_dictionary_size=8,
        text_feature_calcer="binary",
    )

    transformed, _, _ = pipeline.fit_transform_array(data, label)
    state = pipeline.to_state()
    assert state["text_states"][0]["vocabulary"] == ["ab", "ac", "ba", "ca"]
    # Repeated n-grams remain one in binary mode.
    assert set(np.unique(transformed)).issubset({0.0, 1.0})


def test_target_aware_embedding_projection_tracks_label_and_round_trips(tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(912)
    signal = np.linspace(-2.0, 2.0, 80, dtype=np.float32)
    nuisance = rng.normal(size=signal.size).astype(np.float32)
    frame = pd.DataFrame(
        {
            "embedding": [
                np.asarray([value, noise, 0.25 * value], dtype=np.float32)
                for value, noise in zip(signal, nuisance)
            ],
            "numeric": nuisance,
        }
    )
    label = 3.5 * signal + rng.normal(scale=0.03, size=signal.size).astype(np.float32)

    pipeline = ctboost.FeaturePipeline(
        embedding_features=["embedding"],
        embedding_stats=(),
        embedding_target_features=True,
        embedding_target_regularization=0.5,
    )
    transformed, _, names = pipeline.fit_transform_array(frame, label)
    projection_index = names.index("embedding_target_projection")
    assert abs(np.corrcoef(transformed[:, projection_index], label)[0, 1]) > 0.99

    state = pipeline.to_state()
    embedding_state = state["embedding_states"][0]
    assert len(embedding_state["center"]) == 3
    assert len(embedding_state["target_projection_weights"]) == 1
    restored = ctboost.FeaturePipeline.from_state(state)
    restored_values, _, restored_names = restored.transform_array(frame)
    np.testing.assert_allclose(transformed, restored_values, rtol=1e-6, atol=1e-6)
    assert restored_names == names

    model = ctboost.CTBoostRegressor(
        iterations=6,
        max_depth=2,
        alpha=1.0,
        embedding_features=["embedding"],
        embedding_stats=("mean",),
        embedding_target_features=True,
        embedding_target_regularization=0.5,
    )
    # Constructor parameters remain sklearn-clone compatible.
    assert clone(model).get_params()["embedding_target_features"] is True
    model.fit(frame, label)
    assert model._feature_pipeline.to_state()["embedding_target_mode"] == "regression"
    expected = model.predict(frame)
    path = tmp_path / "target_embedding.ctb"
    model.save_model(path)
    restored_model = ctboost.CTBoostRegressor.load_model(path)
    np.testing.assert_allclose(expected, restored_model.predict(frame), rtol=1e-6, atol=1e-6)


def test_multiclass_embedding_target_features_create_one_projection_per_class():
    embeddings = np.empty((9, 1), dtype=object)
    for index in range(9):
        class_index = index % 3
        embeddings[index, 0] = np.eye(3, dtype=np.float32)[class_index]
    labels = np.asarray([index % 3 for index in range(9)], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        embedding_features=[0],
        embedding_stats=(),
        embedding_target_features=True,
    )

    transformed, _, names = pipeline.fit_transform_array(embeddings, labels)
    assert transformed.shape == (9, 3)
    assert names == [
        "f0_target_projection_class0",
        "f0_target_projection_class1",
        "f0_target_projection_class2",
    ]
    assert np.array_equal(np.argmax(transformed, axis=1), labels.astype(np.int64))


def test_old_feature_pipeline_state_loads_with_legacy_text_and_embedding_defaults():
    data = np.empty((3, 2), dtype=object)
    data[:, 0] = ["Red fox", "blue fox", "red hare"]
    data[:, 1] = [
        np.asarray([0.1, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.2], dtype=np.float32),
        np.asarray([0.7, 0.8], dtype=np.float32),
    ]
    label = np.asarray([0.0, 1.0, 0.5], dtype=np.float32)
    pipeline = ctboost.FeaturePipeline(
        text_features=[0],
        text_hash_dim=8,
        embedding_features=[1],
        embedding_stats=("mean", "l2"),
    )
    expected, expected_cat, expected_names = pipeline.fit_transform_array(data, label)
    legacy_state = copy.deepcopy(pipeline.to_state())
    for key in (
        "feature_pipeline_format_version",
        "categorical_key_encoding_version",
        "text_tokenizer",
        "text_ngram_range",
        "text_lowercase",
        "text_min_token_count",
        "text_max_dictionary_size",
        "text_feature_calcer",
        "embedding_target_features",
        "embedding_target_regularization",
        "embedding_target_mode",
    ):
        legacy_state.pop(key, None)
    for text_state in legacy_state["text_states"]:
        for key in (
            "output_dim",
            "uses_dictionary",
            "filters_tokens",
            "vocabulary",
            "idf_values",
        ):
            text_state.pop(key, None)
    for embedding_state in legacy_state["embedding_states"]:
        for key in ("center", "target_projection_weights", "target_output_names"):
            embedding_state.pop(key, None)

    restored = ctboost.FeaturePipeline.from_state(legacy_state)
    actual, actual_cat, actual_names = restored.transform_array(data)
    np.testing.assert_allclose(expected, actual, rtol=1e-7, atol=1e-7)
    assert actual_cat == expected_cat
    assert actual_names == expected_names


def test_old_estimator_pickle_receives_new_preprocessing_defaults():
    model = ctboost.CTBoostRegressor(iterations=2)
    for name in (
        "text_tokenizer",
        "text_ngram_range",
        "text_lowercase",
        "text_min_token_count",
        "text_max_dictionary_size",
        "text_feature_calcer",
        "embedding_target_features",
        "embedding_target_regularization",
        "embedding_target_mode",
    ):
        delattr(model, name)

    restored = pickle.loads(pickle.dumps(model))
    params = restored.get_params(deep=False)
    assert params["text_tokenizer"] == "word"
    assert params["text_ngram_range"] == (1, 1)
    assert params["text_feature_calcer"] == "count"
    assert params["embedding_target_features"] is False
    assert params["embedding_target_mode"] == "auto"


def test_prepare_pool_exposes_text_dictionary_and_embedding_target_parameters():
    data = np.empty((6, 2), dtype=object)
    data[:, 0] = ["red fox", "red fox", "blue fox", "blue hare", "red hare", "fox"]
    data[:, 1] = [np.asarray([float(index), 1.0], dtype=np.float32) for index in range(6)]
    label = np.arange(6, dtype=np.float32)

    pool = ctboost.prepare_pool(
        data,
        label,
        text_features=[0],
        text_ngram_range=(1, 2),
        text_max_dictionary_size=5,
        text_feature_calcer="binary",
        embedding_features=[1],
        embedding_target_features=True,
    )
    state = pool._feature_pipeline.to_state()
    assert state["text_ngram_range"] == (1, 2)
    assert state["text_feature_calcer"] == "binary"
    assert state["embedding_target_features"] is True
    assert pool.num_cols == len(state["output_feature_names_"])


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("text_tokenizer", "sentencepiece", "text_tokenizer"),
        ("text_ngram_range", (2, 1), "text_ngram_range"),
        ("text_feature_calcer", "bm25", "text_feature_calcer"),
        ("text_min_token_count", 0, "text_min_token_count"),
        ("text_max_dictionary_size", -1, "text_max_dictionary_size"),
        ("embedding_target_regularization", -0.1, "embedding_target_regularization"),
        ("embedding_target_mode", "ranking", "embedding_target_mode"),
    ],
)
def test_text_and_embedding_configuration_validation(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        ctboost.FeaturePipeline(**{keyword: value})
