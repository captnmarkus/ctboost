"""Wide multiclass CTR inference preserves per-output arithmetic and routing."""

import copy

import numpy as np
import pytest

from ctboost import FeaturePipeline


@pytest.mark.parametrize("classes", [3, 8])
@pytest.mark.parametrize("format_version", [3, 4])
@pytest.mark.parametrize("strength", [0.0, 0.2, 2.0])
def test_wide_multiclass_ctr_matches_direct_category_statistics(
    classes, format_version, strength
):
    categories = ["red", "blue", None, "__ctboost_missing__", "a||b", "\\literal", "1"]
    width, rows = 32, 37
    train = np.asarray(
        [[categories[(row * (column % 3 + 1) + column) % len(categories)]
          for column in range(width)] for row in range(rows)],
        dtype=object,
    )
    labels = np.arange(rows) % classes
    pipeline = FeaturePipeline(
        cat_features=list(range(width)),
        simple_ctr=["Mean", "Frequency"],
        categorical_combinations=[[0, 1], [2, 3]],
        combinations_ctr=["Frequency", "Mean"],
        ctr_prior_strength=strength,
        random_seed=17,
    ).fit(train, labels)
    state = pipeline.to_state()
    state["feature_pipeline_format_version"] = format_version
    pipeline = FeaturePipeline.from_state(state)
    before = copy.deepcopy(pipeline.to_state())
    query = np.asarray(
        [train[0].tolist(), ["unseen"] * width, [None] * width,
         [np.nan] * width, ["__ctboost_missing__"] * width,
         ["a||b" if column % 2 else "\\literal" for column in range(width)]],
        dtype=object,
    )
    # Reverse row strides exercise the matrix view independently of output order.
    query = query[::-1]
    transformed, _, names = pipeline.transform_array(query)
    assert transformed.flags.f_contiguous and transformed.dtype == np.float32
    targets = np.eye(classes, dtype=np.float32)[labels]
    missing = object()

    def category(value):
        return missing if value is None or isinstance(value, float) and np.isnan(value) else value

    # Derive sufficient statistics directly from raw categories and labels, not
    # the transform or its escaped dictionary keys. Literal sentinels and joined
    # category separators must stay distinct from missing values and tuples.
    for ctr in state["ctr_states"]:
        columns = ctr["source_indices"]
        training_keys = [tuple(category(row[column]) for column in columns) for row in train]
        expected = []
        for row in query:
            key = tuple(category(row[column]) for column in columns)
            selected = np.asarray([key == item for item in training_keys])
            count = float(np.float32(selected.sum()))
            if ctr["ctr_type"] == "Mean":
                sums = targets[selected].sum(axis=0).astype(np.float64)
                prior = np.asarray(ctr["prior_values"], dtype=np.float64)
                denominator = count + strength
                denominator = (
                    max(denominator, 1.0) if format_version == 3
                    else denominator if denominator > 0.0 else 1.0
                )
                values = (sums + strength * prior) / denominator
            else:
                frequency = float(np.float32(count) / np.float32(rows))
                values = [(count + strength * frequency) / max(rows + strength, 1.0)]
            expected.append(values)
        output_columns = [names.index(name) for name in ctr["output_names"]]
        np.testing.assert_array_equal(
            transformed[:, output_columns].view(np.uint32),
            np.asarray(expected, dtype=np.float32).view(np.uint32),
        )
    assert pipeline.to_state() == before
    empty, empty_categories, empty_names = pipeline.transform_array(query[:0])
    assert empty.shape == (0, transformed.shape[1])
    assert empty_names == names
    assert empty_categories == pipeline.cat_feature_indices_
