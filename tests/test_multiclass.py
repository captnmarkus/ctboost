import numpy as np
from sklearn.datasets import make_classification

import ctboost


def test_multiclass_classifier_predict_proba_shape_and_accuracy():
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=17,
    )
    X = X.astype(np.float32)

    clf = ctboost.CBoostClassifier(
        iterations=30,
        learning_rate=0.2,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    probabilities = clf.predict_proba(X)
    predictions = clf.predict(X)
    accuracy = np.mean(predictions == y)

    assert probabilities.shape == (X.shape[0], 3)
    assert predictions.shape == (X.shape[0],)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert accuracy > 0.7


def test_categorical_split_routes_non_ordinal_categories_better_than_random_split():
    rng = np.random.default_rng(5)
    category_names = np.array(["red", "blue", "green", "yellow"])
    raw_categories = rng.choice(category_names, size=400, replace=True)
    mapping = {name: index for index, name in enumerate(category_names)}
    encoded_categories = np.array([mapping[name] for name in raw_categories], dtype=np.float32)
    X = encoded_categories.reshape(-1, 1)
    y = np.isin(raw_categories, ["red", "green"]).astype(np.float32)

    pool = ctboost.Pool(X, y, cat_features=[0])
    clf = ctboost.CBoostClassifier(
        iterations=1,
        learning_rate=1.0,
        max_depth=1,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(pool)

    probabilities = clf.predict_proba(pool)
    predictions = clf.predict(pool).astype(np.float32)
    accuracy = np.mean(predictions == y)

    all_categories = set(range(category_names.size))
    target_left = {mapping["blue"], mapping["yellow"]}
    random_split_rng = np.random.default_rng(11)
    while True:
        random_left = set(
            random_split_rng.choice(category_names.size, size=2, replace=False).tolist()
        )
        if random_left != target_left and random_left != (all_categories - target_left):
            break

    baseline = np.isin(encoded_categories.astype(np.int64), list(random_left)).astype(np.float32)
    baseline_accuracy = max(np.mean(baseline == y), np.mean((1.0 - baseline) == y))

    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert accuracy > 0.95
    assert accuracy > baseline_accuracy + 0.2
