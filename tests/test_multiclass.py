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


def test_multiclass_iteration_shares_tree_structure_across_classes():
    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=29,
    )
    X = X.astype(np.float32)

    clf = ctboost.CBoostClassifier(
        iterations=1,
        learning_rate=0.3,
        max_depth=3,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=19,
    )
    clf.fit(X, y)

    state = dict(clf._booster._handle.export_state())
    trees = state["trees"]
    assert len(trees) == 3

    def structure_signature(tree_state):
        signature = []
        for node in tree_state["nodes"]:
            signature.append(
                (
                    node["is_leaf"],
                    node["is_categorical_split"],
                    node["split_feature_id"],
                    node["split_bin_index"],
                    node["left_child"],
                    node["right_child"],
                    tuple(node["left_categories"]),
                )
            )
        return signature

    signatures = [structure_signature(tree_state) for tree_state in trees]
    assert signatures[0] == signatures[1] == signatures[2]

    leaf_weights = [
        tuple(node["leaf_weight"] for node in tree_state["nodes"] if node["is_leaf"])
        for tree_state in trees
    ]
    assert len(set(leaf_weights)) > 1
