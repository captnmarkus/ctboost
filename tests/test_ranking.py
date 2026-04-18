import numpy as np

import ctboost
import ctboost._core as _core


def _make_ranking_data(num_groups: int = 24, group_size: int = 5):
    rows = []
    labels = []
    group_id = []
    for group in range(num_groups):
        for document in range(group_size):
            relevance = float(document)
            rows.append(
                [
                    relevance + 0.05 * float(group % 3),
                    float(group % 4),
                    float(document == group_size - 1),
                ]
            )
            labels.append(relevance)
            group_id.append(group)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(group_id, dtype=np.int64),
    )


def _pair_accuracy(scores: np.ndarray, labels: np.ndarray, group_id: np.ndarray) -> float:
    correct = 0
    total = 0
    for group in np.unique(group_id):
        mask = group_id == group
        group_scores = scores[mask]
        group_labels = labels[mask]
        for left in range(group_scores.shape[0]):
            for right in range(left + 1, group_scores.shape[0]):
                if group_labels[left] == group_labels[right]:
                    continue
                total += 1
                if (group_scores[left] - group_scores[right]) * (group_labels[left] - group_labels[right]) > 0:
                    correct += 1
    return correct / total


def _make_pair_only_ranking_data(num_groups: int = 32):
    rows = []
    labels = []
    group_id = []
    pairs = []
    for group in range(num_groups):
        base = 2 * group
        rows.append([1.0 + 0.02 * group, 0.0])
        rows.append([-1.0 - 0.02 * group, 1.0])
        labels.extend([0.0, 0.0])
        group_id.extend([group, group])
        pairs.append([base, base + 1])
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(group_id, dtype=np.int64),
        np.asarray(pairs, dtype=np.int64),
    )


def test_low_level_pairlogit_training_with_group_id():
    X, y, group_id = _make_ranking_data()
    train_mask = group_id < 18
    valid_mask = ~train_mask

    train_pool = ctboost.Pool(X[train_mask], y[train_mask], group_id=group_id[train_mask])
    valid_pool = ctboost.Pool(X[valid_mask], y[valid_mask], group_id=group_id[valid_mask])

    booster = ctboost.train(
        train_pool,
        {
            "objective": "PairLogit",
            "eval_metric": "NDCG",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=12,
        eval_set=valid_pool,
    )

    predictions = booster.predict(valid_pool)
    assert _pair_accuracy(predictions, y[valid_mask], group_id[valid_mask]) > 0.9
    assert booster.eval_loss_history
    assert booster.eval_metric_name == "NDCG"


def test_explicit_pairs_enable_pairlogit_training_without_label_order():
    X, y, group_id, pairs = _make_pair_only_ranking_data()
    pool = ctboost.Pool(X, y, group_id=group_id, pairs=pairs)

    booster = ctboost.train(
        pool,
        {
            "objective": "PairLogit",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=10,
    )

    predictions = booster.predict(pool)
    assert np.mean(predictions[pairs[:, 0]] > predictions[pairs[:, 1]]) > 0.95


def test_ranker_estimator_and_grouped_cv():
    X, y, group_id = _make_ranking_data()

    ranker = ctboost.CTBoostRanker(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    ranker.fit(X, y, group_id=group_id)

    predictions = ranker.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert _pair_accuracy(predictions, y, group_id) > 0.9

    cv_result = ctboost.cv(
        ctboost.Pool(X, y, group_id=group_id),
        {
            "objective": "PairLogit",
            "eval_metric": "NDCG",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=8,
        nfold=3,
    )
    assert np.all(np.isfinite(cv_result["valid_loss_mean"]))


def test_ranker_estimator_accepts_explicit_pairs():
    X, y, group_id, pairs = _make_pair_only_ranking_data()

    ranker = ctboost.CTBoostRanker(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    ranker.fit(X, y, group_id=group_id, pairs=pairs)

    predictions = ranker.predict(X)
    assert np.mean(predictions[pairs[:, 0]] > predictions[pairs[:, 1]]) > 0.95


def test_ranking_metrics_map_and_mrr_are_supported():
    X, y, group_id = _make_ranking_data()
    train_pool = ctboost.Pool(X, y, group_id=group_id)

    for metric_name in ("MAP", "MRR"):
        booster = ctboost.train(
            train_pool,
            {
                "objective": "PairLogit",
                "eval_metric": metric_name,
                "learning_rate": 0.2,
                "max_depth": 2,
                "alpha": 1.0,
                "lambda_l2": 1.0,
            },
            num_boost_round=6,
            eval_set=train_pool,
        )
        assert booster.eval_metric_name == metric_name
        assert booster.eval_loss_history
        assert np.isfinite(booster.eval_loss_history[-1])


def test_ranking_metrics_respect_subgroup_id_and_group_weight():
    predictions = np.asarray([1.5, 2.0, 0.0, 2.0, 1.0, 0.0], dtype=np.float32)
    labels = np.asarray([2.0, 1.0, 0.0, 2.0, 1.0, 0.0], dtype=np.float32)
    weights = np.ones(labels.shape[0], dtype=np.float32)
    group_id = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    subgroup_id = np.asarray([0, 0, 1, 0, 1, 2], dtype=np.int64)
    group_weight = np.asarray([4.0, 4.0, 4.0, 1.0, 1.0, 1.0], dtype=np.float32)

    pairlogit_without_subgroup = _core._evaluate_metric(
        predictions,
        labels,
        weights,
        "PairLogit",
        1,
        group_id,
        None,
        None,
        None,
        None,
    )
    pairlogit_with_subgroup = _core._evaluate_metric(
        predictions,
        labels,
        weights,
        "PairLogit",
        1,
        group_id,
        None,
        subgroup_id,
        None,
        None,
    )
    assert pairlogit_with_subgroup < pairlogit_without_subgroup

    ndcg_unweighted = _core._evaluate_metric(
        predictions,
        labels,
        weights,
        "NDCG",
        1,
        group_id,
        None,
        None,
        None,
        None,
    )
    ndcg_weighted = _core._evaluate_metric(
        predictions,
        labels,
        weights,
        "NDCG",
        1,
        group_id,
        group_weight,
        None,
        None,
        None,
    )
    assert ndcg_weighted < ndcg_unweighted
