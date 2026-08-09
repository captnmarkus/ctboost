"""Model-selection and visualization conveniences for CTBoost estimators."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, check_cv, cross_validate
from sklearn.utils.validation import check_is_fitted

from ..core import Pool


def _named_estimators(estimators: Any) -> Dict[str, Any]:
    if isinstance(estimators, Mapping):
        resolved = {}
        for raw_name, estimator in estimators.items():
            name = str(raw_name)
            if name in resolved:
                raise ValueError(
                    "estimator names must remain unique after string conversion"
                )
            resolved[name] = estimator
    else:
        try:
            values = list(estimators)
        except TypeError as exc:
            raise TypeError("estimators must be a mapping or iterable") from exc
        resolved = {}
        for index, estimator in enumerate(values):
            base_name = type(estimator).__name__
            name = base_name
            suffix = 2
            while name in resolved:
                name = "%s-%d" % (base_name, suffix)
                suffix += 1
            resolved[name] = estimator
    if not resolved:
        raise ValueError("estimators must contain at least one model")
    if any(not name for name in resolved):
        raise ValueError("estimator names must be non-empty")
    return resolved


def compare_estimators(
    estimators: Any,
    X: Any,
    y: Any,
    *,
    cv: Any = 3,
    scoring: Any = None,
    groups: Any = None,
    n_jobs: Optional[int] = None,
    return_train_score: bool = False,
    error_score: Any = "raise",
    plot: bool = False,
) -> Dict[str, Any]:
    """Compare sklearn-compatible estimators on identical cross-validation folds.

    Scores retain sklearn's convention that larger is better (loss scorers are
    negated). The returned rows contain every fold value plus mean and standard
    deviation, fit/score timings, and are sorted by the first test metric.
    """

    resolved = _named_estimators(estimators)
    first_estimator = next(iter(resolved.values()))
    splitter = check_cv(cv=cv, y=y, classifier=is_classifier(first_estimator))
    folds = list(splitter.split(X, y, groups))
    if not folds:
        raise ValueError("cv must yield at least one train/test split")
    rows = []
    primary_metric: Optional[str] = None
    for name, estimator in resolved.items():
        scores = cross_validate(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=folds,
            n_jobs=n_jobs,
            return_train_score=bool(return_train_score),
            error_score=error_score,
        )
        metric_names = [
            key[5:]
            for key in scores
            if key.startswith("test_")
        ]
        if not metric_names:
            raise RuntimeError("cross-validation did not return any test scores")
        if primary_metric is None:
            primary_metric = metric_names[0]
        metrics = {}
        for metric_name in metric_names:
            values = np.asarray(scores["test_" + metric_name], dtype=np.float64)
            metric_result: Dict[str, Any] = {
                "values": values.copy(),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            train_key = "train_" + metric_name
            if train_key in scores:
                train_values = np.asarray(scores[train_key], dtype=np.float64)
                metric_result.update(
                    train_values=train_values.copy(),
                    train_mean=float(np.mean(train_values)),
                    train_std=float(np.std(train_values)),
                )
            metrics[metric_name] = metric_result
        fit_times = np.asarray(scores["fit_time"], dtype=np.float64)
        score_times = np.asarray(scores["score_time"], dtype=np.float64)
        rows.append(
            {
                "name": name,
                "estimator": estimator,
                "metrics": metrics,
                "fit_time": {
                    "values": fit_times.copy(),
                    "mean": float(np.mean(fit_times)),
                    "std": float(np.std(fit_times)),
                },
                "score_time": {
                    "values": score_times.copy(),
                    "mean": float(np.mean(score_times)),
                    "std": float(np.std(score_times)),
                },
            }
        )
    assert primary_metric is not None
    rows.sort(
        key=lambda row: (
            not np.isfinite(row["metrics"][primary_metric]["mean"]),
            -row["metrics"][primary_metric]["mean"]
            if np.isfinite(row["metrics"][primary_metric]["mean"])
            else 0.0,
            row["name"],
        )
    )
    response: Dict[str, Any] = {
        "primary_metric": primary_metric,
        "best_model": rows[0]["name"],
        "results": rows,
    }
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise ImportError("plot=True requires matplotlib") from exc
        _, axis = plt.subplots()
        axis.barh(
            [row["name"] for row in reversed(rows)],
            [row["metrics"][primary_metric]["mean"] for row in reversed(rows)],
            xerr=[row["metrics"][primary_metric]["std"] for row in reversed(rows)],
        )
        axis.set(xlabel=primary_metric, title="Estimator comparison")
        response["plot"] = axis
    return response


def _cv_results_dict(search: Any) -> Dict[str, Any]:
    """Return a detached cv-results mapping that is safe to serialize."""

    result: Dict[str, Any] = {}
    for key, value in search.cv_results_.items():
        if isinstance(value, np.ndarray):
            result[key] = value.copy()
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _feature_names(X: Any, feature_count: int) -> Sequence[str]:
    names = getattr(X, "columns", None)
    if names is None and isinstance(X, Pool):
        names = X.feature_names
    if names is None:
        return [str(index) for index in range(feature_count)]
    return [str(name) for name in names]


def _resolve_feature_indices(
    features: Optional[Iterable[Any]],
    names: Sequence[str],
    feature_count: int,
) -> Sequence[int]:
    if features is None:
        return list(range(feature_count))
    name_to_index = {name: index for index, name in enumerate(names)}
    resolved = []
    for feature in features:
        if isinstance(feature, (int, np.integer)):
            index = int(feature)
        elif isinstance(feature, str):
            if feature not in name_to_index:
                raise ValueError("features_for_select references unknown feature %r" % feature)
            index = name_to_index[feature]
        else:
            raise TypeError("features_for_select entries must be feature indices or names")
        if index < 0 or index >= feature_count:
            raise ValueError("features_for_select contains an out-of-range feature index")
        if index not in resolved:
            resolved.append(index)
    if not resolved:
        raise ValueError("features_for_select cannot be empty")
    return resolved


def _subset_columns(X: Any, indices: Sequence[int], names: Sequence[str]) -> Any:
    if isinstance(X, Pool):
        raise TypeError("feature selection requires raw X input, not a Pool")
    if hasattr(X, "iloc"):
        return X.iloc[:, list(indices)]
    module = type(X).__module__.split(".", 1)[0]
    if module == "polars" and hasattr(X, "select"):
        return X.select([names[index] for index in indices])
    if module == "pyarrow" and hasattr(X, "select"):
        return X.select(list(indices))
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError("X must be a 2D feature matrix")
    return array[:, list(indices)]


class _ModelSelectionMixin:
    """CatBoost-style search helpers backed by scikit-learn's stable APIs."""

    def _adopt_best_estimator(self, estimator: Any) -> None:
        # Keep object identity so ``model.grid_search(...); model.predict(...)``
        # behaves like CatBoost's convenience API.
        self.__dict__.clear()
        self.__dict__.update(estimator.__dict__)

    def compare(
        self,
        estimators: Any,
        X: Any,
        y: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Cross-validate this estimator alongside named competitor models."""

        if isinstance(estimators, Mapping):
            competitors = _named_estimators(estimators)
            self_name = "CTBoost"
            suffix = 2
            while self_name in competitors:
                self_name = "CTBoost-%d" % suffix
                suffix += 1
            candidates = {self_name: self, **competitors}
        else:
            candidates = [self, *list(estimators)]
        return compare_estimators(candidates, X, y, **kwargs)

    @staticmethod
    def _plot_search_scores(cv_results: Mapping[str, Any], *, title: str) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise ImportError("plot=True requires matplotlib") from exc
        scores = np.asarray(cv_results["mean_test_score"], dtype=np.float64)
        errors = np.asarray(cv_results.get("std_test_score", np.zeros_like(scores)), dtype=np.float64)
        figure, axis = plt.subplots()
        positions = np.arange(scores.size)
        axis.errorbar(positions, scores, yerr=errors, marker="o", linestyle="none")
        axis.set(xlabel="candidate", ylabel="mean CV score", title=title)
        return axis

    def grid_search(
        self,
        param_grid: Any,
        X: Any,
        y: Any = None,
        *,
        cv: Any = 3,
        scoring: Any = None,
        refit: bool = True,
        n_jobs: Optional[int] = None,
        verbose: Any = False,
        return_train_score: bool = True,
        error_score: Any = "raise",
        plot: bool = False,
        groups: Any = None,
        **fit_params: Any,
    ) -> Dict[str, Any]:
        """Run an exhaustive parameter search and optionally refit this estimator."""

        search = GridSearchCV(
            clone(self),
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=int(verbose),
            return_train_score=return_train_score,
            error_score=error_score,
        )
        search.fit(X, y, groups=groups, **fit_params)
        results = _cv_results_dict(search)
        response: Dict[str, Any] = {
            "params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
            "cv_results": results,
        }
        if refit:
            self._adopt_best_estimator(search.best_estimator_)
        if plot:
            response["plot"] = self._plot_search_scores(results, title="CTBoost grid search")
        return response

    def randomized_search(
        self,
        param_distributions: Any,
        X: Any,
        y: Any = None,
        *,
        n_iter: int = 10,
        cv: Any = 3,
        scoring: Any = None,
        refit: bool = True,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: Any = False,
        return_train_score: bool = True,
        error_score: Any = "raise",
        plot: bool = False,
        groups: Any = None,
        **fit_params: Any,
    ) -> Dict[str, Any]:
        """Run a sampled parameter search and optionally refit this estimator."""

        search = RandomizedSearchCV(
            clone(self),
            param_distributions=param_distributions,
            n_iter=int(n_iter),
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=int(verbose),
            random_state=random_state,
            return_train_score=return_train_score,
            error_score=error_score,
        )
        search.fit(X, y, groups=groups, **fit_params)
        results = _cv_results_dict(search)
        response: Dict[str, Any] = {
            "params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "best_index": int(search.best_index_),
            "cv_results": results,
        }
        if refit:
            self._adopt_best_estimator(search.best_estimator_)
        if plot:
            response["plot"] = self._plot_search_scores(results, title="CTBoost randomized search")
        return response

    def select_features(
        self,
        X: Any,
        y: Any,
        *,
        features_for_select: Optional[Iterable[Any]] = None,
        num_features_to_select: Optional[int] = None,
        scoring: Any = None,
        n_repeats: int = 5,
        random_state: Optional[int] = None,
        sample_weight: Any = None,
        train_final_model: bool = False,
        plot: bool = False,
        **fit_params: Any,
    ) -> Dict[str, Any]:
        """Rank raw features by fitted-model permutation importance.

        The ranking operates in raw input space, so categorical/text/embedding
        pipelines are measured as users see them instead of exposing generated
        internal columns.  ``train_final_model`` is supported for plain numeric
        estimators; transformed-feature pipelines keep the original fitted model.
        """

        feature_count, _ = self._input_feature_metadata(X)
        names = _feature_names(X, feature_count)
        candidates = _resolve_feature_indices(features_for_select, names, feature_count)
        target_count = len(candidates) if num_features_to_select is None else int(num_features_to_select)
        if target_count <= 0 or target_count > len(candidates):
            raise ValueError("num_features_to_select must be between 1 and the candidate count")

        fitted = clone(self)
        if sample_weight is not None:
            fit_params = dict(fit_params)
            fit_params["sample_weight"] = sample_weight
        fitted.fit(X, y, **fit_params)
        importance = permutation_importance(
            fitted,
            X,
            y,
            scoring=scoring,
            n_repeats=int(n_repeats),
            random_state=random_state,
            sample_weight=sample_weight,
        )
        means = np.asarray(importance.importances_mean, dtype=np.float64)
        standard_deviations = np.asarray(importance.importances_std, dtype=np.float64)
        ordered = sorted(candidates, key=lambda index: (-means[index], index))
        selected = ordered[:target_count]
        eliminated = ordered[target_count:]
        response: Dict[str, Any] = {
            "selected_features": list(selected),
            "selected_features_names": [names[index] for index in selected],
            "eliminated_features": list(eliminated),
            "eliminated_features_names": [names[index] for index in eliminated],
            "feature_importances": {
                names[index]: {
                    "mean": float(means[index]),
                    "std": float(standard_deviations[index]),
                }
                for index in candidates
            },
        }

        if train_final_model:
            if fitted._uses_feature_pipeline():
                raise ValueError(
                    "train_final_model=True is not yet supported with categorical, text, "
                    "or embedding feature pipelines; use selected_features to subset X"
                )
            selected_X = _subset_columns(X, selected, names)
            self.fit(selected_X, y, **fit_params)
            self.selected_features_ = np.asarray(selected, dtype=np.int64)
            self.selected_feature_names_ = np.asarray(
                [names[index] for index in selected], dtype=object
            )
        else:
            self._adopt_best_estimator(fitted)

        if plot:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:  # pragma: no cover - dependency-specific
                raise ImportError("plot=True requires matplotlib") from exc
            figure, axis = plt.subplots()
            plot_order = list(reversed(ordered))
            axis.barh(
                [names[index] for index in plot_order],
                [means[index] for index in plot_order],
                xerr=[standard_deviations[index] for index in plot_order],
            )
            axis.set(xlabel="permutation importance", title="CTBoost feature selection")
            response["plot"] = axis
        return response

    def plot_metrics(
        self,
        *,
        metrics: Optional[Iterable[str]] = None,
        datasets: Optional[Iterable[str]] = None,
        ax: Any = None,
    ) -> Any:
        """Plot stored training/evaluation histories and return the axes."""

        check_is_fitted(self, attributes="_booster")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency-specific
            raise ImportError("plot_metrics requires matplotlib") from exc
        if ax is None:
            _, ax = plt.subplots()
        metric_filter = None if metrics is None else {str(value) for value in metrics}
        dataset_filter = None if datasets is None else {str(value) for value in datasets}
        plotted = 0
        for dataset_name, histories in self.get_evals_result().items():
            if dataset_filter is not None and dataset_name not in dataset_filter:
                continue
            for metric_name, history in histories.items():
                if metric_filter is not None and metric_name not in metric_filter:
                    continue
                ax.plot(np.arange(len(history)), history, label="%s:%s" % (dataset_name, metric_name))
                plotted += 1
        if plotted == 0:
            raise ValueError("no stored metric histories matched the requested filters")
        ax.set(xlabel="iteration", ylabel="metric value", title="CTBoost metrics")
        ax.legend()
        return ax


__all__ = ["compare_estimators"]
