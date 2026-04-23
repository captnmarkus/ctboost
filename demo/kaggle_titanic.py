from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ctboost import CTBoostClassifier


def _read_csv(data_dir: Path, file_name: str) -> pd.DataFrame:
    path = data_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"expected {path}")
    return pd.read_csv(path)


def _collapse_rare_titles(titles: pd.Series, minimum_count: int = 10) -> pd.Series:
    counts = titles.value_counts(dropna=False)
    rare_titles = counts[counts < minimum_count].index
    return titles.replace(rare_titles, "Rare")


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()

    family_size = features["SibSp"].fillna(0) + features["Parch"].fillna(0) + 1
    features["FamilySize"] = family_size.astype(np.float32)
    features["IsAlone"] = (family_size == 1).astype(np.int8)
    features["HasCabin"] = features["Cabin"].notna().astype(np.int8)
    features["Deck"] = features["Cabin"].fillna("Unknown").str[0]
    features["FarePerPerson"] = features["Fare"].fillna(features["Fare"].median()) / family_size.clip(lower=1)
    features["Title"] = (
        features["Name"]
        .fillna("Unknown")
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .fillna("Unknown")
    )
    features["Title"] = _collapse_rare_titles(features["Title"])
    features["TicketPrefix"] = (
        features["Ticket"]
        .fillna("Unknown")
        .str.replace(r"[\.\d/ ]", "", regex=True)
        .replace("", "None")
    )

    features = features.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    categorical_columns = [
        "Pclass",
        "Sex",
        "Embarked",
        "Deck",
        "Title",
        "TicketPrefix",
    ]
    for column in categorical_columns:
        features[column] = features[column].fillna("Missing").astype("category")

    return features


def _print_top_features(model: CTBoostClassifier, columns: pd.Index, limit: int = 8) -> None:
    importances = np.asarray(model.feature_importances_, dtype=np.float32)
    if importances.shape[0] == len(columns):
        index = columns
    else:
        index = [f"feature_{idx}" for idx in range(importances.shape[0])]
    importances = pd.Series(importances, index=index).sort_values(ascending=False).head(limit)
    print("\nTop features:")
    for name, value in importances.items():
        print(f"  {name}: {value:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CTBoost on the Kaggle Titanic competition.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with train/test Titanic CSV files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the submission CSV.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    args = parser.parse_args()

    train_frame = _read_csv(args.data_dir, "train.csv")
    test_frame = _read_csv(args.data_dir, "test.csv")

    target = train_frame["Survived"].astype(np.float32)
    test_ids = test_frame["PassengerId"].copy()

    X = _prepare_features(train_frame.drop(columns=["Survived"]))
    X_test = _prepare_features(test_frame)
    cat_features = [column for column in X.columns if pd.api.types.is_categorical_dtype(X[column].dtype)]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        target,
        test_size=args.valid_size,
        random_state=args.random_seed,
        stratify=target,
    )

    model = CTBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        max_depth=4,
        alpha=1.0,
        lambda_l2=2.0,
        eval_metric="AUC",
        cat_features=cat_features,
        random_seed=args.random_seed,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=60,
    )

    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    valid_predictions = (valid_probabilities >= 0.5).astype(np.int32)
    print(f"Validation rows: {len(X_valid)}")
    print(f"Validation accuracy: {accuracy_score(y_valid, valid_predictions):.4f}")
    print(f"Validation ROC AUC: {roc_auc_score(y_valid, valid_probabilities):.4f}")
    if model.best_iteration_ is not None:
        print(f"Best iteration: {model.best_iteration_ + 1}")

    test_predictions = model.predict(X_test).astype(np.int32)
    output_path = args.output or (args.data_dir / "submission_ctboost_titanic.csv")

    if (args.data_dir / "gender_submission.csv").exists():
        submission = _read_csv(args.data_dir, "gender_submission.csv")
        submission["PassengerId"] = test_ids.to_numpy()
        submission["Survived"] = test_predictions
    else:
        submission = pd.DataFrame({"PassengerId": test_ids, "Survived": test_predictions})

    submission.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")
    _print_top_features(model, X.columns)


if __name__ == "__main__":
    main()
