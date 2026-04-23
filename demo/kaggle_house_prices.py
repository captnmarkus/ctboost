from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from ctboost import CTBoostRegressor


def _read_csv(data_dir: Path, file_name: str) -> pd.DataFrame:
    path = data_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"expected {path}")
    return pd.read_csv(path)


def _series_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(0)
    return pd.Series(0, index=frame.index, dtype=np.float32)


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()

    features["TotalSF"] = (
        _series_or_zero(features, "TotalBsmtSF")
        + _series_or_zero(features, "1stFlrSF")
        + _series_or_zero(features, "2ndFlrSF")
    ).astype(np.float32)
    features["TotalBath"] = (
        _series_or_zero(features, "FullBath")
        + 0.5 * _series_or_zero(features, "HalfBath")
        + _series_or_zero(features, "BsmtFullBath")
        + 0.5 * _series_or_zero(features, "BsmtHalfBath")
    ).astype(np.float32)
    features["TotalPorchSF"] = (
        _series_or_zero(features, "OpenPorchSF")
        + _series_or_zero(features, "EnclosedPorch")
        + _series_or_zero(features, "3SsnPorch")
        + _series_or_zero(features, "ScreenPorch")
    ).astype(np.float32)
    features["HouseAge"] = (features["YrSold"] - features["YearBuilt"]).astype(np.float32)
    features["RemodelAge"] = (features["YrSold"] - features["YearRemodAdd"]).astype(np.float32)
    features["HasGarage"] = (_series_or_zero(features, "GarageArea") > 0).astype(np.int8)
    features["HasBasement"] = (_series_or_zero(features, "TotalBsmtSF") > 0).astype(np.int8)

    if "Id" in features.columns:
        features = features.drop(columns=["Id"])
    if "MSSubClass" in features.columns:
        features["MSSubClass"] = features["MSSubClass"].astype(str)

    object_columns = features.select_dtypes(include=["object"]).columns
    for column in object_columns:
        features[column] = features[column].fillna("Missing").astype("category")

    return features


def _print_top_features(model: CTBoostRegressor, columns: pd.Index, limit: int = 10) -> None:
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
    parser = argparse.ArgumentParser(description="Train CTBoost on Kaggle House Prices data.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with train/test House Prices CSV files.")
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

    test_ids = test_frame["Id"].copy()
    y = np.log1p(train_frame["SalePrice"].astype(np.float32))

    X = _prepare_features(train_frame.drop(columns=["SalePrice"]))
    X_test = _prepare_features(test_frame)
    cat_features = [column for column in X.columns if pd.api.types.is_categorical_dtype(X[column].dtype)]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=args.valid_size,
        random_state=args.random_seed,
    )

    model = CTBoostRegressor(
        iterations=1600,
        learning_rate=0.03,
        max_depth=5,
        alpha=1.0,
        lambda_l2=3.0,
        cat_features=cat_features,
        random_seed=args.random_seed,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=80,
    )

    valid_predictions_log = model.predict(X_valid)
    valid_predictions = np.expm1(valid_predictions_log)
    valid_actual = np.expm1(y_valid)
    print(f"Validation rows: {len(X_valid)}")
    print(f"Validation RMSE (log target): {mean_squared_error(y_valid, valid_predictions_log, squared=False):.4f}")
    print(f"Validation RMSE (price): {mean_squared_error(valid_actual, valid_predictions, squared=False):.2f}")
    if model.best_iteration_ is not None:
        print(f"Best iteration: {model.best_iteration_ + 1}")

    test_predictions = np.expm1(model.predict(X_test)).clip(min=0.0)
    output_path = args.output or (args.data_dir / "submission_ctboost_house_prices.csv")

    if (args.data_dir / "sample_submission.csv").exists():
        submission = _read_csv(args.data_dir, "sample_submission.csv")
        submission["Id"] = test_ids.to_numpy()
        submission["SalePrice"] = test_predictions
    else:
        submission = pd.DataFrame({"Id": test_ids, "SalePrice": test_predictions})

    submission.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")
    _print_top_features(model, X.columns)


if __name__ == "__main__":
    main()
