# Demo Workflows

These demos replace the old remote-kernel Kaggle helpers with local, reproducible workflows.
Each example expects competition CSV files in a directory you control and writes a local
submission file that you can inspect before uploading.

## Prerequisites

Install the package with the demo dependencies:

```bash
pip install -e .[dev,sklearn]
```

## Included Examples

- `kaggle_titanic.py`: binary classification for the Titanic competition using mixed
  categorical and numeric inputs.
- `kaggle_house_prices.py`: regression for the House Prices competition using a
  log-target workflow and lightweight feature engineering.

## Expected Data Layout

### Titanic

Place the competition CSVs in one folder, for example:

```text
data/titanic/
  train.csv
  test.csv
  gender_submission.csv
```

Run:

```bash
python demo/kaggle_titanic.py --data-dir data/titanic
```

### House Prices

Place the competition CSVs in one folder, for example:

```text
data/house-prices/
  train.csv
  test.csv
  sample_submission.csv
```

Run:

```bash
python demo/kaggle_house_prices.py --data-dir data/house-prices
```

Both scripts print validation metrics and write a submission CSV next to the dataset unless
you override the output path.
