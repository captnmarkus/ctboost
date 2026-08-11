# Categorical, text, and embedding features

CTBoost can fit a preprocessing pipeline together with the booster. The pipeline is
serialized with the estimator and reused for validation and prediction.

## Categorical features

```python
model = CTBoostClassifier(
    cat_features=["city", "segment"],
    ordered_ctr=True,
    one_hot_max_size=8,
    max_cat_threshold=64,
)
```

Available categorical transforms include one-hot values, smoothed target statistics,
ordered CTRs, feature combinations, and per-feature CTR configuration. Unknown and
missing values have deterministic routes.

Fitted pipelines record `categorical_key_encoding_version` in their state and
inference manifest. New fits use version 2, which keeps actual missing values,
literal strings such as `__ctboost_missing__` and `__ctboost_other__`, synthetic
"other" buckets, backslashes, and categorical-combination delimiters distinct.
Pipelines saved before this field existed load as version 1 and retain their
historical key behavior for prediction and warm starts. CTBoost model documents use
schema version 2 for new saves while the current runtime continues to read schema
version 1 artifacts.

## Text features

```python
model = CTBoostClassifier(
    text_features=["title", "description"],
    text_tokenizer="word",
    text_ngram_range=(1, 2),
    text_lowercase=True,
    text_feature_calcer="tfidf",
    text_min_token_count=2,
    text_max_dictionary_size=50_000,
)
```

Tokenizers are intentionally deterministic and dependency-free. CTBoost currently
provides word, whitespace, and character tokenization; count, binary, and raw
count-times-IDF features. It does not claim CatBoost's tokenizer/dictionary breadth.

## Embeddings

Fixed-width embedding columns can produce descriptive statistics and optional
target-supervised projections. Supervised projections must be fitted only on training
data. They are regularized correlation/ridge-style transforms, not ordered or
leave-one-out target encoders.
