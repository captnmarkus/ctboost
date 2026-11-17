# CTBoost 0.1.63

## Rare numeric values

With automatic quantile borders, a numeric column could receive a single cut
equal to its minimum. Values equal to that cut and values above it then landed
in the same bin. A rare indicator could disappear from training even when it
perfectly predicted the target.

For this case CTBoost now separates the minimum from the next distinct training
value. This applies to dense and sparse inputs. Previously saved ensembles use
their stored borders. Retrain a model to take advantage of the correction.

## Pandas categorical inputs

Scikit-learn estimators trained on a pandas frame can infer categorical columns.
Previously, predicting on a smaller batch could assign a different integer to
the same category and change its prediction. Newly fitted estimators reuse the
training vocabulary. The [categorical guide](https://captnmarkus.github.io/ctboost/guides/categorical-text/)
describes prediction, persistence and compatibility with earlier fits.

The TabArena adapter now recognizes pandas 3's inferred string dtype. Previously
a mixed frame could send these strings into numeric conversion and fail to fit.

## Benchmark evidence

Against 0.1.62 on held-out development splits from all 51 cached TabArena
tasks, 0.1.63 improved three tasks, worsened two and returned byte-identical
predictions on 46. Median relative error reduction was 0.00% and the mean was
+0.17%. The [benchmark guide](https://captnmarkus.github.io/ctboost/benchmarks/)
records the per-task results, protocol and official Space ratings. These
single-model comparisons do not measure release-specific Elo.
