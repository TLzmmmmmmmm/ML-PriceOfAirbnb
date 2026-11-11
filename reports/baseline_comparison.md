# Official baseline comparison

## Result

The official simple linear regression is the local baseline that later experiments must beat. Its mean five-fold RMSE is `111.9119`. The course target of RMSE below 100 requires an additional local improvement of at least `11.9119`, and the working target of RMSE at most 92 requires an improvement of `19.9119`.

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean RMSE | RMSE standard deviation |
|---|---:|---:|---:|---:|---:|---:|---:|
| E001 mean baseline | 127.9094 | 136.8735 | 127.9980 | 135.3162 | 137.0454 | 133.0285 | 4.1872 |
| E002 official linear baseline | 107.7324 | 115.1298 | 108.8584 | 115.1248 | 112.7143 | 111.9119 | 3.1020 |

The linear baseline improves mean RMSE by `21.1165`, or `15.87%`, relative to the mean baseline. It performs better on all five folds.

## Method

Both experiments use the fixed Step 3 fold assignments. For each fold, the model is fitted only on the other four folds and scored on the held-out fold using RMSE in original nightly-price dollars.

E001 predicts the current fold-training mean for every validation row. E002 reproduces the official notebook's five parsed numeric fields plus one-hot neighborhood group features. Neighborhood categories are derived only from each fold's training rows.

## Compatibility with supplied baseline files

- The full-training mean prediction exactly matches `mean_value_baseline.csv`.
- The full-training linear prediction matches `simple_linear_regression_baseline.csv` with maximum absolute difference `3.73e-11`, below the `1e-8` verification tolerance.

This confirms that the deterministic category ordering used here is prediction-equivalent to the notebook's unordered category list.

## Known defects in the official linear baseline

- All 33,538 `extra_people` values are currency strings such as `$25.00`. The notebook calls `int()` directly, so every row becomes the `-1` missing sentinel and the feature carries no information.
- The notebook applies `int()` to `bathrooms`, truncating 1,876 fractional bathroom values; for example, `1.5` becomes `1`.
- The supplied linear submission contains 17 negative price predictions.

These defects are deliberately preserved in E002 because Step 4 measures the official baseline rather than improving it. They can be corrected only in later experiments, where their effect must be measured with the same folds.

## Decision threshold

A later experiment is considered better than the official local baseline only when its mean five-fold RMSE is below `111.9119`. The primary course-performance checkpoint is below `100`, while the working safety target remains at most `92`. Individual fold scores and their standard deviation must also be reported because the high-price tail differs across folds.
