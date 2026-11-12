# E018 CatBoost fair comparison

## Objective

Compare CatBoost with E016 using exactly the same source fields. CatBoost receives the 18 categorical fields natively rather than expanding them into one-hot columns. No host identity or other new predictive fields are introduced.

## Leakage boundary

The fixed five outer folds remain unchanged. For each fold, the amenity vocabulary is learned without outer-validation rows. A deterministic 10% split of the outer training rows is used for CatBoost early stopping; outer-validation targets are used only for the reported RMSE.

## Fixed configuration

- CatBoost `1.2.10`, CPU, four threads, `4gb` CTR RAM limit.
- 2,000 maximum iterations, learning rate `0.03`, depth `8`, L2 leaf regularization `5.0`.
- 100-round early stopping.
- 142 input features per fold: 48 structured/native categorical features plus 94 amenity features.

## Results

| Fold | E016 RMSE | E018 RMSE | E018 improvement |
|---:|---:|---:|---:|
| 0 | 78.886962 | **78.085870** | 0.801092 |
| 1 | **86.390026** | 87.277748 | -0.887722 |
| 2 | 81.645469 | **80.430416** | 1.215053 |
| 3 | 87.338662 | **86.926530** | 0.412132 |
| 4 | **79.336089** | 79.358735 | -0.022646 |
| **Mean** | 82.719442 | **82.415860** | 0.303582 |

E018 improves three of five folds. Its RMSE standard deviation is `3.899254`, compared with E016's `3.524167`. Best iteration counts are `1132`, `1260`, `1977`, `1992`, and `1817`; the two high counts approach but do not hit the 2,000-tree limit. Sequential five-fold runtime is `924.9` seconds.

Prediction correlation with E016 is `0.97146`, and error correlation is `0.95540`. CatBoost therefore adds some useful signal but remains highly similar to E016.

## Decision

Reject E018 as a direct E016 replacement. Although its mean RMSE improves by `0.30358`, it misses the predefined `0.5` threshold, improves only `3/5` folds instead of at least four, and has higher fold dispersion. No tuning or submission is performed in E018.

The OOF predictions remain available locally as a possible controlled ensemble component, but any such blend must be a separate experiment with its own predefined candidates and gate.
