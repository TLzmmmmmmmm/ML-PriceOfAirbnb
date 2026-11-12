# E017 E016/E012 blend comparison

## Objective

Test whether the complementary text errors from E012 improve the new E016 location model without training another model or fitting weights to the public leaderboard.

E016's public RMSE is `82.72167`, only `0.00223` above its local mean CV RMSE of `82.71944`. This public result is recorded as validation evidence but was not used to choose E017 weights.

## Complementarity

- Prediction correlation: `0.72305`.
- Error correlation: `0.72578`.

The correlation is low enough to justify the predeclared small text weights, even though E012 is much weaker alone.

## Fixed-weight results

| E016 | E012 | Mean RMSE | Improvement vs E016 | Folds improved | Gate |
|---:|---:|---:|---:|---:|---|
| 95% | 5% | 82.452366 | 0.267075 | 5/5 | Reject: gain below 0.3 |
| **90%** | **10%** | **82.343309** | **0.376133** | **5/5** | **Retain** |
| 85% | 15% | 82.392871 | 0.326570 | 5/5 | Pass, but not best |
| 80% | 20% | 82.600731 | 0.118710 | 3/5 | Reject |

Selected fold RMSE values are `78.565806`, `86.155529`, `80.993426`, `86.891423`, and `79.110360`. The standard deviation is `3.514599`, and overall OOF RMSE is `82.418240`.

## Decision and submission

Retain the 90% E016 / 10% E012 blend as E017. It exceeds the predefined `0.3` improvement gate and improves every fold. The generated `E017_blend_20261111.csv` contains 17,337 aligned rows, no invalid or negative predictions, and required no clipping. Its SHA-256 is `4c9955487531ed5b5020d896c0d40fa298200cc8d6543ad4b51d587f435b921b`.

The E017 CSV remains local and has not been uploaded to Kaggle.
