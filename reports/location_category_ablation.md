# E016 location-category ablation

## Question

Does adding only `neighbourhood_cleansed` and `zipcode` improve E014 under the fixed five-fold validation protocol?

## Controlled change

E016 preserves E014's structured text features, amenity count, 93 frequent amenity indicators, HGB parameters, and fixed folds. The only change is fold-trained one-hot encoding of the two location columns. Missing values receive the existing `__MISSING__` category, and validation-only categories are ignored.

## Results

| Fold | E014 RMSE | E016 RMSE | Improvement |
|---:|---:|---:|---:|
| 0 | 81.900978 | **78.886962** | 3.014016 |
| 1 | 89.657663 | **86.390026** | 3.267637 |
| 2 | 83.410757 | **81.645469** | 1.765288 |
| 3 | 89.465037 | **87.338662** | 2.126375 |
| 4 | 82.291253 | **79.336089** | 2.955164 |
| **Mean** | **85.345138** | **82.719442** | **2.625696** |

E016 improves all five folds. Its RMSE standard deviation is `3.524167`, close to E014's `3.478555`, so the gain does not come with materially greater fold instability. Overall OOF RMSE is `82.794441`.

## Cost and validation

- Runtime increased from approximately `45.1` seconds for E014 to `228.8` seconds for E016.
- Fold feature counts increased from approximately 210–215 to 787–800.
- The OOF array contains 33,538 finite predictions and is stored with the five fold models under Git-ignored `models/E016/`.
- No Kaggle submission was generated or uploaded.

## Decision

Retain E016 as the new best single model. It exceeds the predefined requirement of at least `0.5` mean-RMSE improvement and improvement in at least four folds: the actual gain is `2.625696`, with `5/5` folds improved.

The experiment establishes that omitted location information was a major limitation of E014. It does not yet establish whether CatBoost is better than HGB; that remains a separate controlled comparison.
