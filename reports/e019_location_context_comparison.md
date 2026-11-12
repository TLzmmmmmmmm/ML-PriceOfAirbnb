# E019 additional location context comparison

## Objective

Measure the isolated effect of adding `host_neighbourhood`, `city`, and `host_location` to E018 CatBoost. All other source fields, amenities, model parameters, outer folds, and internal early-stopping rules remain unchanged.

## Results

| Fold | E018 RMSE | E019 RMSE | E019 improvement |
|---:|---:|---:|---:|
| 0 | **78.085870** | 78.179033 | -0.093163 |
| 1 | **87.277748** | 87.581345 | -0.303597 |
| 2 | **80.430416** | 80.645565 | -0.215148 |
| 3 | 86.926530 | **86.892572** | 0.033958 |
| 4 | 79.358735 | **79.170808** | 0.187927 |
| **Mean** | **82.415860** | 82.493865 | -0.078005 |

E019 improves only two folds and worsens mean RMSE by `0.07800`. Its standard deviation rises from E018's `3.899254` to `3.957468`. It is also `0.15056` worse than the current-best E017 mean RMSE of `82.34331`.

Best iteration counts are `1596`, `874`, `1991`, `1992`, and `1997`. Three folds finish very close to the fixed 2,000-tree cap. Sequential runtime is `1181.9` seconds, approximately 19.7 minutes.

## Complementarity

- E019/E018 prediction correlation: `0.99594`.
- E019/E018 error correlation: `0.99381`.
- E019/E017 prediction correlation: `0.97239`.
- E019/E017 error correlation: `0.95839`.

The new fields do not create enough independent error behavior to compensate for the weaker score.

## Decision

Reject the E019 feature group. It fails every retention condition: mean performance is worse rather than at least `0.3` better, only `2/5` folds improve, and dispersion increases.

E020 is skipped under its predeclared conditional rule. E019 is worse than E017 and its error correlation with E017 exceeds `0.95`, so a dedicated E017/E019 blend search is not justified. No E019 submission is generated.
