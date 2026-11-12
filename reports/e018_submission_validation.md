# E018 CatBoost submission validation

## Outcome

`submissions/E018_catboost_20261112.csv` was generated as a user-requested diagnostic submission. It has not been uploaded to Kaggle and does not change the E018 model-selection decision: E018 remains rejected as a direct replacement for E017/E016 because it missed the predefined CV gate.

## Full-training policy

- All 33,538 training rows participate in fitting.
- The tree count is fixed at `1817`, the median of E018's five recorded best iteration counts: `1132`, `1260`, `1977`, `1992`, and `1817`.
- No early-stopping holdout is created for full training.
- The model uses E018's unchanged 142-feature CatBoost frame, including 94 amenity features.

## CSV validation receipt

| Check | Result |
|---|---:|
| Rows | 17,337 |
| Columns | `Id`, `Predicted` |
| Duplicate IDs | 0 |
| Non-finite predictions | 0 |
| Negative predictions | 0 |
| Predictions clipped to zero | 0 |
| Minimum prediction | 10.049735 |
| Maximum prediction | 1121.423205 |
| Mean prediction | 144.019734 |

The CSV was written, reloaded, and validated against the original test IDs in their original order.

## Reproducibility and repository safety

- Submission SHA-256: `ae950935daf988fa94297b4b0878f27a34f0447cc6a011fd64fd90f565167b79`.
- Model SHA-256: `78a234230ca29ba6398403ed453934f46cdfdff507c8e5b4e9f7d78459ba0508`.
- Both artifacts are excluded from Git by existing ignore rules.
- The submission has not been uploaded.
