# E016 submission validation

## Outcome

`submissions/E016_location_20261111.csv` was generated from an E016 model fitted on all 33,538 training rows. The file has not been uploaded to Kaggle and the existing final selections have not been changed.

## Full-training configuration

- Model: the fixed histogram gradient boosting configuration used by E014/E016.
- Features: E014 structured text and amenity features plus `neighbourhood_cleansed` and `zipcode`.
- HGB iterations selected by the existing internal early-stopping procedure: `471`.
- Total full-training features: `807`.
- Amenity features: `94` (count plus 93 indicators).

## CSV validation receipt

| Check | Result |
|---|---:|
| Rows | 17,337 |
| Columns | `Id`, `Predicted` |
| Duplicate IDs | 0 |
| Non-finite predictions | 0 |
| Negative predictions | 0 |
| Predictions clipped to zero | 0 |
| Minimum prediction | 8.424646 |
| Maximum prediction | 1145.262310 |
| Mean prediction | 144.171759 |

The CSV was written, reloaded, and validated against the original `test.csv` IDs in their original order.

## Reproducibility and repository safety

- Submission SHA-256: `1e60540c0fa533960ddf5511020ab1cfe5936db3007905ba3303870649686c1b`.
- Model SHA-256: `333242e57b97c026b2e8788175e05e35c50c736fe9b4f06aef2c87d4e1f4c049`.
- The model and submission are excluded from Git by the existing ignore rules.
- Only code, tests, the aggregate experiment record, and this validation receipt are committed.
