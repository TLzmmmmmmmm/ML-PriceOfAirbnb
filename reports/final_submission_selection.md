# Step 14 final submission selection

## Status

Complete. The user supplied a Kaggle Submissions screenshot showing `2/2`, with both E015 and E014 checked, captured at `2026-11-11 18:03:44 -08:00`.

## Recommended final candidates

### Primary: E015

- File: `E015_weighted_20261111.csv`
- Local mean CV RMSE: `84.52930`
- Public RMSE: `83.97699`
- Public snapshot rank: 34
- SHA-256: `966aee32247f879a214e9fa0f3416b03af0431964070505cc9581070fe042053`
- Rationale: best local and public score; improves E014 on all five local folds.

### Secondary: E014

- File: `E014_full_20261111.csv`
- Local mean CV RMSE: `85.34514`
- Public RMSE: `84.49803`
- SHA-256: `b11143d93bcc6e0113fda0106d7385091752944b0f5d64e583313a69acafea18`
- Rationale: strongest individual model and does not depend on the TF-IDF/Ridge component used by E015.

## Final file validation

Both files were reloaded immediately before selection and passed:

- exactly 17,337 rows;
- columns exactly `Id` and `Predicted`;
- IDs exactly match `test.csv` in original order;
- zero duplicate IDs;
- zero non-finite predictions;
- zero negative predictions;
- SHA-256 matches the Step 12 receipt.

## Diversification

| Measure | E014 versus E015 |
|---|---:|
| Prediction correlation | 0.996409 |
| Mean absolute difference | 7.104549 |
| Median absolute difference | 5.058615 |
| Maximum absolute difference | 159.426209 |
| Exactly identical rows | 0 / 17,337 |
| Rows where E015 is higher | 9,909 |
| Rows where E015 is lower | 7,428 |

The files are highly correlated because E015 assigns 75% weight to E014, but they are not duplicate predictions. E014 provides a lower-complexity fallback against private-set instability in the text component and ensemble weighting.

## Kaggle final-selection state

Both `E015_weighted_20261111.csv` and `E014_full_20261111.csv` are selected. Kaggle shows `2/2`; no additional final-selection slot remains.
