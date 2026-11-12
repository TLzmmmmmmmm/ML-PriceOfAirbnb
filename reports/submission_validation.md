# Step 12 submission validation

## Full-data training

All 33,538 training rows were used to fit the three recorded E015 components. Test predictions preserve the original 17,337-row `test.csv` order.

| Component | Full-training metadata | Raw test prediction range | Raw negatives |
|---|---|---:|---:|
| E009 | 124 features, 382 HGB iterations | 11.5503–1107.6240 | 0 |
| E014 | 218 features, 93 amenity indicators, 274 HGB iterations | 0.5467–1142.4152 | 0 |
| E012 | 110,000 TF-IDF features | -46.5201–547.5876 | 43 |

The E015 weighted predictions use E014 at 75%, E009 at 10%, and E012 at 15%. Although E012 has negative raw predictions, the weighted E015 predictions are all positive. The nonnegative clipping policy therefore changed zero E014 or E015 predictions.

Full fitted models and component test-prediction arrays are stored under `models/` and excluded from Git.

## Validation receipts

| File | Experiment | Rows | Prediction min | Prediction max | Prediction mean | Clipped | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| `E014_full_20261111.csv` | E014 | 17,337 | 0.546746 | 1142.415222 | 143.899739 | 0 | `b11143d93bcc6e0113fda0106d7385091752944b0f5d64e583313a69acafea18` |
| `E015_weighted_20261111.csv` | E015 | 17,337 | 18.921845 | 1006.433707 | 143.942325 | 0 | `966aee32247f879a214e9fa0f3416b03af0431964070505cc9581070fe042053` |

Both files passed all checks:

- columns are exactly `Id` and `Predicted`;
- IDs exactly match `test.csv` and `sample_submission.csv` in order;
- there are 17,337 data rows and 17,338 physical lines including the header;
- there are no duplicate or missing IDs;
- every prediction is finite and nonnegative;
- reloading each CSV preserves a valid submission.

The submission CSV files are stored under `submissions/` and excluded from Git. E015 is the primary leaderboard candidate; E014 is the strongest single-model alternative.
