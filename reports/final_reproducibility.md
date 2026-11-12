# Final reproducibility record

## Reproduction scope

The final E014 and E015 submissions were independently rebuilt from the official CSV inputs using code at commit `48ad81f`. The reproduction run wrote new models and predictions under Git-ignored `reproduction_step15` directories and did not read or overwrite the Step 12 full-model artifacts.

Reproduction runtime: `152.41` seconds.

## Input fingerprints

| File | Bytes | SHA-256 |
|---|---:|---|
| `train.csv` | 103,806,310 | `63ec83622640063732271a77c9f1d421cc21e198a981cf2b5f2325299d3784b8` |
| `test.csv` | 53,316,267 | `e529f5655071390ee70be7c2a47e3eedb5bab66cf7c2042e30769a004b0a1599` |
| `sample_submission.csv` | 444,353 | `436e6d36452a07bb26428a07b10cf37332380b31f5ba2af0761fe50a9bd7977c` |

## Environment

| Package | Version |
|---|---|
| Python | 3.12.5 |
| NumPy | 2.4.1 |
| pandas | 3.0.0 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |

## Submission reproduction

| Experiment | Rows | Max prediction difference | Prediction arrays exact | Original SHA-256 | Reproduced SHA-256 | Hash match |
|---|---:|---:|---|---|---|---|
| E014 | 17,337 | 0.0 | Yes | `b11143d93bcc6e0113fda0106d7385091752944b0f5d64e583313a69acafea18` | `b11143d93bcc6e0113fda0106d7385091752944b0f5d64e583313a69acafea18` | Yes |
| E015 | 17,337 | 0.0 | Yes | `966aee32247f879a214e9fa0f3416b03af0431964070505cc9581070fe042053` | `966aee32247f879a214e9fa0f3416b03af0431964070505cc9581070fe042053` | Yes |

Both reproduced files have the exact required columns and ID order, with zero duplicate IDs, non-finite predictions, negative predictions, or clipped predictions.

## Final experiment mapping

| Role | Experiment | Method | Local mean CV RMSE | Public RMSE | Final selected |
|---|---|---|---:|---:|---|
| Primary | E015 | 75% E014 + 10% E009 + 15% E012 | 84.52930 | 83.97699 | Yes |
| Secondary | E014 | HGB structured + text-structure + frequent amenities | 85.34514 | 84.49803 | Yes |

The supplied Kaggle confirmation shows `2/2`. The recorded public rank snapshot is 34; the final private rank remains pending until the competition closes.

## Repository safety audit

- No tracked `.csv`, `.joblib`, `.npy`, or `kaggle.json` files were found.
- `train.csv`, `test.csv`, all model artifacts, OOF arrays, reproduction artifacts, and submission files are ignored by Git.
- A tracked-content scan for API-key, password, client-secret, and Kaggle-credential patterns returned only `.gitignore`, where credential filenames are intentionally excluded.
- The final test command is `python -m pytest -q`.

This record contains only aggregate metrics, hashes, package versions, and method descriptions; it contains no row-level competition data or predictions.
