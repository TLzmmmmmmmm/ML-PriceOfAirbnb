# Step 9 text-model comparison

## Controlled setup

All experiments use the Step 3 five folds, raw-dollar targets, Ridge regression with `alpha=20` and `solver=lsqr`, and dollar-scale RMSE. Missing text is replaced with an empty string. The supplied fields are `name`, `summary`, `description`, `space`, `neighborhood_overview`, `house_rules`, and `host_about`.

Each fold fits its TF-IDF vocabulary and inverse-document-frequency weights only on the fold's training rows. Validation text is transformed with that fitted vectorizer. No external corpus, pretrained representation, or test-set statistics are used.

## Results

| Experiment | Representation | Features | Mean RMSE | Std RMSE | Overall OOF RMSE |
|---|---|---:|---:|---:|---:|
| E010 | Word 1–2 grams | 50,000 | 108.100058 | 3.894283 | 108.170004 |
| E011 | Character 3–5 grams | 60,000 | 108.042792 | 3.925244 | 108.113903 |
| E012 | Word + character | 110,000 | **104.492116** | **3.794498** | **104.560826** |

Per-fold RMSE:

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| E010 | 103.390686 | 111.615587 | 103.302373 | 111.497748 | 110.693897 |
| E011 | 103.318825 | 111.868671 | 103.222095 | 111.398737 | 110.405633 |
| E012 | **99.884073** | **108.097488** | **99.934776** | **108.147365** | **106.396879** |

E012 improves on each single TF-IDF representation in all five folds. It also beats the official linear baseline mean RMSE of `111.9119`, confirming that the supplied text has measurable predictive value, although it remains weaker than E009's structured-model mean RMSE of `87.1546`.

## Complementarity with E009

| Text experiment | OOF error correlation with E009 | E009/text 50–50 overall RMSE |
|---|---:|---:|
| E010 | 0.726984 | 90.866556 |
| E011 | 0.715711 | 90.547053 |
| E012 | 0.725292 | 89.121482 |

The correlations are consistently below `0.76` in every fold, so text errors are meaningfully different from E009 errors. Equal weighting is too aggressive and degrades E009, so no ensemble is selected in Step 9. E012 is retained as the strongest text component for a small, controlled weight comparison in Step 11. E011 may be kept as a secondary diversity candidate because its error correlation is the lowest.

E012 was retrained from the original training file under a separate artifact prefix. All five fold scores, feature counts, negative-prediction count, and the OOF SHA-256 hash reproduced exactly.

Raw text-only predictions include some negative values. They are left unchanged for honest OOF evaluation; any final ensemble or submission validity constraint must be evaluated separately rather than retroactively altering these experiment scores.
