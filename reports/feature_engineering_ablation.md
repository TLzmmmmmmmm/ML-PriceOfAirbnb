# Step 8 feature-engineering ablation

## Controlled setup

All experiments use the Step 3 five folds, the E003 histogram-gradient-boosting configuration, raw-dollar targets, squared-error training loss, and dollar-scale RMSE. Each experiment adds exactly one feature group to E003. No feature uses external data, validation targets, or statistics learned from validation/test rows.

Ratios with zero denominators become missing values and are handled by the existing fold-trained median imputer. Date features are differences between dates in the same row; no dataset-wide reference date is used. Text-structure features use only `name`, `summary`, and `description`, not their word meaning.

## Ablation results

Lower RMSE is better. `Change vs E003` is experiment mean minus E003 mean.

| Experiment | Added group | Mean RMSE | Std RMSE | Change vs E003 | Folds improved | Decision |
|---|---|---:|---:|---:|---:|---|
| E003 | None | 87.286950 | 3.407295 | 0.000000 | — | Control |
| E006 | Date relations | 87.411793 | 3.548311 | +0.124843 | 1/5 | Reject |
| E007 | Capacity ratios | 87.430745 | 3.313970 | +0.143795 | 0/5 | Reject |
| E008 | Host scale | 87.333976 | 3.200491 | +0.047026 | 1/5 | Reject |
| E009 | Text structure | **87.154584** | 3.478562 | **-0.132366** | 3/5 | **Retain** |

Per-fold RMSE:

| Experiment | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| E003 | **83.802645** | 90.472924 | 85.653367 | **92.204028** | 84.301783 |
| E006 | 83.847125 | 90.628044 | 85.753844 | 92.606626 | 84.223323 |
| E007 | 83.933891 | 90.500833 | 85.957412 | 92.207194 | 84.554395 |
| E008 | 84.072025 | 90.488250 | 85.735408 | 91.835034 | 84.539161 |
| E009 | 83.840567 | **90.057432** | **85.354560** | 92.467790 | **84.052570** |

## Decision

E009 becomes the best structured tabular experiment. It improves mean RMSE by `0.132366` and improves three folds. Its standard deviation increases by only `0.071267`, and an independent rerun reproduced every fold score, iteration count, and OOF SHA-256 hash exactly.

E006–E008 are rejected because each has a worse mean RMSE and improves at most one fold. They are not combined with E009: searching combinations of individually rejected groups would weaken the controlled-ablation rule and add local-validation selection risk.

No E010 run is created because only one group survived. A “combined retained-groups” experiment would therefore be identical to E009 and would create a duplicate experiment rather than new evidence.
