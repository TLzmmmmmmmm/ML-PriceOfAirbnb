# Step 11 ensemble comparison

## Method

All combinations use aligned Step 3 out-of-fold predictions. Component file lengths, finite values, and recorded SHA-256 hashes were verified before evaluation. Weights were selected only from the small predeclared set below; no public-leaderboard result was used.

E014 is the control. E009 tests variance reduction between related tabular models. E012 is the strongest text model. E011 is a secondary text candidate with slightly lower error correlation but weaker standalone RMSE.

## Correlation

OOF prediction correlation:

| | E009 | E011 | E012 | E014 |
|---|---:|---:|---:|---:|
| E009 | 1.0000 | 0.6526 | 0.6873 | 0.9735 |
| E011 | 0.6526 | 1.0000 | 0.9876 | 0.6663 |
| E012 | 0.6873 | 0.9876 | 1.0000 | 0.7005 |
| E014 | 0.9735 | 0.6663 | 0.7005 | 1.0000 |

OOF error correlation:

| | E009 | E011 | E012 | E014 |
|---|---:|---:|---:|---:|
| E009 | 1.0000 | 0.7157 | 0.7253 | 0.9637 |
| E011 | 0.7157 | 1.0000 | 0.9941 | 0.7138 |
| E012 | 0.7253 | 0.9941 | 1.0000 | 0.7246 |
| E014 | 0.9637 | 0.7138 | 0.7246 | 1.0000 |

E011 and E012 are effectively redundant with each other, so they are never included together.

## Pairwise candidates

| E014 weight | Other component | Other weight | Mean RMSE | Change vs E014 | Folds improved |
|---:|---|---:|---:|---:|---:|
| 1.00 | None | 0.00 | 85.345138 | 0.000000 | — |
| 0.90 | E009 | 0.10 | 85.240840 | -0.104298 | 4/5 |
| 0.75 | E009 | 0.25 | 85.204033 | -0.141104 | 4/5 |
| 0.50 | E009 | 0.50 | 85.461809 | +0.116671 | 2/5 |
| 0.95 | E012 | 0.05 | 84.938083 | -0.407055 | 5/5 |
| 0.90 | E012 | 0.10 | 84.685432 | -0.659706 | 5/5 |
| 0.85 | E012 | 0.15 | **84.588556** | **-0.756581** | **5/5** |
| 0.80 | E012 | 0.20 | 84.647968 | -0.697169 | 5/5 |
| 0.95 | E011 | 0.05 | 85.015956 | -0.329182 | 5/5 |
| 0.90 | E011 | 0.10 | 84.856796 | -0.488342 | 5/5 |
| 0.85 | E011 | 0.15 | 84.868597 | -0.476541 | 5/5 |

## Selected ensemble

Because both E014/E009 and E014/E012 pairs improved E014 across multiple folds, one three-model candidate was evaluated:

- E014: 75%
- E009: 10%
- E012: 15%

| Model | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean RMSE |
|---|---:|---:|---:|---:|---:|---:|
| E014 | 81.900978 | 89.657663 | 83.410757 | 89.465037 | 82.291253 | 85.345138 |
| E015 | **81.058086** | **88.867823** | **82.252063** | **88.762449** | **81.706102** | **84.529305** |

E015 improves all five folds and beats every included component. It has no negative OOF predictions. The mean improvement over E014 is `0.815833`; the improvement is therefore not driven by one anomalous fold.

E015 becomes the primary Step 12 candidate. E014 remains the strongest single-model candidate and a useful lower-complexity alternative.

Recomputing E015 from the three recorded component OOF files reproduced the fold metrics and OOF SHA-256 hash exactly.
