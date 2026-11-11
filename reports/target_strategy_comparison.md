# Target strategy comparison

## Decision

Retain E003: raw dollar price with squared-error training loss. Both alternative strategies improve ordinary-price listings but increase total RMSE by underpredicting the expensive tail.

| Experiment | Target and loss | Mean fold RMSE | Fold RMSE standard deviation | Overall OOF RMSE | Decision |
|---|---|---:|---:|---:|---|
| E003 | Raw price, squared error | **87.2869** | **3.4073** | **87.3534** | Retain |
| E004 | `log1p(price)`, squared error | 88.8462 | 3.5461 | 88.9169 | Reject |
| E005 | Raw price, absolute error | 92.3593 | 3.8365 | 92.4388 | Reject |

E004 is worse than E003 by `1.5592` mean RMSE. E005 is worse by `5.0723`. Both alternatives are worse on all five folds, so their result is not caused by one anomalous split.

## Error by true-price band

Mean error is prediction minus actual price. A negative value means systematic underprediction.

| Price band | Rows | E003 RMSE / mean error | E004 RMSE / mean error | E005 RMSE / mean error |
|---|---:|---:|---:|---:|
| `price ≤ 100` | 16,274 | 38.09 / +20.09 | **29.63 / +11.73** | **29.28 / +11.03** |
| `100 < price ≤ 300` | 14,986 | 61.35 / +0.79 | **56.54 / -11.32** | **55.73 / -12.18** |
| `300 < price ≤ 500` | 1,595 | **148.65 / -74.96** | 154.55 / -100.48 | 156.13 / -113.59 |
| `price > 500` | 683 | **453.80 / -315.47** | 491.31 / -370.12 | 523.53 / -415.59 |

E004 and E005 improve the two lower-price bands, which contain 31,260 of 33,538 records. However, RMSE squares dollar errors. Their additional underprediction on 2,278 records above 300 outweighs the improvement on the majority of listings.

## Interpretation

The log target changes the training objective toward relative error. E004 therefore fits ordinary-price listings more closely but gives less weight to large dollar errors. Its mean prediction error for listings above 500 is `-370.12`, compared with `-315.47` for E003.

Absolute-error loss reduces the influence of large residuals even more. E005 has the best RMSE below 300 but the worst RMSE and underprediction above 300. This behavior is inconsistent with the competition's dollar-scale squared-error metric.

No nonnegative clipping experiment was run because E003, E004, and E005 each produced zero negative OOF predictions. Applying a floor of zero would leave all recorded predictions unchanged.

## Selected strategy

All subsequent structured-model experiments will continue using raw `price` and squared-error loss unless a later experiment provides new evidence. E003 remains the benchmark at mean five-fold RMSE `87.2869`.
