# Local validation protocol

## Purpose

Use one deterministic local score for every experiment. The protocol approximates the competition's random test split while keeping all validation rows unseen during fitting.

## Rules

- Sort rows by unique, non-null `id` before assigning folds.
- Use five-fold shuffled K-fold validation with random seed `151`.
- Each listing appears in validation exactly once.
- Fit all preprocessing and model state only on the training portion of each fold.
- Calculate RMSE after converting predictions back to the original dollar scale.
- Reject missing or non-finite predictions before scoring.

## Fold checks

| Fold | Rows | Mean price | Median price | Price standard deviation | Minimum | Maximum | Price > 500 | Price > 1000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6,708 | 142.489 | 105 | 127.875 | 0 | 1,860 | 126 (1.878%) | 15 (0.224%) |
| 1 | 6,708 | 147.410 | 110 | 136.855 | 0 | 1,820 | 137 (2.042%) | 17 (0.253%) |
| 2 | 6,708 | 142.480 | 105 | 127.963 | 0 | 1,999 | 131 (1.953%) | 14 (0.209%) |
| 3 | 6,707 | 146.378 | 110 | 135.318 | 0 | 1,700 | 147 (2.192%) | 19 (0.283%) |
| 4 | 6,707 | 147.130 | 110 | 137.034 | 0 | 1,750 | 142 (2.117%) | 21 (0.313%) |

All 33,538 training rows receive one fold label. Fold sizes differ by at most one row. The high-price tail varies across folds, so every experiment must report both mean RMSE and the five individual fold scores.

The SHA-256 fingerprint of the sorted `id,fold` assignment is `7232ed5ce9b5b5fe165f2b452c143bfab8271c17e038a4eade537a645edc504b`.
