# Step 13 public leaderboard log

## Evidence scope

The scores below were captured from the user's Kaggle Submissions screenshot on 2026-11-11. Both rows showed a green completion indicator and status `Complete`. Follow-up leaderboard screenshots showed team `Terry Li` at public rank 34 with best score `83.97699` and two entries. Browser read-only automation failed to initialize locally, and no additional Kaggle action was taken.

At capture time Kaggle showed `0/2` final submissions selected. Final selection remains deferred to Step 14.

## Submission log

| Experiment | File | Local mean CV RMSE | Public RMSE | Public minus local | Kaggle status | Final selected | Rank captured |
|---|---|---:|---:|---:|---|---|---|
| E014 | `E014_full_20261111.csv` | 85.34514 | 84.49803 | -0.84711 | Complete | No | Not applicable individually |
| E015 | `E015_weighted_20261111.csv` | 84.52930 | **83.97699** | -0.55231 | Complete | No | **34** |

E015 improves over E014 by `0.81583` locally and `0.52104` on the public half of the test set. The direction agrees even though the public improvement is smaller. Both public scores are also below 100 and below the locally measured official linear baseline.

## Decision

- Retain E015 as the primary candidate because it is best both locally and publicly.
- Retain E014 as the strongest single-model alternative; it provides a different complexity and failure profile.
- Do not alter weights or create leaderboard-driven variants from these two public observations.
- Use no further submission quota in Step 13.

The public leaderboard covers only 50% of the test set, so these results are supporting evidence rather than the final private-leaderboard outcome.

## Leaderboard snapshot

- Public team rank: 34
- Team: `Terry Li`
- Best public score: `83.97699`
- Entries: 2
- Captured at: `2026-11-11 17:54:31 -08:00`
- Rank 1 public score in the supplied snapshot: `75.08503`

This is a point-in-time public rank. It can change as competitors submit, and the final course ranking uses the private half after the deadline.

## E016 optimization update

The user supplied a later Kaggle Submissions screenshot showing `E016_location_20261111.csv` complete with public RMSE `82.72167` and its selection checkbox enabled. Its local mean CV RMSE was `82.71944`, so public minus local is only `0.00223`. This close agreement strengthens confidence in the fixed local validation protocol. No rank was visible in the supplied E016 screenshot.
