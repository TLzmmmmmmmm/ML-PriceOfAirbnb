# CSE 151A Airbnb price challenge — final solution

## Result

The selected primary model is E015, with local five-fold RMSE `84.5293` and public RMSE `83.97699`. The selected secondary model is E014, with local RMSE `85.3451` and public RMSE `84.49803`. The public leaderboard snapshot placed the team at rank 34. Kaggle shows both final slots selected (`2/2`).

The course's two fixed performance thresholds are met on the public score, but the official grade and ranking points depend on the private leaderboard after the deadline.

## Method

E014 is a histogram gradient boosting regressor using leakage-safe structured preprocessing, text-length/missingness features, amenity count, and 93 frequent amenity indicators learned independently inside each training fold.

E015 combines three independently validated components:

- 75% E014;
- 10% E009, the structured model without amenity indicators;
- 15% E012, a Ridge model over 50,000 word TF-IDF and 60,000 character TF-IDF features.

The ensemble weights were selected from a small predeclared set using local OOF predictions only. E015 improved all five folds over E014. Public leaderboard results were recorded only after the model and weights were fixed.

## Validation and reproducibility

- Five deterministic folds assigned by ID with random seed 151.
- Every learned preprocessor, amenity vocabulary, and TF-IDF vocabulary fitted only on training-fold rows.
- No external data or pretrained model.
- Final E014 and E015 CSVs independently regenerated with exact prediction equality and matching SHA-256 hashes.
- Official data, models, OOF arrays, credentials, and submissions remain outside Git.

See `reports/final_reproducibility.md` for input hashes, package versions, final file hashes, and the complete experiment mapping.

Do not publish this repository or summary as a public competition solution until the competition has ended and course-policy review permits publication.
