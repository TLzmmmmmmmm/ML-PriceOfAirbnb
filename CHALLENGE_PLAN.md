# CSE 151A Kaggle Challenge Execution Plan

## Objective

Achieve at least 85% on the challenge while keeping the solution reproducible and compliant with the competition rules.

The challenge score is composed of:

- 50% for a final RMSE strictly below the official simple linear regression baseline.
- An additional 30% for a final RMSE below 100.
- The remaining 20% based on final private-leaderboard ranking.

An 85% score therefore requires both performance thresholds plus at least 5 of the 20 ranking points. Because the ranking-to-grade conversion is unknown, the working target is a mean local cross-validation RMSE of at most 92, ideally below 90, rather than merely reaching 99.x.

## Competition constraints

- Deadline: November 26, 2026 at 11:59 PM Pacific time.
- Metric: root mean squared error (RMSE) in nightly-price dollars.
- No external data.
- No teaming; the work must remain individual.
- Do not hand-label or manually predict test records.
- Maximum 20 Kaggle submissions per day.
- Select at most two submissions for final private-leaderboard evaluation.
- Submission columns must be exactly `Id` and `Predicted`.
- The public leaderboard uses 50% of the test set and is not the final result.

## Execution rules

Before executing any numbered step, first tell the user:

1. the step's objective;
2. exactly what will be done;
3. what result or deliverable the user will receive.

Do not begin that step until the user confirms. After execution, verify the completion criteria, report the evidence, and keep generated work traceable through Git and experiment IDs.

Official competition files must remain unchanged and excluded from Git. Generated submissions, trained models, credentials, and raw competition data must not be committed.

## Progress

| Step | Status | Main result |
|---:|---|---|
| 1 | Complete | Six official files inspected; structure, missingness, target distribution, baseline files, and risks documented in chat |
| 2 | Complete | Reproducible project structure created and pushed in commit `47a27ab` |
| 3 | Complete | Fixed five-fold validation protocol created and pushed in commit `4f6318b` |
| 4 | Complete | Mean and official linear baselines measured in five-fold validation; comparison recorded in `reports/baseline_comparison.md` |
| 5 | Complete | Leakage-safe structured preprocessor implemented; all 65 columns assigned to documented groups |
| 6 | Complete | E003 structured histogram gradient boosting reached mean five-fold RMSE `87.2869` |
| 7 | Complete | Raw price with squared-error loss retained after E004/E005 long-tail comparison |
| 8 | Complete | E009 text-structure features improved mean five-fold RMSE to `87.1546`; three other groups rejected |
| 9–15 | Not started | Must follow the confirmation rule before execution |

## Step 1: Audit files, data, and rules

### Objective

Confirm that the supplied competition files are usable and identify data-quality, formatting, leakage, and rule risks before modeling.

### Actions

- Inventory every supplied file and verify that it is readable and non-empty.
- Compare train and test schemas.
- Check target, ID, missingness, duplicates, data types, ranges, and train/test consistency.
- Validate the sample and baseline submission formats.
- Inspect the official baseline notebook.
- Record competition and grading constraints.

### Deliverables

- Dataset and file inventory.
- Data-quality findings and modeling-risk list.
- Confirmed submission schema and competition constraints.

### Completion criteria

- All official files parse successfully.
- Train and test schemas differ only by `price`.
- IDs and submission rows are verified.
- Critical data issues and modeling risks are explicitly recorded.

## Step 2: Establish the project structure

### Objective

Create a reproducible workspace that protects official files and keeps code, experiments, models, reports, and submissions traceable.

### Actions

- Keep official files unchanged in the repository working directory.
- Create `src/`, `tests/`, `experiments/`, `models/`, `reports/`, and `submissions/` as needed.
- Define experiment IDs such as `E001`.
- Add Git ignore rules for data, credentials, models, reports that should remain local, and submissions.
- Document directory responsibilities and naming rules.

### Deliverables

- Repository structure.
- `README.md` and `.gitignore`.
- Experiment and submission naming conventions.

### Completion criteria

- Official files remain unchanged and ignored by Git.
- Generated artifacts have defined destinations.
- The project structure is committed and pushed.

## Step 3: Establish the local validation protocol

### Objective

Create one deterministic, leakage-resistant local scoring method for every later experiment.

### Actions

- Sort rows by unique `id` to establish a canonical order.
- Assign five shuffled K-fold splits with random seed `151`.
- Map fold labels back to the original DataFrame index.
- Calculate RMSE in the original dollar scale.
- Reject invalid IDs and invalid prediction arrays.
- Test determinism, row-order stability, fold coverage, and RMSE behavior.
- Record aggregate fold statistics without committing row-level competition data.

### Deliverables

- `src/validation.py`.
- `tests/test_validation.py`.
- `reports/validation_protocol.md`.
- README validation documentation.

### Completion criteria

- Every training row belongs to exactly one validation fold.
- Fold sizes differ by at most one row.
- Reordering input rows does not change an ID's fold.
- All validation tests pass.
- The protocol is committed and pushed.

## Step 4: Reproduce and measure the official baselines

### Objective

Measure the mean-value and official simple-linear-regression approaches with the fixed local validation protocol, producing the reference scores every later experiment must beat.

### Actions

- Reproduce the mean-value baseline inside each fold using only the fold's training rows.
- Reproduce the official linear feature construction, including its current parsing behavior.
- Evaluate both methods on all five validation folds.
- Report per-fold RMSE, mean RMSE, and RMSE standard deviation.
- Record runtime, feature list, assumptions, and known baseline defects.

### Deliverables

- Baseline evaluation code and tests.
- `E001` mean-baseline experiment record.
- `E002` official-linear-baseline experiment record.
- Baseline comparison report.

### Completion criteria

- Both baselines use the Step 3 folds.
- No validation target information enters training.
- Results can be reproduced from the official files.
- The official linear baseline becomes the explicit threshold for later experiments.

## Step 5: Build the preprocessing pipeline

### Objective

Create one train-fold-only transformation pipeline that treats train, validation, and test data consistently without leakage.

### Actions

- Remove all-null and constant fields.
- Parse currency and percentage strings.
- parse dates into useful numeric representations without using future information.
- Define missing-value handling for numeric and categorical features.
- Separate numeric, low-cardinality categorical, high-cardinality categorical, and free-text fields.
- Ensure every learned transformation is fitted only on the training portion of each fold.

### Deliverables

- Reusable preprocessing code.
- Unit tests for parsing, missing values, unseen categories, and train/validation isolation.
- Documented feature groups and excluded columns.

### Completion criteria

- The same pipeline transforms train, validation, and test records.
- Unseen test or validation categories do not cause failure.
- No transformer is fitted on validation or test data.
- Tests pass on representative edge cases.

## Step 6: Train the first strong tabular model

### Objective

Create the first competitive model using only structured numeric and categorical information.

### Actions

- Select one strong model appropriate for mixed tabular data and missing values.
- Train and evaluate it across the fixed five folds.
- Save out-of-fold predictions and fold-level metrics locally.
- Compare directly against both official baselines.
- Record model parameters and runtime under a unique experiment ID.

### Deliverables

- First strong tabular-model experiment.
- Five fold scores, mean RMSE, standard deviation, and baseline improvement.
- Locally stored model artifacts and out-of-fold predictions.

### Completion criteria

- The complete five-fold run succeeds.
- The model beats the official linear baseline locally.
- All results are reproducible from the recorded configuration.

## Step 7: Evaluate long-tail target strategies

### Objective

Reduce RMSE sensitivity to rare expensive listings without sacrificing accuracy on ordinary listings.

### Actions

- Compare modeling raw `price` with modeling `log1p(price)` and converting predictions back to dollars.
- Evaluate only with dollar-scale RMSE.
- Test reasonable prediction constraints or robust objectives only as separate experiments.
- Compare fold-level performance, especially folds containing more high-price listings.
- Retain only changes that improve mean RMSE without unacceptable fold instability.

### Deliverables

- Controlled target-strategy comparison.
- Error analysis by price band.
- Selected target treatment for subsequent models.

### Completion criteria

- Every strategy is evaluated on identical folds.
- Dollar-scale RMSE determines the decision.
- The selected strategy improves or stabilizes performance with evidence.

## Step 8: Perform controlled feature engineering

### Objective

Add compact, interpretable features that provide stable cross-validation gains.

### Actions

- Evaluate date-derived ages and recency measures.
- Evaluate listing capacity ratios and host-scale features.
- Evaluate text-length and missingness indicators.
- Add one coherent feature group per experiment.
- Run ablation comparisons against the current best model.
- Remove feature groups that do not produce stable gains.

### Deliverables

- Feature definitions and tests.
- Ablation table showing the effect of each feature group.
- Updated best tabular experiment.

### Completion criteria

- Every retained feature group has repeatable evidence of improvement.
- No feature uses external data or validation-target information.
- The final feature set remains reproducible and documented.

## Step 9: Build the text model

### Objective

Extract predictive information from listing text without external corpora or pretrained external data.

### Actions

- Select the useful supplied text fields.
- Build word- and/or character-level TF-IDF features using only fold-training text.
- Train a regularized regression model on the sparse text representation.
- Evaluate on the same five folds.
- Save out-of-fold text predictions for later ensembling.

### Deliverables

- Reproducible text pipeline.
- Text-only five-fold metrics.
- Out-of-fold text predictions and error comparison with the tabular model.

### Completion criteria

- Vectorizers are fitted only on fold-training records.
- The text model has measurable predictive value.
- Its errors are sufficiently different from the tabular model to justify ensemble testing.

## Step 10: Parse and evaluate amenities

### Objective

Convert the high-cardinality `amenities` string into usable internal features.

### Actions

- Parse amenities into individual supplied tokens.
- Add amenity count and selected frequent amenity indicators.
- Learn any frequency threshold only from fold-training data.
- Evaluate the amenity feature group as a separate ablation experiment.

### Deliverables

- Tested amenities parser.
- Amenity feature specification.
- Cross-validation comparison with and without amenities.

### Completion criteria

- Parsing handles observed formatting consistently.
- No external amenity information is introduced.
- Amenities are retained only if they improve or stabilize validation RMSE.

## Step 11: Build the ensemble

### Objective

Combine complementary models to outperform the strongest individual model without fitting weights to the public leaderboard.

### Actions

- Use out-of-fold predictions from eligible models.
- Compare a small number of transparent weighted averages.
- Select weights using local out-of-fold RMSE only.
- Check per-fold gains and error correlation.
- Avoid large ensemble searches that overfit the local folds.

### Deliverables

- Ensemble-weight comparison.
- Selected ensemble configuration.
- Ensemble out-of-fold predictions and five-fold metrics.

### Completion criteria

- The ensemble improves mean RMSE over every included component.
- Improvement is not caused by one anomalous fold alone.
- Weights and component experiment IDs are fully recorded.

## Step 12: Generate and validate submissions

### Objective

Produce Kaggle-ready prediction files with no schema, ordering, or numerical defects.

### Actions

- Retrain the selected configuration on all training data.
- Generate test predictions.
- Create exactly the `Id` and `Predicted` columns.
- Preserve the original `test.csv` ID order.
- Check row count, duplicate IDs, missing predictions, non-finite values, and invalid negative predictions.
- Link each submission filename to its experiment record.

### Deliverables

- One or more validated CSV files under `submissions/`.
- Submission validation receipt for each file.

### Completion criteria

- The file has exactly 17,337 rows and two required columns.
- IDs exactly match `test.csv` in order.
- All predictions are finite and meet the chosen validity constraints.
- The file can be traced to a complete experiment configuration.

## Step 13: Use the public leaderboard as controlled evidence

### Objective

Use public scores to detect major generalization problems without overfitting to the public half of the test set.

### Actions

- Submit only experiments with a documented local hypothesis and meaningful local evidence.
- Record local CV, public score, submission time, and model ID.
- Classify outcomes based on whether local and public results agree.
- Do not retain a locally worse model solely because of one favorable public score.
- Limit submission frequency well below the daily maximum unless diagnosing a specific issue.

### Deliverables

- Submission and leaderboard experiment log.
- Documented decisions to retain, reject, or investigate each submission.

### Completion criteria

- Every public submission has a corresponding local experiment.
- Public scores are used as secondary evidence rather than the optimization target.
- No undocumented leaderboard-driven changes are introduced.

## Step 14: Select two final submissions

### Objective

Choose two complementary, reproducible submissions that maximize the probability of reaching the 85% course target on the private leaderboard.

### Actions

- Rank candidates primarily by stable local cross-validation evidence.
- Select one candidate with the best validated mean RMSE.
- Select a second candidate with different modeling assumptions or error behavior.
- Consider public performance only as supporting evidence.
- Verify both files again immediately before final selection.

### Deliverables

- Two named final candidates.
- Written selection rationale and risk comparison.
- Final validation receipts.

### Completion criteria

- Both candidates are reproducible and valid.
- The two candidates are not merely duplicate predictions.
- The primary candidate targets best expected RMSE; the secondary candidate provides meaningful diversification.

## Step 15: Reproduce and archive the final solution

### Objective

Prove that the selected result can be recreated from the untouched official files and preserve the evidence needed for grading and later presentation.

### Actions

- Run the final pipeline from the original files in a clean execution path.
- Confirm that it reproduces the selected predictions within the defined numerical tolerance.
- Run all tests and submission checks.
- Record package versions, experiment IDs, final scores when available, and final-submission choices.
- Keep private competition data, models, submissions, and credentials out of Git.
- Prepare a sanitized project summary suitable for later publication only after competition and course-policy review.

### Deliverables

- Reproducibility record.
- Final method summary.
- Final experiment and submission mapping.
- Clean Git repository containing code, tests, documentation, and safe aggregate results.

### Completion criteria

- The final pipeline reruns successfully from the official inputs.
- Tests and submission validators pass.
- Final artifacts match the selected experiments.
- The repository contains no prohibited data or credentials.
