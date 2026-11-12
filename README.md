# CSE 151A Kaggle Challenge

Predict nightly prices for New York City Airbnb listings. The competition metric is RMSE.

## Official files

The six files in the project root are the original competition inputs and baselines. Keep them unchanged so `simple_baseline.ipynb` continues to run with its original relative paths.

- `train.csv`: training features and `price`
- `test.csv`: test features
- `sample_submission.csv`: required submission format
- `simple_baseline.ipynb`: official baseline notebook
- `mean_value_baseline.csv`: official mean baseline submission
- `simple_linear_regression_baseline.csv`: official linear baseline submission

## Project directories

- `src/`: reusable preprocessing, validation, training, and prediction code
- `experiments/`: one record per experiment, including configuration and cross-validation results
- `submissions/`: generated Kaggle submission CSV files
- `reports/`: data-quality and exploratory-analysis outputs
- `models/`: generated model artifacts

## Naming rules

Use a zero-padded experiment ID everywhere, starting with `E001`.

- Experiment record: `experiments/E001_<short_name>.json`
- Submission: `submissions/E001_<short_name>.csv`
- Model artifact: `models/E001_<short_name>.<extension>`

Every submission must be traceable to one experiment record. Do not overwrite an existing experiment or submission; create the next experiment ID instead.

## Competition constraints

- Do not use external data.
- Do not modify the official input files.
- Evaluate experiments with RMSE using the fixed local validation protocol.
- A submission must contain exactly `Id` and `Predicted`, with IDs in `test.csv` order.

## Local validation protocol

All experiments use the shared functions in `src/validation.py`:

- Sort listings by `id`, then assign five shuffled K-fold splits.
- Use `random_state=151` for every experiment.
- Score predictions with RMSE in the original dollar units.
- Fit preprocessing and models only on the training portion of each fold.

The fold assignment is generated in memory and is not committed. Aggregate fold checks are recorded in `reports/validation_protocol.md`.

## Structured preprocessing

`src/preprocessing.py` parses the structured numeric, currency, percentage, date, and low-cardinality categorical fields. Numeric medians and categorical vocabularies are fitted only on the current fold's training rows. High-cardinality categories, free text, and collection-like text fields remain deferred to their dedicated experiments.

The complete field grouping and exclusion rationale are recorded in `reports/feature_groups.md`.

## Current model benchmark

E003 trains histogram gradient boosting on the structured feature matrix. Its mean five-fold RMSE is `87.2869`, compared with `111.9119` for the official linear baseline. The configuration and fold results are recorded in `experiments/E003_hist_gradient_boosting.json` and `reports/first_tabular_model.md`.

Step 8 evaluates four compact feature groups as controlled ablations. Only text length and missingness indicators are retained: E009 reaches mean five-fold RMSE `87.1546`, while date relations, capacity ratios, and host-scale features are rejected. Definitions and per-fold results are recorded in `reports/feature_engineering_ablation.md`.

Step 9 fits fold-local word and character TF-IDF models. E012 combines both sparse representations and reaches mean five-fold RMSE `104.4921`. It is weaker than E009 alone but has OOF error correlation `0.7253` with E009, so it is retained only as a candidate for controlled Step 11 ensemble testing. See `reports/text_model_comparison.md`.

Step 10 parses the supplied amenities and learns frequent indicators inside each training fold. E014 adds amenity count plus 93 stable frequent indicators to E009 and improves mean five-fold RMSE from `87.1546` to `85.3451`, with gains in every fold. See `reports/amenities_ablation.md`.

Step 11 evaluates a small predeclared set of OOF weighted averages. E015 combines E014 (75%), E009 (10%), and E012 (15%), improving all five folds and lowering mean RMSE to `84.5293`. E014 remains the best single-model alternative. See `reports/ensemble_comparison.md`.

Step 12 retrains E014 and every E015 component on all 33,538 training rows. Validated 17,337-row E014 and E015 submission files are stored locally under `submissions/` and remain excluded from Git. Aggregate validation receipts and SHA-256 hashes are recorded in `reports/submission_validation.md`.

Step 13 records public leaderboard scores of `84.49803` for E014 and `83.97699` for E015. E015 placed team `Terry Li` at public rank 34 in the supplied snapshot. The public ordering agrees with local validation, so both candidates are retained without leaderboard-driven model changes. See `reports/leaderboard_log.md`.

Step 7 compared raw squared-error training with a log target and absolute-error loss. Raw dollar price with squared-error loss remains selected because both alternatives increased total RMSE by underpredicting expensive listings. See `reports/target_strategy_comparison.md`.
