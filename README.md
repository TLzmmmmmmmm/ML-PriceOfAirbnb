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
