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
