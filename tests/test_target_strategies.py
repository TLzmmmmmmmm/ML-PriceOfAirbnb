import numpy as np
import pandas as pd


def _training_frame(rows=60):
    from src.preprocessing import (
        DATE_COLUMNS,
        LOW_CARDINALITY_CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )

    row_number = np.arange(rows)
    data = {column: np.ones(rows) for column in NUMERIC_COLUMNS}
    data["accommodates"] = row_number % 8 + 1
    data["bedrooms"] = row_number % 4 + 1
    data["number_of_reviews"] = row_number
    data["extra_people"] = [f"${value}.00" for value in row_number % 30]
    data["host_response_rate"] = [f"{value}%" for value in 80 + row_number % 21]
    data.update({column: ["2018-01-01"] * rows for column in DATE_COLUMNS})
    data.update(
        {
            column: [f"{column}_{value % 3}" for value in row_number]
            for column in LOW_CARDINALITY_CATEGORICAL_COLUMNS
        }
    )
    data["room_type"] = np.where(
        row_number % 2 == 0,
        "Entire home/apt",
        "Private room",
    )
    data["id"] = row_number + 1000
    data["price"] = (
        40.0
        + 22.0 * data["accommodates"]
        + 35.0 * (data["room_type"] == "Entire home/apt")
    )
    return pd.DataFrame(data)


def test_log_target_round_trip_including_zero():
    from src.evaluate_target_strategies import inverse_log_target, transform_log_target

    prices = np.array([0.0, 1.0, 100.0, 1000.0])

    transformed = transform_log_target(prices)
    restored = inverse_log_target(transformed)

    assert transformed[0] == 0.0
    np.testing.assert_allclose(restored, prices, atol=1e-12, rtol=0)


def test_strategy_factory_changes_only_the_loss_function():
    from src.evaluate_target_strategies import build_strategy_model

    log_model = build_strategy_model("log1p_squared_error")
    absolute_model = build_strategy_model("raw_absolute_error")
    log_parameters = log_model.get_params()
    absolute_parameters = absolute_model.get_params()

    assert log_parameters["loss"] == "squared_error"
    assert absolute_parameters["loss"] == "absolute_error"
    absolute_parameters["loss"] = "squared_error"
    assert absolute_parameters == log_parameters


def test_price_band_errors_use_original_dollar_predictions():
    from src.evaluate_target_strategies import price_band_metrics

    actual = np.array([50.0, 200.0, 400.0, 600.0])
    predicted = np.array([60.0, 180.0, 450.0, 500.0])

    metrics = price_band_metrics(actual, predicted)

    assert metrics["price_le_100"] == {
        "count": 1,
        "rmse": 10.0,
        "mean_error": 10.0,
    }
    assert metrics["price_100_to_300"]["rmse"] == 20.0
    assert metrics["price_300_to_500"]["mean_error"] == 50.0
    assert metrics["price_gt_500"]["mean_error"] == -100.0


def test_log_strategy_produces_complete_dollar_scale_oof_predictions(tmp_path):
    from src.evaluate_target_strategies import evaluate_target_strategy_cv

    frame = _training_frame()
    folds = pd.Series(np.arange(len(frame)) % 3, index=frame.index, name="fold")

    result = evaluate_target_strategy_cv(
        frame,
        folds,
        strategy="log1p_squared_error",
        artifact_dir=tmp_path,
        experiment_id="TEST_LOG",
    )

    assert len(result["fold_rmse"]) == 3
    assert result["oof_predictions"].shape == (len(frame),)
    assert np.isfinite(result["oof_predictions"]).all()
    assert sorted(path.name for path in tmp_path.glob("TEST_LOG_fold_*.joblib")) == [
        "TEST_LOG_fold_0.joblib",
        "TEST_LOG_fold_1.joblib",
        "TEST_LOG_fold_2.joblib",
    ]
    saved_oof = np.load(tmp_path / "TEST_LOG_oof_predictions.npy")
    np.testing.assert_allclose(saved_oof, result["oof_predictions"])
