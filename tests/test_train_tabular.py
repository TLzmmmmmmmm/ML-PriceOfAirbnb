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


def test_model_factory_returns_the_fixed_e003_configuration():
    from src.train_tabular import build_e003_model

    model = build_e003_model()
    parameters = model.get_params()

    assert parameters["loss"] == "squared_error"
    assert parameters["learning_rate"] == 0.05
    assert parameters["max_iter"] == 500
    assert parameters["max_leaf_nodes"] == 31
    assert parameters["min_samples_leaf"] == 20
    assert parameters["l2_regularization"] == 1.0
    assert parameters["early_stopping"] is True
    assert parameters["validation_fraction"] == 0.1
    assert parameters["n_iter_no_change"] == 30
    assert parameters["random_state"] == 151


def test_cross_validation_produces_complete_oof_predictions_and_artifacts(tmp_path):
    from src.train_tabular import evaluate_e003_cv

    frame = _training_frame()
    folds = pd.Series(np.arange(len(frame)) % 3, index=frame.index, name="fold")

    result = evaluate_e003_cv(
        frame,
        folds,
        artifact_dir=tmp_path,
        experiment_id="TEST",
    )

    assert len(result["fold_rmse"]) == 3
    assert np.isfinite(result["fold_rmse"]).all()
    assert result["oof_predictions"].shape == (len(frame),)
    assert np.isfinite(result["oof_predictions"]).all()
    assert len(result["iterations"]) == 3
    assert sorted(path.name for path in tmp_path.glob("TEST_fold_*.joblib")) == [
        "TEST_fold_0.joblib",
        "TEST_fold_1.joblib",
        "TEST_fold_2.joblib",
    ]
    saved_oof = np.load(tmp_path / "TEST_oof_predictions.npy")
    np.testing.assert_allclose(saved_oof, result["oof_predictions"])
