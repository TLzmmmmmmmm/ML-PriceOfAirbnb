import numpy as np
import pandas as pd
import pytest


def _full_training_frame(rows=60):
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
    text = np.where(row_number % 2 == 0, "sunny apartment", "quiet room")
    for column in (
        "name",
        "summary",
        "description",
        "space",
        "neighborhood_overview",
        "house_rules",
        "host_about",
    ):
        data[column] = text
    data["amenities"] = np.where(
        row_number % 2 == 0,
        "{Wifi,Kitchen}",
        "{Heating}",
    )
    data["neighbourhood_cleansed"] = np.where(
        row_number % 2 == 0,
        "Chelsea",
        "SoHo",
    )
    data["zipcode"] = np.where(
        row_number % 2 == 0,
        "10001",
        "10002",
    )
    data["id"] = row_number + 1000
    data["price"] = 50.0 + 20.0 * data["accommodates"]
    return pd.DataFrame(data)


def test_fit_full_components_returns_aligned_predictions_and_artifacts(tmp_path):
    from src.submission import fit_full_components

    training = _full_training_frame()
    test = _full_training_frame(rows=12).drop(columns="price")

    result = fit_full_components(training, test, artifact_dir=tmp_path)

    assert result.keys() == {"E009", "E012", "E014"}
    for predictions in result.values():
        assert predictions.shape == (len(test),)
        assert np.isfinite(predictions).all()
    assert sorted(path.name for path in tmp_path.glob("*_full.joblib")) == [
        "E009_full.joblib",
        "E012_full.joblib",
        "E014_full.joblib",
    ]


def test_fit_full_e016_returns_location_model_predictions_and_metadata(tmp_path):
    import joblib

    from src.submission import fit_full_e016

    training = _full_training_frame()
    test = _full_training_frame(rows=12).drop(columns="price")
    artifact_path = tmp_path / "E016_full.joblib"

    result = fit_full_e016(training, test, artifact_path=artifact_path)
    artifact = joblib.load(artifact_path)
    feature_names = (
        artifact["structured_preprocessor"].get_feature_names_out().tolist()
    )

    assert result["predictions"].shape == (len(test),)
    assert np.isfinite(result["predictions"]).all()
    assert result["extra_categorical_columns"] == [
        "neighbourhood_cleansed",
        "zipcode",
    ]
    assert result["feature_count"] > result["amenity_feature_count"]
    assert result["iterations"] > 0
    assert "neighbourhood_cleansed_Chelsea" in feature_names
    assert "zipcode_10001" in feature_names
    assert artifact_path.is_file()
    assert sorted(path.name for path in tmp_path.glob("*.joblib")) == [
        "E016_full.joblib"
    ]


def test_create_submission_preserves_id_order_and_clips_negative_predictions():
    from src.submission import create_submission

    submission, clipped_count = create_submission(
        pd.Series([50, 10], name="id"),
        np.array([-3.0, 125.5]),
    )

    assert submission.columns.tolist() == ["Id", "Predicted"]
    assert submission["Id"].tolist() == [50, 10]
    assert submission["Predicted"].tolist() == [0.0, 125.5]
    assert clipped_count == 1


def test_create_submission_rejects_invalid_ids_and_predictions():
    from src.submission import create_submission

    with pytest.raises(ValueError, match="unique and non-null"):
        create_submission(pd.Series([1, 1]), np.array([10.0, 20.0]))
    with pytest.raises(ValueError, match="same length"):
        create_submission(pd.Series([1, 2]), np.array([10.0]))
    with pytest.raises(ValueError, match="finite"):
        create_submission(pd.Series([1, 2]), np.array([10.0, np.nan]))


def test_validate_submission_checks_schema_order_and_values(tmp_path):
    from src.submission import create_submission, validate_submission

    ids = pd.Series([20, 10], name="id")
    submission, _ = create_submission(ids, np.array([100.0, 200.0]))
    path = tmp_path / "submission.csv"
    submission.to_csv(path, index=False)
    reloaded = pd.read_csv(path)

    receipt = validate_submission(reloaded, ids)

    assert receipt == {
        "row_count": 2,
        "duplicate_ids": 0,
        "nonfinite_predictions": 0,
        "negative_predictions": 0,
        "prediction_min": 100.0,
        "prediction_max": 200.0,
        "prediction_mean": 150.0,
    }

    with pytest.raises(ValueError, match="test ID order"):
        validate_submission(reloaded.iloc[::-1].reset_index(drop=True), ids)
