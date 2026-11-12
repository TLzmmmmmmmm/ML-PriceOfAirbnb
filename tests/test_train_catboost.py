import numpy as np
import pandas as pd
import pytest


pytest.importorskip("catboost")


def _catboost_training_frame(rows=60):
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
    data["name"] = [f"listing {value}" for value in row_number]
    data["summary"] = "summary"
    data["description"] = "description"
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
    data["price"] = (
        50.0
        + 20.0 * data["accommodates"]
        + 15.0 * (data["neighbourhood_cleansed"] == "Chelsea")
    )
    return pd.DataFrame(data)


def test_prepare_e018_features_preserves_categories_and_fold_amenity_vocabulary():
    from src.amenities import AmenitiesTransformer
    from src.preprocessing import MISSING_CATEGORY
    from src.train_catboost import (
        CATBOOST_CATEGORICAL_COLUMNS,
        prepare_e018_features,
    )

    training = _catboost_training_frame(rows=10)
    training.loc[0, "zipcode"] = None
    validation = _catboost_training_frame(rows=2)
    validation["amenities"] = "{Validation Only Pool}"
    amenities = AmenitiesTransformer(
        include_indicators=True,
        min_frequency=0.1,
        max_features=100,
    ).fit(training["amenities"])

    training_features = prepare_e018_features(training, amenities)
    validation_features = prepare_e018_features(validation, amenities)

    assert training_features.columns.tolist() == validation_features.columns.tolist()
    assert set(CATBOOST_CATEGORICAL_COLUMNS).issubset(training_features.columns)
    assert training_features.loc[0, "zipcode"] == MISSING_CATEGORY
    assert all(
        training_features[column].map(type).eq(str).all()
        for column in CATBOOST_CATEGORICAL_COLUMNS
    )
    assert "amenity__validation only pool" not in validation_features.columns
    assert "amenity__wifi" in validation_features.columns


def test_evaluate_e018_cv_produces_finite_oof_and_fold_artifacts(tmp_path):
    import joblib

    from src.train_catboost import evaluate_e018_cv

    frame = _catboost_training_frame()
    folds = pd.Series(np.arange(len(frame)) % 3, index=frame.index, name="fold")
    frame.loc[folds.eq(0), "amenities"] = "{Validation Exclusive Amenity}"

    result = evaluate_e018_cv(
        frame,
        folds,
        artifact_dir=tmp_path,
        experiment_id="TEST_E018",
    )

    assert result["oof_predictions"].shape == (len(frame),)
    assert np.isfinite(result["oof_predictions"]).all()
    assert len(result["fold_rmse"]) == 3
    assert len(result["best_iterations"]) == 3
    assert min(result["best_iterations"]) > 0
    assert len(result["feature_counts"]) == 3
    assert sorted(path.name for path in tmp_path.glob("TEST_E018_fold_*.joblib")) == [
        "TEST_E018_fold_0.joblib",
        "TEST_E018_fold_1.joblib",
        "TEST_E018_fold_2.joblib",
    ]
    fold_zero = joblib.load(tmp_path / "TEST_E018_fold_0.joblib")
    assert (
        "validation exclusive amenity"
        not in fold_zero["amenities_transformer"].selected_amenities_
    )
    np.testing.assert_allclose(
        np.load(tmp_path / "TEST_E018_oof_predictions.npy"),
        result["oof_predictions"],
    )
