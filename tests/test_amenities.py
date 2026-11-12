import numpy as np
import pandas as pd
import pytest


def _amenities_training_frame(rows=60):
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
    data["id"] = row_number + 1000
    data["price"] = 50.0 + 20.0 * data["accommodates"]
    return pd.DataFrame(data)


def test_parse_amenities_handles_quoted_commas_normalization_and_duplicates():
    from src.amenities import parse_amenities

    parsed = parse_amenities(
        '{Wifi,"Wide clearance to shower, toilet",Wifi," Air conditioning "}'
    )

    assert parsed == {
        "wifi",
        "wide clearance to shower, toilet",
        "air conditioning",
    }


def test_parse_amenities_handles_empty_values_and_rejects_invalid_wrapper():
    from src.amenities import parse_amenities

    assert parse_amenities("{}") == set()
    assert parse_amenities(None) == set()
    assert parse_amenities(np.nan) == set()
    with pytest.raises(ValueError, match="braces"):
        parse_amenities("Wifi,Kitchen")


def test_transformer_learns_frequent_tokens_only_from_training_rows():
    from src.amenities import AmenitiesTransformer

    training = pd.Series(
        ["{Wifi,Kitchen}", "{Wifi}", "{Kitchen}", "{TV}"],
        index=[10, 11, 12, 13],
    )
    validation = pd.Series(["{Wifi,Pool,Wifi}", "{Pool}"], index=[20, 21])

    transformer = AmenitiesTransformer(
        include_indicators=True,
        min_frequency=0.5,
        max_features=10,
    ).fit(training)
    transformed = transformer.transform(validation)

    assert transformer.get_feature_names_out().tolist() == [
        "amenity_count",
        "amenity__kitchen",
        "amenity__wifi",
    ]
    np.testing.assert_array_equal(
        transformed,
        np.array([[2.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    )


def test_count_only_transformer_does_not_create_indicator_columns():
    from src.amenities import AmenitiesTransformer

    values = pd.Series(["{Wifi,Wifi,Kitchen}", "{}"])
    transformer = AmenitiesTransformer(include_indicators=False).fit(values)

    assert transformer.get_feature_names_out().tolist() == ["amenity_count"]
    np.testing.assert_array_equal(
        transformer.transform(values),
        np.array([[2.0], [0.0]]),
    )


def test_amenities_cv_keeps_validation_only_tokens_out_of_fold_vocabulary(
    tmp_path,
):
    import joblib

    from src.train_amenities import evaluate_amenities_cv

    frame = _amenities_training_frame()
    folds = pd.Series(np.arange(len(frame)) % 3, index=frame.index, name="fold")
    frame.loc[folds.eq(0), "amenities"] = "{Validation exclusive amenity}"

    result = evaluate_amenities_cv(
        frame,
        folds,
        include_indicators=True,
        min_frequency=0.1,
        max_features=10,
        artifact_dir=tmp_path,
        experiment_id="TEST_AMENITIES",
    )

    fold_zero = joblib.load(tmp_path / "TEST_AMENITIES_fold_0.joblib")
    assert (
        "validation exclusive amenity"
        not in fold_zero["amenities_transformer"].selected_amenities_
    )
    assert result["oof_predictions"].shape == (len(frame),)
    assert np.isfinite(result["oof_predictions"]).all()
    assert result["include_indicators"] is True
    assert sorted(path.name for path in tmp_path.glob("TEST_AMENITIES_fold_*.joblib")) == [
        "TEST_AMENITIES_fold_0.joblib",
        "TEST_AMENITIES_fold_1.joblib",
        "TEST_AMENITIES_fold_2.joblib",
    ]
    np.testing.assert_allclose(
        np.load(tmp_path / "TEST_AMENITIES_oof_predictions.npy"),
        result["oof_predictions"],
    )
