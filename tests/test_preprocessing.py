import numpy as np
import pandas as pd


def _structured_frame(rows=3):
    from src.preprocessing import (
        DATE_COLUMNS,
        LOW_CARDINALITY_CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
    )

    data = {column: [1.0] * rows for column in NUMERIC_COLUMNS}
    data.update(
        {
            "extra_people": ["$25.00"] * rows,
            "host_response_rate": ["90%"] * rows,
        }
    )
    data.update({column: ["2018-01-01"] * rows for column in DATE_COLUMNS})
    data.update(
        {
            column: [f"{column}_value"] * rows
            for column in LOW_CARDINALITY_CATEGORICAL_COLUMNS
        }
    )
    data["id"] = list(range(rows))
    data["price"] = [100.0] * rows
    data["description"] = ["text that must remain excluded"] * rows
    data["host_acceptance_rate"] = [np.nan] * rows
    data["experiences_offered"] = ["none"] * rows
    data["is_business_travel_ready"] = ["f"] * rows
    data["square_feet"] = [np.nan] * rows
    return pd.DataFrame(data)


def test_specialized_parsers_return_numeric_values_and_missing_values():
    from src.preprocessing import parse_currency, parse_date_days, parse_percentage

    currency = parse_currency(pd.Series(["$1,234.50", "", None, "bad"]))
    percentage = parse_percentage(pd.Series(["95%", "100%", None, "bad"]))
    dates = parse_date_days(
        pd.Series(["2000-01-01", "2000-01-03", None, "bad"])
    )

    np.testing.assert_allclose(currency.iloc[:1], [1234.5])
    assert currency.iloc[1:].isna().all()
    np.testing.assert_allclose(percentage.iloc[:2], [95.0, 100.0])
    assert percentage.iloc[2:].isna().all()
    np.testing.assert_allclose(dates.iloc[:2], [0.0, 2.0])
    assert dates.iloc[2:].isna().all()


def test_numeric_imputation_is_fitted_only_on_training_rows():
    from src.preprocessing import StructuredPreprocessor

    training = _structured_frame()
    training["accommodates"] = [1.0, 3.0, np.nan]
    validation = _structured_frame(rows=2)
    validation["accommodates"] = [1000.0, np.nan]

    preprocessor = StructuredPreprocessor().fit(training)
    transformed = preprocessor.transform(validation)
    names = preprocessor.get_feature_names_out().tolist()
    accommodates_position = names.index("accommodates")

    assert transformed[0, accommodates_position] == 1000.0
    assert transformed[1, accommodates_position] == 2.0


def test_unseen_categories_keep_the_same_output_schema():
    from src.preprocessing import StructuredPreprocessor

    training = _structured_frame(rows=2)
    training["room_type"] = ["Entire home/apt", "Private room"]
    validation = _structured_frame(rows=1)
    validation["room_type"] = ["Unseen room type"]

    preprocessor = StructuredPreprocessor().fit(training)
    training_matrix = preprocessor.transform(training)
    validation_matrix = preprocessor.transform(validation)

    assert training_matrix.shape[1] == validation_matrix.shape[1]
    room_columns = [
        index
        for index, name in enumerate(preprocessor.get_feature_names_out())
        if name.startswith("room_type_")
    ]
    assert validation_matrix[0, room_columns].sum() == 0.0


def test_preprocessing_excludes_deferred_fields_and_does_not_mutate_input():
    from src.preprocessing import StructuredPreprocessor

    training = _structured_frame()
    original = training.copy(deep=True)

    preprocessor = StructuredPreprocessor().fit(training)
    preprocessor.transform(training)
    output_names = preprocessor.get_feature_names_out().tolist()

    pd.testing.assert_frame_equal(training, original)
    assert "id" not in output_names
    assert "price" not in output_names
    assert not any("description" in name for name in output_names)
    assert not any("host_acceptance_rate" in name for name in output_names)
    assert not any("square_feet" in name for name in output_names)


def test_date_relation_features_use_only_dates_from_the_same_row():
    from src.preprocessing import add_engineered_features

    frame = _structured_frame(rows=2)
    frame["host_since"] = ["2018-01-01", None]
    frame["first_review"] = ["2018-01-11", "2018-02-01"]
    frame["last_review"] = ["2018-01-31", "2018-03-01"]

    result = add_engineered_features(frame, ("date_relations",))

    assert result["review_history_days"].tolist() == [20.0, 28.0]
    assert result.loc[0, "host_to_first_review_days"] == 10.0
    assert np.isnan(result.loc[1, "host_to_first_review_days"])


def test_capacity_ratio_features_replace_zero_denominators_with_missing():
    from src.preprocessing import add_engineered_features

    frame = _structured_frame(rows=2)
    frame["accommodates"] = [4.0, 2.0]
    frame["bedrooms"] = [2.0, 0.0]
    frame["beds"] = [1.0, 0.0]
    frame["bathrooms"] = [1.0, 1.0]

    result = add_engineered_features(frame, ("capacity_ratios",))

    assert result.loc[0, "accommodates_per_bedroom"] == 2.0
    assert result.loc[0, "accommodates_per_bed"] == 4.0
    assert result.loc[0, "beds_per_bedroom"] == 0.5
    assert result.loc[0, "bathrooms_per_accommodate"] == 0.25
    assert result.loc[1, [
        "accommodates_per_bedroom",
        "accommodates_per_bed",
        "beds_per_bedroom",
    ]].isna().all()


def test_host_scale_features_capture_count_difference_and_ratio():
    from src.preprocessing import add_engineered_features

    frame = _structured_frame(rows=2)
    frame["host_listings_count"] = [10.0, 0.0]
    frame["calculated_host_listings_count"] = [4.0, 2.0]

    result = add_engineered_features(frame, ("host_scale",))

    assert result["host_listing_count_difference"].tolist() == [6.0, -2.0]
    assert result.loc[0, "calculated_to_host_listings_ratio"] == 0.4
    assert np.isnan(result.loc[1, "calculated_to_host_listings_ratio"])


def test_text_structure_features_measure_length_presence_and_row_missingness():
    from src.preprocessing import add_engineered_features

    frame = _structured_frame(rows=2)
    frame["name"] = ["Loft", None]
    frame["description"] = ["abc def", ""]
    frame["summary"] = [None, "Room"]

    result = add_engineered_features(frame, ("text_structure",))

    assert result["name_length"].tolist() == [4.0, 0.0]
    assert result["description_length"].tolist() == [7.0, 0.0]
    assert result["summary_length"].tolist() == [0.0, 4.0]
    assert result["name_missing"].tolist() == [0.0, 1.0]
    assert result["description_missing"].tolist() == [0.0, 1.0]
    assert result["summary_missing"].tolist() == [1.0, 0.0]
    assert result["row_missing_count"].tolist() == [1.0, 2.0]


def test_structured_preprocessor_includes_only_requested_engineered_group():
    from src.preprocessing import StructuredPreprocessor

    training = _structured_frame()
    training["name"] = ["a", "bb", "ccc"]
    training["summary"] = ["x", None, "z"]

    preprocessor = StructuredPreprocessor(
        feature_groups=("text_structure",)
    ).fit(training)
    names = preprocessor.get_feature_names_out().tolist()

    assert "name_length" in names
    assert "summary_missing" in names
    assert "row_missing_count" in names
    assert "review_history_days" not in names
