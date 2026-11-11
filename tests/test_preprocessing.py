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
