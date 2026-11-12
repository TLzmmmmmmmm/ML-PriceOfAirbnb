"""Leakage-safe preprocessing for structured listing attributes."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


NUMERIC_COLUMNS = (
    "host_listings_count",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "guests_included",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "calculated_host_listings_count",
    "reviews_per_month",
)

CURRENCY_COLUMNS = ("extra_people",)
PERCENTAGE_COLUMNS = ("host_response_rate",)
DATE_COLUMNS = ("host_since", "first_review", "last_review")

LOW_CARDINALITY_CATEGORICAL_COLUMNS = (
    "host_response_time",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "neighbourhood_group_cleansed",
    "state",
    "market",
    "country_code",
    "country",
    "property_type",
    "room_type",
    "bed_type",
    "instant_bookable",
    "cancellation_policy",
    "require_guest_profile_picture",
    "require_guest_phone_verification",
)

HIGH_CARDINALITY_CATEGORICAL_COLUMNS = (
    "host_id",
    "host_name",
    "host_location",
    "host_neighbourhood",
    "neighbourhood_cleansed",
    "city",
    "zipcode",
)

TEXT_COLUMNS = (
    "name",
    "summary",
    "space",
    "description",
    "neighborhood_overview",
    "notes",
    "transit",
    "access",
    "interaction",
    "house_rules",
    "host_about",
)

STRUCTURED_TEXT_COLUMNS = ("amenities", "host_verifications")

EXCLUDED_COLUMNS = (
    "id",
    "price",
    "experiences_offered",
    "host_acceptance_rate",
    "is_business_travel_ready",
    "square_feet",
)

DATE_ORIGIN = pd.Timestamp("2000-01-01")
MISSING_CATEGORY = "__MISSING__"

ENGINEERED_FEATURE_GROUPS = {
    "date_relations": (
        "review_history_days",
        "host_to_first_review_days",
    ),
    "capacity_ratios": (
        "accommodates_per_bedroom",
        "accommodates_per_bed",
        "beds_per_bedroom",
        "bathrooms_per_accommodate",
    ),
    "host_scale": (
        "host_listing_count_difference",
        "calculated_to_host_listings_ratio",
    ),
    "text_structure": (
        "name_length",
        "summary_length",
        "description_length",
        "name_missing",
        "summary_missing",
        "description_missing",
        "row_missing_count",
    ),
}

TEXT_STRUCTURE_COLUMNS = ("name", "summary", "description")


def parse_currency(values: pd.Series) -> pd.Series:
    """Convert currency strings such as '$1,234.50' to floating-point values."""
    cleaned = values.astype("string").str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_percentage(values: pd.Series) -> pd.Series:
    """Convert percentage strings such as '95%' to the value 95.0."""
    cleaned = values.astype("string").str.replace(r"[%,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def parse_date_days(values: pd.Series) -> pd.Series:
    """Convert dates to days since the fixed origin 2000-01-01."""
    parsed = pd.to_datetime(values, errors="coerce")
    return (parsed - DATE_ORIGIN).dt.days.astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (numerator / denominator).astype(float)


def add_engineered_features(
    frame: pd.DataFrame,
    feature_groups: tuple[str, ...],
) -> pd.DataFrame:
    """Return a copy with the requested stateless feature groups added."""
    unknown_groups = set(feature_groups) - ENGINEERED_FEATURE_GROUPS.keys()
    if unknown_groups:
        raise ValueError(f"Unknown feature groups: {sorted(unknown_groups)}")

    result = frame.copy()
    if "date_relations" in feature_groups:
        host_since = pd.to_datetime(frame["host_since"], errors="coerce")
        first_review = pd.to_datetime(frame["first_review"], errors="coerce")
        last_review = pd.to_datetime(frame["last_review"], errors="coerce")
        result["review_history_days"] = (last_review - first_review).dt.days.astype(float)
        result["host_to_first_review_days"] = (
            first_review - host_since
        ).dt.days.astype(float)

    if "capacity_ratios" in feature_groups:
        result["accommodates_per_bedroom"] = _safe_ratio(
            frame["accommodates"], frame["bedrooms"]
        )
        result["accommodates_per_bed"] = _safe_ratio(
            frame["accommodates"], frame["beds"]
        )
        result["beds_per_bedroom"] = _safe_ratio(
            frame["beds"], frame["bedrooms"]
        )
        result["bathrooms_per_accommodate"] = _safe_ratio(
            frame["bathrooms"], frame["accommodates"]
        )

    if "host_scale" in feature_groups:
        host_count = pd.to_numeric(frame["host_listings_count"], errors="coerce")
        calculated_count = pd.to_numeric(
            frame["calculated_host_listings_count"], errors="coerce"
        )
        result["host_listing_count_difference"] = host_count - calculated_count
        result["calculated_to_host_listings_ratio"] = _safe_ratio(
            calculated_count, host_count
        )

    if "text_structure" in feature_groups:
        missing_columns = []
        for column in TEXT_STRUCTURE_COLUMNS:
            text = frame[column].astype("string")
            missing = text.isna() | text.str.strip().eq("")
            result[f"{column}_length"] = text.fillna("").str.len().astype(float)
            result[f"{column}_missing"] = missing.astype(float)
            missing_columns.append(missing)
        result["row_missing_count"] = pd.concat(missing_columns, axis=1).sum(
            axis=1
        ).astype(float)

    return result


def prepare_structured_frame(
    frame: pd.DataFrame,
    feature_groups: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Apply stateless parsing and select the structured feature groups."""
    source = add_engineered_features(frame, feature_groups)
    prepared = pd.DataFrame(index=frame.index)

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in CURRENCY_COLUMNS:
        prepared[column] = parse_currency(frame[column])
    for column in PERCENTAGE_COLUMNS:
        prepared[column] = parse_percentage(frame[column])
    for column in DATE_COLUMNS:
        prepared[f"{column}_days"] = parse_date_days(frame[column])
    for column in LOW_CARDINALITY_CATEGORICAL_COLUMNS:
        prepared[column] = (
            frame[column].astype("string").fillna(MISSING_CATEGORY).astype(object)
        )
    for group in feature_groups:
        for column in ENGINEERED_FEATURE_GROUPS[group]:
            prepared[column] = source[column]

    return prepared


class StructuredPreprocessor(BaseEstimator, TransformerMixin):
    """Fit train-only imputers and encoders for structured listing features."""

    def __init__(self, feature_groups: tuple[str, ...] = ()):
        self.feature_groups = feature_groups

    def fit(self, frame: pd.DataFrame, y: object = None) -> "StructuredPreprocessor":
        prepared = prepare_structured_frame(frame, self.feature_groups)
        numeric_features = (
            *NUMERIC_COLUMNS,
            *CURRENCY_COLUMNS,
            *PERCENTAGE_COLUMNS,
            *(f"{column}_days" for column in DATE_COLUMNS),
            *(
                column
                for group in self.feature_groups
                for column in ENGINEERED_FEATURE_GROUPS[group]
            ),
        )
        self.transformer_ = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="median",
                                    keep_empty_features=True,
                                ),
                            )
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        dtype=np.float64,
                    ),
                    LOW_CARDINALITY_CATEGORICAL_COLUMNS,
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )
        self.transformer_.fit(prepared)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        return self.transformer_.transform(
            prepare_structured_frame(frame, self.feature_groups)
        )

    def get_feature_names_out(self) -> np.ndarray:
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()
