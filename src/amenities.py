"""Parsing and fold-trainable features for supplied amenity strings."""

import csv
from collections import Counter
from math import ceil

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def parse_amenities(value: object) -> set[str]:
    """Parse one brace-wrapped CSV-style amenity collection."""
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("Amenities must be wrapped in braces")
    inner = text[1:-1]
    if not inner:
        return set()
    tokens = next(csv.reader([inner], skipinitialspace=True))
    return {token.strip().casefold() for token in tokens if token.strip()}


class AmenitiesTransformer(BaseEstimator, TransformerMixin):
    """Create amenity counts and fold-training frequency indicators."""

    def __init__(
        self,
        *,
        include_indicators: bool = True,
        min_frequency: float = 0.01,
        max_features: int = 100,
    ):
        self.include_indicators = include_indicators
        self.min_frequency = min_frequency
        self.max_features = max_features

    def fit(self, values: pd.Series, y: object = None) -> "AmenitiesTransformer":
        parsed_rows = [parse_amenities(value) for value in values]
        if not self.include_indicators:
            self.selected_amenities_ = ()
            return self

        minimum_count = ceil(self.min_frequency * len(parsed_rows))
        counts = Counter(token for row in parsed_rows for token in row)
        eligible = (
            (token, count)
            for token, count in counts.items()
            if count >= minimum_count
        )
        ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
        self.selected_amenities_ = tuple(
            token for token, _ in ranked[: self.max_features]
        )
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        check_is_fitted(self, "selected_amenities_")
        parsed_rows = [parse_amenities(value) for value in values]
        matrix = np.zeros(
            (len(parsed_rows), 1 + len(self.selected_amenities_)),
            dtype=float,
        )
        for row_index, amenities in enumerate(parsed_rows):
            matrix[row_index, 0] = len(amenities)
            for feature_index, token in enumerate(self.selected_amenities_, start=1):
                matrix[row_index, feature_index] = token in amenities
        return matrix

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self, "selected_amenities_")
        return np.array(
            [
                "amenity_count",
                *(f"amenity__{token}" for token in self.selected_amenities_),
            ],
            dtype=object,
        )
