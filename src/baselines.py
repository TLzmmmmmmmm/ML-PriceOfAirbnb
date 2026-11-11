"""Faithful implementations of the competition's official baselines."""

from collections.abc import Sequence
from math import isnan

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


OFFICIAL_BASE_FEATURES = (
    "guests_included",
    "extra_people",
    "bedrooms",
    "bathrooms",
    "review_scores_rating",
)


def official_to_float(value: object) -> float:
    """Match the official notebook's float conversion and missing sentinel."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if isnan(converted) else converted


def official_to_int(value: object) -> int:
    """Match the official notebook's integer conversion and missing sentinel."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return -1


def build_official_features(
    frame: pd.DataFrame,
    *,
    categories: Sequence[str],
) -> np.ndarray:
    """Build the five notebook features followed by neighborhood one-hot columns."""
    numeric = np.column_stack(
        [
            frame["guests_included"].map(official_to_int),
            frame["extra_people"].map(official_to_int),
            frame["bedrooms"].map(official_to_int),
            frame["bathrooms"].map(official_to_int),
            frame["review_scores_rating"].map(official_to_float),
        ]
    ).astype(float)
    one_hot = np.column_stack(
        [
            frame["neighbourhood_group_cleansed"].eq(category).to_numpy(dtype=float)
            for category in categories
        ]
    )
    return np.column_stack([numeric, one_hot])


def predict_training_mean(
    training_targets: Sequence[float],
    *,
    prediction_count: int,
) -> np.ndarray:
    """Predict the supplied training targets' mean for every requested row."""
    return np.full(prediction_count, np.mean(training_targets), dtype=float)


def predict_official_linear(
    training_features: pd.DataFrame,
    training_targets: Sequence[float],
    prediction_features: pd.DataFrame,
) -> np.ndarray:
    """Fit and apply the official simple linear regression without category leakage."""
    categories = tuple(
        sorted(
            value
            for value in training_features["neighbourhood_group_cleansed"].unique()
            if isinstance(value, str)
        )
    )
    train_matrix = build_official_features(
        training_features,
        categories=categories,
    )
    prediction_matrix = build_official_features(
        prediction_features,
        categories=categories,
    )
    model = LinearRegression()
    model.fit(train_matrix, np.asarray(training_targets, dtype=float))
    return model.predict(prediction_matrix)
