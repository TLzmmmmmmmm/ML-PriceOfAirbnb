"""Full-data training and Kaggle submission validation."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.amenities import AmenitiesTransformer
from src.preprocessing import StructuredPreprocessor
from src.train_tabular import build_e003_model
from src.train_text import TEXT_FIELDS, build_text_vectorizer, combine_text_fields


def fit_full_components(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    artifact_dir: str | Path,
) -> dict[str, np.ndarray]:
    """Fit the E009, E014, and E012 components on every training row."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = training["price"].to_numpy(dtype=float)

    e009_preprocessor = StructuredPreprocessor(feature_groups=("text_structure",))
    e009_training = e009_preprocessor.fit_transform(training)
    e009_test = e009_preprocessor.transform(test)
    e009_model = build_e003_model()
    e009_model.fit(e009_training, target)
    e009_predictions = e009_model.predict(e009_test)
    joblib.dump(
        {"preprocessor": e009_preprocessor, "model": e009_model},
        artifact_dir / "E009_full.joblib",
    )

    e014_preprocessor = StructuredPreprocessor(feature_groups=("text_structure",))
    e014_amenities = AmenitiesTransformer(
        include_indicators=True,
        min_frequency=0.01,
        max_features=100,
    )
    e014_training = np.hstack(
        [
            e014_preprocessor.fit_transform(training),
            e014_amenities.fit_transform(training["amenities"]),
        ]
    )
    e014_test = np.hstack(
        [
            e014_preprocessor.transform(test),
            e014_amenities.transform(test["amenities"]),
        ]
    )
    e014_model = build_e003_model()
    e014_model.fit(e014_training, target)
    e014_predictions = e014_model.predict(e014_test)
    joblib.dump(
        {
            "structured_preprocessor": e014_preprocessor,
            "amenities_transformer": e014_amenities,
            "model": e014_model,
        },
        artifact_dir / "E014_full.joblib",
    )

    e012_vectorizer = build_text_vectorizer("combined")
    e012_training = e012_vectorizer.fit_transform(combine_text_fields(training))
    e012_test = e012_vectorizer.transform(combine_text_fields(test))
    e012_model = Ridge(alpha=20.0, solver="lsqr")
    e012_model.fit(e012_training, target)
    e012_predictions = e012_model.predict(e012_test)
    joblib.dump(
        {
            "text_fields": TEXT_FIELDS,
            "vectorizer": e012_vectorizer,
            "model": e012_model,
        },
        artifact_dir / "E012_full.joblib",
    )

    return {
        "E009": e009_predictions,
        "E012": e012_predictions,
        "E014": e014_predictions,
    }


def fit_full_e016(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    artifact_path: str | Path,
) -> dict[str, object]:
    """Fit E016 on all training rows and predict the aligned test rows."""
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    extra_categorical_columns = ("neighbourhood_cleansed", "zipcode")
    target = training["price"].to_numpy(dtype=float)

    structured = StructuredPreprocessor(
        feature_groups=("text_structure",),
        extra_categorical_columns=extra_categorical_columns,
    )
    amenities = AmenitiesTransformer(
        include_indicators=True,
        min_frequency=0.01,
        max_features=100,
    )
    training_matrix = np.hstack(
        [
            structured.fit_transform(training),
            amenities.fit_transform(training["amenities"]),
        ]
    )
    test_matrix = np.hstack(
        [
            structured.transform(test),
            amenities.transform(test["amenities"]),
        ]
    )
    model = build_e003_model()
    model.fit(training_matrix, target)
    predictions = model.predict(test_matrix)
    joblib.dump(
        {
            "structured_preprocessor": structured,
            "amenities_transformer": amenities,
            "model": model,
        },
        artifact_path,
    )

    return {
        "predictions": predictions,
        "iterations": int(model.n_iter_),
        "feature_count": int(training_matrix.shape[1]),
        "amenity_feature_count": len(amenities.get_feature_names_out()),
        "extra_categorical_columns": list(extra_categorical_columns),
    }


def create_submission(
    ids: pd.Series,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    """Create an ordered, nonnegative Kaggle submission frame."""
    ids = pd.Series(ids, copy=False)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Submission IDs must be unique and non-null")

    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(ids),):
        raise ValueError("IDs and predictions must have the same length")
    if not np.isfinite(predictions).all():
        raise ValueError("Submission predictions must be finite")

    clipped_count = int((predictions < 0).sum())
    clipped = np.clip(predictions, 0.0, None)
    submission = pd.DataFrame(
        {
            "Id": ids.to_numpy(copy=True),
            "Predicted": clipped,
        }
    )
    return submission, clipped_count


def validate_submission(
    submission: pd.DataFrame,
    test_ids: pd.Series,
) -> dict[str, float | int]:
    """Validate schema, row order, IDs, and numerical submission values."""
    if submission.columns.tolist() != ["Id", "Predicted"]:
        raise ValueError("Submission columns must be exactly Id and Predicted")
    test_ids = pd.Series(test_ids, copy=False)
    if len(submission) != len(test_ids):
        raise ValueError("Submission row count must match test IDs")
    if submission["Id"].isna().any() or submission["Id"].duplicated().any():
        raise ValueError("Submission IDs must be unique and non-null")
    if not np.array_equal(submission["Id"].to_numpy(), test_ids.to_numpy()):
        raise ValueError("Submission IDs must preserve the test ID order")

    predictions = pd.to_numeric(submission["Predicted"], errors="coerce").to_numpy()
    nonfinite = int((~np.isfinite(predictions)).sum())
    negative = int((predictions < 0).sum())
    if nonfinite:
        raise ValueError("Submission predictions must be finite")
    if negative:
        raise ValueError("Submission predictions must be nonnegative")

    return {
        "row_count": len(submission),
        "duplicate_ids": int(submission["Id"].duplicated().sum()),
        "nonfinite_predictions": nonfinite,
        "negative_predictions": negative,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
    }
