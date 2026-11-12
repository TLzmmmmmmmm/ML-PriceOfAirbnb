"""Cross-validation for E009 plus parsed amenity features."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.amenities import AmenitiesTransformer
from src.preprocessing import StructuredPreprocessor
from src.train_tabular import build_e003_model
from src.validation import rmse


def evaluate_amenities_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    include_indicators: bool,
    artifact_dir: str | Path,
    experiment_id: str,
    min_frequency: float = 0.01,
    max_features: int = 100,
    extra_categorical_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    """Evaluate E009 with fold-trained amenity count and indicators."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds = folds.reindex(frame.index)
    oof_predictions = np.full(len(frame), np.nan, dtype=float)
    fold_scores: list[float] = []
    iterations: list[int] = []
    feature_counts: list[int] = []
    amenity_feature_counts: list[int] = []
    selected_amenities: list[list[str]] = []
    started = perf_counter()

    for fold in sorted(folds.unique()):
        validation_mask = folds.eq(fold).to_numpy()
        training_frame = frame.loc[~validation_mask]
        validation_frame = frame.loc[validation_mask]

        structured = StructuredPreprocessor(
            feature_groups=("text_structure",),
            extra_categorical_columns=extra_categorical_columns,
        )
        amenities = AmenitiesTransformer(
            include_indicators=include_indicators,
            min_frequency=min_frequency,
            max_features=max_features,
        )
        training_matrix = np.hstack(
            [
                structured.fit_transform(training_frame),
                amenities.fit_transform(training_frame["amenities"]),
            ]
        )
        validation_matrix = np.hstack(
            [
                structured.transform(validation_frame),
                amenities.transform(validation_frame["amenities"]),
            ]
        )
        model = build_e003_model()
        model.fit(training_matrix, training_frame["price"].to_numpy(dtype=float))
        predictions = model.predict(validation_matrix)

        oof_predictions[validation_mask] = predictions
        fold_scores.append(rmse(validation_frame["price"], predictions))
        iterations.append(int(model.n_iter_))
        feature_counts.append(training_matrix.shape[1])
        amenity_feature_counts.append(
            len(amenities.get_feature_names_out())
        )
        selected_amenities.append(list(amenities.selected_amenities_))
        joblib.dump(
            {
                "structured_preprocessor": structured,
                "amenities_transformer": amenities,
                "model": model,
                "fold": int(fold),
            },
            artifact_dir / f"{experiment_id}_fold_{int(fold)}.joblib",
        )

    if not np.isfinite(oof_predictions).all():
        raise ValueError("Every training row must receive one finite OOF prediction")

    np.save(artifact_dir / f"{experiment_id}_oof_predictions.npy", oof_predictions)
    return {
        "fold_rmse": fold_scores,
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "overall_oof_rmse": rmse(frame["price"], oof_predictions),
        "iterations": iterations,
        "feature_counts": feature_counts,
        "amenity_feature_counts": amenity_feature_counts,
        "selected_amenities": selected_amenities,
        "runtime_seconds": perf_counter() - started,
        "negative_predictions": int((oof_predictions < 0).sum()),
        "include_indicators": include_indicators,
        "min_frequency": min_frequency,
        "max_features": max_features,
        "extra_categorical_columns": list(extra_categorical_columns),
        "oof_predictions": oof_predictions,
    }
