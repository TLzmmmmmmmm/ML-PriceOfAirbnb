"""Training and cross-validation for the first structured tabular model."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.preprocessing import StructuredPreprocessor
from src.validation import rmse


def build_e003_model() -> HistGradientBoostingRegressor:
    """Return the fixed E003 histogram gradient boosting configuration."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=151,
    )


def evaluate_e003_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    artifact_dir: str | Path,
    experiment_id: str = "E003",
    feature_groups: tuple[str, ...] = (),
) -> dict[str, object]:
    """Run fixed-fold CV, save fold models, and return OOF metrics."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds = folds.reindex(frame.index)
    oof_predictions = np.full(len(frame), np.nan, dtype=float)
    fold_scores: list[float] = []
    iterations: list[int] = []
    feature_counts: list[int] = []
    started = perf_counter()

    for fold in sorted(folds.unique()):
        validation_mask = folds.eq(fold).to_numpy()
        training_frame = frame.loc[~validation_mask]
        validation_frame = frame.loc[validation_mask]

        preprocessor = StructuredPreprocessor(feature_groups=feature_groups)
        training_matrix = preprocessor.fit_transform(training_frame)
        validation_matrix = preprocessor.transform(validation_frame)
        model = build_e003_model()
        model.fit(training_matrix, training_frame["price"].to_numpy(dtype=float))
        predictions = model.predict(validation_matrix)

        oof_predictions[validation_mask] = predictions
        fold_scores.append(rmse(validation_frame["price"], predictions))
        iterations.append(int(model.n_iter_))
        feature_counts.append(training_matrix.shape[1])
        joblib.dump(
            {
                "preprocessor": preprocessor,
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
        "iterations": iterations,
        "feature_counts": feature_counts,
        "feature_groups": list(feature_groups),
        "runtime_seconds": perf_counter() - started,
        "oof_predictions": oof_predictions,
    }
