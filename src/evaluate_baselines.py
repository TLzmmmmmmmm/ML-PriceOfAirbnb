"""Cross-validation evaluation for the competition's official baselines."""

from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from src.baselines import predict_official_linear, predict_training_mean
from src.validation import rmse


def evaluate_baseline_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    model_name: Literal["mean", "official_linear"],
) -> dict[str, object]:
    """Evaluate one baseline on every supplied validation fold."""
    folds = folds.reindex(frame.index)
    fold_scores: list[float] = []
    started = perf_counter()

    for fold in sorted(folds.unique()):
        validation_mask = folds.eq(fold)
        training_frame = frame.loc[~validation_mask]
        validation_frame = frame.loc[validation_mask]

        if model_name == "mean":
            predictions = predict_training_mean(
                training_frame["price"],
                prediction_count=len(validation_frame),
            )
        elif model_name == "official_linear":
            predictions = predict_official_linear(
                training_frame,
                training_frame["price"],
                validation_frame,
            )
        else:
            raise ValueError(f"Unknown baseline: {model_name}")

        fold_scores.append(rmse(validation_frame["price"], predictions))

    return {
        "fold_rmse": fold_scores,
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "runtime_seconds": perf_counter() - started,
    }
