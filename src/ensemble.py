"""Transparent weighted ensembles evaluated on fixed OOF predictions."""

import numpy as np
import pandas as pd

from src.validation import rmse


def evaluate_blend(
    actual: np.ndarray,
    folds: pd.Series,
    component_predictions: dict[str, np.ndarray],
    *,
    weights: dict[str, float],
) -> dict[str, object]:
    """Blend aligned OOF predictions and calculate fixed-fold RMSE."""
    if component_predictions.keys() != weights.keys():
        raise ValueError("Predictions and weights must use the same component names")

    weight_values = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("Ensemble weights must be finite and nonnegative")
    if not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Ensemble weights must sum to 1")

    actual = np.asarray(actual, dtype=float)
    folds = pd.Series(folds, copy=False)
    predictions = np.zeros_like(actual)
    for name, values in component_predictions.items():
        values = np.asarray(values, dtype=float)
        if values.shape != actual.shape:
            raise ValueError("All prediction arrays must have the same shape as actual")
        if not np.isfinite(values).all():
            raise ValueError("Prediction arrays must contain only finite values")
        predictions += weights[name] * values

    fold_scores = [
        rmse(actual[folds.eq(fold)], predictions[folds.eq(fold)])
        for fold in sorted(folds.unique())
    ]
    return {
        "weights": dict(weights),
        "fold_rmse": fold_scores,
        "mean_rmse": float(np.mean(fold_scores)),
        "std_rmse": float(np.std(fold_scores)),
        "overall_oof_rmse": rmse(actual, predictions),
        "negative_predictions": int((predictions < 0).sum()),
        "predictions": predictions,
    }
