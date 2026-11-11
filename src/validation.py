"""Shared cross-validation and scoring utilities."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def assign_folds(
    ids: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 151,
) -> pd.Series:
    """Assign stable shuffled K-fold labels after sorting rows by listing ID."""
    ids = pd.Series(ids, copy=False)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("IDs must be unique and non-null")

    sorted_positions = np.argsort(ids.to_numpy(), kind="stable")
    folds = np.empty(len(ids), dtype=np.int8)
    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (_, validation_positions) in enumerate(splitter.split(sorted_positions)):
        folds[sorted_positions[validation_positions]] = fold

    return pd.Series(folds, index=ids.index, name="fold")


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute root mean squared error in the supplied target units."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("RMSE inputs must have the same non-zero length")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RMSE inputs must contain only finite values")

    return float(np.sqrt(np.mean(np.square(actual - predicted))))
