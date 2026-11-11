"""Evaluate target transformations and robust objectives for price regression."""

from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.preprocessing import StructuredPreprocessor
from src.train_tabular import build_e003_model
from src.validation import rmse


TargetStrategy = Literal["log1p_squared_error", "raw_absolute_error"]


def transform_log_target(prices: np.ndarray) -> np.ndarray:
    """Transform nonnegative dollar prices with log(1 + price)."""
    return np.log1p(np.asarray(prices, dtype=float))


def inverse_log_target(log_prices: np.ndarray) -> np.ndarray:
    """Convert log(1 + price) predictions back to dollars."""
    return np.expm1(np.asarray(log_prices, dtype=float))


def build_strategy_model(strategy: TargetStrategy) -> HistGradientBoostingRegressor:
    """Build a target-strategy model with the E003 configuration."""
    model = build_e003_model()
    if strategy == "log1p_squared_error":
        return model
    if strategy == "raw_absolute_error":
        return model.set_params(loss="absolute_error")
    raise ValueError(f"Unknown target strategy: {strategy}")


def price_band_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    """Calculate dollar-scale error metrics for four fixed true-price bands."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    masks = {
        "price_le_100": actual <= 100,
        "price_100_to_300": (actual > 100) & (actual <= 300),
        "price_300_to_500": (actual > 300) & (actual <= 500),
        "price_gt_500": actual > 500,
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for name, mask in masks.items():
        if not mask.any():
            result[name] = {"count": 0, "rmse": None, "mean_error": None}
            continue
        errors = predicted[mask] - actual[mask]
        result[name] = {
            "count": int(mask.sum()),
            "rmse": rmse(actual[mask], predicted[mask]),
            "mean_error": float(np.mean(errors)),
        }
    return result


def evaluate_target_strategy_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    strategy: TargetStrategy,
    artifact_dir: str | Path,
    experiment_id: str,
) -> dict[str, object]:
    """Evaluate one target strategy and save fold models plus OOF predictions."""
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
        preprocessor = StructuredPreprocessor()
        training_matrix = preprocessor.fit_transform(training_frame)
        validation_matrix = preprocessor.transform(validation_frame)
        model = build_strategy_model(strategy)

        training_target = training_frame["price"].to_numpy(dtype=float)
        if strategy == "log1p_squared_error":
            model.fit(training_matrix, transform_log_target(training_target))
            predictions = inverse_log_target(model.predict(validation_matrix))
        else:
            model.fit(training_matrix, training_target)
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
                "strategy": strategy,
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
        "runtime_seconds": perf_counter() - started,
        "negative_predictions": int((oof_predictions < 0).sum()),
        "price_band_metrics": price_band_metrics(
            frame["price"].to_numpy(dtype=float),
            oof_predictions,
        ),
        "oof_predictions": oof_predictions,
    }
