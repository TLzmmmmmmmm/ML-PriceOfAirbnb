"""Leakage-safe TF-IDF text models for listing-price regression."""

from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge

from src.validation import rmse


VectorizerKind = Literal["word", "character", "combined"]

TEXT_FIELDS = (
    "name",
    "summary",
    "description",
    "space",
    "neighborhood_overview",
    "house_rules",
    "host_about",
)


def combine_text_fields(
    frame: pd.DataFrame,
    *,
    text_fields: tuple[str, ...] = TEXT_FIELDS,
) -> pd.Series:
    """Join supplied text fields row by row while preserving the input index."""
    text = frame.loc[:, text_fields].astype("string").fillna("")
    return text.agg(" ".join, axis=1)


def build_text_vectorizer(
    kind: VectorizerKind,
) -> TfidfVectorizer | FeatureUnion:
    """Build one bounded TF-IDF configuration."""
    common = {
        "lowercase": True,
        "strip_accents": "unicode",
        "sublinear_tf": True,
        "min_df": 3,
        "dtype": np.float32,
    }
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_df=0.98,
            max_features=50000,
            **common,
        )
    if kind == "character":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=60000,
            **common,
        )
    if kind == "combined":
        return FeatureUnion(
            [
                ("word", build_text_vectorizer("word")),
                ("character", build_text_vectorizer("character")),
            ]
        )
    raise ValueError(f"Unknown vectorizer kind: {kind}")


def evaluate_text_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    vectorizer_kind: VectorizerKind,
    artifact_dir: str | Path,
    experiment_id: str,
) -> dict[str, object]:
    """Fit TF-IDF inside each training fold and return text-only OOF metrics."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds = folds.reindex(frame.index)
    oof_predictions = np.full(len(frame), np.nan, dtype=float)
    fold_scores: list[float] = []
    feature_counts: list[int] = []
    started = perf_counter()

    for fold in sorted(folds.unique()):
        validation_mask = folds.eq(fold).to_numpy()
        training_frame = frame.loc[~validation_mask]
        validation_frame = frame.loc[validation_mask]
        vectorizer = build_text_vectorizer(vectorizer_kind)
        training_matrix = vectorizer.fit_transform(
            combine_text_fields(training_frame)
        )
        validation_matrix = vectorizer.transform(
            combine_text_fields(validation_frame)
        )
        model = Ridge(alpha=20.0, solver="lsqr")
        model.fit(training_matrix, training_frame["price"].to_numpy(dtype=float))
        predictions = model.predict(validation_matrix)

        oof_predictions[validation_mask] = predictions
        fold_scores.append(rmse(validation_frame["price"], predictions))
        feature_counts.append(training_matrix.shape[1])
        joblib.dump(
            {
                "text_fields": TEXT_FIELDS,
                "vectorizer": vectorizer,
                "model": model,
                "fold": int(fold),
                "vectorizer_kind": vectorizer_kind,
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
        "feature_counts": feature_counts,
        "runtime_seconds": perf_counter() - started,
        "negative_predictions": int((oof_predictions < 0).sum()),
        "vectorizer_kind": vectorizer_kind,
        "oof_predictions": oof_predictions,
    }
