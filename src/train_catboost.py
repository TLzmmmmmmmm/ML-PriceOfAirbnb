"""CatBoost evaluation using the same source features as E016."""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from src.amenities import AmenitiesTransformer
from src.preprocessing import (
    LOW_CARDINALITY_CATEGORICAL_COLUMNS,
    prepare_structured_frame,
)
from src.validation import rmse


LOCATION_CATEGORICAL_COLUMNS = ("neighbourhood_cleansed", "zipcode")
CATBOOST_CATEGORICAL_COLUMNS = (
    *LOW_CARDINALITY_CATEGORICAL_COLUMNS,
    *LOCATION_CATEGORICAL_COLUMNS,
)
E019_ADDITIONAL_CATEGORICAL_COLUMNS = (
    "host_neighbourhood",
    "city",
    "host_location",
)
E019_CATEGORICAL_COLUMNS = (
    *CATBOOST_CATEGORICAL_COLUMNS,
    *E019_ADDITIONAL_CATEGORICAL_COLUMNS,
)


def _prepare_catboost_features(
    frame: pd.DataFrame,
    amenities_transformer: AmenitiesTransformer,
    categorical_columns: tuple[str, ...],
) -> pd.DataFrame:
    extra_categorical_columns = tuple(
        column
        for column in categorical_columns
        if column not in LOW_CARDINALITY_CATEGORICAL_COLUMNS
    )
    structured = prepare_structured_frame(
        frame,
        feature_groups=("text_structure",),
        extra_categorical_columns=extra_categorical_columns,
    )
    amenities = pd.DataFrame(
        amenities_transformer.transform(frame["amenities"]),
        columns=amenities_transformer.get_feature_names_out(),
        index=frame.index,
    )
    result = pd.concat([structured, amenities], axis=1)
    for column in categorical_columns:
        result[column] = result[column].map(str)
    return result


def prepare_e018_features(
    frame: pd.DataFrame,
    amenities_transformer: AmenitiesTransformer,
) -> pd.DataFrame:
    """Build CatBoost-native E016 features with a fitted amenity vocabulary."""
    return _prepare_catboost_features(
        frame,
        amenities_transformer,
        CATBOOST_CATEGORICAL_COLUMNS,
    )


def prepare_e019_features(
    frame: pd.DataFrame,
    amenities_transformer: AmenitiesTransformer,
) -> pd.DataFrame:
    """Add the three approved E019 location categories to E018 features."""
    return _prepare_catboost_features(
        frame,
        amenities_transformer,
        E019_CATEGORICAL_COLUMNS,
    )


def build_e018_model(*, iterations: int = 2000) -> CatBoostRegressor:
    """Return the fixed E018 CatBoost regression configuration."""
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=151,
        thread_count=4,
        used_ram_limit="4gb",
        allow_writing_files=False,
        verbose=False,
    )


def fit_full_e018(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    artifact_path: str | Path,
    iterations: int,
) -> dict[str, object]:
    """Fit E018 on every training row with a fixed tree count."""
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    amenities = AmenitiesTransformer(
        include_indicators=True,
        min_frequency=0.01,
        max_features=100,
    ).fit(training["amenities"])
    training_features = prepare_e018_features(training, amenities)
    test_features = prepare_e018_features(test, amenities)
    model = build_e018_model(iterations=iterations)
    model.fit(
        training_features,
        training["price"].to_numpy(dtype=float),
        cat_features=list(CATBOOST_CATEGORICAL_COLUMNS),
        verbose=False,
    )
    predictions = model.predict(test_features)
    joblib.dump(
        {
            "amenities_transformer": amenities,
            "categorical_columns": CATBOOST_CATEGORICAL_COLUMNS,
            "iterations": iterations,
            "model": model,
        },
        artifact_path,
    )
    return {
        "predictions": predictions,
        "training_rows": len(training),
        "iterations": int(model.tree_count_),
        "feature_count": training_features.shape[1],
        "amenity_feature_count": len(amenities.get_feature_names_out()),
    }


def _evaluate_catboost_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    artifact_dir: str | Path,
    experiment_id: str,
    categorical_columns: tuple[str, ...],
) -> dict[str, object]:
    """Run fixed outer-fold CatBoost CV for an explicit category set."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    folds = folds.reindex(frame.index)
    oof_predictions = np.full(len(frame), np.nan, dtype=float)
    fold_scores: list[float] = []
    best_iterations: list[int] = []
    feature_counts: list[int] = []
    amenity_feature_counts: list[int] = []
    selected_amenities: list[list[str]] = []
    started = perf_counter()

    for fold in sorted(folds.unique()):
        validation_mask = folds.eq(fold).to_numpy()
        training_frame = frame.loc[~validation_mask]
        validation_frame = frame.loc[validation_mask]

        amenities = AmenitiesTransformer(
            include_indicators=True,
            min_frequency=0.01,
            max_features=100,
        ).fit(training_frame["amenities"])
        training_features = _prepare_catboost_features(
            training_frame,
            amenities,
            categorical_columns,
        )
        validation_features = _prepare_catboost_features(
            validation_frame,
            amenities,
            categorical_columns,
        )
        training_target = training_frame["price"].to_numpy(dtype=float)
        inner_training, inner_validation = train_test_split(
            np.arange(len(training_frame)),
            test_size=0.1,
            random_state=151,
        )

        model = build_e018_model()
        model.fit(
            training_features.iloc[inner_training],
            training_target[inner_training],
            cat_features=list(categorical_columns),
            eval_set=(
                training_features.iloc[inner_validation],
                training_target[inner_validation],
            ),
            early_stopping_rounds=100,
            verbose=False,
        )
        predictions = model.predict(validation_features)

        oof_predictions[validation_mask] = predictions
        fold_scores.append(rmse(validation_frame["price"], predictions))
        best_iterations.append(int(model.tree_count_))
        feature_counts.append(training_features.shape[1])
        amenity_feature_counts.append(len(amenities.get_feature_names_out()))
        selected_amenities.append(list(amenities.selected_amenities_))
        joblib.dump(
            {
                "amenities_transformer": amenities,
                "categorical_columns": categorical_columns,
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
        "best_iterations": best_iterations,
        "feature_counts": feature_counts,
        "amenity_feature_counts": amenity_feature_counts,
        "selected_amenities": selected_amenities,
        "runtime_seconds": perf_counter() - started,
        "negative_predictions": int((oof_predictions < 0).sum()),
        "oof_predictions": oof_predictions,
    }


def evaluate_e018_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    artifact_dir: str | Path,
    experiment_id: str = "E018",
) -> dict[str, object]:
    """Run the fixed E018 CatBoost cross-validation configuration."""
    return _evaluate_catboost_cv(
        frame,
        folds,
        artifact_dir=artifact_dir,
        experiment_id=experiment_id,
        categorical_columns=CATBOOST_CATEGORICAL_COLUMNS,
    )


def evaluate_e019_cv(
    frame: pd.DataFrame,
    folds: pd.Series,
    *,
    artifact_dir: str | Path,
    experiment_id: str = "E019",
) -> dict[str, object]:
    """Run E018 CV with the three approved E019 location categories."""
    return _evaluate_catboost_cv(
        frame,
        folds,
        artifact_dir=artifact_dir,
        experiment_id=experiment_id,
        categorical_columns=E019_CATEGORICAL_COLUMNS,
    )
