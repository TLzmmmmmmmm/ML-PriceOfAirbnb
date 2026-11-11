import numpy as np
import pandas as pd
import pytest


def _listing_frame(groups):
    size = len(groups)
    return pd.DataFrame(
        {
            "guests_included": np.arange(1, size + 1),
            "extra_people": ["$25.00"] * size,
            "bedrooms": [1.0] * size,
            "bathrooms": [1.5] * size,
            "review_scores_rating": [90.0] * size,
            "neighbourhood_group_cleansed": groups,
        }
    )


def test_official_parsers_preserve_notebook_behavior():
    from src.baselines import official_to_float, official_to_int

    assert official_to_int("$25.00") == -1
    assert official_to_int(1.5) == 1
    assert official_to_int(float("nan")) == -1
    assert official_to_float("95") == 95.0
    assert official_to_float(float("nan")) == -1.0


def test_official_features_use_training_categories_and_notebook_parsing():
    from src.baselines import build_official_features

    frame = pd.DataFrame(
        {
            "guests_included": [2],
            "extra_people": ["$25.00"],
            "bedrooms": [float("nan")],
            "bathrooms": [1.5],
            "review_scores_rating": [float("nan")],
            "neighbourhood_group_cleansed": ["Queens"],
        }
    )

    features = build_official_features(frame, categories=("Brooklyn", "Manhattan"))

    np.testing.assert_array_equal(
        features,
        np.array([[2.0, -1.0, -1.0, 1.0, -1.0, 0.0, 0.0]]),
    )


def test_mean_baseline_uses_only_supplied_training_targets():
    from src.baselines import predict_training_mean

    predictions = predict_training_mean([10.0, 20.0, 30.0], prediction_count=2)

    np.testing.assert_array_equal(predictions, np.array([20.0, 20.0]))


def test_official_linear_baseline_returns_one_finite_prediction_per_row():
    from src.baselines import predict_official_linear

    train = _listing_frame(
        ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"]
    )
    validation = _listing_frame(["Manhattan", "Staten Island"])
    targets = np.array([100.0, 80.0, 60.0, 180.0, 140.0, 120.0])

    predictions = predict_official_linear(train, targets, validation)

    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()


def test_cross_validation_evaluator_scores_each_fold_from_training_only():
    from src.evaluate_baselines import evaluate_baseline_cv

    frame = _listing_frame(
        ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"]
    )
    frame["price"] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    folds = pd.Series([0, 0, 1, 1, 2, 2], index=frame.index, name="fold")

    result = evaluate_baseline_cv(frame, folds, model_name="mean")

    assert result["fold_rmse"] == pytest.approx(
        [30.4138126514911, 5.0, 30.4138126514911]
    )
    assert result["mean_rmse"] == pytest.approx(21.9425417676607)
    assert result["std_rmse"] == pytest.approx(11.9801861744492)
