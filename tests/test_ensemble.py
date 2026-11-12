import numpy as np
import pandas as pd
import pytest


def test_evaluate_blend_returns_weighted_predictions_and_fold_metrics():
    from src.ensemble import evaluate_blend

    actual = np.array([0.0, 2.0, 4.0, 6.0])
    folds = pd.Series([0, 0, 1, 1], name="fold")
    components = {
        "A": np.array([0.0, 0.0, 4.0, 4.0]),
        "B": np.array([2.0, 2.0, 6.0, 6.0]),
    }

    result = evaluate_blend(
        actual,
        folds,
        components,
        weights={"A": 0.5, "B": 0.5},
    )

    np.testing.assert_array_equal(result["predictions"], [1.0, 1.0, 5.0, 5.0])
    assert result["fold_rmse"] == [1.0, 1.0]
    assert result["mean_rmse"] == 1.0
    assert result["overall_oof_rmse"] == 1.0
    assert result["weights"] == {"A": 0.5, "B": 0.5}


def test_evaluate_blend_rejects_mismatched_components_and_weights():
    from src.ensemble import evaluate_blend

    actual = np.array([1.0, 2.0])
    folds = pd.Series([0, 1], name="fold")

    with pytest.raises(ValueError, match="same component names"):
        evaluate_blend(
            actual,
            folds,
            {"A": np.array([1.0, 2.0])},
            weights={"B": 1.0},
        )


def test_evaluate_blend_rejects_invalid_weights_and_prediction_shapes():
    from src.ensemble import evaluate_blend

    actual = np.array([1.0, 2.0])
    folds = pd.Series([0, 1], name="fold")
    components = {"A": np.array([1.0, 2.0]), "B": np.array([2.0, 3.0])}

    with pytest.raises(ValueError, match="sum to 1"):
        evaluate_blend(
            actual,
            folds,
            components,
            weights={"A": 0.8, "B": 0.3},
        )
    with pytest.raises(ValueError, match="nonnegative"):
        evaluate_blend(
            actual,
            folds,
            components,
            weights={"A": 1.1, "B": -0.1},
        )
    with pytest.raises(ValueError, match="same shape"):
        evaluate_blend(
            actual,
            folds,
            {"A": np.array([1.0]), "B": np.array([2.0, 3.0])},
            weights={"A": 0.5, "B": 0.5},
        )
