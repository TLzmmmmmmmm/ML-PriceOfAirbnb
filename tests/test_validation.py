import math

import pandas as pd
import pytest


def test_assign_folds_is_deterministic_and_balanced():
    from src.validation import assign_folds

    ids = pd.Series(range(101, 121), name="id")

    first = assign_folds(ids, n_splits=5, random_state=151)
    second = assign_folds(ids, n_splits=5, random_state=151)

    pd.testing.assert_series_equal(first, second)
    assert set(first) == {0, 1, 2, 3, 4}
    assert first.value_counts().sort_index().tolist() == [4, 4, 4, 4, 4]


def test_assign_folds_is_stable_when_rows_are_reordered():
    from src.validation import assign_folds

    original = pd.Series([40, 10, 30, 20, 60, 50], name="id")
    reordered = original.sample(frac=1, random_state=7)

    original_by_id = dict(zip(original, assign_folds(original, n_splits=3)))
    reordered_by_id = dict(zip(reordered, assign_folds(reordered, n_splits=3)))

    assert original_by_id == reordered_by_id


@pytest.mark.parametrize(
    "ids",
    [
        pd.Series([1, 1, 2], name="id"),
        pd.Series([1, None, 2], name="id"),
    ],
)
def test_assign_folds_rejects_ids_that_cannot_define_stable_rows(ids):
    from src.validation import assign_folds

    with pytest.raises(ValueError, match="IDs must be unique and non-null"):
        assign_folds(ids, n_splits=2)


def test_rmse_matches_hand_calculation_in_original_units():
    from src.validation import rmse

    actual = [100.0, 200.0]
    predicted = [100.0, 202.0]

    assert rmse(actual, predicted) == pytest.approx(math.sqrt(2.0))


def test_rmse_rejects_non_finite_predictions():
    from src.validation import rmse

    with pytest.raises(ValueError, match="finite"):
        rmse([100.0, 200.0], [100.0, float("nan")])


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([100.0, 200.0], [100.0]),
        ([], []),
    ],
)
def test_rmse_rejects_inputs_without_matching_nonzero_lengths(actual, predicted):
    from src.validation import rmse

    with pytest.raises(ValueError, match="same non-zero length"):
        rmse(actual, predicted)
