import joblib
import numpy as np
import pandas as pd


def _text_training_frame(rows=60):
    row_number = np.arange(rows)
    room_text = np.where(
        row_number % 2 == 0,
        "sunny entire apartment",
        "quiet private room",
    )
    return pd.DataFrame(
        {
            "id": row_number + 1000,
            "name": room_text,
            "summary": np.where(row_number % 3 == 0, None, "near subway"),
            "description": [f"listing number {value}" for value in row_number],
            "space": "comfortable space",
            "neighborhood_overview": "local neighborhood",
            "house_rules": "no smoking",
            "host_about": "experienced host",
            "price": 80.0 + 70.0 * (row_number % 2 == 0),
        }
    )


def test_combine_text_fields_preserves_rows_and_replaces_missing_values():
    from src.train_text import combine_text_fields

    frame = pd.DataFrame(
        {
            "name": ["Loft", None],
            "summary": [None, "Near subway"],
            "description": ["Bright home", "Quiet room"],
        },
        index=[8, 3],
    )

    combined = combine_text_fields(
        frame,
        text_fields=("name", "summary", "description"),
    )

    assert combined.index.tolist() == [8, 3]
    assert combined.tolist() == ["Loft  Bright home", " Near subway Quiet room"]
    assert not combined.str.contains("nan", case=False).any()


def test_text_vectorizers_have_bounded_word_and_character_configurations():
    from src.train_text import build_text_vectorizer

    word = build_text_vectorizer("word")
    character = build_text_vectorizer("character")

    assert word.analyzer == "word"
    assert word.ngram_range == (1, 2)
    assert word.max_features == 50000
    assert character.analyzer == "char_wb"
    assert character.ngram_range == (3, 5)
    assert character.max_features == 60000


def test_combined_vectorizer_contains_word_and_character_branches():
    from src.train_text import build_text_vectorizer

    combined = build_text_vectorizer("combined")

    assert [name for name, _ in combined.transformer_list] == [
        "word",
        "character",
    ]


def test_text_cv_fits_vocabulary_without_validation_only_tokens(tmp_path):
    from src.train_text import evaluate_text_cv

    frame = _text_training_frame()
    folds = pd.Series(np.arange(len(frame)) % 3, index=frame.index, name="fold")
    frame.loc[folds.eq(0), "description"] += " validationonlytoken"

    result = evaluate_text_cv(
        frame,
        folds,
        vectorizer_kind="word",
        artifact_dir=tmp_path,
        experiment_id="TEST_TEXT",
    )

    fold_zero = joblib.load(tmp_path / "TEST_TEXT_fold_0.joblib")
    assert "validationonlytoken" not in fold_zero["vectorizer"].vocabulary_
    assert len(result["fold_rmse"]) == 3
    assert result["oof_predictions"].shape == (len(frame),)
    assert np.isfinite(result["oof_predictions"]).all()
    assert result["vectorizer_kind"] == "word"
    assert sorted(path.name for path in tmp_path.glob("TEST_TEXT_fold_*.joblib")) == [
        "TEST_TEXT_fold_0.joblib",
        "TEST_TEXT_fold_1.joblib",
        "TEST_TEXT_fold_2.joblib",
    ]
    np.testing.assert_allclose(
        np.load(tmp_path / "TEST_TEXT_oof_predictions.npy"),
        result["oof_predictions"],
    )
