# E016 Location Categories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the isolated five-fold RMSE effect of adding `neighbourhood_cleansed` and `zipcode` to E014.

**Architecture:** Extend the existing structured preprocessor with an optional tuple of extra categorical columns, then pass the two approved location columns through the existing E014 amenities cross-validation path. Preserve all E014 model, fold, amenity, and feature-group settings so the comparison changes only the two location columns.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, pytest, joblib

**Spec:** User-approved E016 design in the 2026-11-11 conversation; project-wide rules are in `CHALLENGE_PLAN.md`.

## Global Constraints

- Use the fixed five-fold assignment with random seed `151`.
- Fit category vocabularies only on the current training fold.
- Keep E014's HGB parameters, text-structure features, and amenity settings unchanged.
- Do not create or upload a Kaggle submission.
- Retain E016 only if mean RMSE improves by at least `0.5` and at least four of five folds improve.
- Keep raw data, OOF predictions, fitted models, and submissions out of Git.

## Review Focus

- A validation-only location category must not enter the training vocabulary.
- Missing location values must have a stable categorical representation.
- Unknown location categories must transform without failure.
- Omitting extra columns must preserve the existing preprocessor behavior.
- E016 must differ from E014 only by the two approved location columns.

---

### Task 1: Optional extra categorical columns

**Files:**
- Modify: `src/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: raw listing `DataFrame` and an optional tuple of column names.
- Produces: `StructuredPreprocessor(extra_categorical_columns: tuple[str, ...] = ())` with fold-trained one-hot output.

- [x] Add failing tests for included columns, missing values, unseen categories, and unchanged default behavior.
- [x] Run `python -m pytest tests/test_preprocessing.py -q` and verify the new tests fail.
- [x] Add the minimum optional-column support to `prepare_structured_frame` and `StructuredPreprocessor`.
- [x] Run `python -m pytest tests/test_preprocessing.py -q` and verify it passes.

### Task 2: E016 cross-validation path

**Files:**
- Modify: `src/train_amenities.py`
- Modify: `tests/test_amenities.py`

**Interfaces:**
- Consumes: `extra_categorical_columns` in `evaluate_amenities_cv`.
- Produces: fold artifacts and metrics identifying the exact extra columns.

- [x] Add a failing test proving the requested columns are present in a saved fold preprocessor and reported in results.
- [x] Run the targeted amenities test and verify it fails.
- [x] Pass the tuple to `StructuredPreprocessor` and include it in the result dictionary.
- [x] Run the targeted and full test suites and verify they pass.

### Task 3: Run and record E016

**Files:**
- Create: `experiments/E016_location_categories.json`
- Create: `reports/location_category_ablation.md`
- Modify: `CHALLENGE_PLAN.md`

**Interfaces:**
- Consumes: untouched `train.csv`, fixed folds, E014 settings, and the two approved location columns.
- Produces: E016 OOF metrics, ignored local artifacts, and a retain/reject decision.

- [x] Run five-fold E016 with `neighbourhood_cleansed` and `zipcode`.
- [x] Verify every row has one finite OOF prediction and compare all five folds with E014.
- [x] Record aggregate results and the threshold-based decision without committing row-level predictions.
- [x] Run `python -m pytest -q` and verify the complete suite passes.
- [x] Verify Git contains no raw data, model, OOF, or submission artifact changes.
- [ ] Commit and push the E016 code, tests, aggregate experiment record, report, and plan update.
