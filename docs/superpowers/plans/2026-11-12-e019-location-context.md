# E019 Additional Location Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the isolated effect of adding `host_neighbourhood`, `city`, and `host_location` to E018 CatBoost.

**Architecture:** Generalize the tested CatBoost feature/CV core just enough to accept an explicit categorical-column tuple, while retaining E018 wrappers and defaults. Add E019 wrappers that supply the three approved fields, then run the unchanged sequential outer/inner validation protocol.

**Tech Stack:** Python 3.12, CatBoost 1.2.10, pandas, NumPy, scikit-learn, pytest, joblib

**Spec:** User-approved E019 design in the 2026-11-12 conversation; project-wide rules are in `CHALLENGE_PLAN.md`.

## Global Constraints

- E019 differs from E018 only by `host_neighbourhood`, `city`, and `host_location`.
- Do not add `host_id`, `host_name`, new engineered features, or new parameters.
- Preserve the fixed outer folds and internal 10% early-stopping split.
- Use sequential CPU training with the existing 2,000-tree cap and memory settings.
- Retain the feature group only if E019 improves E018 by at least `0.3`, improves at least four folds, and does not materially worsen dispersion.
- Do not generate or upload a submission.

## Review Focus

- E018 defaults and prior reproducibility must remain unchanged.
- All 21 E019 categorical fields must be stable strings with explicit missing values.
- Outer-validation rows must not enter the amenity vocabulary or early-stopping data.
- Metrics must be compared with both E018 and E017.
- Models and OOF predictions must remain Git-ignored.

---

### Task 1: Parameterized CatBoost feature/CV core

**Files:**
- Modify: `src/train_catboost.py`
- Modify: `tests/test_train_catboost.py`

- [x] Add failing tests for the three E019 fields, 21 categorical fields, and unchanged E018 defaults.
- [x] Verify RED failures before implementation.
- [x] Add the minimum shared feature/CV path plus E018/E019 wrappers.
- [x] Run targeted and complete CatBoost-environment tests.

### Task 2: Five-fold E019 experiment

**Files:**
- Create: `experiments/E019_additional_location_context.json`
- Create: `reports/e019_location_context_comparison.md`
- Modify: `CHALLENGE_PLAN.md`

- [x] Run sequential five-fold E019 CV.
- [x] Verify finite aligned OOF predictions and five ignored fold artifacts.
- [x] Compare folds, means, dispersion, iterations, and correlations with E018/E017.
- [x] Apply the predefined gate without tuning or submission generation.
- [x] Run fresh CatBoost/base suites, artifact checks, and staged-diff checks.
- [ ] Commit and push safe code, tests, plan, and aggregate records.
