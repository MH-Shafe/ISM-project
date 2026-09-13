# Leakage Analysis — CERT r4.2 Insider Threat Project

Status: Phase-1 foundation. This document lists identified leakage risks and the
controls adopted so far. It will be revisited as features evolve.

## Analytical unit and default evaluation design

- Analytical unit: **user x day** (one row = one user on one day).
- Default evaluation design: **chronological TRAIN -> CALIBRATION -> TEST**
  split on the day axis. Random splitting of temporal behavior is rejected by
  default because daily behavior is autocorrelated and malicious windows are
  contiguous in time (`src/preprocessing/splits.py`).
- Default boundaries (baseline, `src/config.py` SPLIT_DEFAULTS):
  train <= 2011-01-31, calibration 2011-02-01..2011-03-31, test >= 2011-04-01
  (dataset spans 2010-01-02 .. 2011-05-17).

## The core question

> Would this information actually be available at prediction time?

Every feature and statistic must answer "yes" for the day being scored.

## Identified risks and controls

| # | Risk | Where it can appear | Control |
|---|------|---------------------|---------|
| 1 | **Future events in features** | Any rolling/frequency statistic computed over the full timeline | All Phase-7 features are computed strictly within one (user, day): verified by the day-independence test (`tests/test_aggregation.py::test_day_independence_no_future_leakage`) |
| 2 | **Answer-key (label) leakage** | `answers/insiders.csv` and scenario observables define malicious users and windows | Labels are consumed only for evaluation, never in feature construction; structural test forbids label columns in the feature table (`tests/test_leakage.py`) |
| 3 | **Target-derived features** | e.g. `sensitive_file_access_count` if "sensitive" is defined from scenario files | Feature is marked **PENDING**; a leakage-safe definition must come from time-of-event information only |
| 4 | **Future frequency statistics** | e.g. user baseline counts, "normal" usage profiles fitted on the whole timeline | Any baseline (e.g. `unusual_access_count`, rare-device flags) must be computed from a strictly past window relative to the scored day; feature marked **PENDING** until such a window is defined |
| 5 | **Graph relationships that only exist in the future** | Interaction graph edges derived from the full timeline | Graph construction (future phase) must build edges from events with timestamp < prediction time only |
| 6 | **Normalization / scaling using test data** | Standard scalers, quantile transforms fit on the union | All scaling must be fit on train only and frozen for calibration/test |
| 7 | **Entity overlap** | Users appear across all splits (a user can be malicious in both train and test windows) | User-day is the unit; evaluation is on user-DAYS, and per-split statistics are reported so overlap effects are visible; no user-level holdout by default (documented limitation) |
| 8 | **Calibration/test contamination** | Conformal calibration scores computed with test information | Calibration set is a separate chronological partition used only for calibration; never merged into train |
| 9 | **Preprocessing before temporal split** | Deduplication, imputation, or filtering using whole-timeline statistics | Preprocessing is per-event and per-day (no global statistics); anything global must be train-only |
| 10 | **Malformed label dates** | `insiders.csv` has 1 unparseable start date (`CDE1846`, `/21/2011 11:43:39`) | Reported, never silently fixed; that incident is excluded from label mapping until manually resolved |
| 11 | **ID collisions across files** | Errata: IDs unique within a file, not globally | Joins never use `id` across files; keys are (user, day) or (user, pc) |
| 12 | **insiders.csv contains other releases** | Rows for datasets 2.0..6.2 are not part of r4.2 | Label mapping filters `dataset == 4.2` (70 incidents) |

## Controls already enforced in code

1. Features are day-local (`tests/test_aggregation.py` day-independence test).
2. No label columns in feature tables (`tests/test_leakage.py`).
3. Temporal split is chronological and exhaustive (`tests/test_splits.py`).
4. Validation is read-only (`tests/test_leakage.py`).
5. Feature registry forces explicit status (defined/pending) and documents
   source + definition per feature (`src/config.py`).

## Open questions (no invented answers)

- ~~Sensitive-file definition for r4.2~~ **Resolved (Phase 3)**: a-priori rule over
  event-time file-type magic bytes (OLE2/PDF/ZIP), declared in
  `config.SENSITIVE_MAGIC_PREFIXES`; never derived from the answer key.
- ~~"Normal usage" baselines for unusual/rare-access features~~ **Resolved (Phase 3)**:
  `unusual_access_count` uses a strictly past 28-day window
  (`config.UNUSUAL_ACCESS_WINDOW_DAYS`); windowed features are held to
  future-independence instead of single-day equality.
- Whether to hold out malicious users entirely at evaluation time (trade-off
  against temporal design; decision deferred until model phase).