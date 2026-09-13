# Environment & Dependency Record

Two verified environments. No version is invented: every value below is
either OBSERVED on a machine in this project (2026-08-17, Phase 16) or read
from a verified project record.

## 1. PC — verification & tests (OBSERVED 2026-08-17)

Python 3.11.9 (Windows), `pip` 24.0.

| Package | Version | Evidence |
|---|---|---|
| lightgbm | 4.7.0 | `pip`/importlib metadata on this PC, Phase 16 |
| numpy | 2.4.6 | same |
| pandas | 3.0.5 | same |
| scikit-learn | 1.9.0 | same |
| pyarrow | 25.0.1 | same |
| pytest | 9.1.1 | same |

Not installed on the PC (not required for local verification):
duckdb, shap, xgboost, catboost, networkx, torch.

`requirements.txt` pins exactly these versions.

## 2. Kaggle kernel — training & execution (OBSERVED 2026-08-16)

- Python 3.12.13, 4 CPU, ~33.7 GB RAM, Tesla T4
  (source: `docs/kaggle-execution-policy.md`).
- **lightgbm 4.6.0** — the version that trained the frozen `lgbm-graph-v1`
  model (source: `reports/artifacts/phase7_freeze_lgbm-graph-v1.json`
  `model.version` = "4.6.0").
- duckdb, shap and the full smoke-test stack are present on Kaggle
  (source: `smoke_test.py` checks imports of pandas, numpy, torch,
  lightgbm, xgboost, catboost, shap, networkx).
- Exact versions of the remaining Kaggle packages are **NOT RECORDED** in
  project artifacts — the kernel image floats. They are therefore not pinned
  anywhere; do not fabricate them.

## 3. Version compatibility evidence

- **Training (Kaggle)**: lightgbm 4.6.0 — authoritative for the frozen model.
- **Verification (PC)**: lightgbm 4.7.0 — the Phase 15 gate reproduced the
  frozen TEST score vector with max abs diff **5.551115123125783e-17**
  (54,081/54,108 rows bit-exact; tolerance 1e-12) and reproduced the CAL AUC
  exactly (diff 0.0). Source: `reports/artifacts/phase15_experiment.json`
  `gates`. The two versions are interchangeable for deterministic scoring of
  this model within project tolerances (OBSERVED).
- numpy 2.4.6 / pandas 3.0.5 (PC) vs the Kaggle image: local test suite and
  Phase 13–15 local runs ran green on the PC stack (OBSERVED 2026-08-17).

## 4. Raw data and secrets

- Raw CERT data exists only on Kaggle; never copy it to the PC.
- Local Kaggle credentials live in `.env.kaggle` and session identifiers in
  `KAGGLE_RUNTIME_ID.txt` / `KAGGLE_RUNTIME_URL.txt`. They are local
  configuration, excluded from the repository's secret scan, never committed,
  and documented generically only.