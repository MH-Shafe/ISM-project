# ISM Project Shafe

Leakage-resistant insider-threat detection using the CERT r4.2 dataset.

## Objective

Build a production-ready insider-threat detection system that:

1. Prevents data leakage at every pipeline stage
2. Achieves strong detection performance on chronological test data
3. Provides uncertainty quantification via conformal prediction
4. Offers explainability via SHAP
5. Produces deterministic, auditable decisions for analysts

## Final Architecture

```
CERT r4.2
→ validation/preprocessing
→ leakage-safe user-day aggregation
→ behavioral features (8)
→ graph/trust features (4)
→ frozen LightGBM (lgbm-graph-v1)
→ frozen decision policy (4 risk levels)
→ conformal uncertainty (prediction sets)
→ SHAP explanations
→ trust/context diagnostics
→ deterministic Phase 20 decision engine
→ analyst-facing output (21 columns)
```

## Key Results

| Metric | Value |
|---|---|
| Dataset | CERT r4.2 (501,000 user-days) |
| Production model | lgbm-graph-v1 |
| Features | 12 (8 behavioral + 4 graph) |
| ROC-AUC | 0.9392 |
| PR-AUC | 0.2678 |
| F1 | 0.354 |
| MCC | 0.365 |
| TEST alerts | 49 |
| Alert threshold | 0.9186 |
| Conformal coverage (pos) | 100% (30/30) |
| Conformal coverage (neg) | 98.8% |

**Adaptive Risk:** Evaluated across 10 algorithm families (Phases 8, 18, 19). REJECTED for production — delta ROC-AUC = -0.028.

## Installation

### Windows

```bash
git clone git@github.com:MH-Shafe/ISM-Project-Shafe.git
cd ISM-Project-Shafe
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
git clone git@github.com:MH-Shafe/ISM-Project-Shafe.git
cd ISM-Project-Shafe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Setup

1. Download CERT r4.2 from: https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099
2. Extract the archive
3. Place files under `data/raw/cert_r4.2/` as documented in `data/README.md`

## Run Tests

```bash
pytest
```

From a clean clone, expect **0 failures**. Tests are organized in tiers:

| Tier | Count | When skipped |
|---|---|---|
| Core (synthetic fixtures) | ~448 | Never — always pass |
| Dataset-dependent | ~16 | CERT r4.2 not in `data/raw/` |
| Kaggle-artifact-dependent | ~16 | Frozen artifacts not in `artifacts/` or `kaggle_scripts/` |
| Optional-dependency | ~16 | torch/PyG not installed |

Skipped tests document exactly which artifact is needed. To run the full validation suite on Kaggle, see `docs/RUNBOOK.md`.

## Reproduce Workflow

```bash
# Validate dataset
python -c "from src.data.validation import *; ..."

# Generate features
python -c "from src.data.aggregation import *; ..."

# Load frozen model
python -c "import lightgbm as lgb; m = lgb.Booster(model_file='artifacts/model/lgbm-graph-v1.txt')"

# Load demo parquet
python -c "import pandas as pd; df = pd.read_parquet('demo_artifacts/final_user_day_decisions.parquet'); print(df.shape)"
```

See `docs/RUNBOOK.md` for detailed step-by-step instructions.

## Run Teacher Notebook

```bash
jupyter notebook notebooks/ISM_Project_Shafe_Full_Workflow_Demo.ipynb
```

The notebook requires:
- `demo_artifacts/final_user_day_decisions.parquet`
- `artifacts/model/lgbm-graph-v1.txt`
- `configs/phase20/` (loaded automatically)

## Reports

- **Master report:** `reports/ISM_MASTER_REPORT.md`
- **Final reports:** `reports/final/`
- **Reproducibility:** `docs/REPRODUCIBILITY.md`
- **Artifact registry:** `docs/ARTIFACTS.md`

## Reproducibility

- **Seed:** 42
- **Split:** Chronological (TRAIN through 2011-01-31, CAL 2011-02-01 to 2011-03-31, TEST from 2011-04-01)
- **Model:** Frozen since Phase 7 — no retraining allowed
- **Thresholds:** Frozen since Phase 9 — no tuning allowed
- **Adaptive Risk:** Rejected — frozen system unchanged
- **All artifacts:** Hash-verified (MD5, SHA256)

## Limitations

- **Class imbalance:** 0.38% malicious user-days (1,892 / 501,000)
- **Small TEST set:** 30 positives in chronological TEST
- **External validity:** CERT r4.2 is synthetic — real-world generalization is uncertain
- **Conformal assumptions:** Exchangeability under chronological split may not hold perfectly
- **SHAP is not causal:** Feature importance does not imply causation

## License

MIT License — see [LICENSE](LICENSE)

## Citation

See [CITATION.cff](CITATION.cff) for citation metadata.
