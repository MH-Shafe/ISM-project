# Runbook

Step-by-step instructions to run the ISM Project Shafe pipeline.

## Prerequisites

- Python 3.11+
- Git
- ~2 GB free disk space

## 1. Clone and set up

```bash
git clone git@github.com:MH-Shafe/ISM-Project-Shafe.git
cd ISM-Project-Shafe

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Obtain CERT r4.2

Download from: https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099

Place files under `data/raw/cert_r4.2/` as documented in `data/README.md`.

## 3. Run tests

```bash
pytest
```

From a clean clone, expect **0 failures**. Tests fall into tiers:

- **Core tests (~448):** Use synthetic fixtures, always pass without any data.
- **Dataset-dependent (~16):** Require CERT r4.2 in `data/raw/cert_r4.2/`. Skipped with explanation when dataset is absent.
- **Kaggle-artifact-dependent (~16):** Require frozen intermediate artifacts from a full Kaggle execution (e.g., `artifacts/phase14_test_predictions.parquet`, `kaggle_scripts/run_phase18.py`, `reports/artifacts/phase19_user_allocation.json`). These are produced by the Kaggle pipeline and not included in the Git repository.
- **Optional-dependency (~16):** Require torch or PyG. Skipped when not installed.

To run the full validation suite, execute the notebooks on Kaggle where the dataset and all artifacts are available.

## 4. Run the pipeline

The main pipeline scripts are in `src/`. Key modules:

- `src/data/validation.py` — validate CERT CSV files
- `src/data/aggregation.py` — user-day aggregation
- `src/data/labels.py` — label generation
- `src/graph/features.py` — graph feature engineering
- `src/models/lightgbm_baseline.py` — model training
- `src/evaluation/metrics.py` — evaluation metrics
- `src/experiments/phase20.py` — final decision engine

## 5. Load frozen model

The frozen LightGBM model is at `artifacts/model/lgbm-graph-v1.txt`.

Load with:
```python
import lightgbm as lgb
model = lgb.Booster(model_file='artifacts/model/lgbm-graph-v1.txt')
```

## 6. Use demo parquet

The pre-computed 501K x 21 decision table is at `demo_artifacts/final_user_day_decisions.parquet`.

```python
import pandas as pd
df = pd.read_parquet('demo_artifacts/final_user_day_decisions.parquet')
print(df.shape)  # (501000, 21)
```

## 7. Open teacher notebook

```bash
jupyter notebook notebooks/ISM_Project_Shafe_Full_Workflow_Demo.ipynb
```

The notebook requires the demo parquet in `demo_artifacts/` and the frozen model in `artifacts/model/`.

## 8. Regenerate reports (if supported)

Report generation is supported through the experiment scripts in `src/experiments/`. See individual phase scripts for details.

## Troubleshooting

- **Import errors:** Ensure `src/` is on your Python path (the notebook handles this automatically).
- **Missing dataset:** Real-data integration tests will be skipped. Core tests use synthetic fixtures.
- **Model loading errors:** Verify LightGBM 4.6.0 is installed.
