# Project Structure

```
ISM-Project-Shafe/
├── README.md                           # Project overview and quick start
├── requirements.txt                    # Python dependencies
├── CITATION.cff                        # Citation metadata
├── LICENSE                             # License
├── .gitignore                          # Git ignore rules
│
├── src/                                # Core ML source code
│   ├── __init__.py
│   ├── config.py                       # Central configuration, schemas, feature registry
│   ├── data/                           # Data loading, validation, aggregation, labels
│   │   ├── validation.py               # CERT r4.2 schema validation
│   │   ├── aggregation.py              # User-day aggregation (leakage-safe)
│   │   └── labels.py                   # Malicious label generation
│   ├── preprocessing/                  # Splits, alignment
│   │   ├── splits.py                   # Chronological temporal splits
│   │   └── alignment.py               # Key-alignment verification
│   ├── graph/                          # Graph feature engineering
│   │   └── features.py                 # Leakage-safe graph features (Phase 5)
│   ├── models/                         # Model training
│   │   └── lightgbm_baseline.py        # LightGBM trainer (feature-registry enforced)
│   ├── evaluation/                     # Metrics, threshold selection
│   │   ├── metrics.py                  # AUPRC, AUROC, MCC, etc.
│   │   └── threshold.py               # CALIBRATION-only threshold selection
│   └── experiments/                    # Phase experiment scripts (6-20)
│       ├── phase6.py                   # Graph-vs-behavioral regression
│       ├── phase7.py                   # Robustness/freeze
│       ├── phase8.py                   # Adaptive risk v1
│       ├── phase9.py                   # Alert prioritization
│       ├── phase10.py                  # Conformal prediction
│       ├── phase11.py                  # SHAP explainability
│       ├── phase12.py                  # Operational envelope
│       ├── phase13.py                  # Temporal stability
│       ├── phase14.py                  # User holdout
│       ├── phase15.py                  # Unseen-user diagnosis
│       ├── phase17.py                  # Calibration transfer
│       ├── phase18.py                  # Adaptive risk v2
│       ├── phase19.py                  # Fusion benchmark
│       └── phase20.py                  # Final decision engine
│
├── configs/                            # Phase 20 frozen configurations
│   └── phase20/
│       ├── conformal_integration.yaml  # Conformal prediction config
│       ├── decision_policy.yaml        # Risk-level decision rules
│       ├── final_feature_schema.yaml   # 12-feature model schema
│       └── output_schema.yaml          # 21-column output schema
│
├── tests/                              # Pytest test suite (24 files)
│   ├── conftest.py                     # Shared fixtures
│   ├── test_validation.py              # Data validation tests
│   ├── test_labels.py                  # Label generation tests
│   ├── test_aggregation.py             # User-day aggregation tests
│   ├── test_splits.py                  # Temporal split tests
│   ├── test_threshold.py               # Threshold selection tests
│   ├── test_baseline.py                # Model + evaluation tests
│   ├── test_graph.py                   # Graph feature tests
│   ├── test_leakage.py                 # Leakage prevention tests
│   ├── test_real_data.py               # Real CERT data integration
│   └── test_phase{6-20}.py             # Per-phase experiment tests
│
├── artifacts/                          # Frozen production artifacts
│   ├── model/
│   │   └── lgbm-graph-v1.txt           # Frozen LightGBM model
│   ├── frozen/                         # Freeze records and manifests
│   └── summaries/                      # Feature importance summaries
│
├── demo_artifacts/                     # Demo/teaching artifacts
│   └── final_user_day_decisions.parquet  # 501K x 21 decision table
│
├── notebooks/                          # Jupyter notebooks
│   └── ISM_Project_Shafe_Full_Workflow_Demo.ipynb
│
├── reports/                            # Authoritative reports
│   ├── ISM_MASTER_REPORT.md            # Cumulative project handoff
│   ├── final/                          # Final consolidated reports
│   └── artifacts/                      # Phase-specific artifacts
│
├── knowledge/                          # Project knowledge base (Markdown)
│   ├── decisions/                      # Established decisions
│   ├── architecture/                   # System architecture
│   └── features/                       # Feature registry
│
├── data/                               # Dataset directory (Git-tracked structure only)
│   ├── README.md                       # Dataset placement instructions
│   ├── raw/cert_r4.2/                  # CERT r4.2 (user-provided)
│   ├── interim/                        # Intermediate processing
│   └── processed/                      # Processed features
│
├── docs/                               # Documentation
│   ├── REPRODUCIBILITY.md              # Reproducibility guide
│   ├── RUNBOOK.md                      # Step-by-step runbook
│   ├── ARTIFACTS.md                    # Artifact registry
│   ├── PROJECT_STRUCTURE.md            # This file
│   └── ...                             # Additional docs
│
├── experiments/                        # Experiment ledger (JSON manifests)
│
└── scripts/                            # Utility scripts
    └── verify_project.py               # Project verification
```
