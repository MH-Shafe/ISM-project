# Architecture (Verified)

Current, verified architecture of the CERT r4.2 Insider Threat Detection project.
Future components are marked **PLANNED** and are not implemented.

## Data flow

```
PC
└─ OpenCode
   └─ Jupyter MCP (opencode.json: jupyter-mcp-server --sandbox-variant kaggle)
      └─ Kaggle kernel (Jupyter server; files exchanged via contents API;
         root = /kaggle/working)
         └─ CERT r4.2 dataset (mounted read-only at /kaggle/input/...)
            └─ user-day pipeline (kaggle_scripts/build_user_day.py
               -> src/data/aggregation.py, duckdb path; pandas path for tests)
               -> artifacts/user_day_features.parquet (501,000 x 11)
                  └─ LightGBM baseline (src/models/lightgbm_baseline.py)
                     -> TEST evaluation (src/evaluation/metrics.py, threshold.py)
                        -> frozen record: lgbm-baseline-v2 (see evaluation/README)
```

## Verified components (OBSERVED)

| Component | Where | Status |
|---|---|---|
| Data validation | `src/data/validation.py`, `kaggle_scripts/run_validation.py` | verified, read-only on `/kaggle/input` |
| Label mapping | `src/data/labels.py` (answer key, `dataset == 4.2` only) | verified |
| User-day aggregation | `src/data/aggregation.py` (duckdb streaming for 14.5 GB http; pandas for tests) | verified |
| Chronological split | `src/preprocessing/splits.py` | verified |
| LightGBM baseline | `src/models/lightgbm_baseline.py` | verified, frozen v2 |
| Evaluation | `src/evaluation/metrics.py` (+ `binary_decision_metrics`, `bootstrap_ci`), `threshold.py` | verified |
| Graph feature table | `src/graph/features.py`, `kaggle_scripts/build_graph_features.py` | built, experimental only |
| Conformal overlay | `src/experiments/phase10.py`, `kaggle_scripts/run_phase10.py` | verified, diagnostic overlay on the frozen system (accepted) |
| Explainability layer | `src/experiments/phase11.py`, `kaggle_scripts/run_phase11.py` | verified, reporting layer on the frozen system (accepted); native `pred_contrib`, margin-space additive, counterfactuals NOT SUPPORTED |
| Execution pattern | background subprocess + log polling on Kaggle (inline cells: 30 s default cap, ~120 s with explicit timeout; longer = background) | verified |

## Planned components (NOT implemented)

- **Graph/trust** — graph-derived signals (see [[graph/README]]); the Phase 5
  feature table exists but no graph model is trained.
- **Adaptive risk** — per-user risk adaptation — PLANNED (Phase 8 candidate
  rejected; not implemented).

## Split protocol (frozen)

TRAIN <= 2011-01-31 (395,000 rows / 1,539 malicious) -> CALIBRATION 2011-02-01..2011-03-31
(59,000 / 323) -> TEST >= 2011-04-01 (47,000 / 30). TEST is evaluated once, never tuned.

## Related

- `docs/kaggle-execution-policy.md` — verified Kaggle execution policy (timeouts, recovery, heartbeats)
- [[dataset/cert-r4.2]] — the dataset
- [[features/feature-registry]] — registered features
- [[experiments/experiment-index]] — reports
- [[decisions/decision-log]] — established decisions
- [[risks/README]] — leakage risks