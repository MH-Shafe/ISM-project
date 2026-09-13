# ISM Project — Final Project Conclusions

Definitive summary of the CERT r4.2 insider-threat detection project:
retained components, rejected approaches, architecture, detection quality,
limitations, and future work.

**Last updated**: 2026-09-14
**Project status**: Phase 20 complete. Model development STOPPED.
**Frozen system**: `lgbm-graph-v1` + `frozen_max_f1` threshold + conformal overlay + explainability layer

---

## E1. What Was Retained

| Component | Status | Phase | Evidence |
|---|---|---|---|
| User-day analytical unit | FROZEN | 0 | 501K × 9 table, 1,892 malicious rows |
| 8 behavioral features | FROZEN | 3-4 | AUC-PR +34% vs 6-feature baseline |
| 4 graph features | FROZEN | 5-7 | Complementary; 20.5% of alert contribution |
| lgbm-graph-v1 model | FROZEN | 7 | AUC-ROC 0.939, AUC-PR 0.268, 49 alerts |
| frozen_max_f1 threshold | FROZEN | 9 | t = 0.9186, CAL F1 0.589, 49 TEST alerts |
| Conformal confidence | ACCEPTED | 10 | Pos coverage 1.000, Wilson LB 0.917 |
| Explainability layer | ACCEPTED | 11 | 21K rows, recon exact, shap max diff 0.0 |
| Decision engine (21-col) | COMPLETE | 20 | 501K rows, 54/54 tests, determinism PASS |

---

## E2. What Was Rejected (Complete List)

| Approach | Phase | Reason | Evidence |
|---|---|---|---|
| lr 0.05 baseline | 1-2 | Collapsed to best_iter 1, degenerate scores | CAL AUC-ROC 1.000, 0 alerts |
| Graph-only model (Arm B) | 6 | Weak standalone: AUC-PR 0.014, best_iter 3 | Phase 6 experiment |
| department_file_type_mismatch_count | 7 | Zero gain 5/5 seeds, 49 non-zero rows | Phase 7 feature stability |
| Adaptive risk (equal-weight) | 8 | Degrades all metrics vs ML-only | Phase 8 experiment |
| Adaptive risk (learned weights) | 8 | Collapses to ML-only [1,0,0] | Phase 8 optimization |
| Capacity/percentile alert policies | 9 | Fail S1-S3 selection rules | Phase 9 policy screen |
| De-duplication beyond 1 day | 12 | Destroys recall (3d: 0.248, 7d: 0.139) | Phase 12 operational envelope |
| Rolling threshold | 12 | Never beats frozen policy | Phase 12 operational envelope |
| Counterfactual explanations | 11 | Not supported under frozen-output contract | Phase 11 decision |
| Platt calibration (production) | 17 | Mechanical FAIL: AUC-ROC not invariant | Phase 17 verdict |
| Binning calibration (production) | 17 | 21,127 alerts, precision 0.018 — unusable | Phase 17 verdict |
| ECDF calibration (production) | 17 | Brier -216% — percentile map, not calibration | Phase 17 verdict |
| Adaptive risk A0/Family C | 19 | CONFIRM FAIL: delta ROC-AUC -0.0281 < -0.01 | Phase 19 CONFIRM |

---

## E3. Production Architecture

```
Input: CERT r4.2 logs (7 files + LDAP + psychometric + answers)
  ↓
User-Day Aggregation (Phase 0)
  → 501,000 rows × 9 base columns
  ↓
Feature Computation (Phase 3-7, frozen)
  → 8 behavioral + 4 graph = 12 features
  ↓
LightGBM lgbm-graph-v1 (Phase 7, frozen)
  → ml_risk score [0, 1]
  ↓
Alert Threshold (Phase 9, frozen)
  → t = 0.9186015432508062
  ↓
Risk Level Assignment (Phase 11)
  → ALERT / BORDERLINE / MONITOR / NON-ALERT
  ↓
Conformal Confidence (Phase 10, accepted)
  → conformal_set, p0, p1, confidence
  ↓
Explanations (Phase 11, accepted)
  → top_3_reasons, contributions, trust_flag
  ↓
Decision Table (Phase 20, complete)
  → 21 columns, 501,000 rows
  → Dashboard-ready output
```

---

## E4. Detection Quality Summary

### Primary Metrics (TEST, 47,000 rows, 30 positives)

| Metric | Value | Notes |
|---|---|---|
| AUC-ROC | 0.93916 | Bootstrap CI [0.900, 0.975] |
| AUC-PR | 0.26778 | Bootstrap CI [0.140, 0.460] |
| P@10 | 0.600 | 6 of top-10 are true positives |
| P@30 | 0.367 | 11 of top-30 are true positives |
| R@50 | 0.467 | 14 of 30 true positives in top-50 |
| F1 (at threshold) | 0.354 | Threshold = 0.9186 |
| MCC (at threshold) | 0.365 | Balanced measure |
| Alerts | 49 | 1.04/day over 47 days |
| Scenario 2 recall | 0.464 | 13/28 rows |
| Scenario 3 recall | 0.500 | 1/2 rows |

### Full Decision Table (501,000 rows)

| Risk Level | Count | % | Action |
|---|---|---|---|
| ALERT | 3,785 | 0.76% | Immediate escalation |
| BORDERLINE | 1,484 | 0.30% | Analyst review within 24h |
| MONITOR | 178,204 | 35.57% | Weekly watchlist |
| NON-ALERT | 317,527 | 63.38% | No action |

### What the System Detects

- **usb_connection_count** is the dominant alert driver (36/49 alerts top-1)
- **http_activity_count** provides broad behavioral context (bipolar contribution)
- **device_consistency_score** is the strongest graph signal (top-3 in 72% of CAL alerts)
- Detection is incident-length dependent: 2-5 d: 3/3, 6-12 d: 3/4, 45-100 d: 7/8

### What the System Misses

- 2 malicious users (JJM0203, WDD0366) with 56 missed days — ranking failures, not cold start
- 53% of true positives above threshold (16/30) — recall limited by extreme class imbalance
- Tail-window degradation (Phase 13 W8) — score distribution shifts at span boundaries

---

## E5. Limitations

### Data Limitations

1. **30 TEST positives** — every top-k count is an integer multiple of 1/30; wide uncertainty
2. **No post-2011 data** — temporal stability beyond observed span unmeasured
3. **Synthetic dataset** — CERT r4.2 is a simulation, not real organizational data
4. **0.38% prevalence** — extreme imbalance limits precision at any practical recall

### Model Limitations

1. **Entity-level transport loss** — user-disjoint AUC-ROC 0.787 vs 0.939 (Phase 14)
2. **Graph features weak standalone** — AUC-PR 0.014 (graph-only); useful only combined
3. **Feature shift small but present** — top PSI 0.026, top KS 0.047 (Phase 15)
4. **Gain ranking seed-sensitive** — Kendall tau min 0.143 (Phase 4)

### Operational Limitations

1. **Tail-window degradation** — W8 tail FAIL (Phase 13); risk documented
2. **Alert fatigue** — one user (HBO0413) accounts for 15% of CAL alerts
3. **Capacity projections unverified** — 10K-user projection is INFERENCE, not OBSERVED
4. **Conformal guarantee conditional** — assumes within-class exchangeability

### Methodological Limitations

1. **Chronological split only** — no random or spatial splits justified
2. **Single dataset** — CERT r4.2 only; no cross-dataset validation
3. **No online learning** — static model, no adaptation to concept drift
4. **Explanations are attributions, not causes** — top-3 rank churn ≈ 95%
5. **Algorithm scope limited to LightGBM** — Random Forest, XGBoost, CatBoost,
   and Logistic Regression were proposed but never trained or evaluated; no
   comparison artifacts exist. A broader algorithmic comparison is an open
   direction for future authorized work.

---

## E6. Future Work (HYPOTHESIS — Not Demonstrated)

### Requires New Authorization

| Direction | Rationale | Estimated Effort |
|---|---|---|
| Production deployment | Decision table ready; dashboard integration needed | 1-2 weeks |
| Larger dataset validation | CERT r4.2 is small; generalization unknown | External data required |
| Online threshold adaptation | Tail-window degradation could be mitigated | New authorized phase |
| Graph feature engineering | Current graph features are weak; deeper network analysis | New authorized phase |
| Multi-model ensemble | Behavioral + graph could be separate models with late fusion | New authorized phase |
| Alert fatigue mitigation | Per-user de-duplication, priority scoring | Operational, not ML |

### Out of Scope (Per Project Principles)

- GNNs, transformers, embeddings
- Real-time streaming architecture
- Cross-organization transfer learning
- Adversarial robustness testing

---

## E7. Evidence Classification

All claims in this document are classified per ISM Principles §12:

| Label | Count | Examples |
|---|---|---|
| OBSERVED | ~80% | All metrics, thresholds, test results, artifact hashes |
| INFERENCE | ~15% | "Graph features earn their place", "ranking failures not cold start" |
| HYPOTHESIS | ~5% | Capacity projections, tail-window cause, future work directions |
| LITERATURE RESULT | 0% | None referenced |

---

*Document generated from authoritative phase artifacts. No scientific experiments were conducted or modified during consolidation.*
