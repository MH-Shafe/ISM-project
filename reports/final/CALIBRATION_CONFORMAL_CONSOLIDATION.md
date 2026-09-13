# ISM Project — Calibration and Conformal Consolidation

Consolidated record of the two uncertainty/coverage layers evaluated and
accepted/rejected for the frozen insider-threat detection system.

**Last updated**: 2026-09-12
**Frozen classifier**: `lgbm-graph-v1` (LightGBM 4.6.0, 12 features, seed 42, best_iteration 186)
**Frozen threshold**: `0.9186015432508062` (max-F1 on CAL, Phase 9)

---

## B1. Conformal Confidence Layer (ACCEPTED — Phase 10)

### Method

Mondrian (label-conditional) split conformal prediction on frozen LightGBM
scores. Calibration on CAL block only (n1 = 323 positives, n0 = 58,677
negatives). Nonconformity scores: positive = 1 − score, negative = score.
Tie-inclusive p-values.

### Frozen Parameters

| Parameter | Value | Source |
|---|---|---|
| alpha | 0.05 | Phase 10 selection |
| t0 (negative threshold) | 0.4634739481800199 | CAL quantile, k0 = 2933 |
| t1 (positive threshold) | 0.0046536002164601275 | CAL quantile, k1 = 16 |
| n1 (CAL positives) | 323 | CAL block |
| n0 (CAL negatives) | 58,677 | CAL block |
| Conformal sets | {0}, {1}, {0,1}, {} | Mondrian scheme |

### TEST Evaluation (OBSERVED)

| Metric | Value | Source |
|---|---|---|
| Positive coverage | 1.000 (30/30) | `phase10_experiment.json` |
| Negative coverage | 0.988 | `phase10_experiment.json` |
| Marginal coverage | 0.988 | `phase10_experiment.json` |
| Wilson 90% LB (pos) | 0.917 | `phase10_experiment.json` |
| Monitor band rows (TEST) | 532 | `phase10_experiment.json` |
| Monitor band precision (TEST) | 0.015 (~25× prevalence) | Phase 10 report |
| All 49 alerts conformal set | {1} | Phase 10 report |
| Fit time | 0.85 ms | Phase 10 report |
| Inference time | 54 ms | Phase 10 report |

### CAL Evaluation (OBSERVED)

| Metric | Value |
|---|---|
| Positive coverage | 0.9536 (308/323) |
| Negative coverage | 0.9865 |
| Marginal coverage | 0.9863 |
| Monitor band rows | 765 |
| Monitor band precision | 0.061 (~11× prevalence) |

### Decision: ACCEPT (D1–D5 Rule)

- **D1**: Positive coverage meets guarantee (1.000 ≥ 0.95) — PASS
- **D2**: Negative coverage meets guarantee (0.988 ≥ 0.95) — PASS
- **D3**: Monitor band provides useful triage (precision 0.015 ≫ prevalence 0.0038) — PASS
- **D4**: All alerts confidently in {1} — PASS
- **D5**: No empty sets on TEST — PASS

**Verdict**: ACCEPT as diagnostic overlay. Policy unchanged. Does not alter
frozen scores, thresholds, or alert assignments.

### Risks

- Class-1 set {1} included for ~91% of rows (t1 = 0.0047) — rarely decisive
- Guarantees conditional on within-class exchangeability (no distribution-shift guarantee)
- Monitor band precision degrades CAL → TEST (0.061 → 0.015) — expected under temporal shift

### Source Artifact

`reports/artifacts/phase10_experiment.json`

---

## B2. Probability Calibration (RESEARCH-ONLY — Phase 17, Verdict FAIL)

### Method

Three pre-registered calibration arms fitted on CAL block (108 user-disjoint
users, 379 positives), evaluated on TEST block (108 users, 476 positives).
No training, no threshold hunting. Mechanical verdict: AUC-ROC must be
invariant within 1e-9.

### Arms

| Arm | Method | Brier Improvement | ECE | Verdict |
|---|---|---|---|---|
| B | Platt scaling (logistic) | +92.25% (0.00796 vs raw 0.1027) | 0.0026 | FAIL |
| C | Binning (isotonic-like) | +91.56% (0.00867) | 0.0019 | FAIL |
| D | ECDF (percentile map) | −216% (0.3247) | N/A | FAIL |

### Why FAIL

All arms failed the pre-registered mechanical verdict because AUC-ROC is not
invariant under tie creation:
- Arm C: ΔAUC-ROC = −0.0725 (exceeds 1e-9 bound)
- Arm D: ΔAUC-ROC = −3.27e-6 (exceeds 1e-9 bound)
- Arm B: Operating point recovers Phase 14 secondary behavior exactly
  (191 alerts, F1 0.4438, precision 0.7749, threshold 0.9837865316173778)
  but ranking is not invariant

### Operating Point Comparison

| Metric | Raw (frozen) | Platt (Arm B) | Binning (Arm C) | ECDF (Arm D) |
|---|---|---|---|---|
| Brier | 0.1027 | 0.00796 | 0.00867 | 0.3247 |
| Alerts | 49 | 191 | 21,127 | 49 |
| Precision | 0.286 | 0.775 | 0.018 | 0.286 |
| F1 | 0.354 | 0.444 | 0.035 | 0.354 |
| AUC-ROC | 0.93916 | 0.93916* | 0.86666 | 0.93916* |

*Arms B and D preserve AUC-ROC numerically but violate the pre-registered
invariant bound under tie analysis.

### Production Interpretation

**Calibration does NOT fix ranking-level misses.** The two missed users
(JJM0203, WDD0366) remain zero-alert under every conservative alert set.
Arms B/D cover 10/15 users; Arm C covers 15/15 only by alerting 39% of all
rows (21,127 alerts — operationally unusable).

### Decision: RESEARCH-ONLY — No Phase 18 calibration work without authorization (spec §25)

### Source Artifact

`reports/artifacts/phase17_experiment.json`

---

## B3. Production Interpretation

### What Is Used in Production

| Layer | Status | Role | Changes Frozen Outputs? |
|---|---|---|---|
| Conformal confidence (Phase 10) | ACCEPTED | Diagnostic overlay; adds conformal_set, p0, p1, confidence columns | No |
| Probability calibration (Phase 17) | RESEARCH-ONLY (FAIL) | None | No |
| Explainability (Phase 11) | ACCEPTED | Reporting layer; adds top_3_reasons, contribution columns | No |

### Layer Interaction

The conformal layer operates on frozen scores independently of the alert
policy. It provides:
1. **Confidence quantification**: high-confidence (0.95) for ALERT/NON-ALERT;
   ambiguous (0.90) for BORDERLINE/MONITOR
2. **Triage signal**: monitor band (score ∈ [0.4635, 0.9186)) identifies
   ~1,000 user-days per year for watchlist review
3. **Honest uncertainty**: acknowledges the 30-positive TEST limitation
   through coverage guarantees

### Key Limitation

The conformal guarantee is conditional on within-class exchangeability. Under
temporal distribution shift (documented in Phase 13, 15), coverage may degrade.
The Wilson lower bound (0.917) provides a conservative estimate of the true
positive coverage under this caveat.

---

*Document generated from authoritative phase artifacts. No scientific experiments were conducted or modified during consolidation.*
