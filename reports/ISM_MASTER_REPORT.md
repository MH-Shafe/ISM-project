# ISM Project — Master Report

Single authoritative cumulative report for the CERT r4.2 Insider Threat
Detection project. Handoff document for external AI review: complete history,
current verified state, frozen configuration, decisions, rejections, risks.

---

## 0. Document Status

| Item | Value |
|---|---|
| Report path | `reports/ISM_MASTER_REPORT.md` |
| Purpose | One-file project handoff for independent GPT review, planning, debugging, and research discussion |
| Last updated | 2026-09-14 |
| Latest completed phase | Final Project Consolidation (all documentation complete) |
| Project status | Final consolidation complete — all ablation evidence, calibration/conformal, explainability/decision, reproducibility, and conclusions documented; master report updated; consistency audit passed; final archive created. Model development STOPPED. |
| Current production candidate | `lgbm-graph-v1` (frozen, Phase 7) |
| Current alert policy | `frozen_max_f1`, threshold `0.9186015432508062` (frozen, Phase 9) |
| Dataset | CERT r4.2 (Kaggle, read-only) |
| Analytical unit | user × day |
| Update policy | Master report is updated at the END of every completed phase/run per Section 25; detailed phase reports remain the authoritative records and are never modified |

Evidence classification labels used throughout: **OBSERVED** (measured,
verified), **INFERENCE** (reasoned from evidence), **HYPOTHESIS** (untested
proposal), **LITERATURE RESULT** (external), **NOT VERIFIED** (absent from
project records).

---

## 4. Executive Summary

### Problem

Detect insider-threat activity (unauthorized/malicious use of legitimate
access) per **user × day** in CERT r4.2, as a production-style alerting
system: rank 47,000 test user-days and trigger a small, stable set of
alerts for analyst review. The system detects, it does not adjudicate:
alerts are a queue, not a verdict.

### Dataset (OBSERVED)

CERT r4.2 (7 log files + LDAP + psychometric + answer key), span
2010-01-02 .. 2011-05-17, 1,000 users, 501 days → **501,000 user-day rows**,
**1,892 malicious rows** (70 users, 0.38% prevalence). Validated: 0
unparseable timestamps, 0 duplicate rows across all logs; email cc/bcc nulls
expected. Labels from `answers/insiders.csv`, filtered `dataset == 4.2`
(70 incidents: scenario 1 ×30, scenario 2 ×30, scenario 3 ×10).

### Analytical unit

**user × day** — one row = one user on one day. User-level aggregation is
diagnostic only (Phase 9), never a model input or policy unit.

### Pipeline

```
CERT logs
→ validation (read-only on /kaggle/input)
→ leakage-safe user-day aggregation (day-local; 28-day strictly-past window)
→ 8 behavioral features (frozen Phase 4)
→ 4 graph/trust features (built Phase 5, merged Phase 6, frozen Phase 7)
→ LightGBM (lgbm-graph-v1)
→ alert policy (frozen_max_f1 threshold, Phase 9)
→ conformal confidence overlay (accepted diagnostic, Phase 10)
→ explainability layer (accepted reporting, Phase 11)
→ operational envelope evaluated (Phase 12: capacity, de-duplication,
  rolling threshold; policy unchanged)
→ temporal stability evaluated (Phase 13: pre-registered CAL verdict FAIL on
  the 3-day tail window; system unchanged; risk documented)
→ user-level holdout evaluated (Phase 14: evaluation-only auxiliary model
  on user-disjoint blocks; coverage 13/15, AUC-ROC 0.786 vs 0.939; frozen
  system unchanged)
→ degradation diagnosed (Phase 15: descriptive decomposition — feature
  shift small (|d| ≤ 0.179, PSI ≤ 0.026), graph features intrinsically
  weak in both cohorts, alert explosion = score-calibration failure on
  unseen benign rows (79.8% of FPs within 0.10 of the threshold), the 2
  user misses = ranking failures far below the threshold; frozen system
  unchanged)
→ end-to-end CONFIRM evaluated (Phase 19: one-time authorized evaluation;
  verdict FAIL — adaptive risk A0 vs Family C, delta AUC-ROC -0.0281
  < -0.01 threshold; frozen system unchanged; adaptive risk rejected
  for production)
→ final system integration (Phase 20: deterministic decision engine;
  frozen components integrated into dashboard-ready output;
  16-column schema with ML risk, risk level, confidence, trust
  diagnostics, explanations, alert flag, recommended action;
  39/39 tests pass; Adaptive Risk excluded from production path)
```

*(See [Figure 1](figures/fig1_system_architecture.png) for the visual
pipeline diagram.)*

Component status: implemented + frozen — aggregation, features, model,
threshold, explainability; accepted overlays — conformal, explainability;
evaluated with no change — operational envelope (Phase 12: no policy beat
frozen P0 under the recorded R1–R4 rule), temporal stability (Phase 13:
pre-registered CAL verdict FAIL on the 3-day tail window W8 — documented
risk, system unchanged), user-level holdout (Phase 14: evaluation-only —
auxiliary model on user-disjoint blocks; 13/15 unseen users detected,
ranking degraded AUC-ROC 0.786 vs 0.939; system unchanged);
diagnosed with no change — unseen-user degradation (Phase 15: descriptive
decomposition; all candidate causes ranked; system unchanged);
evaluated with no change — end-to-end CONFIRM (Phase 19: one-time
authorized evaluation; verdict FAIL — adaptive risk A0 vs Family C;
delta AUC-ROC -0.0281 < -0.01 threshold; adaptive risk rejected for
production; frozen system unchanged);
integrated — final system integration (Phase 20: deterministic decision
engine; frozen components integrated into dashboard-ready output;
Adaptive Risk excluded from production path; no ML change);
rejected — adaptive risk (Phase 8, confirmed Phase 19), department graph feature (Phase 7),
capacity/percentile alert policies (Phase 9), counterfactual explanations
(Phase 11), graph-only model (Phase 6); out of scope — GNNs, transformers,
embeddings, dashboards, production deployment.

### Current model

`lgbm-graph-v1` — LightGBM 4.6.0, **12 features** (8 behavioral + 4 graph),
seed 42, best_iteration 186, frozen threshold `0.9186015432508062`.
(OBSERVED — `src/experiments/phase7.py` FROZEN_CONFIG + Phase 7 freeze record.)

### Current performance (TEST once, 47,000 rows, 30 positives; OBSERVED)

| Metric | Value |
|---|---|
| AUC-ROC | 0.939157 (bootstrap CI [0.900, 0.975]) |
| AUC-PR | 0.267776 (bootstrap CI [0.140, 0.460]) |
| Precision@10 / @30 / Recall@50 | 0.600 / 0.367 / 0.467 |
| Policy (t = 0.91860) | precision 0.286 / recall 0.467 / F1 0.354 / MCC 0.365 / balanced acc 0.733 |
| Alerts | 49 (1.04/day over 47 days) |
| Scenario recall | scenario 2: 0.464 (28 rows), scenario 3: 0.500 (2 rows) |

### Current decision

The frozen system stands as specified: `lgbm-graph-v1` + frozen max-F1
threshold + user × day unit. Conformal layer and explainability layer are
accepted as analysis/reporting overlays that do not change any frozen output.
Why: every candidate change tested since the freeze (adaptive risk, policy
families) was measurably worse or degenerate; the end-to-end CONFIRM
evaluation (Phase 19) confirmed the adaptive risk approach FAILS for
production (delta AUC-ROC -0.0281 < -0.01); the frozen operating point is
reproducible and temporally stable across all full 7-day windows, with a
documented tail-window degradation (Phase 13 verdict FAIL — risk, not
repair); overlays add honest uncertainty and attribution at negligible cost.
(OBSERVED + INFERENCE.)

### Current limitations

- **Only 30 positive TEST user-days** (incl. 2 scenario-3 rows): TEST-level
  operating-point metrics have wide uncertainty; every top-k count is an
  integer multiple of 1/30. Decisions therefore rest on CALIBRATION evidence
  (323 positives) where possible.
- **Temporal evaluation**: chronological split only; user-level holdout done
  in Phase 14 (evaluation-only): coverage 13/15 unseen users, user-level
  AUC 0.965, but ranking degrades (AUC-ROC 0.786 vs 0.939; precision at the
  frozen threshold 0.202 vs 0.286) — entity-level transport loss measured;
  no unseen production window exists inside r4.2 (data ends 2011-05-17;
  strict replication infeasible — Phase 13 spec §0).
- **Unseen-user degradation diagnosed (Phase 15, OBSERVED/INFERENCE)**:
  feature shift TRAIN→CAL→TEST is small on all 12 frozen features (|d| ≤
  0.179, PSI ≤ 0.026, KS ≤ 0.047) — not a dominant driver; the two missed
  users (JJM0203, WDD0366) have long histories (170/289 active days before
  onset) and max scores 0.498/0.726 — ranking failures, not cold start or
  threshold-edge; graph features are weak discriminators in BOTH blocks
  (row AUC 0.50–0.69; rare counts ~99.9% zero) — intrinsic weakness, not
  unseen-specific; the 1,026-alert explosion is an absolute-score
  calibration failure — 79.8% of false positives lie within 0.10 above the
  frozen threshold (median margin 0.045) on unseen benign rows; detection
  is incident-length dependent (2-5 d: 3/3, 6-12 d: 3/4, 45-100 d: 7/8);
  84.9% of alerts concentrate on 8 long-incident users; no temporal
  position effect (early 0.875 vs late 0.857); unseen-user coverage
  improves (13/15 vs 5/5 chronological) at 12–21x alert cost.
- **Temporal stability (Phase 13, OBSERVED)**: pre-registered CAL verdict
  FAIL — the 3-day tail window W8 (2011-03-29..03-31) has score median
  0.04141 (28.6% beyond the 10% margin of B5) and precision 0.3333
  (marginal B4); all 8 full 7-day CAL windows pass every bound; the 15-day
  cross-check passes every checkable bound. TEST annex: 2/7 consistent
  (low-positive late windows → precision 0). System unchanged; risk
  documented (Section 21).
- **Uncertainty**: AUC-PR TEST CI [0.140, 0.460]; gain-based importance is
  seed-sensitive (Kendall tau min 0.143 in Phase 4).
- **Graph features**: weak standalone (graph-only AUC-PR 0.014); useful only
  combined with behavioral features.
- **Conformal**: class-1 set included for ~91% of rows (rarely decisive);
  validity assumes within-class exchangeability (no distribution-shift
  guarantee).
- **Explainability**: contributions are attributions of the frozen model,
  not causes; top-reason rank is fragile under unit perturbations (top-3
  churn ~95%, decision flips ≤3.5%); `http_activity_count` is bipolar
  (≈49% of |contribution|, signed alert mean ≈ 0).
- **Kaggle runtime**: ephemeral — runtime recycling wipes `/kaggle/working`;
  background jobs are not persistent; 30 s default / ~120 s client execution
  caps (Section 16).
- **Calibration transfer (Phase 17, OBSERVED)**: Platt scaling fitted on the
  108-user CAL block transfers calibration to unseen users (TEST Brier
  0.00796 vs raw 0.1027, ECE 0.0026, improvement 92.25%) and its operating
  point recovers the recorded Phase 14 secondary behavior exactly (191
  alerts, F1 0.4438, precision 0.7749); binning (arm C) transfers in
  Brier/ECE terms (91.56%) but its operating point is unusable (21,127
  alerts, precision 0.018); ECDF (arm D) is a percentile map, not a
  calibration (Brier −216%); mechanical verdict FAIL — the empirical
  rank-based AUC-ROC is not invariant under tie creation, so arms C (−0.0725)
  and D (−3.27e-6) violate the pre-registered 1e-9 bound even though all
  transforms are non-decreasing. Calibration does NOT fix ranking-level
  misses: JJM0203 and WDD0366 remain zero-alert under every conservative
  alert set (B/D 10/15 coverage; C covers 15/15 only by alerting 39% of all
  rows). Per spec §25, FAIL ⇒ no Phase 18 calibration work on this mechanism
  without separate authorization.
- **End-to-end CONFIRM (Phase 19, OBSERVED)**: one-time authorized evaluation
  of the adaptive risk approach (A0 vs Family C). Final verdict: FAIL.
  A0: ROC-AUC 0.7789, PR-AUC 0.2984, F1 0.3407, alerts 72, precision
  0.7500, recall 0.2204, MCC 0.4059, balanced accuracy 0.6101. Family C:
  ROC-AUC 0.7507, PR-AUC 0.2971, F1 0.3386, alerts 74, precision 0.7297,
  recall 0.2204, MCC 0.4003, balanced accuracy 0.6101. Deltas: ROC-AUC
  -0.0281, PR-AUC -0.0013, F1 -0.0021. Verdict FAIL triggered by delta
  ROC-AUC -0.0281 < -0.01. Adaptive risk rejected for production.

---

## 5. Phase Status Table

| Phase | Objective | Main Change | Model/Experiment | Key Result | Decision | Status | Detailed Report |
|---|---|---|---|---|---|---|---|
| Foundation | Integration, data validation, leakage analysis, user-day pipeline | Bridge OpenCode→Kaggle; validators, labels, aggregation, splits | none (infrastructure) | 501,000 × 9 table, 1,892 malicious rows; 12 leakage risks + controls; 31 tests | ACCEPT | Complete | [foundation_report.md](reports/foundation_report.md) |
| Baseline (Phase 1–2) | First model on the 6-feature table | `lgbm-baseline-v1` | LightGBM, 6 features | TEST AUC-ROC 0.9136 / AUC-PR 0.1140; max-F1 precision 0.80→0.22 (transfer loss) | Baseline recorded | Complete | [baseline_report.md](reports/baseline_report.md) |
| Phase 3 | Controlled behavioral feature expansion | +`sensitive_file_access_count`, +`unusual_access_count` | A (6f) vs B (8f) | B: AUC-PR +34% rel.; TEST max-F1 precision 0.216→0.409 | B recommended as new baseline | Complete | [phase3_report.md](reports/phase3_report.md) |
| Phase 4 | Ablation + robustness + finalize behavioral baseline | Registry order fix; ablation arms a/asen/aun/b | 4 arms, seed/window sweeps, bootstrap | b dominates all arms (F1 0.346, MCC 0.350, 22 alerts); byte-exact repro | **FREEZE `lgbm-baseline-v2`** | Complete | [phase4_report.md](reports/phase4_report.md) |
| Phase 5 | Leakage-safe graph construction | 5 experimental graph features | none (infrastructure) | 501,000 × 7 table, exact alignment, 63 tests, build 13 s | Infrastructure only; no accuracy claim | Complete | [phase5_graph_report.md](reports/phase5_graph_report.md) |
| Phase 6 | Controlled graph-vs-behavioral regression | A (8) vs B (5 graph) vs C (13) | 3 arms, identical protocol | C: AUC-PR 0.153→0.268, P@10 0.4→0.6; B weak standalone | Graph features complementary — earn their place | Complete | [phase6_graph_regression_report.md](reports/phase6_graph_regression_report.md) |
| Phase 7 | Robustness + freeze of the candidate | `department_file_type_mismatch_count` REJECTED | seed sweep (5), feature stability, freeze | C reproduces bit-for-bit; 12-feature candidate robust; TEST once: AUC-PR 0.2678 | **FREEZE CANDIDATE `lgbm-graph-v1`** | Complete | [phase7_freeze_report.md](reports/phase7_freeze_report.md) |
| Phase 8 | Adaptive risk (α·ML+β·Trust+γ·Behavior) | 3 arms: ML / equal-weight / learned | CAL-only weight learning | Learned weights collapse to [1,0,0]; equal-weight loses CAL AUC-PR 0.550→0.218 | **NO — rejected** | Complete | [phase8_adaptive_risk_report.md](reports/phase8_adaptive_risk_report.md) |
| Phase 9 | Production alert-prioritization + threshold stability | 13 candidate policies screened on CAL | S1/S2/S3 rule, bootstrap CI | `frozen_max_f1` selected (CAL F1 0.589, CI width 0.0642); TEST: 49 alerts, F1 0.354 | **FROZEN policy** | Complete | [phase9_alert_prioritization_report.md](reports/phase9_alert_prioritization_report.md) |
| Phase 10 | Conformal confidence layer | Mondrian split-conformal on frozen scores | alphas 0.01–0.20, CAL-only fit | TEST pos coverage 1.000 (LB 0.917), neg 0.988; monitor band 532 rows | **ACCEPT (diagnostic overlay)** | Complete | [phase10_conformal_report.md](reports/phase10_conformal_report.md) |
| Phase 11 | Explainability layer | LightGBM pred_contrib, margin space | native TreeSHAP + shap cross-check | 21,043 TEST rows explained; recon exact (3.6e-14); all alerts confident-positive | **ACCEPT (reporting layer); counterfactuals NOT SUPPORTED** | Complete | [phase11_explainability_report.md](reports/phase11_explainability_report.md) |
| Phase 12 | Operational envelope: capacity, de-duplication, rolling threshold | 4 candidate policies (P1–P4) vs frozen P0; pure functions of frozen scores | R1–R4 rule (CAL only), TEST once | no policy satisfies R1–R4 (dedup cuts recall; rolling loses recall by 0.004); 10k-user projection 41.4 alerts/day (INFERENCE); bug in `cal_windows` found & fixed | **RETAIN P0 (frozen policy unchanged)** | Complete | [phase12_operational_envelope_report.md](reports/phase12_operational_envelope_report.md) |
| Phase 13 | Temporal stability of the frozen system across all model-unseen days (CAL 59 + TEST 47 = 106 days) | pre-registered 7-day grid (16 windows) + B1–B6 bounds; CAL-only verdict; TEST annex | frozen outputs only; local run; deterministic (bit-identical) | **CAL verdict FAIL**: W8 tail (03-29..03-31) B5 hard (median 0.04141, 28.6% beyond margin) + B4 marginal (precision 0.3333); 8/8 full windows pass; 15-day cross-check clean; TEST annex 2/7 consistent | **descriptive risk documented; system unchanged** | Complete | [phase13_temporal_stability_report.md](reports/phase13_temporal_stability_report.md) |
| Phase 14 | User-level holdout generalization (evaluation-only) | auxiliary `lgbm-graph-v1-uhold` on user-disjoint blocks (pre-registered 70-user allocation); frozen threshold as reference point; user-block bootstrap | LightGBM, 12 frozen features, seed 42, ES on user-disjoint CAL, best_iter 214 | TEST once (54,108 rows, 476 pos): AUC-ROC 0.7865 (vs 0.9392), AUC-PR 0.3757; coverage 13/15 users (Wilson [0.688, 0.933]); P@10 1.0; 1,026 alerts at frozen threshold; user-level AUC 0.965; determinism bit-identical | **evaluation only; frozen system unchanged** | Complete | [phase14_user_holdout_report.md](reports/phase14_user_holdout_report.md) |
| Phase 15 | Unseen-user generalization diagnosis (descriptive) | decomposition of the Phase 14 degradation; fully local on frozen artifacts; no model, no threshold, no TEST rerun | none (statistics only; A: strata, B: feature shift, C: detected vs missed, D: graph, E: temporal, F: threshold, G: cohort comparison) | gates PASS (score repro max diff 5.551e-17; CAL AUC exact; bit-identical double run); feature shift small (|d| ≤ 0.179, PSI ≤ 0.026); 2 misses = ranking failures (max scores 0.498/0.726, 0 alerts in 501 d); graph features weak in both blocks (row AUC 0.50–0.69); 79.8% of 1,026 FPs within 0.10 of the threshold (score-calibration failure); coverage improves 13/15 vs 5/5 at 12–21x alert cost | **diagnosis only; frozen system unchanged** | Complete | [phase15_unseen_user_diagnosis_report.md](reports/phase15_unseen_user_diagnosis_report.md) |
| Phase 16 | Final packaging & reproducibility verification | README, requirements.txt + docs/environment.md, verification-only gate `scripts/verify_project.py`, artifact index; no ML | none (verification only) | gate PASS: 37 checks (structure, phase14/15 manifests md5+size 20/20, frozen model 186 trees/12 features/651,047 B, threshold exact, JSON/parquet load, master sections+links, secrets, tests 379 passed/4 skipped); fresh baseline md5s recorded for all artifacts; TEST never scored | **packaging complete; frozen system unchanged** | Complete | [phase16_final_packaging_report.md](reports/phase16_final_packaging_report.md) |
| Phase 17 | Calibration transfer on unseen users (research-only) | 3 pre-registered arms (B Platt, C binning, D ECDF) + A reference; CAL-fitted, TEST-once; no training, no threshold hunting | fits+bootstrap on CAL (108 users, 379 pos), TEST once (108 users, 476 pos) | gates 1/1b/2/3/4 all PASS; determinism 3× bit-identical; TEST Brier improvement: B +92.25% (0.00796), C +91.56% (0.00867), D −216% (0.3247); B recovers the Phase 14 secondary point exactly (191 alerts, F1 0.4438, precision 0.7749, threshold 0.9837865316173778); C unusable operating point (21,127 alerts); **mechanical verdict FAIL** — ΔAUC-ROC C −0.0725 / D −3.27e-6 (empirical AUC not tie-invariant), per pre-registered §14 | **evaluation only; frozen system unchanged; no Phase 18 calibration work without authorization (spec §25)** | Complete | [phase17_calibration_transfer_report.md](reports/phase17_calibration_transfer_report.md) |
| Phase 18 | Leakage-safe adaptive risk v2 (evaluation-only) | Redesigned adaptive-risk formulation (α·ML + β·Trust + γ·Context + δ·Behavioral); user-disjoint DEV-like evaluation; no chronological TEST access; deterministic repeated execution | DEV-like adaptive-risk optimization with deterministic repeated evaluation | deterministic science reproduced; optimized fusion collapsed toward ML-only (α ≥ 0.95); no adaptive-risk arm met the required improvement criteria; one run ~2.57 h; expensive context computation dominated runtime; earlier runner crash after science due to implementation/state issue; artifacts from one execution lost after Kaggle reset | **FAIL — no production change; frozen system unchanged** | Complete | (no standalone phase report; evaluation-only) |
| Phase 19 | End-to-end CONFIRM evaluation (one-time authorized) | Adaptive risk A0 vs Family C; DEV=588, CONFIRM=196, overlap=0; A0 threshold 0.989445, C threshold 0.358667 | adaptive risk on CONFIRM (98,196 rows, 245 positives) | A0: ROC-AUC 0.7789, PR-AUC 0.2984, F1 0.3407, alerts 72; C: ROC-AUC 0.7507, PR-AUC 0.2971, F1 0.3386, alerts 74; delta ROC-AUC -0.0281; delta PR-AUC -0.0013 | **FAIL (delta ROC-AUC -0.0281 < -0.01); adaptive risk rejected for production; frozen system unchanged** | Complete | [phase19 CONFIRM result](reports/artifacts/phase19_v1_1_2_confirm_result.json) |
| Phase 20 | Final system integration + full-scale operational validation | Complete decision table (501,000 user-days, 21 columns) materialized from frozen components; no ML change | `final_user_day_decisions.parquet` (2.76 MB); `scripts/phase20_materialize_decisions.py`; 54/54 tests | 501K rows, 1000 users; ALERT=3785, BORDERLINE=1484, MONITOR=178204, NON-ALERT=317527; score integrity 0 mismatches; determinism PASS (MD5 match); leakage audit PASS; no Adaptive Risk; runtime 180s | **COMPLETE — full decision table materialized; frozen system unchanged; ready for dashboard** | Complete | [phase20 report](reports/artifacts/phase20/PHASE20_REPORT.md) |

---

## 6. Phase-by-Phase Detailed Summary

### Foundation — Integration, Validation, Leakage Analysis, User-Day Pipeline

- **Objective**: stand up the OpenCode → Jupyter MCP → Kaggle execution bridge;
  validate CERT r4.2; establish leakage discipline; build the user-day table.
- **Inputs**: raw r4.2 logs (read-only) on Kaggle; no CERT data on the PC.
- **Method**: validators (pandas + duckdb streaming for the 14.5 GB http log),
  label mapper (`dataset == 4.2` only), user-day aggregation (grid 1000 × 501),
  chronological splitter, feature registry in `src/config.py`.
- **Key results (OBSERVED)**: all logs validated clean; 501,000 × 9 table;
  1,892 malicious rows / 70 users; split TRAIN 395,000/1,539 · CAL 59,000/323 ·
  TEST 47,000/30; 12 leakage risks documented with controls.
- **Resource cost (OBSERVED)**: http.csv pandas path OOM'd (~28.7 GB RSS) →
  duckdb streaming path adopted (14.5 GB handled).
- **Decision**: ACCEPT — foundation established.
- **Verification**: 31 tests passed on Kaggle; raw data never copied to PC.
- **Risks**: pandas path unsuitable for http.csv (duckdb is the production
  path); one label start date (`CDE1846`) unparseable → skipped, documented.
- **Detailed record**: [foundation_report.md](reports/foundation_report.md)

### Baseline — lgbm-baseline-v1 (Phase 1–2)

- **Objective**: first LightGBM model on the 6-feature table under the
  chronological protocol.
- **Method**: LightGBM 4.6.0, lr 0.03, 31 leaves, min_data_in_leaf 100,
  feature_fraction 0.8, bagging 0.8/freq 1, seed 42, up to 3000 rounds, early
  stopping on CALIBRATION auc (patience 100); scale_pos_weight 255.66 from
  TRAIN; max-F1 and 50%-precision thresholds on CALIBRATION; TEST once.
- **Key results (OBSERVED)**: best_iteration 182 (4.36 s); TEST AUC-ROC
  0.9136, AUC-PR 0.1140; P@10 0.30; max-F1 point: CAL precision 0.799 →
  TEST 0.216 (F1 0.239, 37 alerts); 50%-precision point unreachable on TEST
  (0 alerts); scenario recall 2: 0.25, 3: 0.50.
- **Resource cost (OBSERVED)**: train 4.36 s; model 635,440 B (182 trees).
- **Decision**: baseline recorded (lr 0.05 config REJECTED — collapsed to
  best_iteration 1 with degenerate 0.0/1.0 score saturation).
- **Why**: a usable reference point; transfer loss (precision 0.80→0.22)
  documented as the calibration-transportability problem.
- **Verification**: 40 tests; artifacts pulled and verified.
- **Risks**: max-F1 calibration point does not transfer to TEST; extreme
  imbalance (30 positives).
- **Detailed record**: [baseline_report.md](reports/baseline_report.md)

### Phase 3 — Controlled Behavioral Feature Expansion

- **Objective**: test two pending registry features under a controlled
  A/B design.
- **Inputs**: rebuilt 8-column table (501,000 × 11), verified.
- **Method**: A (6 features) vs B (8 features); identical protocol.
- **Feature definitions (declared a priori)**: `sensitive_file_access_count`
  = file rows/day matching declared magic-byte prefixes (OLE2 `D0-CF-11-E0-A1-B1-1A-E1`,
  PDF `25-50-44-46-2D`, ZIP `50-4B-03-04`); `unusual_access_count` = logon
  events/day on a PC unused by the user in the trailing 28 days (strictly
  past; first-ever use counts).
- **Key results (OBSERVED)**: B beats A: AUC-ROC +0.0217 (0.91360→0.93534),
  AUC-PR +0.0390 (+34% rel.), TEST max-F1 precision 0.216→0.409 at recall
  0.267→0.30, 37→22 alerts; arm A reproduces the baseline record exactly.
- **Decision**: B (8 features) is the recommended new baseline.
- **Verification**: 43 tests; predictions independently recomputed and
  matched records exactly.
- **Detailed record**: [phase3_report.md](reports/phase3_report.md)

### Phase 4 — Ablation, Robustness, Finalization (FREEZE lgbm-baseline-v2)

- **Objective**: isolate each new feature's contribution, check robustness,
  and freeze the behavioral baseline.
- **Method**: arms a (6f), asen (+sensitive), aun (+unusual), b (+both);
  reproducibility hard gates; seed sweep {5,7,13,21}; window sweep
  W ∈ {14,21,28,35,42}; bootstrap CIs (evaluation-only).
- **Key results (OBSERVED)**: b dominates all arms — AUC-PR 0.15304
  (+34.2% rel. vs a), MCC 0.350, F1 0.346, P@10 0.40, R@50 0.333, scenario
  recall 0.286/0.500, 22 alerts; gates byte-exact (a and b reproduce prior
  records at 1e-9); CAL AUC-ROC seed spread ±0.3% but gain-rank seed-sensitive
  (τ min 0.143); window sweep flat (0.9323–0.9326).
- **Resource cost (OBSERVED)**: b trains 4.13 s, predicts TEST 1.06 s, model
  559,479 B (160 trees) — best model is also cheapest.
- **Decision**: **FREEZE `lgbm-baseline-v2`** (8 features, seed 42, W=28
  frozen); regression discipline: any future model must beat this record.
- **Verification**: 48 tests; reproducibility gates; artifacts pulled.
- **Risks**: wide TEST CIs (AUC-PR 0.059–0.318); gain ranking not over-read.
- **Detailed record**: [phase4_report.md](reports/phase4_report.md)

### Phase 5 — Leakage-Safe Graph Construction

- **Objective**: build experimental graph/trust features, leakage-safe, no
  model.
- **Inputs**: logon.csv, file.csv, 18 monthly LDAP snapshots.
- **Method**: graph model (user/pc/file/content-type/department); temporal
  rule: day-d term from day-d edges; every historical term from first-use
  day **strictly before** d; department from latest LDAP snapshot strictly
  before month(d).
- **Features (5)**: `device_consistency_score` (Jaccard day-PCs vs
  strictly-past PCs), `rare_device_usage_count`, `file_type_consistency_score`
  (Jaccard over content types), `rare_file_type_access_count`,
  `department_file_type_mismatch_count`.
- **Key results (OBSERVED)**: 501,000 × 7 table, exactly aligned (0
  graph-only / 0 base-only keys); every filename in r4.2 globally unique →
  filename-based features degenerate; device features richest (consistency
  65.7% non-zero), file-type sparse, department mismatch fires on 49
  user-days.
- **Resource cost (OBSERVED)**: build 12.99 s; total script 17.9 s; parquet
  556,021 B.
- **Decision**: infrastructure only; experimental; no accuracy claim.
- **Verification**: 63 tests (15 new graph tests, incl. future-invariance,
  strictly-past rule, hand-computed Jaccard); cross-env pandas versions.
- **Risks**: coarse content taxonomy; month-granular LDAP; department
  feature is the only cross-user signal — do not over-read.
- **Detailed record**: [phase5_graph_report.md](reports/phase5_graph_report.md)

### Phase 6 — Controlled Graph-vs-Behavioral Regression

- **Objective**: do the 5 graph features add predictive value beyond the
  frozen 8-feature baseline?
- **Arms**: A (8 behavioral; control/repro gate), B (5 graph; standalone),
  C (13; combined). Identical protocol, no per-arm tuning.
- **Key results (OBSERVED)**: A reproduces the Phase 4 record exactly
  (incl. confusion matrix). B collapses (best_iteration 3, AUC-PR 0.0143,
  FPR 0.86%, 417 alerts). C: AUC-ROC 0.9391566 (+0.0038), AUC-PR 0.2677761
  (+0.1147), P@10 0.6, R@50 0.467, scenario-2 recall 0.464 (+0.179), 49
  alerts. Paired bootstrap: C−A AUC-ROC CI [+0.0020, +0.0056] (100% of
  resamples > 0), AUC-PR CI [+0.022, +0.218] (99.2%); B−A entirely negative.
- **Resource cost (OBSERVED)**: C train 4.76 s, predict 1.21 s, model
  651,085 B (+16% vs A); merged table 501,000 × 16.
- **Decision**: graph features earn their place — complementary, not
  standalone; improvement chiefly from `device_consistency_score`.
- **Verification**: 82 tests on Kaggle (78 local + 4 env skips).
- **Risks**: 30 TEST positives; arm B is a stump (not a tuned graph model);
  per-arm thresholds mix model quality with threshold behavior.
- **Detailed record**: [phase6_graph_regression_report.md](reports/phase6_graph_regression_report.md)

### Phase 7 — Robustness and Freeze of lgbm-graph-v1

- **Objective**: calibration-only robustness evidence and freeze of the
  Phase 6 Arm C candidate.
- **Method**: seed sweep {7,42,123,2024,2026}; feature-stability ranks;
  deterministic REJECT rule for the department feature; freeze record with
  rerun gate; TEST once afterwards.
- **Key results (OBSERVED)**: seed-robust (CAL AUC-ROC 0.888 ± 0.016, CV
  1.8%; AUC-PR 0.526 ± 0.039); rank τ 0.79–1.00; `department_file_type_mismatch_count`
  REJECTED (zero gain 5/5 seeds, 49/454,000 non-zero rows, single-feature CAL
  AUC-ROC 0.500, 12-vs-13 removal delta exactly 0.0); freeze rerun reproduced
  bit-for-bit.
- **Final TEST (once)**: AUC-ROC 0.9391565538286849, AUC-PR 0.2677761042396373,
  P@10 0.600, R@50 0.467, F1 0.354 / MCC 0.365, 49 alerts, scenario recall
  0.464 / 0.500 — bit-identical to Phase 6 Arm C.
- **Resource cost (OBSERVED)**: train 4.64 s, predict 1.29 s, model 651,047 B.
- **Decision**: **FREEZE CANDIDATE `lgbm-graph-v1`** (12 features, seed 42,
  threshold 0.9186015432508062).
- **Verification**: 95 passed / 4 skipped locally; MD5-verified sources;
  artifacts reloaded; rerun gate passed.
- **Risks**: 30 TEST positives; CAL alert-rate CV 22% across seeds; graph
  tables must be regenerated by the Phase 5 pipeline in production.
- **Detailed record**: [phase7_freeze_report.md](reports/phase7_freeze_report.md)

### Phase 8 — Adaptive Risk (REJECTED)

- **Objective**: test `Final Risk = α·ML + β·Trust + γ·Behavior` beyond the
  frozen prediction.
- **Method**: components from registry features only (trust = 1 − equal-weight
  graph-consistency score; behavior = normalized anomalous-work indicators);
  TRAIN-only empirical-CDF normalization; exhaustive simplex weight grid
  (step 0.1, 66 combos), CALIBRATION selection; arms A (1,0,0), B
  (1/3,1/3,1/3), C (learned); TEST once.
- **Key results (OBSERVED)**: learned weights collapse to **ML-only [1,0,0]**
  (arm C ≡ arm A by construction, reproducing Phase 7 bit-for-bit); equal-weight
  control degrades CAL AUC-PR 0.550→0.218 and doubles alerts (244→548); TEST:
  B worse on every metric (AUC-PR −0.201, P@10 −0.5, 256 alerts, F1 0.105);
  trust_risk redundant with ml_risk (Spearman +0.629) and negatively correlated
  with the target (−0.048); behavior_risk standalone CAL AUC-PR 0.008.
- **Resource cost (OBSERVED)**: component build+scoring 0.197 s (CAL block) —
  free but useless.
- **Decision**: **NO — adaptive risk rejected; `lgbm-graph-v1` remains the
  production candidate.** Only the studied component definitions are rejected.
- **Verification**: 116 passed / 4 skipped; arm A bit-equal to Phase 7 file.
- **Risks**: 30 TEST positives (decision rests on CAL evidence); TRAIN
  in-sample weight fitting could only favor ML (selection on CAL protects the
  conclusion).
- **Detailed record**: [phase8_adaptive_risk_report.md](reports/phase8_adaptive_risk_report.md)

### Phase 9 — Alert Prioritization and Threshold Stability (FROZEN POLICY)

- **Objective**: convert the frozen model into a defensible production alert
  policy under analyst capacity constraints.
- **Method**: 13 candidates (2 thresholds + 6 daily-capacity + 5 global
  percentiles), declared a priori; deterministic selection rule on
  CALIBRATION: S1 threshold-identifiability (bootstrap CI width ≤ 0.2), S2
  window alert-rate stability ([0.5, 2.0]× global per 2-week window), S3
  window coverage (recall > 0 in every window); highest CALIBRATION F1 wins.
- **Key results (OBSERVED)**: `frozen_max_f1` (t = 0.9186015432508062):
  CI width 0.0642 (median exactly 0.9186), CAL F1 0.589, alerts 244 (4.1/day),
  precision 0.684; windows stable in rate (5.27→2.79/day within bounds) and
  precision (±0.09); recall drifts 0.575→0.429 (documented, not corrected);
  244 alerts across 35 users (Gini 0.607; HBO0413 alone 37 alerts). Rejected:
  50p threshold (1 CAL alert — degenerate), tight percentiles (zero-alert
  window; S2/S3 fail), high capacities (precision < 0.2).
- **TEST (once)**: 49 alerts (1.04/day), precision 0.286, recall 0.467, F1
  0.354, MCC 0.365, balanced acc 0.733 — identical to the Phase 7 max-F1
  point (all 16 comparison deltas 0.000000).
- **Resource cost (OBSERVED)**: frozen scores CAL+TEST 0.287 s; 13 policies
  over 59,000 rows 0.105 s; test-stage application < 0.001 s.
- **Decision**: **FROZEN — `frozen_max_f1` is the production alert policy.**
- **Verification**: 146 passed / 4 skipped; TEST scores bit-equal to Phase 7;
  rerun gate; MD5-verified sources.
- **Risks**: window recall drift; repeat-alert fatigue (one user 15% of
  alerts) is an operational lever outside the frozen policy; percentile
  counts scale with daily volume.
- **Detailed record**: [phase9_alert_prioritization_report.md](reports/phase9_alert_prioritization_report.md)

### Phase 10 — Conformal Confidence Layer (ACCEPTED)

- **Objective**: honest, leakage-safe uncertainty for frozen scores without
  touching the model or policy.
- **Method**: Mondrian (label-conditional) split conformal on frozen scores;
  calibration = CALIBRATION block only (n1 = 323, n0 = 58,677); tie-inclusive
  p-values; alphas 0.01/0.05/0.10/0.20; sets {0},{1},{0,1},empty.
- **Key results (OBSERVED)**: CAL coverage meets guarantees at every alpha
  (0.05: pos 0.9536, neg 0.9865); TEST pos coverage 1.000 (30/30; Wilson 90%
  LB 0.917), neg 0.988, marginal 0.988; all 49 alerts confident-positive {1}
  at every alpha; **monitor band** (t0 = 0.4635 ≤ score < 0.9186): CAL 765
  rows precision 0.061 (≈11× prevalence), TEST 532 rows precision 0.015
  (≈25× prevalence); score tie at the 0.46347395 leaf (~10% of each class);
  fit 0.85 ms, inference 54 ms.
- **Decision**: **ACCEPT** (D1–D5 rule; D3 via the monitor band — the
  uncertain-alert branch never fires). Diagnostic overlay only; policy
  unchanged.
- **Verification**: 33 Phase-10 tests (hand-computed p-values, tie mirrors,
  determinism, calibration isolation); 179 passed / 4 skipped; TEST once;
  bit-identical rerun.
- **Risks**: class-1 included for ~91% of rows (t1 = 0.0047) — rarely
  decisive; guarantees conditional on within-class exchangeability.
- **Detailed record**: [phase10_conformal_report.md](reports/phase10_conformal_report.md)

### Phase 11 — Explainability Layer (ACCEPTED)

- **Objective**: deterministic, label-free, leakage-safe explanations for
  frozen outputs, as the reporting format.
- **Method**: LightGBM native `pred_contrib` (path-dependent TreeSHAP) in
  margin space; `margin = bias + Σ contrib` (reconstruction identity, tol
  1e-6); shap TreeExplainer as validation cross-check only; deterministic
  non-causal reason templates (≤3 reasons); label-free selection policy
  (alerts / monitor t0≤s<thr / borderline within 0.05 / decile sample);
  counterfactuals explicitly NOT SUPPORTED.
- **Key results (OBSERVED)**: reconstruction exact (max err TRAIN 4.9e-14,
  CAL 4.8e-14, TEST 3.6e-14); shap cross-check max diff **0.0**; 21,043 TEST
  rows explained (49 alerts + 20,494 monitor + 500 sample); all 49 alerts
  confident-positive {1}; `usb_connection_count` top-1 positive reason in
  36/49 TEST alerts (top-1 in 50% of the 244 CAL alerts); `device_consistency_score`
  top-3 in 72% of CAL alerts; graph features 20.5% of alert-row |contribution|;
  `http_activity_count` bipolar (48.8% of global |contribution|, signed alert
  mean ≈ 0); decision flips ≤ 3.5% per feature under unit perturbations while
  top-3 rank churn ≈ 95% (reason rank is fragile, decisions are not);
  monitor band set distribution {1}: 532, {0,1}: 19,962.
- **Resource cost (OBSERVED)**: contributions TRAIN+CAL 199.8 s, stability
  171.4 s, shap 0.61 s, TEST 20.7 s; peak RSS 1,266 MB; explanations parquet
  4.66 MB.
- **Decision**: **ACCEPT (G1–G9); counterfactuals NOT SUPPORTED** under the
  frozen-output contract.
- **Verification**: 35 Phase-11 tests; full suite 214 passed / 4 skipped
  (final gate); bit-identical rerun; 14 artifacts pulled and re-verified.
- **Risks**: attributions ≠ causes; rank fragility; bipolar http summary
  misuse; explanations inherit model ranking quality.
- **Detailed record**: [phase11_explainability_report.md](reports/phase11_explainability_report.md)

### Phase 12 — Operational Envelope: Capacity, De-duplication, Rolling Threshold (EVALUATED; POLICY UNCHANGED)

- **Objective**: measure the operational envelope of the frozen alerting
  system — alert capacity at larger scale, per-user alert fatigue, threshold
  stability — using **pure functions of the frozen scores**; adopt a better
  operational policy only if CALIBRATION evidence supports it. Never
  retrains/retunes; all frozen artifacts are inputs.
- **Method**: 4 candidate policies vs frozen P0 (P1/P2/P3 = frozen threshold
  + 1/3/7-day per-user de-duplication; P4 = rolling calibration threshold,
  window chosen by CALIBRATION-only F1 rule); recorded R1–R4 selection rule
  (recall retention ≥ 0.9×, alert reduction ≥ 10%, Gini not worse than
  +0.05, zero-alert windows not more than P0); capacity projections are
  arithmetic on OBSERVED CALIBRATION behavior only; TEST evaluated once.
- **Key results (OBSERVED)**: de-duplication beyond 1 day destroys recall
  (3d: 148 alerts / recall 0.248; 7d: 100 alerts / recall 0.139 vs P0 244 /
  0.517); B1 (1d) ≡ P0 by construction (user × day has no same-day
  duplicates); rolling threshold never beats frozen (best wd=7 aggregate F1
  0.567 vs P0 0.589; P4 misses R1 by 0.004); **eligible = []; RETAIN P0**.
  Capacity (INFERENCE): 10,000 users → 2,440 alerts/59 d (41.4/day) under
  uniform scaling; 244 (4.1/day) only under an untested clean-population
  HYPOTHESIS. TEST (once, P0): 49 alerts, precision 0.2857, recall 0.4667,
  F1 0.3544 — identical to the Phase 7/9 record, as expected (policy
  unchanged).
- **Bug found & fixed (OBSERVED)**: `cal_windows` chunked rows instead of
  unique days (missing `.unique()`): wd=7 produced 8,429 windows → O(windows²)
  concat + 8,429 threshold fits (hours of CPU); fixed to unique-day chunking
  (9/5/2 windows; rolling analysis 1.06 s); regression test
  `test_cal_windows_chunks_unique_days_not_rows` added; phase12.py md5
  `a801ed182828134ab997bcd10bff803f` local=kernel. Also corrected
  `test_frozen_threshold_matches_phase9_record` (was asserting a nonexistent
  key and had always been skipped locally).
- **Resource cost (OBSERVED)**: full phase < 20 s (rolling analysis 1.06 s);
  model never retrained; 106,000 rows scored; artifacts ~129 KB total.
- **Decision**: **RETAIN P0 — the frozen Phase 9 policy stands; no
  operational change adopted.**
- **Verification**: 28 Phase-12 tests (incl. hand-computed masks, per-user
  independence, no-future-reference rolling, unique-day windows); 4 artifact
  reloadability tests; full suite **277 passed / 4 skipped**; all 10
  artifacts md5-verified against the kernel manifest; TEST-once honored.
- **Risks**: capacity projections NOT VERIFIED beyond 1,000 users (S2/S3 are
  HYPOTHESIS); rolling evidence thin (wd=30 rests on 1 defined window);
  de-duplication/rolling are alert-management levers, not detection
  improvements; operational adoption (if ever) requires a new authorized
  phase.
- **Detailed record**: [phase12_operational_envelope_report.md](reports/phase12_operational_envelope_report.md)

### Phase 13 — Temporal Stability of the Frozen System (EVALUATED; SYSTEM UNCHANGED; VERDICT FAIL — DESCRIPTIVE RISK)

- **Objective**: measure temporal stability of the frozen system (model +
  threshold) over **all model-unseen labeled days** (CAL 59 + TEST 47 = 106
  days, 353 positives) at a pre-registered 7-day grid (16 windows, W0–W15,
  spec §5) against pre-registered bounds B1–B6 (spec §25); verdict from CAL
  windows only; TEST as descriptive annex. Never retrains/retunes/regenerates
  predictions; frozen artifacts are the only inputs.
- **Feasibility (established at planning)**: a strict new-unseen-window
  replication is NOT FEASIBLE inside r4.2 (data ends 2011-05-17; spec §0);
  design D (frozen-output temporal stability across CAL+TEST) was adopted and
  approved verbatim with recorded amendments (16-window correction in spec §5;
  6→7 TEST windows, 15→16 total).
- **Safety gates (all PASS, OBSERVED)**: 59/47/106 unique days, disjoint;
  every day/row in exactly one window; no window crosses the CAL/TEST
  boundary; computed windows == pre-registered table verbatim; TEST never
  regenerated; verdict function CAL-only (test-enforced).
- **Key results (OBSERVED)**: CAL 7-day windows — 8/8 full windows pass all
  six bounds (alerts/day 2.857–5.429 vs band [2.068, 8.271]; precision
  0.500–0.806 vs floor 0.342; recall 0.400–0.592 vs floor 0.25); **W8 tail
  (03-29..03-31, 3 days, 12 positives) violates B5 hard** (score median
  0.04141 vs band [0.05798, 0.23191]; deviation 28.6% > 10% margin) **and B4
  marginally** (precision 0.3333 vs floor 0.3422; deviation within margin).
  Per the pre-registered rule (any violation beyond 10% margin → FAIL):
  **CAL verdict = FAIL**. 15-day cross-check (Phase 9 record): all checkable
  bounds pass (B5/B6 NOT VERIFIED on that grid — no medians in the record).
  TEST annex (descriptive re-read of the frozen parquet, never re-scored):
  2/7 windows consistent; 5/7 inconsistent driven by late low-positive
  windows (W11/W13/W14: 0–1 positives, precision 0; W13 zero recall; W15
  rate 0.400/day below the band); block totals asserted equal to the
  published record (49 alerts, AUC-ROC 0.9391566, AUC-PR 0.2677761).
- **Interpretation (INFERENCE)**: degradation concentrates at span tails
  (CAL W8, late TEST windows) where the score bulk shifts down (W8 median
  0.0414) and positives thin out; the effect is diluted below detection at
  15-day granularity. HYPOTHESIS (not demonstrated): tail behavior is a
  data/behavioral regime effect, not a model defect.
- **Resource cost (OBSERVED)**: fully local run — 42 s total (analysis ×2
  for the determinism gate, bit-identical); artifacts ≈ 60 KB; **no Kaggle
  execution required** (approved default path: frozen parquet + records).
- **Decision**: **descriptive risk-register result — the frozen system
  stands unchanged**; CAUTION/FAIL would never have changed the system by
  design (spec §25). Risk entries R13-1..R13-5 recorded (§21).
- **Verification**: 35 Phase-13 tests (pre-registered table verbatim, unique-
  day partition, day gates, per-bound margins incl. exact W8 cases, verdict
  aggregation + CAL-only enforcement, Wilson 90% CI hand-computed, annex
  isolation, artifact reloadability, record reconciliation, determinism);
  full suite **312 passed / 4 skipped**; determinism gate bit_identical=True;
  frozen input md5s match Phase 12 manifest (parquet
  `efca7668…`, rolling `d3455748…`); phase9 record recorded fresh (no prior
  baseline); TEST-once honored.
- **Risks**: tail-window degradation (R13-1); false-alert bursts in
  low-positive windows (R13-2); no labeled data after 2011-05-17 → stability
  beyond the observed span unmeasured (R13-4); CAL per-row checks NOT
  VERIFIED (record-based only; optional CAL-only scoring pass documented in
  spec §12 for a future authorized run).
- **Detailed record**: [phase13_temporal_stability_report.md](reports/phase13_temporal_stability_report.md)

### Phase 14 — User-Level Holdout Generalization (EVALUATED; SYSTEM UNCHANGED)

- **Objective**: measure how the frozen feature set and training protocol
  generalize to **users never seen in training** (the previously documented
  entity-overlap limitation, master §21), evaluation-only.
- **Method**: pre-registered deterministic user allocation (spec §6,
  asserted verbatim): malicious users sorted by (onset, id) → round-robin
  period 14 (positions 0–7 TRAIN, 8–10 CAL, 11–13 TEST); benign users
  period 10 (0–7 TRAIN, 8 CAL, 9 TEST). Blocks: TRAIN 784 users / 392,784
  rows / 1,037 pos / 40 mal; CAL 108 / 54,108 / 379 / 15; TEST 108 /
  54,108 / 476 / 15; pairwise disjoint. Auxiliary `lgbm-graph-v1-uhold`:
  12 frozen features, seed 42, scale_pos_weight 377.77 (TRAIN), early
  stopping on user-disjoint CAL AUC, best_iteration 214. Operating points:
  frozen threshold `0.9186015432508062` (immutable reference) + CAL
  best-F1 secondary (descriptive only). Interpretation vs pre-registered
  bands (§20) against the frozen chronological record (cited, never
  recomputed); authoritative chronological TEST not re-evaluated.
- **Key results (OBSERVED)**: user-disjoint CAL AUC-ROC 0.7355 / AUC-PR
  0.3344. Auxiliary TEST (once): AUC-ROC **0.786513** (−0.153 vs 0.939157 →
  band "materially worse"), AUC-PR 0.375740 (+40.3% rel. → band
  "inconclusive"; +direction, cohort-conditional, not comparable), P@10 1.0
  / P@30 1.0 / P@50 1.0 / P@100 0.95; at frozen threshold 1,026 alerts,
  precision 0.2018, recall 0.4349, F1 0.2756, MCC 0.2873; **coverage 13/15
  malicious users** (Wilson 90% [0.688, 0.933]) → band "high user-level
  coverage"; user-level AUC 0.9649; zero-detection users JJM0203, WDD0366;
  alerts precede onset for most detected users (delay median −229 d);
  all 15 malicious TEST users are in the low-activity tercile of the TEST
  block (OBSERVED concentration); scenario recall 0.267 / 0.445 / 0.500
  (1/2/3) = 207 TP; user-block bootstrap AUC-ROC CI [0.666, 0.901], AUC-PR
  CI [0.236, 0.505] (1000 resamples).
- **Discrepancy resolved (documented)**: `phase14_scenario.json` field
  `n_users` actually holds stratum **row** counts (2004/4008/1503 =
  4/8/3 users × 501); source-traced to a per-row `mask.sum()` in
  `_scenario_counts`/`_scenario_recall` (phase14.py); `n_malicious_rows`
  and all recalls are correct; mathematically consistent with
  `user_diagnostics` strata; artifact NOT regenerated (TEST-once);
  conflict record §26 #7.
- **Recovery (OBSERVED)**: runtime recycled before the phase → sources
  re-pushed gz+b64 (20 files, md5-verified), derived tables rebuilt
  (user_day 20 s, graph 20 s, merged md5 `9a3b1885…9def` == frozen record);
  3 runner bugs fixed (numpy-2 `is False` gate; `alignment['both']` →
  `common_keys`; artifacts dir creation); PC crash after pull → artifacts
  re-verified, kernel alive/idle, no rerun, TEST-once preserved.
- **Resource cost (OBSERVED)**: protocol ×2 = 66.1 s (run1 32.96 s, run2
  32.67 s; bit-identical); artifacts ≈ 921 KB (model 750,807 B).
- **Decision**: **evaluation-only — the frozen system stands unchanged;
  no production decision** (per spec §20; STOP, no Phase 15).
- **Verification**: 34/34 Phase 14 tests (allocation rules + pre-registered
  table, real-parquet block/label gates, CDE1846 absence, feature
  isolation, operating points, user-block bootstrap, interpretation bands,
  determinism, artifact reloadability + manifest md5s); full suite **346
  passed / 4 skipped**; determinism bit_identical=True; TEST-once honored.
- **Risks**: entity-level transport loss measured (ranking −0.153 AUC-ROC,
  precision 0.202 at the frozen threshold); cohort-conditional metrics not
  comparable to the chronological record; small low-activity TEST cohort.
- **Detailed record**: [phase14_user_holdout_report.md](reports/phase14_user_holdout_report.md)

### Phase 15 — Unseen-User Generalization Diagnosis (DIAGNOSED; SYSTEM UNCHANGED)

- **Objective**: decompose the Phase 14 degradation into candidate causes —
  feature shift, sparse-user behavior, graph-feature weakness,
  threshold/operating-point effects — without touching the production
  system or re-running anything.
- **Method**: fully local, IO-free deterministic module
  (`src/experiments/phase15.py`) over verified frozen artifacts; the
  authoritative chronological TEST was never read or scored; the Phase 14
  allocation preserved verbatim; CAL block scores reproduced locally from
  the frozen auxiliary model under two gates. Pre-registered analyses
  (spec `docs/phase15_specification.md`): A activity strata, B feature
  shift (user-level Cohen's d / KS / row-level PSI, bins fitted on TRAIN
  only), C detected vs missed, D graph diagnosis (single-feature row AUC
  per block, consistency stability), E temporal, F threshold diagnostic
  (descriptive), G cohort comparison with pre-registered attribution.
- **Gates (OBSERVED)**: input md5s all match; Gate 1 local TEST-score
  reproduction max diff 5.551e-17 ≤ 1e-12 (54,081/54,108 bit-exact;
  lightgbm 4.7.0 local vs 4.6.0 kernel); Gate 2 CAL AUC reproduced exactly
  (0.735512422820, diff 0.0); double-run bit_identical=True; runtime 6.0 s.
- **Key results (OBSERVED)**: (A) detection is incident-length dependent —
  2–5 d: 3/3, 6–12 d: 3/4, 45–100 d: 7/8; the 2 missed users hold 56
  missed malicious days and 0 alerts in 501 days; 84.9% of alerts
  concentrate on the 8 long-incident users. (B) feature shift is small on
  all 12 frozen features (top: file_type_consistency_score d +0.179,
  file_access_count d +0.164, sensitive_file_access_count d +0.160; all
  PSI ≤ 0.026, KS ≤ 0.047) — not a dominant driver (INFERENCE). (C) missed
  users JJM0203/WDD0366: 48/8 malicious days, 170/289 active days before
  onset, max scores 0.4981/0.7264 (vs detected median 0.9992) — systematic
  ranking failures; cold start excluded (INFERENCE). (D) graph features
  discriminate poorly in BOTH blocks (row AUC 0.50–0.69; rare counts
  ~99.9% zero) — intrinsic feature-set weakness, not unseen-specific
  (INFERENCE); weak negative consistency-vs-activity link
  (file_type_consistency r = −0.225, p = 0.019). (E) no temporal position
  effect (early 0.875 vs late 0.857; 2010 H2 0.917, 2011 0.667, n=3);
  detection delay median −229 d (alerts precede onset; alerts also fire on
  benign days). (F) TEST malicious-day scores are bimodal (p50 0.696, p75
  0.994); 13/56 missed positives within 0.05 of the threshold (edge mass
  small; the misses are far below); **79.8% (819/1,026) of false positives
  lie within 0.10 above the frozen threshold** (median FP margin 0.045) —
  absolute-score calibration inflation on unseen benign rows (INFERENCE);
  CAL block reproduces the same pattern (580 alerts, 135 TP). (G)
  coverage improves with unseen users (13/15 vs 5/5 chronological) at
  12–21x alert cost; cross-phase comparisons mix model instances and time
  scopes — the unseen-user effect cannot be numerically isolated across
  phases (INFERENCE).
- **Attribution (pre-registered labels)**: OBSERVED — both unseen cohorts
  degrade vs chronological; all 15 unseen malicious users are
  low-activity-tercile; INFERENCE — the drop is entity-associated (blocks
  share the timeline), the alert explosion is calibration transport
  failure, the 2 misses are ranking failures, graph weakness is intrinsic;
  HYPOTHESIS — dominant mechanism is calibration inflation of unseen
  benign rows plus ranking collapse of a minority of low-activity
  malicious users.
- **Decision**: **diagnosis only — the frozen system stands unchanged**;
  risks R15-1..R15-4 recorded (§21); no new threshold, no model change.
- **Verification**: 22/22 Phase 15 unit tests + 11 artifact/reloadability
  tests; full suite **379 passed / 4 skipped**; manifest md5-verified;
  frozen-input immutability test PASS.
- **Detailed record**: [phase15_unseen_user_diagnosis_report.md](reports/phase15_unseen_user_diagnosis_report.md)

### Phase 16 — Final Packaging & Reproducibility Verification

- **Scope (authorized 2026-08-17, verification-only)**: make the frozen
  system understandable, reproducible from documented inputs, independently
  verifiable, auditable without rerunning TEST, and packaged with clear
  environment/dependency information. No ML of any kind; no Kaggle restore
  required (fully local).
- **Deliverables (OBSERVED, all created/verified this phase)**:
  `README.md` (frozen system, pipeline, split, features, overlays, phases
  12–15 results, risks, TEST-once, commands, research-not-production
  disclaimer); `requirements.txt` (PC verification env, versions OBSERVED
  2026-08-17); `docs/environment.md` (PC env OBSERVED + Kaggle env from the
  policy doc and the Phase 7 freeze record; Kaggle package versions beyond
  lightgbm 4.6.0 explicitly unrecorded, not fabricated);
  `scripts/verify_project.py` (verification-only gate);
  `reports/artifacts/ARTIFACT_INDEX.md` (every artifact: phase, purpose,
  format, status, hash source, regeneration policy);
  `reports/artifacts/final_verification_report.json` (gate output);
  `reports/phase16_final_packaging_report.md`; master report update;
  decision-log #22; experiment-index + evaluation README notes; journal
  entry.
- **Gate result (OBSERVED)**: `python scripts/verify_project.py` → 37
  checks, all PASS, exit code 0. Structure (packaging/docs/source/tests/
  reports/knowledge/artifacts ≥ 90 files); manifests md5+size verified
  (phase14 10/10, phase15 10/10); frozen-input md5s match the Phase 15
  gate3 record; frozen model integrity parse-only (num_trees 186, feature
  names = frozen 12, size 651,047 B); `src/config.py` registry yields the
  frozen 12; threshold exact (0.9186015432508062) in phase9_freeze.json
  and the Phase 7 record; all artifact JSON parses; all parquet loads (no
  scoring); master sections 0/4–30 + links; README links; Phase 16 STOP
  recorded; seven credential patterns → no matches; pytest 379 passed / 4
  skipped; fresh baseline md5s for every artifact recorded in the report.
- **Non-actions (verified)**: no training/fitting/scoring/threshold/
  calibration; chronological TEST never re-scored (prediction parquet
  loaded for parse + hash only); no artifact altered; no historical report
  modified; the Phase 15 report stale "Runtime 9.5 s" line and the stale
  knowledge notes were left untouched and recorded (§26 #8).
- **Decision**: **packaging complete — frozen system final and
  verification-gated**; acceptance gates G1–G17 all PASS.
- **Detailed record**: [phase16_final_packaging_report.md](reports/phase16_final_packaging_report.md)

### Phase 17 — Calibration Transfer on Unseen Users (EVALUATED; SYSTEM UNCHANGED; VERDICT FAIL — PRE-REGISTERED MECHANICAL)

- **Scope (approved `docs/phase17_specification.md`, research-only)**: test
  whether a calibration fitted on the user-disjoint CAL block (108 users,
  379 positives) transfers to unseen users (user-disjoint TEST, 108 users,
  476 positives) under three pre-registered arms — B Platt
  (logistic-regression scale), C binning (20 equal-frequency bins, n<30
  merge, PAV), D ECDF (midpoint-rank percentile map) — vs the frozen
  reference A. No training, no threshold hunting (one pre-registered
  `best_f1_threshold` per arm, CAL-only), no user-conditioning, no answer-
  key contact; authoritative chronological TEST never read or re-scored.
- **Gates (all OBSERVED)**: g1 arm-A reproduction exact (AUC-ROC
  0.786513147174144, AUC-PR 0.3757395445913874, 1,026 alerts, precision
  0.2018, recall 0.4349, F1 0.2756, MCC 0.2873); g1b raw CAL best-F1
  threshold 0.9837865316173778 = recorded Phase 14 secondary threshold,
  B/D alert sets identical to raw (191 rows, 0 differing), C differs
  (20,936 rows — pre-registered OBSERVED); g2 rows 54,108 / positives 379 /
  580 ≥ frozen threshold / 135 TP / CAL AUC 0.7355124228200639 (diff 0.0);
  g3 phase14+phase15 manifests (20 files) all md5-exact; g4 151 baselines
  exact before AND after, with one documented project-owner correction —
  `final_verification_report.json`'s own self-baseline excluded
  (by-construction stale: `verify_project.py` hashes every artifact
  including its own output before writing the report; decision recorded in
  `phase17_experiment.json` gate 4 and §26 #9); determinism 3× bit-identical.
- **Results (OBSERVED, TEST once)**: B TEST Brier 0.007962 (improvement
  +92.25%), ECE 0.002588, operating F1 0.4438 / precision 0.7749 / 191
  alerts / 10-of-15 coverage (Wilson 90% [0.4836, 0.7988]); C Brier
  0.008669 (+91.56%), ECE 0.005172, but operating point unusable — 21,127
  alerts (39% of rows), precision 0.0182, coverage 15/15 only by alerting
  everything; D Brier 0.324673 (−216.0%), ECE 0.481091 — the ECDF is a
  percentile map, not a probability calibration. B and D operating
  behavior byte-identical (pre-registered monotone-invariance expectation,
  OBSERVED). Ranking verification: B exactly invariant (ΔAUC-ROC 0.0,
  ΔAUC-PR 0.0, P@10 1.0, R@100 0.19958); **C ΔAUC-ROC −0.07249, ΔAUC-PR
  −0.35919, P@10 = 0**; **D ΔAUC-ROC −3.271e-06** — cause (OBSERVED): the
  empirical (average-rank) AUC estimator is not invariant under tie
  creation (C: 3 distinct p̂ values on TEST; D: 1,731 of 4,174 distinct
  TEST scores tied via ECDF flattening across the 1,840 gaps of the CAL
  score distribution). The spec's pre-registered claim that arm C's
  AUC-ROC is "unchanged" is REFUTED by measurement (report §5; §26 #10).
- **Bootstrap (seed 42, n=1000, 90% CIs)**: user-block — B/D AUC-ROC CI
  [0.66645, 0.90150] (= recorded Phase 14 interval exactly, reproducibility
  anchor), B operating F1 mean 0.440 CI [0.296, 0.561], alerts mean 189;
  C AUC-ROC [0.64153, 0.78014]. Calibration-transfer (CAL users resampled,
  calibration REFIT inside each resample) — TEST Brier B 0.00805
  [0.00787, 0.00849], C 0.00868 [0.00866, 0.00873] (tight: transfer robust
  to fit variance), D 0.32652 [0.30545, 0.34686] (no transfer).
- **Zero-alert users (OBSERVED)**: A [JJM0203, WDD0366] (pre-registered
  expectation met — the Phase 15 ranking failures); B/D [JGT0221, JJM0203,
  JLM0364, TNM0961, WDD0366]; C none. Scenario recall B 0.200/0.320/0.167
  (all 15 TEST malicious users are in the low-activity tercile, as in
  Phase 14).
- **Cost (OBSERVED)**: 655.6 s total local; ≈218 s per pipeline run (spec
  §21 expectation < 5 min met); no GPU/Kaggle quota; bootstrap dominates.
  Env: Python 3.11.9, numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0,
  scipy 1.17.1.
- **Decision (mechanical, pre-registered §14)**: **FAIL** — criterion 3
  (|ΔAUC-ROC| ≤ 1e-9 every arm) violated by C and D; D also violates the
  1e-12 strictly-monotone bound. Spec §25: FAIL ⇒ **no Phase 18
  calibration work on this mechanism**; negative result recorded;
  alternative directions (feature work per R15-3, ranking interventions
  per R15-2) require separate fully-specified authorized phases.
- **Interpretation (evidence-labeled)**: calibration CAN transfer across
  user blocks (B, robust under refit variance); it cannot repair
  ranking-level failures (JJM0203/WDD0366 zero-alert under every
  conservative alert set); calibration quality alone is insufficient for
  deployment (C). NOT VERIFIED: production behavior of any calibrated
  operating point (chronological TEST off-limits forever; R17-5).
- **Verification**: `tests/test_phase17.py` 28 tests (unit transforms,
  PAV monotonicity, ECDF ties, hand-computed Brier/ECE, operating-point
  semantics, coverage/Wilson, small-panel bootstrap, verdict paths,
  full-pipeline determinism on exact-shape synthetic frames 54,108 rows,
  structural guards, artifact reloadability + manifest md5s + frozen-input
  immutability); full suite with `CERT_WORKING=reports` **404 passed /
  7 skipped** (4 Kaggle-only real-data + 3 artifact-dependent phase17
  tests that all pass after execution); `scripts/verify_project.py` 37/37
  PASS pre-execution; re-run after bookkeeping.
- **Detailed record**: [phase17_calibration_transfer_report.md](reports/phase17_calibration_transfer_report.md)

---

## 7. Complete Model Evolution

```
6 behavioral features (lgbm-baseline-v1)
  ↓ Phase 3 — +2 behavioral features (controlled A/B, TEST once)
8 behavioral features (lgbm-baseline-v2, Phase 4 FREEZE)
  ↓ Phase 5 — graph feature table built (experimental, no model)
  ↓ Phase 6 — controlled regression: C (13f) beats A; graph-only weak
13-feature arm C
  ↓ Phase 7 — department feature REJECTED; freeze
12 features (lgbm-graph-v1, Phase 7 FREEZE — current candidate)
  ↓ Phase 8 — adaptive risk REJECTED (model unchanged)
  ↓ Phase 9 — alert policy FROZEN (threshold 0.9186015432508062)
  ↓ Phase 10 — conformal overlay ACCEPTED (model/policy unchanged)
  ↓ Phase 11 — explainability layer ACCEPTED (model/policy unchanged)
  ↓ Phase 12 — operational envelope EVALUATED (model/policy unchanged)
  ↓ Phase 13 — temporal stability EVALUATED (model/policy unchanged)
  ↓ Phase 14 — user-level holdout EVALUATED (auxiliary model, evaluation-only)
```

| Model | Features | Config (LightGBM 4.6.0) | Seed | Best iter | Threshold (CAL) | TEST AUC-ROC / AUC-PR | Decision | Status |
|---|---|---|---|---|---|---|---|---|
| `lgbm-baseline-v1` | 6 (behavioral) | lr 0.03, 31 leaves, mdi 100, ff 0.8, bag 0.8/f1, up to 3000 rds, ES CAL auc p100 | 42 | 182 | 0.9506 (max-F1) | 0.9136 / 0.1140 | baseline recorded | superseded |
| `lgbm-baseline-v2` | 8 (behavioral) | same | 42 | 160 | 0.96485 (max-F1) | 0.93534 / 0.15304 | FREEZE (Phase 4) | frozen reference |
| graph-only (Arm B) | 5 (graph) | same | 42 | 3 (stump) | 0.87160 | 0.78635 / 0.01431 | REJECT standalone (Phase 6) | experimental only |
| Arm C | 13 (8+5) | same | 42 | 186 | 0.91860 | 0.93916 / 0.26778 | candidate (Phase 6) | superseded by v1 |
| **`lgbm-graph-v1`** | **12 (8+4)** | **same; scale_pos_weight 255.66 (TRAIN)** | **42** | **186** | **0.9186015432508062 (frozen_max_f1)** | **0.93916 / 0.26778** | **FREEZE CANDIDATE (Phase 7) — current** | **frozen/current** |
| adaptive risk B (equal-weight) | 12 → 3 components | α=β=γ=1/3 | 42 | — | 0.80070 | 0.90977 / 0.06720 | REJECT (Phase 8) | not adopted |
| adaptive risk C (learned) | = ML-only | [1,0,0] ≡ arm A | 42 | — | 0.918602 | 0.93916 / 0.26778 | REJECT (Phase 8) | degenerate ≡ model |
| `lgbm-graph-v1-uhold` (auxiliary, Phase 14) | 12 (frozen) | same; scale_pos_weight 377.77 (user-disjoint TRAIN) | 42 | 214 (user-disjoint CAL ES) | frozen 0.91860 (reference) | 0.78651 / 0.37574 (unseen-user TEST, 476 pos) | EVALUATION ONLY (Phase 14); never a production candidate | auxiliary, not adopted |

Auxiliary models are evaluation instruments only and never enter the
production candidate path.

All TEST metrics above evaluated exactly once per finalized experiment on the
47,000-row TEST (30 positives).

---

## 8. Current Frozen System Specification

### Dataset

- **CERT r4.2** (Kaggle, read-only): `/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2`
- Raw data handling: never copied to the PC; validation/pipeline read-only.
- Date coverage: 2010-01-02 .. 2011-05-17 (501 days).

### Unit

- **user × day** — 501,000 rows (1,000 users × 501 days); 1,892 malicious
  rows; 70 malicious users; prevalence 0.38%.

### Split (chronological, frozen in `src/config.py` SPLIT_DEFAULTS)

| Split | Date range | Rows | Positives |
|---|---|---|---|
| TRAIN | ≤ 2011-01-31 | 395,000 | 1,539 |
| CALIBRATION | 2011-02-01 .. 2011-03-31 | 59,000 | 323 |
| TEST | ≥ 2011-04-01 | 47,000 | 30 |

*(See [Figure 2](figures/fig2_chronological_split.png) for the split
timeline and [Figure 3](figures/fig3_class_distribution.png) for class
distribution.)*

### Features

Exact frozen 12-feature list (verified against `src/experiments/phase7.py`
FROZEN_CONFIG and the freeze record):

1. `login_count` (behavioral)
2. `after_hours_login_count` (behavioral)
3. `usb_connection_count` (behavioral)
4. `file_access_count` (behavioral)
5. `sensitive_file_access_count` (behavioral, magic-byte rule)
6. `http_activity_count` (behavioral)
7. `unique_device_count` (behavioral)
8. `unusual_access_count` (behavioral, 28-day strictly-past window)
9. `device_consistency_score` (graph)
10. `rare_device_usage_count` (graph)
11. `file_type_consistency_score` (graph)
12. `rare_file_type_access_count` (graph)

**`department_file_type_mismatch_count` was REJECTED** (Phase 7: zero gain in
5/5 seeds, 49/454,000 non-zero rows, single-feature CAL AUC-ROC 0.500,
removal delta exactly 0.0). It remains documented in `config.GRAPH_FEATURES`
and `REJECTED_GRAPH_FEATURES`.

### Model

`lgbm-graph-v1` — LightGBM 4.6.0; objective binary, metric auc; lr 0.03;
num_leaves 31; min_data_in_leaf 100; feature_fraction 0.8; bagging 0.8,
bagging_freq 1; seed 42; scale_pos_weight 255.66 (TRAIN only); early
stopping on CALIBRATION auc (patience 100, up to 3000 rounds);
best_iteration **186**. Feature version: v2 (8 behavioral + 4 graph;
department feature REJECTED in Phase 7). Source of truth:
`src/experiments/phase7.py` (`FROZEN_CONFIG`); model file
`reports/artifacts/phase7_model_lgbm-graph-v1.txt` (651,047 B).

### Alert policy (frozen, Phase 9)

`frozen_max_f1` — threshold **0.9186015432508062**; alert every user-day with
score ≥ t. Selected by the documented CALIBRATION-only rule (S1/S2/S3 +
highest CAL F1 0.589). CALIBRATION 50%-precision threshold
(0.9962470269668189) is recorded as a candidate, not adopted. Source of
truth: `src/experiments/phase9.py`, `reports/artifacts/phase9_freeze.json`.

### Conformal layer (accepted diagnostic overlay, Phase 10)

Mondrian (label-conditional) split conformal on the **frozen scores**; fit on
CALIBRATION only (n1 323 / n0 58,677); alpha 0.05 (also 0.01/0.10/0.20
recorded); tie-inclusive p-values; sets {0}/{1}/{0,1}/empty; inclusion
thresholds t1 = 0.0047, t0 = 0.4634739481800199. Used for: honest
class-conditional coverage reporting and the **monitor band** (t0 ≤ score <
threshold) as a watch-list. It never modifies scores, the model, or the
policy.

### Explainability (accepted reporting layer, Phase 11)

LightGBM native `pred_contrib` (path-dependent TreeSHAP) in margin space;
`margin = bias + Σ contrib` exact (tol 1e-6; observed err ~1e-14); shap
TreeExplainer cross-check (validation only); deterministic non-causal reason
templates (≤3 reasons per row); risk labels ALERT / BORDERLINE / MONITOR /
NON-ALERT derived from frozen scores and t0; counterfactual explanations
NOT SUPPORTED. Selection policy is label-free (scores only).

### Adaptive risk

**Phase 8 decision: NO — rejected.** `lgbm-graph-v1` remains the production
candidate; no adaptive layer exists in the system.

---

## 9. Final Performance Table

TEST only, evaluated once per finalized experiment (47,000 rows, 30
positives). OBSERVED values from phase reports and experiment records.

| Model | Features | AUC-ROC | AUC-PR | P@10 | R@50 | F1 | MCC | Alerts | Decision |
|---|---|---|---|---|---|---|---|---|---|
| `lgbm-baseline-v1` | 6 | 0.91360 | 0.11401 | 0.300 | 0.300 | 0.239 | 0.240 | 37 | superseded |
| `lgbm-baseline-v2` | 8 | 0.93534 | 0.15304 | 0.400 | 0.333 | 0.346 | 0.350 | 22 | frozen reference |
| graph-only (Arm B) | 5 | 0.78635 | 0.01431 | 0.000 | 0.033 | 0.054 | 0.105 | 417 | rejected standalone |
| Arm C | 13 | 0.93916 | 0.26778 | 0.600 | 0.467 | 0.354 | 0.365 | 49 | superseded by v1 |
| **`lgbm-graph-v1`** ⭐ | **12** | **0.93916** | **0.26778** | **0.600** | **0.467** | **0.354** | **0.365** | **49** | **current production candidate** |
| adaptive risk B | 3-component | 0.90977 | 0.06720 | 0.100 | 0.067 | 0.105 | 0.170 | 256 | rejected |
| adaptive risk C | ≡ ML | 0.93916 | 0.26778 | 0.600 | 0.467 | 0.354 | 0.365 | 49 | rejected (degenerate) |
| CONFIRM A0 (Phase 19) | 12 | 0.77886 | 0.29842 | — | — | 0.3407 | 0.4059 | 72 | FAIL (Phase 19) |
| CONFIRM Family C (Phase 19) | 12 | 0.75074 | 0.29711 | — | — | 0.3386 | 0.4003 | 74 | FAIL (Phase 19) |

F1/MCC/alerts are at each system's own CALIBRATION max-F1 threshold (or the
frozen Phase 9 policy for the current system). Precisely:
`lgbm-graph-v1` = Phase 9 policy F1 0.354 / MCC 0.365 / 49 alerts, precision
0.286, recall 0.467, balanced accuracy 0.733.

Phase 19 CONFIRM results are on the CONFIRM partition (196 users, 98,196 rows,
245 positives) using adaptive risk scoring. A0 uses threshold 0.989445
(source: DEV_OOF); Family C uses threshold 0.358667 (source: DEV_OOF).
Additional CONFIRM metrics: A0 precision 0.7500, recall 0.2204, FPR 0.000184,
FNR 0.7796; C precision 0.7297, recall 0.2204, FPR 0.000204, FNR 0.7796.
(OBSERVED — `phase19_v1_1_2_confirm_result.json`.)

---

## 10. Feature Evolution and Evidence

| Feature | Type | Introduced | Status | Evidence | Current Use |
|---|---|---|---|---|---|
| `login_count` | behavioral | foundation | frozen | Phase 3/4 ablation | in lgbm-graph-v1 |
| `after_hours_login_count` | behavioral | foundation | frozen | Phase 3/4 ablation | in lgbm-graph-v1 |
| `usb_connection_count` | behavioral | foundation | frozen | dominant gain feature; top-1 reason in 36/49 TEST alerts | in lgbm-graph-v1 |
| `file_access_count` | behavioral | foundation | frozen | Phase 3/4 ablation | in lgbm-graph-v1 |
| `http_activity_count` | behavioral | foundation | frozen | ~49% of |contribution|; bipolar (Phase 11) | in lgbm-graph-v1 |
| `unique_device_count` | behavioral | foundation | frozen | Phase 3/4 ablation | in lgbm-graph-v1 |
| `sensitive_file_access_count` | behavioral | Phase 3 | frozen | magic-byte a-priori rule; precision-side gain (Phase 4 asen) | in lgbm-graph-v1 |
| `unusual_access_count` | behavioral | Phase 3 | frozen | 28-day strictly-past window (W=28 frozen); ranking-side gain; robust 14–42 days | in lgbm-graph-v1 |
| `device_consistency_score` | graph | Phase 5 | frozen | strongest graph feature; top-3 in 72% of CAL alerts | in lgbm-graph-v1 |
| `rare_device_usage_count` | graph | Phase 5 | frozen | small consistent gain (Phase 7) | in lgbm-graph-v1 |
| `file_type_consistency_score` | graph | Phase 5 | frozen | strongest graph single-feature signal (CAL AUC-ROC 0.707) | in lgbm-graph-v1 |
| `rare_file_type_access_count` | graph | Phase 5 | frozen | small consistent gain | in lgbm-graph-v1 |
| `department_file_type_mismatch_count` | graph | Phase 5 | **rejected** | zero gain 5/5 seeds; 0.011% non-zero; single-feature AUC-ROC 0.500 | none |
| `file_access_consistency_score` | graph (filename) | — | rejected (degenerate) | all r4.2 filenames globally unique → identically 0 | none |
| `rare_file_access` | graph (filename) | — | rejected (degenerate) | = `file_access_count` by construction | none |
| `url_consistency_score` | graph | — | deferred | 14.5 GB http scan for unproven signal; cost-based deferral | none |

Temporal/leakage rules: all behavioral features day-local except
`unusual_access_count` (strictly past [day-28, day-1]); all graph features use
strictly-past first-use history; department from LDAP snapshot strictly
before month(d); no future information, enforced by tests.

---

## 11. Graph/Trust Findings

- **Graph relationships (OBSERVED, Phase 5)**: nodes user/pc/file/content-type/
  department; edges logon(user,pc,day), file(user,file,day), filetype(user,type,day),
  department(user,dept,month). Most device edges are repeat edges (41.9% of
  (user,pc) pairs on 2+ days); **every filename globally unique** → content
  type is the only defensible file unit.
- **Leakage safeguards**: strictly-past first-use rule; same-day edges only in
  the day term; LDAP strictly before month(d); future-invariance tests;
  exact key alignment with the behavioral table.
- **Graph-only performance (OBSERVED, Phase 6)**: weak standalone — AUC-ROC
  0.786, AUC-PR 0.014, collapsed to a stump (best_iteration 3), FPR 0.86%.
- **Combined performance (OBSERVED)**: +5 graph features over 8 behavioral:
  AUC-PR 0.153→0.268 (+0.115), P@10 0.4→0.6, R@50 +0.133, scenario-2 recall
  0.286→0.464; paired bootstrap CIs exclude zero. Cost: +16% model size,
  <1 s more train/inference.
- **Strongest graph features (OBSERVED)**: `device_consistency_score`
  (gain rank 3 overall in Phase 6; top-3 in 72% of CAL alerts; top-1 in 9/49
  TEST alerts; top-1 positive in 36/49 for `usb` overall); `file_type_consistency_score`
  strongest graph single-feature (CAL AUC-ROC 0.707).
- **Rejected graph features (OBSERVED)**: `department_file_type_mismatch_count`
  (zero gain 5/5 seeds, removal delta exactly 0.0); filename-based features
  (degenerate).
- **Adaptive trust-risk result (OBSERVED, Phase 8)**: NO — trust semantics
  (high consistency = low risk) contradicted by the data; `trust_risk`
  negatively correlated with the target (−0.048 on CAL); weights collapse to
  ML.

**Conclusion (OBSERVED + INFERENCE)**: graph features are **weak as a
standalone model** and **useful as complementary features** to behavioral
counts — this remains supported by the Phase 6/7 records.

---

## 12. Alerting and Threshold Findings

- **Frozen threshold**: `0.9186015432508062` (Phase 9 `frozen_max_f1`).
- **Selection**: CALIBRATION-only rule S1 (identifiability, bootstrap CI
  width 0.0642 ≤ 0.2) + S2 (window alert-rate stability within [0.5, 2.0]×)
  + S3 (window coverage > 0) → highest CAL F1 0.589 (best alternatives:
  pct_0.0050 0.570, daily_top_5 0.511).
- **Calibration evidence (OBSERVED)**: 244 alerts (0.414% rate), precision
  0.684, recall 0.517; window rates 5.27→2.79/day; window precision flat
  ±0.09; window recall drifts 0.575→0.429 (documented, not corrected).
- **Alert volume**: ~4/day on 1,000 users (CAL); TEST 49 alerts (1.04/day).
- **TEST (once)**: precision 0.286 / recall 0.467 / F1 0.354 / MCC 0.365 /
  balanced acc 0.733; scenario recall 0.464 / 0.500; identical to Phase 7
  max-F1 point.
- **Concentration (OBSERVED)**: 244 CAL alerts across 35 of 1,000 users
  (Gini 0.607); top-1%/5%/10% of alerted users → 15.2%/29.5%/50.8% of alerts;
  HBO0413 alone = 37 alerts (15%) — repeat-alert fatigue is an operational
  lever, outside the frozen policy. User-level aggregation is diagnostic only
  (user-level AUC up to 0.9977; P@5 users = 1.0).
- **Alternatives considered and not adopted (OBSERVED)**: 50p-precision
  threshold (1 CAL alert — degenerate); tight percentiles pct_0.0005/0.0010
  (zero-alert window, S2/S3 fail); high daily capacities (precision collapses
  < 0.2); pct_0.0050 was eligible and competitive (F1 0.570) but inferior to
  the frozen threshold on F1 and MCC.
- **Interpretation (INFERENCE)**: the threshold is well identified and
  temporally stable in volume, but it is **not claimed to be universally
  optimal**; it is the best eligible policy on CALIBRATION evidence with a
  documented trade (recall drift, precision 0.286 on TEST).
- **Operational envelope (Phase 12, OBSERVED)**: de-duplication beyond 1
  day never improves the operating point (3d/7d: recall 0.248/0.139 vs 0.517;
  both fail the R1 recall-retention rule); a rolling calibration threshold
  (best: 7-day windows, aggregate F1 0.567) never beats the frozen threshold
  (F1 0.589), and the combined best candidate (P4, rolling 7d) misses the
  recall-retention bar by 0.004; **no candidate satisfied the recorded
  R1–R4 rule → RETAIN frozen P0**. Alert capacity at 10,000 users is an
  INFERENCE projection (≈41.4 alerts/day), not a measurement (S2/S3 are
  HYPOTHESIS).
- **Temporal stability (Phase 13, OBSERVED)**: pre-registered 7-day-grid
  verdict on CAL windows = **FAIL** — W8 tail (03-29..03-31) score median
  0.04141 (B5, 28.6% beyond the 10% margin) and precision 0.3333 (B4,
  marginal); every full 7-day window passes all bounds (alerts/day
  2.857–5.429; precision 0.500–0.806; recall 0.400–0.592); 15-day
  cross-check clean on every checkable bound. TEST annex (descriptive):
  2/7 consistent; late low-positive windows drive the inconsistency
  (precision 0). Recorded as a documented risk, not repaired (no adaptation
  allowed); the system stands unchanged.
- **User-level holdout (Phase 14, OBSERVED, evaluation-only)**: at the
  frozen threshold on 108 unseen users (476 positives): 1,026 alerts
  (0.019/user-day), precision 0.202, recall 0.435, F1 0.276, MCC 0.287;
  P@10/30/50 = 1.0, P@100 0.95; 13/15 malicious users detected (Wilson 90%
  [0.688, 0.933]); user-level AUC 0.965. Ranking degrades vs the
  chronological record (AUC-ROC 0.786 vs 0.939) — entity-level transport
  loss, now measured; no production decision (auxiliary cohort is not
  comparable to the chronological TEST by design).
- **Threshold diagnostic (Phase 15, OBSERVED, descriptive)**: unseen-user
  false positives hug the frozen threshold — median margin 0.045, max
  0.078; **79.8% (819/1,026) of TEST false positives lie within 0.10 above
  the threshold**; the missed-positive edge mass is small (13/56 within
  0.05) and the 2 missed users are far below it (0.498/0.726) — the alert
  explosion is an absolute-score calibration inflation on unseen benign
  rows, not a threshold-edge effect (INFERENCE); CAL block reproduces the
  pattern (580 alerts at the frozen threshold, 135 TP). No threshold was
  selected or recommended.

---

## 13. Conformal Confidence Findings

- **Method**: Mondrian (label-conditional) split conformal on frozen scores
  (`src/experiments/phase10.py`); tie-inclusive conservative p-values
  (`p_y(x) > alpha` inclusion); sets {0},{1},{0,1},empty.
- **Calibration set**: CALIBRATION only — n1 = 323, n0 = 58,677; alpha levels
  0.01 / 0.05 / 0.10 / 0.20.
- **Coverage (OBSERVED)**: TEST positive coverage 1.000 (30/30; Wilson 90% LB
  0.917), negative 0.988, marginal 0.988 at alpha 0.05 (and at every alpha).
  CAL coverage meets the 1−α guarantees.
- **Monitor band (OBSERVED)**: scores in (t0 = 0.4634739481800199, threshold
  0.9186015432508062): CAL 765 rows, precision 0.0614 (~11× prevalence);
  TEST 532 rows, precision 0.0150 (~25× prevalence). This is the layer's
  primary operational value — a defensible watch-list.
- **Confidence/coverage interpretation (INFERENCE, per method)**: conformal
  p-values are frequentist validity objects — the layer guarantees
  *coverage*, not a probability of correctness, and the guarantee is
  conditional on within-class exchangeability of calibration and test rows.
  Class-1 is included for ~91% of rows (t1 = 0.0047), so sets are rarely
  decisive for the positive class.
- **Inference cost (OBSERVED)**: fit 0.85 ms; TEST inference 54 ms (4 alphas,
  47,000 rows); negligible vs score computation.
- **Limitations**: no distribution-shift guarantee; ~10% of each class sits
  on the 0.46347395 tie leaf; the layer does not improve ranking.
- **Status**: ACCEPTED as an uncertainty/coverage overlay; it does **not**
  replace the frozen alert model or modify any score/policy.

---

## 14. Explainability Findings

- **Method**: LightGBM native `pred_contrib` — path-dependent TreeSHAP as
  implemented by LightGBM (identical algorithm to `shap.TreeExplainer` for
  this Booster); margin space (raw logit): `margin = bias + Σ contrib`,
  `prob = sigmoid(margin)`.
- **SHAP relationship (OBSERVED)**: shap TreeExplainer cross-check on 1,000
  rows: max absolute diff vs native **0.0** (bit-identical); used as
  validation only — `pred_contrib` is the runtime path (no new dependency).
- **Reconstruction validation (OBSERVED)**: margin == bias + Σ contrib to
  max err 4.9e-14 (TRAIN), 4.8e-14 (CAL), 3.6e-14 (TEST); tol 1e-6.
- **Explained rows (TEST, once; OBSERVED)**: **21,043 rows = 49 alerts +
  20,494 monitor + 500 non-alert sample**; all **49 alerts** are
  confident-positive conformal set {1}; monitor set dist {1}: 532, {0,1}:
  19,962; mean alert margin 3.2535 (mean prob 0.9565), mean base −2.3178.
- **Top reasons (OBSERVED)**: `usb_connection_count` top-1 positive reason in
  **36/49** TEST alerts (and top-1 in 50% of the 244 CAL alert rows);
  `device_consistency_score` top-1 in 9/49 (top-3 in 72% of CAL alerts);
  `http_activity_count` top-1 in 3/49.
- **Global importance (OBSERVED)**: `http_activity_count` ≈ 48.8% of
  |contribution| but **bipolar** — signed effect ≈ 0 on alert rows (raises
  risk for some rows, lowers it for others); do not summarize with one sign.
  Kendall τ (gain vs explanation) 0.727 CAL / 0.697 TRAIN (descriptive).
  *(See [Figure 5](figures/fig5_feature_importance.png) for the global
  SHAP importance bar chart.)*
- **Graph contribution (OBSERVED)**: 4 graph features carry **20.5%** of
  alert-row |contribution|.
- **Stability (OBSERVED)**: bit-identical recompute; row-order invariant;
  perturbation: decision flips ≤ 3.52% per feature for unit deltas, but
  top-3 rank churn ≈ 95% → reason rank is descriptive, not stable ground
  truth.
- **Counterfactuals**: **NOT SUPPORTED** — a counterfactual reports a
  hypothetical score the frozen model never produced, breaking the
  frozen-output contract; re-reporting altered scores is forbidden.
- **Resource cost (OBSERVED)**: contributions TRAIN+CAL (454k rows) 199.8 s;
  stability 171.4 s; shap 0.61 s; TEST (47k rows) 20.7 s; peak RSS 1,266 MB;
  explanations parquet 4.66 MB (21,043 × 39).
- **Limitations**: attributions are not causal claims (templates omit causal
  verbs, test-enforced); explanations inherit model ranking quality;
  bipolar `http_activity_count` needs per-row reasoning.

---

## 15. Leakage and Research Integrity

Consolidated controls (all OBSERVED/verified by tests; full register:
[docs/leakage_analysis.md](docs/leakage_analysis.md), 12 risks):

| Area | Control |
|---|---|
| Temporal split | Chronological TRAIN → CAL → TEST only; random splits rejected (autocorrelation, contiguous malicious windows) |
| Prediction-time availability | Every feature must exist at prediction time (day-local or strictly-past windows) |
| User-day independence | One row = one user-day; no future rows affect past rows (day-independence test) |
| Future-information restriction | `unusual_access_count` strictly past [day-28, day-1]; no full-timeline statistics |
| Graph first-use rules | Historical terms from first-use day strictly before d; same-day excluded from history; LDAP strictly before month(d) |
| Calibration isolation | CAL is a separate chronological block; never merged into TRAIN |
| TEST isolation | TEST evaluated exactly once per finalized experiment; never used for tuning, early stopping, threshold selection, or model selection |
| Threshold isolation | Thresholds from CALIBRATION only (Phases 2–9) |
| Conformal calibration isolation | Fit on CALIBRATION only (n1/n0); TEST labels never enter the inference path |
| Explainability isolation | Selection policy is label-free (frozen scores only); method frozen before TEST; global analyses on TRAIN+CAL only |
| Label handling | Answer key consumed only by the label mapper (dataset == 4.2); no label columns in feature tables (structural test) |
| Scaling/normalization | TRAIN-only fits, frozen for CAL/TEST (Phase 8 CDFs) |
| ID collisions | Joins use (user, day) / (user, pc) keys, never cross-file ids |
| Malformed label date | `CDE1846` skipped, reported, never silently fixed |
| Rejected risks | Filename-based features (degenerate); answer-key-derived sensitivity (replaced by magic-byte rule) |

**TEST protocol**: TEST is final evaluation evidence, not a tuning
environment. Every phase: decisions locked on TRAIN/CAL evidence, then a
single TEST evaluation, then (for freeze phases) a deterministic rerun gate
that asserts bit-identical reproduction.

---

## 16. Reproducibility

- **Fixed seeds**: model seed 42 everywhere; no seed selection (Phase 7
  robustness explicitly non-selective); deterministic thresholds.
- **Configuration**: `src/config.py` (features, splits, constants) and
  `src/experiments/phase7.py` FROZEN_CONFIG are the sources of truth;
  freeze records must agree (test-enforced).
- **Artifact storage**: every measured claim backed by a JSON/parquet record
  in `reports/artifacts/`; artifacts pulled to the PC and re-parsed
  (row counts, keys, score ranges, reconstruction identities).
- **Source synchronization**: sources byte-verified vs local via MD5 before
  each Kaggle run; pushed `.py` files are jupytext-wrapped by the server →
  gzip+b64 push with kernel-side `gzip.decompress` (verified byte-identical).
- **PC/Kaggle separation**: PC holds source/tests/docs/reports/knowledge;
  Kaggle holds dataset + execution + derived artifacts; raw data never
  leaves Kaggle.
- **Byte-identical gates**: Phase 4 (baselines at 1e-9), Phase 6 (arm A =
  phase4-b confusion matrix identical), Phase 7/8 (TEST scores bit-equal to
  the frozen predictions parquet), Phase 9 (alert mask reproducible), Phase
  10/11 (rerun gates reproduce saved records bit-identically).
- **Model integrity checks**: num_trees 186 = record; feature names = frozen
  12; gain importance matches the Phase 7 record (Phase 11 re-verified).

### Phase 19 archive (OBSERVED)

The Phase 19 archive is integrity-verified and contains the final scientific
result, report, manifest, state, execution log, threshold evidence, source,
environment metadata, inventory, and checksums. It is not fully self-contained:
the recovery-amendment source, some test files, and phase14_split.json remain
available outside the archive and are documented in the missing-artifact
inventory.

**Archive hashes:**
- `PHASE19_COMPLETE_ARCHIVE.zip`:
  MD5 `692d5c77f57703ee597b9f9aa7868597`,
  SHA256 `8e4cbd73df6af491ceec053ee19ba32970fee9cfcff68f7cf53e12b5f19837a2`
- `PHASE19_COMPLETE_ARCHIVE.tar.gz`:
  MD5 `9ab29208dd5d0d817278c8c3dc615675`,
  SHA256 `61cc17a14521a00c0fbedf327df5c733cd571a357318596f35417f5eff453c57`

**Critical scientific hashes:**
- `phase19.py` MD5: `6cd6f089e1f7277f3a3b79c52f5ba37f`
- Threshold finalization MD5: `985b11b4cdc3f778e536e9021e93a131`
- Confirm result MD5: `fb835ce71b0596e9988e2e31bd03319d`
- Confirm report MD5: `75a40c1e0802aab0970d54e5691f841e`
- Confirm manifest MD5: `0c8bc8df3d125288ce5510a3366af5f6`

**Archive contents status:**
- `PHASE19_SOURCE_INCLUDED`: YES
- `DEV_EVIDENCE_INCLUDED`: YES
- `THRESHOLD_EVIDENCE_INCLUDED`: YES
- `FINAL_CONFIRM_RESULT_INCLUDED`: YES
- `FINAL_CONFIRM_REPORT_INCLUDED`: YES
- `FINAL_CONFIRM_MANIFEST_INCLUDED`: YES
- `FINAL_CONFIRM_STATE_INCLUDED`: YES
- `EXECUTION_LOG_INCLUDED`: YES
- `ENVIRONMENT_CAPTURED`: YES
- `INVENTORY_CREATED`: YES
- `CHECKSUMS_CREATED`: YES

**Known archive limitations:**
- `RECOVERY_AMENDMENT_INCLUDED`: NO (available locally/project-side;
  documented in missing-artifact report)
- `TESTS_INCLUDED`: PARTIAL (final status documented; not all test source
  files archived)
- `DEPENDENCIES_INCLUDED`: PARTIAL (`phase14_split.json` not included in
  archive; documented as missing / external)
- **Tests**: full local suite green at every phase (Section 17); real-data
  tests run on Kaggle.
- **Report generation**: master report updated only from verified records;
  conflicts recorded per Section 26.

### Phase 19 recovery chronology (OBSERVED)

Phase 19 encountered an implementation issue during deployment; the
recovery path preserved all scientific values and produced a valid final
result. Chronology:

**A. Phase 19 v1.1 — initial learned adaptive-risk experiment.**
Candidate families: A0 ML-only, B convex fusion, C logistic stacking,
D/E residual models, F temporal persistence, G Bayesian evidence fusion,
H gating models, I/J deep Option-B exploratory models. 32 lightweight
candidates evaluated. DEV selected: Family C, hp = 1.0. Historical DEV
canonical MD5: `a4f1fa5f772ab5526aac0b66649c1d41`.

**B. Threshold leakage discovered.** Before scientific CONFIRM
evaluation, the v1.1 CONFIRM implementation was found to derive a
best-F1 threshold using CONFIRM labels. This violated the intended
leakage-safe protocol. Important: PH19_CONFIRM had NOT yet been
scientifically evaluated at this point.

**C. Phase 19 v1.1.1 — corrected threshold protocol.** Thresholds
derived from DEV-only OOF predictions. A0 threshold frozen before
CONFIRM: `0.989445417431935`. Family C threshold frozen before CONFIRM:
`0.35866671520672855`. Raw scores used for ROC-AUC / PR-AUC; frozen
thresholds used for threshold-dependent metrics. Threshold canonical
MD5: `f73d2414902e2091fe109eedec3af906`. Threshold finalization MD5:
`985b11b4cdc3f778e536e9021e93a131`.

**D. Premature administrative state failure.** A v1.1.1 runner
persisted `confirm_opened = true` too early, before the first
label-dependent CONFIRM evaluation. It then crashed during prerequisite
loading because `load_dev_checkpoints()` was called incorrectly without
required `ckpt_dir` and `run_id` parameters. Observed after that
failed attempt: no CONFIRM metrics computed, no CONFIRM result JSON, no
CONFIRM report, no CONFIRM manifest, no CONFIRM outcome observed.
Classification: administrative open = YES; scientific CONFIRM
exposure = NO.

**E. Phase 19 v1.1.2 recovery amendment.** A recovery-only runner
amendment was created. Changes: fixed frozen-data loading; moved
`confirm_opened` state transition to immediately before the first
label-dependent CONFIRM operation; preserved all scientific values
(A0 vs C(hp=1.0), thresholds, verdict rules). Full preflight path
passed; 80/80 tests passed. Recovery runner MD5:
`3ac43b1249b4db933d89646cea62f7fb`. Recovery runner SHA256:
`3b21af66be5db402bf7cd10c0fbe70a460e7f6a94edd1e3075d7137607c5e783`.
Final one-time recovery evaluation completed successfully. Final state:
`confirm_opened = true`, `confirm_completed = true`. PH19_CONFIRM rerun
allowed: NO.

### Kaggle execution policy (OBSERVED — [docs/kaggle-execution-policy.md](docs/kaggle-execution-policy.md))

- Kaggle workspace `/kaggle/working` is **ephemeral**: runtime recycling
  wipes it; kernels respawn under the same ID after death (~35 s); child
  processes of a dead kernel do not survive.
- **30 s default** `execute_code` timeout interrupts and kills the cell;
  callers MUST pass an explicit `timeout` for any real work.
- **~120 s client cap**: calls returning empty after >120 s mean the job is
  still running in the background — poll artifacts, do not relaunch.
- Long work runs as **background subprocesses** (nohup + log polling) which
  survive call aborts but NOT kernel death or recycling.
- Recovery: check kernel status before relaunching; never relaunch a job
  speculatively (duplicate concurrent training).
- Outcome journal: `logs/kaggle_execution.jsonl` via `kaggle_exec.py log`.

---

## 17. Test and Verification History

| Phase | Local suite | Kaggle | Reproduction gate | Artifact checks | TEST-once |
|---|---|---|---|---|---|
| Foundation | — | 31 passed | — | validation JSON + summary pulled | n/a |
| Baseline | — | 40 passed | — | predictions 47,000×4 re-verified | once |
| Phase 3 | — | 43 passed | arm A = baseline record exactly | predictions recomputed locally, match records | once per arm |
| Phase 4 | — | 48 passed | a/b byte-exact vs prior records (1e-9) | 4 arms pulled, sizes verified | once per arm |
| Phase 5 | 63 incl. graph | 63 passed (15 graph) | deterministic output test | graph parquet 501,000×7 re-verified | n/a |
| Phase 6 | 78 passed + 4 env skips | 82 passed | arm A = phase4-b exact (all diffs 0.0) | JSON/parquet reloaded locally | once per arm |
| Phase 7 | 95 passed / 4 skipped | verified (real-data skips) | freeze rerun bit-for-bit | record parses; predictions 47,000×4 | once |
| Phase 8 | 116 passed / 4 skipped | verified | arm A bit-equal to Phase 7 file | 3 records + predictions reloaded | once per arm |
| Phase 9 | 146 passed / 4 skipped | verified | TEST scores bit-equal; rerun reproduces record | predictions 47,000×5 | once |
| Phase 10 | 179 passed / 4 skipped | verified | rerun reproduces record (TEST identical) | predictions 47,000×11; set cols from fit JSON | once |
| Phase 11 | final gate **214 passed / 4 skipped** (Phase 11 file 35/35) | verified | rerun reproduces record (TEST identical) | 14 artifacts; explanations 21,043×39; recon 2.4e-14 | once |
| Phase 12 | final gate **277 passed / 4 skipped** (Phase 12 file 28/28, incl. 4 artifact reloadability tests) | verified | rerun reproduces record (TEST identical) | all 10 artifacts md5-verified; parquet 47,000×7 reloaded | once |
| Phase 13 | final gate **312 passed / 4 skipped** (Phase 13 file 35/35: pre-registered table, partition, gates, bounds, verdict + CAL-only enforcement, Wilson CI, determinism) | not needed (local run) | deterministic double run bit-identical (`bit_identical=true`) | frozen inputs md5-matched (parquet `efca7668…`, rolling `d3455748…`); 5 new artifacts reloadable | once (frozen parquet re-read, never re-scored) |
| Phase 14 | final gate **346 passed / 4 skipped** (Phase 14 file 34/34: allocation rules + pre-registered 70-user table, real-parquet block/label gates, CDE1846 absence, feature isolation, operating points, user-block bootstrap incl. degenerate, Wilson/Gini, interpretation bands, determinism, artifact reloadability + manifest md5s) | not needed (run on kernel; local suite with `CERT_WORKING=reports`) | deterministic double run bit-identical (`determinism: True`, run1 32.96 s / run2 32.67 s) | 13 artifacts md5-verified against the kernel manifest; merged input md5 `9a3b1885…9def` == frozen record (rebuilt after recycle); predictions 54,108×6 reloaded | once (auxiliary TEST; authoritative chronological TEST not re-evaluated) |
| Phase 15 | final gate **379 passed / 4 skipped** (Phase 15 file 33/33: 22 unit tests — buckets, d/KS/PSI formulas, percentile, zero-share clip, cold-start exclusion, onset-half labels, determinism — + 11 artifact reloadability/manifest tests) | not needed (fully local run) | gates PASS: TEST score repro max diff 5.551e-17 ≤ 1e-12 (54,081/54,108 bit-exact); CAL AUC reproduced exactly (diff 0.0); double run bit_identical=True | 10 data artifacts md5-verified via manifest (manifest self-excluded per Phase 14 convention; runner fixed to write the manifest after the final log line); frozen-input immutability test PASS (merged `9a3b1885…9def`, model `3778a4d8…fa19`, p14 predictions `71eeb3f1…d213d`) | once (authoritative chronological TEST never read or re-scored) |
| Phase 16 | full local suite re-run inside the gate: **379 passed / 4 skipped** (identical to Phase 15 — packaging introduced no behavior change) | not needed (fully local verification) | verification-only gate `scripts/verify_project.py`: 37/37 checks PASS, exit 0 (structure, manifests md5+size 20/20, frozen model parse-only 186 trees/12 features, threshold exact, JSON/parquet load, master sections+links, README links, secrets clean, test suite) | fresh baseline md5s for ALL artifacts recorded in `final_verification_report.json` (verification record, not re-evaluation); `final_verification_report.json` reloadable (G14) | none (no TEST evaluation occurred; prediction parquet loaded for parse/hash only) |
| Phase 17 | full local suite with `CERT_WORKING=reports`: **404 passed / 7 skipped** (Phase 17 file 28/28: unit transforms incl. merge rule, PAV monotonicity, ECDF ties, hand-computed Brier/ECE, operating-point semantics, coverage/Wilson, small-panel bootstrap, verdict paths incl. tie→CAUTION, full-pipeline determinism on exact-shape synthetic frames 54,108 rows (379/476 pos), structural guards, artifact reloadability + manifest md5s + frozen-input immutability; 3 artifact tests pending at first run — pass after execution) | not needed (fully local run) | gates PASS: arm-A reproduction exact (AUC-ROC 0.786513147174144, 1,026 alerts); CAL AUC exact (diff 0.0); g1b raw threshold 0.9837865316173778 = Phase 14 record, B/D alert sets identical; determinism 3× bit-identical (`determinism: True`, runs=3, ≈218 s per run) | 6 artifacts md5-verified via `phase17_manifest.json` (self-excluded; `phase17_cost.json` added to the manifest in the bookkeeping pass); frozen-input immutability PASS (`phase14_test_predictions.parquet` `71eeb3f1…213d`, `phase15_cal_scores.parquet` `c955a4ec…1330`, `phase14_user_diagnostics.json` `cd893661…dd07`); `scripts/verify_project.py` 37/37 PASS pre-execution | once (user-disjoint TEST read once from the frozen prediction parquet; authoritative chronological TEST never read or re-scored) |
| Phase 18 | evaluation-only; no standalone retained test artifact. Deterministic science was reproduced during execution, but exact standalone Phase 18 test-suite evidence was not retained after the Kaggle reset | not needed (Kaggle execution) | deterministic repeated execution required; deterministic science reproduced; no CONFIRM or chronological TEST accessed | no new artifacts retained (artifacts from one execution lost after Kaggle reset) | none (evaluation-only; no chronological TEST) |
| Phase 19 | full local suite: **80 passed / 0 skipped** (36 v1.1.1+v1.1.2 tests + 44 backup tests; static-leakage-audit comment-filtering fix, regression tests for recovery runner) | not needed (Kaggle execution) | CONFIRM executed once on Kaggle; verdict FAIL; all hashes verified (6/6 PASS); archive created | 7 artifacts: confirm result, report, manifest, state JSON, recovery amendment, user allocation, threshold finalization | once (CONFIRM evaluated once; no re-evaluation) |

Skips are pre-existing real-data tests that require the Kaggle environment;
model-integrity gates (tree counts, feature names, gain matches) were
re-verified in Phases 9–11.

---

## 18. Resource/Performance Profile

Measured values (OBSERVED); everything not listed is NOT VERIFIED.

### Computational Efficiency Summary

The frozen system is designed for lightweight continuous operation. All
measurements below are wall-clock time and peak RSS on a single CPU core;
**CPU utilization % was NOT MEASURED** and must not be inferred from wall-clock
times. GPU was not used for any component.

| Metric | Value | Source |
|---|---|---|
| Model training (lgbm-graph-v1, 12 features, 186 trees) | 4.64 s | phase7_freeze record |
| Model inference (47,000 TEST rows) | 1.29 s | phase7_freeze record |
| Graph feature generation (501K user-days) | 13.0 s | phase6_cost.json |
| Full decision engine (501K rows, all overlays) | 180.3 s | phase20_cost |
| Peak memory (full decision engine) | 1,744 MB | phase20_cost |
| Model disk size | 651 KB | phase7_freeze record |
| Explanations parquet (21,043 TEST-alert rows) | 4.66 MB | phase11_cost |

**Sustained throughput estimate (INFERENCE)**: scoring 1,000 user-days takes
~0.26 s (linear extrapolation from 47K in 1.29 s). A 10,000-user deployment
would score one day in ~2.6 s; the full 501K-day table in ~130 s. These are
INFERENCE from OBSERVED single-run measurements, not a benchmark.

| Component | Time | Memory | Artifact size | GPU | Notes |
|---|---|---|---|---|---|
| User-day aggregation (501,000 rows, incl. 14.5 GB http) | not measured | duckdb path (pandas path OOM ~28.7 GB RSS) | table on Kaggle | no | rebuild `build_user_day.py` |
| Graph feature build (Phase 5) | 12.99 s (script 17.9 s) | not measured | graph parquet 556,021 B | no | single pass over logon+file+LDAP |
| Baseline training (6f, 182 trees) | 4.36 s | not measured | model 635,440 B | no | |
| Phase 4 arm b (8f, 160 trees) | 4.13 s train / 1.06 s predict | not measured | model 559,479 B | no | |
| Phase 6 arm C (13f, 186 trees) | 4.76 s train / 1.21 s predict | not measured | model 651,085 B | no | merged table 501,000×16 (1,077,577 B) |
| Phase 7 freeze (12f) | 4.64 s train / 1.29 s predict | not measured | model 651,047 B | no | |
| Phase 8 component build + scoring (CAL) | 0.197 s | not measured | artifacts 4–174 KB | no | |
| Phase 9 frozen scores CAL+TEST | 0.287 s | not measured | — | no | 13 policies over 59,000 rows: 0.105 s |
| Phase 10 conformal fit / inference | 0.85 ms / 54 ms (TEST, 4 alphas) | not measured | predictions parquet 1.8 MB | no | |
| Phase 11 contributions (454k rows) | 199.8 s | peak RSS 1,266 MB (test stage); 867 MB (cal) | explanations parquet 4.66 MB | no | stability checks 171.4 s; shap 0.61 s; TEST 20.7 s |
| Phase 12 operational envelope | whole phase 19.1 s (rolling analysis 1.06 s) | peak RAM NOT VERIFIED | 9 artifacts ≈ 129 KB | no | no retraining; 106,000 rows scored; 4 policies + 4 dedup variants + 3 rolling sizes |
| Phase 13 temporal stability | local only: analysis ×2 = 42.0 s | peak RAM NOT VERIFIED | 5 artifacts ≈ 60 KB | no | no Kaggle execution; no scoring; frozen records + parquet re-read; determinism gate bit-identical |
| Phase 14 user-level holdout | protocol ×2 (determinism gate) = 66.1 s (run1 32.96 s, run2 32.67 s; write 0.034 s); rebuild after recycle: user_day 20 s + graph 20 s + merged 1 s | NOT VERIFIED | 13 artifacts ≈ 921 KB (model 750,807 B; predictions 115,086 B) | no (T4 unused) | 1000 user-block bootstraps per run; 108,216 rows scored per run; md5-gated inputs |
| Phase 15 unseen-user diagnosis | local only: full protocol 6.02 s (analysis ×2 5.40 s, bit-identical) | NOT VERIFIED (pandas DF ≤ ~1 GB RSS expected) | 10 data artifacts ≈ 171 KB (cal scores parquet 110,026 B) + manifest/log/cost | no | no Kaggle execution; no model, no scoring of the chronological TEST; CAL block re-scored once for the AUC gate (gate-only) |
| Phase 16 final packaging gate | `python scripts/verify_project.py` (final run 2026-08-17): checks ≈ 4 s + full pytest suite 125.82 s = **130.3 s total**, exit 0 (suite time observed 94–170 s across identical runs) | NOT VERIFIED (trivial) | `final_verification_report.json` 27,360 B; README/requirements/environment/artifact index ≈ 27 KB | no | verification-only; no model/scoring/calibration; Kaggle not required |
| Phase 17 calibration transfer | local only: total 655.6 s (gate3 0.02 s, gate4 before 0.38 s, load 0.13 s, analysis ×3 = 654.7 s ≈ 218 s per pipeline run, gate4 after 0.16 s, write ≈ 0.3 s) | NOT VERIFIED (pandas DFs ≤ ~1 GB RSS expected) | 7 artifacts ≈ 250 KB + `phase17.log` (incl. manifest) | no | no Kaggle execution; no model training (3 calibration fits on 54,108 CAL rows + 1000× refit bootstrap dominate the runtime); frozen records + parquet re-read; determinism gate 3× bit-identical |

---

## 19. Decisions Log Summary

Authoritative: [knowledge/decisions/decision-log.md](knowledge/decisions/decision-log.md).

| # | Decision | Evidence | Phase | Status |
|---|---|---|---|---|
| 1 | CERT r4.2 is the fixed dataset | foundation | Foundation | established |
| 2 | user × day is the analytical unit | leakage analysis | Foundation | established |
| 3 | Chronological TRAIN→CAL→TEST; random splits rejected | splits.py | Foundation | established |
| 4 | LightGBM main model config (lr 0.03, 31 leaves, etc.) | baseline/phase4 | Baseline | established |
| 5 | `lgbm-baseline-v2` frozen behavioral baseline | phase4 | Phase 4 | established (superseded as candidate, still the regression reference) |
| 6 | TEST never used for tuning; evaluated once | all reports | all | established |
| 7 | scale_pos_weight 255.66 from TRAIN only | baseline | Baseline | established |
| 8 | Raw data never leaves Kaggle | ism-principles* | Foundation | established |
| 9 | `unusual_access_count` 28-day strictly-past window | phase4 §5.2 | Phase 4 | established |
| 10 | `sensitive_file_access_count` magic-byte a-priori rule | phase3 | Phase 3 | established |
| 11 | Graph features experimental only (no model) | phase5 | Phase 5 | established (superseded: Phase 6/7 merged 4 into the frozen model) |
| 12 | External Kaggle notebooks are research material, not dependencies | research/README | — | established |
| 13 | `lgbm-graph-v1` (12 features) frozen candidate | phase7 | Phase 7 | established — **current** |
| 14 | Adaptive risk NOT adopted | phase8 | Phase 8 | established — rejected |
| 15 | Production alert policy = frozen max-F1 threshold 0.9186015432508062 | phase9 | Phase 9 | established — **frozen** |
| 16 | Conformal layer ACCEPTED as diagnostic overlay | phase10 | Phase 10 | established — accepted |
| 17 | Explainability layer ACCEPTED; counterfactuals NOT SUPPORTED | phase11 | Phase 11 | established — accepted |
| 18 | Operational envelope: RETAIN frozen P0 (no de-duplication / rolling policy adopted) | phase12 (R1–R4 rule) | Phase 12 | established — policy unchanged |
| 19 | Temporal stability: pre-registered CAL verdict FAIL (W8 tail: B5 hard + B4 marginal) → documented risk, system unchanged | phase13 (pre-registered B1–B6 + margin rule, CAL-only) | Phase 13 | established — descriptive finding, no system change |
| 20 | User-level holdout (Phase 14): evaluation-only auxiliary model on user-disjoint blocks — measured (coverage 13/15, AUC-ROC 0.786 vs 0.939); NO production decision; frozen system unchanged | phase14 (pre-registered spec §6 allocation + §20 bands; determinism gate) | Phase 14 | established — evaluation only, no adoption |
| 21 | Phase 15 diagnosis: degradation is entity-associated score-calibration (79.8% of unseen-user FPs within 0.10 of the frozen threshold) + ranking failure for a low-activity minority (2/15 users, max scores 0.498/0.726); feature shift small, graph weakness intrinsic; NO threshold/model change — frozen system unchanged; future unseen-user work must define an entity-conditional protocol (new data or explicit authorization) | phase15 (pre-registered spec analyses A–G; gates: score repro 5.551e-17, CAL AUC exact, bit-identical double run) | Phase 15 | established — diagnosis only, no adoption |
| 22 | **Phase 16 final packaging (no ML)**: the frozen system is packaged for external handoff — README.md, requirements.txt (PC env, versions OBSERVED), docs/environment.md (PC + Kaggle env record; unpinned Kaggle versions documented as unrecorded), scripts/verify_project.py (verification-only gate: structure, manifests md5+size 20/20, frozen model parse-only 186 trees/12 features, threshold exact, JSON/parquet load, master sections+links, secrets, full suite), reports/artifacts/ARTIFACT_INDEX.md, final_verification_report.json, phase16 report, master update, decision #22 | phase16 (acceptance gates G1–G17 all PASS; no training/scoring/calibration; TEST never re-scored) | Phase 16 | established — packaging complete, frozen system unchanged |
| 23 | **Phase 17 calibration transfer (research-only, mechanical verdict FAIL)**: calibration fitted on the user-disjoint CAL block transfers to unseen users only for Platt scaling (B: TEST Brier 0.00796, +92.25%, ECE 0.0026; operating point recovers the Phase 14 secondary record exactly — 191 alerts, F1 0.4438, precision 0.7749, threshold 0.9837865316173778); binning (C) transfers Brier/ECE but not its operating point (21,127 alerts); ECDF (D) is a percentile map, not calibration (−216%); FAIL triggered by the pre-registered ranking bound (|ΔAUC-ROC| ≤ 1e-9): C −0.0725, D −3.27e-6 — the empirical rank-based AUC is not invariant under tie creation. Spec §25: FAIL ⇒ no Phase 18 calibration work on this mechanism without separate authorization; no production change; frozen system unchanged | phase17 (approved spec; gates 1/1b/2/3/4 PASS; determinism 3× bit-identical; bootstraps seed 42 n=1000; suite 404 passed / 7 skipped; verify_project 37/37 PASS) | Phase 17 | established — evaluation only, no adoption |
| 24 | **Phase 18 adaptive risk v2 (evaluation-only, verdict FAIL)**: redesigned adaptive-risk formulation (α·ML + β·Trust + γ·Context + δ·Behavioral); user-disjoint DEV-like evaluation; no chronological TEST access; deterministic repeated execution required. Results: deterministic science reproduced; optimized fusion collapsed toward ML-only (α ≥ 0.95); no adaptive-risk arm met the required improvement criteria; expensive context computation dominated runtime (~2.57 h/run); earlier runner crash after science due to implementation/state issue; artifacts from one execution lost after Kaggle reset. Verdict: FAIL. No production change; frozen system unchanged. No standalone phase report retained | established (Phase 18, evaluation only, no adoption) | (evaluation-only; no standalone phase report; artifacts lost after Kaggle reset) |
| 25 | **Phase 19 end-to-end CONFIRM (verdict FAIL)**: one-time authorized evaluation of adaptive risk A0 vs Family C on the CONFIRM partition (196 users, 98,196 rows, 245 positives, overlap=0). Final verdict: FAIL — delta ROC-AUC -0.0281 < -0.01 (A0: 0.7789, C: 0.7507); delta PR-AUC -0.0013; delta F1 -0.0021; alerts 72/74. Adaptive risk rejected for production. Frozen system unchanged since Phase 7/9. No ML phase running or scheduled. | phase19 (approved spec; CONFIRM executed once; verdict FAIL per pre-registered rule) | Phase 19 | established — evaluation only, no adoption; adaptive risk rejected for production |
| 26 | **Phase 20 final system integration + full-scale operational validation**: complete dashboard-ready user-day decision table materialized (501,000 rows x 21 columns). Frozen LightGBM model (lgbm-graph-v1, 12 features, seed=42, iter=186) scored all 501K user-days. Risk levels computed (ALERT=3785, BORDERLINE=1484, MONITOR=178204, NON-ALERT=317527). Conformal confidence integrated (high-confidence=61017, ambiguous=439983). SHAP explanations for 21043 TEST-alert rows; feature-value fallback for 479957 remaining. Trust diagnostics for 476200 rows. Score integrity: 0 mismatches vs Phase 7 predictions. Alert integrity: 0 ALERT-only mismatches vs Phase 10 (58 BORDERLINE additions expected per output_schema.yaml). Determinism: PASS (MD5 match). Leakage audit: PASS. Runtime 180s, peak memory 1.7GB. Parquet MD5=6476f791d1cc9f21327a94e1373f9e09. 54/54 tests pass. No ML change; frozen system unchanged. | phase20 (integration + full-scale validation; 54 tests; artifact cross-check) | Phase 20 | established — full decision table materialized; ready for dashboard |

\* `ism-principles.md` (`.opencode/instructions/ism-principles.md`) is
referenced by the decision log; verified to exist on 2026-08-16 and its
content corroborates these claims (see Section 26 conflict record #2,
RESOLVED).

---

## 20. Rejected / Failed Approaches

### Baseline Scope Note

The model comparison in this project was restricted to LightGBM variants
within the same training framework. **Random Forest, XGBoost, CatBoost, and
Logistic Regression** were proposed as candidate baselines during early
planning but were **never trained, evaluated, or recorded** in any
authoritative artifact. No RF/XGBoost/CatBoost/LR result exists in
`reports/artifacts/`; any comparison table that includes these algorithms
would contain fabricated data. The ablation in Table 1 (Section 31) is the
only published model comparison and covers LightGBM arms A–C plus an
adaptive-risk variant. A broader algorithmic comparison remains an open
direction for future authorized work.

| Approach | What was tested | Evidence | Decision | Reconsideration possible? |
|---|---|---|---|---|
| lr 0.05 baseline config | first training attempt | collapsed at best_iteration 1; degenerate 0.0/1.0 scores | rejected (baseline) | no value seen |
| 50%-precision calibration point | Phase 2/3/4/9 | unreachable on TEST (0–1 alerts); degenerate | not adopted | only with different calibration data |
| `file_access_consistency_score`, `rare_file_access` | filename-based graph features | all r4.2 filenames globally unique → degenerate | rejected | no (dataset property) |
| `url_consistency_score` | URL-based consistency | cost of 14.5 GB scan for unproven signal | deferred | yes (cost-based, not evidence-based) |
| `department_file_type_mismatch_count` | 13th graph feature | zero gain 5/5 seeds; 0.011% non-zero; CAL single-feature AUC-ROC 0.500; removal delta exactly 0 | rejected (Phase 7) | unlikely (feature is near-constant) |
| Graph-only model (Arm B) | 5 graph features alone | AUC-PR 0.014, stump, FPR 0.86% | rejected standalone | only as part of a larger design |
| Equal-weight adaptive risk | α=β=γ=1/3 | CAL AUC-PR 0.550→0.218; TEST worse everywhere; 5× alerts | rejected (Phase 8) | only with different component semantics |
| Learned adaptive-risk mixture | simplex weight grid | collapsed to [1,0,0] = ML; trust_risk anti-correlated with target | rejected (Phase 8) | only with new components |
| Tight percentile alert policies (pct_0.0005/0.0010) | global top-p | zero-alert window; S2/S3 fail | rejected (Phase 9) | as documented candidates only |
| High-capacity policies (daily_top_20+) | top-N per day | precision collapses < 0.2 | rejected (Phase 9) | as documented candidates only |
| Counterfactual explanations | "what if" altered-score outputs | violates frozen-output contract; forbidden | NOT SUPPORTED (Phase 11) | yes, via a new authorized phase under the recorded criteria |
| Adaptive-risk "trust" semantics | graph consistency = low risk | trust_risk negatively correlated with target | rejected | different trust definitions untested |
| Per-user de-duplication 3d/7d (B2/B3) | suppress repeat alerts per user | recall halves (0.248/0.139 vs 0.517); fails R1 | not adopted (Phase 12) | only if recall loss is accepted |
| Rolling calibration threshold (7/14/30 d) | re-estimate threshold per window on strictly-past CAL data | aggregate F1 never beats frozen (0.567 vs 0.589); wd=30 rests on 1 window | not adopted (Phase 12) | as a review procedure, not a policy |
| Recalibration as unseen-user remedy (Phase 17) | 3 pre-registered arms fitted on the user-disjoint CAL block, applied to unseen TEST users | mechanical verdict FAIL per the pre-registered §14 rule: rank-based AUC-ROC not invariant under tie creation — arm C (binning) ΔAUC-ROC −0.0725 (3 distinct p̂ values on TEST), arm D (ECDF) −3.27e-6 (1,731 tie collisions across CAL-score gaps); D also degrades Brier (−216%); calibration cannot repair the Phase 15 ranking failures (JJM0203/WDD0366 zero-alert under every conservative alert set) | not adopted (Phase 17; spec §25: no Phase 18 calibration work on this mechanism) | only via a new authorized phase with tie-bounded pre-registration and a different operating design |
| Adaptive risk — end-to-end CONFIRM (Phase 19) | A0 vs Family C; CONFIRM partition (196 users, 98,196 rows, 245 positives, overlap=0); thresholds from DEV_OOF; no training, no threshold tuning | verdict FAIL: delta ROC-AUC -0.0281 < -0.01 (A0: 0.7789, C: 0.7507); delta PR-AUC -0.0013; delta F1 -0.0021; alerts 72/74; precision 0.75/0.73; recall 0.2204 both; MCC 0.4059/0.4003; balanced accuracy 0.6101 both | **rejected for production** (Phase 19; verdict FAIL) | no — frozen system unchanged; adaptive risk component definitions rejected; new definitions would require a new authorized phase |
| Adaptive risk v2 (Phase 18) | redesigned formulation (α·ML + β·Trust + γ·Context + δ·Behavioral); user-disjoint DEV-like evaluation; no chronological TEST access | optimized fusion collapsed toward ML-only (α ≥ 0.95); no arm met required improvement criteria; expensive context computation (~2.57 h/run); artifacts lost after Kaggle reset | **rejected** (Phase 18; verdict FAIL) | no — frozen system unchanged; no production change |

---

## 21. Current Open Risks

### Data risks
- Only 30 positive TEST rows (2 scenario-3); prevalence 0.38%. (non-blocker;
  all decisions mitigated via CALIBRATION evidence)
- Label date `CDE1846` unparseable — excluded, documented. (non-blocker)

### Evaluation risks
- Entity overlap across splits: now measured by Phase 14 (evaluation-only
  user-level holdout) — 13/15 unseen users detected, but ranking degrades
  (AUC-ROC 0.786 vs 0.939; precision at frozen threshold 0.202 vs 0.286):
  entity-level transport loss quantified; auxiliary cohort is not
  comparable to the chronological record (476 vs 30 positives).
- Calibration transportability: max-F1 precision drops CAL→TEST (v1 0.80→0.22;
  v2 0.90→0.41; graph-v1 0.68→0.29). (non-blocker) Phase 17 (OBSERVED):
  a CAL-fitted Platt calibration transfers to unseen users (TEST Brier
  0.00796, ECE 0.0026, operating point = Phase 14 secondary record
  exactly), but no production change was authorized; ranking-level misses
  are not calibration-fixable (R17-2). Phase 19 (OBSERVED): adaptive risk
  CONFIRM verdict FAIL — delta ROC-AUC -0.0281 < -0.01; adaptive risk
  rejected for production (R19-1).
- Wide TEST CIs (AUC-PR [0.140, 0.460]); bootstrap assumes row exchangeability
  within users. (non-blocker)
- Gain-based importance is seed-sensitive (τ min 0.143). (non-blocker)

### Model risks
- Window recall drift 0.575→0.429 (CAL, Phase 9) — no adaptation allowed.
  (non-blocker, documented)
- Small-α alert counts noisy (CAL alert-rate CV 22% across seeds). (non-blocker)

### Adaptive risk rejection (Phase 19)
- **R19-1 (CONFIRM verdict FAIL, OBSERVED)**: end-to-end evaluation of
  adaptive risk A0 vs Family C on the CONFIRM partition (196 users, 98,196
  rows, 245 positives, overlap=0). Delta ROC-AUC -0.0281 < -0.01 threshold.
  Delta PR-AUC -0.0013 (not positive). Delta F1 -0.0021. Verdict: FAIL.
  Adaptive risk rejected for production. Frozen system unchanged. No ML
  phase running or scheduled. (non-blocker, documented; adaptive risk
  component definitions rejected; new definitions would require a new
  authorized phase)

### Temporal stability risks (Phase 13)
- **R13-1 (tail degradation, OBSERVED)**: CAL 3-day tail window W8
  (03-29..03-31) — score median 0.04141 (B5 hard violation, 28.6% beyond
  margin) and precision 0.3333 (B4 marginal); TEST late windows W11–W15 show
  precision 0 with 0–1 positives. Pre-registered CAL verdict FAIL — a
  descriptive risk-register statement; system unchanged by design (no
  adaptation allowed). (non-blocker, documented)
- **R13-2 (false-alert bursts, OBSERVED)**: low-positive windows can produce
  4–8 alerts with precision 0 (TEST W11/W13/W14). (non-blocker, documented;
  block-level FPR stays 0.0007 TEST)
- **R13-3 (tail-window uncertainty, OBSERVED)**: tail windows are small
  (3–5 days, ≤12 positives); Wilson 90% CIs wide (W8 recall [0.140, 0.452]);
  the verdict is descriptive, not a significance test. (non-blocker)
- **R13-4 (unmeasured future span, OBSERVED)**: no labeled data after
  2011-05-17; temporal stability beyond the observed 106 days is unmeasured;
  strict replication requires a new dataset/release (out of contract).
  (non-blocker)
- **R13-5 (record-based CAL checks)**: CAL per-row partition and CAL
  balanced-acc/FPR/FNR are NOT VERIFIED locally (no per-row CAL data);
  record-based assertions only; the optional CAL-only scoring pass is
  documented (spec §12) for a future authorized run. (non-blocker)

### Feature risks
- Content taxonomy limited to OLE2/PDF/ZIP/other; month-granular LDAP.
  (non-blocker)
- `rare_*` graph features contribute small signal; revisit possible. (future
  investigation)

### Graph risks
- Graph features weak standalone; production must regenerate tables with the
  same parquet layouts (separate Phase 5 pipeline). (non-blocker)
- Department feature near-constant (49 user-days) — no signal. (non-blocker)

### Explainability risks
- Attributions ≠ causal claims; top-reason rank fragile under perturbation.
  (non-blocker)
- `http_activity_count` bipolar — single-sign summaries misleading.
  (non-blocker)

### Operational risks
- One user 15% of alerts — repeat-alert fatigue (lever outside frozen policy).
- No production deployment, dashboard, or live evaluation exists.
- Percentile-policy counts scale with daily volume (if ever adopted).
- Capacity at scale NOT VERIFIED: 10,000-user projections (41.4 alerts/day)
  are INFERENCE on OBSERVED behavior; S2/S3 rest on HYPOTHESIS assumptions.
  (non-blocker, documented Phase 12)
- Rolling-threshold evidence is thin (wd=30: 1 defined window). (non-blocker,
  documented Phase 12)

### Reproducibility risks
- Kaggle runtime is ephemeral: recycling wipes `/kaggle/working`; background
  jobs not persistent; sources must be re-pushed (gz+b64) and derived tables
  rebuilt after any recycle. (OBSERVED — occurred during Phase 11 and
  before Phase 14; Phase 14 rebuild md5-matched the frozen record exactly)

### Phase 14 risks (user-level holdout, evaluation-only)
- **R14-1 (unseen-user ranking loss, OBSERVED)**: AUC-ROC −0.153 vs the
  chronological record; precision at the frozen threshold 0.202 (vs 0.286) —
  entity-level calibration-transportability measured for the first time.
  (non-blocker, documented; no production decision by design)
- **R14-2 (cohort comparability)**: auxiliary TEST (476 positives) is not
  directly comparable to the chronological TEST (30 positives);
  interpretation rests on the pre-registered bands only. (non-blocker)
- **R14-3 (small/selected cohort)**: all 15 malicious TEST users fall in the
  low-activity tercile of the 108-user block; mid/high activity strata
  contain no malicious TEST users. (non-blocker)
- **R14-4 (artifact field mislabel)**: `phase14_scenario.json` `n_users`
  holds stratum row counts (see §26 #7); cosmetic, affects no metric.
  (non-blocker, documented)

### Phase 15 risks (unseen-user diagnosis, descriptive)
- **R15-1 (unseen-user score calibration, OBSERVED)**: 79.8% of the 1,026
  unseen-user false positives lie within 0.10 above the frozen threshold
  (median margin 0.045) — absolute-score inflation on unseen benign rows;
  any future entity-conditional calibration must be pre-registered, never
  TEST-derived. (non-blocker; no threshold selected by Phase 15)
- **R15-2 (ranking collapse for a minority, OBSERVED)**: JJM0203 and
  WDD0366 never alert (0 alerts in 501 days; max scores 0.498/0.726) —
  incidents of ≤ 8 malicious days in low-activity profiles are invisible
  to the frozen ranking; alerting-only monitoring has no floor safeguard.
  (non-blocker)
- **R15-3 (intrinsic graph-feature weakness, OBSERVED)**: single-feature
  row AUCs 0.50–0.69 and ~99.9% zero rates for `rare_*` counts hold in
  BOTH cohorts — the weakness is inherent to the feature set, not
  unseen-specific. (non-blocker)
- **R15-4 (coverage vs precision trade-off on unseen users, OBSERVED)**:
  unseen-user coverage improves (13/15 vs 5/5 chronological) only at
  12–21x alert volume — entity-dependent operating points diverge;
  documented, not remedied. (non-blocker)

### Phase 17 risks (calibration transfer, research-only; verdict FAIL)
- **R17-1 (rank-based AUC is not tie-invariant, OBSERVED)**: the empirical
  (average-rank) AUC-ROC estimator changes when distinct scores are mapped
  to equal values even when row order is preserved — arm C −0.0725
  (3 distinct p̂ values on TEST), arm D −3.27e-6 (1,731 of 4,174 distinct
  TEST scores tied across CAL-score gaps). Any future pre-registration
  involving recalibration must bound tie creation explicitly, not rank
  preservation. (non-blocker; drove the mechanical FAIL)
- **R17-2 (calibration cannot repair ranking failures, OBSERVED)**:
  JJM0203 and WDD0366 remain zero-alert under every conservative alert
  set (A 13/15, B/D 10/15 coverage); only arm C "covers" them by
  alerting 39% of all rows. Calibration transfer is orthogonal to the
  Phase 15 ranking problem (R15-2). (non-blocker)
- **R17-3 (D is not a calibration, OBSERVED)**: the ECDF percentile map
  degrades TEST Brier (−216%) and ECE (0.481); it should not be reused
  as a probability calibration. (non-blocker; recorded)
- **R17-4 (spec §10.1/§12.2 invariance claim refuted, OBSERVED)**: the
  approved spec claimed arm C preserves AUC-ROC ("order preserved");
  measurement shows order preservation is insufficient for the discrete
  estimator — recorded in §26 #10. (non-blocker; pre-registration gap)
- **R17-5 (no production claim, holds)**: calibrated operating points
  were never evaluated on the authoritative chronological TEST (off-limits
  forever); NOT VERIFIED for deployment; report states explicitly.

No **blocker** risks are open for the frozen system as specified.

---

## 22. Planned / Not Yet Implemented

Nothing below is an active task; each requires explicit authorization.

- **Production deployment** — not implemented; the frozen system exists as
  code + records only.
- **Dashboard / operational UI** — not implemented; out of scope per plan.
- **Unseen production-surrogate evaluation** — not possible inside r4.2 (data
  ends 2011-05-17; no later window exists). Feasibility analyzed and recorded
  in Phase 13 (spec §0): a strict replication window requires a new dataset
  or release (out of contract; would need separate authorization). If ever
  run, it supersedes the descriptive Phase 13 verdict.
- **Threshold stability / rolling review** — recommended by Phase 9 and
  examined in Phase 12 (rolling thresholds never beat the frozen threshold on
  CALIBRATION; wd=7 closest at aggregate F1 0.567). A future operational
  procedure (re-estimate only the threshold on a rolling CAL window under
  S1/S2/S3); not adopted as a policy.
- **Per-user alert de-duplication / cap policy** — identified operational
  lever (HBO0413); measured in Phase 12 (3d/7d de-duplication halves recall —
  not adopted under R1–R4); not part of the frozen policy.
- **User-level holdout evaluation** — DONE in Phase 14 (evaluation-only):
  coverage 13/15 unseen users; ranking degraded vs chronological; no
  production decision; a user-disjoint re-allocation for a different
  purpose would require a new authorized phase.
- **Unseen-user degradation diagnosis** — DONE in Phase 15 (descriptive,
  local-only): feature shift small, graph weakness intrinsic, alert
  explosion = score-calibration inflation on unseen benign rows, 2 misses
  = ranking failures; frozen system unchanged; a future fix would require
  a new authorized phase with pre-registered entity-conditional
  calibration (R15-1).
- **Final packaging & verification gate** — DONE in Phase 16 (no ML):
  README.md, requirements.txt, docs/environment.md,
  `scripts/verify_project.py` (verification-only, 37/37 PASS),
  ARTIFACT_INDEX.md, final_verification_report.json, phase16 report. The
  "single end-to-end reproducibility" item is closed as a
  verification-only gate (a full ML rerun would violate TEST-once).
- **Advanced graph methods (GNNs, embeddings, PageRank)** — out of scope;
  not implemented.
- **Transformers / neural models** — out of scope; not implemented.
- **Counterfactual explanations** — explicitly NOT SUPPORTED under the
  frozen-output contract; a future phase could authorize them under the
  recorded criteria (Section 14).
- **Trust/adaptive-risk variants** — only the Phase 8 component definitions
  were rejected; new definitions are untested (HYPOTHESIS space). Phase 19
  CONFIRM evaluation confirmed the adaptive risk approach FAILS for
  production (delta ROC-AUC -0.0281 < -0.01). No further adaptive risk work
  recommended without fundamentally new component definitions.

---

## 23. Current State — What GPT Should Know Before Recommending Anything

### Current objective

Maintain and hand off the verified frozen system (model + policy + overlays);
no ML phase is in progress. The full-scale operational validation is complete —
a complete 501,000-row decision table has been materialized with all frozen
components integrated. Any next step (dashboard deployment, new authorized
phase) requires explicit authorization.
### Current candidate

`lgbm-graph-v1` — LightGBM 4.6.0, 12 features (8 behavioral + 4 graph), seed
42, best_iteration 186, alert threshold 0.9186015432508062, user × day unit,
chronological split (395k/59k/47k; 1,539/323/30 positives).

### What has already been proven (OBSERVED)

- TEST (once): AUC-ROC 0.939157, AUC-PR 0.267776, P@10 0.600, R@50 0.467,
  49 alerts (F1 0.354, MCC 0.365, precision 0.286).
- Graph features complementary (+0.115 AUC-PR) but weak standalone.
- Freeze is bit-reproducible; scores bit-equal across Phases 7–11.
- Conformal: positive coverage 30/30 (Wilson LB 0.917); monitor band enriches
  precision ~11× (CAL) / ~25× (TEST).
- Explainability: exact additive reconstruction (err ~1e-14); shap
  cross-check bit-identical; `usb_connection_count` dominant alert reason.
- Operational envelope (Phase 12): no de-duplication / rolling-threshold
  policy satisfies the recorded R1–R4 rule on CALIBRATION — frozen P0 stands;
  alert capacity at 10,000 users ≈ 41.4/day (INFERENCE).
- Temporal stability (Phase 13): pre-registered CAL verdict FAIL is a
  descriptive risk-register finding — the 3-day tail W8 (03-29..03-31)
  violates B5 hard (median 0.04141) and B4 marginally (precision 0.3333);
  8/8 full 7-day CAL windows pass all bounds; 15-day cross-check clean;
  TEST annex 2/7 consistent; system unchanged (risk, not repair).
- User-level holdout (Phase 14, evaluation-only): 13/15 unseen users
  detected (user-level AUC 0.965); ranking degrades vs chronological
  (AUC-ROC 0.786 vs 0.939; precision 0.202 at the frozen threshold);
  P@10/30/50 = 1.0 on the auxiliary TEST; determinism bit-identical;
  NO production decision; frozen system unchanged.
- Unseen-user diagnosis (Phase 15, descriptive): feature shift small on all
  12 frozen features (|d| ≤ 0.179, PSI ≤ 0.026, KS ≤ 0.047); graph
  features intrinsically weak in both cohorts (row AUC 0.50–0.69); 79.8%
  of unseen-user false positives within 0.10 of the frozen threshold
  (calibration inflation, INFERENCE); the 2 missed users are ranking
  failures far below the threshold (max scores 0.498/0.726); detection is
  incident-length dependent (2–5 d: 3/3, 6–12 d: 3/4, 45–100 d: 7/8);
  coverage improves 13/15 vs 5/5 only at 12–21x alert cost; frozen system
  unchanged.
- Calibration transfer (Phase 17, research-only): Platt scaling fitted on
  the 108-user CAL block transfers to unseen users — TEST Brier 0.00796
  (+92.25%), ECE 0.0026, and recovers the Phase 14 secondary operating
  point exactly (191 alerts, F1 0.4438, precision 0.7749); transfer robust
  under refit variance (bootstrap Brier CI [0.00787, 0.00849]); binning
  transfers Brier/ECE but not its operating point (21,127 alerts);
  ECDF does not calibrate (−216%); mechanical verdict FAIL (rank-based
  AUC not tie-invariant: C −0.0725, D −3.27e-6); calibration cannot fix
  ranking-level misses (JJM0203/WDD0366 zero-alert under every
  conservative set); frozen system unchanged.
- All decisions rest on CALIBRATION evidence; TEST evaluated once.

### What has already been rejected (do not re-suggest without new evidence)

- Adaptive risk (learned and equal-weight mixtures; trust semantics) —
  **confirmed rejected by Phase 19 CONFIRM** (delta ROC-AUC -0.0281 < -0.01;
  verdict FAIL; adaptive risk component definitions rejected; new definitions
  would require a new authorized phase).
- `department_file_type_mismatch_count` and filename-based graph features.
- Graph-only model; 50%-precision operating point; tight percentile and
  high-capacity alert policies.
- Per-user de-duplication (3d/7d) and rolling-calibration-threshold policies
  (Phase 12, under the recorded R1–R4 rule).
- Counterfactual explanations (under the frozen-output contract).
- lr 0.05 training config.
- Recalibration as a mechanism for unseen-user detection (Phase 17,
  mechanical verdict FAIL under the pre-registered §14 rule): arms C
  (binning) and D (ECDF) violate ranking preservation on the discrete
  AUC estimator; per spec §25 no Phase 18 calibration work on this
  mechanism without separate authorization. Calibration (arm B, Platt)
  remains a measured, transferable tool but is NOT adopted as a
  production change — no production decision was authorized.

### Current limitations

- 30-positive TEST; wide CIs; no unseen window in r4.2; window recall drift
  documented; conformal sets rarely decisive; explanation ranks fragile.
- Tail-window temporal degradation documented (Phase 13 verdict FAIL: W8 CAL
  tail median/precision; TEST late-window precision 0) — risk R13-1, not
  repaired; stability beyond 2011-05-17 unmeasured (R13-4).
- User-level holdout measured (Phase 14, evaluation-only): 13/15 unseen
  users detected, but ranking degrades vs the chronological record
  (AUC-ROC 0.786 vs 0.939; precision 0.202 at the frozen threshold) —
  R14-1, documented, no production decision.
- Unseen-user diagnosis (Phase 15, descriptive): alert explosion on unseen
  users = absolute-score calibration inflation near the threshold
  (R15-1); 2 users invisible to ranking (R15-2); graph features
  intrinsically weak (R15-3); coverage/precision trade-off diverges by
  entity (R15-4) — documented, no system change.
- Calibration transfer (Phase 17, research-only): rank-based AUC not
  tie-invariant (R17-1); calibration cannot repair ranking failures
  (R17-2); ECDF is not a calibration (R17-3); no production claim
  (R17-5) — verdict FAIL recorded, no system change.
- End-to-end CONFIRM (Phase 19): adaptive risk A0 vs Family C; verdict
  FAIL (delta ROC-AUC -0.0281 < -0.01; delta PR-AUC -0.0013); adaptive
  risk rejected for production (R19-1); frozen system unchanged.

### Current execution constraints

- Kaggle runtime is ephemeral (recycling wipes the workspace); background
  jobs are not persistent; 30 s default / ~120 s client caps; check kernel
  status before relaunching; sources pushed via gz+b64 and MD5-verified;
  derived tables rebuilt after recycling. PC holds sources/tests/reports;
  raw data never leaves Kaggle.

### Current stop point

Phase 19 complete / STOP (2026-09-11): the approved end-to-end CONFIRM
evaluation executed and verified — adaptive risk A0 vs Family C on the
CONFIRM partition (196 users, 98,196 rows, 245 positives, overlap=0);
verdict FAIL (delta ROC-AUC -0.0281 < -0.01); adaptive risk rejected for
production; frozen system unchanged since Phase 7/9 (151 baselines exact,
incl. one documented self-baseline exclusion); chronological TEST never
re-scored; all scientific history preserved; no ML phase running or
scheduled; spec §25: any new ML phase requires explicit authorization.

### What requires authorization

- Any new ML phase, retraining, retuning, threshold/feature/model change,
  rerun of experiments for exploration, modification of frozen records,
  deletion of history, counterfactual work, new dataset windows, or
  production deployment. Updates to THIS report (Section 25) and the
  knowledge layer are routine maintenance. Phase 19 CONFIRM confirmed
  adaptive risk rejection for production — no further adaptive risk work
  without fundamentally new component definitions and explicit authorization.

---

## 24. Source Index

### Foundation
- [Foundation report](reports/foundation_report.md)
- [Leakage analysis (12 risks + controls)](docs/leakage_analysis.md)
- [Kaggle execution policy](docs/kaggle-execution-policy.md)

### Project Instructions
- [ISM project principles (incl. Section 15 master-report maintenance workflow)](.opencode/instructions/ism-principles.md)

### Phase Reports
- [Baseline report (lgbm-baseline-v1)](reports/baseline_report.md)
- [Phase 3 — behavioral expansion](reports/phase3_report.md)
- [Phase 4 — ablation/robustness/freeze (lgbm-baseline-v2)](reports/phase4_report.md)
- [Phase 5 — graph construction](reports/phase5_graph_report.md)
- [Phase 6 — graph regression comparison](reports/phase6_graph_regression_report.md)
- [Phase 7 — freeze (lgbm-graph-v1)](reports/phase7_freeze_report.md)
- [Phase 8 — adaptive risk (rejected)](reports/phase8_adaptive_risk_report.md)
- [Phase 9 — alert policy freeze](reports/phase9_alert_prioritization_report.md)
- [Phase 10 — conformal layer](reports/phase10_conformal_report.md)
- [Phase 11 — explainability layer](reports/phase11_explainability_report.md)
- [Phase 11 gap analysis (internal)](docs/phase11_gap_analysis.md)
- [Phase 12 — operational envelope](reports/phase12_operational_envelope_report.md)
- [Phase 13 — temporal stability](reports/phase13_temporal_stability_report.md)
- [Phase 13 specification (approved protocol)](docs/phase13_specification.md)
- [Phase 14 — user-level holdout](reports/phase14_user_holdout_report.md)
- [Phase 14 specification (approved protocol)](docs/phase14_specification.md)
- [Phase 15 — unseen-user diagnosis](reports/phase15_unseen_user_diagnosis_report.md)
- [Phase 15 specification (approved protocol)](docs/phase15_specification.md)
- [Phase 16 — final packaging & verification](reports/phase16_final_packaging_report.md)
- [Phase 17 — calibration transfer (research-only, verdict FAIL)](reports/phase17_calibration_transfer_report.md)
- [Phase 17 specification (approved protocol)](docs/phase17_specification.md)
- Phase 18 — Leakage-Safe Adaptive Risk v2: evaluation-only phase. No standalone Phase 18 report artifact was retained after the Kaggle reset. Evidence and final status are summarized in the master report phase history (§5), decision log (§19 decision #24), and adaptive-risk discussion (§20). Verdict: FAIL; no production change.
- [Phase 19 — end-to-end CONFIRM (verdict FAIL)](reports/artifacts/phase19_v1_1_2_confirm_result.json)
- [Phase 19 recovery runner](phase19_v1_1_1/run_phase19_confirm_once_v1_1_2.py)
- [Phase 19 threshold finalization](phase19_v1_1_1/phase19_v1_1_1_threshold_finalization.json)
- [Phase 19 user allocation](phase19_v1_1_1/phase19_user_allocation.json)
- [Phase 20 — final system integration (decision engine)](reports/artifacts/phase20/PHASE20_REPORT.md)
- [Phase 20 decision table](reports/artifacts/phase20/final_user_day_decisions.parquet) — 501,000 user-day decisions
- [Phase 20 decision table (CSV)](reports/artifacts/phase20/final_user_day_decisions.csv)
- [Phase 20 decision table summary](reports/artifacts/phase20/final_user_day_decisions_summary.json)
- [Phase 20 generation script](scripts/phase20_materialize_decisions.py)
- [Phase 20 input inventory](reports/artifacts/phase20/phase20_input_inventory.json)
- [Phase 20 dashboard sample](reports/artifacts/phase20/phase20_dashboard_sample.json)
- [Phase 20 analyst summary](reports/artifacts/phase20/phase20_analyst_summary.md)
- [Phase 20 traceability](reports/artifacts/phase20/phase20_traceability.md)
- [Phase 20 leakage audit](reports/artifacts/phase20/phase20_leakage_audit.md)
- [Phase 20 reproducibility manifest](reports/artifacts/phase20/phase20_reproducibility_manifest.json)
- [Phase 20 archive V2](reports/artifacts/phase20/PHASE20_COMPLETE_ARCHIVE_V2.zip)
- [Project README (packaging deliverable)](README.md)
- [Environment & dependency record](docs/environment.md)
- [Verification-only gate](scripts/verify_project.py)
- [Artifact index](reports/artifacts/ARTIFACT_INDEX.md)

### Knowledge
- [Knowledge base README](knowledge/README.md)
- [Architecture](knowledge/architecture/architecture.md)
- [Dataset facts](knowledge/dataset/cert-r4.2.md)
- [Feature registry](knowledge/features/feature-registry.md)
- [Experiment index](knowledge/experiments/experiment-index.md)
- [Decision log](knowledge/decisions/decision-log.md)
- [Evaluation methodology + records](knowledge/evaluation/README.md)
- [Risks overview](knowledge/risks/README.md)
- [Graph direction](knowledge/graph/README.md)
- [Research policy](knowledge/research/README.md)

### Source Code
- `src/config.py` — feature registry, splits, constants (source of truth)
- `src/experiments/phase7.py` — FROZEN_CONFIG (source of truth for the model)
- `src/experiments/phase9.py` — policy selection rule
- `src/experiments/phase10.py` — conformal method + D1–D5 rule
- `src/experiments/phase11.py` — explanation layer
- `src/experiments/phase12.py` — operational envelope (md5
  `a801ed182828134ab997bcd10bff803f`, local=kernel)
- `src/experiments/phase13.py` — temporal-stability analysis (pre-registered
  bounds, CAL-only verdict, safety gates)
- `src/experiments/phase14.py` — user-level holdout protocol (pre-registered
  allocation, user-disjoint blocks, user-block bootstrap, interpretation
  bands, scenario counts)
- `src/experiments/phase15.py` — unseen-user diagnosis (IO-free,
  deterministic: activity strata, feature shift, detected vs missed, graph
  diagnosis, temporal, threshold diagnostic, cohort comparison)
- `src/experiments/phase17.py` — calibration-transfer protocol (IO-free,
  deterministic: gates 1/1b/2, Platt/binning/ECDF fit+apply, Brier/ECE,
  operating points, coverage/Wilson, strata/scenario, ranking
  verification, user-block + calibration-transfer bootstrap, mechanical
  verdict)
- `kaggle_scripts/run_phase15.py` — Phase 15 local runner (md5-gated
  inputs, gates, double-run determinism)
- `src/experiments/phase20.py` — Phase 20 deterministic decision engine
  (integrates frozen model, threshold, conformal, explainability, graph
  diagnostics into dashboard-ready output)
- `kaggle_scripts/run_phase17.py` — Phase 17 local runner (frozen-input
  loads, gates 3/4 before-after, triple-run determinism, artifacts,
  manifest, cost)
- `phase19_v1_1_1/phase19.py` — Phase 19 core source (MD5
  `6cd6f089e1f7277f3a3b79c52f5ba37f`; static-leakage-audit, user-day
  allocation, CONFIRM evaluation)
- `phase19_v1_1_1/run_phase19_confirm_once_v1_1_2.py` — Phase 19
  recovery runner (MD5 `3ac43b1249b4db933d89646cea62f7fb`; parquet/JSON
  data loading, preflight-only mode, state transition at Step 7)
- `src/data/` (validation, labels, aggregation), `src/preprocessing/splits.py`,
  `src/graph/features.py`, `src/evaluation/` (metrics, threshold),
  `src/models/lightgbm_baseline.py`
- `kaggle_scripts/` — runners (`run_phase*.py`, `rebuild_derived_tables.py`,
  `decode_sources.py`, `pull_from_kaggle.py`, `build_user_day.py`,
  `build_graph_features.py`, `kaggle_exec.py`, `push_to_kaggle.py`)
- `tests/` — full suite (80 passed / 0 skipped at Phase 19; 404 passed / 7 skipped at Phase 17)

### Execution Policy
- [Kaggle execution policy](docs/kaggle-execution-policy.md)
- Journal: `logs/kaggle_execution.jsonl`

### Artifacts
- `reports/artifacts/` — all generated evidence (JSON records, parquet
  predictions/explanations/models, logs). Key files:
  `phase7_freeze_lgbm-graph-v1.json`, `phase7_model_lgbm-graph-v1.txt`,
  `phase7_predictions_lgbm-graph-v1.parquet`, `phase9_freeze.json`,
  `phase9_experiment.json`, `phase10_fit.json`, `phase10_decision.json`,
  `phase11_explanations.parquet`, `phase11_model_integrity.json`,
  `graph_features.parquet`, `validation_report.json`.
  Phase 12: `phase12_{experiment,calibration_policies,deduplication,
  rolling_threshold,capacity_projection,comparison,test_policy_results,
  cost}.json`, `phase12_test_predictions.parquet`, `phase12.log`.
  Phase 13: `phase13_{experiment,window_analysis,verdict,cost}.json`,
  `phase13.log`.
  Phase 14: `phase14_{split,model_record,calibration,test_metrics,
  user_diagnostics,bootstrap,scenario,cost,manifest,experiment}.json`,
  `phase14_model.txt`, `phase14_test_predictions.parquet`,
  `phase14_rebuild_manifest.json` (all md5-verified against the kernel
  manifest; merged input md5 `9a3b1885…9def` == frozen record).
  Phase 15: `phase15_{experiment,activity_strata,feature_shift,
  detected_vs_missed,graph_diagnosis,temporal_analysis,threshold_diagnostic,
  cohort_comparison,manifest,cost}.json`, `phase15_cal_scores.parquet`,
  `phase15.log` (md5-verified via the manifest; local run).
  Phase 16: `final_verification_report.json` (gate output; fresh baseline
  md5s for all artifacts; self-excluded from scans).
  Phase 17: `phase17_{experiment,calibration_fit,calibration_metrics,
  operating_points,ranking_verification,cost,manifest}.json`,
  `phase17.log` (local run; md5-verified via the manifest; gate-4
  self-baseline exclusion documented in `phase17_experiment.json`).
  Phase 19 (in `phase19_v1_1_1/`):
  `phase19_dev_result_run1.json` (MD5 `c981d5250a1d6b09555c8e2a199268fc`;
  selected=C),
  `phase19_dev_result_run2.json` (MD5 `0d45042527662134fa8f68f0165feffe`;
  selected=C),
  `phase19_user_allocation.json` (DEV=588/CONFIRM=196; MD5
  `09521712780275643d46777fbb8c20b6`),
  `phase19_v1_1_1_threshold_finalization.json` (MD5
  `985b11b4cdc3f778e536e9021e93a131`),
  `run_phase19_confirm_once_v1_1_2.py` (recovery runner; MD5
  `3ac43b1249b4db933d89646cea62f7fb`),
  `phase19_v1_1_2_confirm_result.json` (final CONFIRM result; MD5
  `fb835ce71b0596e9988e2e31bd03319d`),
  `phase19_v1_1_2_confirm_report.md` (MD5
  `75a40c1e0802aab0970d54e5691f841e`),
  `phase19_v1_1_2_confirm_manifest.json` (MD5
  `0c8bc8df3d125288ce5510a3366af5f6`),
  `phase19_v1_1_2_confirm_state.json` (opened=true, completed=true),
  `phase19_v1_1_2_confirm_execution.log`,
  `phase19_v1_1_2_confirm_recovery_amendment.json` (available locally;
  not in archive).
  Archive: `PHASE19_COMPLETE_ARCHIVE.zip` (MD5
  `692d5c77f57703ee597b9f9aa7868597`; SHA256
  `8e4cbd73df6af491ceec053ee19ba32970fee9cfcff68f7cf53e12b5f19837a2`),
  `PHASE19_COMPLETE_ARCHIVE.tar.gz` (MD5
  `9ab29208dd5d0d817278c8c3dc615675`; SHA256
  `61cc17a14521a00c0fbedf327df5c733cd571a357318596f35417f5eff453c57`).

---

## 25. Master Report Maintenance Rules

Future OpenCode runs MUST, at the end of every completed phase/run:

1. Read this master report before updating it.
2. Read the new phase report and relevant artifacts; verify numbers.
3. Never delete previous phase history.
4. Never overwrite historical results with new results.
5. Add the new phase to the phase table (Section 5).
6. Update the executive summary (Section 4).
7. Update the frozen-system section (Section 8) only if the production
   candidate legitimately changes.
8. Update the performance table (Section 9).
9. Update decisions/rejections (Sections 19–20) if new evidence exists.
10. Update risks (Section 21).
11. Update resource measurements (Section 18).
12. Update the verification/test history (Section 17).
13. Update the AI handoff (Section 23) and document status (Section 0).
14. Validate all internal links (Section 24).
15. Record the date and latest completed phase (Section 0).
16. Label claims OBSERVED / INFERENCE / HYPOTHESIS / LITERATURE RESULT.
17. Never invent missing metrics — write NOT VERIFIED instead.
18. Never silently reconcile conflicting records — use Section 26.
19. If records conflict, report the conflict and identify the authoritative
    source.
20. Do not modify detailed phase reports to make this report consistent.

---

## 26. Conflict Resolution Rule

If two project files contain different values: (1) identify the conflict;
(2) identify the authoritative source; (3) determine the cause (rerun /
formatting / corrected artifact / stale report / actual change); (4) record
the resolution here.

Source priority: 1. frozen/source-of-truth configuration (`src/config.py`,
`src/experiments/phase7.py` FROZEN_CONFIG) → 2. verified experiment artifact
(`reports/artifacts/*.json`/`.parquet`) → 3. final phase report →
4. knowledge-layer summary → 5. previous chat output. Chat memory is never
experimental evidence when artifacts exist.

**Known conflict records (as of 2026-09-11):**

1. **Phase 8 vs Phase 9 max-F1 threshold bootstrap CI**: Phase 8 reported CI
   [0.0001, 0.953] (mean 0.332); Phase 9 reported CI [0.8978, 0.962] (mean
   0.9204, width 0.0642) for the same estimator. Cause: different tie-break
   semantics in the bootstrap estimator (Phase 8 broke F1 ties toward the
   lowest threshold; Phase 9 mirrors the actual `best_f1_threshold`
   implementation, keeping the higher threshold). Resolution: Phase 9's
   estimator is authoritative (it matches the selection implementation);
   documented in the Phase 9 report §9.
2. **`ism-principles.md` — RESOLVED**: the decision log (#1, #6, #8) and
   research/README reference `.opencode/instructions/ism-principles.md`. At
   report creation the file did not exist (`.opencode/` absent) and was
   recorded NOT VERIFIED as a file. On 2026-08-16 the file was verified to
   exist; its content corroborates the cited claims (raw-data claim appears
   in §1; decision-log citation of "section 9" is approximate). Resolution:
   the referenced file now exists and matches the recorded principles; this
   record is closed, not deleted. A master-report maintenance section (§15)
   was added to the file on the same date.
3. **`knowledge/risks/README.md` stale entries**: its HYPOTHESIS section
   ("whether graph features add signal beyond the 8 behavioral features —
   untested") is outdated: Phase 6/7 measured this (graph features are
   complementary). Resolution: phase reports are authoritative; the
   knowledge note lags and is flagged here, not silently edited.
4. **Knowledge-layer test counts**: `knowledge/evaluation/README.md` and
   experiment-index list per-phase suites; the master report Section 17 uses
   the counts recorded in each phase report and the Phase 11 final local gate
   (214 passed / 4 skipped). No conflict found among authoritative records.
5. **`phase12_calibration_policies.json` first transcription — RESOLVED**:
   an early b64 transcription of the kernel artifact produced a different md5
   (`c81dcae2…`) than the kernel manifest (`58d24624…`). Cause: corrupted
   transcription. Resolution: the artifact was refetched cleanly and the
   verified copy (md5 `58d24624a0942c8d911822f0fcc11fc6`) is authoritative;
   all 10 Phase 12 artifacts were md5-verified against the kernel manifest
   before being accepted.
6. **`test_frozen_threshold_matches_phase9_record` wrong key — RESOLVED**:
   the test asserted `rec["threshold_max_f1"]`, but the authoritative
   `reports/artifacts/phase9_freeze.json` stores the threshold as
   `policy.param` (`0.9186015432508062`). Cause: stale test written against a
   guessed schema; it had always been skipped locally (artifacts not staged).
   Resolution: test corrected to `rec["policy"]["param"] ==
   p12.FROZEN_THRESHOLD`; now runs and passes; test_phase9.py schema matches
   the artifact.
7. **`phase14_scenario.json` `n_users` field — DOCUMENTED (not a numeric
   conflict)**: the field named `n_users` (per scenario per split) actually
   holds the number of user-day **rows** in the stratum (TEST 2004/4008/
   1503 = 4/8/3 users × 501), while `phase14_user_diagnostics.json`
   `strata.by_scenario` reports true user counts (4/8/3). Cause:
   `src/experiments/phase14.py` `_scenario_counts`/`_scenario_recall`
   compute `n_users = mask.sum()` over the per-row array. `n_malicious_rows`
   (30/440/6; sum 476 = TEST positives) and every recall value are correct
   (8+196+3 = 207 = TP at primary). Resolution: units reconciled
   mathematically (row counts ÷ 501 = user counts; strata identical);
   artifact NOT regenerated (TEST-once + artifact immutability);
   authoritative source: the artifacts themselves, consistent with each
   other; a future run should rename the field to `n_rows`.
8. **Stale documentation, recorded not rewritten (Phase 16)**: (a)
   `reports/phase15_unseen_user_diagnosis_report.md` §2 states "Runtime
   9.5 s" (pre-manifest-fix run); the final verified run is 6.02 s (this
   report §6/§18; phase15_experiment.json). Cause: the report predates the
   legitimate runner fix; it was NOT edited (history preservation). (b)
   `knowledge/graph/README.md` ("graph phase is the next major development
   stage… no graph model"), `knowledge/features/feature-registry.md`
   ("experimental, NOT in the model") and `knowledge/architecture/
   architecture.md` (graph/adaptive-risk marked PLANNED) lag Phases 6–8
   (graph features are frozen inside `lgbm-graph-v1`; adaptive risk
   REJECTED). Cause: knowledge layer not refreshed after Phases 7–8.
   Resolution: authoritative sources are `src/config.py`,
   `reports/phase7_freeze_report.md`, `reports/phase8_adaptive_risk_report.md`
   and this report §10/§11; the stale notes are flagged here, not silently
   edited (per §25 history preservation and the Phase 16 authorization
   "update only the necessary knowledge records").
9. **`final_verification_report.json` self-baseline — RESOLVED by
   authorized exclusion (Phase 17)**: the gate-4 before/after baseline of
   Phase 16 flagged one mismatch: the verification report's own recorded
   self-md5 (84e10ac1…) never equals the written file (c3c31f20…) because
   `scripts/verify_project.py` hashes every artifact (including its own
   output, data_checks lines 447–450) BEFORE writing the report (line
   230). Cause: by-construction stale self-baseline, not a modification.
   Resolution: project owner authorized (2026-08-19) "Exclude self-baseline,
   proceed"; the exclusion is recorded in the gate record
   (`phase17_experiment.json` gate 4: `{"excluded":
   {"final_verification_report.json": "by-construction stale
   self-baseline…"}}`); all other 151 baselines verified exact before AND
   after the Phase 17 run. The exclusion applies to future gate runs too;
   `final_verification_report.json` remains the authoritative verification
   record.
10. **Spec §10.1/§12.2 "arm C preserves AUC-ROC" claim — REFUTED by
   measurement (Phase 17)**: the approved Phase 17 spec stated arm C
   (binning) preserves row order so "AUC-ROC is unchanged". Measured:
   ΔAUC-ROC −0.07249 (and arm D −3.27e-6). Cause (OBSERVED): the empirical
   average-rank `roc_auc_score` estimator is not invariant under tie
   creation even when row order is preserved (C: 3 distinct p̂ values on
   TEST; D: 1,731 tie collisions across CAL-score gaps). Resolution:
   artifacts and the phase report are authoritative; the pre-registered
   claim is recorded as refuted (§17 phase report §5, R17-1); the
   mechanical verdict FAIL is applied per the pre-registered §14 rule.
   Future pre-registrations must bound tie creation explicitly.

---

## 27. Validation Requirements

After creating/updating this report, verify: file exists; Markdown parses and
is readable; all sections exist (0, 4–27); all phase reports are represented;
the latest phase is represented; no duplicate/conflicting phase entries; all
numerical values match authoritative sources; all internal links resolve; no
unsupported claims; no raw dataset included; no Kaggle notebook dump
included; no credentials/secrets; no sensitive runtime URLs/tokens (kernel
URLs and tokens appear in foundation/phase logs only where they were part of
the environment record — they are deliberately not reproduced here).

## 28. Size Constraint

Target ≈ 10–20 pages of Markdown. Comprehensive context in one file; do NOT
duplicate phase reports. Use tables, concise summaries, and links. If it
grows past ~20 pages, prune the phase-by-phase sections toward the phase
table + links.

## 29. Future Run Behavior

After EVERY future completed phase/experiment/major authorized milestone:
read the new phase report → verify results against artifacts → update this
report (Sections 0, 4–24 as applicable) → add the phase to the cumulative
history → update system state/performance/decisions/risks/resources/handoff →
validate links → run the Section 27 verification checks → STOP. Do not wait
until project end. The master report always represents the latest VERIFIED
project state.

## 30. Final Deliverable

This file (`reports/ISM_MASTER_REPORT.md`) is the deliverable. Completion
summary of the run that created/updated it is provided to the user after
verification; no ML phase is started by this task.

---

## 31. Final Project Consolidation (2026-09-12)

### Overview

All consolidation documents created in `reports/final/`:

| Document | Purpose |
|---|---|
| `END_TO_END_ABLATION_SUMMARY.md` | Two-table ablation evidence (direct-comparable + incremental layers) |
| `END_TO_END_ABLATION_SUMMARY.csv` | Machine-readable ablation data |
| `END_TO_END_ABLATION_SUMMARY.json` | Structured ablation data with all exact values |
| `CALIBRATION_CONFORMAL_CONSOLIDATION.md` | Conformal layer (accepted) + calibration transfer (research-only, FAIL) |
| `EXPLAINABILITY_DECISION_CONSOLIDATION.md` | SHAP validation, full 21-column decision table, layer stack |
| `FINAL_REPRODUCIBILITY_PACKAGE.md` | Experiment ledger, frozen config, environment, artifact hashes, tests |
| `final_reproducibility_manifest.json` | Machine-readable reproducibility manifest |
| `FINAL_PROJECT_CONCLUSIONS.md` | Retained/rejected components, architecture, quality, limitations |

### Ablation Summary (Table 1 — Directly Comparable)

| Entry | Model | AUC-ROC | AUC-PR | F1 | MCC | Alerts | Verdict |
|---|---|---|---|---|---|---|---|
| A | lgbm-baseline-v2 (8f) | 0.93534 | 0.15304 | 0.346 | 0.350 | 22 | FREEZE |
| B | Graph-only (5f) | 0.78635 | 0.01431 | 0.054 | 0.105 | 417 | WEAK |
| C | lgbm-graph-v1 (12f) | 0.93916 | 0.26778 | 0.354 | 0.365 | 49 | FREEZE CANDIDATE |
| D1 | Adaptive Risk (equal-weight) | 0.90977 | 0.06720 | 0.105 | 0.170 | 256 | REJECTED |

*(See [Figure 4](figures/fig4_ablation_comparison.png) for a visual
comparison of the ablation arms.)*

### Ablation Summary (Table 2 — Incremental System Layers)

| Entry | Layer | Evidence | Verdict |
|---|---|---|---|
| C | Frozen classifier | Chronological TEST | FREEZE |
| D2 | Phase 19 CONFIRM | Separate user-disjoint cohort | FAIL |
| E | Conformal confidence | CAL-only fit, TEST evaluation | ACCEPT |
| F | Full decision table | 501K materialization | COMPLETE |

*(See [Figure 6](figures/fig6_decision_distribution.png) for the Phase 20
decision distribution and [Figure 7](figures/fig7_runtime_modelsize.png)
for computational cost comparison.)*

### Key Artifacts

| Artifact | MD5 |
|---|---|
| Phase 20 decision table (parquet) | 6476f791d1cc9f21327a94e1373f9e09 |
| Phase 20 archive V2 | 2fd2fb1f9320be545818e7f0355b6bfa |
| Phase 19 archive | 692d5c77f57703ee597b9f9aa7868597 |
| Master report backup | ISM_MASTER_REPORT_BACKUP_2026-09-12.md |

### Consolidation Decisions

1. **Adaptive Risk**: CONFIRM FAIL (delta ROC-AUC -0.0281 < -0.01). Rejected for production. No further work without fundamentally new component definitions.
2. **Conformal**: ACCEPTED as diagnostic overlay. Pos coverage 1.000, Wilson LB 0.917.
3. **Calibration**: RESEARCH-ONLY (FAIL). Platt transfers but ranking not invariant. No Phase 18 work without authorization.
4. **Decision table**: Complete. 501K rows, 21 columns, 54/54 tests, determinism PASS.

### Status

All documentation complete. No ML phase running or scheduled. Frozen system unchanged since Phase 7/9. Any new authorized phase requires explicit authorization.

---

*End of master report. Detailed phase reports remain the authoritative
sources; this report is the cumulative, verified summary.*