# Feature Registry (Overview)

Authoritative registry: `src/config.py` (`BEHAVIORAL_FEATURES`, `GRAPH_FEATURES`,
`REJECTED_GRAPH_FEATURES`). This note is a navigable overview; the code is the source
of truth for definitions.

## Behavioral features (all status `defined`)

All are per-user-day aggregates (day-local), except `unusual_access_count` (strictly
past 28-day window). None use future or label information (leakage-safe; verified by
`tests/test_aggregation.py`, `tests/test_leakage.py`).

| Feature | Source | Definition (concise) | Temporal availability | Leakage status | In frozen baseline |
|---|---|---|---|---|---|
| `login_count` | logon.csv | Logon rows per user-day | end of day | safe (day-local) | yes |
| `after_hours_login_count` | logon.csv | Logon rows per user-day with hour < 8 or >= 18 | end of day | safe (day-local) | yes |
| `usb_connection_count` | device.csv | Connect rows per user-day (removable media) | end of day | safe (day-local) | yes |
| `file_access_count` | file.csv | File rows per user-day (copy to removable media) | end of day | safe (day-local) | yes |
| `sensitive_file_access_count` | file.csv | File rows per user-day with declared sensitive magic-byte prefix (OLE2 `D0-CF-11-E0-A1-B1-1A-E1`, PDF `25-50-44-46-2D`, ZIP `50-4B-03-04`) | end of day | safe (a-priori event-time rule; never answer-key derived) | yes |
| `http_activity_count` | http.csv | Http rows per user-day | end of day | safe (day-local) | yes |
| `unique_device_count` | logon.csv | Distinct PCs with logon rows per user-day | end of day | safe (day-local) | yes |
| `unusual_access_count` | logon.csv | Logon events per user-day on a PC with no logon by that user in trailing 28 days (`[day-28, day-1]`, `UNUSUAL_ACCESS_WINDOW_DAYS=28`); first-ever use counts | end of day (window strictly past) | safe (future-independence test; same-day excluded) | yes |

All 8 are part of the frozen behavioral baseline **lgbm-baseline-v2** (Phase 4
freeze; see [[../reports/phase4_report]]). `model_features()` enforces the exact
registry list — extra/missing columns raise.

## Graph features (experimental, NOT in the model)

Built in Phase 5, `artifacts/graph_features.parquet` (501,000 x 7). Experimental
only — never merged into the behavioral model; no accuracy claim.

| Feature | Source | Definition (concise) | Temporal availability | Leakage status |
|---|---|---|---|---|
| `device_consistency_score` | logon.csv | Jaccard(day PCs, strictly-past PCs) | end of day d | safe (strictly past; first-use-day < d) |
| `rare_device_usage_count` | logon.csv | Day PCs never used strictly before d | end of day d | safe |
| `file_type_consistency_score` | file.csv | Jaccard(day content types, strictly-past types); types = OLE2/PDF/ZIP/other | end of day d | safe (filenames globally unique, so types are the unit) |
| `rare_file_type_access_count` | file.csv | File rows on d whose type was never accessed strictly before d | end of day d | safe |
| `department_file_type_mismatch_count` | file.csv + LDAP | Rows on d whose type is not in the department's strictly-past types; department from latest LDAP snapshot strictly before month(d) | end of day d | safe (cross-user within department; context signal only) |

Rejected/deferred (machine-readable reasons in `REJECTED_GRAPH_FEATURES`):
`file_access_consistency_score`, `rare_file_access` (degenerate: all filenames
globally unique), `url_consistency_score` (deferred: 14.5 GB http scan, unproven
signal).

## Explainability notes (Phase 11, on the frozen `lgbm-graph-v1`)

Margin-space contributions from LightGBM `pred_contrib` (see
[[../reports/phase11_explainability_report]]). Observed on CALIBRATION/TEST,
descriptive only:

- `http_activity_count` is **bipolar**: ~49% of total |contribution| but its
  signed effect is ≈ 0 on alert rows (raises risk for some rows, lowers it for
  others); never summarize it with a single sign.
- `usb_connection_count` is the dominant alert driver (top-1 positive
  contribution in 50% of CALIBRATION alerts, 36/49 of TEST alerts).
- `device_consistency_score` is the top graph contributor (top-3 in 72% of
  alerts; rank 2 overall by |contribution| share 14%).
- `rare_file_type_access_count` is near-inert (0.07% share, never a top-3
  alert reason).
- Contributions are attributions of the frozen model's score, not causal
  claims; reason templates omit causal verbs (test-enforced).

## Related

- [[dataset/cert-r4.2]] — source logs
- [[experiments/experiment-index]] — how features were measured
- [[graph/README]] — planned graph direction
- [[decisions/decision-log]] — feature-related decisions