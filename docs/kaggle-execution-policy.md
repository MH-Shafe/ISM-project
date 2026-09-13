# Kaggle Execution Policy

Operational policy for executing project code on Kaggle through OpenCode +
Jupyter MCP. Applies to the verified environment (2026-08-16):

- PC: authoritative source, tests, docs. OpenCode + Jupyter MCP (opencode.json).
- Kaggle: execution only. Kernel: Python 3.12.13, 4 CPU, ~33.7 GB RAM, Tesla T4.
- Dataset (read-only): `/kaggle/input/datasets/andrihjonior/cert-insider-threat-dataset-r4-2`.
- Workspace (ephemeral): `/kaggle/working` — wiped on runtime recycling.
- Raw CERT data is never copied to the PC.

All behavioral statements below are OBSERVED (verified experimentally
2026-08-16) unless marked otherwise.

---

## 0. Phase execution workflow (permanent, adopted 2026-08-20)

The PC repository is the authoritative source for code, specifications,
reports, tests, configurations, historical artifacts and experiment
definitions. Kaggle is the compute/execution environment only. During
normal phase execution there is NO slow bulk per-file synchronization
between PC and Kaggle: the transport unit is one phase bundle ZIP in and
one results bundle ZIP out.

### 0.1 Bundle construction (PC side)

- One phase = exactly one execution package: `phase<PHASE>_kaggle_bundle.zip`.
- Contents: `src/`, `tests/`, `kaggle_scripts/`, `configs/`, `docs/`
  (with the phase specification), `reports/` (only required reference
  reports), `artifacts/` (only required frozen previous-phase artifacts),
  `knowledge/`, `requirements.txt`, `run_phase<PHASE>.py`,
  `phase<PHASE>_bundle_manifest.json`, `phase<PHASE>_bundle_metadata.json`.
- Excludes: raw CERT r4.2 (already available as a Kaggle dataset/input),
  secrets, unnecessary caches, unrelated historical files.
- Manifest: for every included file `relative_path`, `size_bytes`, `md5`.
- Metadata: `phase`, `bundle filename`, `bundle size`, `bundle MD5`,
  `creation timestamp`, `relevant specification`, `file count`.
- Before handoff: re-open the ZIP, verify every file against the manifest,
  verify ZIP size and MD5, run package validation, run applicable local
  tests. Only then report "BUNDLE READY — UPLOAD TO /kaggle/working/"
  and STOP.

### 0.2 Upload and verification (Kaggle side)

- The user places the bundle at `/kaggle/working/phase<PHASE>_kaggle_bundle.zip`
  (preferred, authoritative runtime location). Not a Kaggle Dataset, not
  under `/kaggle/input`. Raw CERT r4.2 may remain under `/kaggle/input/`.
- On "THE PHASE <PHASE> BUNDLE IS UPLOADED": verify filename, size, MD5
  against `phase<PHASE>_bundle_metadata.json` (or the authoritative values
  supplied at creation). Missing or incorrect → STOP, do not execute.
- Extract to `/kaggle/working/ism_project/`; then locate the manifest,
  verify every internal file (size + md5), verify the directory structure.
  Log "PHASE <PHASE> BUNDLE VERIFIED" only after all required files pass.
- Keep the ZIP at `/kaggle/working/phase<PHASE>_kaggle_bundle.zip` until
  the phase finishes (allows re-extraction if the workspace is corrupted).
  Do not delete it just after extraction.

### 0.3 Data and previous phases

- CERT r4.2: detect and structurally verify under `/kaggle/input/` when
  raw data is required. Dataset filename alone is NOT proof of version.
  Never silently substitute another CERT release or another dataset.
- Previous phases are consumed as frozen, verified artifacts. Never rerun
  earlier phases. For every previous-phase artifact used: locate → hash →
  size → schema/shape → manifest comparison → load. Missing or mismatched
  → STOP and report exactly what failed.
- Safe rebuild exception: deterministic preprocessing tables may be
  rebuilt only when the current specification permits it, with the correct
  frozen code/config, with no TEST/future leakage, and verified against
  expected shape/hash anchors. This is NOT permission to rerun previous
  experimental evaluations.

### 0.4 Execution mechanics

- CPU by default; GPU only if the approved phase genuinely requires it.
  Record Python version, package versions, CPU info, relevant environment
  variables, seeds.
- Runtime state under `/kaggle/working/`: `phase<PHASE>_live.log`,
  `phase<PHASE>_status.json`, `phase<PHASE>_checkpoints/`.
- Status file fields: `phase`, `run_id`, `stage`, `stage_name`, `status`,
  `started_at`, `updated_at`, `completed_at`, `test_opened`,
  `test_completed`, `last_checkpoint`, `last_error`, `hard_gates_passed`,
  `artifacts_created`. Statuses: INITIALIZING / RUNNING / PASS / FAILED /
  WAITING / COMPLETE.
- Live log: timestamps, stage start/end, actions, elapsed time, gate
  outcomes, row counts, warnings, experiment progress, bootstrap progress,
  tracebacks on failure. Flush frequently.
- Checkpoint after every expensive or scientifically important stage;
  never mark an incomplete stage as PASS.

### 0.5 Scientific discipline (unchanged by this workflow)

- Leakage: nothing that would not be available at prediction time;
  no statistics, baselines, graph stats, normalizers, thresholds, weights,
  model/calibration parameters or operating thresholds computed from
  TEST/future data. Chronological TRAIN → CAL → TEST by default.
- Analytical unit: user × day unless the approved spec defines another.
- TEST discipline: code/features/weights/calibration/thresholds/metrics/
  acceptance criteria frozen and pre-TEST gates passed before any
  authorized TEST access; `pretest_freeze.json` written first;
  `test_opened = true` immediately before first TEST access;
  `test_completed = true` + `test_complete.json` only after a successful
  evaluation. Never rerun TEST because a later stage fails.
- Failure before TEST (`test_opened == false`): capture traceback, update
  status, preserve checkpoints, diagnose, fix authoritative PC code, run
  local tests, rebuild ZIP + manifest, ask for upload. Clean rerun
  permitted if scientifically safe.
- Failure during TEST: STOP, no automatic retry, report exactly what TEST
  information was accessed and whether predictions/metrics were produced,
  wait for explicit authorization.
- Failure after TEST (`test_completed == true`): never rerun the phase,
  never evaluate TEST again; resume only downstream stages from saved
  checkpoints; preserve `test_complete.json`.
- Run tests (unit, artifact integrity, leakage, split/entity integrity,
  hard gates) before scientific execution; a hard-gate failure → STOP,
  never weaken the gate after seeing the failure.
- Execute exactly the approved spec: no unregistered arms, no metric/
  threshold/weight changes after seeing results, no TEST-based tuning,
  no silent model substitution, no improvement claim without evidence.
- Long runs: small timing probe first (e.g., 10 bootstrap repetitions)
  for runtime estimation only, then the full registered count with
  progress lines. Do not restart a healthy run because it looks slow.
- Results: classify OBSERVED / INFERENCE / HYPOTHESIS / LITERATURE RESULT;
  use the phase's pre-registered PASS/FAIL criteria exactly; a promising
  sub-arm in a failed phase is reported as such, never converted to PASS.

### 0.6 Outputs and preservation

- Generate only spec-authorized outputs; never overwrite historical
  artifacts. Create `phase<PHASE>_artifact_manifest.json` (size/hash per
  artifact) and verify every generated file against it.
- Results bundle: `phase<PHASE>_results_bundle.zip` — generated artifacts,
  experiment manifest, metrics/results, status/log, checkpoint summaries,
  phase report, authorized knowledge/digest updates, tests/verification
  summary. No raw CERT data. Report ZIP size, ZIP MD5, file count.
- Preserve results back to the PC before intentionally ending/resetting
  the Kaggle runtime; the PC copy becomes authoritative after integrity
  verification (bundle MD5, internal manifest, artifact hashes).
- Kaggle reset recovery: before TEST → re-upload the same verified ZIP,
  recover from preserved checkpoints; after TEST completion → protect
  TEST-once discipline, use preserved `test_complete`/results, never
  rerun TEST.
- Phase completion report: Problem → Evidence → What was executed →
  Gates → Results → PASS/FAIL → Tests → Leakage checks → Runtime →
  Artifacts → Bundle MD5 → Remaining risks → Recommended next phase.

### 0.7 Authority and conflicts

This workflow changes execution mechanics only; it does NOT weaken the
research methodology. Priority: correctness, leakage safety,
reproducibility, valid evaluation, interpretability, compute efficiency,
implementation simplicity. Every major component must earn its place
through evidence. If this workflow conflicts with a scientific requirement
of the current approved specification, preserve the scientific
specification, STOP, and report the conflict.

---

## 1. Failure classes and verified behavior

### A. Normal long-running job (>30 s)

- Caller MUST pass an explicit `timeout` to `execute_code`.
- Verified: 60 s job with `timeout=90` completes; 135 s job with `timeout=150`
  keeps running; heartbeats stream back.
- The default ~30 s timeout is an infrastructure limit, NOT a project failure.

### B. MCP timeout semantics (two distinct regimes — do not confuse them)

1. **MCP-server default (30 s)** — the Kaggle execution IS interrupted and
   killed. Verified: a 60-iteration 1 s-tick loop stopped at tick 29 when the
   call hit the 30 s default.
2. **MCP-client cap (~120 s, opencode.json `timeout: 120000`)** — only the
   waiting call is aborted; the Kaggle execution CONTINUES in the background.
   Verified: a 150 s loop wrote 148 ticks after the call returned empty.
   The kernel is busy until the job finishes; subsequent calls queue.

Implication: interactive calls are bounded at ~120 s. Do not treat an empty
return after >120 s as failure — poll for the expected artifact instead.
Before launching any new job, check that the kernel is idle and the previous
job actually finished (poll the job log / status marker). Never relaunch a
training or experiment job speculatively after an aborted call: a background
process may still be running, so verify it first (jobs write a start marker
with PID) to prevent duplicate concurrent training.

### C. Temporary kernel recovery

- A dead kernel: calls hang, then return `[TIMEOUT ERROR ... was interrupted]`.
- Verified: after SIGKILL, the Jupyter server respawns the kernel under the
  SAME kernel ID in ~35 s; a retried health probe succeeds shortly after.
- Rule: on a timeout/hang, wait ~10 s, probe, retry up to 3 times before any
  restart action. Do not recreate/restart everything immediately.
- Note: child processes of the dead kernel do NOT survive (verified).

### D. Runtime recycling (workspace replacement)

Signals: `/kaggle/working` is wiped (no previously pushed files), or the
kernel ID in `KAGGLE_RUNTIME_ID.txt` no longer matches the live kernel.

Recovery runbook:
1. Report: "Kaggle runtime was recycled" (do not classify as code failure).
2. Refresh runtime config via the approved mechanism
   (`get-kaggle-runtime.ps1`/`.bat` with a fresh `KAGGLE_URL`, or manual
   update of `.env.kaggle` + `KAGGLE_RUNTIME_ID.txt`/`URL.txt`).
3. Restart the OpenCode session so the MCP server reads the new
   `KAGGLE_RUNTIME_URL`/`KAGGLE_RUNTIME_ID` environment variables.
4. Re-push project source from PC (gz+b64 push; kernel side MUST decode with
   `gzip.decompress` — verified byte-identical).
5. Rebuild derived artifacts on Kaggle (they do not survive recycling).
6. Never attempt to recover raw CERT data into the PC.

Recovery incident 2026-08-19 (kernel `8326d7d7`, OBSERVED):
- **Contents-API 500/502 under load**: bulk file PUTs triggered transient
  HTTP 500/502 (rate limiting). Retry with backoff in small batches; retried
  requests succeeded.
- **Directory persistence**: a dir PUT may return 201 yet the dir is not
  visible immediately (eventual consistency). Create parent dirs one level at
  a time and verify existence (GET) before uploading files; do not rely on a
  single recursive mkdir call.
- **Jupytext wrapping**: GET of a pushed `.py` file returns a jupytext
  notebook (`type: notebook`, `format: json`), NOT the raw blob text — the
  file on disk is the blob and is correct. PC-side GET verification of `.py`
  files is therefore invalid; authoritative verification is kernel-side
  disk-bytes md5 via `decode_sources.py`.
- **PC-side verification rule**: after a push, decode and md5-verify ON the
  kernel (byte-identical vs the local manifest). Do not re-verify `.py` files
  by reading them back through the contents API.
- **MCP staleness**: after a runtime recycle + session restart, Jupyter MCP
  tools can still point at the old runtime (404/empty results). Workaround:
  direct contents-API calls + a websocket kernel executor
  (`kaggle_exec.py`); restarting the OpenCode/MCP session restores MCP tools.
- Recovery outcome (OBSERVED): 274 files pushed, 273-entry manifest
  (self-excluded), decode 273/273 byte-identical; derived tables rebuilt with
  merged md5 `9a3b1885…9def` matching the frozen record; phase14/15/17
  manifests md5-verified (10/10, 10/10, 7/7); pytest 411 passed on kernel;
  `verify_project.py` 35/35 on kernel + 2 expected failures for PC-only files
  (`opencode.json`, `smoke_test.py`, `start-opencode.bat`,
  `get-kaggle-runtime.*`, `.env.kaggle` — intentionally never pushed).

### E. Code failure (real traceback)

- A Python traceback returned by `execute_code` is a CODE failure.
- Preserve the traceback verbatim; report it as code failure; do NOT retry
  indefinitely; do NOT relabel it as infrastructure failure.

### F. Test failure (pytest)

- Run pytest locally on the PC; report failures separately from Kaggle
  infrastructure status; never hide failures behind automatic retries.

---

## 2. Timeout policy (verified values)

| Job class                      | `execute_code` timeout | Notes                                   |
|--------------------------------|------------------------|-----------------------------------------|
| Health probe / smoke test      | 10–15 s                | trivial cells only                      |
| Short job (preprocessing step) | 60 s                   | heartbeat every 15–30 s if streaming    |
| Experiment step                | 90–120 s               | hard interactive ceiling is ~120 s      |
| Long training / heavy job      | NOT interactive        | background subprocess + log polling     |

Background mechanism (verified): launch with
`subprocess.Popen("nohup sh -c '...' > /kaggle/working/job.log 2>&1 &", shell=True)`,
return immediately, then poll the log/status file with short calls.
Background jobs survive MCP-call aborts but NOT kernel death (verified) and
NOT runtime recycling.

---

## 3. Heartbeat / progress format

For any job likely to exceed 30 s, emit `[JOB]`-prefixed lines with
`flush=True` (streamed when the tool supports it):

```text
[JOB] started
[JOB] elapsed=30s
[JOB] elapsed=60s
[JOB] stage=training
[JOB] stage=validation
[JOB] completed
```

Rules:
- one line every ~30 s (or per stage change), timestamped where useful;
- final line MUST be `[JOB] completed` (or `[JOB] failed <reason>`);
- background jobs write the same format to their log file for polling.

---

## 4. Standard snippets

### Health probe (kernel side, ≤15 s)

```python
import os, time
print("KAGGLE_SMOKE_TEST", os.getcwd(), os.path.isdir("/kaggle/input"))
print("uptime_s", round(float(open("/proc/uptime").read().split()[0]), 1))
```

### Heartbeat wrapper (kernel side)

```python
import time
t0 = time.time()
print(f"[JOB] started", flush=True)
try:
    ...  # work
    print(f"[JOB] elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[JOB] completed total={time.time()-t0:.1f}s", flush=True)
except Exception as e:
    print(f"[JOB] failed {type(e).__name__}: {e}", flush=True)
    raise
```

### Background job (kernel side, for >120 s work)

```python
import subprocess
subprocess.Popen("nohup sh -c 'python /kaggle/working/job.py > /kaggle/working/job.log 2>&1' &", shell=True)
# poll: read /kaggle/working/job.log and look for "[JOB] completed"
```

---

## 5. Outcome journaling (PC side)

Record every noteworthy outcome so failures are attributable:

```
python kaggle_scripts/kaggle_exec.py health          # config consistency check
python kaggle_scripts/kaggle_exec.py log probe ok "kernel alive" --elapsed 1.2
python kaggle_scripts/kaggle_exec.py log job timeout "no explicit timeout" --elapsed 30.0
python kaggle_scripts/kaggle_exec.py log push ok "phase11.py restored" --elapsed 4.1
python kaggle_scripts/kaggle_exec.py status          # recent journal
```

Journal file: `logs/kaggle_execution.jsonl`.

---

## 6. Classification quick reference

| Symptom                                        | Class                    | Action                          |
|------------------------------------------------|--------------------------|---------------------------------|
| `[TIMEOUT ERROR ... interrupted]` (≤30 s)      | timeout/interrupted     | retry with explicit timeout     |
| empty return after >120 s                      | busy (job still running)| poll for artifact               |
| hang → timeout → probe succeeds (~35 s later)  | kernel_dead (recovered)  | continue after probe            |
| `/kaggle/working` wiped / kernel ID changed    | runtime_recycled         | runbook D                       |
| traceback in cell output                       | code_error               | fix code, preserve traceback    |
| pytest failures                                | test_failure             | fix tests, report separately    |