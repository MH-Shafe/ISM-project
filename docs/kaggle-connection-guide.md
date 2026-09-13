# Kaggle Connection Guide — Full Setup for a New Project

Step-by-step guide to connect your PC to a Kaggle kernel for remote code
execution. This covers authentication, runtime discovery, OpenCode MCP
integration, file push/pull, and recovery after runtime recycling.

---

## 1. Prerequisites

### On Kaggle
- A Kaggle account with a running **Notebook** (not a Dataset or Competition).
- The notebook must be in **Edit** mode (not View-only) with a running
  Python 3 kernel.
- Optionally add the CERT r4.2 dataset via "Add Data" so it appears at
  `/kaggle/input/datasets/...`.

### On Your PC
- Python 3.11+ installed.
- `jupyter-mcp-server` installed globally:
  ```bash
  pip install jupyter-mcp-server
  ```
- PowerShell 5.1+ (built into Windows 10/11).
- The project repository cloned locally.

---

## 2. Understanding the Architecture

This project does **NOT** use the Kaggle CLI or `kagglehub`. Instead it
connects directly to the Kaggle kernel's **Jupyter REST API** via a
temporary proxy URL provided by Kaggle.

```
PC (OpenCode)                         Kaggle Kernel
┌──────────────┐    Jupyter API      ┌──────────────────────┐
│ opencode     │───PUT /api/contents/──▶  /kaggle/working/  │
│ jupyter-mcp  │◀──GET /api/contents/──  (ephemeral workspace)│
│ push/pull.py │                      │  /kaggle/input/      │
└──────────────┘                      │  (read-only dataset) │
                                      └──────────────────────┘
```

### Three credential files (never committed to git)
| File | Purpose |
|------|---------|
| `.env.kaggle` | `KAGGLE_RUNTIME_URL` and `KAGGLE_RUNTIME_ID` as env vars |
| `KAGGLE_RUNTIME_URL.txt` | The full Jupyter proxy URL (used by push/pull scripts) |
| `KAGGLE_RUNTIME_ID.txt` | The Python 3 kernel UUID (used by MCP + health checks) |

---

## 3. Start a Kaggle Notebook Session

1. Go to https://www.kaggle.com → Code → New Notebook.
2. In the notebook editor, click **"Enable Edit Mode"** if not already.
3. Click the **three-dot menu** → **"Connect to VS Code"** or
   **"Copy Jupyter Server URL"**. You need the URL that looks like:
   ```
   https://www.kaggle.com/api/v1/kernels/<something>/jupyter/
   ```
   Alternatively: **Settings** → **Jupyter Server** → copy the URL shown.

4. Keep this URL — it changes every time you restart the session.

---

## 4. Run the Runtime Discovery Script

This script takes the Kaggle URL, queries the kernel API, finds the
Python 3 kernel, and writes the three credential files.

### Option A: Interactive Batch File (Recommended for First-Time Setup)

```cmd
get-kaggle-runtime.bat
```

- Prompts for the Kaggle URL.
- Calls the PowerShell script.
- Creates `.env.kaggle`, `KAGGLE_RUNTIME_ID.txt`, `KAGGLE_RUNTIME_URL.txt`.

### Option B: PowerShell Directly

```powershell
$env:KAGGLE_URL = "https://www.kaggle.com/api/v1/kernels/YOUR_KERNEL_ID/jupyter/"
.\get-kaggle-runtime.ps1
```

### Option C: Manual File Creation

If the scripts don't run, create the three files manually:

**`.env.kaggle`**
```
KAGGLE_RUNTIME_URL=https://www.kaggle.com/api/v1/kernels/YOUR_KERNEL_ID/jupyter/
KAGGLE_RUNTIME_ID=YOUR_KERNEL_UUID
```

**`KAGGLE_RUNTIME_URL.txt`**
```
https://www.kaggle.com/api/v1/kernels/YOUR_KERNEL_ID/jupyter/
```

**`KAGGLE_RUNTIME_ID.txt`**
```
YOUR_KERNEL_UUID
```

Replace `YOUR_KERNEL_UUID` with the UUID portion of the URL (the part
after `/kernels/` and before `/jupyter/`).

### Verify

```cmd
python kaggle_scripts\kaggle_exec.py health
```

This checks that the three files are consistent and the kernel is alive.

---

## 5. Configure OpenCode MCP Integration

Edit `opencode.json` at the project root. The MCP server definition:

```json
{
  "mcp": {
    "kaggle": {
      "type": "local",
      "command": [
        "jupyter-mcp-server",
        "--sandbox-variant", "kaggle",
        "--code-sandbox-url", "{env:KAGGLE_RUNTIME_URL}",
        "--code-sandbox-id", "{env:KAGGLE_RUNTIME_ID}"
      ],
      "enabled": true,
      "timeout": 120000
    }
  }
}
```

The MCP server reads `KAGGLE_RUNTIME_URL` and `KAGGLE_RUNTIME_ID` from
the environment. If you placed them in `.env.kaggle`, OpenCode should
pick them up. If not, set them as system/user environment variables:

```powershell
[System.Environment]::SetEnvironmentVariable("KAGGLE_RUNTIME_URL", "YOUR_URL", "User")
[System.Environment]::SetEnvironmentVariable("KAGGLE_RUNTIME_ID", "YOUR_ID", "User")
```

**Restart OpenCode** after changing these variables so the MCP server
reads the new values.

---

## 6. Test the Connection

### 6a. Health Check (PC Side)

```cmd
python kaggle_scripts\kaggle_exec.py health
```

Expected output: all three config files present, IDs match, kernel reachable.

### 6b. Run a Smoke Test on the Kernel

Via OpenCode MCP (if configured):
```
Run this code on Kaggle:
import os, sys
print("cwd:", os.getcwd())
print("kaggle_input exists:", os.path.isdir("/kaggle/input"))
print("python:", sys.version)
```

Or push a test script:
```cmd
python kaggle_scripts\push_to_kaggle.py smoke_test.py
```

---

## 7. Pushing Files to Kaggle

### Manual Push (Individual Files)

```cmd
python kaggle_scripts\push_to_kaggle.py src/experiments/phase11.py kaggle_scripts/run_phase11.py
```

### Automated Push (Full Bundle — Recommended)

For production phase execution, use the bundle workflow:

1. **Build the bundle** on the PC:
   ```cmd
   python kaggle_scripts\build_phase18_bundle.py
   ```

2. **Upload the bundle** to Kaggle via the Jupyter UI or MCP:
   - Place `phase<PHASE>_kaggle_bundle.zip` at `/kaggle/working/`.

3. **Verify and extract** on the kernel:
   ```python
   # Run via MCP execute_code or the notebook
   import zipfile, hashlib, json
   # ... (see kaggle-execution-policy.md section 0.2)
   ```

### Key Notes on File Pushing
- The Jupyter API wraps `.py` files as jupytext notebooks — the file on
  disk is still correct (byte-identical), but GET verification of `.py`
  files through the API is unreliable.
- If a plain push fails verification, `push_to_kaggle.py` falls back to
  gzip+base64 encoding. On the kernel, decode with:
  ```python
  import gzip, base64
  data = gzip.decompress(base64.b64decode(content))
  ```
- Under heavy load (bulk pushes), the API may return transient HTTP 500/502.
  Retry with backoff in small batches.

---

## 8. Pulling Artifacts from Kaggle

```cmd
python kaggle_scripts\pull_from_kaggle.py
```

This pulls Phase 14 artifacts from `/kaggle/working/artifacts/` to
`reports/artifacts/` on the PC. Every file is md5-verified against the
kernel-side manifest.

To pull different artifacts, edit the `ARTIFACTS` list in
`pull_from_kaggle.py` or adapt it for your project.

---

## 9. Handling Runtime Recycling

Kaggle runtimes are **ephemeral**. The workspace (`/kaggle/working`) is
wiped when:
- The notebook goes idle and is shut down.
- Kaggle recycles the runtime for maintenance.
- The kernel crashes and is restarted.

### Signs of Recycling
- Previously pushed files are gone.
- `KAGGLE_RUNTIME_ID.txt` no longer matches the live kernel.
- API calls return 404 or empty results.

### Recovery Steps

1. **Get a fresh URL** from the Kaggle notebook (repeat step 3).
2. **Run the discovery script** (repeat step 4):
   ```cmd
   get-kaggle-runtime.bat
   ```
3. **Restart OpenCode** so MCP reads the new credentials.
4. **Re-push project source** from the PC:
   ```cmd
   python kaggle_scripts\push_to_kaggle.py src/config.py src/features.py ...
   ```
5. **Rebuild derived artifacts** on the kernel (they don't survive
   recycling).

### Automatic Recovery Detection

The project logs all execution outcomes to `logs/kaggle_execution.jsonl`.
Check recent entries:
```cmd
python kaggle_scripts\kaggle_exec.py status
```

---

## 10. Quick Reference — Common Commands

| Task | Command |
|------|---------|
| Set up credentials | `get-kaggle-runtime.bat` |
| Health check | `python kaggle_scripts\kaggle_exec.py health` |
| Push files | `python kaggle_scripts\push_to_kaggle.py <files>` |
| Pull artifacts | `python kaggle_scripts\pull_from_kaggle.py` |
| Execution journal | `python kaggle_scripts\kaggle_exec.py status` |
| Log an outcome | `python kaggle_scripts\kaggle_exec.py log <kind> <outcome> "<detail>" --elapsed <s>` |
| Build phase bundle | `python kaggle_scripts\build_phase<N>_bundle.py` |
| Smoke test (kernel) | `python smoke_test.py` |

---

## 11. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ERROR: KAGGLE_URL is empty` | Env var not set | Set `$env:KAGGLE_URL` or use the `.bat` file |
| `No python3 kernel was found` | Kernel not started | Open the Kaggle notebook, run a cell to start the kernel |
| HTTP 500/502 on push | Rate limiting under load | Retry with backoff; push in smaller batches |
| MCP returns 404 | Stale runtime config | Re-run discovery + restart OpenCode |
| `KAGGLE_RUNTIME_ID.txt` mismatch | Runtime recycled | Full recovery (section 9) |
| `.py` files appear different via GET | Jupytext wrapping | Normal; verify on kernel disk with md5 instead |
| `jupyter-mcp-server` not found | Not installed | `pip install jupyter-mcp-server` |
| Import errors on kernel | Missing packages | Add to the kernel: `!pip install <pkg>` or update the notebook's `requirements.txt` |

---

## 12. Security Notes

- **Never commit** `.env.kaggle`, `KAGGLE_RUNTIME_ID.txt`, `KAGGLE_RUNTIME_URL.txt`, or `KAGGLE_RUNTIME_ID.txt` to git. They are already in `.gitignore` (or should be).
- The Kaggle URL is **temporary** — it expires when the session ends.
- Do not put API keys, passwords, or tokens in any pushed file.
- Raw CERT data stays on Kaggle; never copy it to the PC.

---

## 13. File Reference

| File | Location | Purpose |
|------|----------|---------|
| `get-kaggle-runtime.ps1` | Project root | Discovers runtime credentials |
| `get-kaggle-runtime.bat` | Project root | Interactive wrapper for the above |
| `.env.kaggle` | Project root | Runtime env vars (gitignored) |
| `KAGGLE_RUNTIME_ID.txt` | Project root | Kernel UUID (gitignored) |
| `KAGGLE_RUNTIME_URL.txt` | Project root | Jupyter proxy URL (gitignored) |
| `opencode.json` | Project root | MCP server config |
| `kaggle_scripts/push_to_kaggle.py` | `kaggle_scripts/` | Push files PC → Kaggle |
| `kaggle_scripts/pull_from_kaggle.py` | `kaggle_scripts/` | Pull artifacts Kaggle → PC |
| `kaggle_scripts/kaggle_exec.py` | `kaggle_scripts/` | Health check + outcome journal |
| `docs/kaggle-execution-policy.md` | `docs/` | Full operational policy |
