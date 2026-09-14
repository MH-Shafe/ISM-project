"""Build the ISM_Report_Figure_Generator.ipynb notebook."""
import json

def _split_source(source):
    """Split source into notebook-format lines (each ending with \\n except last)."""
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": _split_source(source), "outputs": [], "execution_count": None}

cells = []

# ── Cell 1: Title ──
cells.append(md("""# ISM Project Shafe — Report Figure Generator

This notebook regenerates publication/report figures from already-preserved
experiment artifacts.

**It performs:**
- NO model training
- NO threshold tuning
- NO TEST optimization
- NO modification of frozen scientific artifacts

All figures appear directly in notebook output cells."""))

# ── Cell 2: Configuration ──
cells.append(code("""# ── USER CONFIGURATION ──────────────────────────────────────────────
# Edit this cell to change appearance or behavior.

GITHUB_REPO = "https://github.com/MH-Shafe/ISM-project.git"
REPO_DIR    = "ISM-project"
AUTO_CLONE  = True       # clone from GitHub if repo not found locally
SAVE_FIGURES = False     # True → save PNGs; False → display inline only

DPI         = 150
FIGSIZE     = (10, 6)
TITLE_SIZE  = 15
LABEL_SIZE  = 12
TICK_SIZE   = 10
LEGEND_SIZE = 10
LINE_WIDTH  = 2.0
BAR_WIDTH   = 0.22
GRID_ALPHA  = 0.20"""))

# ── Cell 3: Auto-detect repo ──
cells.append(code("""import os, subprocess
from pathlib import Path

def find_repo():
    \"\"\"Locate the ISM-project repository root.\"\"\"
    cwd = Path.cwd()
    # 1. Check cwd itself
    if (cwd / "scripts").is_dir() and (cwd / "reports").is_dir():
        return cwd
    # 2. Check ./ISM-project
    if (cwd / REPO_DIR).is_dir():
        return cwd / REPO_DIR
    # 3. Check /kaggle/working/ISM-project
    kaggle_path = Path("/kaggle/working") / REPO_DIR
    if kaggle_path.is_dir():
        return kaggle_path
    # 4. Check parent directories
    for p in [cwd.parent, cwd.parent.parent]:
        if (p / "scripts").is_dir() and (p / "reports").is_dir():
            return p
    return None

REPO_ROOT = find_repo()

if REPO_ROOT is None and AUTO_CLONE:
    print("Repository not found locally. Cloning from GitHub...")
    try:
        subprocess.check_call(
            ["git", "clone", GITHUB_REPO, REPO_DIR],
            cwd=str(Path.cwd()),
            timeout=60,
        )
        REPO_ROOT = Path.cwd() / REPO_DIR
        print(f"Cloned to {REPO_ROOT}")
    except Exception as e:
        print(f"Clone failed: {e}")
        print("Enable Internet in Kaggle, or upload/attach the repository.")
        REPO_ROOT = None

if REPO_ROOT is not None:
    print(f"Repository root: {REPO_ROOT}")
    try:
        sha_before = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
        print(f"Git commit before update: {sha_before}")
    except Exception:
        sha_before = None
        print("Git commit before update: (unknown)")

    # Attempt safe pull if worktree is clean
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
        if not status:
            subprocess.check_call(
                ["git", "fetch", "origin", "main"],
                cwd=str(REPO_ROOT), timeout=30,
            )
            subprocess.check_call(
                ["git", "pull", "--ff-only", "origin", "main"],
                cwd=str(REPO_ROOT), timeout=30,
            )
            sha_after = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO_ROOT), text=True,
            ).strip()
            print(f"Git commit after update:  {sha_after}")
        else:
            print("Repository has local changes; skipping automatic pull.")
    except Exception as e:
        print(f"Git update failed: {e}")
else:
    print("ERROR: Repository not found. Cannot continue.")"""))

# ── Cell 4: Imports ──
cells.append(code("""%matplotlib inline

from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({
    "figure.dpi": DPI,
    "font.size": TICK_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

def show_figure(fig, filename=None):
    \"\"\"Display figure inline and optionally save it.\"\"\"
    if SAVE_FIGURES and filename:
        out = REPO_ROOT / "reports" / "figures" / "notebook_generated"
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {out / filename}")
    display(fig)
    plt.close(fig)"""))

# ── Cell 5: Artifact discovery ──
cells.append(code("""def first_existing(*paths):
    \"\"\"Return the first path that exists, or None.\"\"\"
    for p in paths:
        if p.exists():
            return p
    return None

ablation_path = first_existing(
    REPO_ROOT / "reports/final/END_TO_END_ABLATION_SUMMARY.csv",
    REPO_ROOT / "reports/final/END_TO_END_ABLATION_SUMMARY.json",
)
shap_path = first_existing(
    REPO_ROOT / "reports/artifacts/phase11_global_importance.csv",
    REPO_ROOT / "artifacts/summaries/phase11_global_importance.json",
)
baseline_csv_path = first_existing(
    REPO_ROOT / "reports/final/BASELINE_MODEL_COMPARISON.csv",
    REPO_ROOT / "reports/artifacts/baseline_benchmark/BASELINE_MODEL_COMPARISON.csv",
)

artifacts = {
    "ablation":              ablation_path,
    "baseline_csv":          baseline_csv_path,
    "all_results":           REPO_ROOT / "reports/artifacts/baseline_benchmark/all_results.json",
    "freeze":                REPO_ROOT / "reports/artifacts/baseline_benchmark/BENCHMARK_FREEZE.json",
    "bootstrap_ci":          REPO_ROOT / "reports/artifacts/baseline_benchmark/bootstrap_confidence_intervals.json",
    "pred_dir":              REPO_ROOT / "reports/artifacts/baseline_benchmark/predictions",
    "shap":                  shap_path,
    "decisions_parquet":     first_existing(
        REPO_ROOT / "demo_artifacts/final_user_day_decisions.parquet",
        REPO_ROOT / "reports/artifacts/phase20/final_user_day_decisions.parquet",
    ),
}

rows = []
for name, path in artifacts.items():
    exists = path is not None and path.exists() if isinstance(path, Path) else False
    display = str(path.relative_to(REPO_ROOT)) if exists and path is not None else str(path)
    rows.append({"Artifact": name, "Path": display, "Exists": "YES" if exists else "NO"})

pd.DataFrame(rows)"""))

# ── Cell 6: Validate predictions ──
cells.append(code("""PRED_DIR = artifacts["pred_dir"]

with open(artifacts["all_results"]) as f:
    all_results = json.load(f)

MODEL_MAP = {
    "logistic":         ("Logistic Regression", "logistic_test_predictions.parquet"),
    "random_forest":    ("Random Forest",       "random_forest_test_predictions.parquet"),
    "xgboost":          ("XGBoost",             "xgboost_test_predictions.parquet"),
    "catboost":         ("CatBoost",            "catboost_test_predictions.parquet"),
    "lightgbm":         ("LightGBM",            "lightgbm_benchmark_test_predictions.parquet"),
}

EXPECTED = {
    "logistic":      {"roc_auc": 0.882866, "pr_auc": 0.009934},
    "random_forest": {"roc_auc": 0.772914, "pr_auc": 0.183094},
    "xgboost":       {"roc_auc": 0.940020, "pr_auc": 0.269326},
    "catboost":      {"roc_auc": 0.919902, "pr_auc": 0.179216},
    "lightgbm":      {"roc_auc": 0.939157, "pr_auc": 0.267776},
}

pred_data = {}  # model_key -> {"y": array, "score": array, "df": DataFrame}
val_rows = []

for key, (display_name, fname) in MODEL_MAP.items():
    fpath = PRED_DIR / fname
    if not fpath.exists():
        val_rows.append({"Model": display_name, "Status": "MISSING"})
        continue
    df = pd.read_parquet(fpath)
    y = df["malicious"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()
    roc = roc_auc_score(y, s)
    pr  = average_precision_score(y, s)
    exp = EXPECTED[key]
    roc_ok = abs(roc - exp["roc_auc"]) < 0.001
    pr_ok  = abs(pr  - exp["pr_auc"])  < 0.001
    status = "PASS" if (roc_ok and pr_ok) else "FAIL"
    pred_data[key] = {"y": y, "score": s, "df": df, "display": display_name}
    val_rows.append({
        "Model": display_name, "Rows": len(df), "Positives": int(y.sum()),
        "ROC-AUC": f"{roc:.6f}", "PR-AUC": f"{pr:.6f}", "Status": status,
    })

val_df = pd.DataFrame(val_rows)
print(f"Models loaded: {len(pred_data)}/5")
print(f"All PASS: {(val_df['Status'] == 'PASS').all()}")
val_df"""))

# ── Cell 7: Figure 1 — Architecture ──
cells.append(md("## Figure 1 — Final System Architecture"))
cells.append(code("""fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

steps = [
    "CERT r4.2",
    "Validation & Preprocessing",
    "Leakage-Safe User-Day Aggregation",
    "Behavioral + Graph Features (12)",
    "Frozen LightGBM (lgbm-graph-v1)",
    "Frozen Decision Policy (t=0.9186)",
    "Conformal Uncertainty",
    "SHAP Explanations",
    "Trust / Context Diagnostics",
    "Deterministic Decision Layer",
    "Analyst-Facing User-Day Output",
]

ys = np.linspace(0.94, 0.08, len(steps))
for i, (label, y) in enumerate(zip(steps, ys)):
    ax.text(0.5, y, label, ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.2))
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, ys[i+1]+0.026), xytext=(0.5, y-0.026),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

ax.text(0.5, 0.985, "Figure 1. Final System Architecture",
        ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold")
ax.text(0.5, 0.025,
        "Adaptive Risk was evaluated and rejected; it is not part of the production path.",
        ha="center", va="bottom", fontsize=9, style="italic")

show_figure(fig, "figure_01_system_architecture.png")"""))

# ── Cell 8: Figure 2 — Chronological Split ──
cells.append(md("## Figure 2 — Leakage-Safe Chronological Split"))
cells.append(code("""with open(artifacts["ablation"]) as f:
    abl_splits = json.load(f).get("splits", {})

sp_train = abl_splits.get("train", {})
sp_cal   = abl_splits.get("cal", {})
sp_test  = abl_splits.get("test", {})

train_rows = sp_train.get("rows", 0)
cal_rows   = sp_cal.get("rows", 0)
test_rows  = sp_test.get("rows", 0)
train_pos  = sp_train.get("positives", 0)
cal_pos    = sp_cal.get("positives", 0)
test_pos   = sp_test.get("positives", 0)

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)

box_w, box_h = 3.0, 2.8
box_y = 0.8
box_centers = [2.0, 6.0, 10.0]
box_colors  = ["#4C72B0", "#55A868", "#C44E52"]
stage_names = ["TRAIN", "CAL", "TEST"]
date_lines  = [
    ["≤ 2011-01-31"],
    ["2011-02-01", "to", "2011-03-31"],
    ["≥ 2011-04-01"],
]
row_lines = [f"{train_rows:,} user-days", f"{cal_rows:,} user-days", f"{test_rows:,} user-days"]
pos_lines = [f"{train_pos:,} malicious", f"{cal_pos:,} malicious", f"{test_pos:,} malicious"]

for i, (cx, color, name, dates, rows_txt, pos_txt) in enumerate(
    zip(box_centers, box_colors, stage_names, date_lines, row_lines, pos_lines)
):
    rect = plt.Rectangle((cx - box_w/2, box_y), box_w, box_h,
                          linewidth=1.5, edgecolor=color, facecolor="white",
                          zorder=2)
    ax.add_patch(rect)
    header_rect = plt.Rectangle((cx - box_w/2, box_y + box_h - 0.55), box_w, 0.55,
                                linewidth=0, edgecolor="none", facecolor=color, alpha=0.15, zorder=3)
    ax.add_patch(header_rect)
    ax.text(cx, box_y + box_h - 0.25, name, ha="center", va="center",
            fontsize=14, fontweight="bold", color=color, zorder=4)
    date_y = box_y + box_h - 0.9
    for line in dates:
        ax.text(cx, date_y, line, ha="center", va="center", fontsize=9.5, zorder=4)
        date_y -= 0.28
    ax.text(cx, box_y + 0.7, rows_txt, ha="center", va="center", fontsize=10, zorder=4)
    ax.text(cx, box_y + 0.35, pos_txt, ha="center", va="center", fontsize=9.5,
            color="#C44E52", zorder=4)

for i in range(len(box_centers) - 1):
    x_start = box_centers[i] + box_w/2 + 0.05
    x_end   = box_centers[i+1] - box_w/2 - 0.05
    x_mid   = (x_start + x_end) / 2
    ax.annotate("", xy=(x_end, box_y + box_h/2), xytext=(x_start, box_y + box_h/2),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#555555", mutation_scale=18))

ax.text(6.0, 0.25, "No temporal overlap  |  TEST used only for final evaluation",
        ha="center", va="center", fontsize=10, style="italic", color="#555555")

ax.text(6.0, 4.2, "Figure 2. Leakage-Safe Chronological Data Split",
        ha="center", va="center", fontsize=TITLE_SIZE, fontweight="bold")

fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
show_figure(fig, "figure_02_chronological_split.png")"""))

# ── Cell 9: Figure 3 — Class Distribution ──
cells.append(md("## Figure 3 — CERT r4.2 Class Distribution"))
cells.append(code("""total      = train_rows + cal_rows + test_rows
malicious  = train_pos + cal_pos + test_pos
benign     = total - malicious

fig, ax = plt.subplots(figsize=(10, 5))
categories = ["Benign", "Malicious"]
counts     = [benign, malicious]
colors     = ["#4C72B0", "#C44E52"]

bars = ax.barh(categories, counts, color=colors, height=0.55)
ax.set_xscale("log")
ax.set_xlabel("User-Day Count (log scale)", fontsize=LABEL_SIZE)
ax.set_title("Figure 3. CERT r4.2 User-Day Class Distribution",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.grid(axis="x", alpha=GRID_ALPHA)
ax.set_xlim(1, max(counts) * 5)

for bar, val, color in zip(bars, counts, colors):
    pct = 100.0 * val / total
    label = f"{val:,} ({pct:.2f}%)"
    x_pos = val * 1.25
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
            ha="left", va="center", fontsize=11, fontweight="bold", color=color)

ax.text(0.5, -0.14,
        f"Extreme class imbalance: {malicious:,} malicious user-days out of {total:,} total",
        transform=ax.transAxes, ha="center", fontsize=9.5, style="italic", color="#555555")
fig.subplots_adjust(bottom=0.18, top=0.92)

show_figure(fig, "figure_03_class_distribution.png")"""))

# ── Cell 10: Figure 4 — Ablation ──
cells.append(md("## Figure 4 — Leakage-Safe Ablation Comparison"))
cells.append(code("""abl_path = artifacts["ablation"]
if abl_path is None:
    raise FileNotFoundError("Ablation artifact not found")

if abl_path.suffix == ".csv":
    abl_raw = pd.read_csv(abl_path)
    abl_raw = abl_raw[abl_raw.get("table", pd.Series(["table1"]*len(abl_raw))) == "table1"].head(3)
    models_abl = abl_raw["label"].tolist()
    pr_auc     = abl_raw["auc_pr"].astype(float).tolist()
    f1_vals    = abl_raw["f1"].astype(float).tolist()
    mcc_vals   = abl_raw["mcc"].astype(float).tolist()
else:
    with open(abl_path) as f:
        abl_json = json.load(f)
    table1 = abl_json.get("table1_directly_comparable", [])
    models_abl, pr_auc, f1_vals, mcc_vals = [], [], [], []
    for entry in table1[:3]:
        models_abl.append(entry["label"])
        m = entry.get("metrics", {})
        pr_auc.append(float(m.get("auc_pr", 0)))
        f1_vals.append(float(m.get("f1", 0)))
        mcc_vals.append(float(m.get("mcc", 0)))

x = np.arange(len(models_abl))
w = BAR_WIDTH

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.bar(x - w, pr_auc,  w, label="PR-AUC", color="#4C72B0")
ax.bar(x,     f1_vals, w, label="F1",     color="#55A868")
ax.bar(x + w, mcc_vals,w, label="MCC",    color="#C44E52")

ax.set_xticks(x)
ax.set_xticklabels(["Behavioral\\nonly", "Graph\\nonly", "Behavioral\\n+ Graph"],
                    fontsize=TICK_SIZE)
ax.set_ylabel("Score", fontsize=LABEL_SIZE)
ax.set_ylim(bottom=0)
ax.set_title("Figure 4. Leakage-Safe Ablation Comparison (Arms A / B / C)",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.legend(fontsize=LEGEND_SIZE)
ax.grid(axis="y", alpha=GRID_ALPHA)

for i, (p, f, m) in enumerate(zip(pr_auc, f1_vals, mcc_vals)):
    ax.text(i - w, p + 0.005, f"{p:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i,     f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

fig.subplots_adjust(bottom=0.20, top=0.90)
show_figure(fig, "figure_04_ablation.png")"""))

# ── Cell 11: Figure 5A — ROC ──
cells.append(md("## Figure 5A — ROC Curves on Chronological TEST"))
cells.append(code("""from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ORDER = ["logistic", "random_forest", "xgboost", "catboost", "lightgbm"]
COLORS = {"logistic": "#C44E52", "random_forest": "#DD8452",
          "xgboost": "#55A868", "catboost": "#4C72B0", "lightgbm": "#937860"}
STYLES = {"logistic": "-", "random_forest": "--",
          "xgboost": "-", "catboost": "-.", "lightgbm": "-"}

fig, ax = plt.subplots(figsize=FIGSIZE)
for key in ORDER:
    if key not in pred_data:
        continue
    d = pred_data[key]
    fpr, tpr, _ = roc_curve(d["y"], d["score"])
    auc_val = roc_auc_score(d["y"], d["score"])
    lw = 2.8 if key == "lightgbm" else LINE_WIDTH
    ax.step(fpr, tpr, where="post", lw=lw, color=COLORS[key],
            linestyle=STYLES[key],
            label=f'{d["display"]} (AUC={auc_val:.3f})')

ax.step([0, 1], [0, 1], where="post", ls="--", lw=1, color="grey", label="Random")
ax.set_xlabel("False Positive Rate", fontsize=LABEL_SIZE)
ax.set_ylabel("True Positive Rate", fontsize=LABEL_SIZE)
ax.set_title("Figure 5A. ROC Curves — All Five Models on Chronological TEST",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.grid(alpha=GRID_ALPHA)

axins = inset_axes(ax, width="38%", height="38%", loc="lower right",
                   borderpad=0.35)
for key in ORDER:
    if key not in pred_data:
        continue
    d = pred_data[key]
    fpr, tpr, _ = roc_curve(d["y"], d["score"])
    auc_val = roc_auc_score(d["y"], d["score"])
    lw = 2.8 if key == "lightgbm" else LINE_WIDTH
    axins.step(fpr, tpr, where="post", lw=lw, color=COLORS[key],
               linestyle=STYLES[key])
axins.step([0, 1], [0, 1], where="post", ls="--", lw=0.8, color="grey")
axins.set_xlim(0, 0.10)
tpr_max = 1.02
for key in ORDER:
    if key not in pred_data:
        continue
    fpr_k, tpr_k, _ = roc_curve(pred_data[key]["y"], pred_data[key]["score"])
    tpr_at_01 = tpr_k[fpr_k <= 0.10]
    if len(tpr_at_01) > 0:
        tpr_max = max(tpr_max, tpr_at_01.max() * 1.15)
axins.set_ylim(0, min(tpr_max, 1.02))
axins.set_title("Low-FPR region", fontsize=8, pad=2)
axins.tick_params(labelsize=7)
axins.grid(alpha=GRID_ALPHA)

ax.legend(fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22))

show_figure(fig, "figure_05a_roc.png")"""))

# ── Cell 12: Figure 5B — PR ──
cells.append(md("## Figure 5B — Precision–Recall Curves on Chronological TEST"))
cells.append(code("""fig, ax = plt.subplots(figsize=FIGSIZE)
for key in ORDER:
    if key not in pred_data:
        continue
    d = pred_data[key]
    prec, rec, _ = precision_recall_curve(d["y"], d["score"])
    ap = average_precision_score(d["y"], d["score"])
    lw = 2.8 if key == "lightgbm" else LINE_WIDTH
    ax.step(rec, prec, where="post", lw=lw, color=COLORS[key],
            linestyle=STYLES[key],
            label=f'{d["display"]} (AP={ap:.3f})')

prevalence = pred_data[ORDER[0]]["y"].mean()
ax.axhline(y=prevalence, ls=":", lw=1.2, color="grey",
           label=f"Prevalence baseline ({prevalence:.6f})")
ax.text(0.98, prevalence * 3, f"Positive prevalence = {prevalence*100:.3f}%",
        ha="right", va="bottom", fontsize=9, color="grey", style="italic")

ax.set_xlabel("Recall", fontsize=LABEL_SIZE)
ax.set_ylabel("Precision", fontsize=LABEL_SIZE)
ax.set_title("Figure 5B. Precision–Recall Curves on Chronological TEST",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.grid(alpha=GRID_ALPHA)
ax.legend(fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))

show_figure(fig, "figure_05b_pr.png")"""))
cells.append(md("""> **Note:** ROC and precision–recall curves are computed from the preserved row-level
> chronological TEST predictions (47,000 user-days; 30 malicious user-days).
> The step-like appearance reflects the small number of positive TEST examples
> and is not smoothed."""))

# ── Cell 13: Figure 6 — SHAP ──
cells.append(md("## Figure 6 — Global SHAP Feature Importance"))
cells.append(code("""shap_path = artifacts["shap"]
if shap_path is None:
    raise FileNotFoundError("SHAP importance artifact not found")

if shap_path.suffix == ".csv":
    shap_df = pd.read_csv(shap_path)
    shap_df = shap_df.rename(columns={"mean_abs_shap": "importance"})
else:
    with open(shap_path) as f:
        shap_json = json.load(f)
    per_feat = shap_json.get("calibration", shap_json).get("importance", shap_json).get("per_feature", {})
    rows = []
    for feat, vals in per_feat.items():
        rows.append({"feature": feat, "importance": vals.get("mean_abs", 0)})
    shap_df = pd.DataFrame(rows)

shap_df = shap_df.sort_values("importance", ascending=True)

fig_h = max(6, 0.42 * len(shap_df) + 2)
fig, ax = plt.subplots(figsize=(10, fig_h))
bars = ax.barh(shap_df["feature"], shap_df["importance"], color="#4C72B0")
ax.set_xlabel("Mean Absolute SHAP Contribution", fontsize=LABEL_SIZE)
ax.set_title("Figure 6. Global SHAP Feature Importance (Phase 11, 12 features)",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.grid(axis="x", alpha=GRID_ALPHA)

for bar, val in zip(bars, shap_df["importance"]):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

fig.subplots_adjust(left=0.32, top=0.94)
show_figure(fig, "figure_06_shap.png")"""))

# ── Cell 14: Figure 7 — Decision Distribution ──
cells.append(md("## Figure 7 — Final Operational Decision Distribution"))
cells.append(code("""ddf = pd.read_parquet(artifacts["decisions_parquet"])
counts = ddf["risk_level"].astype(str).str.upper().value_counts()

preferred = ["ALERT", "BORDERLINE", "MONITOR", "NON-ALERT"]
vals  = [int(counts.get(x, 0)) for x in preferred]
total_d = sum(vals)

fig, ax = plt.subplots(figsize=FIGSIZE)
bars = ax.bar(preferred, vals, color=["#C44E52", "#DD8452", "#55A868", "#4C72B0"])
ax.set_ylabel("User-Day Count", fontsize=LABEL_SIZE)
ax.set_title("Figure 7. Final Operational Decision Distribution (Phase 20)",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.grid(axis="y", alpha=GRID_ALPHA)

ymax = max(vals)
for bar, v in zip(bars, vals):
    pct = 100.0 * v / total_d
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ymax*0.015,
            f"{v:,}\\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

ax.text(0.5, -0.13, f"Complete Phase 20 population: {total_d:,} user-days",
        transform=ax.transAxes, ha="center", fontsize=9)
fig.subplots_adjust(bottom=0.18, top=0.92)

show_figure(fig, "figure_07_decisions.png")"""))

# ── Cell 15: Figure 8 — Baseline Comparison ──
cells.append(md("## Figure 8 — Post-Freeze Baseline Comparison"))
cells.append(code("""if artifacts["baseline_csv"] is not None and artifacts["baseline_csv"].exists():
    bdf = pd.read_csv(artifacts["baseline_csv"])
    print("Figure 8 source: BASELINE_MODEL_COMPARISON.csv")
else:
    print("Figure 8 source: all_results.json fallback")
    required_models = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "LightGBM (frozen)"]
    rows = []
    for model in required_models:
        r = all_results.get(model, {})
        rows.append({
            "Model": model,
            "PR-AUC": r.get("pr_auc", 0),
            "F1": r.get("f1", 0),
            "MCC": r.get("mcc", 0),
        })
    bdf = pd.DataFrame(rows)

MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "LightGBM (frozen)"]
name_col = "Model"
if "PR-AUC" not in bdf.columns and "pr_auc" in bdf.columns:
    bdf = bdf.rename(columns={"pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC"})
kept = bdf[bdf[name_col].isin(MODEL_ORDER)].copy()
kept["_order"] = kept[name_col].map({m: i for i, m in enumerate(MODEL_ORDER)})
kept = kept.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

DISPLAY = {
    "Logistic Regression": "Logistic\\nRegression",
    "Random Forest":       "Random\\nForest",
    "XGBoost":             "XGBoost",
    "CatBoost":            "CatBoost",
    "LightGBM (frozen)":   "LightGBM\\n(frozen)",
}

metrics = ["PR-AUC", "F1", "MCC"]
mcolors = ["#4C72B0", "#55A868", "#C44E52"]
x = np.arange(len(kept))
w = BAR_WIDTH

fig, ax = plt.subplots(figsize=(12, 6.5))
for i, (metric, color) in enumerate(zip(metrics, mcolors)):
    vals = pd.to_numeric(kept[metric], errors="coerce").to_numpy()
    bars = ax.bar(x + (i-1)*w, vals, w, label=metric, color=color)
    for bar, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([DISPLAY.get(m, m) for m in kept[name_col]], fontsize=TICK_SIZE)
ax.set_ylabel("Score", fontsize=LABEL_SIZE)
ax.set_ylim(0, 0.45)
ax.set_title("Figure 8. Post-Freeze Baseline Model Comparison",
             fontsize=TITLE_SIZE, fontweight="bold")
ax.legend(fontsize=LEGEND_SIZE)
ax.grid(axis="y", alpha=GRID_ALPHA)
ax.text(0.5, -0.18, "Same 12 features and chronological split; thresholds selected on CAL only.",
        transform=ax.transAxes, ha="center", fontsize=9)
fig.subplots_adjust(bottom=0.22, top=0.92)

show_figure(fig, "figure_08_baselines.png")"""))

# ── Cell 16: Figure 9 — Confusion Matrices ──
cells.append(md("## Figure 9 — TEST Confusion Matrices"))
cells.append(code("""with open(artifacts["all_results"]) as f:
    results = json.load(f)

CM_ORDER = [
    ("Logistic Regression", "logistic"),
    ("Random Forest",       "random_forest"),
    ("XGBoost",             "xgboost"),
    ("CatBoost",            "catboost"),
    ("LightGBM (frozen)",   "lightgbm"),
]

for display_name, key in CM_ORDER:
    if key not in pred_data:
        print(f"Skipping {display_name}: predictions not loaded")
        continue
    d = pred_data[key]
    threshold = results[display_name]["threshold"]
    y_pred = (d["score"] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(d["y"], y_pred, labels=[0,1]).ravel()

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\\nNegative", "Predicted\\nPositive"], fontsize=10)
    ax.set_yticklabels(["Actual\\nNegative", "Actual\\nPositive"], fontsize=10)
    ax.set_title(f"Confusion Matrix — {display_name}\\n(threshold={threshold:.4f})",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    show_figure(fig, f"figure_09_confusion_{key}.png")"""))

# ── Cell 17: Bootstrap CI ──
cells.append(md("## Supplementary Figure — Bootstrap 95% Confidence Intervals"))
cells.append(code("""with open(artifacts["bootstrap_ci"]) as f:
    ci = json.load(f)

ci_models = ci if "Logistic Regression" in ci else ci.get("models", {})
ci_keys = ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "LightGBM (benchmark)"]
ci_keys = [k for k in ci_keys if k in ci_models]

metrics_ci = ["roc_auc", "pr_auc", "f1", "mcc"]
metric_labels = ["ROC-AUC", "PR-AUC", "F1", "MCC"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, metric, label in zip(axes, metrics_ci, metric_labels):
    points, lows, highs, names = [], [], [], []
    for mname in ci_keys:
        mdata = ci_models[mname].get(metric, {})
        if not mdata:
            continue
        points.append(mdata.get("mean", mdata.get("point_estimate", 0)))
        lows.append(mdata.get("ci_95_lower", mdata.get("ci_lower", 0)))
        highs.append(mdata.get("ci_95_upper", mdata.get("ci_upper", 0)))
        names.append(mname.replace(" ", "\\n") if len(mname) > 12 else mname)

    x = np.arange(len(names))
    yerr = np.array([np.array(points) - np.array(lows), np.array(highs) - np.array(points)])
    ax.errorbar(x, points, yerr=yerr, fmt="o", capsize=5, lw=1.5, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(label, fontsize=LABEL_SIZE)
    ax.set_title(f"{label} — 95% Bootstrap CI (1000 resamples)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)

fig.suptitle("Supplementary Figure. Bootstrap Confidence Intervals",
             fontsize=TITLE_SIZE, fontweight="bold", y=1.01)
fig.tight_layout()
show_figure(fig, "figure_10_bootstrap_ci.png")"""))

# ── Cell 18: Validation summary ──
cells.append(md("## Final Validation Summary"))
cells.append(code("""baseline_source_ok = (
    (artifacts["baseline_csv"] is not None and artifacts["baseline_csv"].exists())
    or
    (artifacts["all_results"].exists() and all(
        m in all_results for m in ["Logistic Regression", "Random Forest", "XGBoost", "CatBoost"]
    ))
)

checks = {
    "Repository detected":              REPO_ROOT is not None,
    "Ablation artifact":                artifacts["ablation"] is not None and artifacts["ablation"].exists(),
    "Baseline comparison source":       baseline_source_ok,
    "All results JSON":                 artifacts["all_results"].exists(),
    "Freeze JSON":                      artifacts["freeze"].exists(),
    "Bootstrap CI JSON":                artifacts["bootstrap_ci"].exists(),
    "SHAP artifact":                    artifacts["shap"] is not None and artifacts["shap"].exists(),
    "Decisions parquet":                artifacts["decisions_parquet"] is not None and artifacts["decisions_parquet"].exists(),
    "Prediction dir":                   artifacts["pred_dir"].is_dir(),
    "All 5 models loaded":              len(pred_data) == 5,
    "All predictions PASS":             (val_df["Status"] == "PASS").all() if len(val_df) == 5 else False,
    "SAVE_FIGURES default":             SAVE_FIGURES == False,
}

summary = pd.DataFrame([
    {"Check": k, "Result": "PASS" if v else "FAIL"}
    for k, v in checks.items()
])
print(summary.to_string(index=False))
print(f"\\nOverall: {'ALL PASS' if all(checks.values()) else 'SOME FAILED'}")"""))

# ── Build notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = r"D:\Class\ISM\ISM-Project-Shafe-Git\notebooks\ISM_Report_Figure_Generator.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
