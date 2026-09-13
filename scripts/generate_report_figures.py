#!/usr/bin/env python3
"""
ISM Project Shafe - reproducible report figure generator.

This script NEVER trains or tunes a model. It only reads saved project artifacts
and regenerates report figures.

Typical use from the repository root:
    python scripts/generate_report_figures.py

Optional explicit inputs:
    python scripts/generate_report_figures.py \
        --baseline-csv reports/final/BASELINE_MODEL_COMPARISON.csv \
        --all-results reports/artifacts/baseline_benchmark/all_results.json \
        --freeze reports/artifacts/baseline_benchmark/BENCHMARK_FREEZE.json \
        --bootstrap-ci reports/artifacts/baseline_benchmark/bootstrap_confidence_intervals.json

The STYLE section below is intentionally simple so you can change figure size,
fonts, labels, line widths, etc. without changing scientific values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# STYLE: EDIT THESE VALUES TO CHANGE APPEARANCE ONLY
# ---------------------------------------------------------------------------

DPI = 300
FIGSIZE = (10, 6)
TITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
VALUE_SIZE = 9
LINE_WIDTH = 2.0
BAR_WIDTH = 0.22
GRID_ALPHA = 0.20
BOTTOM_MARGIN = 0.22
TOP_MARGIN = 0.88

# Canonical model order for the post-freeze benchmark.
MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "CatBoost",
    "LightGBM (frozen)",
]

DISPLAY_NAME = {
    "Logistic Regression": "Logistic\nRegression",
    "Random Forest": "Random\nForest",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "LightGBM (frozen)": "LightGBM\n(frozen)",
    "LightGBM (benchmark)": "LightGBM\n(benchmark)",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def normalise_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def discover_file(root: Path, patterns: Iterable[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[CREATED] {path}")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one comparable LightGBM row (frozen preferred) plus four baselines."""
    name_col = find_column(df, ["Model", "model"])
    if name_col is None:
        raise ValueError("Could not find model-name column in baseline CSV.")

    d = df.copy()
    d[name_col] = d[name_col].astype(str)

    keep = []
    for name in MODEL_ORDER:
        exact = d[d[name_col] == name]
        if len(exact):
            keep.append(exact.iloc[0])

    if not keep:
        raise ValueError("No recognised benchmark models found.")

    return pd.DataFrame(keep).reset_index(drop=True)


def find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {normalise_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalise_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def format_pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f}%"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        obj = load_json(path)
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            # Common record structures.
            for key in ["records", "data", "results", "features"]:
                if key in obj and isinstance(obj[key], list):
                    return pd.DataFrame(obj[key])
            return pd.DataFrame(obj)
    raise ValueError(f"Unsupported table format: {path}")


# ---------------------------------------------------------------------------
# FIGURE 1 - FINAL SYSTEM ARCHITECTURE
# ---------------------------------------------------------------------------

def figure_01_architecture(out_dir: Path) -> Path:
    steps = [
        "CERT r4.2",
        "Validation & Preprocessing",
        "Leakage-Safe User-Day Aggregation",
        "Behavioral + Graph Features",
        "Frozen LightGBM (lgbm-graph-v1)",
        "Frozen Decision Policy",
        "Conformal Uncertainty",
        "SHAP Explanations",
        "Trust / Context Diagnostics",
        "Deterministic Decision Layer",
        "Analyst-Facing User-Day Output",
    ]

    fig = plt.figure(figsize=(9, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ys = np.linspace(0.94, 0.08, len(steps))
    for i, (label, y) in enumerate(zip(steps, ys)):
        ax.text(
            0.5, y, label,
            ha="center", va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.2),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, ys[i + 1] + 0.026),
                xytext=(0.5, y - 0.026),
                arrowprops=dict(arrowstyle="->", lw=1.2),
            )

    ax.text(
        0.5, 0.985,
        "Figure 1. Final System Architecture",
        ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold",
    )
    ax.text(
        0.5, 0.025,
        "Adaptive Risk was evaluated separately and rejected; it is not part of the production path.",
        ha="center", va="bottom", fontsize=9,
    )

    path = out_dir / "figure_01_final_system_architecture.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# FIGURE 2 - CHRONOLOGICAL SPLIT
# ---------------------------------------------------------------------------

def figure_02_split(freeze: dict, out_dir: Path) -> Path:
    split = freeze["split"]
    names = ["TRAIN", "CAL", "TEST"]
    rows = [split["train_rows"], split["cal_rows"], split["test_rows"]]
    pos = [split["train_pos"], split["cal_pos"], split["test_pos"]]
    dates = [
        "≤ 2011-01-31",
        "2011-02-01 to 2011-03-31",
        "≥ 2011-04-01",
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(names))
    bars = ax.bar(x, rows)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=TICK_SIZE)
    ax.set_ylabel("User-Days", fontsize=LABEL_SIZE)
    ax.set_title("Figure 2. Leakage-Safe Chronological Data Split",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    ymax = max(rows)
    for bar, n, p, date in zip(bars, rows, pos, dates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.025,
            f"{date}\n{n:,} rows\n{p:,} malicious",
            ha="center", va="bottom", fontsize=VALUE_SIZE,
        )

    ax.text(
        0.5, -0.19,
        "TRAIN → CAL → TEST   |   No temporal overlap   |   TEST not used for training or threshold fitting",
        transform=ax.transAxes, ha="center", fontsize=9,
    )

    fig.subplots_adjust(bottom=BOTTOM_MARGIN, top=TOP_MARGIN)
    path = out_dir / "figure_02_chronological_split.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# FIGURE 3 - CLASS DISTRIBUTION
# ---------------------------------------------------------------------------

def figure_03_class_distribution(freeze: dict, out_dir: Path) -> Path:
    split = freeze["split"]
    total = split["train_rows"] + split["cal_rows"] + split["test_rows"]
    malicious = split["train_pos"] + split["cal_pos"] + split["test_pos"]
    benign = total - malicious

    labels = ["Benign", "Malicious"]
    counts = [benign, malicious]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, counts)

    ax.set_ylabel("User-Day Count", fontsize=LABEL_SIZE)
    ax.set_title("Figure 3. CERT r4.2 User-Day Class Distribution",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    for bar, value in zip(bars, counts):
        pct = value / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts) * 0.018,
            f"{value:,}\n({format_pct(pct)})",
            ha="center", va="bottom", fontsize=VALUE_SIZE,
        )

    ax.text(
        0.5, -0.14,
        f"Total = {total:,} user-days | Malicious = {malicious:,}",
        transform=ax.transAxes, ha="center", fontsize=9,
    )

    fig.subplots_adjust(bottom=0.18, top=TOP_MARGIN)
    path = out_dir / "figure_03_class_distribution.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# FIGURE 4 - ABLATION COMPARISON
# ---------------------------------------------------------------------------

def load_ablation(repo_root: Path) -> Optional[pd.DataFrame]:
    path = discover_file(repo_root, [
        "reports/final/END_TO_END_ABLATION_SUMMARY.csv",
        "reports/final/*ABLATION*.csv",
        "reports/**/*ablation*.csv",
    ])
    if path is None:
        return None

    df = pd.read_csv(path)
    # Keep only directly comparable A/B/C arms from table1.
    table_col = find_column(df, ["table"])
    if table_col:
        df = df[df[table_col].astype(str).str.lower() == "table1"].reset_index(drop=True)
    model_col = find_column(df, ["arm", "label", "model", "configuration", "variant", "name"])
    pr_col = find_column(df, ["auc_pr", "pr_auc", "PR-AUC", "prauc"])
    f1_col = find_column(df, ["f1", "F1"])
    mcc_col = find_column(df, ["mcc", "MCC"])

    if not all([model_col, pr_col, f1_col, mcc_col]):
        print(f"[SKIP] Figure 4: could not infer columns in {path}")
        return None

    d = df[[model_col, pr_col, f1_col, mcc_col]].copy()
    d.columns = ["Model", "PR-AUC", "F1", "MCC"]

    # Prefer directly comparable A/B/C rows only.
    norm = d["Model"].astype(str).map(normalise_name)
    mask = (
        norm.str.contains("behavioralonly")
        | norm.str.contains("graphonly")
        | norm.str.contains("behavioralgraph")
        | norm.str.match(r"^a$")
        | norm.str.match(r"^b$")
        | norm.str.match(r"^c$")
    )
    if mask.any():
        d = d[mask].head(3)

    return d.head(3)


def figure_04_ablation(repo_root: Path, out_dir: Path) -> Optional[Path]:
    df = load_ablation(repo_root)
    if df is None or df.empty:
        print("[SKIP] Figure 4: ablation source not found.")
        return None

    x = np.arange(len(df))
    metrics = ["PR-AUC", "F1", "MCC"]
    width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[metric].astype(float), width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"].astype(str), rotation=12, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=0)
    ax.set_title("Figure 4. Leakage-Safe Ablation Comparison",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    fig.subplots_adjust(bottom=BOTTOM_MARGIN, top=TOP_MARGIN)
    path = out_dir / "figure_04_ablation_comparison.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# FIGURE 5A / 5B - ROC / PR CURVES
# ---------------------------------------------------------------------------

TRUE_COLUMNS = [
    "y_true", "label", "target", "is_malicious", "malicious", "ground_truth",
]
SCORE_COLUMNS = [
    "score", "probability", "prediction", "pred", "y_score",
    "ml_probability", "risk_score", "proba",
]


def detect_prediction_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    return find_column(df, TRUE_COLUMNS), find_column(df, SCORE_COLUMNS)


def infer_model_from_filename(path: Path) -> Optional[str]:
    name = normalise_name(path.stem)
    mapping = [
        ("lightgbmbenchmark", "LightGBM (benchmark)"),
        ("logistic", "Logistic Regression"),
        ("randomforest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("catboost", "CatBoost"),
        ("lightgbm", "LightGBM (frozen)"),
    ]
    for key, model in mapping:
        if key in name:
            return model
    return None


def discover_prediction_frames(repo_root: Path) -> dict[str, pd.DataFrame]:
    base = repo_root / "reports" / "artifacts" / "baseline_benchmark"
    if not base.exists():
        return {}

    files = list(base.rglob("*test*prediction*.parquet"))
    files += list(base.rglob("*test*prediction*.csv"))
    files += list(base.rglob("*test*pred*.parquet"))
    files += list(base.rglob("*test*pred*.csv"))
    files = sorted(set(files))

    frames: dict[str, pd.DataFrame] = {}
    for path in files:
        try:
            df = read_table(path)
        except Exception:
            continue

        y_col, score_col = detect_prediction_columns(df)
        if not y_col or not score_col:
            continue

        model = infer_model_from_filename(path)
        if model:
            frames[model] = df[[y_col, score_col]].rename(
                columns={y_col: "y_true", score_col: "score"}
            )

    return frames


def figure_05_curves(repo_root: Path, out_dir: Path) -> list[Path]:
    frames = discover_prediction_frames(repo_root)
    if not frames:
        print("[SKIP] Figure 5: no saved TEST prediction files found.")
        return []

    created: list[Path] = []

    # ROC - one plot, no subplot.
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for model in MODEL_ORDER:
        if model not in frames:
            continue
        d = frames[model].dropna()
        y = d["y_true"].astype(int).to_numpy()
        s = d["score"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        roc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, linewidth=LINE_WIDTH, label=f"{model} (AUC={roc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("True Positive Rate", fontsize=LABEL_SIZE)
    ax.set_title("Figure 5A. ROC Curves on Chronological TEST",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    path = out_dir / "figure_05a_roc_curves.png"
    save_figure(fig, path)
    created.append(path)

    # PR - separate plot.
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for model in MODEL_ORDER:
        if model not in frames:
            continue
        d = frames[model].dropna()
        y = d["y_true"].astype(int).to_numpy()
        s = d["score"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(recall, precision, linewidth=LINE_WIDTH, label=f"{model} (AP={ap:.3f})")
    ax.set_xlabel("Recall", fontsize=LABEL_SIZE)
    ax.set_ylabel("Precision", fontsize=LABEL_SIZE)
    ax.set_title("Figure 5B. Precision-Recall Curves on Chronological TEST",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(alpha=GRID_ALPHA)
    path = out_dir / "figure_05b_precision_recall_curves.png"
    save_figure(fig, path)
    created.append(path)

    return created


# ---------------------------------------------------------------------------
# FIGURE 6 - SHAP FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def load_shap_importance(repo_root: Path, explicit: Optional[Path]) -> Optional[pd.DataFrame]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend([
        repo_root / "reports" / "artifacts" / "phase11" / "global_shap_importance.csv",
        repo_root / "reports" / "artifacts" / "phase11_global_importance.csv",
    ])

    path = first_existing(candidates)
    if path is None:
        path = discover_file(repo_root, [
            "reports/**/*shap*importance*.csv",
            "reports/**/*shap*importance*.parquet",
            "reports/**/*global*importance*.csv",
            "reports/**/*global*importance*.parquet",
        ])
    if path is None:
        return None

    try:
        df = read_table(path)
    except Exception:
        return None

    feature_col = find_column(df, ["feature", "feature_name", "name"])
    value_col = find_column(df, [
        "mean_abs_shap", "mean_absolute_shap", "mean_abs_contribution",
        "importance", "shap_importance", "mean_abs_value",
    ])
    if not feature_col or not value_col:
        return None

    out = df[[feature_col, value_col]].copy()
    out.columns = ["Feature", "Importance"]
    out["Importance"] = pd.to_numeric(out["Importance"], errors="coerce")
    return out.dropna().sort_values("Importance", ascending=True)


def figure_06_shap(repo_root: Path, out_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    df = load_shap_importance(repo_root, explicit)
    if df is None or df.empty:
        print("[SKIP] Figure 6: authoritative SHAP importance artifact not found.")
        return None

    fig_height = max(6, 0.42 * len(df) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(df["Feature"], df["Importance"])
    ax.set_xlabel("Mean Absolute SHAP Contribution", fontsize=LABEL_SIZE)
    ax.set_title("Figure 6. Global SHAP Feature Importance",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="x", alpha=GRID_ALPHA)
    fig.subplots_adjust(left=0.30, top=TOP_MARGIN)

    path = out_dir / "figure_06_shap_feature_importance.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# FIGURE 7 - PHASE 20 DECISION DISTRIBUTION
# ---------------------------------------------------------------------------

def discover_decision_table(repo_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit and explicit.exists():
        return explicit
    return first_existing([
        repo_root / "demo_artifacts" / "final_user_day_decisions.parquet",
        repo_root / "reports" / "artifacts" / "phase20" / "final_user_day_decisions.parquet",
    ])


def figure_07_decisions(repo_root: Path, out_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    path = discover_decision_table(repo_root, explicit)
    if path is None:
        print("[SKIP] Figure 7: Phase 20 decision table not found.")
        return None

    df = pd.read_parquet(path)
    level_col = find_column(df, [
        "risk_level", "decision_level", "decision", "risk_category", "risk_band",
    ])
    if not level_col:
        print(f"[SKIP] Figure 7: risk-level column not found in {path}")
        return None

    preferred = ["ALERT", "BORDERLINE", "MONITOR", "NON-ALERT"]
    counts = df[level_col].astype(str).value_counts()
    counts.index = counts.index.str.upper()

    values = [int(counts.get(x, 0)) for x in preferred]
    total = sum(values)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(preferred, values)
    ax.set_ylabel("User-Day Count", fontsize=LABEL_SIZE)
    ax.set_title("Figure 7. Final Operational Decision Distribution",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    ymax = max(values) if values else 1
    for bar, value in zip(bars, values):
        pct = value / total if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.018,
            f"{value:,}\n({format_pct(pct, 1)})",
            ha="center", va="bottom", fontsize=VALUE_SIZE,
        )

    ax.text(
        0.5, -0.14,
        f"Complete Phase 20 population: {total:,} user-days",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    fig.subplots_adjust(bottom=0.18, top=TOP_MARGIN)

    out = out_dir / "figure_07_final_decision_distribution.png"
    save_figure(fig, out)
    return out


# ---------------------------------------------------------------------------
# FIGURE 8 - FIVE-MODEL BASELINE COMPARISON
# ---------------------------------------------------------------------------

def figure_08_baselines(baseline_df: pd.DataFrame, out_dir: Path) -> Path:
    df = choose_baseline_rows(baseline_df)

    model_col = find_column(df, ["Model"])
    metric_cols = {
        "PR-AUC": find_column(df, ["PR-AUC", "pr_auc"]),
        "F1": find_column(df, ["F1", "f1"]),
        "MCC": find_column(df, ["MCC", "mcc"]),
    }
    if not all(metric_cols.values()):
        raise ValueError("Baseline CSV is missing one or more required metrics: PR-AUC, F1, MCC.")

    models = [DISPLAY_NAME.get(x, x) for x in df[model_col].astype(str)]
    x = np.arange(len(models))
    width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, (label, col) in enumerate(metric_cols.items()):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        offsets = x + (i - 1) * width
        bars = ax.bar(offsets, vals, width, label=label)
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.set_ylim(0, max(0.42, float(
        np.nanmax([
            pd.to_numeric(df[col], errors="coerce").max()
            for col in metric_cols.values()
        ])
    ) + 0.08))
    ax.set_title("Figure 8. Post-Freeze Baseline Model Comparison",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.text(
        0.5, -0.19,
        "Same 12 features and chronological split; thresholds selected on CAL only.",
        transform=ax.transAxes, ha="center", fontsize=9,
    )

    fig.subplots_adjust(bottom=BOTTOM_MARGIN, top=TOP_MARGIN)
    path = out_dir / "figure_08_model_baseline_comparison.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# FIGURE 9 - CONFUSION COUNTS
# ---------------------------------------------------------------------------

def figure_09_confusion_counts(all_results: dict, out_dir: Path) -> Path:
    names = [m for m in MODEL_ORDER if m in all_results]

    values = {
        "TP": [all_results[m]["tp"] for m in names],
        "FP": [all_results[m]["fp"] for m in names],
        "FN": [all_results[m]["fn"] for m in names],
    }

    # TN is intentionally omitted from the grouped bars because ~47k TN values
    # would make TP/FP/FN unreadable. TN remains annotated below each model.
    x = np.arange(len(names))
    width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, (label, vals) in enumerate(values.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME.get(x, x) for x in names], fontsize=TICK_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.set_title("Figure 9. Confusion-Matrix Error/Detection Counts",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    tn_text = "   |   ".join(
        f"{DISPLAY_NAME.get(m, m).replace(chr(10), ' ')} TN={all_results[m]['tn']:,}"
        for m in names
    )
    ax.text(
        0.5, -0.20,
        tn_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
    )

    fig.subplots_adjust(bottom=0.24, top=TOP_MARGIN)
    path = out_dir / "figure_09_confusion_counts.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# EXTRA - BOOTSTRAP 95% CI COMPARISON
# ---------------------------------------------------------------------------

def figure_10_bootstrap_ci(ci_data: dict, out_dir: Path) -> Optional[Path]:
    models = ci_data.get("models", {})
    names = [m for m in [
        "Logistic Regression", "Random Forest", "XGBoost", "CatBoost", "LightGBM"
    ] if m in models]
    if not names:
        return None

    metric = "f1"
    points = [models[m][metric]["point_estimate"] for m in names]
    low = [models[m][metric]["ci_lower"] for m in names]
    high = [models[m][metric]["ci_upper"] for m in names]
    yerr = np.array([
        np.array(points) - np.array(low),
        np.array(high) - np.array(points),
    ])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(names))
    ax.errorbar(x, points, yerr=yerr, fmt="o", capsize=5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DISPLAY_NAME.get(m, m).replace("\n", " ") for m in names],
        rotation=15, ha="right", fontsize=TICK_SIZE,
    )
    ax.set_ylabel("F1", fontsize=LABEL_SIZE)
    ax.set_title(
        f"Supplementary Figure. F1 Bootstrap {int(ci_data.get('confidence_level', .95)*100)}% Confidence Intervals",
        fontsize=TITLE_SIZE, fontweight="bold",
    )
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.subplots_adjust(bottom=BOTTOM_MARGIN, top=TOP_MARGIN)

    path = out_dir / "figure_10_bootstrap_f1_ci.png"
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ISM Project Shafe report figures.")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--baseline-csv", type=Path, default=None)
    p.add_argument("--all-results", type=Path, default=None)
    p.add_argument("--freeze", type=Path, default=None)
    p.add_argument("--bootstrap-ci", type=Path, default=None)
    p.add_argument("--shap-importance", type=Path, default=None)
    p.add_argument("--decision-table", type=Path, default=None)
    p.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any repository-dependent figure cannot be generated.",
    )
    return p.parse_args()


def resolve_inputs(args: argparse.Namespace) -> dict[str, Optional[Path]]:
    root = args.repo_root.resolve()

    def resolve(explicit: Optional[Path], candidates: list[Path], patterns: list[str]) -> Optional[Path]:
        if explicit:
            p = explicit if explicit.is_absolute() else root / explicit
            if p.exists():
                return p
        p = first_existing(candidates)
        if p:
            return p
        return discover_file(root, patterns)

    return {
        "baseline_csv": resolve(
            args.baseline_csv,
            [
                root / "reports" / "final" / "BASELINE_MODEL_COMPARISON.csv",
                root / "reports" / "artifacts" / "baseline_benchmark" / "BASELINE_MODEL_COMPARISON.csv",
            ],
            ["**/BASELINE_MODEL_COMPARISON.csv"],
        ),
        "all_results": resolve(
            args.all_results,
            [
                root / "reports" / "artifacts" / "baseline_benchmark" / "all_results.json",
            ],
            ["**/all_results.json"],
        ),
        "freeze": resolve(
            args.freeze,
            [
                root / "reports" / "artifacts" / "baseline_benchmark" / "BENCHMARK_FREEZE.json",
            ],
            ["**/BENCHMARK_FREEZE.json"],
        ),
        "bootstrap_ci": resolve(
            args.bootstrap_ci,
            [
                root / "reports" / "artifacts" / "baseline_benchmark" / "bootstrap_confidence_intervals.json",
            ],
            ["**/bootstrap_confidence_intervals.json"],
        ),
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    out_dir = (args.out_dir or (repo_root / "reports" / "figures")).resolve()
    ensure_dir(out_dir)

    paths = resolve_inputs(args)
    print("Resolved inputs:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    required = ["baseline_csv", "all_results", "freeze", "bootstrap_ci"]
    missing_required = [k for k in required if paths[k] is None]
    if missing_required:
        raise FileNotFoundError(
            "Missing required benchmark inputs: " + ", ".join(missing_required)
        )

    baseline_df = pd.read_csv(paths["baseline_csv"])
    all_results = load_json(paths["all_results"])
    freeze = load_json(paths["freeze"])
    ci_data = load_json(paths["bootstrap_ci"])

    created: list[Path] = []
    skipped: list[str] = []

    created.append(figure_01_architecture(out_dir))
    created.append(figure_02_split(freeze, out_dir))
    created.append(figure_03_class_distribution(freeze, out_dir))

    p = figure_04_ablation(repo_root, out_dir)
    if p:
        created.append(p)
    else:
        skipped.append("Figure 4: ablation source unavailable/unreadable")

    curve_paths = figure_05_curves(repo_root, out_dir)
    if curve_paths:
        created.extend(curve_paths)
    else:
        skipped.append("Figure 5: TEST prediction artifacts unavailable")

    shap_explicit = args.shap_importance
    if shap_explicit and not shap_explicit.is_absolute():
        shap_explicit = repo_root / shap_explicit
    p = figure_06_shap(repo_root, out_dir, shap_explicit)
    if p:
        created.append(p)
    else:
        skipped.append("Figure 6: SHAP importance artifact unavailable")

    decision_explicit = args.decision_table
    if decision_explicit and not decision_explicit.is_absolute():
        decision_explicit = repo_root / decision_explicit
    p = figure_07_decisions(repo_root, out_dir, decision_explicit)
    if p:
        created.append(p)
    else:
        skipped.append("Figure 7: Phase 20 decision table unavailable")

    created.append(figure_08_baselines(baseline_df, out_dir))
    created.append(figure_09_confusion_counts(all_results, out_dir))

    p = figure_10_bootstrap_ci(ci_data, out_dir)
    if p:
        created.append(p)

    # Write a simple reproducibility manifest.
    manifest = {
        "generator": "scripts/generate_report_figures.py",
        "repo_root": str(repo_root),
        "inputs": {k: str(v) if v else None for k, v in paths.items()},
        "created": [str(p) for p in created],
        "skipped": skipped,
        "note": "No model fitting or threshold tuning occurs in this script.",
    }
    manifest_path = out_dir / "figure_generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[CREATED] {manifest_path}")

    print("\nSummary")
    print(f"  created: {len(created)}")
    print(f"  skipped: {len(skipped)}")
    for item in skipped:
        print(f"    - {item}")

    if args.strict and skipped:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
