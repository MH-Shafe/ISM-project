"""Generate 7 publication-quality figures for the ISM Master Report.

All data is taken from verified project artifacts (OBSERVED).
No raw dataset is accessed; no model inference is performed.

Source artifacts:
  - reports/artifacts/phase7_freeze_lgbm-graph-v1.json
  - reports/artifacts/phase9_freeze.json
  - reports/artifacts/phase11_global_importance.json
  - reports/artifacts/phase20/final_user_day_decisions_summary.json
  - reports/artifacts/phase6_cost.json
  - reports/ISM_MASTER_REPORT.md (Section 9 performance table)
"""

import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = pathlib.Path(__file__).parent
REPO = OUT.parent.parent

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ---------------------------------------------------------------------------
# Figure 1: System Architecture (pipeline flow)
# ---------------------------------------------------------------------------

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Figure 1. System Architecture — lgbm-graph-v1 Pipeline", fontweight="bold", pad=12)

    boxes = [
        (0.5, 3.5, "CERT r4.2\n7 logs + LDAP\n501K user-days", "#d4e6f1"),
        (2.5, 3.5, "Validation &\nAggregation\n(day-local, 28d lag)", "#d5f5e3"),
        (4.5, 4.2, "8 Behavioral\nFeatures\n(Phase 4 frozen)", "#fcf3cf"),
        (4.5, 2.8, "4 Graph\nFeatures\n(Phase 5 frozen)", "#fadbd8"),
        (6.5, 3.5, "LightGBM\nlgbm-graph-v1\n12 features, 186 iter", "#f9e79f"),
        (8.5, 3.5, "Alert Policy\nmax-F1 threshold\n0.9186", "#f5b7b1"),
        (10.0, 4.3, "Conformal\n(overlay)", "#d6eaf8"),
        (10.0, 2.7, "Explainability\n(SHAP, 3 reasons)", "#e8daef"),
        (11.2, 3.5, "Decision\nEngine\n(501K rows)", "#d5f5e3"),
    ]

    for x, y, text, color in boxes:
        w, h = 1.6, 1.0
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor="gray",
                                        linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
                linespacing=1.3)

    arrows = [
        (1.3, 3.5, 1.7, 3.5),
        (3.3, 3.8, 3.7, 4.2),
        (3.3, 3.2, 3.7, 2.8),
        (5.3, 4.2, 5.7, 3.7),
        (5.3, 2.8, 5.7, 3.3),
        (7.3, 3.5, 7.7, 3.5),
        (9.3, 3.7, 9.2, 4.3),
        (9.3, 3.3, 9.2, 2.7),
        (10.8, 4.0, 10.6, 3.7),
        (10.8, 3.0, 10.6, 3.3),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2))

    ax.text(6.0, 0.6, "Frozen since Phase 7/9. No ML change since. Adaptive Risk rejected (Phase 19 FAIL).",
            ha="center", va="center", fontsize=8, style="italic", color="#555")

    fig.savefig(OUT / "fig1_system_architecture.png")
    plt.close(fig)
    print("  fig1_system_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2: Chronological Split Timeline
# ---------------------------------------------------------------------------

def fig2_chronological_split():
    fig, ax = plt.subplots(figsize=(10, 4))

    splits = [
        ("TRAIN", "2010-01-02", "2011-01-31", 395000, 1539, "#2ecc71"),
        ("CALIBRATION", "2011-02-01", "2011-03-31", 59000, 323, "#3498db"),
        ("TEST", "2011-04-01", "2011-05-17", 47000, 30, "#e74c3c"),
    ]

    from datetime import datetime
    t0 = datetime(2010, 1, 2).toordinal()
    t_end = datetime(2011, 5, 17).toordinal()
    total = t_end - t0

    y = 0.5
    h = 0.35
    for name, d_start, d_end, rows, pos, color in splits:
        ds = datetime.strptime(d_start, "%Y-%m-%d").toordinal()
        de = datetime.strptime(d_end, "%Y-%m-%d").toordinal()
        x0 = (ds - t0) / total
        w = (de - ds) / total
        rect = mpatches.FancyBboxPatch((x0, y - h/2), w, h,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor="gray",
                                        alpha=0.85)
        ax.add_patch(rect)
        ax.text(x0 + w/2, y + 0.02, name, ha="center", va="bottom",
                fontsize=9, fontweight="bold")
        ax.text(x0 + w/2, y - 0.02, f"{rows:,} rows\n{pos} pos",
                ha="center", va="top", fontsize=7.5)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    ax.set_title("Figure 2. Chronological Split Strategy (user x day)", fontweight="bold", pad=10)

    ax.annotate("No random split. Temporal ordering preserved.", xy=(0.5, 0.05),
                xycoords="axes fraction", ha="center", fontsize=8, style="italic", color="#555")

    fig.savefig(OUT / "fig2_chronological_split.png")
    plt.close(fig)
    print("  fig2_chronological_split.png")


# ---------------------------------------------------------------------------
# Figure 3: Class Distribution Across Splits
# ---------------------------------------------------------------------------

def fig3_class_distribution():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    data = [
        ("TRAIN", 395000, 1539, "#2ecc71"),
        ("CALIBRATION", 59000, 323, "#3498db"),
        ("TEST", 47000, 30, "#e74c3c"),
    ]

    for ax, (name, total, pos, color) in zip(axes, data):
        neg = total - pos
        bars = ax.bar(["Non-malicious", "Malicious"], [neg, pos],
                      color=["#bdc3c7", color], edgecolor="gray", linewidth=0.8)
        ax.set_title(f"{name}\n({total:,} rows)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        ax.set_ylim(1, total * 2)
        for bar, val in zip(bars, [neg, pos]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                    f"{val:,}", ha="center", va="bottom", fontsize=8)
        prevalence = pos / total * 100
        ax.text(0.5, 0.88, f"Prevalence: {prevalence:.2f}%",
                transform=ax.transAxes, ha="center", fontsize=8, color="#555")

    fig.suptitle("Figure 3. Class Distribution Across Splits (0.38% overall prevalence)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_class_distribution.png")
    plt.close(fig)
    print("  fig3_class_distribution.png")


# ---------------------------------------------------------------------------
# Figure 4: Ablation Comparison (Phase 6 arms)
# ---------------------------------------------------------------------------

def fig4_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    arms = {
        "A: Behavioral\n(8 features)": {"auc_roc": 0.93534, "auc_pr": 0.15304, "f1": 0.346, "alerts": 22},
        "B: Graph-only\n(5 features)": {"auc_roc": 0.78635, "auc_pr": 0.01431, "f1": 0.054, "alerts": 417},
        "C: Merged\n(12 features)": {"auc_roc": 0.93916, "auc_pr": 0.26778, "f1": 0.354, "alerts": 49},
    }
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    metrics = [
        ("AUC-ROC", "auc_roc", [0.7, 1.0]),
        ("AUC-PR", "auc_pr", [0, 0.3]),
        ("F1 Score", "f1", [0, 0.4]),
    ]

    for ax, (label, key, ylim) in zip(axes, metrics):
        vals = [d[key] for d in arms.values()]
        bars = ax.bar(range(3), vals, color=colors, edgecolor="gray", linewidth=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["A", "B", "C"], fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(ylim)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ylim[1] - ylim[0]) * 0.02,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontweight="bold")

    fig.suptitle("Figure 4. Phase 6 Ablation — Directly Comparable Arms (TEST)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ablation_comparison.png")
    plt.close(fig)
    print("  fig4_ablation_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5: SHAP Feature Importance
# ---------------------------------------------------------------------------

def fig5_feature_importance():
    importance = {
        "http_activity_count": 1.4210,
        "device_consistency_score": 0.4060,
        "usb_connection_count": 0.2222,
        "file_access_count": 0.2205,
        "login_count": 0.2108,
        "after_hours_login_count": 0.1321,
        "unique_device_count": 0.0941,
        "sensitive_file_access_count": 0.0878,
        "file_type_consistency_score": 0.0445,
        "rare_device_usage_count": 0.0369,
        "unusual_access_count": 0.0317,
        "rare_file_type_access_count": 0.0019,
    }

    sorted_items = sorted(importance.items(), key=lambda x: x[1])
    names = [k.replace("_count", "").replace("_score", " score") for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vals)))
    bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Figure 5. Global Feature Importance (SHAP, CAL + TRAIN)", fontweight="bold", pad=10)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7.5)

    ax.text(0.98, 0.02, "Source: phase11_global_importance.json\nKendall tau (gain vs SHAP): 0.73",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, style="italic", color="#777")

    fig.tight_layout()
    fig.savefig(OUT / "fig5_feature_importance.png")
    plt.close(fig)
    print("  fig5_feature_importance.png")


# ---------------------------------------------------------------------------
# Figure 6: Decision Level Distribution (Phase 20)
# ---------------------------------------------------------------------------

def fig6_decision_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Risk level distribution
    risk = {"ALERT": 3785, "BORDERLINE": 1484, "MONITOR": 178204, "NON-ALERT": 317527}
    risk_colors = ["#e74c3c", "#f39c12", "#3498db", "#bdc3c7"]
    labels_r = list(risk.keys())
    sizes_r = list(risk.values())
    wedges, texts, autotexts = axes[0].pie(
        sizes_r, labels=labels_r, colors=risk_colors,
        autopct=lambda p: f"{p:.2f}%\n({int(round(p*sum(sizes_r)/100)):,})",
        startangle=90, textprops={"fontsize": 8})
    axes[0].set_title("Risk Level", fontweight="bold")

    # Right: Confidence distribution
    conf = {"high-confidence": 61017, "ambiguous": 439983}
    conf_colors = ["#2ecc71", "#f39c12"]
    sizes_c = list(conf.values())
    wedges2, texts2, autotexts2 = axes[1].pie(
        sizes_c, labels=list(conf.keys()), colors=conf_colors,
        autopct=lambda p: f"{p:.2f}%\n({int(round(p*sum(sizes_c)/100)):,})",
        startangle=90, textprops={"fontsize": 8})
    axes[1].set_title("Conformal Confidence", fontweight="bold")

    fig.suptitle("Figure 6. Phase 20 Decision Distribution (501,000 user-days)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_decision_distribution.png")
    plt.close(fig)
    print("  fig6_decision_distribution.png")


# ---------------------------------------------------------------------------
# Figure 7: Model Size & Runtime Comparison
# ---------------------------------------------------------------------------

def fig7_runtime_modelsize():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    models = ["Baseline\n(6f, 182t)", "Arm A\n(8f, 160t)", "Arm B\n(5f, graph)", "Arm C\n(13f, 186t)", "lgbm-graph-v1\n(12f, 186t)"]
    train_t = [4.36, 4.13, 1.38, 4.76, 4.64]
    predict_t = [None, 1.06, 0.048, 1.21, 1.29]
    model_bytes = [635440, 559479, 13357, 651085, 651047]

    x = np.arange(len(models))
    w = 0.35

    # Left: Runtime
    bars1 = axes[0].bar(x - w/2, train_t, w, label="Train (s)", color="#3498db", edgecolor="gray")
    pred_vals = [p if p is not None else 0 for p in predict_t]
    bars2 = axes[0].bar(x + w/2, pred_vals, w, label="Predict TEST (s)", color="#e74c3c", edgecolor="gray")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=7)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Training & Inference Time", fontweight="bold")
    axes[0].legend(fontsize=8)
    for bar, val in zip(bars1, train_t):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, pred_vals):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Right: Model size
    axes[1].bar(x, [b / 1024 for b in model_bytes], color=["#95a5a6", "#3498db", "#e74c3c", "#f39c12", "#2ecc71"],
                edgecolor="gray")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=7)
    axes[1].set_ylabel("KB")
    axes[1].set_title("Model Size", fontweight="bold")
    for i, b in enumerate(model_bytes):
        axes[1].text(i, b / 1024 + 5, f"{b/1024:.0f} KB", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Figure 7. Computational Cost Comparison (OBSERVED, no GPU)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_runtime_modelsize.png")
    plt.close(fig)
    print("  fig7_runtime_modelsize.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating ISM Master Report figures...")
    fig1_architecture()
    fig2_chronological_split()
    fig3_class_distribution()
    fig4_ablation()
    fig5_feature_importance()
    fig6_decision_distribution()
    fig7_runtime_modelsize()
    print("Done. 7 figures saved to", OUT)
