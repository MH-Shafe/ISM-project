"""Model evaluation metrics for the user-day insider-threat task.

All functions are pure and deterministic; they never look at features or
table structure, only at labels and scores, so they cannot leak.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


def classification_metrics(y_true, y_score) -> dict:
    """Threshold-free metrics. AUC-PR = average precision (positive class)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
    }


def top_k_metrics(y_true, y_score, ks=(10, 30, 50, 100)) -> dict:
    """Precision/recall at the top-k scored rows (threshold-free)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    n_pos = max(1, int(y_true.sum()))
    out = {}
    for k in ks:
        k = min(k, len(y_true))
        hits = int(y_true[order[:k]].sum())
        out[f"precision_at_{k}"] = hits / max(1, k)
        out[f"recall_at_{k}"] = hits / n_pos
    return out


def confusion_at_threshold(y_true, y_score, threshold: float) -> dict:
    """Precision / recall / F1 at a fixed decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_score) >= threshold
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_alerts": int(pred.sum()),
    }


def binary_decision_metrics(y_true, y_score, threshold: float) -> dict:
    """Full confusion-matrix metric set at a fixed decision threshold.

    Adds MCC, balanced accuracy, FPR, FNR, and alert rate to the
    precision/recall/F1 point already provided by confusion_at_threshold.
    """
    base = confusion_at_threshold(y_true, y_score, threshold)
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_score) >= threshold
    tn = int((~pred & (y_true == 0)).sum())
    tp, fp, fn = base["tp"], base["fp"], base["fn"]
    n = max(1, len(y_true))
    out = dict(base)
    out["tn"] = tn
    out["mcc"] = float(matthews_corrcoef(y_true, pred))
    out["balanced_accuracy"] = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    out["fpr"] = fp / max(1, fp + tn)
    out["fnr"] = fn / max(1, fn + tp)
    out["alert_rate"] = (tp + fp) / n
    return out


def bootstrap_ci(y_true, y_score, metric: str = "auc_roc", n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI of an evaluation metric over resampled rows.

    Evaluation-only uncertainty estimation: resamples the observed labels
    and scores, never refits the model or touches the threshold.

    metric: "auc_roc" or "auc_pr". Resamples lacking a positive class are
    skipped (metric undefined), and the count is reported.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    values = []
    skipped = 0
    for _ in range(n_boot):
        boot = rng.choice(idx, size=n, replace=True)
        t, s = y_true[boot], y_score[boot]
        if t.sum() == 0 or t.sum() == len(t):
            skipped += 1
            continue
        if metric == "auc_roc":
            values.append(roc_auc_score(t, s))
        elif metric == "auc_pr":
            values.append(average_precision_score(t, s))
        else:
            raise ValueError(f"unsupported metric: {metric}")
    values = np.asarray(values)
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return {
        "metric": metric,
        "n_boot": n_boot,
        "n_valid": int(len(values)),
        "n_skipped": skipped,
        "ci_low": lo,
        "ci_high": hi,
        "mean": float(values.mean()),
    }


def paired_bootstrap_delta(y_true, y_score_base, y_score_new,
                           metrics=("auc_roc", "auc_pr"), n_boot: int = 1000,
                           seed: int = 42, alpha: float = 0.05) -> dict:
    """Paired percentile bootstrap of metric(new) - metric(base).

    Evaluation-only uncertainty: TEST rows are resampled jointly so both
    arms face the same rows; the models are never refit and no threshold
    is touched. Descriptive uncertainty only -- with few positive TEST
    rows it must not be read as a significance test.
    """
    y_true = np.asarray(y_true, dtype=int)
    s_base = np.asarray(y_score_base, dtype=float)
    s_new = np.asarray(y_score_new, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    out = {}
    for metric in metrics:
        values = []
        skipped = 0
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            t = y_true[idx]
            if t.sum() == 0 or t.sum() == len(t):
                skipped += 1
                continue
            if metric == "auc_roc":
                va = roc_auc_score(t, s_base[idx])
                vb = roc_auc_score(t, s_new[idx])
            elif metric == "auc_pr":
                va = average_precision_score(t, s_base[idx])
                vb = average_precision_score(t, s_new[idx])
            else:
                raise ValueError(f"unsupported metric: {metric}")
            values.append(vb - va)
        values = np.asarray(values)
        lo = float(np.percentile(values, 100 * alpha / 2))
        hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
        out[metric] = {
            "metric": metric,
            "n_boot": n_boot,
            "n_valid": int(len(values)),
            "n_skipped": skipped,
            "ci_low": lo,
            "ci_high": hi,
            "mean": float(values.mean()),
            "frac_gt_0": float((values > 0).mean()),
        }
    return out


def per_scenario_recall(y_true, y_score, scenario_of_user, keys, threshold: float) -> dict:
    """Recall per CERT scenario on a given split (evaluation only).

    scenario_of_user: dict user -> scenario (from the answer key; used only
    to stratify evaluation, never as a feature).
    """
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_score) >= threshold
    users = np.asarray(keys["user"])
    out = {}
    for scenario in sorted({v for v in scenario_of_user.values()}):
        mask = np.array([scenario_of_user.get(u, -1) == scenario for u in users])
        n_pos = int((mask & (y_true == 1)).sum())
        if n_pos == 0:
            continue
        out[str(scenario)] = {
            "n_malicious_rows": n_pos,
            "n_users": int(mask.sum()),
            "recall": float((mask & pred & (y_true == 1)).sum()) / n_pos,
        }
    return out