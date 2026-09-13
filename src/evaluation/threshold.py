"""Threshold selection on CALIBRATION only (protocol: never on TEST)."""
from __future__ import annotations

import numpy as np


def best_f1_threshold(y_true, y_score) -> tuple[float, float]:
    """Threshold maximizing F1 on the calibration scores.

    Returns (threshold, best_f1). Ties are broken toward the higher
    threshold (fewer alerts).

    Implementation: O(n log n) — a stable ascending sort plus one O(k)
    scan over unique score values using a cumulative-positive prefix.
    Each candidate threshold is evaluated with the exact reference
    semantics (ascending unique-score grid, `pred = score >= t`, and a
    strict-greater F1 update with the 1e-12 tolerance), so outputs are
    identical to the original O(k*n) unique-threshold scan.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_score)
    if n == 0:
        return None, -1.0
    order = np.argsort(y_score, kind="stable")
    s = y_score[order]
    y = y_true[order]
    n_nan = int(np.isnan(s).sum())
    n_nan_pos = int(y[n - n_nan:].sum())
    total_pos = int(y.sum())
    total_neg = n - total_pos
    cum_pos = np.cumsum(y)
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    best_t, best_f1 = None, -1.0
    for i in starts:
        if s[i] != s[i]:  # NaN threshold: no row is >= NaN (reference semantics)
            tp = 0
            fp = 0
            fn = total_pos
        else:
            pos_below = cum_pos[i - 1] if i > 0 else 0
            tp = total_pos - n_nan_pos - pos_below
            fp = total_neg - (i - pos_below) - (n_nan - n_nan_pos)
            fn = pos_below + n_nan_pos
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, float(s[i])
    return float(best_t), float(best_f1)


def threshold_at_precision(y_true, y_score, min_precision: float = 0.5) -> float:
    """Highest threshold with calibration precision >= min_precision.

    Falls back to the score of the single highest-ranked positive if the
    target precision is unreachable on calibration.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    tp = 0
    for rank, idx in enumerate(order, start=1):
        tp += int(y_true[idx])
        if tp / rank >= min_precision:
            return float(y_score[idx])
    pos_idx = order[y_true[order] == 1][0] if (y_true == 1).any() else order[0]
    return float(y_score[pos_idx])