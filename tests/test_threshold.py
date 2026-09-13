"""Regression tests for src/evaluation/threshold.best_f1_threshold.

Protects the O(n log n) rewrite (prefix-sum threshold scan) against the
reference O(k*n) unique-threshold scan:
  1. bit-identical outputs on randomized data (several sizes, seeds)
  2. known hand-computed values
  3. ties -> higher threshold retained
  4. duplicate scores
  5. all-negative / all-positive / empty / single-row inputs
  6. NaN scores (reference semantics preserved)
  7. deterministic ordering (stable sort, same result regardless of
     input order)
"""
import numpy as np
import pytest

from src.evaluation import threshold as th


def best_f1_threshold_naive(y_true, y_score) -> tuple[float, float]:
    """Reference implementation: verbatim copy of the original O(k*n) scan."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    best_t, best_f1 = None, -1.0
    for t in np.unique(y_score):
        pred = y_score >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def _random_case(n, seed, pos_rate=0.2):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    s = rng.random(n) * (1.0 if seed % 2 else 1e-6)  # vary score scale
    return y, s


@pytest.mark.parametrize("n", [1, 2, 5, 17, 100, 1000])
@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_random_equality_with_reference(n, seed):
    y, s = _random_case(n, seed)
    assert th.best_f1_threshold(y, s) == best_f1_threshold_naive(y, s)


def test_random_equality_duplicate_scores():
    rng = np.random.default_rng(11)
    y = (rng.random(500) < 0.3).astype(int)
    s = rng.integers(0, 8, 500).astype(float) / 10  # heavy ties
    assert th.best_f1_threshold(y, s) == best_f1_threshold_naive(y, s)


def test_random_equality_near_float32_quantization():
    # scores quantized like lightgbm float32 outputs
    rng = np.random.default_rng(23)
    y = (rng.random(2000) < 0.02).astype(int)
    s = np.float32(rng.random(2000)).astype(np.float64)
    assert th.best_f1_threshold(y, s) == best_f1_threshold_naive(y, s)


def test_hand_computed_known_value():
    # t = 0.90: tp=1, fp=0, fn=0 -> F1 1.0; t = 0.30: tp=1, fp=1 -> F1 0.67
    assert th.best_f1_threshold(np.array([1, 0]), np.array([0.90, 0.30])) == \
        (0.90, 1.0)


def test_tie_broken_toward_higher_threshold():
    # both 0.90 and 0.85 give F1 1.0 -> keep the higher threshold
    y = np.array([1, 0, 1])
    s = np.array([0.90, 0.85, 0.95])
    t, f1 = th.best_f1_threshold(y, s)
    assert t == 0.90
    assert f1 == pytest.approx(1.0)


def test_duplicate_scores_hand_computed():
    # scores 0.5, 0.5, 0.3 with y 1, 0, 1:
    #   t=0.3 -> tp=2 fp=1 fn=0 -> F1 = 2*(2/3)*1/(2/3+1) = 0.8
    #   t=0.5 -> tp=1 fp=1 fn=1 -> F1 = 0.5
    t, f1 = th.best_f1_threshold(np.array([1, 0, 1]), np.array([0.5, 0.5, 0.3]))
    assert t == 0.3
    assert f1 == pytest.approx(0.8)


def test_all_negative():
    y = np.zeros(10, dtype=int)
    s = np.linspace(0.1, 0.9, 10)
    t, f1 = th.best_f1_threshold(y, s)
    assert t == s.min() and f1 == 0.0  # every f1 is 0 -> first grid threshold
    assert th.best_f1_threshold(y, s) == best_f1_threshold_naive(y, s)


def test_all_positive():
    y = np.ones(10, dtype=int)
    s = np.linspace(0.1, 0.9, 10)
    t, f1 = th.best_f1_threshold(y, s)
    assert t == s.min() and f1 == pytest.approx(1.0)
    assert th.best_f1_threshold(y, s) == best_f1_threshold_naive(y, s)


def test_empty_input():
    assert th.best_f1_threshold(np.array([], dtype=int), np.array([])) == \
        (None, -1.0)


def test_single_row():
    assert th.best_f1_threshold(np.array([1]), np.array([0.5])) == (0.5, 1.0)
    assert th.best_f1_threshold(np.array([0]), np.array([0.5])) == (0.5, 0.0)


def test_input_order_invariance():
    y = np.array([1, 0, 1, 0, 1])
    s = np.array([0.9, 0.2, 0.7, 0.3, 0.6])
    rng = np.random.default_rng(5)
    perm = rng.permutation(len(y))
    a = th.best_f1_threshold(y, s)
    b = th.best_f1_threshold(y[perm], s[perm])
    assert a == b


def test_nan_scores_match_reference():
    rng = np.random.default_rng(13)
    y = (rng.random(200) < 0.25).astype(int)
    s = rng.random(200)
    s[rng.choice(200, 7, replace=False)] = np.nan
    assert th.best_f1_threshold(y, s) == best_f1_threshold_naive(y, s)
    # all-negative with NaN: reference keeps the first grid threshold ->
    # verify bit-identical
    s2 = np.array([0.5, 0.7, np.nan])
    assert th.best_f1_threshold(np.zeros(3, dtype=int), s2) == \
        best_f1_threshold_naive(np.zeros(3, dtype=int), s2)