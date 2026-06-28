"""Tests for reservoirs.baselines, the honest-comparison API (numpy-only HARD deps).

These bite: they pin a hand-computed chance value (and deliberately use a case where chance
does NOT win), pin the strong no-reservoir linear bar on linearly-separable data plus the
honest real-Iris ceiling, check the z-guard returns ~0 exactly when the score ignores
wiring (and a finite z when it detects wiring-dependence), pin compare_classification's exact
hand-computed deltas, and exercise both shapes of format_comparison plus its error branch.

Hard deps: numpy + pytest + stdlib only. scikit-learn is touched solely via
``pytest.importorskip`` so the suite stays green with or without it.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, REPO)


# data helpers (numpy-only)
def _load_iris_numpy():
    """Load iris.csv with numpy only. Returns (X[150,4] float, y[150] str) or None if missing.

    Columns: col0=Id (skipped), cols1-4 = float features, col5 = Species string labels.
    """
    path = os.path.join(REPO, "iris.csv")
    if not os.path.exists(path):
        return None
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    X = raw[:, 1:5].astype(float)
    y = raw[:, 5]
    return X, y


def _separable_3class(seed=0, n_per=50):
    """A clearly linearly-separable 3-class problem with ORIGIN-AVOIDING centers.

    The no-bias default of linear_readout_baseline cannot separate clusters that differ only
    in magnitude from the origin, so the centers are placed away from [0,0] (load-bearing:
    verified 1.0 accuracy across seeds under the no-bias default).
    """
    g = np.random.default_rng(seed)
    centers = np.array([[8.0, 8.0], [-8.0, 8.0], [0.0, -8.0]])
    X = np.vstack([g.normal(c, 0.6, (n_per, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n_per)
    idx = g.permutation(X.shape[0])
    return X[idx], y[idx]


def _sym_weighted_adj(n=30, p=0.25, seed=0):
    """Symmetric, weighted, zero-diagonal adjacency (same recipe as tests/test_nulls._adj)."""
    g = np.random.default_rng(seed)
    M = (g.random((n, n)) < p).astype(float) * g.uniform(0.5, 2.0, (n, n))
    M = np.triu(M, 1)
    return M + M.T


# A. majority_class_baseline
def test_majority_matches_hand_computed_and_is_the_true_mode():
    """A1: predicted_label is the TRUE train mode (0), not the best-on-test label."""
    from reservoirs.baselines import majority_class_baseline
    y_train = np.array([0, 0, 0, 1, 1, 2])  # mode is 0 (count 3 > 2 > 1)
    y_test = np.array([0, 0, 1, 2, 2, 2])
    out = majority_class_baseline(y_train, y_test)
    assert out["predicted_label"] == 0, "predicted_label must be the TRAIN mode"
    assert abs(out["accuracy"] - 2.0 / 6.0) < 1e-12, "hand-computed: 2 of 6 test labels are 0"


def test_majority_does_not_assume_it_wins():
    """A2: chance pinned to its true (poor) value. 'Always predict 0' is the WRONG call on
    this test set, where class 2 dominates. Catches code that secretly picks best-on-test."""
    from reservoirs.baselines import majority_class_baseline
    y_train = np.array([0, 0, 0, 1, 1, 2])
    y_test = np.array([2, 2, 2, 2, 0, 1])
    out = majority_class_baseline(y_train, y_test)
    assert out["predicted_label"] == 0
    assert abs(out["accuracy"] - 1.0 / 6.0) < 1e-12  # only one 0 in y_test


def test_majority_string_labels_and_defaults_to_train():
    """A3: string labels + y_test defaults to y_train."""
    from reservoirs.baselines import majority_class_baseline
    y_train = np.array(["b", "a", "a", "c", "a"])
    out = majority_class_baseline(y_train)  # y_test defaults to y_train
    assert out["predicted_label"] == "a"
    assert abs(out["accuracy"] - 3.0 / 5.0) < 1e-12


# B. linear_readout_baseline
def test_linear_readout_is_strong_on_separable_data():
    """B1: on cleanly separable data the no-reservoir linear model clears 0.9 (observed 1.0),
    carrying the spec's >0.9 claim via the spec's own synthetic fallback (DEFAULTS, no bias)."""
    from reservoirs.baselines import linear_readout_baseline
    X, y = _separable_3class(seed=0)
    cut = int(0.7 * len(X))
    out = linear_readout_baseline(X[:cut], y[:cut], X[cut:], y[cut:], ridge=1.0)
    assert out["accuracy"] > 0.9, f"separable linear bar should exceed 0.9, got {out['accuracy']:.3f}"
    assert set(out["classes"]) == {0, 1, 2}
    assert out["predictions"].shape[0] == len(X) - cut


def test_linear_readout_on_real_iris_is_a_strong_bar():
    """B2: on real 3-class Iris the no-reservoir linear readout is strong (well above chance).

    NOTE: one-hot least-squares is subject to the multiclass *masking* problem and AVERAGES
    around ~0.80 on Iris (individual splits can exceed 0.9, but it trails logistic on average).
    So we assert a robust >0.75 floor AND that it crushes chance (>chance+0.3), the honest claim.
    (`logistic_readout_baseline` is the optional stronger linear-style bar.)"""
    from reservoirs.baselines import linear_readout_baseline, majority_class_baseline
    iris = _load_iris_numpy()
    if iris is None:
        pytest.skip("iris.csv not available")
    X, y = iris
    ii = np.arange(150)
    te = ii % 5 == 0  # 30 test, evenly stratified
    tr = ~te
    acc = linear_readout_baseline(X[tr], y[tr], X[te], y[te], ridge=1.0)["accuracy"]
    chance = majority_class_baseline(y[tr], y[te])["accuracy"]  # = 1/3
    assert acc > 0.75, f"linear bar on Iris should clear 0.75, got {acc:.3f}"
    assert acc > chance + 0.3, "linear readout must crush majority-class chance on Iris"


# C. reservoir_vs_null
def test_null_z_is_zero_when_score_ignores_wiring():
    """C1: total edge weight is preserved by the degree/weight-preserving null, so every null
    scores essentially identically to the real graph -> z must be ~0. The null spread is only
    float summation jitter (~5e-14); the guard must NOT divide that ~0 numerator by the
    ~0 std and report the ~1.1 garbage z a literal `==0` guard would give."""
    from reservoirs.baselines import reservoir_vs_null
    A = _sym_weighted_adj(seed=0)
    res = reservoir_vs_null(A, score_fn=lambda M: float(M.sum()), n_null=10, seed=0)
    assert set(res) == {"real", "null_mean", "null_std", "z", "null_scores", "n_null"}
    assert res["n_null"] == 10 and len(res["null_scores"]) == 10
    assert abs(res["null_std"]) < 1e-9               # total weight ~invariant (jitter ~5e-14)
    assert abs(res["z"]) < 1e-9                       # guarded to ~0, not a noise-driven blowup
    assert abs(res["real"] - res["null_mean"]) < 1e-9


def test_null_detects_wiring_dependence():
    """C2: a wiring-sensitive score (largest eigenvalue) varies across nulls, so null_std > 0
    and z is finite, the real division branch, proving the machinery distinguishes topology."""
    from reservoirs.baselines import reservoir_vs_null
    A = _sym_weighted_adj(seed=1)
    res = reservoir_vs_null(A, score_fn=lambda M: float(np.linalg.eigvalsh(M).max()), n_null=12, seed=0)
    assert res["n_null"] == 12 and len(res["null_scores"]) == 12
    assert res["null_std"] > 0.0
    assert np.isfinite(res["z"])


# D. compare_classification
def test_compare_classification_keys_and_exact_deltas():
    """D: all 8 keys + HAND-COMPUTED exact deltas on a tiny direction-separable fixture
    (separable by direction; majority=0.5, linear=1.0 with the with-intercept bar)."""
    from reservoirs.baselines import compare_classification
    Xtr = [[5, 0], [6, 0], [5, 1], [0, 5], [0, 6]]
    ytr = [0, 0, 0, 1, 1]
    Xte = [[5, 0.5], [6, 1], [0, 5.5], [1, 6]]
    yte = [0, 0, 1, 1]
    res = compare_classification(0.75, Xtr, ytr, Xte, yte, ridge=1.0)
    assert set(res) == {
        "reservoir", "majority", "linear", "majority_label",
        "delta_vs_majority", "delta_vs_linear", "beats_majority", "beats_linear",
    }
    assert res["reservoir"] == 0.75
    assert res["majority"] == pytest.approx(0.5)      # predict 0 -> 2/4
    assert res["linear"] == pytest.approx(1.0)
    assert res["majority_label"] == 0
    assert res["delta_vs_majority"] == pytest.approx(0.25)
    assert res["delta_vs_linear"] == pytest.approx(-0.25)
    assert res["beats_majority"] is True
    assert res["beats_linear"] is False
    # internal consistency
    assert res["delta_vs_majority"] == pytest.approx(res["reservoir"] - res["majority"])
    assert res["delta_vs_linear"] == pytest.approx(res["reservoir"] - res["linear"])


# E. format_comparison
def test_format_comparison_comparison_shape():
    """E1: the compare_classification shape renders a non-empty table naming each baseline."""
    from reservoirs.baselines import compare_classification, format_comparison
    Xtr = [[5, 0], [6, 0], [5, 1], [0, 5], [0, 6]]
    ytr = [0, 0, 0, 1, 1]
    Xte = [[5, 0.5], [6, 1], [0, 5.5], [1, 6]]
    yte = [0, 0, 1, 1]
    res = compare_classification(0.75, Xtr, ytr, Xte, yte, ridge=1.0)
    s = format_comparison(res)
    assert isinstance(s, str) and s.strip()
    assert "reservoir" in s and "majority" in s and "linear" in s


def test_format_comparison_null_shape_and_error_branch():
    """E2: the reservoir_vs_null shape renders a non-empty table naming 'real'/'null'; an
    unrecognized dict raises ValueError."""
    from reservoirs.baselines import reservoir_vs_null, format_comparison
    A = _sym_weighted_adj(seed=1)
    res = reservoir_vs_null(A, score_fn=lambda M: float(np.linalg.eigvalsh(M).max()), n_null=12, seed=0)
    n = format_comparison(res)
    assert isinstance(n, str) and n.strip()
    assert "real" in n and "null" in n
    with pytest.raises(ValueError):
        format_comparison({"foo": 1})


# F. logistic_readout_baseline (OPTIONAL, sklearn-guarded)
def test_logistic_readout_clears_0p9_on_iris():
    """F: the genuine >0.9 linear-style bar that one-hot least-squares trails on average
    (verified 0.967). Skips cleanly if sklearn or iris.csv is absent."""
    pytest.importorskip("sklearn")
    from reservoirs.baselines import logistic_readout_baseline
    iris = _load_iris_numpy()
    if iris is None:
        pytest.skip("iris.csv not available")
    X, y = iris
    te = np.arange(150) % 5 == 0
    tr = ~te
    acc = logistic_readout_baseline(X[tr], y[tr], X[te], y[te])["accuracy"]
    assert acc > 0.9, f"logistic readout should clear 0.9 on Iris, got {acc:.3f}"
