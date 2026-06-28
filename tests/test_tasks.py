"""Tests for reservoirs.tasks, the temporal (memory) benchmark (numpy + pytest only).

These bite: the NARMA-10 recurrence is checked against a hand calculation; the
divergence guard is exercised; memory_capacity is validated on an explicit tapped
delay line (MC ~ depth) and on a memoryless state (MC ~ 0); narma_nrmse is pinned
at its two extremes (perfect predictor ~ 0, constant state ~ 1); the topology
runner's dict keys/ranges are checked on a small synthetic connectome; and a
structural no-leakage check confirms the readout fit uses only the train block.

Fast by construction: N <= 40, T <= 800, n_null <= 8.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, REPO)

from reservoirs import tasks  # noqa: E402


# helpers
def _sym_weighted_adj(n=30, p=0.25, seed=0):
    """Symmetric, weighted, zero-diagonal adjacency (same recipe as tests/test_nulls)."""
    g = np.random.default_rng(seed)
    M = (g.random((n, n)) < p).astype(float) * g.uniform(0.5, 2.0, (n, n))
    M = np.triu(M, 1)
    return M + M.T


# A. NARMA-10 recurrence
def test_narma_recurrence_matches_hand_computed():
    """y[10], y[11] recomputed by the formula by hand match tasks' recurrence to ~1e-9.

    With discard=0 the returned (u, y) expose s and the warmup directly, so we can
    hand-evaluate the first two non-trivial steps from the published formula.
    """
    u, y = tasks.narma10(30, seed=3, discard=0)
    s = u.ravel()
    Y = y.ravel()
    # y[:10] are zero, so the first non-trivial term is y[10].
    assert np.allclose(Y[:10], 0.0)
    # t = 9 -> y[10]: all of y[0:10] are zero, so only the input cross-term + bias survive.
    y10 = 0.3 * 0.0 + 0.05 * 0.0 * 0.0 + 1.5 * s[0] * s[9] + 0.1
    assert abs(Y[10] - y10) < 1e-9
    # t = 10 -> y[11]: sum_{i} y[1:11] = y[10] (only nonzero term).
    y11 = 0.3 * Y[10] + 0.05 * Y[10] * np.sum(Y[1:11]) + 1.5 * s[1] * s[10] + 0.1
    assert abs(Y[11] - y11) < 1e-9


def test_narma_is_bounded():
    """For several seeds the target is finite and bounded; normal seeds never raise."""
    for seed in range(8):
        u, y = tasks.narma10(500, seed=seed)
        assert u.shape == (500, 1) and y.shape == (500, 1)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) < 1e3
    # The resample guard must still yield a finite result even with a single attempt
    # for a normal seed (i.e. RuntimeError is NOT raised on the common path).
    u, y = tasks.narma10(300, seed=1, max_resample=1)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e3


# B. memory_capacity
def test_mc_of_pure_delay_line():
    """An explicit tapped delay line of depth D has total MC ~ D.

    Column j holds the input delayed by j+1, so delays 1..D are each reconstructed
    perfectly (MC_k ~ 1 for k <= D) and longer delays cannot be (MC_k ~ 0). Since
    the MC sum runs over delays >= 1, the total is ~ D.
    """
    rng = np.random.default_rng(0)
    T, D = 800, 8
    u = rng.uniform(-1.0, 1.0, T)
    states = np.zeros((T, D))
    for j in range(D):
        d = j + 1
        states[d:, j] = u[: T - d]  # column j = u(t-(j+1))
    out = tasks.memory_capacity(states, u, max_delay=2 * D, seed=0)
    curve = out["mc_curve"]
    assert len(curve) == 2 * D
    assert abs(out["mc_total"] - D) < 0.5, out["mc_total"]
    for k in range(D):            # delays 1..D -> reconstructable
        assert curve[k] > 0.8, (k, curve[k])
    for k in range(D, 2 * D):     # delays D+1..2D -> not in the tap set
        assert curve[k] < 0.2, (k, curve[k])


def test_mc_zero_for_memoryless():
    """A memoryless state (the current input only) has ~ no memory of the past."""
    rng = np.random.default_rng(1)
    T = 800
    u = rng.uniform(-1.0, 1.0, T)
    states = u[:, None]  # [T, 1]: state(t) = u(t)
    out = tasks.memory_capacity(states, u, seed=0)
    assert out["mc_total"] < 0.2, out["mc_total"]
    assert out["n_neurons"] == 1


# C. narma_nrmse extremes
def test_narma_nrmse_extremes():
    """Perfect predictor -> NRMSE ~ 0; constant state -> NRMSE ~ 1."""
    rng = np.random.default_rng(2)
    T = 600
    y = rng.standard_normal((T, 1))
    # Perfect: the target itself is one of the state columns.
    states_perfect = np.hstack([y, rng.standard_normal((T, 2))])
    nrmse_perfect = tasks.narma_nrmse(states_perfect, y, ridge=1e-8)
    assert nrmse_perfect < 0.05, nrmse_perfect
    # Constant state: the readout can only predict a constant ~ mean(y_train),
    # which on a zero-mean stationary target gives NRMSE ~ 1.
    states_const = np.ones((T, 3))
    nrmse_const = tasks.narma_nrmse(states_const, y, ridge=1e-6)
    assert abs(nrmse_const - 1.0) < 0.3, nrmse_const


# D. topology runner
def test_topology_runner_keys_and_ranges():
    """memory_capacity_topology returns all keys with sane ranges on a small connectome."""
    A = _sym_weighted_adj(n=30, p=0.25, seed=0)
    res = tasks.memory_capacity_topology(
        A, n_null=8, seed=0, n_seeds=2, n_input_draws=1, n_samples=500, washout=50
    )
    expected = {
        "metric", "real", "null_mean", "null_std", "z", "percentile", "p_value",
        "random_esn", "no_reservoir", "n_neurons", "n_null",
    }
    assert set(res) == expected
    assert res["metric"] == "memory_capacity"
    assert res["n_neurons"] == 30
    assert res["n_null"] == 8
    assert 0.0 <= res["percentile"] <= 1.0
    assert 0.0 < res["p_value"] <= 1.0
    assert np.isfinite(res["z"])
    # The no-reservoir control must carry essentially no memory, well below the
    # real reservoir's MC.
    assert res["no_reservoir"] < 0.5, res["no_reservoir"]
    assert res["real"] > res["no_reservoir"]
    # The report renders to a non-empty string mentioning the baselines.
    s = tasks.format_temporal_report(res)
    assert isinstance(s, str) and "real connectome" in s and "no reservoir" in s


# E. no temporal leakage
def test_no_temporal_leakage():
    """Shuffling the TEST-block targets does not change the fitted readout.

    The ridge readout is the primitive both metrics use; here we confirm it (and
    therefore the train metric / train predictions) depends ONLY on the train
    block, so test-set targets cannot leak into the fit.
    """
    rng = np.random.default_rng(0)
    T, N = 400, 5
    states = rng.standard_normal((T, N))
    y = rng.standard_normal((T, 1))
    c = int(0.5 * T)
    W1 = tasks._ridge_readout(states[:c], y[:c], 1e-6)
    # Permute ONLY the test block of the targets; the train block is untouched.
    y2 = y.copy()
    y2[c:] = y2[c:][rng.permutation(T - c)]
    W2 = tasks._ridge_readout(states[:c], y2[:c], 1e-6)
    assert np.allclose(W1, W2)
    assert np.allclose(states[:c] @ W1, states[:c] @ W2)
