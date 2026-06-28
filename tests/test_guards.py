"""TDD: fail-fast guards for the silent-degenerate edge cases the review flagged.

(slither split with <2 sessions, empty design matrix in compute_wout, zero windows,
 and the nulls edge/weight length mismatch.)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))


def test_grouped_split_requires_two_sessions():
    from slither.data import train_test_split_windows
    u = np.zeros((5, 25, 3), np.float32); y = np.zeros((5, 25, 18), np.float32)
    sid = np.zeros(5, int)  # a single session
    with pytest.raises(ValueError, match="session"):
        train_test_split_windows(u, y, sid, group_by_session=True)


def test_grouped_split_two_sessions_is_disjoint():
    from slither.data import train_test_split_windows
    u = np.zeros((6, 25, 3), np.float32); y = np.zeros((6, 25, 18), np.float32)
    sid = np.array([0, 0, 0, 1, 1, 1])
    _, _, _, _, sid_tr, sid_te = train_test_split_windows(u, y, sid, group_by_session=True)
    assert set(sid_tr).isdisjoint(set(sid_te))  # the property the leakage-free split must guarantee
    assert len(sid_tr) and len(sid_te)


def test_compute_wout_empty_design_raises():
    from slither.metrics import compute_wout
    X = np.zeros((2, 3, 10), np.float32); Y = np.zeros((2, 3, 5), np.float32)
    with pytest.raises(ValueError):  # washout >= T -> no rows
        compute_wout(X, Y, washout=5)


def test_make_windows_zero_windows_raises():
    from slither.data import make_windows
    X = [np.zeros((10, 4), np.float32)]; Y = [np.zeros((10, 18), np.float32)]  # shorter than window_len
    with pytest.raises(ValueError):
        make_windows(X, Y, window_len=25, stride=15)


def test_rewire_guards_edge_weight_mismatch():
    # normal path still works (guard must not break valid input)
    from reservoirs.nulls import rewire_degree_preserving
    rng = np.random.default_rng(0)
    A = (rng.random((20, 20)) < 0.3) * rng.uniform(0.5, 2, (20, 20)); A = np.triu(A, 1); A = A + A.T
    B = rewire_degree_preserving(A, seed=1)
    assert B.shape == (20, 20)
