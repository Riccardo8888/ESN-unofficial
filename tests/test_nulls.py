"""TDD: degree-preserving rewired null connectomes (Maslov-Sneppen), for the null-model baseline.

A connectome-RC claim ("the brain topology helps") only holds if the real connectome beats a null
that keeps the same degree sequence but randomizes the wiring. This module builds that null.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))


def _adj(n=30, p=0.25, seed=0):
    g = np.random.default_rng(seed)
    M = (g.random((n, n)) < p).astype(float) * g.uniform(0.5, 2.0, (n, n))
    M = np.triu(M, 1)
    return M + M.T  # symmetric, zero diagonal


def test_rewire_preserves_degree_sequence():
    from reservoirs.nulls import rewire_degree_preserving
    A = _adj()
    B = rewire_degree_preserving(A, seed=1)
    np.testing.assert_array_equal(np.sort((A != 0).sum(1)), np.sort((B != 0).sum(1)))


def test_rewire_changes_the_wiring():
    from reservoirs.nulls import rewire_degree_preserving
    A = _adj()
    B = rewire_degree_preserving(A, seed=1)
    assert not np.array_equal(A != 0, B != 0), "rewiring should change which edges exist"


def test_rewire_preserves_weight_multiset():
    from reservoirs.nulls import rewire_degree_preserving
    A = _adj()
    B = rewire_degree_preserving(A, seed=1)
    wa = np.sort(A[np.triu_indices(len(A), 1)]); wa = wa[wa != 0]
    wb = np.sort(B[np.triu_indices(len(B), 1)]); wb = wb[wb != 0]
    np.testing.assert_allclose(wa, wb)


def test_rewire_symmetric_zero_diagonal():
    from reservoirs.nulls import rewire_degree_preserving
    B = rewire_degree_preserving(_adj(), seed=2)
    np.testing.assert_array_equal(B, B.T)
    assert np.all(np.diag(B) == 0)


def test_rewire_is_deterministic_under_seed():
    from reservoirs.nulls import rewire_degree_preserving
    A = _adj()
    np.testing.assert_array_equal(rewire_degree_preserving(A, seed=3), rewire_degree_preserving(A, seed=3))
