"""TDD: vectorized batched state collection must match the per-window forward loop.

Phase 5 adds `collect_states_batch(U)` (U: [B, T, n_inputs] -> states [B, T, N]), which processes
all B windows in parallel via GEMM. It does not replace the golden-pinned `forward`: it has to
reproduce the same dynamics (within float tolerance) while running much faster when there are many
windows.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
GRAPH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)),
                     "generated_artifacts", "graphs")


def _seqs(B=8, T=20, nin=4, seed=0):
    return np.random.default_rng(seed).standard_normal((B, T, nin))


def test_connectome_batch_matches_per_window():
    from reservoirs.connectome import ConnectomeReservoir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ConnectomeReservoir(4, graph_dir=GRAPH, spectral_radius=0.9, seed=7)
    U = _seqs(nin=4)
    batched = r.collect_states_batch(U)
    perwin = np.stack([np.asarray(r.forward(u, collect_states=True)) for u in U])
    assert batched.shape == perwin.shape == (8, 20, r.n_neurons)
    np.testing.assert_allclose(batched, perwin, atol=1e-4, rtol=1e-4)


def test_random_batch_matches_per_window():
    from reservoirs.random import Reservoir
    np.random.seed(7)
    r = Reservoir(4, 40, rhow=1.25)
    U = _seqs(nin=4)
    batched = r.collect_states_batch(U)
    perwin = np.stack([np.asarray(r.forward(u, collect_states=True)) for u in U])
    np.testing.assert_allclose(batched, perwin, atol=1e-8, rtol=1e-8)  # float64 -> tight


def test_batch_handles_single_sequence():
    from reservoirs.connectome import ConnectomeReservoir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ConnectomeReservoir(4, graph_dir=GRAPH, spectral_radius=0.9, seed=7)
    U = _seqs(B=1, nin=4)
    out = r.collect_states_batch(U)
    assert out.shape == (1, 20, r.n_neurons)
    np.testing.assert_allclose(out[0], np.asarray(r.forward(U[0], collect_states=True)), atol=1e-4)
