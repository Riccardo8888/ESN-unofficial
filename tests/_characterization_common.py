"""
Shared logic for the characterization (golden) tests.

These pin the behavior of the kept engines, now imported from the `reservoirs/` package
(Phase 2), against goldens that were generated (Phase 1) from the original pre-package code.
The package reproduces those goldens bit-for-bit, which is the proof that the move into the
package (plus the shim) did not change any numbers. The two kept engines are
`reservoirs.connectome.ConnectomeReservoir` (formerly brain_connectome_reservoir_v0_1.py) and
`reservoirs.random.Reservoir` (formerly reservoir.py).

The old connectome engine (`brain_connectome_reservoir.py`) was retired; its `old_native_Win` /
`old_native_leak` goldens are now checked directly in `test_characterization.py`
(`test_old_new_native_rng_identical_static`), which proves the new engine draws them identically.

Determinism: the connectome engine seeds an internal default_rng(7); the random engine uses the
global numpy RNG, so we call np.random.seed(0) immediately before building it. Everything runs on
the committed mock connectome (60 nodes, undirected), n_inputs=4, with a fixed deterministic input.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GRAPH_DIR = os.path.join(REPO_ROOT, "generated_artifacts", "graphs")
SEED = 7
N_INPUTS = 4
UPSCALE = 120  # > native 60 -> exercises random-projection upscale (new engine default)


def tiny_input(T=20, n=N_INPUTS):
    """Deterministic input sequence [T, n], no RNG involved."""
    idx = np.arange(T * n, dtype=np.float64).reshape(T, n)
    return np.sin(idx / 3.0)


def sr_exact(W):
    """Exact spectral radius via dense eigenvalues (reference, independent of engine power-iter)."""
    return np.array(float(np.max(np.abs(np.linalg.eigvals(np.asarray(W, dtype=np.float64))))))


def build_new(**kw):
    """Build the canonical connectome engine from the package."""
    from reservoirs.connectome import ConnectomeReservoir
    kw.setdefault("graph_dir", GRAPH_DIR)
    kw.setdefault("seed", SEED)
    return ConnectomeReservoir(N_INPUTS, **kw)


def snapshot():
    """Pin the KEPT engines (now from the package) against the Phase-1 goldens."""
    snaps = {}
    u = tiny_input()

    # canonical connectome engine (reservoirs.connectome)
    rnew = build_new(spectral_radius=0.9)
    snaps["new_native_Win"] = np.asarray(rnew.Win)
    snaps["new_native_leak"] = np.asarray(rnew.leak)
    snaps["new_native_Win_bias"] = np.asarray(rnew.Win_bias)
    snaps["new_native_W"] = np.asarray(rnew.W)
    snaps["new_native_sr"] = sr_exact(rnew.W)
    snaps["new_native_states"] = np.asarray(rnew.transform(u, washout=0))
    rnew_up = build_new(spectral_radius=0.9, n_neurons=UPSCALE)  # random-projection upscale (default)
    snaps["new_up120_W"] = np.asarray(rnew_up.W)

    # random engine (reservoirs.random; global RNG, so seed first)
    from reservoirs.random import Reservoir
    np.random.seed(0)
    rr = Reservoir(N_INPUTS, 30, rhow=1.25)
    snaps["rand_R1_win"] = np.asarray(rr.win)
    snaps["rand_R1_w"] = np.asarray(rr.w)
    snaps["rand_R1_leak"] = np.asarray(rr.leak)
    snaps["rand_R1_sr"] = np.array(float(rr.spectral_radius))
    snaps["rand_R1_states"] = np.asarray(rr.forward(u, collect_states=True))

    return snaps
