"""Cross-engine sanity test on Iris (Phase-3 port of the ad-hoc `test_reservoirs.ipynb`).

Builds each package engine on the sine-encoded Iris benchmark and asserts the readout
learns (test accuracy above a conservative floor), finite states, correct shapes. This is
a guardrail: it would have caught the historical blow-ups where a reservoir produced garbage.
"""
import os
import sys
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
GRAPH = os.path.join(REPO, "generated_artifacts", "graphs")

pd = pytest.importorskip("pandas")


def _iris():
    df = pd.get_dummies(pd.read_csv(os.path.join(REPO, "iris.csv"), index_col=0))
    raw = df.to_numpy().astype(float)
    rng = random.Random(7)
    tr = rng.sample(range(0, 50), 40) + rng.sample(range(50, 100), 40) + rng.sample(range(100, 150), 40)
    te = [i for i in range(150) if i not in tr]
    data = raw.copy()
    data[:, :4] = data[:, :4] / data[tr, :4].max(axis=0)
    ts = np.arange(0, 50, 1.0)  # 50 steps — fast
    u = np.array([np.vstack([np.sin(ts * 2 * np.pi * p[i]) for i in range(4)]).T for p in data[:, :4]])
    y = np.array([data[:, 4:]] * len(ts)).swapaxes(0, 1).astype(float)
    return u[tr], y[tr], u[te], y[te]


U_TR, Y_TR, U_TE, Y_TE = _iris()


def _collect(res, batch):
    return np.stack([res.forward(seq, collect_states=True) for seq in batch], axis=0)


def _eval(res, washout=5, alpha=1e-4):
    Xtr, Xte = _collect(res, U_TR), _collect(res, U_TE)
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all(), "reservoir produced non-finite states"
    X = Xtr[:, washout:, :].reshape(-1, Xtr.shape[-1]); Y = Y_TR[:, washout:, :].reshape(-1, Y_TR.shape[-1])
    w = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)
    acc = float(np.mean((Xte @ w).mean(1).argmax(1) == Y_TE.mean(1).argmax(1)))
    assert (Xtr @ w).shape == (U_TR.shape[0], U_TR.shape[1], Y_TR.shape[-1])
    return acc


@pytest.mark.parametrize("name", ["fully_connected", "gaussian", "erdos_renyi"])
def test_learning_engines_clear_floor(name):
    from reservoirs.random import Reservoir, Reservoir3, ErdosRenyiReservoir
    np.random.seed(7)
    res = {"fully_connected": lambda: Reservoir(4, 60, rhow=1.25),
           "gaussian": lambda: Reservoir3(4, 60, rhow=1.25),
           "erdos_renyi": lambda: ErdosRenyiReservoir(4, 60, rhow=1.25, density=0.15, seed=7)}[name]()
    acc = _eval(res)
    assert acc >= 0.6, f"{name} iris test_acc={acc:.3f} below floor 0.6"


def test_ring_runs_finite():
    """Ring reservoir collapses to ~chance on this task — we only assert it runs & is finite."""
    from reservoirs.random import Reservoir2
    np.random.seed(7)
    acc = _eval(Reservoir2(4, 60, rhow=1.25))
    assert 0.0 <= acc <= 1.0


def test_connectome_engine_learns():
    import warnings
    from reservoirs.connectome import ConnectomeReservoir
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ConnectomeReservoir(4, graph_dir=GRAPH, spectral_radius=0.9, seed=7)
    acc = _eval(res)
    assert acc >= 0.5, f"connectome iris test_acc={acc:.3f} below floor 0.5"
