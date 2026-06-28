"""Integration: the continuous-learning readouts compose with a FROZEN reservoir substrate.

Drives a reservoir on sine-encoded Iris, then (a) learns an OnlineReadout one flower at a time
(streaming), and (b) builds a forgetting-free ConceptorClassifier on the reservoir states.
This is the end-to-end proof that `reservoirs.learning` works on real reservoir states.
"""
import os
import sys
import random

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
pd = pytest.importorskip("pandas")


def _iris_states(n_neurons=80, seed=7):
    from reservoirs.random import Reservoir
    df = pd.get_dummies(pd.read_csv(os.path.join(REPO, "iris.csv"), index_col=0))
    raw = df.to_numpy().astype(float)
    rng = random.Random(seed)
    tr = rng.sample(range(0, 50), 40) + rng.sample(range(50, 100), 40) + rng.sample(range(100, 150), 40)
    te = [i for i in range(150) if i not in tr]
    data = raw.copy(); data[:, :4] = data[:, :4] / data[tr, :4].max(axis=0)
    ts = np.arange(0, 50, 1.0)
    u = np.array([np.vstack([np.sin(ts * 2 * np.pi * p[i]) for i in range(4)]).T for p in data[:, :4]])
    labels = data[:, 4:].argmax(axis=1)
    np.random.seed(seed)
    res = Reservoir(4, n_neurons, rhow=1.25)
    states = np.stack([res.forward(seq, collect_states=True) for seq in u], axis=0)  # [150, T, N]
    return states[tr], labels[tr], states[te], labels[te]


S_TR, Y_TR, S_TE, Y_TE = _iris_states()


def test_online_readout_streams_over_reservoir_states():
    from reservoirs.learning.online import OnlineReadout
    feat_tr = S_TR.mean(axis=1)   # time-averaged reservoir state per flower
    feat_te = S_TE.mean(axis=1)
    clf = OnlineReadout(forgetting=1.0, ridge=1e-4)
    clf.partial_fit(feat_tr[:1], Y_TR[:1], classes=[0, 1, 2])   # declare classes, first sample
    for i in range(1, len(feat_tr)):
        clf.partial_fit(feat_tr[i:i + 1], Y_TR[i:i + 1])        # stream the rest one at a time
    acc = float(np.mean(clf.predict(feat_te) == Y_TE))
    assert acc > 0.7, f"online readout iris test_acc={acc:.3f}"


def test_conceptor_classifier_on_reservoir_states():
    from reservoirs.learning.continual import ConceptorClassifier
    clf = ConceptorClassifier(aperture=8.0)
    for c in (0, 1, 2):                       # learn each class incrementally (forgetting-free)
        clf.learn_class(S_TR[Y_TR == c], c)
    acc = float(np.mean(clf.predict(S_TE) == Y_TE))
    # positive-evidence conceptors are modest on *static* iris (~0.70 as it comes out); the point of
    # this integration test is that the layer composes with the reservoir and clears chance (0.33).
    assert acc > 0.5, f"conceptor classifier iris test_acc={acc:.3f}"
