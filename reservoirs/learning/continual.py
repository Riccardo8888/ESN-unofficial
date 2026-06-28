"""Conceptors (Jaeger 2014) and a simple per-class conceptor-bank classifier.

A conceptor ``C = R (R + a^-2 I)^-1`` is a soft projector onto the ellipsoid the reservoir
state cloud occupies while driven by a class/pattern (``R`` = state correlation, ``a`` = aperture).

`ConceptorClassifier` is deliberately a **per-class conceptor bank**: it stores one independent
conceptor per class and classifies by the positive evidence ``mean_t x^T C_k x``. Because each
conceptor depends only on its own class's data, learning a later class cannot alter an earlier
conceptor — so the stored *representation* is forgetting-free.

This is NOT Jaeger's full conceptor scheme. It does NOT implement:
  - negative evidence (the ``x^T NOT(OR_{m!=k} C_m) x`` term),
  - ``x^T x`` normalisation of the evidence (so raw-energy differences across samples can bias it),
  - incremental "free-subspace loading" (``N = C AND NOT(OR previous)``).
The boolean primitives below (``conceptor_not / _and / _or``) are the correct algebra for building
those, but the bundled classifier does not use them — they are provided for that future work.
See ../../docs/CONTINUOUS_LEARNING_DESIGN.md.
"""
import numpy as np


def conceptor_from_states(states, aperture: float) -> np.ndarray:
    """C = R (R + aperture^-2 I)^-1, with R the state correlation over all timesteps/examples."""
    S = np.asarray(states, dtype=float)
    X = S.reshape(-1, S.shape[-1])
    n, N = X.shape
    R = (X.T @ X) / n
    C = R @ np.linalg.inv(R + (aperture ** -2) * np.eye(N))
    return 0.5 * (C + C.T)  # symmetric by construction; enforce against roundoff


def conceptor_not(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0]) - C


def conceptor_and(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    return np.linalg.pinv(np.linalg.pinv(A) + np.linalg.pinv(B) - np.eye(N))


def conceptor_or(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return conceptor_not(conceptor_and(conceptor_not(A), conceptor_not(B)))


class ConceptorClassifier:
    """Per-class conceptor bank (positive-evidence only; see module docstring for what it is NOT).

    learn_class(states, label) stores/updates that class's conceptor; predict(states) returns the
    class with the highest mean positive evidence ``x^T C x``. `states` are reservoir states of
    shape [n_examples, T, N] (or [T, N] for a single example). Independent per-class conceptors
    ⇒ the representation is forgetting-free, but note: multi-class *decisions* can still shift as
    competing classes are added (the class-incremental difficulty), and evidence is un-normalised.
    """

    def __init__(self, aperture: float = 4.0):
        self.aperture = float(aperture)
        self.conceptors_ = {}

    def learn_class(self, states, label):
        self.conceptors_[label] = conceptor_from_states(states, self.aperture)
        return self

    def evidence(self, states):
        S = np.asarray(states, dtype=float)
        single = (S.ndim == 2)
        if single:
            S = S[None]
        out = {}
        for label, C in self.conceptors_.items():
            # per-example mean over time of the quadratic form x^T C x (vectorized)
            ev = np.einsum("nti,ij,ntj->nt", S, C, S).mean(axis=1)
            out[label] = ev[0] if single else ev
        return out

    def predict(self, states):
        if not self.conceptors_:
            raise RuntimeError("learn at least one class before predict().")
        ev = self.evidence(states)
        labels = list(self.conceptors_.keys())
        scores = np.stack([np.atleast_1d(ev[l]) for l in labels], axis=-1)  # [n, n_classes]
        idx = np.argmax(scores, axis=-1)
        labels_arr = np.array(labels, dtype=object)
        out = labels_arr[idx]
        return out if np.asarray(states).ndim == 3 else out[0]
