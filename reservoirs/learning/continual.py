"""Conceptors (Jaeger 2014) and a simple per-class conceptor-bank classifier.

A conceptor ``C = R (R + a^-2 I)^-1`` is a soft projector onto the ellipsoid the reservoir
state cloud occupies while driven by a class or pattern (``R`` = state correlation, ``a`` = aperture).

`ConceptorClassifier` is deliberately a per-class conceptor bank: it stores one independent
conceptor per class and classifies by the positive evidence ``mean_t x^T C_k x``. Because each
conceptor depends only on its own class's data, learning a later class cannot alter an earlier
conceptor, so the stored representation is forgetting-free.

This is NOT Jaeger's full conceptor scheme. It does NOT implement negative evidence (the
``x^T NOT(OR_{m!=k} C_m) x`` term), ``x^T x`` normalisation of the evidence (so raw-energy
differences across samples can bias it), or incremental "free-subspace loading"
(``N = C AND NOT(OR previous)``). The boolean primitives below (``conceptor_not / _and / _or``)
are the correct algebra for building those, but the bundled classifier does not use them; they
are provided for that future work.
See ../../docs/CONTINUOUS_LEARNING_DESIGN.md.
"""
import numpy as np


def conceptor_from_correlation(R, aperture: float) -> np.ndarray:
    """C = R (R + aperture^-2 I)^-1 from a state-correlation matrix R.

    Computed via a symmetric eigendecomposition of the PSD correlation R (C shares R's
    eigenvectors; C's eigenvalues are s/(s + aperture^-2)). This is exact, guarantees the
    conceptor eigenvalues lie in [0, 1] even when R is rank-deficient (fewer samples than
    neurons), and never raises the way a plain `inv` does at very large aperture, because the
    aperture term is the inversion regulariser.
    """
    R = np.asarray(R, dtype=float)
    R = 0.5 * (R + R.T)                          # symmetrise against roundoff before eigh
    s, U = np.linalg.eigh(R)
    s = np.clip(s, 0.0, None)                    # R is PSD; clamp tiny negative roundoff
    denom = s + (aperture ** -2)
    ev = np.divide(s, denom, out=np.zeros_like(s), where=denom > 0)
    C = (U * ev) @ U.T
    return 0.5 * (C + C.T)


def conceptor_from_states(states, aperture: float) -> np.ndarray:
    """C = R (R + aperture^-2 I)^-1, with R the state correlation over all timesteps/examples."""
    S = np.asarray(states, dtype=float)
    X = S.reshape(-1, S.shape[-1])
    n = X.shape[0]
    R = (X.T @ X) / n
    return conceptor_from_correlation(R, aperture)


def conceptor_not(C: np.ndarray) -> np.ndarray:
    return np.eye(C.shape[0]) - C


def conceptor_and(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    return np.linalg.pinv(np.linalg.pinv(A) + np.linalg.pinv(B) - np.eye(N))


def conceptor_or(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return conceptor_not(conceptor_and(conceptor_not(A), conceptor_not(B)))


class ConceptorClassifier:
    """Per-class conceptor bank (positive-evidence only; see module docstring for what it is NOT).

    learn_class(states, label) stores or updates that class's conceptor; predict(states) returns
    the class with the highest mean positive evidence ``x^T C x``. `states` are reservoir states of
    shape [n_examples, T, N] (or [T, N] for a single example). Because the per-class conceptors are
    independent the representation is forgetting-free, but note that multi-class decisions can still
    shift as competing classes are added (the class-incremental difficulty), and the evidence is
    un-normalised.
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


class ConceptorReadout:
    """Benchmark-compatible per-class conceptor bank (scikit-learn-style incremental API).

    A thin adapter so the conceptor method can be driven by `ContinualBenchmark`, which expects a
    per-sample `partial_fit(X, y[, classes])` / `predict(X)` (X is [n_samples, n_features], one
    label per row). It accumulates each class's state-correlation across `partial_fit` calls and
    refits that class's conceptor; `predict` classifies each row x by argmax_k x^T C_k x.

    Why this exists (see ../../docs/AUDIT.md F3): the per-class representation is forgetting-free
    (each conceptor depends only on its own class's data), but the multi-class decisions are NOT,
    and running this through the benchmark is exactly what surfaces that (BWT/forgetting can be
    strongly negative on overlapping class-incremental streams). Evidence is positive-only and
    un-normalised, the same caveat as `ConceptorClassifier`.
    """

    def __init__(self, aperture: float = 4.0):
        self.aperture = float(aperture)
        self.classes_ = None
        self._Rsum = {}
        self._n = {}
        self.conceptors_ = {}

    def partial_fit(self, X, y, classes=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        n_features = X.shape[1]
        if self.classes_ is None:
            self.classes_ = list(classes) if classes is not None else []
        for c in np.unique(y):
            if c not in self.classes_:
                self.classes_.append(c)
            if c not in self._Rsum:
                self._Rsum[c] = np.zeros((n_features, n_features))
                self._n[c] = 0
            Xc = X[y == c]
            self._Rsum[c] = self._Rsum[c] + Xc.T @ Xc
            self._n[c] += len(Xc)
            R = self._Rsum[c] / max(self._n[c], 1)
            self.conceptors_[c] = conceptor_from_correlation(R, self.aperture)
        return self

    def predict(self, X):
        if not self.conceptors_:
            raise RuntimeError("call partial_fit before predict().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        labels = [c for c in self.classes_ if c in self.conceptors_]
        scores = np.stack(
            [np.einsum("ni,ij,nj->n", X, self.conceptors_[c], X) for c in labels], axis=1
        )
        return np.asarray(labels, dtype=object)[scores.argmax(axis=1)]
