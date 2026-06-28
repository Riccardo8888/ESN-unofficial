"""Online (per-sample) readouts: RLS (default), LMS, NLMS.

The reservoir is frozen; only these readout weights adapt, one sample at a time via
``partial_fit``. RLS is exact recursive least squares with a forgetting factor and a ridge
seed (P0 = I/ridge ⇔ Tikhonov penalty `ridge·‖w‖²`). The inverse-correlation matrix is
re-symmetrised each step — a lightweight guard against roundoff drift, NOT the full
square-root/UD-factored form or the state-noise injection discussed in the design doc (those
remain future work). See ../../docs/CONTINUOUS_LEARNING_DESIGN.md.
"""
import numpy as np


class RLSReadout:
    """Multi-output linear readout trained online.

    Parameters
    ----------
    method : {'rls','lms','nlms'}   (default 'rls')
    forgetting : RLS forgetting factor λ ∈ (0,1]   (1 = infinite memory)
    ridge : Tikhonov strength δ; RLS seeds P0 = I/δ
    mu : (N)LMS step size
    """

    def __init__(self, n_features, n_outputs, method="rls",
                 forgetting=1.0, ridge=1e-2, mu=0.1, eps=1e-8):
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.method = method
        self.forgetting = float(forgetting)
        self.ridge = float(ridge)
        self.mu = float(mu)
        self.eps = float(eps)
        self.W_ = np.zeros((self.n_features, self.n_outputs))
        self.P_ = np.eye(self.n_features) / self.ridge
        self._fitted = False

    def _update(self, x, d):
        e = d - self.W_.T @ x  # a-priori error (n_outputs,)
        if self.method == "rls":
            lam = self.forgetting
            pi = self.P_ @ x
            k = pi / (lam + x @ pi)
            self.W_ = self.W_ + np.outer(k, e)
            self.P_ = (self.P_ - np.outer(k, pi)) / lam
            self.P_ = 0.5 * (self.P_ + self.P_.T)  # numerical stability: keep symmetric
        elif self.method == "lms":
            self.W_ = self.W_ + self.mu * np.outer(x, e)
        elif self.method == "nlms":
            self.W_ = self.W_ + (self.mu / (self.eps + x @ x)) * np.outer(x, e)
        else:
            raise ValueError(f"unknown method {self.method!r}")

    def partial_fit(self, X, Y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        for x, d in zip(X, Y):
            self._update(x, d)
        self._fitted = True
        return self

    def predict(self, X):
        return np.atleast_2d(np.asarray(X, dtype=float)) @ self.W_


class OnlineReadout:
    """Online classifier on top of an `RLSReadout` (one-hot targets, argmax decode).

    Mirrors scikit-learn's incremental contract: `classes` must be passed on the first
    `partial_fit` call; later calls update (do not reset) the readout.
    """

    def __init__(self, method="rls", forgetting=1.0, ridge=1e-2, mu=0.1):
        self.method = method
        self.forgetting = forgetting
        self.ridge = ridge
        self.mu = mu
        self.classes_ = None
        self._rls = None

    def partial_fit(self, X, y, classes=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y)
        if self.classes_ is None:
            if classes is None:
                raise ValueError("`classes` must be provided on the first partial_fit call.")
            self.classes_ = list(classes)
            self._index = {c: i for i, c in enumerate(self.classes_)}
            self._rls = RLSReadout(X.shape[1], len(self.classes_), method=self.method,
                                   forgetting=self.forgetting, ridge=self.ridge, mu=self.mu)
        Y = np.zeros((len(y), len(self.classes_)))
        for i, label in enumerate(y):
            Y[i, self._index[label]] = 1.0
        self._rls.partial_fit(X, Y)
        return self

    def decision_function(self, X):
        return self._rls.predict(X)

    def predict(self, X):
        idx = self._rls.predict(X).argmax(axis=1)
        return np.asarray(self.classes_)[idx]
