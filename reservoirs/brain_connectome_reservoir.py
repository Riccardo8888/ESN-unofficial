"""Smoke-test copy of the canonical ConnectomeReservoir."""
from __future__ import annotations
import os, glob, math, warnings
from typing import Optional, Tuple, List
import numpy as np
import networkx as nx

__all__ = ["ConnectomeReservoir"]


class ConnectomeReservoir:
    def __init__(self, n_inputs, graph_dir=None, *, folder=None, path=None,
                 n_neurons=None, target_size=None, file_glob="*.graphml",
                 edge_attr="weight", combine="mean",
                 spectral_radius=1.25, rhow=None,
                 leak_range=(0.1, 0.3), symmetric=True, seed=42, input_scale=1.0):
        graph_dir = graph_dir or folder or path
        if graph_dir is None:
            raise ValueError("graph_dir required")
        if rhow is not None:
            spectral_radius = float(rhow)
        if combine not in ("mean", "median", "first"):
            raise ValueError("combine in {mean,median,first}")
        lo, hi = leak_range
        if hi <= lo:
            raise ValueError("leak_range")
        if n_neurons is None:
            n_neurons = target_size

        self.rng = np.random.default_rng(seed)
        files = sorted(glob.glob(os.path.join(graph_dir, file_glob)))
        if not files:
            raise FileNotFoundError(f"no graphml in {graph_dir}")

        graphs = [nx.read_graphml(f) for f in files]
        all_nodes = sorted({n for g in graphs for n in g.nodes()})
        if edge_attr is not None and not _attr_exists(graphs, edge_attr):
            edge_attr = None
        As = []
        for g in graphs:
            if set(g.nodes()) != set(all_nodes):
                g = g.copy(); g.add_nodes_from(set(all_nodes) - set(g.nodes()))
            As.append(nx.to_numpy_array(g, nodelist=all_nodes, weight=edge_attr, dtype=float))
        if len(As) == 1 or combine == "first":
            A = As[0]
        elif combine == "median":
            A = np.median(np.stack(As, axis=2), axis=2)
        else:
            A = np.mean(np.stack(As, axis=2), axis=2)
        if symmetric:
            A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)
        if n_neurons is not None and n_neurons != A.shape[0]:
            A = self._resize(A, int(n_neurons))
        sr = _spectral_radius(A)
        if not math.isfinite(sr) or sr <= 0:
            raise ValueError("non-pos sr")
        W = (A / sr) * spectral_radius

        self.n_inputs = int(n_inputs)
        self.n_neurons = int(W.shape[0])
        self.target_spectral_radius = float(spectral_radius)
        self.W = W.astype(np.float32, copy=False)
        self.Win = self.rng.uniform(-input_scale, input_scale,
                                    size=(self.n_neurons, self.n_inputs)).astype(np.float32)
        self.leak = self.rng.uniform(lo, hi, size=(self.n_neurons,)).astype(np.float32)
        self.Win_bias = self.rng.uniform(-input_scale, input_scale,
                                         size=(self.n_neurons,)).astype(np.float32)
        self.Wout = None
        self.w = self.W
        self.win = np.concatenate([self.Win, self.Win_bias[:, None]], axis=1)

    def _step(self, x, u):
        pre = self.Win @ u + self.Win_bias + self.W @ x
        return (1.0 - self.leak) * x + self.leak * np.tanh(pre)

    def _collect(self, U, washout):
        T = U.shape[0]
        if washout >= T:
            raise ValueError("washout >= T")
        x = np.zeros(self.n_neurons, dtype=np.float32)
        out = []
        for t in range(T):
            x = self._step(x, U[t])
            if t >= washout:
                out.append(x.copy())
        return np.array(out, dtype=np.float32)

    def forward(self, u, collect_states=False, wout=None):
        U = _vinp(u, self.n_inputs)
        T = U.shape[0]; N = self.n_neurons
        x = np.zeros(N, dtype=np.float32)
        need = collect_states or (wout is not None)
        X = np.zeros((T, N), dtype=np.float32) if need else None
        for t in range(T):
            x = self._step(x, U[t])
            if need:
                X[t] = x
        Y = None
        if wout is not None:
            Wout = np.asarray(wout, dtype=np.float32)
            Y = X @ Wout
        if (wout is not None) and collect_states:
            return X, Y
        if wout is not None:
            return Y
        if collect_states:
            return X
        return None

    def fit(self, U, Y, washout=100, ridge=1e-4):
        U = _vinp(U, self.n_inputs)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        X = self._collect(U, washout)
        Y_w = Y[washout:]
        A = X.T @ X
        A[np.diag_indices_from(A)] += ridge
        self.Wout = np.linalg.solve(A, X.T @ Y_w).astype(np.float32)
        return self

    def predict(self, U, washout=0):
        if self.Wout is None:
            raise RuntimeError("fit first")
        U = _vinp(U, self.n_inputs)
        return self._collect(U, washout) @ self.Wout

    def transform(self, U, washout=100):
        return self._collect(_vinp(U, self.n_inputs), washout)

    def _resize(self, A, k):
        N = A.shape[0]
        if k == N: return A
        if k < N:
            deg = A.sum(axis=1) + A.sum(axis=0)
            keep = np.sort(np.argsort(-deg)[:k])
            Ak = A[np.ix_(keep, keep)]; np.fill_diagonal(Ak, 0.0); return Ak
        P = self.rng.standard_normal((k, N)).astype(np.float64)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        Ak = (P @ A) @ P.T; np.fill_diagonal(Ak, 0.0); return Ak.astype(A.dtype, copy=False)


def _vinp(U, n_inputs):
    U = np.asarray(U, dtype=np.float32)
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if U.shape[1] != n_inputs:
        raise ValueError(f"got {U.shape[1]} inputs; expected {n_inputs}")
    return U


def _attr_exists(graphs, attr):
    for g in graphs:
        for _, _, d in g.edges(data=True):
            if attr in d:
                try:
                    float(d[attr]); return True
                except Exception:
                    continue
    return False


def _spectral_radius(A, max_iter=200, tol=1e-6):
    rng = np.random.default_rng(0)
    n = A.shape[0]
    x = rng.standard_normal(n); x /= np.linalg.norm(x) + 1e-30
    lam_prev = 0.0
    for _ in range(max_iter):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny < 1e-30 or not math.isfinite(ny): break
        x_new = y / ny
        lam = float(x_new @ (A @ x_new))
        if abs(lam - lam_prev) < tol:
            return abs(lam)
        x, lam_prev = x_new, lam
    try:
        return float(np.max(np.abs(np.linalg.eigvals(A))))
    except Exception:
        return 0.0
