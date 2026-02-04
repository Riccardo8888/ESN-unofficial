import os
import glob
import math
from typing import Optional, Tuple, List
import numpy as np
import networkx as nx

__all__ = ["ConnectomeReservoir"]


class ConnectomeReservoir:
    def __init__(
        self,
        n_inputs: int,
        graph_dir: Optional[str] = None,
        *,
        # aliases for graph_dir
        folder: Optional[str] = None,
        path: Optional[str] = None,
        n_neurons: Optional[int] = None,
        # options
        file_glob: str = "*.graphml",
        edge_attr: Optional[str] = "weight",
        combine: str = "mean",
        rhow: float = 1.25,
        leak_range: Tuple[float, float] = (0.1, 0.3),
        symmetric: bool = True,
        seed: int = 42,
        input_scale: float = 1.0,
        target_size: Optional[int] = None,
    ) -> None:
        graph_dir = graph_dir or folder or path
        if graph_dir is None:
            raise ValueError("Please provide graph_dir= (or folder=/path=) pointing to the .graphml folder.")

        if n_neurons is not None and target_size is None:
            target_size = int(n_neurons)

        if combine not in ("mean", "median", "first"):
            raise ValueError("combine must be one of {'mean','median','first'}")

        if not os.path.isdir(graph_dir):
            raise FileNotFoundError(f"graph_dir not found: {graph_dir}")

        self.rng = np.random.default_rng(seed)

        files = sorted(glob.glob(os.path.join(graph_dir, file_glob)))
        if not files:
            raise FileNotFoundError(f"No files matching {file_glob!r} in {graph_dir}")

        graphs = [nx.read_graphml(f) for f in files]

        # Determine unified node set (string IDs allowed)
        all_nodes = sorted({n for g in graphs for n in g.nodes()})
        N0 = len(all_nodes)
        if N0 == 0:
            raise ValueError("Loaded graphs contain zero nodes.")

        # Use provided edge_attr only if present in at least one graph
        if edge_attr is not None:
            if not _edge_attr_exists_any(graphs, edge_attr):
                edge_attr = None

        # Convert each graph to adjacency on the unified node order
        As = []
        for g in graphs:
            # Ensure all nodes exist in g (isolated nodes allowed)
            if set(g.nodes()) != set(all_nodes):
                g = g.copy()
                missing = set(all_nodes) - set(g.nodes())
                if missing:
                    g.add_nodes_from(missing)
            A = nx.to_numpy_array(g, nodelist=all_nodes, weight=edge_attr, dtype=float)
            As.append(A)

        # Combine multiple connectomes
        if len(As) == 1 or combine == "first":
            A = As[0]
        else:
            stack = np.stack(As, axis=2)
            if combine == "median":
                A = np.median(stack, axis=2)
            else:  # mean
                A = np.mean(stack, axis=2)

        # Clean up adjacency
        if symmetric:
            A = 0.5 * (A + A.T)

        # Remove self-loops; not necessary but cleaner
        np.fill_diagonal(A, 0.0)

        # resize to target_size
        if target_size is not None and target_size != A.shape[0]:
            A = self._resize_adjacency(A, int(target_size))

        # Scale to target spectral radius
        sr = _spectral_radius(A)
        if not math.isfinite(sr) or sr <= 0.0:
            raise ValueError("Adjacency has non-positive spectral radius (graph is empty or degenerate).")
        W = (A / sr) * float(rhow)

        # Store
        self.n_inputs = int(n_inputs)
        self.n_neurons = int(W.shape[0])
        self.W = W.astype(np.float32, copy=False)

        # Random input weights and per-neuron leak
        self.Win = self.rng.uniform(-input_scale, input_scale, size=(self.n_neurons, self.n_inputs)).astype(np.float32)
        lo, hi = leak_range
        if hi <= lo:
            raise ValueError("leak_range must satisfy high > low (e.g., (0.1, 0.3)).")
        self.leak = self.rng.uniform(lo, hi, size=(self.n_neurons,)).astype(np.float32)
        # Random input bias term so we can do win @ [u; 1]
        self.input_bias = self.rng.uniform(-input_scale, input_scale, size=(self.n_neurons,)).astype(np.float32)

        # Lowercase aliases (for compatibility with your snippet)
        self.w = self.W  # [N, N]
        # win will expect input augmented with a bias 1.0: shape [N, n_inputs + 1]
        self.win = np.concatenate([self.Win, self.input_bias[:, None]], axis=1)  # [N, Nin+1]

    # Dynamics
    def reset_state(self, x0: np.ndarray | None = None) -> np.ndarray:
        if x0 is None:
            self.x = np.zeros(self.n_neurons, dtype=np.float32)
        else:
            x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
            if x0.shape[0] != self.n_neurons:
                raise ValueError(f"x0 has {x0.shape[0]} dims; expected {self.n_neurons}")
            self.x = x0
        return self.x
    
    def forward(self, u, collect_states=False, wout=None):
        #run reservoir forward in time

        U = np.asarray(u, dtype=np.float32)
        if U.ndim == 1:
            if self.n_inputs != 1:
                raise ValueError(f"1D input given but reservoir expects n_inputs={self.n_inputs}")
            U = U.reshape(-1, 1)
        if U.shape[1] != self.n_inputs:
            raise ValueError(f"U has {U.shape[1]} inputs; expected {self.n_inputs}")

        T = U.shape[0]
        N = self.n_neurons

        # state
        x = np.zeros((N,), dtype=np.float32)

        # allocate design matrix if needed
        X = None
        need_X = collect_states or (wout is not None)
        if need_X:
            X = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            ut = U[t]
            x_next = np.tanh(self.win @ np.concatenate([ut, [1.0]], axis=0) + self.w @ x)
            x = (1.0 - self.leak) * x + self.leak * x_next
            if need_X:
                X[t, :] = x

        Y = None
        if wout is not None:
            Wout = np.asarray(wout, dtype=np.float32)
            # Accept wout as [N] or [N,K]
            if Wout.ndim == 1:
                if Wout.shape[0] != N:
                    raise ValueError(f"wout shape {Wout.shape} incompatible with N={N}")
                Y = X @ Wout
            elif Wout.ndim == 2:
                if Wout.shape[0] != N:
                    raise ValueError(f"wout shape {Wout.shape} incompatible with N={N}")
                Y = X @ Wout
            else:
                raise ValueError("wout must be 1D [N] or 2D [N,K].")

        if (wout is not None) and collect_states:
            return X, Y
        elif (wout is not None):
            return Y
        elif collect_states:
            return X
        else:
            return None
    
    # helpers / internals
    def _resize_adjacency(self, A: np.ndarray, k: int) -> np.ndarray:
        N = A.shape[0]
        if k <= 0:
            raise ValueError("target_size / n_neurons must be positive.")
        if k == N:
            return A

        if k < N:
            deg = A.sum(axis=1) + A.sum(axis=0)
            keep = np.argsort(-deg)[:k]
            keep.sort()
            Ak = A[np.ix_(keep, keep)]
            # Clean up numerical issues
            np.fill_diagonal(Ak, 0.0)
            return Ak
        
        reps = int(np.ceil(k / N))
        Abig = np.tile(A, (reps, reps))[:k, :k]
        eps = self.rng.normal(0.0, 1e-4, size=Abig.shape)
        Abig = np.clip(Abig + eps, 0.0, None)
        np.fill_diagonal(Abig, 0.0)
        return Abig.astype(A.dtype, copy=False)


# module-level utilities
def _edge_attr_exists_any(graphs: List[nx.Graph], attr: str) -> bool:
    for g in graphs:
        for _, _, d in g.edges(data=True):
            if attr in d:
                try:
                    float(d[attr])
                    return True
                except Exception:
                    continue
    return False


def _spectral_radius(A: np.ndarray, iters: int = 100) -> float:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(A.shape[0],))
    nrm = np.linalg.norm(x)
    if nrm == 0:
        x[0] = 1.0
    else:
        x /= nrm

    lam = 0.0
    for _ in range(iters):
        y = A @ x
        nrm = np.linalg.norm(y)
        if nrm == 0.0 or not math.isfinite(nrm):
            break
        x = y / nrm
        lam = float(np.linalg.norm(A @ x))

    if not math.isfinite(lam) or lam <= 0.0:
        try:
            vals = np.linalg.eigvals(A)
            lam = float(np.max(np.abs(vals)))
        except Exception:
            lam = 0.0
    return lam

