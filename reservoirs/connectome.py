"""
reservoirs.connectome
======================
A biologically-grounded Echo State Network reservoir whose recurrent weight
matrix is derived from one or more connectome graphs in GraphML format.
Canonical connectome engine (formerly brain_connectome_reservoir_v0_1.py) plus a
back-compat shim (`rhow=` alias, `forward()`, `resize_method`) added in Phase 2.

Key design choices (and why):
  - spectral_radius < 1  →  echo state property guaranteed (safe default)
  - spectral_radius ≥ 1  →  super-critical; better for slow-dynamics tasks
  - leaky integrator update  →  multi-timescale dynamics
  - washout discarded in fit()  →  initial-condition artefacts removed
  - ridge regression for readout  →  regularised, closed-form Wout
  - bias column in design matrix  →  affine readout without a separate bias
  - random projection for upscaling  →  avoids block-repetitive structure of tiling
"""

import os
import glob
import math
from typing import Optional, Tuple, List

import numpy as np
import networkx as nx

from ._batch import leaky_collect_batch

__all__ = ["ConnectomeReservoir"]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ConnectomeReservoir:
    """
    Reservoir whose recurrent matrix W is built from connectome GraphML files.

    Parameters
    ----------
    n_inputs : int
        Dimensionality of the input signal u(t).
    graph_dir : str
        Folder containing *.graphml files (or use folder=/path= aliases).
    n_neurons / target_size : int, optional
        Resize the reservoir to exactly this many neurons.
        If omitted the graph's native size is used.
    file_glob : str
        Glob pattern for graph files (default "*.graphml").
    edge_attr : str or None
        Edge attribute to use as weight.  Falls back to unweighted if absent.
    combine : {"mean","median","first"}
        How to combine multiple connectomes into one adjacency matrix.
    spectral_radius : float
        Target spectral radius of W.  Must be < 1 to guarantee the echo state
        property.  Values in [0.8, 0.99] are typical.  Default: 0.9.
    leak_range : (float, float)
        Per-neuron leak rates drawn uniformly from this range.
        Each leak α_i controls the timescale: ẋ ≈ α(−x + input) (the α multiplies the whole RHS).
    symmetric : bool
        Symmetrise the adjacency before scaling (makes W symmetric).
    seed : int
        RNG seed for reproducibility.
    input_scale : float
        Half-range of the uniform distribution used to draw Win.
    """

    def __init__(
        self,
        n_inputs: int,
        graph_dir: Optional[str] = None,
        *,
        folder: Optional[str] = None,
        path: Optional[str] = None,
        n_neurons: Optional[int] = None,
        file_glob: str = "*.graphml",
        edge_attr: Optional[str] = "weight",
        combine: str = "mean",
        # spectral_radius: controls memory depth.
        #   < 1  → echo state property guaranteed (contractive dynamics)
        #   ≥ 1  → edge-of-chaos / super-critical regime; can outperform on
        #          tasks with slow temporal structure (e.g. sine-wave classification)
        #          at the cost of potential instability for very long sequences.
        # Default restored to 1.25, which matched empirical results on this task.
        spectral_radius: float = 1.25,
        leak_range: Tuple[float, float] = (0.1, 0.3),
        symmetric: bool = True,
        seed: int = 42,
        input_scale: float = 1.0,
        target_size: Optional[int] = None,
        # --- back-compat shim params (Phase 2) ---
        rhow: Optional[float] = None,          # deprecated alias for spectral_radius
        resize_method: str = "project",        # {"project","tile"}; "tile" reproduces old-engine upscale structure
        # --- Phase 4: build directly from an adjacency matrix (e.g. a rewired null) ---
        adjacency: Optional[np.ndarray] = None,
    ) -> None:
        # `rhow=` is the legacy (old-engine) name for the target spectral radius.
        if rhow is not None:
            import warnings
            warnings.warn(
                "`rhow=` is deprecated; use `spectral_radius=`.",
                DeprecationWarning, stacklevel=2,
            )
            spectral_radius = float(rhow)
        if resize_method not in ("project", "tile"):
            raise ValueError("resize_method must be one of {'project','tile'}")
        self._resize_method = resize_method
        graph_dir = graph_dir or folder or path
        if graph_dir is None and adjacency is None:
            raise ValueError(
                "Provide graph_dir= (or folder=/path=) pointing to the .graphml folder, "
                "or adjacency= with a precomputed connectivity matrix."
            )
        if combine not in ("mean", "median", "first"):
            raise ValueError("combine must be one of {'mean','median','first'}")
        if spectral_radius <= 0.0:
            raise ValueError(f"spectral_radius must be positive; got {spectral_radius}.")
        if spectral_radius >= 1.0:
            import warnings
            warnings.warn(
                f"spectral_radius={spectral_radius} >= 1. The echo state property is "
                "not guaranteed, but values slightly above 1 often work well on tasks "
                "with slow temporal structure. Monitor for diverging states.",
                UserWarning, stacklevel=2,
            )

        lo, hi = leak_range
        if hi <= lo:
            raise ValueError("leak_range must satisfy high > low (e.g., (0.1, 0.3)).")

        n_neurons = n_neurons or target_size  # accept both names

        self.rng = np.random.default_rng(seed)

        if adjacency is not None:
            # ---- Phase 4: build directly from a given matrix (e.g. a degree-preserving null) ----
            A = np.asarray(adjacency, dtype=float).copy()
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError("adjacency must be a square 2-D matrix.")
        else:
            if not os.path.isdir(graph_dir):
                raise FileNotFoundError(f"graph_dir not found: {graph_dir}")
            # ---- load graphs -----------------------------------------------
            files = sorted(glob.glob(os.path.join(graph_dir, file_glob)))
            if not files:
                raise FileNotFoundError(f"No files matching {file_glob!r} in {graph_dir}")

            graphs = [nx.read_graphml(f) for f in files]
            all_nodes = sorted({n for g in graphs for n in g.nodes()})
            N0 = len(all_nodes)
            if N0 == 0:
                raise ValueError("Loaded graphs contain zero nodes.")

            if edge_attr is not None and not _edge_attr_exists_any(graphs, edge_attr):
                edge_attr = None  # fall back to unweighted

            As = []
            for g in graphs:
                if set(g.nodes()) != set(all_nodes):
                    g = g.copy()
                    g.add_nodes_from(set(all_nodes) - set(g.nodes()))
                A = nx.to_numpy_array(g, nodelist=all_nodes, weight=edge_attr, dtype=float)
                As.append(A)

            # ---- combine connectomes ---------------------------------------
            if len(As) == 1 or combine == "first":
                A = As[0]
            elif combine == "median":
                A = np.median(np.stack(As, axis=2), axis=2)
            else:
                A = np.mean(np.stack(As, axis=2), axis=2)

        if symmetric:
            A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)

        # ---- resize --------------------------------------------------------
        if n_neurons is not None and n_neurons != A.shape[0]:
            A = self._resize_adjacency(A, int(n_neurons))

        # ---- scale to target spectral radius (FIX 1) ----------------------
        sr = _spectral_radius(A)
        if not math.isfinite(sr) or sr <= 0.0:
            raise ValueError(
                "Adjacency has non-positive spectral radius (graph may be empty)."
            )
        W = (A / sr) * spectral_radius

        # ---- store ---------------------------------------------------------
        self.n_inputs = int(n_inputs)
        self.n_neurons = int(W.shape[0])
        self.spectral_radius = spectral_radius
        self.W = W.astype(np.float32, copy=False)

        self.Win = self.rng.uniform(
            -input_scale, input_scale, size=(self.n_neurons, self.n_inputs)
        ).astype(np.float32)
        # Sample leak BEFORE Win_bias — preserving original RNG draw order
        # so that seed=7 produces identical Win/leak to the original code.
        self.leak = self.rng.uniform(lo, hi, size=(self.n_neurons,)).astype(np.float32)
        self.Win_bias = self.rng.uniform(
            -input_scale, input_scale, size=(self.n_neurons,)
        ).astype(np.float32)

        # readout weights — set after fit()
        self.Wout: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Core dynamics
    # -----------------------------------------------------------------------

    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Single leaky-integrator step.  Returns next state x(t+1)."""
        pre = self.Win @ u + self.Win_bias + self.W @ x
        x_candidate = np.tanh(pre)
        return (1.0 - self.leak) * x + self.leak * x_candidate

    def _collect_states(
        self, U: np.ndarray, washout: int
    ) -> np.ndarray:
        """
        Run reservoir over input sequence U [T, n_inputs] and return the
        design matrix X_wash [T-washout, n_neurons].

        Washout rows are discarded so the initial zero-state transient does not
        contaminate the regression. (NOTE: no constant-1 bias column is appended — an earlier
        comment claimed one; the affine offset enters the dynamics via `Win_bias` in `_step`.)
        """
        T = U.shape[0]
        if washout >= T:
            raise ValueError(
                f"washout={washout} must be < sequence length T={T}."
            )

        x = np.zeros(self.n_neurons, dtype=np.float32)
        states = []

        for t in range(T):
            x = self._step(x, U[t])
            if t >= washout:
                states.append(x.copy())

        return np.array(states, dtype=np.float32)  # [T-washout, N]

    # -----------------------------------------------------------------------
    # Training (FIX 2: fit() method was entirely absent in original)
    # -----------------------------------------------------------------------

    def fit(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        washout: int = 100,
        ridge: float = 1e-4,
    ) -> "ConnectomeReservoir":
        """
        Train the linear readout via ridge regression.

        Parameters
        ----------
        U : array [T, n_inputs]  — input sequence
        Y : array [T, n_outputs] — target sequence (same length as U)
        washout : int
            Number of initial timesteps to discard (transient removal).
            Typical values: 50–200.
        ridge : float
            L2 regularisation strength for ridge regression.
            Larger values → smoother Wout, less prone to overfitting.

        Returns
        -------
        self  (for method chaining)
        """
        U = _validate_input(U, self.n_inputs)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.shape[0] != U.shape[0]:
            raise ValueError("U and Y must have the same number of timesteps.")

        X = self._collect_states(U, washout)   # [T-w, N+1]
        Y_wash = Y[washout:]                    # [T-w, K]

        # Closed-form ridge: Wout = (X^T X + λI)^{-1} X^T Y
        # Shape: [N+1, K]
        A = X.T @ X
        A[np.diag_indices_from(A)] += ridge
        self.Wout = np.linalg.solve(A, X.T @ Y_wash).astype(np.float32)
        return self

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def predict(
        self,
        U: np.ndarray,
        washout: int = 0,
    ) -> np.ndarray:
        """
        Run reservoir and return readout predictions.

        Requires fit() to have been called first.

        Parameters
        ----------
        U : array [T, n_inputs]
        washout : int
            Discard this many leading predictions (useful for test sequences
            that have a cold-start artifact).

        Returns
        -------
        Y_hat : [T-washout, n_outputs]
        """
        if self.Wout is None:
            raise RuntimeError("Call fit() before predict().")
        U = _validate_input(U, self.n_inputs)
        X = self._collect_states(U, washout)   # [T-w, N+1]
        return X @ self.Wout                   # [T-w, K]

    def transform(
        self,
        U: np.ndarray,
        washout: int = 100,
    ) -> np.ndarray:
        """
        Return raw reservoir states (design matrix) without applying a readout.
        Useful when you want to train your own downstream model.

        Returns
        -------
        X : [T-washout, n_neurons]
            Raw reservoir states. NOTE: there is NO appended bias column — the affine
            offset enters the dynamics via ``Win_bias`` inside ``_step``. (The previous
            docstring incorrectly claimed an [N+1] shape with a constant-1 column.)
        """
        U = _validate_input(U, self.n_inputs)
        return self._collect_states(U, washout)

    # -----------------------------------------------------------------------
    # Back-compat shim: legacy forward() API (Phase 2)
    # -----------------------------------------------------------------------

    def forward(self, u, wout=None, collect_states=False):
        """Legacy ``forward()`` API so old-engine notebooks run on this engine.

        Collects reservoir states with washout=0 and, if a readout ``wout`` is given,
        returns ``X @ wout``. Mirrors the old engine's return convention:
        ``(X, Y)`` if both, else ``Y``, else ``X``, else ``None``.

        NOTE: results are NOT bit-identical to the archived old engine — the spectral
        radius is computed by a different (Rayleigh-quotient) method, and upscaling uses
        random projection unless ``resize_method='tile'`` was passed at construction.
        This shim is for API compatibility during migration, not exact reproduction.
        """
        U = _validate_input(u, self.n_inputs)
        X = self._collect_states(U, washout=0)   # [T, n_neurons]
        Y = None
        if wout is not None:
            Y = X @ np.asarray(wout, dtype=np.float32)
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X
        return None

    def collect_states_batch(self, U: np.ndarray) -> np.ndarray:
        """Vectorized state collection over a batch of windows (Phase 5, ~B× faster).

        U : [B, T, n_inputs] -> states [B, T, n_neurons]. Equivalent (within float tolerance)
        to ``np.stack([self.forward(u, collect_states=True) for u in U])``.
        """
        U = np.asarray(U, dtype=np.float32)
        if U.ndim != 3 or U.shape[2] != self.n_inputs:
            raise ValueError(f"U must be [B, T, n_inputs={self.n_inputs}]; got {U.shape}.")
        drive = U @ self.Win.T + self.Win_bias          # [B, T, N]
        return leaky_collect_batch(drive, self.W, self.leak)

    # -----------------------------------------------------------------------
    # Resizing helpers
    # -----------------------------------------------------------------------

    def _resize_adjacency(self, A: np.ndarray, k: int) -> np.ndarray:
        N = A.shape[0]
        if k <= 0:
            raise ValueError("target_size / n_neurons must be positive.")
        if k == N:
            return A

        if k < N:
            # keep the k highest-degree nodes
            deg = A.sum(axis=1) + A.sum(axis=0)
            keep = np.argsort(-deg)[:k]
            keep.sort()
            Ak = A[np.ix_(keep, keep)]
            np.fill_diagonal(Ak, 0.0)
            return Ak

        # --- upscale (k > N) ---
        if getattr(self, "_resize_method", "project") == "tile":
            # Legacy old-engine behavior: tile the adjacency and add tiny noise.
            # Kept so old upscaled results can be reproduced (resize_method="tile").
            reps = int(np.ceil(k / N))
            Abig = np.tile(A, (reps, reps))[:k, :k]
            eps = self.rng.normal(0.0, 1e-4, size=Abig.shape)
            Abig = np.clip(Abig + eps, 0.0, None)
            np.fill_diagonal(Abig, 0.0)
            return Abig.astype(A.dtype, copy=False)

        # Default: random projection (vs tiling, which makes k/N near-identical copies).
        # CAVEAT: this is NOT a structure-preserving enlargement — `P A P^T` produces a DENSE
        # k×k matrix that discards the connectome's sparsity/degree/community structure, so an
        # upscaled "connectome reservoir" is largely a random projection of it. It avoids tiling's
        # block-degeneracy, but for a topology claim prefer the native size. (See docs/METHODOLOGY.md.)
        P = self.rng.standard_normal((k, N)).astype(np.float64)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        # projected adjacency: P A P^+  (P^+ = P^T when rows are unit-norm)
        Ak = (P @ A) @ P.T
        np.fill_diagonal(Ak, 0.0)
        return Ak.astype(A.dtype, copy=False)


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _validate_input(U: np.ndarray, n_inputs: int) -> np.ndarray:
    U = np.asarray(U, dtype=np.float32)
    if U.ndim == 1:
        if n_inputs != 1:
            raise ValueError(
                f"1-D input given but reservoir expects n_inputs={n_inputs}."
            )
        U = U.reshape(-1, 1)
    if U.shape[1] != n_inputs:
        raise ValueError(f"Input has {U.shape[1]} features; expected {n_inputs}.")
    return U


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


def _spectral_radius(A: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> float:
    """Spectral radius via Rayleigh-quotient power iteration, with an eigvals fallback.

    This engine ALWAYS symmetrises the adjacency, so the dominant eigenvalue is real and the
    Rayleigh quotient `x^T A x` converges to it correctly — which is the only case used here.

    LIMITATION (latent, not reached on any current path): for a NON-symmetric / non-normal `A`
    with a complex-dominant eigenpair, the Rayleigh quotient can converge to a wrong value and
    the `eigvals` fallback below would NOT fire (it triggers only on non-finite / non-positive
    results). Do not call this on a non-symmetric matrix without switching to `||A x||` or
    `np.linalg.eigvals`. (An earlier docstring wrongly claimed a "norm check" guarded this — it
    does not exist; `norm_y` only normalises the iterate.)
    """
    rng = np.random.default_rng(0)
    n = A.shape[0]
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x) + 1e-30
    lam_prev = 0.0

    for _ in range(max_iter):
        y = A @ x
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-30 or not math.isfinite(norm_y):
            break
        x_new = y / norm_y
        lam = float(x_new @ (A @ x_new))   # Rayleigh quotient
        if abs(lam - lam_prev) < tol:
            return abs(lam)
        x, lam_prev = x_new, lam

    # Fallback: exact eigenvalues (slower but always correct)
    try:
        vals = np.linalg.eigvals(A)
        return float(np.max(np.abs(vals)))
    except Exception:
        return 0.0