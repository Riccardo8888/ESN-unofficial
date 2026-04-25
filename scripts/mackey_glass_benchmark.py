#!/usr/bin/env python3
"""
Mackey-Glass benchmark for the four reservoir types.

Recreates the Section VIII comparison in the paper by training each reservoir
to predict the Mackey-Glass time series one step ahead and reporting Memory
Capacity (MC = 1 - NRMSE) and R^2 on a held-out test split.

Reservoir types tested
----------------------
* ErdosRenyiReservoir         (random sparse, p=0.1 by default)
* FullyConnectedReservoir     (dense uniform, all-to-all)
* GaussianReservoir           (Gaussian weights with sigma=J/sqrt(N))
* ConnectomeReservoir         (HCP adjacency, default 234-node folder)

All four reservoirs use the same N (default 234), same target spectral radius
(default 1.0), same leak range, same washout, same ridge regularisation, and
the same Mackey-Glass series so that differences come from the recurrent
weight structure alone.

Usage
-----
    python scripts/mackey_glass_benchmark.py                        # all four, 234 nodes
    python scripts/mackey_glass_benchmark.py --n-neurons 463        # different size
    python scripts/mackey_glass_benchmark.py --graph-dir data/folder_path_1015
    python scripts/mackey_glass_benchmark.py --reservoir connectome # only one type
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reservoirs.ErdosRenyi import ErdosRenyiReservoir
from reservoirs.FullyConnected import FullyConnectedReservoir
from reservoirs.Gaussian import GaussianReservoir
from reservoirs.brain_connectome_reservoir import ConnectomeReservoir


# ---------------------------------------------------------------------------
# Mackey-Glass generator
# ---------------------------------------------------------------------------

def mackey_glass(n_steps: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, n: float = 10.0,
                 dt: float = 1.0, x0: float = 1.2, seed: int = 42) -> np.ndarray:
    """
    Generate a Mackey-Glass series via Euler integration of:

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Default parameters (tau=17, n=10, beta=0.2, gamma=0.1) put the system in
    its standard mildly-chaotic regime and match what is used in most
    reservoir-computing benchmarks (Jaeger 2001 etc.).
    """
    rng = np.random.default_rng(seed)
    history = max(int(tau / dt), 1) + 1
    x = np.full(history, x0, dtype=np.float64)
    # Tiny perturbation so different seeds give different burn-ins.
    x += rng.normal(0.0, 1e-3, size=history)

    out = np.empty(n_steps, dtype=np.float64)
    for t in range(n_steps):
        x_now = x[-1]
        x_delay = x[0]
        dxdt = beta * x_delay / (1.0 + x_delay ** n) - gamma * x_now
        x_next = x_now + dt * dxdt
        out[t] = x_next
        x = np.roll(x, -1)
        x[-1] = x_next
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised RMSE: RMSE divided by the std of the target signal."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sd = float(np.std(y_true))
    return rmse / sd if sd > 0 else float("inf")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Reservoir builders
# ---------------------------------------------------------------------------

def build_reservoir(kind: str, *, n_inputs: int, n_neurons: int,
                    spectral_radius: float, leak_range, seed: int,
                    graph_dir: Path | None = None,
                    p_er: float = 0.1):
    if kind == "erdos_renyi":
        return ErdosRenyiReservoir(
            n_inputs=n_inputs, n_neurons=n_neurons, rhow=spectral_radius,
            leak_range=leak_range, p=p_er, seed=seed,
        )
    if kind == "fully_connected":
        return FullyConnectedReservoir(
            n_inputs=n_inputs, n_neurons=n_neurons, rhow=spectral_radius,
            leak_range=leak_range, seed=seed,
        )
    if kind == "gaussian":
        return GaussianReservoir(
            n_inputs=n_inputs, n_neurons=n_neurons, rhow=spectral_radius,
            leak_range=leak_range, J=1.0, seed=seed,
        )
    if kind == "connectome":
        if graph_dir is None:
            raise ValueError("connectome requires --graph-dir")
        return ConnectomeReservoir(
            n_inputs=n_inputs, graph_dir=str(graph_dir), n_neurons=n_neurons,
            spectral_radius=spectral_radius, leak_range=leak_range,
            edge_attr="weight", combine="mean", symmetric=True, seed=seed,
        )
    raise ValueError(f"Unknown reservoir kind: {kind}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _ridge_with_bias(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ridge regression that fits an affine readout: Y ~= [X | 1] @ W.

    Without the bias column, the readout cannot reproduce a non-zero mean
    target (Mackey-Glass mean is ~0.93), which is what was breaking the
    initial benchmark.
    """
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    A = X1.T @ X1
    A[np.diag_indices_from(A)] += alpha
    return np.linalg.solve(A, X1.T @ Y)


def _apply_readout(X: np.ndarray, Wout: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
    return X1 @ Wout


def benchmark_one(kind: str, series: np.ndarray, *,
                  n_neurons: int, spectral_radius: float, leak_range,
                  washout: int, ridge: float, train_len: int, test_len: int,
                  seed: int, graph_dir: Path | None, p_er: float) -> dict:
    """
    Train one reservoir on the one-step-ahead prediction task and evaluate.
    Uses a uniform manual-ridge path (with bias column) for all reservoir
    types so the comparison is apples-to-apples.
    """
    u = series[:-1].reshape(-1, 1).astype(np.float32)
    y = series[1:].reshape(-1, 1).astype(np.float32)

    u_train, y_train = u[:train_len], y[:train_len]
    u_test, y_test = u[train_len:train_len + test_len], y[train_len:train_len + test_len]

    t0 = time.time()
    res = build_reservoir(
        kind, n_inputs=1, n_neurons=n_neurons,
        spectral_radius=spectral_radius, leak_range=leak_range, seed=seed,
        graph_dir=graph_dir, p_er=p_er,
    )

    # Run forward over training, drop washout, fit affine readout.
    X_train = np.asarray(res.forward(u_train, collect_states=True))
    X_train_w = X_train[washout:]
    Y_train_w = y_train[washout:]
    Wout = _ridge_with_bias(X_train_w, Y_train_w, alpha=ridge)

    # Predict on the test split (no washout — keep state from training? No,
    # the reservoir resets at the start of the next forward(). To avoid a
    # cold-start transient on the test split, we apply a small washout there
    # too and align both arrays.)
    X_test = np.asarray(res.forward(u_test, collect_states=True))
    test_washout = min(washout, max(1, X_test.shape[0] // 10))
    y_pred = _apply_readout(X_test[test_washout:], Wout)
    y_eval = y_test[test_washout:]

    elapsed = time.time() - t0
    nrmse_val = nrmse(y_eval, y_pred)
    mc = 1.0 - nrmse_val
    r2 = r2_score(y_eval, y_pred)
    return {
        "reservoir": kind,
        "n_neurons": int(getattr(res, "n_neurons", n_neurons)),
        "spectral_radius": float(spectral_radius),
        "MC": float(mc),
        "R2": float(r2),
        "NRMSE": float(nrmse_val),
        "fit_time_s": float(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mackey-Glass benchmark.")
    parser.add_argument("--reservoir", default="all",
                        choices=["all", "erdos_renyi", "fully_connected",
                                 "gaussian", "connectome"])
    parser.add_argument("--n-neurons", type=int, default=234,
                        help="Reservoir size (default 234, matching the paper).")
    parser.add_argument("--spectral-radius", type=float, default=1.0)
    parser.add_argument("--leak-min", type=float, default=0.1)
    parser.add_argument("--leak-max", type=float, default=0.3)
    parser.add_argument("--washout", type=int, default=200)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--train-len", type=int, default=2000)
    parser.add_argument("--test-len", type=int, default=2000)
    parser.add_argument("--graph-dir", type=str, default=None,
                        help="Connectome folder. Defaults to data/folder_path_<n_neurons> if it exists.")
    parser.add_argument("--p-er", type=float, default=0.1,
                        help="Connection probability for the Erdős-Rényi reservoir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path for a JSON results file.")
    args = parser.parse_args()

    # Resolve the connectome folder.
    if args.graph_dir is None:
        cand = REPO_ROOT / "data" / f"folder_path_{args.n_neurons}"
        graph_dir = cand if cand.is_dir() and any(cand.glob("*.graphml")) else None
    else:
        graph_dir = Path(args.graph_dir)
        if not graph_dir.is_dir():
            raise SystemExit(f"--graph-dir not found: {graph_dir}")

    if args.reservoir in ("all", "connectome") and graph_dir is None:
        print("WARNING: no connectome folder found; skipping connectome reservoir.")

    # Generate the Mackey-Glass series.
    n_steps = args.train_len + args.test_len + args.washout + 1
    print(f"Generating Mackey-Glass series with {n_steps} steps...")
    series = mackey_glass(n_steps=n_steps, seed=args.seed)
    print(f"  series stats:  mean={series.mean():.4f}  std={series.std():.4f}  "
          f"min={series.min():.4f}  max={series.max():.4f}")

    leak_range = (args.leak_min, args.leak_max)

    if args.reservoir == "all":
        kinds = ["erdos_renyi", "fully_connected", "gaussian"]
        if graph_dir is not None:
            kinds.append("connectome")
    else:
        kinds = [args.reservoir]

    print(f"\nBenchmarking on N={args.n_neurons}, spectral_radius={args.spectral_radius}, "
          f"washout={args.washout}, ridge={args.ridge}\n")
    rows = []
    for kind in kinds:
        try:
            r = benchmark_one(
                kind, series,
                n_neurons=args.n_neurons,
                spectral_radius=args.spectral_radius,
                leak_range=leak_range,
                washout=args.washout, ridge=args.ridge,
                train_len=args.train_len, test_len=args.test_len,
                seed=args.seed,
                graph_dir=graph_dir,
                p_er=args.p_er,
            )
            rows.append(r)
            print(f"  {kind:18s}  MC={r['MC']:.4f}  R2={r['R2']:.4f}  "
                  f"NRMSE={r['NRMSE']:.4f}  ({r['fit_time_s']:.2f}s)")
        except Exception as e:
            print(f"  {kind:18s}  FAILED: {e}")

    print()
    print("=" * 60)
    print("Mackey-Glass benchmark summary")
    print("=" * 60)
    print(f"{'reservoir':<18s} {'N':>5s} {'rhow':>5s} {'MC':>7s} {'R2':>7s} {'NRMSE':>7s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['reservoir']:<18s} {r['n_neurons']:>5d} {r['spectral_radius']:>5.2f}"
              f" {r['MC']:>7.4f} {r['R2']:>7.4f} {r['NRMSE']:>7.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps({
            "config": vars(args),
            "graph_dir": str(graph_dir) if graph_dir else None,
            "results": rows,
        }, indent=2, default=str))
        print(f"\nResults JSON saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
