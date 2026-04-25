#!/usr/bin/env python3
"""
Regenerate the four legacy Mackey-Glass per-reservoir figures with the new
tuned HPs so their plot titles match Table I (mg_main).

Outputs (filenames match the existing \\includegraphics calls in main.tex):

    scripts/figures/Mackey-Glass_Erdos-Renyi.png
    scripts/figures/Mackey-Glass_fully_connected.png
    scripts/figures/Mackey-Glass_Gaussian.png
    scripts/figures/Strength-weighted_234_connectome.png

Each plot shows the target Mackey-Glass series vs the one-step-ahead reservoir
prediction on the test split, with the actual MC / R^2 / NRMSE in the title.

Usage
-----
    python scripts/regenerate_legacy_mg_figures.py \
        --graph-dir data/folder_path_234

Defaults reuse the same HPs that produced Table I:
    spectral_radius = 0.7, leak = [0.2, 0.7], washout = 200, ridge = 1e-6, N = 234.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mackey_glass_benchmark import (
    mackey_glass, build_reservoir, _ridge_with_bias, _apply_readout,
    nrmse, r2_score,
)

FIG_DIR = REPO_ROOT / "scripts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (kind, output filename, plot title prefix, line color)
LEGACY = [
    ("erdos_renyi",     "Mackey-Glass_Erdos-Renyi.png",        "Erdős-Rényi reservoir", "#888888"),
    ("fully_connected", "Mackey-Glass_fully_connected.png",    "Fully-connected reservoir", "#f08c00"),
    ("gaussian",        "Mackey-Glass_Gaussian.png",           "Gaussian reservoir", "#3aa648"),
    ("connectome",      "Strength-weighted_234_connectome.png", "Strength-weighted 234-node connectome", "#1f77ff"),
]


def run_one(kind, *, series, n_neurons, spectral_radius, leak_range, washout,
            ridge, train_len, test_len, seed, graph_dir, p_er):
    u = series[:-1].reshape(-1, 1).astype(np.float32)
    y = series[1:].reshape(-1, 1).astype(np.float32)
    u_train, y_train = u[:train_len], y[:train_len]
    u_test, y_test = u[train_len:train_len + test_len], y[train_len:train_len + test_len]

    res = build_reservoir(
        kind, n_inputs=1, n_neurons=n_neurons,
        spectral_radius=spectral_radius, leak_range=leak_range, seed=seed,
        graph_dir=graph_dir, p_er=p_er,
    )
    X_train = np.asarray(res.forward(u_train, collect_states=True))
    Wout = _ridge_with_bias(X_train[washout:], y_train[washout:], alpha=ridge)
    X_test = np.asarray(res.forward(u_test, collect_states=True))
    test_washout = min(washout, max(1, X_test.shape[0] // 10))
    y_pred = _apply_readout(X_test[test_washout:], Wout).reshape(-1)
    y_eval = y_test[test_washout:].reshape(-1)
    n_used = int(getattr(res, "n_neurons", n_neurons))
    return y_eval, y_pred, n_used


def plot_one(out_path, title_prefix, color, y_eval, y_pred, n_used,
             show_steps=600):
    """Two stacked panels: target vs prediction (top) and residual (bottom)."""
    err = y_pred - y_eval
    nrmse_v = nrmse(y_eval, y_pred)
    mc = 1.0 - nrmse_v
    r2 = r2_score(y_eval, y_pred)

    s = min(show_steps, len(y_eval))
    t = np.arange(s)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 4.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax_top.plot(t, y_eval[:s], color="k", lw=1.4, label="target")
    ax_top.plot(t, y_pred[:s], color=color, lw=1.1, ls="--", label="prediction")
    ax_top.set_ylabel("$x(t)$")
    ax_top.legend(loc="upper right", fontsize=8, frameon=False)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(
        f"{title_prefix}  (N={n_used}, MC={mc:.4f}, "
        f"$R^2$={r2:.4f}, NRMSE={nrmse_v:.4f})",
        fontsize=10,
    )

    ax_bot.plot(t, err[:s], color=color, lw=0.8)
    ax_bot.axhline(0, color="k", lw=0.5)
    ax_bot.set_ylabel("residual")
    ax_bot.set_xlabel("test step")
    ax_bot.grid(True, alpha=0.3)
    # Symmetric residual scale (drop large early transient if any).
    ymax = float(np.max(np.abs(err[:s]))) if s else 0.0
    if ymax > 0:
        ax_bot.set_ylim(-1.05 * ymax, 1.05 * ymax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Regenerate legacy MG figures.")
    p.add_argument("--n-neurons", type=int, default=234)
    p.add_argument("--spectral-radius", type=float, default=0.7)
    p.add_argument("--leak-min", type=float, default=0.2)
    p.add_argument("--leak-max", type=float, default=0.7)
    p.add_argument("--washout", type=int, default=200)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--train-len", type=int, default=2000)
    p.add_argument("--test-len", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--p-er", type=float, default=0.1)
    p.add_argument("--show-steps", type=int, default=600,
                   help="Number of test steps to show in each plot.")
    p.add_argument("--graph-dir", type=str, default=None,
                   help="Connectome graph folder. Defaults to data/folder_path_<n>.")
    p.add_argument("--only", default=None,
                   choices=[k for k, *_ in LEGACY],
                   help="Regenerate just one figure.")
    args = p.parse_args()

    # Resolve connectome folder.
    if args.graph_dir is None:
        cand = REPO_ROOT / "data" / f"folder_path_{args.n_neurons}"
        graph_dir = cand if cand.is_dir() and any(cand.glob("*.graphml")) else None
    else:
        graph_dir = Path(args.graph_dir)
        if not graph_dir.is_dir():
            raise SystemExit(f"--graph-dir not found: {graph_dir}")

    n_steps = args.train_len + args.test_len + args.washout + 1
    print(f"Generating Mackey-Glass series ({n_steps} steps)...")
    series = mackey_glass(n_steps=n_steps, seed=args.seed)
    leak_range = (args.leak_min, args.leak_max)

    todo = LEGACY if args.only is None else [r for r in LEGACY if r[0] == args.only]
    for kind, fname, title_prefix, color in todo:
        if kind == "connectome" and graph_dir is None:
            print(f"  SKIP {kind}: no graph_dir found.")
            continue
        try:
            y_eval, y_pred, n_used = run_one(
                kind, series=series,
                n_neurons=args.n_neurons,
                spectral_radius=args.spectral_radius,
                leak_range=leak_range, washout=args.washout, ridge=args.ridge,
                train_len=args.train_len, test_len=args.test_len,
                seed=args.seed, graph_dir=graph_dir, p_er=args.p_er,
            )
        except Exception as e:
            print(f"  FAILED {kind}: {e}")
            continue
        out = FIG_DIR / fname
        plot_one(out, title_prefix, color, y_eval, y_pred, n_used,
                 show_steps=args.show_steps)
        mc = 1.0 - nrmse(y_eval, y_pred)
        print(f"  -> {out}   (MC={mc:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
