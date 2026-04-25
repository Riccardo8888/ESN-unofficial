#!/usr/bin/env python3
"""
End-to-end Slither.io sweep: connectome vs ER across reservoir sizes.

What it does:
  1. Loads all session data from vnicktest/scripts/data/ via the existing loader.
  2. Selects the top-X% of sessions by frame count (default 30%, AI_bot excluded
     by default since it is a scripted bot, not a human player).
  3. For each (reservoir_kind, N) in the sweep, trains a ridge readout and
     records angle / boost / angular-error metrics on a held-out test split.
  4. Saves per-run JSON + a summary CSV + a four-panel comparison PNG figure.

Output:
  scripts/results/slither_<kind>_n<N>.json     (one per run)
  scripts/results/slither_summary.csv          (all runs, one row each)
  scripts/figures/slither_top30_comparison.png

Usage (Windows / Git Bash / WSL / Linux):
  cd ESN_claude/ESN-unofficial
  python scripts/run_slither_connectome_sweep.py
        # default: top 30% humans-only, sweep N=300/500/1000, both reservoirs

  python scripts/run_slither_connectome_sweep.py \
        --top 0.30 --exclude AI_bot --graph-dir data/folder_path_234 \
        --sizes 300 500 1000 2000

If the connectome graph load is slow because folder_path_234 holds 1064
graphml files, point --graph-dir at a small subset (e.g. a folder with 5
copied .graphml files); the spectral-radius rescaling makes results
qualitatively the same.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "vnicktest" / "scripts" / "data"
RES_DIR = REPO / "scripts" / "results"
FIG_DIR = REPO / "scripts" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data subset
# ---------------------------------------------------------------------------

def session_frame_count(session_path: Path) -> int:
    """Read player_inputs/.zarray to get the frame count without loading data."""
    pi = session_path / "player_inputs"
    zarray = pi / ".zarray"
    if zarray.exists():
        try:
            meta = json.loads(zarray.read_text())
            shape = meta.get("shape", [])
            return int(shape[0]) if shape else 0
        except Exception:
            pass
    try:
        import zarr
        return int(zarr.open(str(pi), mode="r").shape[0])
    except Exception:
        return 0


def collect_sessions(data_dir: Path, exclude=()):
    sessions = []
    for user_dir in sorted(data_dir.iterdir()):
        if not user_dir.is_dir() or user_dir.name in exclude:
            continue
        for sess in sorted(user_dir.iterdir()):
            if sess.is_dir() and (sess.name.startswith("session_")
                                  or sess.name.startswith("game_")):
                if (sess / "grids").exists():
                    sessions.append(sess)
    return sessions


def select_top_sessions(data_dir: Path, top: float, exclude=()):
    sessions = collect_sessions(data_dir, exclude=exclude)
    rows = [(s, session_frame_count(s)) for s in sessions]
    rows.sort(key=lambda r: r[1], reverse=True)
    k = max(1, int(round(top * len(rows))))
    return rows[:k]


def make_subset_tree(top_sessions, dst: Path):
    """Build a <user>/<session> directory tree of symlinks for the selected sessions."""
    import shutil, os
    # OneDrive on Windows sometimes holds locks; rmtree may not fully succeed.
    if dst.exists():
        try:
            shutil.rmtree(dst)
        except OSError:
            pass
    dst.mkdir(parents=True, exist_ok=True)
    for sess_path, _ in top_sessions:
        user = sess_path.parent.name
        target_dir = dst / user
        target_dir.mkdir(parents=True, exist_ok=True)
        link = target_dir / sess_path.name
        if link.exists() or link.is_symlink():
            # Already staged; leave it in place.
            continue
        # On Windows symlinks need admin/dev-mode; fall back to copy.
        try:
            os.symlink(str(sess_path.resolve()), str(link))
        except (OSError, NotImplementedError):
            shutil.copytree(sess_path, link)
    return dst


# ---------------------------------------------------------------------------
# Reservoir runs
# ---------------------------------------------------------------------------

def build_reservoir(kind, *, n_inputs, n_neurons, graph_dir, edge_attr,
                    spectral_radius, leak_range, seed):
    if kind == "connectome":
        from reservoirs.brain_connectome_reservoir import ConnectomeReservoir
        return ConnectomeReservoir(
            n_inputs=n_inputs, graph_dir=str(graph_dir),
            n_neurons=n_neurons, spectral_radius=spectral_radius,
            leak_range=leak_range, edge_attr=edge_attr, combine="mean",
            symmetric=True, seed=seed, input_scale=1.0,
        )
    if kind == "erdos_renyi":
        from reservoirs.ErdosRenyi import ErdosRenyiReservoir
        return ErdosRenyiReservoir(
            n_inputs=n_inputs, n_neurons=n_neurons, rhow=spectral_radius,
            leak_range=leak_range, p=0.1, seed=seed,
        )
    raise ValueError(f"unknown kind: {kind}")


def run_one(kind, n_neurons, *, Xt, yt, Xv, yv, graph_dir, edge_attr,
            spectral_radius=1.05, leak_range=(0.1, 0.3), washout=50,
            ridge=0.01, seed=42):
    print(f"\n=== {kind:<14s} N={n_neurons} ===", flush=True)
    t = time.time()
    res = build_reservoir(kind, n_inputs=Xt.shape[1], n_neurons=n_neurons,
                          graph_dir=graph_dir, edge_attr=edge_attr,
                          spectral_radius=spectral_radius, leak_range=leak_range,
                          seed=seed)
    print(f"  build  ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    Xs_tr = res.forward(Xt, collect_states=True)
    print(f"  forward(train)  {Xs_tr.shape}  ({time.time()-t:.1f}s)", flush=True)
    t = time.time()
    Xs_te = res.forward(Xv, collect_states=True)
    print(f"  forward(test)   {Xs_te.shape}  ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    Xw, yw = Xs_tr[washout:], yt[washout:]
    R = Xw.T @ Xw
    R[np.diag_indices_from(R)] += ridge
    wout = np.linalg.solve(R, Xw.T @ yw)
    yp_tr = Xs_tr[washout:] @ wout
    yp_te = Xs_te[washout:] @ wout
    yt_te = yv[washout:]
    print(f"  ridge + predict  ({time.time()-t:.1f}s)", flush=True)

    from utilities.metrics import compute_all_metrics
    m_tr = compute_all_metrics(yw, yp_tr)
    m_te = compute_all_metrics(yt_te, yp_te)
    print(f"  TRAIN  ang={m_tr['accuracy']:.4f}  top3={m_tr['top3_accuracy']:.4f}  "
          f"err={m_tr['angular_error_deg']:.2f}°  boost={m_tr['boost_accuracy']:.4f}",
          flush=True)
    print(f"  TEST   ang={m_te['accuracy']:.4f}  top3={m_te['top3_accuracy']:.4f}  "
          f"err={m_te['angular_error_deg']:.2f}°  boost={m_te['boost_accuracy']:.4f}",
          flush=True)
    return {
        "reservoir": kind,
        "n_neurons": int(n_neurons),
        "edge_attr": edge_attr if kind == "connectome" else None,
        "spectral_radius": spectral_radius,
        "leak_range": list(leak_range),
        "washout": washout,
        "ridge": ridge,
        "seed": seed,
        "n_train_frames": int(Xt.shape[0]),
        "n_test_frames": int(Xv.shape[0]),
        "train": m_tr,
        "test": m_te,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(rows, out_path):
    rows = sorted(rows, key=lambda r: (r["reservoir"], r["n_neurons"]))
    labels = [f"{r['reservoir'].replace('erdos_renyi','ER').replace('connectome','Connectome')} "
              f"N={r['n_neurons']}" for r in rows]
    ang = [r["test"]["accuracy"] for r in rows]
    top3 = [r["test"]["top3_accuracy"] for r in rows]
    err = [r["test"]["angular_error_deg"] for r in rows]
    bst = [r["test"]["boost_accuracy"] for r in rows]
    colors = ["#888" if r["reservoir"] == "erdos_renyi" else "#48f" for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    specs = [
        ("Angle accuracy (test)", ang, (0, 1), "%.3f"),
        ("Top-3 accuracy (test)", top3, (0, 1), "%.3f"),
        ("Angular error (test, °)", err, (0, max(err) * 1.15 if err else 25), "%.2f"),
        ("Boost accuracy (test)", bst, (0, 1), "%.3f"),
    ]
    for ax, (title, vals, ylim, fmt) in zip(axes, specs):
        bars = ax.bar(labels, vals, color=colors, edgecolor="k")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.02,
                    fmt % v, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Slither.io — connectome vs Erdős-Rényi reservoir", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=float, default=0.30)
    p.add_argument("--exclude", nargs="*", default=["AI_bot"])
    p.add_argument("--graph-dir", default=str(REPO / "data" / "folder_path_234"))
    p.add_argument("--edge-attr", default="number_of_fibers")
    p.add_argument("--sizes", nargs="+", type=int, default=[300, 500, 1000])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir() or not any(data_dir.iterdir()):
        raise SystemExit(f"No data found under {data_dir}")

    print(f"Selecting top {args.top:.0%} sessions from {data_dir} "
          f"(exclude={args.exclude}) ...", flush=True)
    top = select_top_sessions(data_dir, args.top, exclude=set(args.exclude))
    total_frames = sum(n for _, n in top)
    print(f"  {len(top)} sessions, {total_frames:,} frames", flush=True)

    # Stage to a flat tree so the loader picks them up cleanly.
    subset = REPO / "scripts" / "results" / "_top_subset_data"
    print(f"Staging subset tree -> {subset}", flush=True)
    make_subset_tree(top, subset)

    # Hot-patch SLITHER_DATA_PATH for the loader.
    import vnicktest.scripts.configuration as cfg
    cfg.SLITHER_DATA_PATH = subset

    print("Loading data...", flush=True)
    t = time.time()
    from utilities.data_loader import load_all_data, train_test_split
    X_list, y_list, names, users = load_all_data(subset, verbose=False)
    Xt, yt, Xv, yv, _ = train_test_split(
        X_list, y_list, names, users,
        test_size=0.2, random_seed=args.seed, use_chunks=True,
    )
    print(f"  {len(X_list)} sessions, {sum(len(x) for x in X_list):,} frames -> "
          f"train {Xt.shape}, test {Xv.shape}  ({time.time()-t:.1f}s)", flush=True)

    rows = []
    for kind in ("connectome", "erdos_renyi"):
        for n in args.sizes:
            try:
                row = run_one(
                    kind, n,
                    Xt=Xt, yt=yt, Xv=Xv, yv=yv,
                    graph_dir=args.graph_dir, edge_attr=args.edge_attr,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"FAILED {kind} N={n}: {e}", flush=True)
                continue
            rows.append(row)
            (RES_DIR / f"slither_{kind}_n{n}.json").write_text(
                json.dumps(row, indent=2, default=float))

    # Summary CSV
    csv_path = RES_DIR / "slither_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reservoir", "N", "test_accuracy", "test_top3", "test_ang_err_deg",
                    "test_boost_acc", "train_accuracy", "train_ang_err_deg",
                    "n_train_frames", "n_test_frames"])
        for r in rows:
            w.writerow([
                r["reservoir"], r["n_neurons"],
                r["test"]["accuracy"], r["test"]["top3_accuracy"],
                r["test"]["angular_error_deg"], r["test"]["boost_accuracy"],
                r["train"]["accuracy"], r["train"]["angular_error_deg"],
                r["n_train_frames"], r["n_test_frames"],
            ])
    print(f"\nSummary CSV -> {csv_path}", flush=True)

    fig_path = FIG_DIR / "slither_top30_comparison.png"
    plot_summary(rows, fig_path)
    print(f"Comparison figure -> {fig_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
