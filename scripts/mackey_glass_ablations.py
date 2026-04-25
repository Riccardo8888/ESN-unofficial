#!/usr/bin/env python3
"""
Connectome-reservoir ablations on Mackey-Glass.

Runs four families of experiments and saves both JSON results and PNG plots:

  1. Edge-attribute ablation     -> edge_attr in {unweighted, FA_mean, number_of_fibers, fiber_length_mean}
  2. Combine-method ablation     -> combine in {first, mean, median} x #subjects in {1, 5, 50}
  3. Parcellation-N sweep        -> N in {83, 129, 234, 463, 1015}
  4. Multi-seed reproducibility  -> 5 seeds x 4 reservoir types

The HP setup matches the tuned Section VIII config:
    rho_W = 0.7, leak = (0.2, 0.7), washout = 200, ridge = 1e-6.

Outputs are written to scripts/figures/ and scripts/results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mackey_glass_benchmark import (
    benchmark_one,
    mackey_glass,
)

DATA_ROOT = REPO_ROOT / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
RES_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Common HP setup (tuned Section VIII config)
HP = dict(
    spectral_radius=0.7,
    leak_range=(0.2, 0.7),
    washout=200,
    ridge=1e-6,
    train_len=1500,
    test_len=2000,
    p_er=0.1,
)


def _make_subject_subset(src_dir: Path, k: int, dst_dir: Path) -> Path:
    """Copy the first k .graphml files of src_dir into dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.graphml"))[:k]
    if not files:
        raise FileNotFoundError(f"No graphml files in {src_dir}")
    for f in files:
        target = dst_dir / f.name
        if not target.exists():
            shutil.copy(f, target)
    return dst_dir


# ---------------------------------------------------------------------------
# 1. Edge-attribute ablation
# ---------------------------------------------------------------------------

def ablation_edge_attr(series, n_neurons=234, seed=42) -> List[dict]:
    print("\n" + "=" * 60)
    print("ABLATION 1: edge attribute")
    print("=" * 60)

    # Single brain (subject 0) at N=234
    src = DATA_ROOT / "folder_path_234"
    single = _make_subject_subset(src, 1, Path("/tmp/ablation_n234_k1"))

    rows = []
    # We pass the attribute name to ConnectomeReservoir via benchmark_one ->
    # build_reservoir, but build_reservoir doesn't expose `edge_attr` directly.
    # We use a tiny patch: reach in via monkey-patching the call path.
    from reservoirs.brain_connectome_reservoir import ConnectomeReservoir
    import scripts.mackey_glass_benchmark as mg
    original = mg.build_reservoir

    def patched(kind, *, n_inputs, n_neurons, spectral_radius, leak_range, seed,
                graph_dir=None, p_er=0.1):
        return ConnectomeReservoir(
            n_inputs=n_inputs, graph_dir=str(graph_dir),
            n_neurons=n_neurons, spectral_radius=spectral_radius,
            leak_range=leak_range, edge_attr=patched.edge_attr,
            combine="first", symmetric=True, seed=seed,
        )

    for attr_label, attr_value in [
        ("unweighted", None),
        ("FA_mean", "FA_mean"),
        ("number_of_fibers", "number_of_fibers"),
        ("fiber_length_mean", "fiber_length_mean"),
    ]:
        patched.edge_attr = attr_value
        mg.build_reservoir = patched
        r = benchmark_one(
            "connectome", series,
            n_neurons=n_neurons,
            spectral_radius=HP["spectral_radius"],
            leak_range=HP["leak_range"],
            washout=HP["washout"], ridge=HP["ridge"],
            train_len=HP["train_len"], test_len=HP["test_len"],
            seed=seed, graph_dir=single, p_er=HP["p_er"],
        )
        r["edge_attr"] = attr_label
        rows.append(r)
        print(f"  edge_attr={attr_label:22s}  MC={r['MC']:.4f}  R2={r['R2']:.4f}")

    mg.build_reservoir = original
    return rows


# ---------------------------------------------------------------------------
# 2. Combine-method ablation
# ---------------------------------------------------------------------------

def ablation_combine(series, n_neurons=234, seed=42) -> List[dict]:
    print("\n" + "=" * 60)
    print("ABLATION 2: combine method x #subjects")
    print("=" * 60)

    src = DATA_ROOT / "folder_path_234"
    rows = []

    from reservoirs.brain_connectome_reservoir import ConnectomeReservoir
    import scripts.mackey_glass_benchmark as mg
    original = mg.build_reservoir

    def patched(kind, *, n_inputs, n_neurons, spectral_radius, leak_range,
                seed, graph_dir=None, p_er=0.1):
        return ConnectomeReservoir(
            n_inputs=n_inputs, graph_dir=str(graph_dir),
            n_neurons=n_neurons, spectral_radius=spectral_radius,
            leak_range=leak_range, edge_attr=patched.edge_attr,
            combine=patched.combine, symmetric=True, seed=seed,
        )
    patched.edge_attr = "number_of_fibers"

    for k in (1, 5, 50):
        subset_dir = _make_subject_subset(src, k, Path(f"/tmp/ablation_n234_k{k}"))
        for combine in ("first", "mean", "median"):
            if k == 1 and combine != "first":
                continue
            patched.combine = combine
            mg.build_reservoir = patched
            r = benchmark_one(
                "connectome", series,
                n_neurons=n_neurons,
                spectral_radius=HP["spectral_radius"],
                leak_range=HP["leak_range"],
                washout=HP["washout"], ridge=HP["ridge"],
                train_len=HP["train_len"], test_len=HP["test_len"],
                seed=seed, graph_dir=subset_dir, p_er=HP["p_er"],
            )
            r["combine"] = combine
            r["n_subjects"] = k
            rows.append(r)
            print(f"  k={k:>3d}  combine={combine:<8s}  MC={r['MC']:.4f}  R2={r['R2']:.4f}")

    mg.build_reservoir = original
    return rows


# ---------------------------------------------------------------------------
# 3. Parcellation N sweep
# ---------------------------------------------------------------------------

def ablation_n_sweep(series, seed=42) -> List[dict]:
    print("\n" + "=" * 60)
    print("ABLATION 3: parcellation N sweep (single brain, number_of_fibers)")
    print("=" * 60)
    rows = []
    from reservoirs.brain_connectome_reservoir import ConnectomeReservoir
    import scripts.mackey_glass_benchmark as mg
    original = mg.build_reservoir

    def patched(kind, *, n_inputs, n_neurons, spectral_radius, leak_range,
                seed, graph_dir=None, p_er=0.1):
        return ConnectomeReservoir(
            n_inputs=n_inputs, graph_dir=str(graph_dir),
            n_neurons=n_neurons, spectral_radius=spectral_radius,
            leak_range=leak_range, edge_attr="number_of_fibers",
            combine="first", symmetric=True, seed=seed,
        )

    for N in (83, 129, 234, 463, 1015):
        src = DATA_ROOT / f"folder_path_{N}"
        if not src.is_dir():
            print(f"  N={N:<5d}  SKIPPED (folder missing)")
            continue
        single = _make_subject_subset(src, 1, Path(f"/tmp/ablation_n{N}_k1"))
        mg.build_reservoir = patched
        r = benchmark_one(
            "connectome", series,
            n_neurons=N,
            spectral_radius=HP["spectral_radius"],
            leak_range=HP["leak_range"],
            washout=HP["washout"], ridge=HP["ridge"],
            train_len=HP["train_len"], test_len=HP["test_len"],
            seed=seed, graph_dir=single, p_er=HP["p_er"],
        )
        r["N"] = N
        rows.append(r)
        print(f"  N={N:<5d}  MC={r['MC']:.4f}  R2={r['R2']:.4f}")

    mg.build_reservoir = original
    return rows


# ---------------------------------------------------------------------------
# 4. Multi-seed reproducibility
# ---------------------------------------------------------------------------

def ablation_multiseed(series, seeds=(42, 7, 11, 23, 99), n_neurons=234) -> List[dict]:
    print("\n" + "=" * 60)
    print(f"ABLATION 4: multi-seed reproducibility ({len(seeds)} seeds)")
    print("=" * 60)
    src = DATA_ROOT / "folder_path_234"
    single = _make_subject_subset(src, 1, Path("/tmp/ablation_n234_k1"))

    rows = []
    for kind in ("erdos_renyi", "fully_connected", "gaussian", "connectome"):
        for s in seeds:
            r = benchmark_one(
                kind, series,
                n_neurons=n_neurons,
                spectral_radius=HP["spectral_radius"],
                leak_range=HP["leak_range"],
                washout=HP["washout"], ridge=HP["ridge"],
                train_len=HP["train_len"], test_len=HP["test_len"],
                seed=s, graph_dir=single, p_er=HP["p_er"],
            )
            r["seed"] = s
            rows.append(r)
        # Aggregate
        these = [x["MC"] for x in rows if x["reservoir"] == kind]
        print(f"  {kind:<18s}  MC mean={np.mean(these):.4f}  std={np.std(these):.4f}  "
              f"({len(these)} seeds)")
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_edge_attr(rows):
    labels = [r["edge_attr"] for r in rows]
    mcs = [r["MC"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(labels, mcs, color=["#888", "#4c8", "#f80", "#48f"], edgecolor="k")
    ax.set_ylabel("Memory Capacity")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Connectome reservoir on Mackey-Glass — edge attribute")
    for bar, mc in zip(bars, mcs):
        ax.text(bar.get_x() + bar.get_width() / 2, mc + 0.01,
                f"{mc:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="k", linewidth=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "ablation_edge_attr.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_combine(rows):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ks = sorted({r["n_subjects"] for r in rows})
    combines = ["first", "mean", "median"]
    width = 0.25
    x = np.arange(len(ks))
    for i, c in enumerate(combines):
        ys = []
        for k in ks:
            cand = [r["MC"] for r in rows if r["combine"] == c and r["n_subjects"] == k]
            ys.append(cand[0] if cand else np.nan)
        ax.bar(x + (i - 1) * width, ys, width, label=c, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("Memory Capacity")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Connectome reservoir — combine method × #subjects")
    ax.legend(title="combine")
    plt.tight_layout()
    out = FIG_DIR / "ablation_combine.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_n_sweep(rows):
    Ns = [r["N"] for r in rows]
    mcs = [r["MC"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(Ns, mcs, marker="o", color="#48f", linewidth=2)
    for n, mc in zip(Ns, mcs):
        ax.annotate(f"{mc:.3f}", (n, mc), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Parcellation size N")
    ax.set_ylabel("Memory Capacity")
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_title("Connectome reservoir on Mackey-Glass — parcellation N")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "ablation_n_sweep.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_multiseed(rows):
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["reservoir"], []).append(r["MC"])

    kinds = ["erdos_renyi", "fully_connected", "gaussian", "connectome"]
    means = [np.mean(by_kind[k]) for k in kinds]
    stds = [np.std(by_kind[k]) for k in kinds]
    colors = ["#888", "#4c8", "#f80", "#48f"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(kinds, means, yerr=stds, capsize=5, color=colors, edgecolor="k")
    ax.set_ylabel("Memory Capacity")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Reservoir comparison on Mackey-Glass — 5 seeds")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.01,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "ablation_multiseed.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ablations", default="all",
                   help="Comma-separated subset of: edge_attr, combine, n_sweep, multiseed, all")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    n_steps = HP["train_len"] + HP["test_len"] + HP["washout"] + 1
    print(f"Generating Mackey-Glass series ({n_steps} steps, seed={args.seed})...")
    series = mackey_glass(n_steps=n_steps, seed=args.seed)

    requested = set(args.ablations.split(",")) if args.ablations != "all" else {
        "edge_attr", "combine", "n_sweep", "multiseed"}

    all_results = {}
    figs = []
    t0 = time.time()

    if "edge_attr" in requested:
        rows = ablation_edge_attr(series, seed=args.seed)
        all_results["edge_attr"] = rows
        figs.append(plot_edge_attr(rows))

    if "combine" in requested:
        rows = ablation_combine(series, seed=args.seed)
        all_results["combine"] = rows
        figs.append(plot_combine(rows))

    if "n_sweep" in requested:
        rows = ablation_n_sweep(series, seed=args.seed)
        all_results["n_sweep"] = rows
        figs.append(plot_n_sweep(rows))

    if "multiseed" in requested:
        rows = ablation_multiseed(series)
        all_results["multiseed"] = rows
        figs.append(plot_multiseed(rows))

    elapsed = time.time() - t0

    out_json = RES_DIR / "mackey_glass_ablations.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))

    print()
    print("=" * 60)
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results JSON: {out_json}")
    print("Figures:")
    for f in figs:
        print(f"  {f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
