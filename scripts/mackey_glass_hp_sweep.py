#!/usr/bin/env python3
"""
Hyperparameter sweep for the connectome reservoir on the Mackey-Glass task.

Wraps `scripts/mackey_glass_benchmark.benchmark_one` with a simple grid over
spectral_radius, leak_range, washout, ridge and reports the best (MC, R^2)
configuration.  Also prints the matched single-config baselines for the
three reference reservoirs at the SAME training/test split, so the table
that ends up in Section VIII is internally consistent.

Usage
-----
    python scripts/mackey_glass_hp_sweep.py --graph-dir data/folder_path_234
    python scripts/mackey_glass_hp_sweep.py --graph-dir <single-brain-folder> \
                                            --output mackey_glass_results_n234.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.mackey_glass_benchmark import benchmark_one, mackey_glass


GRID = {
    "spectral_radius": [0.7, 0.9, 1.0, 1.1, 1.25, 1.4],
    "leak_lo":         [0.05, 0.1, 0.2],
    "leak_hi":         [0.3, 0.5, 0.7],
    "washout":         [100, 200, 400],
    "ridge":           [1e-8, 1e-6, 1e-4, 1e-2],
}


def all_combinations():
    keys = list(GRID.keys())
    for vals in product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, vals))
        if cfg["leak_hi"] <= cfg["leak_lo"]:
            continue
        yield cfg


def main() -> int:
    p = argparse.ArgumentParser(description="Mackey-Glass HP sweep for connectome reservoir.")
    p.add_argument("--graph-dir", type=str, required=True,
                   help="GraphML folder for the connectome reservoir.")
    p.add_argument("--n-neurons", type=int, default=234)
    p.add_argument("--train-len", type=int, default=1500)
    p.add_argument("--test-len", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max-time-s", type=float, default=120.0,
                   help="Stop the sweep once this many seconds have elapsed.")
    args = p.parse_args()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.is_dir():
        raise SystemExit(f"--graph-dir not found: {graph_dir}")

    n_steps = args.train_len + args.test_len + max(GRID["washout"]) + 1
    print(f"Generating MG series with {n_steps} steps (seed={args.seed})...")
    series = mackey_glass(n_steps=n_steps, seed=args.seed)

    print(f"\nSweeping {sum(1 for _ in all_combinations())} configurations on the connectome reservoir "
          f"(N={args.n_neurons}, graph_dir={graph_dir})\n")

    best = None
    rows = []
    t_start = time.time()
    for i, cfg in enumerate(all_combinations(), 1):
        if time.time() - t_start > args.max_time_s:
            print(f"  -- max-time-s={args.max_time_s} reached after {i-1} runs, stopping.")
            break
        try:
            r = benchmark_one(
                "connectome", series,
                n_neurons=args.n_neurons,
                spectral_radius=cfg["spectral_radius"],
                leak_range=(cfg["leak_lo"], cfg["leak_hi"]),
                washout=cfg["washout"],
                ridge=cfg["ridge"],
                train_len=args.train_len,
                test_len=args.test_len,
                seed=args.seed,
                graph_dir=graph_dir,
                p_er=0.1,
            )
        except Exception as e:
            print(f"  [{i:3d}] FAILED  {cfg}  -- {e}")
            continue
        row = {**cfg, **r}
        rows.append(row)
        if best is None or r["MC"] > best["MC"]:
            best = row
            print(f"  [{i:3d}]  rho={cfg['spectral_radius']:.2f}  leak=[{cfg['leak_lo']},{cfg['leak_hi']}]"
                  f"  wash={cfg['washout']}  ridge={cfg['ridge']:.0e}  ->  MC={r['MC']:.4f}  R2={r['R2']:.4f}  *NEW BEST*")

    print()
    print("=" * 70)
    print("BEST CONNECTOME CONFIG ON MG")
    print("=" * 70)
    if best is None:
        print("  (no successful runs)")
        return 1
    print(f"  spectral_radius : {best['spectral_radius']}")
    print(f"  leak_range      : ({best['leak_lo']}, {best['leak_hi']})")
    print(f"  washout         : {best['washout']}")
    print(f"  ridge           : {best['ridge']}")
    print(f"  ----")
    print(f"  MC              : {best['MC']:.4f}")
    print(f"  R^2             : {best['R2']:.4f}")
    print(f"  NRMSE           : {best['NRMSE']:.4f}")

    # Match reference reservoirs at the SAME HP setting, to keep the
    # comparison internally consistent.
    print()
    print("=" * 70)
    print("REFERENCE RESERVOIRS AT THE SAME HP SETTING")
    print("=" * 70)
    ref_rows = []
    for kind in ("erdos_renyi", "fully_connected", "gaussian"):
        try:
            rr = benchmark_one(
                kind, series,
                n_neurons=args.n_neurons,
                spectral_radius=best["spectral_radius"],
                leak_range=(best["leak_lo"], best["leak_hi"]),
                washout=best["washout"],
                ridge=best["ridge"],
                train_len=args.train_len,
                test_len=args.test_len,
                seed=args.seed,
                graph_dir=graph_dir,
                p_er=0.1,
            )
            ref_rows.append({"reservoir": kind, **rr})
            print(f"  {kind:<18s} MC={rr['MC']:.4f}  R^2={rr['R2']:.4f}  NRMSE={rr['NRMSE']:.4f}")
        except Exception as e:
            print(f"  {kind:<18s} FAILED: {e}")

    print()
    print("=" * 70)
    print("FINAL SECTION VIII TABLE  (one row per reservoir)")
    print("=" * 70)
    final_rows = ref_rows + [{"reservoir": "connectome", **best}]
    print(f"{'reservoir':<18s} {'MC':>7s} {'R^2':>7s} {'NRMSE':>8s}")
    print("-" * 50)
    for r in final_rows:
        print(f"{r['reservoir']:<18s} {r['MC']:>7.4f} {r['R2']:>7.4f} {r['NRMSE']:>8.4f}")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "graph_dir": str(graph_dir),
            "n_neurons": args.n_neurons,
            "train_len": args.train_len,
            "test_len": args.test_len,
            "seed": args.seed,
            "best_config": best,
            "reference_at_best": ref_rows,
            "all_runs": rows,
        }, indent=2, default=str))
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
