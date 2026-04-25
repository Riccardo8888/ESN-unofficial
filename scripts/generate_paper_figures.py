#!/usr/bin/env python3
"""
Generate the figures and tables that the paper imports.

Reads pre-existing JSON / CSV result files (no experiments are re-run here)
and produces:

  scripts/figures/slither_train_test_overfit.png
  scripts/figures/slither_metric_diff.png
  scripts/figures/mackey_glass_summary_panel.png
  scripts/figures/mackey_glass_n_sweep_compare.png
  scripts/tables/mg_main.tex
  scripts/tables/mg_edge_attr.tex
  scripts/tables/mg_combine.tex
  scripts/tables/mg_n_sweep.tex
  scripts/tables/mg_multiseed.tex
  scripts/tables/slither_sweep.tex

Run after either of:
    python scripts/mackey_glass_benchmark.py        # populates root JSONs
    python scripts/mackey_glass_hp_sweep.py
    python scripts/mackey_glass_ablations.py
    python scripts/run_slither_connectome_sweep.py

Then:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RES_DIR = REPO / "scripts" / "results"
FIG_DIR = REPO / "scripts" / "figures"
TBL_DIR = REPO / "scripts" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# Common colour palette: grey for ER, orange for FC, green for Gaussian, blue for connectome.
COLORS = {
    "erdos_renyi": "#888888",
    "ER": "#888888",
    "fully_connected": "#f08c00",
    "FC": "#f08c00",
    "gaussian": "#3aa648",
    "GR": "#3aa648",
    "connectome": "#1f77ff",
    "Connectome": "#1f77ff",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _label(reservoir: str) -> str:
    return {
        "erdos_renyi": "ER",
        "fully_connected": "FC",
        "gaussian": "Gaussian",
        "connectome": "Connectome",
    }.get(reservoir, reservoir)


def _save_table(path: Path, header: list[str], rows: list[list[str]],
                caption: str = "", label: str = "", colspec: str | None = None):
    if colspec is None:
        colspec = "l" + "r" * (len(header) - 1)
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("    \\centering")
    if caption:
        lines.append(f"    \\caption{{{caption}}}")
    if label:
        lines.append(f"    \\label{{{label}}}")
    lines.append(f"    \\begin{{tabular}}{{{colspec}}}")
    lines.append("        \\hline")
    lines.append("        " + " & ".join(header) + " \\\\")
    lines.append("        \\hline")
    for r in rows:
        lines.append("        " + " & ".join(r) + " \\\\")
    lines.append("        \\hline")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Mackey-Glass artifacts
# ---------------------------------------------------------------------------

def make_mg_main_table_and_panel():
    """4-reservoir headline at tuned HPs (from mackey_glass_results_n234_tuned.json)."""
    p = REPO / "mackey_glass_results_n234_tuned.json"
    data = _load_json(p)
    if not data:
        print(f"  skip MG main: {p.name} not found")
        return
    best = data["best_config"]
    refs = data["reference_at_best"]
    rows_by_kind = {r["reservoir"]: r for r in refs}
    rows_by_kind["connectome"] = best

    order = ["erdos_renyi", "fully_connected", "gaussian", "connectome"]
    mc = [rows_by_kind[k]["MC"] for k in order]
    r2 = [rows_by_kind[k]["R2"] for k in order]
    nrmse = [rows_by_kind[k]["NRMSE"] for k in order]
    times = [rows_by_kind[k].get("fit_time_s", 0.0) for k in order]

    # LaTeX table
    rows = [
        [_label(k),
         f"{rows_by_kind[k]['MC']:.4f}",
         f"{rows_by_kind[k]['R2']:.4f}",
         f"{rows_by_kind[k]['NRMSE']:.4f}",
         f"{rows_by_kind[k].get('fit_time_s', 0.0):.2f}"]
        for k in order
    ]
    _save_table(
        TBL_DIR / "mg_main.tex",
        ["Reservoir", "MC", "$R^2$", "NRMSE", "fit time (s)"],
        rows,
        caption=("Mackey--Glass one-step prediction at tuned hyper-parameters "
                 f"($\\rho_W{{=}}{best['spectral_radius']}$, leak "
                 f"$[{best['leak_lo']},{best['leak_hi']}]$, washout "
                 f"${best['washout']}$, ridge $\\alpha{{=}}{best['ridge']:.0e}$, "
                 "$N{=}234$, single-brain connectome)."),
        label="tab:mg_main",
    )

    # Four-panel summary figure
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    labels = [_label(k) for k in order]
    bar_colors = [COLORS[k] for k in order]
    for ax, vals, title, ylim, fmt in zip(
        axes,
        [mc, r2, nrmse, times],
        ["Memory Capacity", "$R^2$", "NRMSE", "fit time (s, log)"],
        [(0.5, 1.05), (0.95, 1.005), (0, max(nrmse) * 1.5 if max(nrmse) > 0 else 0.1), None],
        ["%.4f", "%.4f", "%.4f", "%.1fs"],
    ):
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="k")
        ax.set_title(title, fontsize=10)
        if title == "fit time (s, log)":
            ax.set_yscale("log")
        else:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v * 1.02 if title == "fit time (s, log)"
                    else v + (ylim[1] - ylim[0]) * 0.02,
                    fmt % v, ha="center", va="bottom", fontsize=8)
    plt.suptitle("Mackey-Glass at tuned HPs (single-brain 234-node connectome)", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mackey_glass_summary_panel.png", dpi=150)
    plt.close(fig)
    print(f"  -> {FIG_DIR / 'mackey_glass_summary_panel.png'}")
    print(f"  -> {TBL_DIR / 'mg_main.tex'}")


def make_mg_ablation_tables():
    p = RES_DIR / "mackey_glass_ablations.json"
    data = _load_json(p)
    if not data:
        print(f"  skip MG ablations: {p.name} not found")
        return

    if "edge_attr" in data:
        rows = [[r["edge_attr"].replace("_", "\\_"),
                 f"{r['MC']:.4f}", f"{r['R2']:.4f}", f"{r['NRMSE']:.4f}"]
                for r in data["edge_attr"]]
        _save_table(
            TBL_DIR / "mg_edge_attr.tex",
            ["Edge attribute", "MC", "$R^2$", "NRMSE"], rows,
            caption=("Effect of the edge weighting on the connectome reservoir "
                     "(single-brain, $N{=}234$, tuned HPs)."),
            label="tab:mg_edge_attr",
        )
        print(f"  -> {TBL_DIR / 'mg_edge_attr.tex'}")

    if "combine" in data:
        rows = [[str(r["n_subjects"]), r["combine"],
                 f"{r['MC']:.4f}", f"{r['R2']:.4f}", f"{r['NRMSE']:.4f}"]
                for r in data["combine"]]
        _save_table(
            TBL_DIR / "mg_combine.tex",
            ["\\# subjects", "Combine", "MC", "$R^2$", "NRMSE"], rows,
            caption=("Effect of the multi-subject aggregation method on the connectome "
                     "reservoir ($N{=}234$, edge\\_attr=number\\_of\\_fibers)."),
            label="tab:mg_combine",
        )
        print(f"  -> {TBL_DIR / 'mg_combine.tex'}")

    if "n_sweep" in data:
        rows = [[str(r["N"]), f"{r['MC']:.4f}", f"{r['R2']:.4f}", f"{r['NRMSE']:.4f}"]
                for r in data["n_sweep"]]
        _save_table(
            TBL_DIR / "mg_n_sweep.tex",
            ["$N$", "MC", "$R^2$", "NRMSE"], rows,
            caption=("Connectome reservoir on Mackey--Glass across the five HCP "
                     "parcellations (single-brain, edge\\_attr=number\\_of\\_fibers)."),
            label="tab:mg_n_sweep",
        )
        print(f"  -> {TBL_DIR / 'mg_n_sweep.tex'}")

    if "multiseed" in data:
        from collections import defaultdict
        bucket = defaultdict(list)
        for r in data["multiseed"]:
            bucket[r["reservoir"]].append(r["MC"])
        rows = []
        for kind in ("erdos_renyi", "fully_connected", "gaussian", "connectome"):
            if kind not in bucket:
                continue
            mcs = bucket[kind]
            rows.append([_label(kind),
                         f"{np.mean(mcs):.4f}",
                         f"{np.std(mcs):.4f}",
                         f"{np.min(mcs):.4f}",
                         f"{np.max(mcs):.4f}",
                         str(len(mcs))])
        _save_table(
            TBL_DIR / "mg_multiseed.tex",
            ["Reservoir", "mean MC", "std", "min", "max", "$n$ seeds"], rows,
            caption=("Reproducibility check: Mackey--Glass MC across multiple "
                     "RNG seeds ($N{=}234$, tuned HPs, single-brain connectome)."),
            label="tab:mg_multiseed",
        )
        print(f"  -> {TBL_DIR / 'mg_multiseed.tex'}")


def make_mg_n_sweep_compare():
    """One curve per reservoir: MC vs N. Currently only connectome has an N sweep
    in scripts/results/; reference reservoirs are independent of N at native HPs."""
    p = RES_DIR / "mackey_glass_ablations.json"
    data = _load_json(p)
    if not data or "n_sweep" not in data:
        print("  skip MG n-sweep compare: ablation data missing")
        return
    Ns = [r["N"] for r in data["n_sweep"]]
    mc_connectome = [r["MC"] for r in data["n_sweep"]]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(Ns, mc_connectome, "o-", color=COLORS["connectome"], label="Connectome", lw=2)
    # Tuned-HP single point for the references (at N=234 where the table was produced)
    p_tuned = REPO / "mackey_glass_results_n234_tuned.json"
    tuned = _load_json(p_tuned)
    if tuned and "reference_at_best" in tuned:
        for r in tuned["reference_at_best"]:
            ax.scatter([234], [r["MC"]], color=COLORS[r["reservoir"]], marker="s",
                       s=80, edgecolor="k", label=_label(r["reservoir"]) + " (N=234)")
    for n, mc in zip(Ns, mc_connectome):
        ax.annotate(f"{mc:.3f}", (n, mc), textcoords="offset points",
                    xytext=(5, -12), fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("Parcellation / reservoir size $N$")
    ax.set_ylabel("Memory Capacity")
    ax.set_title("Mackey-Glass MC vs reservoir size")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "mackey_glass_n_sweep_compare.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Slither artifacts
# ---------------------------------------------------------------------------

def _read_slither_summary():
    p = RES_DIR / "slither_summary.csv"
    if not p.exists():
        return None
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("reservoir",) else v)
                         for k, v in r.items()})
    return rows


def make_slither_table():
    rows = _read_slither_summary()
    if not rows:
        print("  skip slither table: slither_summary.csv missing")
        return
    rows = sorted(rows, key=lambda r: (r["reservoir"], int(r["N"])))
    table_rows = []
    for r in rows:
        table_rows.append([
            _label(r["reservoir"]),
            f"{int(r['N'])}",
            f"{r['test_accuracy']:.4f}",
            f"{r['test_top3']:.4f}",
            f"{r['test_ang_err_deg']:.2f}",
            f"{r['test_boost_acc']:.4f}",
            f"{r['train_accuracy']:.4f}",
            f"{r['train_ang_err_deg']:.2f}",
        ])
    _save_table(
        TBL_DIR / "slither_sweep.tex",
        ["Reservoir", "$N$", "test ang.", "test top-3",
         "test err.~(\\textdegree)", "test boost",
         "train ang.", "train err.~(\\textdegree)"],
        table_rows,
        caption=("Slither.io online prediction sweep on the top-30\\% sessions by "
                 "frame count (AI\\_bot excluded). Connectome and Erd\\H{o}s--R\\'enyi "
                 "compared at three reservoir sizes under matched HPs ($\\rho_W{=}1.05$, "
                 "leak~$[0.1, 0.3]$, washout~50, ridge~$\\alpha{=}10^{-2}$)."),
        label="tab:slither_sweep",
    )
    print(f"  -> {TBL_DIR / 'slither_sweep.tex'}")


def make_slither_overfit_plot():
    rows = _read_slither_summary()
    if not rows:
        print("  skip slither overfit plot: slither_summary.csv missing")
        return
    rows.sort(key=lambda r: (r["reservoir"], int(r["N"])))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    Ns_unique = sorted({int(r["N"]) for r in rows})
    width = 0.16
    x = np.arange(len(Ns_unique))
    series = {("connectome", "train"): [], ("connectome", "test"): [],
              ("erdos_renyi", "train"): [], ("erdos_renyi", "test"): []}
    for n in Ns_unique:
        for kind in ("connectome", "erdos_renyi"):
            r = next((r for r in rows
                      if int(r["N"]) == n and r["reservoir"] == kind), None)
            if r is None:
                series[(kind, "train")].append(np.nan)
                series[(kind, "test")].append(np.nan)
            else:
                series[(kind, "train")].append(r["train_accuracy"])
                series[(kind, "test")].append(r["test_accuracy"])

    bars_specs = [
        (-1.5 * width, ("connectome", "train"), "Connectome (train)", COLORS["connectome"], 0.6),
        (-0.5 * width, ("connectome", "test"),  "Connectome (test)",  COLORS["connectome"], 1.0),
        ( 0.5 * width, ("erdos_renyi", "train"), "ER (train)",         COLORS["erdos_renyi"], 0.6),
        ( 1.5 * width, ("erdos_renyi", "test"),  "ER (test)",          COLORS["erdos_renyi"], 1.0),
    ]
    for off, key, label, color, alpha in bars_specs:
        ax.bar(x + off, series[key], width, label=label, color=color, alpha=alpha,
               edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$N{{=}}{n}$" for n in Ns_unique])
    ax.set_ylabel("Angle accuracy")
    ax.set_ylim(0, 0.8)
    ax.set_title("Slither.io: train vs test angle accuracy across $N$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    out = FIG_DIR / "slither_train_test_overfit.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


def make_slither_diff_plot():
    rows = _read_slither_summary()
    if not rows:
        print("  skip slither diff plot: slither_summary.csv missing")
        return
    Ns_unique = sorted({int(r["N"]) for r in rows})

    metrics = [
        ("test_accuracy", "Angle accuracy"),
        ("test_top3", "Top-3 accuracy"),
        ("test_ang_err_deg", "Angular error (°)\n(lower is better)"),
        ("test_boost_acc", "Boost accuracy"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.4), sharey=False)
    for ax, (key, title) in zip(axes, metrics):
        diffs = []
        for n in Ns_unique:
            r_c = next(r for r in rows
                       if int(r["N"]) == n and r["reservoir"] == "connectome")
            r_e = next(r for r in rows
                       if int(r["N"]) == n and r["reservoir"] == "erdos_renyi")
            diffs.append(r_c[key] - r_e[key])
        colors = ["#3aa648" if d > 0 else "#d33" for d in diffs]
        bars = ax.bar([f"$N{{=}}{n}$" for n in Ns_unique], diffs,
                      color=colors, edgecolor="k")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        for b, d in zip(bars, diffs):
            ax.text(b.get_x() + b.get_width() / 2,
                    d + (0.005 if d >= 0 else -0.005),
                    f"{d:+.4f}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8)
        ax.tick_params(axis="x")
    fig.suptitle("Slither.io: connectome − ER difference per metric "
                 "(green: connectome better; red: ER better)", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "slither_metric_diff.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Mackey-Glass artifacts:")
    make_mg_main_table_and_panel()
    make_mg_ablation_tables()
    make_mg_n_sweep_compare()
    print("\nSlither artifacts:")
    make_slither_table()
    make_slither_overfit_plot()
    make_slither_diff_plot()
    print("\nDone.")


if __name__ == "__main__":
    main()
