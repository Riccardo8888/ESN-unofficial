#!/usr/bin/env python3
"""
Train Slither.io ESN using the Brain-Connectome Reservoir.

Mirrors `train_slither_reservoir.py` (the random-reservoir baseline) but
substitutes the recurrent matrix with a real Human-Connectome-Project
adjacency loaded from a directory of GraphML files.  All other steps -- data
loading, train/test split, ridge regression, evaluation, per-user analysis,
and result logging -- are identical, so results are directly comparable to
the random-reservoir runs in TRAINING_RESULTS.txt.
"""

import sys
from pathlib import Path
import numpy as np
import json
import argparse
import warnings
from datetime import datetime

# Add parent directory to path so the vnicktest scripts can import
# `vnicktest.scripts.configuration`, `utilities`, and `reservoirs`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vnicktest.scripts.configuration import (
    SLITHER_DATA_PATH, OUTPUT_PATH, TEST_SPLIT, RANDOM_SEED,
    N_RESERVOIR, SPECTRAL_RADIUS, INPUT_SCALE, LEAK_RATE_MIN, LEAK_RATE_MAX,
    ALPHA, WASHOUT, print_config,
)
from utilities.data_loader import load_all_data, train_test_split
from utilities.metrics import (
    compute_all_metrics,
    print_metrics,
    compute_angle_classification_metrics,
    compute_boost_metrics,
    compare_metrics,
)
from reservoirs.brain_connectome_reservoir import ConnectomeReservoir

warnings.filterwarnings("ignore")


def compute_wout(X, Y, T_washout, alpha):
    """Ridge regression: Wout = (X^T X + alpha I)^{-1} X^T Y."""
    X_train = X[T_washout:]
    Y_train = Y[T_washout:]
    R = X_train.T @ X_train
    P = X_train.T @ Y_train
    return np.linalg.solve(R + alpha * np.eye(X_train.shape[1]), P)


def find_default_graph_dir() -> Path:
    """
    Pick a sensible default GraphML folder if the user did not specify one.
    Prefer the largest available connectome resolution under data/folder_path_*.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data"
    candidates = [
        data_root / "folder_path_1015",
        data_root / "folder_path_463",
        data_root / "folder_path_234",
        data_root / "folder_path_129",
        data_root / "folder_path_83",
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.graphml")):
            return c
    raise FileNotFoundError(
        f"No connectome .graphml folder found under {data_root}. "
        f"Pass --graph-dir explicitly."
    )


def main():
    parser = argparse.ArgumentParser(description="Train Slither.io ESN with the brain connectome reservoir.")
    parser.add_argument("--graph-dir", type=str, default=None,
                        help="Folder of .graphml connectome files. Defaults to the largest folder_path_* under data/.")
    parser.add_argument("--n-reservoir", type=int, default=N_RESERVOIR)
    parser.add_argument("--spectral-radius", type=float, default=SPECTRAL_RADIUS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--washout", type=int, default=WASHOUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print_config()
    graph_dir = Path(args.graph_dir) if args.graph_dir else find_default_graph_dir()
    print(f"\n[connectome] Using graph directory: {graph_dir}")

    # ----- Step 1: load data -----------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    X_list, y_list, session_names, usernames = load_all_data(SLITHER_DATA_PATH, verbose=True)
    if not X_list:
        print("\nERROR: no Slither sessions found.")
        return 1

    # ----- Step 2: train / test split --------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN/TEST SPLIT (random chunks per session)")
    print("=" * 60)
    X_train_cat, y_train_cat, X_test_cat, y_test_cat, test_user_indices = train_test_split(
        X_list, y_list, session_names, usernames,
        test_size=TEST_SPLIT,
        random_seed=args.seed,
        use_chunks=True,
    )
    print(f"  Train: X={X_train_cat.shape}  y={y_train_cat.shape}")
    print(f"  Test : X={X_test_cat.shape}  y={y_test_cat.shape}")

    # ----- Step 3: build the connectome reservoir --------------------------
    print("\n" + "=" * 60)
    print("STEP 3: BUILDING CONNECTOME RESERVOIR")
    print("=" * 60)
    reservoir = ConnectomeReservoir(
        n_inputs=X_train_cat.shape[1],
        graph_dir=str(graph_dir),
        n_neurons=args.n_reservoir,
        spectral_radius=args.spectral_radius,
        leak_range=(LEAK_RATE_MIN, LEAK_RATE_MAX),
        edge_attr="weight",
        combine="mean",
        symmetric=True,
        seed=args.seed,
        input_scale=INPUT_SCALE,
    )
    print(f"  Reservoir size      : {reservoir.n_neurons}")
    print(f"  Target spectral radius: {reservoir.target_spectral_radius}")
    print(f"  Leak rate range     : [{LEAK_RATE_MIN}, {LEAK_RATE_MAX}]")

    # ----- Step 4: collect training states ---------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: COLLECTING RESERVOIR STATES (TRAIN)")
    print("=" * 60)
    X_train_states = reservoir.forward(X_train_cat, collect_states=True)
    print(f"  States: {X_train_states.shape}")

    # ----- Step 5: fit readout ---------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: TRAINING OUTPUT WEIGHTS (ridge regression)")
    print("=" * 60)
    wout = compute_wout(X_train_states, y_train_cat,
                        T_washout=args.washout, alpha=args.alpha)
    print(f"  W_out: {wout.shape}")

    # ----- Step 6: evaluate ------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATING")
    print("=" * 60)
    y_train_pred = X_train_states @ wout
    y_train_true = y_train_cat
    y_train_pred = y_train_pred[args.washout:]
    y_train_true = y_train_true[args.washout:]
    train_metrics = compute_all_metrics(y_train_true, y_train_pred)
    print_metrics(train_metrics, "Training")

    X_test_states = reservoir.forward(X_test_cat, collect_states=True)
    y_test_pred = X_test_states @ wout
    y_test_true = y_test_cat
    y_test_pred = y_test_pred[args.washout:]
    y_test_true = y_test_true[args.washout:]
    test_metrics = compute_all_metrics(y_test_true, y_test_pred)
    print_metrics(test_metrics, "Test")

    compare_metrics(train_metrics, test_metrics)

    # ----- Step 6b: per-user analysis -------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6b: PER-USER ANALYSIS")
    print("=" * 60)
    test_user_indices_adj = test_user_indices[args.washout:]
    unique_users = {}
    for idx in range(len(X_list)):
        username = usernames[idx]
        unique_users.setdefault(username, []).append(idx)

    print(f"\n{'User':<25} {'Frames':<10} {'Boost Acc':<12} {'Angle Acc':<12} {'Angular Err':<12}")
    print("-" * 75)
    user_stats = {}
    for username, session_indices in unique_users.items():
        user_mask = np.isin(test_user_indices_adj, session_indices)
        if not user_mask.any():
            continue
        y_user_pred = y_test_pred[user_mask]
        y_user_true = y_test_true[user_mask]
        ang = compute_angle_classification_metrics(y_user_true, y_user_pred)
        bst = compute_boost_metrics(y_user_true, y_user_pred)
        user_stats[username] = {
            "n_frames": int(len(y_user_pred)),
            "boost_accuracy": float(bst["boost_accuracy"]),
            "angle_accuracy": float(ang["accuracy"]),
            "angular_error_deg": float(ang["angular_error_deg"]),
        }
        print(f"{username:<25} {len(y_user_pred):<10} "
              f"{bst['boost_accuracy']:>10.2%}  "
              f"{ang['accuracy']:>10.2%}  "
              f"{ang['angular_error_deg']:>10.2f}°")

    # ----- Step 7: save model + results ------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_PATH / f"connectome_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reservoir + W_out as a single npz so it can be reloaded later.
    np.savez(
        output_dir / "reservoir_model.npz",
        W=reservoir.W, Win=reservoir.Win, Win_bias=reservoir.Win_bias,
        leak=reservoir.leak, W_out=wout,
        n_inputs=reservoir.n_inputs, n_neurons=reservoir.n_neurons,
        spectral_radius=reservoir.target_spectral_radius,
        graph_dir=str(graph_dir),
    )

    results = {
        "timestamp": timestamp,
        "config": {
            "reservoir_type": "brain_connectome",
            "graph_dir": str(graph_dir),
            "n_reservoir": int(reservoir.n_neurons),
            "spectral_radius": float(reservoir.target_spectral_radius),
            "leak_rate_min": LEAK_RATE_MIN,
            "leak_rate_max": LEAK_RATE_MAX,
            "alpha": float(args.alpha),
            "washout": int(args.washout),
            "test_split": float(TEST_SPLIT),
            "random_seed": int(args.seed),
        },
        "training": train_metrics,
        "test": test_metrics,
        "per_user": user_stats,
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"\nSaved model and results to: {output_dir}")

    # ----- Step 8: append to TRAINING_RESULTS.txt for parity with the baseline
    log_file = Path(__file__).parent / "TRAINING_RESULTS.txt"
    if not log_file.exists():
        log_file.write_text("=" * 80 + "\nSLITHER.IO ESN TRAINING RESULTS LOG\n" + "=" * 80 + "\n")
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training: {timestamp}  (CONNECTOME RESERVOIR)\n")
        f.write(f"{'='*80}\n")
        f.write(f"Graph dir         : {graph_dir}\n")
        f.write(f"Reservoir size    : {reservoir.n_neurons}\n")
        f.write(f"Spectral radius   : {reservoir.target_spectral_radius}\n")
        f.write(f"Alpha             : {args.alpha}\n")
        f.write(f"Washout           : {args.washout} frames\n")
        f.write(f"Total frames      : {X_train_cat.shape[0] + X_test_cat.shape[0]}\n\n")
        f.write("Test metrics:\n")
        f.write(f"  Angle Accuracy : {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Top-3 Accuracy : {test_metrics['top3_accuracy']*100:.2f}%\n")
        f.write(f"  Angular Error  : {test_metrics['angular_error_deg']:.2f} deg\n")
        f.write(f"  Boost Accuracy : {test_metrics['boost_accuracy']*100:.2f}%\n")
        f.write(f"Output: {output_dir.name}\n")
        f.write("-" * 80 + "\n")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
