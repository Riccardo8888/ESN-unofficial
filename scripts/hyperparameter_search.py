#!/usr/bin/env python3
"""
Hyperparameter Search per Slither.io ESN

Questo script cerca automaticamente i migliori iperparametri per ridurre
l'overfitting e massimizzare le prestazioni sul test set.

Parametri ottimizzati:
- ALPHA (regolarizzazione ridge regression): 1e-4 to 1.0
- N_RESERVOIR (numero neuroni): 250 to 2000
- WASHOUT (periodo di stabilizzazione): 25 to 150
- SPECTRAL_RADIUS: 0.9 to 1.5

Usa cross-validation per valutazione robusta.

Autori: Nick & Riccardo
Data: 27 Ottobre 2025
"""

import sys
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime
import json
from itertools import product
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vnicktest.scripts.configuration import *
from utilities.data_loader import load_all_data
from utilities.metrics import compute_direction_metrics, compute_boost_metrics
from reservoir import Reservoir

warnings.filterwarnings('ignore')


def compute_wout(X, Y, alpha=1e-3):
    """Ridge regression to compute output weights."""
    R = X.T @ X
    P = X.T @ Y
    Wout = np.linalg.solve(R + alpha * np.eye(X.shape[1]), P)
    return Wout


def train_and_evaluate_single_fold(
    X_train, Y_train, X_val, Y_val,
    n_inputs, n_reservoir, spectral_radius, leak_range,
    alpha, washout, seed
):
    """
    Train ESN on train fold and evaluate on validation fold.
    
    X_train, Y_train are lists of sessions, each session is 2D [frames, features]
    
    Returns:
        metrics: dict with train and validation metrics
    """
    # Create reservoir
    reservoir = Reservoir(
        n_inputs=n_inputs,
        n_neurons=n_reservoir,
        rhow=spectral_radius,
        inp_scaling=1.0,
        leak_range=leak_range,
        verbose=False
    )
    
    # Collect training states from all sessions
    all_train_states = []
    all_train_targets = []
    
    for sess_idx, (X_session, Y_session) in enumerate(zip(X_train, Y_train)):
        # Run reservoir on this session
        X_states = reservoir.forward(X_session, collect_states=True)
        
        # Skip if session too short
        if len(X_states) <= washout:
            continue
            
        # Remove washout period
        X_states = X_states[washout:]
        Y_targets = Y_session[washout:]
        
        all_train_states.append(X_states)
        all_train_targets.append(Y_targets)
    
    if not all_train_states:
        raise ValueError("All training sessions too short!")
    
    X_train_states = np.vstack(all_train_states)
    Y_train_targets = np.vstack(all_train_targets)
    
    # Train output weights
    wout = compute_wout(X_train_states, Y_train_targets, alpha=alpha)
    
    # Evaluate on train fold - concatenate all predictions
    train_preds = []
    train_targets = []
    
    for X_session, Y_session in zip(X_train, Y_train):
        X_states = reservoir.forward(X_session, collect_states=True)
        if len(X_states) <= washout:
            continue
        X_states = X_states[washout:]
        Y_targets = Y_session[washout:]
        
        Y_pred = X_states @ wout
        train_preds.append(Y_pred)
        train_targets.append(Y_targets)
    
    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)
    
    # Ensure 2D shape
    if train_preds.ndim == 1:
        train_preds = train_preds.reshape(-1, 1)
    if train_targets.ndim == 1:
        train_targets = train_targets.reshape(-1, 1)
    
    # Evaluate on validation fold
    val_preds = []
    val_targets = []
    
    for X_session, Y_session in zip(X_val, Y_val):
        X_states = reservoir.forward(X_session, collect_states=True)
        if len(X_states) <= washout:
            continue
        X_states = X_states[washout:]
        Y_targets = Y_session[washout:]
        
        Y_pred = X_states @ wout
        val_preds.append(Y_pred)
        val_targets.append(Y_targets)
    
    if not val_preds:
        raise ValueError("All validation sessions too short!")
    
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    # Ensure 2D shape
    if val_preds.ndim == 1:
        val_preds = val_preds.reshape(-1, 1)
    if val_targets.ndim == 1:
        val_targets = val_targets.reshape(-1, 1)
    
    # Compute metrics
    # Ensure we have correct shape [n_samples, 3]
    if train_targets.shape[1] != 3 or train_preds.shape[1] != 3:
        raise ValueError(f"Expected shape [n, 3], got train_targets={train_targets.shape}, train_preds={train_preds.shape}")
    
    train_dir = compute_direction_metrics(train_targets[:, :2], train_preds[:, :2])
    train_boost = compute_boost_metrics(train_targets, train_preds)  # Pass full array!
    
    if val_targets.shape[1] != 3 or val_preds.shape[1] != 3:
        raise ValueError(f"Expected shape [n, 3], got val_targets={val_targets.shape}, val_preds={val_preds.shape}")
    
    val_dir = compute_direction_metrics(val_targets[:, :2], val_preds[:, :2])
    val_boost = compute_boost_metrics(val_targets, val_preds)  # Pass full array!
    
    return {
        'train': {'direction': train_dir, 'boost': train_boost},
        'val': {'direction': val_dir, 'boost': val_boost}
    }


def cross_validate(
    X_list, Y_list, n_inputs,
    n_reservoir, spectral_radius, leak_range,
    alpha, washout, seed, n_folds=3
):
    """
    Perform k-fold cross-validation on sessions.
    
    X_list, Y_list are lists of sessions (not flattened!)
    
    Returns:
        mean_metrics: averaged metrics across folds
    """
    n_sessions = len(X_list)
    fold_size = n_sessions // n_folds
    
    if fold_size == 0:
        fold_size = 1
        n_folds = n_sessions
    
    fold_metrics = []
    
    for fold in range(n_folds):
        # Split sessions
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_sessions
        
        val_indices = list(range(val_start, val_end))
        train_indices = [i for i in range(n_sessions) if i not in val_indices]
        
        if not train_indices or not val_indices:
            continue
        
        X_train = [X_list[i] for i in train_indices]
        Y_train = [Y_list[i] for i in train_indices]
        X_val = [X_list[i] for i in val_indices]
        Y_val = [Y_list[i] for i in val_indices]
        
        # Train and evaluate
        metrics = train_and_evaluate_single_fold(
            X_train, Y_train, X_val, Y_val,
            n_inputs, n_reservoir, spectral_radius, leak_range,
            alpha, washout, seed + fold  # Different seed per fold
        )
        
        fold_metrics.append(metrics)
    
    # Average metrics across folds
    mean_train_rmse = np.mean([m['train']['direction']['rmse_direction'] for m in fold_metrics])
    mean_train_angular = np.mean([m['train']['direction']['angular_error_deg'] for m in fold_metrics])
    mean_train_boost_acc = np.mean([m['train']['boost']['boost_accuracy'] for m in fold_metrics])
    
    mean_val_rmse = np.mean([m['val']['direction']['rmse_direction'] for m in fold_metrics])
    mean_val_angular = np.mean([m['val']['direction']['angular_error_deg'] for m in fold_metrics])
    mean_val_boost_acc = np.mean([m['val']['boost']['boost_accuracy'] for m in fold_metrics])
    
    # Overfitting metrics
    mse_ratio = (mean_val_rmse ** 2) / (mean_train_rmse ** 2) if mean_train_rmse > 0 else float('inf')
    acc_diff = mean_train_boost_acc - mean_val_boost_acc
    
    return {
        'train_rmse': mean_train_rmse,
        'train_angular': mean_train_angular,
        'train_boost_acc': mean_train_boost_acc,
        'val_rmse': mean_val_rmse,
        'val_angular': mean_val_angular,
        'val_boost_acc': mean_val_boost_acc,
        'mse_ratio': mse_ratio,
        'acc_diff': acc_diff,
        # Combined score for ranking (higher is better)
        'score': mean_val_boost_acc - 0.5 * acc_diff  # Maximize val accuracy, minimize overfitting
    }


def hyperparameter_search(
    X_list, Y_list, n_inputs,
    alphas=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0],
    n_reservoirs=[250, 500, 1000, 1500],
    washouts=[25, 50, 75, 100],
    spectral_radii=[0.9, 1.0, 1.25, 1.5],
    leak_range=(0.1, 0.3),
    n_folds=3,
    seed=42,
    max_combinations=None,
    quick_mode=False
):
    """
    Grid search over hyperparameters with cross-validation.
    
    Args:
        X_list, Y_list: Lists of sessions (will be modified if quick_mode!)
        n_inputs: Input dimension
        alphas: List of alpha values to try
        n_reservoirs: List of reservoir sizes to try
        washouts: List of washout periods to try
        spectral_radii: List of spectral radius values to try
        leak_range: Fixed leak range
        n_folds: Number of cross-validation folds
        seed: Random seed
        max_combinations: Maximum combinations to try (None = all)
        quick_mode: If True, use fewer combinations for fast testing
    
    Returns:
        results: List of dicts with hyperparameters and metrics
        best_config: Best hyperparameter configuration
    """
    if quick_mode:
        print("\n⚡ QUICK MODE: Usando parametri ridotti per test veloce")
        alphas = [1e-3, 1e-2, 1e-1]
        n_reservoirs = [500, 1000]
        washouts = [50, 75]
        spectral_radii = [1.0, 1.25]
        n_folds = 2  # Reduce folds for speed
        
        # ALSO: Use only first 2000 frames per session for speed!
        print("   ⚡ Subsampling data: max 2000 frames per session")
        X_list_quick = []
        Y_list_quick = []
        for X, Y in zip(X_list, Y_list):
            if len(X) > 2000:
                # Take evenly spaced samples
                indices = np.linspace(0, len(X)-1, 2000, dtype=int)
                X_list_quick.append(X[indices])
                Y_list_quick.append(Y[indices])
            else:
                X_list_quick.append(X)
                Y_list_quick.append(Y)
        X_list = X_list_quick
        Y_list = Y_list_quick
        total_frames_quick = sum(len(x) for x in X_list)
        print(f"   ✓ Reduced to {total_frames_quick} frames total, {n_folds} folds")
    
    # Generate all combinations
    all_combinations = list(product(alphas, n_reservoirs, washouts, spectral_radii))
    
    if max_combinations and len(all_combinations) > max_combinations:
        print(f"\n⚠️  Troppe combinazioni ({len(all_combinations)}), campionamento casuale di {max_combinations}")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_combinations), size=max_combinations, replace=False)
        all_combinations = [all_combinations[i] for i in indices]
    
    n_total = len(all_combinations)
    
    print(f"\n🔍 HYPERPARAMETER SEARCH")
    print(f"   Parametri da testare:")
    print(f"   - Alpha: {alphas}")
    print(f"   - N_reservoir: {n_reservoirs}")
    print(f"   - Washout: {washouts}")
    print(f"   - Spectral radius: {spectral_radii}")
    print(f"   - Leak range: {leak_range}")
    print(f"\n   Totale combinazioni: {n_total}")
    print(f"   Cross-validation folds: {n_folds}")
    print(f"   Tempo stimato: {n_total * 2:.0f}-{n_total * 5:.0f} secondi")
    print(f"\n{'='*80}")
    
    results = []
    best_score = -float('inf')
    best_config = None
    
    start_time = time.time()
    
    for idx, (alpha, n_res, wash, rho) in enumerate(all_combinations, 1):
        iter_start = time.time()
        
        print(f"\n[{idx}/{n_total}] Testing: alpha={alpha:.0e}, N={n_res}, washout={wash}, rho={rho}")
        
        try:
            # Cross-validate
            metrics = cross_validate(
                X_list, Y_list, n_inputs,
                n_reservoir=n_res,
                spectral_radius=rho,
                leak_range=leak_range,
                alpha=alpha,
                washout=wash,
                seed=seed,
                n_folds=n_folds
            )
            
            # Store results
            config = {
                'alpha': alpha,
                'n_reservoir': n_res,
                'washout': wash,
                'spectral_radius': rho,
                'leak_range': leak_range
            }
            
            result = {
                'config': config,
                'metrics': metrics
            }
            results.append(result)
            
            # Check if best so far
            if metrics['score'] > best_score:
                best_score = metrics['score']
                best_config = config
                print(f"   ✨ NEW BEST! Score={metrics['score']:.4f}")
            
            # Print analysis
            print(f"   Val Boost Acc: {metrics['val_boost_acc']:.2%}", end="")
            print(f" | Angular: {metrics['val_angular']:.1f}°", end="")
            print(f" | MSE ratio: {metrics['mse_ratio']:.2f}", end="")
            
            if metrics['mse_ratio'] > 2.0:
                print(" ⚠️ Overfitting")
            elif metrics['mse_ratio'] > 1.5:
                print(" ⚠️")
            else:
                print(" ✓")
            
            iter_time = time.time() - iter_start
            remaining = (n_total - idx) * iter_time
            print(f"   Time: {iter_time:.1f}s | ETA: {remaining:.0f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ SEARCH COMPLETATO in {total_time:.0f} secondi ({total_time/60:.1f} minuti)")
    print(f"\n🏆 MIGLIOR CONFIGURAZIONE:")
    print(f"   Alpha: {best_config['alpha']:.0e}")
    print(f"   N_reservoir: {best_config['n_reservoir']}")
    print(f"   Washout: {best_config['washout']}")
    print(f"   Spectral radius: {best_config['spectral_radius']}")
    
    # Find best result metrics
    best_result = [r for r in results if r['config'] == best_config][0]
    best_metrics = best_result['metrics']
    
    print(f"\n📊 METRICHE MIGLIORI:")
    print(f"   Val Boost Accuracy: {best_metrics['val_boost_acc']:.2%}")
    print(f"   Val Angular Error: {best_metrics['val_angular']:.1f}°")
    print(f"   Train Boost Accuracy: {best_metrics['train_boost_acc']:.2%}")
    print(f"   MSE Ratio: {best_metrics['mse_ratio']:.2f}", end="")
    
    if best_metrics['mse_ratio'] > 2.0:
        print(" ⚠️  Overfitting ancora presente")
    elif best_metrics['mse_ratio'] > 1.5:
        print(" ⚠️  Lieve overfitting")
    else:
        print(" ✅ Buono!")
    
    print(f"   Accuracy Diff: {best_metrics['acc_diff']:.2%}")
    
    # Sort results by score
    results_sorted = sorted(results, key=lambda r: r['metrics']['score'], reverse=True)
    
    return results_sorted, best_config


def save_search_results(results, best_config, output_dir):
    """Save hyperparameter search results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_serializable = []
    for r in results:
        config = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                  for k, v in r['config'].items()}
        metrics = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in r['metrics'].items()}
        results_serializable.append({'config': config, 'metrics': metrics})
    
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    # Save best config
    best_config_serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                for k, v in best_config.items()}
    
    with open(output_dir / "best_config.json", 'w') as f:
        json.dump(best_config_serializable, f, indent=2)
    
    # Create analysis
    summary_lines = [
        "=" * 80,
        "HYPERPARAMETER SEARCH RESULTS",
        "=" * 80,
        "",
        f"Best Configuration:",
        f"  Alpha: {best_config['alpha']:.0e}",
        f"  N_reservoir: {best_config['n_reservoir']}",
        f"  Washout: {best_config['washout']}",
        f"  Spectral radius: {best_config['spectral_radius']}",
        f"  Leak range: {best_config['leak_range']}",
        "",
        "Top 10 Configurations:",
        "-" * 80,
    ]
    
    for idx, r in enumerate(results[:10], 1):
        cfg = r['config']
        met = r['metrics']
        summary_lines.extend([
            f"{idx}. Score={met['score']:.4f} | Val Acc={met['val_boost_acc']:.2%} | MSE ratio={met['mse_ratio']:.2f}",
            f"   alpha={cfg['alpha']:.0e}, N={cfg['n_reservoir']}, wash={cfg['washout']}, rho={cfg['spectral_radius']}"
        ])
    
    with open(output_dir / "analysis.txt", 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n💾 Risultati salvati in: {output_dir}")
    print(f"   - all_results.json (tutti i risultati)")
    print(f"   - best_config.json (miglior configurazione)")
    print(f"   - analysis.txt (riassunto)")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter search for Slither.io ESN")
    parser.add_argument('--quick', action='store_true', help="Quick mode: fewer combinations")
    parser.add_argument('--max-combinations', type=int, default=None, help="Max combinations to try")
    parser.add_argument('--n-folds', type=int, default=3, help="Number of CV folds")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔍 SLITHER.IO ESN - HYPERPARAMETER SEARCH")
    print("="*80)
    
    # Load data
    print(f"\n📁 Caricamento dati da: {SLITHER_DATA_PATH}")
    
    try:
        X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=True)
    except Exception as e:
        print(f"\n❌ Errore nel caricamento dati: {e}")
        return 1
    
    # DON'T concatenate - keep as list of sessions!
    print(f"\n🎯 Preparazione {len(X_list)} sessioni per cross-validation...")
    
    # Get input dimension from first session
    n_inputs = X_list[0].shape[1]
    
    print(f"   ✓ Total sessions: {len(X_list)}")
    print(f"   ✓ Input dimension: {n_inputs}")
    total_frames = sum(len(x) for x in X_list)
    print(f"   ✓ Total frames: {total_frames}")
    
    # Run hyperparameter search
    results, best_config = hyperparameter_search(
        X_list, y_list, n_inputs,
        n_folds=args.n_folds,
        seed=RANDOM_SEED,
        max_combinations=args.max_combinations,
        quick_mode=args.quick
    )
    
    # Save results
    output_dir = Path(__file__).parent / "slither_esn_results" / f"hypersearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_search_results(results, best_config, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ HYPERPARAMETER SEARCH COMPLETATO!")
    print("="*80)
    print(f"\n💡 Prossimi passi:")
    print(f"   1. Copia i parametri migliori in configuration.py:")
    print(f"      ALPHA = {best_config['alpha']:.0e}")
    print(f"      N_RESERVOIR = {best_config['n_reservoir']}")
    print(f"      WASHOUT = {best_config['washout']}")
    print(f"      SPECTRAL_RADIUS = {best_config['spectral_radius']}")
    print(f"   2. Ri-esegui train_slither_reservoir.py con i nuovi parametri")
    print(f"   3. Confronta i risultati con quelli precedenti")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
