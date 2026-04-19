"""
Hyperparameter Search con Binary Search Intelligente per ESN
============================================================

Invece di testare tutti i parametri in una griglia, usa una binary search
assumendo che la funzione di performance sia monotona.

Algoritmo:
1. Testa gli estremi di ogni range
2. Se uno è migliore, dividi quello spazio a metà
3. Ripeti per SEARCH_DEPTH iterazioni
4. Questo riduce drasticamente il numero di test necessari

Esempio: invece di testare 10 valori (10 test), binary search con depth=3
testa solo: [estremo_sx, estremo_dx, metà, quarto_1, quarto_3] = 5 test
"""

import numpy as np
import sys
import time
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from vnicktest.scripts.configuration import *
from utilities.data_loader import load_all_data
from utilities.metrics import compute_direction_metrics, compute_boost_metrics
from reservoir import Reservoir


def compute_wout(X, Y, alpha=1e-3):
    """
    Compute output weights using ridge regression.
    
    Args:
        X: Reservoir states [n_samples, n_reservoir]
        Y: Target outputs [n_samples, n_outputs]
        alpha: Regularization parameter
        
    Returns:
        W_out: Output weight matrix [n_reservoir, n_outputs]
    """
    # Ridge regression: W = (X^T X + alpha*I)^-1 X^T Y
    XtX = X.T @ X
    XtY = X.T @ Y
    
    # Add regularization
    regularization = alpha * np.eye(XtX.shape[0])
    
    # Solve
    W_out = np.linalg.solve(XtX + regularization, XtY)
    
    return W_out


def train_and_evaluate(
    X_list, Y_list, n_inputs,
    n_reservoir, spectral_radius, leak_range,
    alpha, washout, seed, n_folds=3
):
    """
    Train ESN and evaluate con cross-validation.
    Uguale a hyperparameter_search.py ma senza la ricorsione.
    """
    n_sessions = len(X_list)
    fold_size = max(1, n_sessions // n_folds)
    
    fold_metrics = []
    
    for fold in range(n_folds):
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
        
        # Create reservoir
        reservoir = Reservoir(
            n_inputs=n_inputs,
            n_neurons=n_reservoir,
            rhow=spectral_radius,
            inp_scaling=1.0,
            leak_range=leak_range,
            verbose=False
        )
        
        # Collect states from training sessions
        train_states_list = []
        train_targets_list = []
        
        for X_session, Y_session in zip(X_train, Y_train):
            X_states = reservoir.forward(X_session, collect_states=True)
            if len(X_states) <= washout:
                continue
            X_states = X_states[washout:]
            Y_targets = Y_session[washout:]
            train_states_list.append(X_states)
            train_targets_list.append(Y_targets)
        
        if not train_states_list:
            continue
        
        train_states = np.vstack(train_states_list)
        train_targets = np.vstack(train_targets_list)
        
        # Train with ridge regression
        wout = compute_wout(train_states, train_targets, alpha=alpha)
        
        # Predict on training
        train_preds = train_states @ wout
        
        # Collect validation states
        val_states_list = []
        val_targets_list = []
        
        for X_session, Y_session in zip(X_val, Y_val):
            X_states = reservoir.forward(X_session, collect_states=True)
            if len(X_states) <= washout:
                continue
            X_states = X_states[washout:]
            Y_targets = Y_session[washout:]
            val_states_list.append(X_states)
            val_targets_list.append(Y_targets)
        
        if not val_states_list:
            continue
        
        val_states = np.vstack(val_states_list)
        val_targets = np.vstack(val_targets_list)
        
        # Predict on validation
        val_preds = val_states @ wout
        
        # Compute metrics
        train_dir = compute_direction_metrics(train_targets[:, :2], train_preds[:, :2])
        train_boost = compute_boost_metrics(train_targets, train_preds)
        val_dir = compute_direction_metrics(val_targets[:, :2], val_preds[:, :2])
        val_boost = compute_boost_metrics(val_targets, val_preds)
        
        fold_metrics.append({
            'train': {'direction': train_dir, 'boost': train_boost},
            'val': {'direction': val_dir, 'boost': val_boost}
        })
    
    if not fold_metrics:
        return None
    
    # Average across folds
    mean_train_rmse = np.mean([m['train']['direction']['rmse_direction'] for m in fold_metrics])
    mean_train_boost_acc = np.mean([m['train']['boost']['boost_accuracy'] for m in fold_metrics])
    mean_val_rmse = np.mean([m['val']['direction']['rmse_direction'] for m in fold_metrics])
    mean_val_boost_acc = np.mean([m['val']['boost']['boost_accuracy'] for m in fold_metrics])
    
    mse_ratio = (mean_val_rmse ** 2) / (mean_train_rmse ** 2) if mean_train_rmse > 0 else float('inf')
    acc_diff = mean_train_boost_acc - mean_val_boost_acc
    
    # IMPROVED SCORING FUNCTION:
    # 1. Prioritize absolute validation performance (80% weight)
    # 2. Penalize overfitting only if severe (20% weight, only if ratio > 1.3)
    overfitting_penalty = max(0, (mse_ratio - 1.3)) * 0.1  # Penalty only if MSE ratio > 1.3
    acc_penalty = max(0, (acc_diff - 0.05)) * 0.2  # Penalty only if acc diff > 5%
    
    score = mean_val_boost_acc - overfitting_penalty - acc_penalty
    
    return {
        'val_boost_acc': mean_val_boost_acc,
        'val_rmse': mean_val_rmse,
        'mse_ratio': mse_ratio,
        'acc_diff': acc_diff,
        'score': score  # Now optimizes for absolute performance first
    }


def binary_search_parameter(
    param_name, param_range, 
    X_list, Y_list, n_inputs,
    fixed_params, 
    search_depth=3,
    n_folds=2
):
    """
    Binary search su un singolo parametro.
    
    Args:
        param_name: Nome del parametro ('alpha', 'n_reservoir', etc)
        param_range: [min, max] range da esplorare
        fixed_params: Altri parametri fissi
        search_depth: Quante iterazioni di binary search
        
    Returns:
        best_value, best_score, tested_values
    """
    print(f"\n🔍 Binary search su '{param_name}': range {param_range}")
    
    # Start con gli estremi
    left, right = param_range
    tested = {}
    
    # Test estremi
    for val in [left, right]:
        params = fixed_params.copy()
        params[param_name] = val
        
        val_str = f"{val:.2e}" if param_name == 'alpha' else str(val)
        print(f"   Testing {param_name}={val_str}...", end=" ")
        
        result = train_and_evaluate(
            X_list, Y_list, n_inputs,
            n_reservoir=params['n_reservoir'],
            spectral_radius=params['spectral_radius'],
            leak_range=params['leak_range'],
            alpha=params['alpha'],
            washout=params['washout'],
            seed=params['seed'],
            n_folds=n_folds
        )
        
        if result:
            tested[val] = result['score']
            print(f"score={result['score']:.4f}")
        else:
            tested[val] = -float('inf')
            print("FAILED")
    
    # Iterative binary search
    for depth in range(search_depth):
        # Find best so far
        best_val = max(tested, key=tested.get)
        best_score = tested[best_val]
        
        # Determine search direction
        left_score = tested.get(left, -float('inf'))
        right_score = tested.get(right, -float('inf'))
        
        # Se left è meglio, cerca tra left e mid
        # Se right è meglio, cerca tra mid e right
        if left_score > right_score:
            # Search left half
            new_val = (left + best_val) / 2 if param_name == 'alpha' else int((left + best_val) / 2)
            search_left, search_right = left, best_val
        else:
            # Search right half
            new_val = (best_val + right) / 2 if param_name == 'alpha' else int((best_val + right) / 2)
            search_left, search_right = best_val, right
        
        # Skip if already tested or too close
        if new_val in tested:
            break
        if param_name != 'alpha' and abs(new_val - best_val) < 10:
            break
        
        # Test new value
        params = fixed_params.copy()
        params[param_name] = new_val
        
        val_str = f"{new_val:.2e}" if param_name == 'alpha' else str(new_val)
        print(f"   [Depth {depth+1}] Testing {param_name}={val_str}...", end=" ")
        
        result = train_and_evaluate(
            X_list, Y_list, n_inputs,
            n_reservoir=params['n_reservoir'],
            spectral_radius=params['spectral_radius'],
            leak_range=params['leak_range'],
            alpha=params['alpha'],
            washout=params['washout'],
            seed=params['seed'],
            n_folds=n_folds
        )
        
        if result:
            tested[new_val] = result['score']
            print(f"score={result['score']:.4f}")
        else:
            tested[new_val] = -float('inf')
            print("FAILED")
    
    # Return best
    best_val = max(tested, key=tested.get)
    best_score = tested[best_val]
    
    val_str = f"{best_val:.2e}" if param_name == 'alpha' else str(best_val)
    print(f"   ✨ Best {param_name}: {val_str} (score={best_score:.4f})")
    
    return best_val, best_score, tested


def main():
    parser = argparse.ArgumentParser(description='Binary Search Hyperparameter Optimization')
    parser.add_argument('--depth', type=int, default=3, help='Binary search depth (default: 3)')
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer combinations and less data')
    args = parser.parse_args()
    
    SEARCH_DEPTH = args.depth
    
    print("\n" + "="*80)
    print("🔍 SLITHER.IO ESN - BINARY SEARCH HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load data
    print(f"\n📁 Caricamento dati da: {SLITHER_DATA_PATH}")
    X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) == 0:
        print("\n❌ No data found!")
        return 1
    
    n_inputs = X_list[0].shape[1]
    n_folds = 2 if args.quick else 3
    
    # Subsample in quick mode
    if args.quick:
        print("\n⚡ QUICK MODE: Subsampling to 1000 frames per session")
        X_quick, y_quick = [], []
        for X, y in zip(X_list, y_list):
            if len(X) > 1000:
                indices = np.linspace(0, len(X)-1, 1000, dtype=int)
                X_quick.append(X[indices])
                y_quick.append(y[indices])
            else:
                X_quick.append(X)
                y_quick.append(y)
        X_list, y_list = X_quick, y_quick
        total = sum(len(x) for x in X_list)
        print(f"   ✓ Reduced to {total} frames total")
    
    print(f"\n🎯 Binary Search Configuration:")
    print(f"   - Search depth: {SEARCH_DEPTH}")
    print(f"   - Cross-validation folds: {n_folds}")
    print(f"   - Sessions: {len(X_list)}")
    
    # Define parameter ranges
    param_ranges = {
        'alpha': (1e-4, 1.0),
        'n_reservoir': (250, 2000),
        'washout': (25, 150),
        'spectral_radius': (0.5, 2.0)
    }
    
    # Start with defaults from configuration
    best_params = {
        'alpha': ALPHA,
        'n_reservoir': N_RESERVOIR,
        'washout': WASHOUT,
        'spectral_radius': SPECTRAL_RADIUS,
        'leak_range': (LEAK_RATE_MIN, LEAK_RATE_MAX),
        'seed': RANDOM_SEED
    }
    
    print("\n📊 Starting parameters (from configuration.py):")
    for k, v in best_params.items():
        if k != 'leak_range' and k != 'seed':
            print(f"   - {k}: {v}")
    
    start_time = time.time()
    
    # Optimize each parameter sequentially
    # L'ordine è importante: alpha è più critico, poi n_reservoir, poi il resto
    param_order = ['alpha', 'n_reservoir', 'spectral_radius', 'washout']
    
    for param_name in param_order:
        best_val, best_score, tested = binary_search_parameter(
            param_name,
            param_ranges[param_name],
            X_list, y_list, n_inputs,
            best_params,
            search_depth=SEARCH_DEPTH,
            n_folds=n_folds
        )
        
        # Update best params
        best_params[param_name] = best_val
        
        print(f"\n   📊 Updated best configuration:")
        for k, v in best_params.items():
            if k != 'leak_range' and k != 'seed':
                val_str = f"{v:.2e}" if k == 'alpha' else str(v)
                print(f"      {k}: {val_str}")
    
    total_time = time.time() - start_time
    
    # Final evaluation with best params
    print(f"\n{'='*80}")
    print("🏆 FINAL BEST CONFIGURATION")
    print(f"{'='*80}")
    
    for k, v in best_params.items():
        if k != 'leak_range' and k != 'seed':
            val_str = f"{v:.2e}" if k == 'alpha' else str(v)
            print(f"   {k}: {val_str}")
    
    # Final detailed evaluation
    print(f"\n📊 Final evaluation con {n_folds}-fold CV...")
    final_result = train_and_evaluate(
        X_list, y_list, n_inputs,
        n_reservoir=best_params['n_reservoir'],
        spectral_radius=best_params['spectral_radius'],
        leak_range=best_params['leak_range'],
        alpha=best_params['alpha'],
        washout=best_params['washout'],
        seed=best_params['seed'],
        n_folds=n_folds
    )
    
    if final_result:
        print(f"\n   Val Boost Accuracy: {final_result['val_boost_acc']:.2%}")
        print(f"   Val RMSE (direction): {final_result['val_rmse']:.4f}")
        print(f"   MSE Ratio: {final_result['mse_ratio']:.2f}")
        print(f"   Acc Difference: {final_result['acc_diff']:.2%}")
        print(f"   Combined Score: {final_result['score']:.4f}")
    
    print(f"\n⏱️  Total search time: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Save best config
    output_file = Path("best_config_binary_search.json")
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types
        config_to_save = {}
        for k, v in best_params.items():
            if isinstance(v, (np.integer, np.floating)):
                config_to_save[k] = float(v)
            elif isinstance(v, tuple):
                config_to_save[k] = list(v)
            else:
                config_to_save[k] = v
        
        json.dump({
            'best_config': config_to_save,
            'final_metrics': final_result if final_result else {},
            'search_time': total_time,
            'search_depth': SEARCH_DEPTH
        }, f, indent=2)
    
    print(f"\n💾 Best configuration saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("✅ BINARY SEARCH COMPLETATO!")
    print(f"{'='*80}")
    
    print(f"\n💡 Prossimi passi:")
    print(f"   1. Copia i parametri migliori in configuration.py:")
    print(f"      ALPHA = {best_params['alpha']:.2e}")
    print(f"      N_RESERVOIR = {best_params['n_reservoir']}")
    print(f"      WASHOUT = {best_params['washout']}")
    print(f"      SPECTRAL_RADIUS = {best_params['spectral_radius']:.2f}")
    print(f"   2. Ri-esegui train_slither_reservoir.py con i nuovi parametri")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
