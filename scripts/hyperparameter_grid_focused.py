"""
Focused Grid Search around the most promising hyperparameter region.

By default this targets the **brain-connectome** reservoir (the canonical
research target).  Pass ``--reservoir random`` for the random-matrix
baseline that produced the original best_config_focused_grid.json numbers.
"""

import argparse
import numpy as np
import sys
import time
from pathlib import Path

# Ensure we can import sibling scripts and the package modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hyperparameter_search_binary import train_and_evaluate, _find_default_graph_dir
from vnicktest.scripts.configuration import *
from utilities.data_loader import load_all_data


def main():
    parser = argparse.ArgumentParser(description="Focused grid search")
    parser.add_argument('--reservoir', choices=['connectome', 'random'], default='connectome')
    parser.add_argument('--graph-dir', type=str, default=None)
    args = parser.parse_args()
    reservoir_type = args.reservoir
    graph_dir = Path(args.graph_dir) if args.graph_dir else (
        _find_default_graph_dir() if reservoir_type == 'connectome' else None
    )
    print(f"\n[focused-grid] reservoir_type={reservoir_type}"
          + (f"  graph_dir={graph_dir}" if graph_dir else ""))
    print("\n" + "="*80)
    print("🎯 FOCUSED GRID SEARCH - Parametri Promettenti")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {SLITHER_DATA_PATH}")
    X_list, y_list, session_names, _usernames = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) == 0:
        print("\n❌ No data found!")
        return 1
    
    n_inputs = X_list[0].shape[1]
    
    # Subsample for speed
    print("\n⚡ Subsampling to 2000 frames per session for speed")
    X_quick, y_quick = [], []
    for X, y in zip(X_list, y_list):
        if len(X) > 2000:
            indices = np.linspace(0, len(X)-1, 2000, dtype=int)
            X_quick.append(X[indices])
            y_quick.append(y[indices])
        else:
            X_quick.append(X)
            y_quick.append(y)
    X_list, y_list = X_quick, y_quick
    
    # FOCUSED parameter ranges - solo valori promettenti!
    params_grid = {
        'alpha': [0.005, 0.01, 0.015, 0.02],  # Attorno a 0.01
        'n_reservoir': [500, 625, 750],  # Da 500 a 750
        'spectral_radius': [0.95, 1.0, 1.1],  # Attorno a 1.0
        'washout': [40, 50, 60]  # Attorno a 50
    }
    
    print(f"\n🔍 Testing Focused Grid:")
    for k, v in params_grid.items():
        print(f"   - {k}: {v}")
    
    n_combinations = (len(params_grid['alpha']) * 
                     len(params_grid['n_reservoir']) * 
                     len(params_grid['spectral_radius']) * 
                     len(params_grid['washout']))
    
    print(f"\n   Totale combinazioni: {n_combinations}")
    print(f"   Tempo stimato: {n_combinations * 1:.0f}-{n_combinations * 2:.0f} secondi")
    
    # Test all combinations
    results = []
    best_score = -float('inf')
    best_config = None
    
    start_time = time.time()
    test_num = 0
    
    for alpha in params_grid['alpha']:
        for n_res in params_grid['n_reservoir']:
            for rho in params_grid['spectral_radius']:
                for wash in params_grid['washout']:
                    test_num += 1
                    
                    print(f"\n[{test_num}/{n_combinations}] α={alpha:.3f}, N={n_res}, ρ={rho:.2f}, w={wash}")
                    
                    result = train_and_evaluate(
                        X_list, y_list, n_inputs,
                        n_reservoir=n_res,
                        spectral_radius=rho,
                        leak_range=(0.1, 0.3),
                        alpha=alpha,
                        washout=wash,
                        seed=42,
                        n_folds=2,
                        reservoir_type=reservoir_type, graph_dir=graph_dir,
                    )
                    
                    if result:
                        config = {
                            'alpha': alpha,
                            'n_reservoir': n_res,
                            'spectral_radius': rho,
                            'washout': wash
                        }
                        
                        results.append({
                            'config': config,
                            'metrics': result
                        })
                        
                        print(f"   Val Acc: {result['val_boost_acc']:.2%} | Score: {result['score']:.4f} | MSE ratio: {result['mse_ratio']:.2f}")
                        
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_config = config
                            print(f"   ✨ NEW BEST!")
                    else:
                        print(f"   ❌ FAILED")
    
    total_time = time.time() - start_time
    
    # Sort results by score
    results.sort(key=lambda r: r['metrics']['score'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"✅ SEARCH COMPLETATO in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}")
    
    print(f"\n🏆 TOP 5 CONFIGURAZIONI:")
    for i, r in enumerate(results[:5], 1):
        c = r['config']
        m = r['metrics']
        print(f"\n{i}. α={c['alpha']:.3f}, N={c['n_reservoir']}, ρ={c['spectral_radius']:.2f}, w={c['washout']}")
        print(f"   Val Acc: {m['val_boost_acc']:.2%} | MSE ratio: {m['mse_ratio']:.2f} | Score: {m['score']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"🥇 BEST CONFIGURATION:")
    print(f"{'='*80}")
    print(f"   ALPHA = {best_config['alpha']:.3f}")
    print(f"   N_RESERVOIR = {best_config['n_reservoir']}")
    print(f"   SPECTRAL_RADIUS = {best_config['spectral_radius']:.2f}")
    print(f"   WASHOUT = {best_config['washout']}")
    
    best_result = [r for r in results if r['config'] == best_config][0]
    best_metrics = best_result['metrics']
    
    print(f"\n📊 BEST METRICS:")
    print(f"   Val Boost Accuracy: {best_metrics['val_boost_acc']:.2%}")
 