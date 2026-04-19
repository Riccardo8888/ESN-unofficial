#!/usr/bin/env python3
"""
Train Slither.io ESN - Compatible with existing reservoir.py approach
======================================================================

Questo script usa lo stesso approccio del notebook humand_data.ipynb:
- Usa reservoir.forward() per raccogliere stati
- Training con compute_wout() usando ridge regression manuale
- Stesso workflow: collect states -> compute wout -> predict

"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from vnicktest.scripts.configuration import *
from utilities.data_loader import (
    load_all_data, 
    train_test_split
)
from reservoir import Reservoir  # Usa il reservoir esistente!
from utilities.metrics import (
    compute_all_metrics, 
    print_metrics,
    compute_angle_classification_metrics,
    compute_boost_metrics,
    compare_metrics
)


def compute_wout(X, Y, T_washout=WASHOUT, alpha=ALPHA):
    """
    Compute output weights using ridge regression.
    
    Questo è lo stesso approccio usato in humand_data.ipynb:
    - Rimuove washout period
    - Ridge regression: (X^T X + alpha I)^-1 X^T Y
    
    Args:
        X: Reservoir states [n_timesteps, n_reservoir]
        Y: Target outputs [n_timesteps, n_outputs]
        T_washout: Frames to discard at start
        alpha: Regularization parameter
        
    Returns:
        W_out: Output weights [n_reservoir, n_outputs]
    """
    # Remove washout period
    X_train = X[T_washout:]
    Y_train = Y[T_washout:]
    
    # Ridge regression
    R = X_train.T @ X_train  # State correlation matrix
    P = X_train.T @ Y_train  # State-output cross-correlation
    
    # Solve: W_out = (R + alpha I)^-1 P
    # Use solve() instead of inv() for speed (2-3x faster)
    W_out = np.linalg.solve(R + alpha * np.eye(X_train.shape[1]), P)
    
    return W_out


def main():
    """Main training pipeline - compatible with humand_data.ipynb approach"""
    
    # Print configuration
    print_config()
    
    # ===========================================
    # STEP 1: LOAD DATA
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    
    X_list, y_list, session_names, usernames = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) == 0:
        print("\n❌ ERROR: No data found!")
        print(f"   Please check that data exists in: {SLITHER_DATA_PATH}")
        print(f"   This should be the 'data/' folder in this workspace")
        return 1
    
    # ===========================================
    # STEP 2: TRAIN/TEST SPLIT
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN/TEST SPLIT (usando chunk casuali)")
    print("=" * 60)
    
    # Use new chunk-based splitting for more homogeneous test set
    X_train_cat, y_train_cat, X_test_cat, y_test_cat, test_user_indices = train_test_split(
        X_list, y_list, session_names, usernames,
        test_size=TEST_SPLIT, 
        random_seed=RANDOM_SEED,
        use_chunks=True  # NEW: Take random chunks from each session
    )
    
    print(f"\nTaking {TEST_SPLIT:.0%} random chunks from each session for test set")
    print(f"This creates a more representative and homogeneous test set")
    
    print(f"\nData shapes:")
    print(f"  Training:   X={X_train_cat.shape}, y={y_train_cat.shape}")
    print(f"  Test:       X={X_test_cat.shape}, y={y_test_cat.shape}")
    print(f"  Test ratio: {X_test_cat.shape[0] / (X_train_cat.shape[0] + X_test_cat.shape[0]):.1%}")
    print(f"  Input dim:  {X_train_cat.shape[1]}")
    print(f"  Output dim: {y_train_cat.shape[1]}")
    
    # ===========================================
    # STEP 3: CREATE RESERVOIR
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 3: CREATING RESERVOIR (usando reservoir.py)")
    print("=" * 60)
    
    # Usa la classe Reservoir esistente (come Reservoir, Reservoir2, Reservoir3)
    reservoir = Reservoir(
        n_inputs=X_train_cat.shape[1],
        n_neurons=N_RESERVOIR,
        rhow=SPECTRAL_RADIUS,
        inp_scaling=INPUT_SCALE,
        leak_range=(LEAK_RATE_MIN, LEAK_RATE_MAX),
        verbose=True
    )
    
    print(f"\nReservoir properties:")
    print(f"  Input dim: {reservoir.n_inputs}")
    print(f"  Neurons: {reservoir.n_neurons}")
    print(f"  Spectral radius: {reservoir.spectral_radius:.3f}")
    print(f"  Leak rate range: [{LEAK_RATE_MIN}, {LEAK_RATE_MAX}]")
    
    # ===========================================
    # STEP 4: COLLECT RESERVOIR STATES (TRAINING)
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 4: COLLECTING RESERVOIR STATES (Training)")
    print("=" * 60)
    
    print(f"Running forward pass on training data...")
    X_train_states = reservoir.forward(X_train_cat, collect_states=True, show_progress=True)
    print(f"  ✓ Collected states: {X_train_states.shape}")
    
    # ===========================================
    # STEP 5: TRAIN OUTPUT WEIGHTS
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 5: TRAINING OUTPUT WEIGHTS (Ridge Regression)")
    print("=" * 60)
    
    print(f"Computing W_out with washout={WASHOUT}, alpha={ALPHA}...")
    wout = compute_wout(X_train_states, y_train_cat, T_washout=WASHOUT, alpha=ALPHA)
    print(f"  ✓ W_out shape: {wout.shape}")
    
    # ===========================================
    # STEP 6: EVALUATE ON TRAINING SET
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATING ON TRAINING SET")
    print("=" * 60)
    
    # Make predictions: Y = X @ W_out
    y_train_pred = X_train_states @ wout
    y_train_true = y_train_cat
    
    # Remove washout for fair comparison
    y_train_pred = y_train_pred[WASHOUT:]
    y_train_true = y_train_true[WASHOUT:]
    
    print(f"Computing metrics (after washout)...")
    train_metrics = compute_all_metrics(y_train_true, y_train_pred)
    print_metrics(train_metrics, "Training")
    
    # ===========================================
    # STEP 7: COLLECT STATES & EVALUATE TEST SET
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 7: EVALUATING ON TEST SET")
    print("=" * 60)
    
    print(f"Running forward pass on test data...")
    X_test_states = reservoir.forward(X_test_cat, collect_states=True, show_progress=True)
    print(f"  ✓ Collected states: {X_test_states.shape}")
    
    # Make predictions
    y_test_pred = X_test_states @ wout
    y_test_true = y_test_cat
    
    # Remove washout
    y_test_pred = y_test_pred[WASHOUT:]
    y_test_true = y_test_true[WASHOUT:]
    
    print(f"Computing metrics (after washout)...")
    test_metrics = compute_all_metrics(y_test_true, y_test_pred)
    print_metrics(test_metrics, "Test")
    
    # ===========================================
    # STEP 8: COMPARE RESULTS
    # ===========================================
    compare_metrics(train_metrics, test_metrics)
    
    # ===========================================
    # STEP 8.5: PER-USER ANALYSIS
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 8.5: PER-USER ANALYSIS (Test Set)")
    print("=" * 60)
    
    # Adjust test_user_indices for washout
    test_user_indices_adj = test_user_indices[WASHOUT:]
    
    # Group by user
    unique_users = {}
    for idx in range(len(X_list)):
        username = usernames[idx]
        if username not in unique_users:
            unique_users[username] = []
        unique_users[username].append(idx)
    
    print(f"\nAnalyzing performance across {len(unique_users)} users:")
    print(f"{'User':<15} {'Frames':<10} {'Boost Acc':<12} {'Angle Acc':<12} {'Angular Err':<12}")
    print("=" * 65)
    
    user_stats = {}
    for username, session_indices in unique_users.items():
        # Find test samples belonging to this user's sessions
        user_mask = np.isin(test_user_indices_adj, session_indices)
        
        if not user_mask.any():
            continue
        
        # Get predictions and targets for this user
        y_user_pred = y_test_pred[user_mask]
        y_user_true = y_test_true[user_mask]
        
        # Compute metrics
        user_angle = compute_angle_classification_metrics(y_user_true, y_user_pred)
        user_boost = compute_boost_metrics(y_user_true, y_user_pred)
        
        user_stats[username] = {
            'n_frames': len(y_user_pred),
            'boost_accuracy': user_boost['boost_accuracy'],
            'angle_accuracy': user_angle['accuracy'],
            'angular_error_deg': user_angle['angular_error_deg']
        }
        
        print(f"{username:<15} {len(y_user_pred):<10} "
              f"{user_boost['boost_accuracy']:>10.2%}  "
              f"{user_angle['accuracy']:>10.2%}  "
              f"{user_angle['angular_error_deg']:>10.2f}°")
    
    # Summary statistics
    if len(user_stats) > 1:
        all_boost_accs = [s['boost_accuracy'] for s in user_stats.values()]
        all_angle_accs = [s['angle_accuracy'] for s in user_stats.values()]
        
        print("\n" + "-" * 65)
        print(f"{'MEAN':<15} {'':<10} {np.mean(all_boost_accs):>10.2%}  {np.mean(all_angle_accs):>10.2%}")
        print(f"{'STD':<15} {'':<10} {np.std(all_boost_accs):>10.2%}  {np.std(all_angle_accs):>10.2%}")
        print(f"{'MIN':<15} {'':<10} {np.min(all_boost_accs):>10.2%}  {np.min(all_angle_accs):>10.2%}")
        print(f"{'MAX':<15} {'':<10} {np.max(all_boost_accs):>10.2%}  {np.max(all_angle_accs):>10.2%}")
        
        # Identify best and worst performers
        best_user = max(user_stats, key=lambda u: user_stats[u]['boost_accuracy'])
        worst_user = min(user_stats, key=lambda u: user_stats[u]['boost_accuracy'])
        
        print(f"\n💡 Insights:")
        print(f"   🏆 Best performer: {best_user} ({user_stats[best_user]['boost_accuracy']:.2%} accuracy)")
        print(f"   ⚠️  Worst performer: {worst_user} ({user_stats[worst_user]['boost_accuracy']:.2%} accuracy)")
        
        acc_range = np.max(all_boost_accs) - np.min(all_boost_accs)
        if acc_range > 0.10:  # 10% difference
            print(f"   📊 Large variance ({acc_range:.1%}) suggests user-specific patterns")
        else:
            print(f"   ✓ Low variance ({acc_range:.1%}) indicates good generalization across users")
    
    # ===========================================
    # STEP 9: SAVE RESULTS
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 9: SAVING RESULTS")
    print("=" * 60)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_PATH / f"training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reservoir and weights
    reservoir_data = {
        'W_in': reservoir.win,
        'W': reservoir.w,
        'W_out': wout,
        'leak': reservoir.leak,
        'n_inputs': reservoir.n_inputs,
        'n_neurons': reservoir.n_neurons,
        'spectral_radius': reservoir.spectral_radius
    }
    
    model_path = output_dir / "reservoir_model.npz"
    np.savez(model_path, **reservoir_data)
    
    # Save training statistics for fine-tuning (after washout)
    X_train_clean = X_train_states[WASHOUT:]
    y_train_clean = y_train_cat[WASHOUT:]
    R = X_train_clean.T @ X_train_clean
    P = X_train_clean.T @ y_train_clean
    stats_path = output_dir / "training_stats.npz"
    np.savez(stats_path, R=R, P=P)
    
    # Save metrics
    results = {
        'timestamp': timestamp,
        'config': {
            'prediction_horizon': PREDICTION_HORIZON,
            'n_reservoir': N_RESERVOIR,
            'spectral_radius': SPECTRAL_RADIUS,
            'input_scale': INPUT_SCALE,
            'leak_rate_min': LEAK_RATE_MIN,
            'leak_rate_max': LEAK_RATE_MAX,
            'alpha': ALPHA,
            'washout': WASHOUT,
            'test_split': TEST_SPLIT,
            'random_seed': RANDOM_SEED,
            'split_method': 'random_chunks_per_session',
            'approach': 'reservoir.py (same as humand_data.ipynb)'
        },
        'data': {
            'total_sessions': len(session_names),
            'n_train_frames': X_train_cat.shape[0],
            'n_test_frames': X_test_cat.shape[0],
            'input_dim': X_train_cat.shape[1],
            'output_dim': y_train_cat.shape[1]
        },
        'training': train_metrics,
        'test': test_metrics
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - Model: {model_path.name}")
    print(f"  - Results: {results_path.name}")
    
    # ===========================================
    # FINAL SUMMARY
    # ===========================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 Key Results:")
    print(f"  Angle Accuracy (test):  {test_metrics['accuracy']*100:.2f}%")
    print(f"  Top-3 Accuracy (test):  {test_metrics['top3_accuracy']*100:.2f}%")
    print(f"  Angular Error (test):   {test_metrics['angular_error_deg']:.2f}°")
    print(f"  Boost Accuracy (test):  {test_metrics['boost_accuracy']*100:.2f}%")
    
    print(f"\n📁 Output Directory: {output_dir}")
    
    print(f"\n📝 Note: Questo script usa lo stesso approccio di humand_data.ipynb:")
    print(f"   - reservoir.forward() per raccogliere stati")
    print(f"   - compute_wout() per training con ridge regression")
    print(f"   - X @ W_out per predizioni")
    
    # ===========================================
    # APPEND TO TRAINING HISTORY LOG
    # ===========================================
    log_file = Path(__file__).parent / "TRAINING_RESULTS.txt"
    
    # Create header if file doesn't exist
    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SLITHER.IO ESN TRAINING RESULTS LOG\n")
            f.write("="*80 + "\n\n")
    
    # Append this training's results
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training: {timestamp} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("ESN Configuration:\n")
        f.write(f"  - Reservoir size: {N_RESERVOIR}\n")
        f.write(f"  - Spectral radius: {SPECTRAL_RADIUS}\n")
        f.write(f"  - Input scale: {INPUT_SCALE}\n")
        f.write(f"  - Leak rate: [{LEAK_RATE_MIN}, {LEAK_RATE_MAX}]\n")
        f.write(f"  - Sparsity: {SPARSITY}\n\n")
        
        f.write("Training Configuration:\n")
        f.write(f"  - Alpha (regularization): {ALPHA}\n")
        f.write(f"  - Washout: {WASHOUT} frames\n")
        f.write(f"  - Prediction horizon: {PREDICTION_HORIZON} frames\n")
        f.write(f"  - Test split: {TEST_SPLIT*100:.0f}%\n")
        f.write(f"  - Random seed: {RANDOM_SEED}\n")
        f.write(f"  - Total frames: {X_train_cat.shape[0] + X_test_cat.shape[0]}\n\n")
        
        f.write("🎯 Key Results:\n")
        f.write(f"  Angle Accuracy (test):  {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Top-3 Accuracy (test):  {test_metrics['top3_accuracy']*100:.2f}%\n")
        f.write(f"  Angular Error (test):   {test_metrics['angular_error_deg']:.2f}°\n")
        f.write(f"  Boost Accuracy (test):  {test_metrics['boost_accuracy']*100:.2f}%\n")
        f.write(f"  Error Ratio (test/train): {test_metrics['angular_error_deg']/train_metrics['angular_error_deg']:.3f}\n\n")
        
        # Per-user stats if available
        if len(user_stats) > 0:
            f.write("👥 Per-User Performance:\n")
            for username, stats in sorted(user_stats.items(), key=lambda x: x[1]['boost_accuracy'], reverse=True):
                f.write(f"  - {username:<12} Boost Acc: {stats['boost_accuracy']*100:>6.2f}%  "
                       f"Angle Acc: {stats['angle_accuracy']*100:>6.2f}%  ({stats['n_frames']} frames)\n")
            f.write("\n")
        
        f.write(f"📁 Output: {output_dir.name}\n")
        f.write(f"{'-'*80}\n")
    
    print(f"\n✅ Results appended to: {log_file}")
    
    print("\n" + "=" * 60)
    print("✓ ALL DONE!")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
