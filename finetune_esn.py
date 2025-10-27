#!/usr/bin/env python3
"""
Fine-tune ESN - Seconda Passata di Training
============================================

Carica un modello ESN già addestrato e fa una seconda passata
sui dati per migliorare le performance.
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from configuration import *
from utilities.data_loader import (
    load_all_data, 
    train_test_split, 
    concatenate_sessions
)
from reservoir import Reservoir
from utilities.metrics import (
    compute_all_metrics, 
    print_metrics,
)


def compute_wout_incremental(X_new, Y_new, W_out_old, R_old, P_old, alpha=ALPHA, learning_rate=0.5):
    """
    Aggiorna W_out con nuovi dati usando update incrementale con learning rate.
    
    Due strategie:
    1. Se R_old/P_old disponibili: accumula statistiche e risolve
    2. Se non disponibili: gradient descent partendo da W_out_old
    
    Args:
        X_new: Nuovi stati reservoir [n_samples, n_reservoir]
        Y_new: Nuovi target [n_samples, n_outputs]
        W_out_old: Pesi output precedenti [n_reservoir, n_outputs]
        R_old: Matrice correlazione stati precedente [n_reservoir, n_reservoir] (opzionale)
        P_old: Matrice cross-correlazione precedente [n_reservoir, n_outputs] (opzionale)
        alpha: Regolarizzazione
        learning_rate: Peso da dare ai nuovi dati (0=nessun cambiamento, 1=solo nuovi dati)
        
    Returns:
        W_out_new, R_new, P_new
    """
    if R_old is not None and P_old is not None:
        # Strategia 1: Accumula statistiche con learning rate
        R_new = R_old + learning_rate * (X_new.T @ X_new)
        P_new = P_old + learning_rate * (X_new.T @ Y_new)
        
        # Ricalcola W_out
        W_out_new = np.linalg.inv(R_new + alpha * np.eye(X_new.shape[1])) @ P_new
    else:
        # Strategia 2: Gradient descent (quando non abbiamo R_old/P_old)
        # Calcola gradiente: dL/dW = -2 * X^T * (Y - X*W) + 2*alpha*W
        predictions = X_new @ W_out_old
        residuals = Y_new - predictions
        gradient = -2 * (X_new.T @ residuals) + 2 * alpha * W_out_old
        
        # Update con learning rate ridotto per stabilità
        W_out_new = W_out_old - learning_rate * 0.001 * gradient / X_new.shape[0]
        
        # Calcola nuove statistiche
        R_new = X_new.T @ X_new
        P_new = X_new.T @ Y_new
    
    return W_out_new, R_new, P_new


def main():
    """Fine-tune existing ESN model"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune ESN model')
    parser.add_argument('--mode', type=str, default='optimize', 
                       choices=['optimize', 'generalize'],
                       help='optimize: use all data (maximize performance), '
                            'generalize: keep test set separate (measure generalization)')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(f"ESN FINE-TUNING - SECONDA PASSATA (MODE: {args.mode.upper()})")
    print("=" * 70)
    
    if args.mode == 'optimize':
        print("\n⚙️  OPTIMIZATION MODE:")
        print("   - Training on ALL data (train + test)")
        print("   - Goal: Maximize overall performance")
        print("   - No generalization evaluation")
    else:
        print("\n🧪 GENERALIZATION MODE:")
        print("   - Training on TRAIN data only")
        print("   - Testing on held-out TEST data")
        print("   - Goal: Evaluate generalization ability")
    
    # ===========================================
    # STEP 1: LOAD EXISTING MODEL
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING EXISTING MODEL")
    print("=" * 70)
    
    # Find latest model
    training_dirs = sorted([d for d in OUTPUT_PATH.iterdir() 
                           if d.is_dir() and d.name.startswith('training_')], 
                          reverse=True)
    
    if not training_dirs:
        print("❌ No existing models found! Train a model first:")
        print("   python3 train_slither_reservoir.py")
        return 1
    
    model_dir = training_dirs[0]
    model_path = model_dir / "reservoir_model.npz"
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model_data = np.load(model_path)
    W_in = model_data['W_in']
    W = model_data['W']
    W_out_old = model_data['W_out']
    leak = model_data['leak']
    
    # Reconstruct reservoir
    reservoir = Reservoir(
        n_inputs=int(model_data['n_inputs']),
        n_neurons=int(model_data['n_neurons']),
        rhow=float(model_data['spectral_radius']),
        inp_scaling=1.0,
        leak_range=(leak.min(), leak.max())
    )
    reservoir.win = W_in
    reservoir.w = W
    reservoir.leak = leak
    
    print(f"✓ Model loaded: {reservoir.n_neurons} neurons")
    
    # Load old statistics (if saved)
    stats_path = model_dir / "training_stats.npz"
    if stats_path.exists():
        stats = np.load(stats_path)
        R_old = stats['R']
        P_old = stats['P']
        print(f"✓ Training statistics loaded")
    else:
        print("⚠️  No training statistics found, will recompute from scratch")
        R_old = None
        P_old = None
    
    # ===========================================
    # STEP 2: LOAD DATA (SAME AS BEFORE)
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATA")
    print("=" * 70)
    
    X_list, y_list, session_names, usernames = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) == 0:
        print("\n❌ ERROR: No data found!")
        return 1
    
    # ===========================================
    # STEP 3: TRAIN/TEST SPLIT
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("=" * 70)
    
    if args.mode == 'optimize':
        # Use ALL data for training (maximize performance)
        X_train_cat, y_train_cat = concatenate_sessions(X_list, y_list)
        X_test_cat = X_train_cat  # Use same data for evaluation
        y_test_cat = y_train_cat
        test_user_indices = np.zeros(len(y_train_cat), dtype=int)
        
        print(f"\n✓ Using ALL data: {X_train_cat.shape[0]} samples")
        print("  (No train/test split - optimizing for maximum performance)")
    else:
        # Keep train/test separate (measure generalization)
        X_train_cat, y_train_cat, X_test_cat, y_test_cat, test_user_indices = train_test_split(
            X_list, y_list, session_names, usernames,
            test_size=TEST_SPLIT, 
            random_seed=RANDOM_SEED
        )
        
        print(f"\n✓ Train: {X_train_cat.shape[0]} samples")
        print(f"✓ Test:  {X_test_cat.shape[0]} samples (held out)")
    
    # ===========================================
    # STEP 4: COLLECT RESERVOIR STATES (SECOND PASS)
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 4: COLLECTING RESERVOIR STATES (PASS 2)")
    print("=" * 70)
    
    # Add noise to inputs for data augmentation (aiuta convergenza)
    print("Adding input noise for better convergence...")
    noise_scale = 0.01  # 1% noise
    X_train_noisy = X_train_cat + np.random.randn(*X_train_cat.shape) * noise_scale
    
    print(f"Running reservoir on training data (noise: {noise_scale*100:.1f}%)...")
    X_train_states = reservoir.forward(X_train_noisy, collect_states=True)
    
    print(f"✓ Collected {X_train_states.shape[0]} training states")
    
    # ===========================================
    # STEP 5: FINE-TUNE OUTPUT WEIGHTS
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 5: FINE-TUNING OUTPUT WEIGHTS (INCREMENTAL)")
    print("=" * 70)
    
    # Remove washout
    X_train_clean = X_train_states[WASHOUT:]
    y_train_clean = y_train_cat[WASHOUT:]
    
    if R_old is not None and P_old is not None:
        # Incremental update with learning rate
        print("Using incremental update with learning rate...")
        learning_rate = 0.3  # Peso per i nuovi dati (30% nuovi, 70% vecchi)
        W_out_new, R_new, P_new = compute_wout_incremental(
            X_train_clean, y_train_clean, W_out_old, R_old, P_old, 
            alpha=ALPHA, learning_rate=learning_rate
        )
        print(f"  Learning rate: {learning_rate:.2f}")
    else:
        # Gradient descent from old weights
        print("Using gradient descent from old weights...")
        learning_rate = 0.5  # Più aggressivo quando partiamo da zero
        W_out_new, R_new, P_new = compute_wout_incremental(
            X_train_clean, y_train_clean, W_out_old, None, None,
            alpha=ALPHA, learning_rate=learning_rate
        )
        print(f"  Learning rate: {learning_rate:.2f}")
    
    # Calculate improvement
    train_pred_old = X_train_clean @ W_out_old
    train_pred_new = X_train_clean @ W_out_new
    
    mse_old = np.mean((y_train_clean - train_pred_old) ** 2)
    mse_new = np.mean((y_train_clean - train_pred_new) ** 2)
    
    print(f"\n📊 Training MSE:")
    print(f"   Old model: {mse_old:.6f}")
    print(f"   New model: {mse_new:.6f}")
    print(f"   Improvement: {(mse_old - mse_new) / mse_old * 100:+.2f}%")
    
    # ===========================================
    # STEP 6: EVALUATE ON TEST SET
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATING ON TEST SET")
    print("=" * 70)
    
    # Test with old model
    X_test_states = reservoir.forward(X_test_cat, collect_states=True)
    y_test_pred_old = X_test_states @ W_out_old
    y_test_pred_new = X_test_states @ W_out_new
    
    metrics_old = compute_all_metrics(y_test_cat, y_test_pred_old)
    metrics_new = compute_all_metrics(y_test_cat, y_test_pred_new)
    
    print(f"\n{'Metric':<30} {'Old Model':<15} {'New Model':<15} {'Change':<10}")
    print("-" * 70)
    print(f"{'Direction RMSE':<30} {metrics_old['rmse_direction']:>14.4f} {metrics_new['rmse_direction']:>14.4f} "
          f"{(metrics_new['rmse_direction'] - metrics_old['rmse_direction']):>+9.4f}")
    print(f"{'Angular Error (deg)':<30} {metrics_old['angular_error_deg']:>14.2f} {metrics_new['angular_error_deg']:>14.2f} "
          f"{(metrics_new['angular_error_deg'] - metrics_old['angular_error_deg']):>+9.2f}")
    print(f"{'Boost Accuracy':<30} {metrics_old['boost_accuracy']*100:>13.2f}% {metrics_new['boost_accuracy']*100:>13.2f}% "
          f"{(metrics_new['boost_accuracy'] - metrics_old['boost_accuracy'])*100:>+8.2f}%")
    
    # ===========================================
    # STEP 7: SAVE FINE-TUNED MODEL
    # ===========================================
    print("\n" + "=" * 70)
    print("STEP 7: SAVING FINE-TUNED MODEL")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_PATH / f"finetuned_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    reservoir_data = {
        'W_in': reservoir.win,
        'W': reservoir.w,
        'W_out': W_out_new,
        'leak': reservoir.leak,
        'n_inputs': reservoir.n_inputs,
        'n_neurons': reservoir.n_neurons,
        'spectral_radius': reservoir.spectral_radius
    }
    model_path = output_dir / "reservoir_model.npz"
    np.savez(model_path, **reservoir_data)
    
    # Save training statistics for next fine-tuning
    stats_path = output_dir / "training_stats.npz"
    np.savez(stats_path, R=R_new, P=P_new)
    
    print(f"✓ Fine-tuned model saved to: {output_dir}")
    print(f"  - Model: {model_path.name}")
    print(f"  - Stats: {stats_path.name}")
    
    print("\n" + "=" * 70)
    print("✅ FINE-TUNING COMPLETED!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
