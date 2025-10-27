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

from configuration import *
from utilities.data_loader import (
    load_all_data, 
    train_test_split, 
    concatenate_sessions
)
from reservoir import Reservoir  # Usa il reservoir esistente!
from utilities.metrics import (
    compute_all_metrics, 
    print_metrics, 
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
    W_out = np.linalg.inv(R + alpha * np.eye(X_train.shape[1])) @ P
    
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
    
    X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) == 0:
        print("\n❌ ERROR: No data found!")
        print(f"   Please check that data exists in: {SLITHER_DATA_PATH}")
        print(f"   This should be the 'data/' folder in this workspace")
        return 1
    
    # ===========================================
    # STEP 2: TRAIN/TEST SPLIT
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN/TEST SPLIT")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test, train_names, test_names = train_test_split(
        X_list, y_list, session_names, 
        test_size=TEST_SPLIT, 
        random_seed=RANDOM_SEED
    )
    
    print(f"\nTraining sessions ({len(X_train)}):")
    for name in train_names:
        print(f"  - {name}")
    
    print(f"\nTest sessions ({len(X_test)}):")
    for name in test_names:
        print(f"  - {name}")
    
    # Concatenate sessions (come in humand_data.ipynb)
    X_train_cat, y_train_cat = concatenate_sessions(X_train, y_train)
    X_test_cat, y_test_cat = concatenate_sessions(X_test, y_test)
    
    print(f"\nData shapes:")
    print(f"  Training:   X={X_train_cat.shape}, y={y_train_cat.shape}")
    print(f"  Test:       X={X_test_cat.shape}, y={y_test_cat.shape}")
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
    X_train_states = reservoir.forward(X_train_cat, collect_states=True)
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
    X_test_states = reservoir.forward(X_test_cat, collect_states=True)
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
            'approach': 'reservoir.py (same as humand_data.ipynb)'
        },
        'data': {
            'train_sessions': train_names,
            'test_sessions': test_names,
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
    print(f"  Direction RMSE (test):  {test_metrics['rmse_direction']:.4f}")
    print(f"  Angular Error (test):   {test_metrics['angular_error_deg']:.2f}°")
    print(f"  Boost Accuracy (test):  {test_metrics['boost_accuracy']*100:.2f}%")
    print(f"  Overall MSE (test):     {test_metrics['overall_mse']:.6f}")
    
    print(f"\n📁 Output Directory: {output_dir}")
    
    print(f"\n📝 Note: Questo script usa lo stesso approccio di humand_data.ipynb:")
    print(f"   - reservoir.forward() per raccogliere stati")
    print(f"   - compute_wout() per training con ridge regression")
    print(f"   - X @ W_out per predizioni")
    
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
