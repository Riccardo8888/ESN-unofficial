#!/usr/bin/env python3
"""
Train Slither.io Echo State Network
====================================

Main script to train an ESN to predict player actions from game state.

This script:
1. Loads Slither.io data from Zarr format
2. Prepares training and test sets
3. Trains an Echo State Network
4. Evaluates performance on both sets
5. Saves the trained model and results

Usage:
    python train_slither_esn.py
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Add utilities to path
sys.path.append(str(Path(__file__).parent))

from vnicktest.scripts.configuration import *
from utilities.data_loader import (
    load_all_data, 
    train_test_split, 
    concatenate_sessions
)
from utilities.esn_model import SlitherESN
from utilities.metrics import (
    compute_all_metrics, 
    print_metrics, 
    compare_metrics
)


def main():
    """Main training pipeline"""
    
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
        print(f"   Make sure you have played some games with the slither.io scraper active.")
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
    
    # Concatenate sessions
    X_train_cat, y_train_cat = concatenate_sessions(X_train, y_train)
    X_test_cat, y_test_cat = concatenate_sessions(X_test, y_test)
    
    print(f"\nData shapes:")
    print(f"  Training:   X={X_train_cat.shape}, y={y_train_cat.shape}")
    print(f"  Test:       X={X_test_cat.shape}, y={y_test_cat.shape}")
    print(f"  Input dim:  {X_train_cat.shape[1]}")
    print(f"  Output dim: {y_train_cat.shape[1]}")
    
    # ===========================================
    # STEP 3: CREATE ESN
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 3: CREATING ECHO STATE NETWORK")
    print("=" * 60)
    
    esn = SlitherESN(
        n_inputs=X_train_cat.shape[1],
        n_reservoir=N_RESERVOIR,
        n_outputs=y_train_cat.shape[1],
        spectral_radius=SPECTRAL_RADIUS,
        input_scale=INPUT_SCALE,
        leak_rate_range=(LEAK_RATE_MIN, LEAK_RATE_MAX),
        sparsity=SPARSITY,
        random_seed=RANDOM_SEED
    )
    
    print(f"\n{esn}")
    print(f"\nReservoir properties:")
    print(f"  Neurons: {esn.n_reservoir}")
    print(f"  Spectral radius: {esn.spectral_radius}")
    print(f"  Sparsity: {esn.sparsity}")
    print(f"  Leak rate range: [{esn.leak_rate_range[0]}, {esn.leak_rate_range[1]}]")
    
    # ===========================================
    # STEP 4: TRAIN ESN
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING ESN")
    print("=" * 60)
    
    train_stats = esn.train(
        X_train_cat, 
        y_train_cat, 
        alpha=ALPHA, 
        washout=WASHOUT,
        verbose=True
    )
    
    # ===========================================
    # STEP 5: EVALUATE ON TRAINING SET
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATING ON TRAINING SET")
    print("=" * 60)
    
    print(f"\nMaking predictions on training data...")
    y_train_pred = esn.predict(X_train_cat, washout=WASHOUT)
    y_train_true = y_train_cat[WASHOUT:]
    
    print(f"Computing metrics...")
    train_metrics = compute_all_metrics(y_train_true, y_train_pred)
    print_metrics(train_metrics, "Training")
    
    # ===========================================
    # STEP 6: EVALUATE ON TEST SET
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATING ON TEST SET")
    print("=" * 60)
    
    print(f"\nMaking predictions on test data...")
    y_test_pred = esn.predict(X_test_cat, washout=WASHOUT)
    y_test_true = y_test_cat[WASHOUT:]
    
    print(f"Computing metrics...")
    test_metrics = compute_all_metrics(y_test_true, y_test_pred)
    print_metrics(test_metrics, "Test")
    
    # ===========================================
    # STEP 7: COMPARE RESULTS
    # ===========================================
    compare_metrics(train_metrics, test_metrics)
    
    # ===========================================
    # STEP 8: SAVE RESULTS
    # ===========================================
    print("\n" + "=" * 60)
    print("STEP 8: SAVING RESULTS")
    print("=" * 60)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_PATH / f"training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "slither_esn_model.pkl"
    esn.save(model_path)
    
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
            'sparsity': SPARSITY,
            'alpha': ALPHA,
            'washout': WASHOUT,
            'test_split': TEST_SPLIT,
            'random_seed': RANDOM_SEED
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
        # Convert numpy types to native Python types for JSON serialization
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
