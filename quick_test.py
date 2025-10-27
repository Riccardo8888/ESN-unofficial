#!/usr/bin/env python3
"""
Quick Test Script for Slither.io ESN
=====================================

This script runs a quick test to verify the ESN system is working.
It uses a small subset of data for fast testing.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from configuration import *
from utilities.data_loader import load_all_data, train_test_split, concatenate_sessions
from utilities.esn_model import SlitherESN
from utilities.metrics import compute_all_metrics, print_metrics

def quick_test():
    """Run a quick test of the ESN system"""
    
    print("\n" + "=" * 60)
    print("SLITHER.IO ESN - QUICK TEST")
    print("=" * 60)
    
    # Test configuration
    print(f"\nChecking configuration...")
    print(f"  Data path: {SLITHER_DATA_PATH}")
    print(f"  Prediction horizon: {PREDICTION_HORIZON} frames")
    print(f"  Input dimension: {INPUT_DIM}")
    print(f"  Output dimension: {OUTPUT_DIM}")
    
    # Load data
    print(f"\nLoading data...")
    X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=False)
    
    if len(X_list) == 0:
        print("\n❌ No data found!")
        print(f"   Check: {SLITHER_DATA_PATH}")
        return 1
    
    print(f"  ✓ Found {len(X_list)} session(s)")
    
    # Use a small subset for quick testing
    if X_list[0].shape[0] > 500:
        print(f"\n  Using first 500 frames for quick test...")
        X_list = [X[:500] for X in X_list]
        y_list = [y[:500] for y in y_list]
    
    # Split data
    X_train, y_train, X_test, y_test, train_names, test_names = train_test_split(
        X_list, y_list, session_names
    )
    
    X_train_cat, y_train_cat = concatenate_sessions(X_train, y_train)
    X_test_cat, y_test_cat = concatenate_sessions(X_test, y_test)
    
    print(f"  Train: {X_train_cat.shape[0]} frames")
    print(f"  Test:  {X_test_cat.shape[0]} frames")
    
    # Create small ESN for quick testing
    print(f"\nCreating small ESN (200 neurons)...")
    esn = SlitherESN(
        n_inputs=X_train_cat.shape[1],
        n_reservoir=200,  # Small for quick test
        n_outputs=y_train_cat.shape[1],
        spectral_radius=1.25,
        input_scale=1.0,
        leak_rate_range=(0.1, 0.3),
        sparsity=0.9,
        random_seed=RANDOM_SEED
    )
    
    print(f"  {esn}")
    
    # Train
    print(f"\nTraining...")
    esn.train(X_train_cat, y_train_cat, alpha=ALPHA, washout=30, verbose=False)
    print(f"  ✓ Training complete")
    
    # Evaluate
    print(f"\nEvaluating...")
    
    # Train set
    y_train_pred = esn.predict(X_train_cat, washout=30)
    y_train_true = y_train_cat[30:]
    train_metrics = compute_all_metrics(y_train_true, y_train_pred)
    
    # Test set
    y_test_pred = esn.predict(X_test_cat, washout=30)
    y_test_true = y_test_cat[30:]
    test_metrics = compute_all_metrics(y_test_true, y_test_pred)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("QUICK TEST RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Training Set:")
    print(f"  Direction RMSE:  {train_metrics['rmse_direction']:.4f}")
    print(f"  Angular Error:   {train_metrics['angular_error_deg']:.2f}°")
    print(f"  Boost Accuracy:  {train_metrics['boost_accuracy']*100:.2f}%")
    
    print(f"\n📊 Test Set:")
    print(f"  Direction RMSE:  {test_metrics['rmse_direction']:.4f}")
    print(f"  Angular Error:   {test_metrics['angular_error_deg']:.2f}°")
    print(f"  Boost Accuracy:  {test_metrics['boost_accuracy']*100:.2f}%")
    
    # Check if results are reasonable
    print(f"\n✓ Quick test completed!")
    
    if test_metrics['angular_error_deg'] < 45:
        print(f"  ✓ Angular error is reasonable (<45°)")
    else:
        print(f"  ⚠ Angular error is high (>45°) - may need more data or tuning")
    
    if test_metrics['boost_accuracy'] > 0.7:
        print(f"  ✓ Boost accuracy is reasonable (>70%)")
    else:
        print(f"  ⚠ Boost accuracy is low (<70%) - may need more data or tuning")
    
    print(f"\nFor full training, run: python train_slither_esn.py")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = quick_test()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
