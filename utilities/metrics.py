"""
Evaluation Metrics for Slither.io ESN
======================================

Functions to evaluate ESN performance on predicting player actions.
"""

import numpy as np
from typing import Dict, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configuration import *


def compute_direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for direction prediction (mx, my).
    
    Args:
        y_true: True outputs [n_samples, 3] (mx, my, boost)
        y_pred: Predicted outputs [n_samples, 3]
        
    Returns:
        Dictionary with direction metrics
    """
    mx_true = y_true[:, 0]
    my_true = y_true[:, 1]
    mx_pred = y_pred[:, 0]
    my_pred = y_pred[:, 1]
    
    # Mean Squared Error for each component
    mse_mx = np.mean((mx_true - mx_pred) ** 2)
    mse_my = np.mean((my_true - my_pred) ** 2)
    mse_direction = (mse_mx + mse_my) / 2
    
    # Root Mean Squared Error
    rmse_mx = np.sqrt(mse_mx)
    rmse_my = np.sqrt(mse_my)
    rmse_direction = np.sqrt(mse_direction)
    
    # Mean Absolute Error
    mae_mx = np.mean(np.abs(mx_true - mx_pred))
    mae_my = np.mean(np.abs(my_true - my_pred))
    mae_direction = (mae_mx + mae_my) / 2
    
    # Angular error (angle between predicted and true direction vectors)
    # Normalize vectors
    norm_true = np.sqrt(mx_true**2 + my_true**2 + 1e-10)
    norm_pred = np.sqrt(mx_pred**2 + my_pred**2 + 1e-10)
    
    mx_true_norm = mx_true / norm_true
    my_true_norm = my_true / norm_true
    mx_pred_norm = mx_pred / norm_pred
    my_pred_norm = my_pred / norm_pred
    
    # Dot product for cosine similarity
    cos_sim = mx_true_norm * mx_pred_norm + my_true_norm * my_pred_norm
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Numerical stability
    
    # Angular error in radians
    angular_error = np.arccos(cos_sim)
    mean_angular_error = np.mean(angular_error)
    mean_angular_error_deg = np.degrees(mean_angular_error)
    
    return {
        'mse_mx': mse_mx,
        'mse_my': mse_my,
        'mse_direction': mse_direction,
        'rmse_mx': rmse_mx,
        'rmse_my': rmse_my,
        'rmse_direction': rmse_direction,
        'mae_mx': mae_mx,
        'mae_my': mae_my,
        'mae_direction': mae_direction,
        'angular_error_rad': mean_angular_error,
        'angular_error_deg': mean_angular_error_deg
    }


def compute_boost_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          threshold: float = BOOST_THRESHOLD) -> Dict[str, float]:
    """
    Compute metrics for boost prediction (binary classification).
    
    Args:
        y_true: True outputs [n_samples, 3] (mx, my, boost)
        y_pred: Predicted outputs [n_samples, 3]
        threshold: Threshold for classifying boost as active
        
    Returns:
        Dictionary with boost metrics
    """
    boost_true = y_true[:, 2]
    boost_pred = y_pred[:, 2]
    
    # Binarize predictions
    boost_pred_binary = (boost_pred >= threshold).astype(int)
    boost_true_binary = (boost_true >= threshold).astype(int)
    
    # Accuracy
    accuracy = np.mean(boost_pred_binary == boost_true_binary)
    
    # Confusion matrix elements
    tp = np.sum((boost_true_binary == 1) & (boost_pred_binary == 1))
    tn = np.sum((boost_true_binary == 0) & (boost_pred_binary == 0))
    fp = np.sum((boost_true_binary == 0) & (boost_pred_binary == 1))
    fn = np.sum((boost_true_binary == 1) & (boost_pred_binary == 0))
    
    # Precision, Recall, F1
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # MSE for continuous boost values
    mse_boost = np.mean((boost_true - boost_pred) ** 2)
    
    # Distribution statistics
    boost_true_mean = np.mean(boost_true)
    boost_pred_mean = np.mean(boost_pred)
    
    return {
        'boost_accuracy': accuracy,
        'boost_precision': precision,
        'boost_recall': recall,
        'boost_f1': f1,
        'boost_mse': mse_boost,
        'boost_true_mean': boost_true_mean,
        'boost_pred_mean': boost_pred_mean,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True outputs [n_samples, 3] (mx, my, boost)
        y_pred: Predicted outputs [n_samples, 3]
        
    Returns:
        Dictionary with all metrics
    """
    direction_metrics = compute_direction_metrics(y_true, y_pred)
    boost_metrics = compute_boost_metrics(y_true, y_pred)
    
    # Overall MSE
    overall_mse = np.mean((y_true - y_pred) ** 2)
    
    # Combine all metrics
    metrics = {
        'overall_mse': overall_mse,
        **direction_metrics,
        **boost_metrics
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Print metrics in a readable format.
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset (e.g., "Training", "Test")
    """
    print(f"\n{'=' * 60}")
    print(f"{dataset_name.upper()} SET METRICS")
    print("=" * 60)
    
    print(f"\n📊 Overall:")
    print(f"  MSE (all outputs): {metrics['overall_mse']:.6f}")
    
    print(f"\n🎯 Direction Prediction (mx, my):")
    print(f"  MSE:  mx={metrics['mse_mx']:.6f}, my={metrics['mse_my']:.6f}, avg={metrics['mse_direction']:.6f}")
    print(f"  RMSE: mx={metrics['rmse_mx']:.4f}, my={metrics['rmse_my']:.4f}, avg={metrics['rmse_direction']:.4f}")
    print(f"  MAE:  mx={metrics['mae_mx']:.4f}, my={metrics['mae_my']:.4f}, avg={metrics['mae_direction']:.4f}")
    print(f"  Angular Error: {metrics['angular_error_deg']:.2f}° ({metrics['angular_error_rad']:.4f} rad)")
    
    print(f"\n⚡ Boost Prediction (binary):")
    print(f"  Accuracy:  {metrics['boost_accuracy']:.4f} ({metrics['boost_accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['boost_precision']:.4f}")
    print(f"  Recall:    {metrics['boost_recall']:.4f}")
    print(f"  F1 Score:  {metrics['boost_f1']:.4f}")
    print(f"  MSE:       {metrics['boost_mse']:.6f}")
    
    print(f"\n📈 Boost Distribution:")
    print(f"  True mean:      {metrics['boost_true_mean']:.4f}")
    print(f"  Predicted mean: {metrics['boost_pred_mean']:.4f}")
    
    print(f"\n🔢 Confusion Matrix:")
    print(f"  TP: {metrics['tp']:.0f}  FN: {metrics['fn']:.0f}")
    print(f"  FP: {metrics['fp']:.0f}  TN: {metrics['tn']:.0f}")


def compare_metrics(train_metrics: Dict[str, float], 
                   test_metrics: Dict[str, float]):
    """
    Compare training and test metrics side by side.
    
    Args:
        train_metrics: Training set metrics
        test_metrics: Test set metrics
    """
    print(f"\n{'=' * 60}")
    print("TRAINING vs TEST COMPARISON")
    print("=" * 60)
    
    def print_comparison(name: str, train_val: float, test_val: float, 
                        format_str: str = "{:.6f}"):
        train_str = format_str.format(train_val)
        test_str = format_str.format(test_val)
        print(f"  {name:30s}  Train: {train_str:12s}  Test: {test_str:12s}")
    
    print(f"\n📊 Overall MSE:")
    print_comparison("MSE", train_metrics['overall_mse'], test_metrics['overall_mse'])
    
    print(f"\n🎯 Direction Metrics:")
    print_comparison("RMSE (direction)", train_metrics['rmse_direction'], 
                    test_metrics['rmse_direction'], "{:.4f}")
    print_comparison("MAE (direction)", train_metrics['mae_direction'], 
                    test_metrics['mae_direction'], "{:.4f}")
    print_comparison("Angular Error (deg)", train_metrics['angular_error_deg'], 
                    test_metrics['angular_error_deg'], "{:.2f}")
    
    print(f"\n⚡ Boost Metrics:")
    print_comparison("Accuracy", train_metrics['boost_accuracy'], 
                    test_metrics['boost_accuracy'], "{:.4f}")
    print_comparison("F1 Score", train_metrics['boost_f1'], 
                    test_metrics['boost_f1'], "{:.4f}")
    
    # Check for overfitting
    print(f"\n🔍 Overfitting Analysis:")
    mse_ratio = test_metrics['overall_mse'] / train_metrics['overall_mse']
    acc_diff = train_metrics['boost_accuracy'] - test_metrics['boost_accuracy']
    
    print(f"  Test/Train MSE ratio: {mse_ratio:.3f}", end="")
    if mse_ratio < 1.2:
        print(" ✓ (good generalization)")
    elif mse_ratio < 1.5:
        print(" ~ (acceptable)")
    else:
        print(" ⚠ (possible overfitting)")
    
    print(f"  Train-Test Accuracy diff: {acc_diff:.4f}", end="")
    if acc_diff < 0.05:
        print(" ✓ (good generalization)")
    elif acc_diff < 0.10:
        print(" ~ (acceptable)")
    else:
        print(" ⚠ (possible overfitting)")


if __name__ == "__main__":
    """Test metrics computation"""
    print_config()
    
    print("\n" + "=" * 60)
    print("TESTING METRICS")
    print("=" * 60)
    
    # Create dummy predictions
    n_samples = 1000
    y_true = np.random.randn(n_samples, 3)
    y_true[:, 2] = (np.random.rand(n_samples) > 0.7).astype(float)  # 30% boost active
    
    # Simulate predictions with some noise
    y_pred = y_true + np.random.randn(n_samples, 3) * 0.2
    y_pred[:, 2] = np.clip(y_pred[:, 2], 0, 1)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_pred)
    print_metrics(metrics, "Dummy Test")
    
    # Test comparison
    train_metrics = compute_all_metrics(y_true, y_pred * 0.95)
    test_metrics = compute_all_metrics(y_true, y_pred * 1.1)
    compare_metrics(train_metrics, test_metrics)
