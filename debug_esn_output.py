#!/usr/bin/env python3
"""
Analisi Output ESN - Debug Predictions
=======================================
Testa il modello con dati reali per vedere le predizioni.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from configuration import *
from reservoir import Reservoir

# Load model
model_path = Path("slither_esn_results/training_20251027_210927/reservoir_model.npz")
print(f"Loading model: {model_path}")
model_data = np.load(model_path)

W_in = model_data['W_in']
W = model_data['W']
W_out = model_data['W_out']
leak = model_data['leak']

# Create reservoir
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

print(f"Model loaded: {reservoir.n_neurons} neurons\n")

# Load real test data
from utilities.data_loader import load_all_data, train_test_split

X_list, y_list, session_names, usernames = load_all_data(SLITHER_DATA_PATH, verbose=False)
X_train_cat, y_train_cat, X_test_cat, y_test_cat, test_user_indices = train_test_split(
    X_list, y_list, session_names, usernames,
    test_size=TEST_SPLIT, 
    random_seed=RANDOM_SEED
)

print(f"Loaded test data: {X_test_cat.shape[0]} frames\n")

# Test predictions
print("=" * 80)
print("ANALYZING ESN OUTPUT ON REAL TEST DATA")
print("=" * 80)

# Sample 100 random frames
n_samples = 100
indices = np.random.choice(X_test_cat.shape[0], n_samples, replace=False)

# Initialize reservoir state
x = np.zeros(reservoir.n_neurons)

# Statistics
boost_predictions = []
boost_truth = []
angle_deltas = []

print(f"\nSample Predictions (first 20):")
print(f"{'Frame':<8} {'True mx':<10} {'Pred mx':<10} {'True my':<10} {'Pred my':<10} {'True B':<8} {'Pred B':<10} {'Angle Δ':<10}")
print("-" * 90)

for i, idx in enumerate(indices):
    # Get input and target
    features = X_test_cat[idx]
    target = y_test_cat[idx]
    
    # Predict
    x = reservoir.update(features, x)
    prediction = x @ W_out
    
    mx_pred, my_pred, boost_pred = prediction
    mx_true, my_true, boost_true = target
    
    # Calculate angles
    angle_true = np.arctan2(my_true, mx_true)
    angle_pred = np.arctan2(my_pred, mx_pred)
    angle_delta = np.degrees(angle_pred - angle_true)
    
    # Boost classification
    boost_pred_binary = int(boost_pred > 0.5)
    boost_true_binary = int(boost_true > 0.5)
    
    boost_predictions.append(boost_pred)
    boost_truth.append(boost_true_binary)
    angle_deltas.append(angle_delta)
    
    # Print first 20
    if i < 20:
        print(f"{i:<8} {mx_true:>9.3f} {mx_pred:>9.3f} {my_true:>9.3f} {my_pred:>9.3f} "
              f"{boost_true_binary:>7} {boost_pred:>9.3f} {angle_delta:>9.1f}°")

# Statistics
print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)

boost_predictions = np.array(boost_predictions)
boost_truth = np.array(boost_truth)
angle_deltas = np.array(angle_deltas)

print(f"\nBoost Predictions:")
print(f"  Mean boost_pred:  {boost_predictions.mean():.3f}")
print(f"  Std boost_pred:   {boost_predictions.std():.3f}")
print(f"  Min boost_pred:   {boost_predictions.min():.3f}")
print(f"  Max boost_pred:   {boost_predictions.max():.3f}")
print(f"\n  Distribution:")
print(f"    < 0.3: {np.sum(boost_predictions < 0.3):3d} ({np.sum(boost_predictions < 0.3)/len(boost_predictions)*100:.1f}%)")
print(f"    0.3-0.5: {np.sum((boost_predictions >= 0.3) & (boost_predictions < 0.5)):3d} ({np.sum((boost_predictions >= 0.3) & (boost_predictions < 0.5))/len(boost_predictions)*100:.1f}%)")
print(f"    0.5-0.7: {np.sum((boost_predictions >= 0.5) & (boost_predictions < 0.7)):3d} ({np.sum((boost_predictions >= 0.5) & (boost_predictions < 0.7))/len(boost_predictions)*100:.1f}%)")
print(f"    > 0.7: {np.sum(boost_predictions > 0.7):3d} ({np.sum(boost_predictions > 0.7)/len(boost_predictions)*100:.1f}%)")

print(f"\nBoost Ground Truth:")
print(f"  Boost ON:  {np.sum(boost_truth):3d} ({np.sum(boost_truth)/len(boost_truth)*100:.1f}%)")
print(f"  Boost OFF: {np.sum(boost_truth == 0):3d} ({np.sum(boost_truth == 0)/len(boost_truth)*100:.1f}%)")

boost_pred_binary = (boost_predictions > 0.5).astype(int)
boost_accuracy = np.mean(boost_pred_binary == boost_truth)
print(f"\nBoost Accuracy: {boost_accuracy*100:.2f}%")

print(f"\nAngle Deltas:")
print(f"  Mean: {angle_deltas.mean():.1f}°")
print(f"  Std:  {angle_deltas.std():.1f}°")
print(f"  Median: {np.median(angle_deltas):.1f}°")
print(f"  |Mean|: {np.abs(angle_deltas).mean():.1f}°")

print(f"\n⚠️  DIAGNOSIS:")
if boost_predictions.mean() < 0.3:
    print(f"  ❌ Model NEVER predicts boost (mean={boost_predictions.mean():.3f})")
    print(f"     Problem: Model biased towards no-boost")
elif boost_predictions.mean() > 0.7:
    print(f"  ❌ Model ALWAYS predicts boost (mean={boost_predictions.mean():.3f})")
    print(f"     Problem: Model biased towards boost")
elif boost_predictions.std() < 0.1:
    print(f"  ⚠️  Model outputs very low variance (std={boost_predictions.std():.3f})")
    print(f"     Problem: Model not confident in decisions")
else:
    print(f"  ✅ Boost predictions look reasonable")

if np.abs(angle_deltas).mean() > 50:
    print(f"  ❌ Angle predictions very inaccurate (|mean|={np.abs(angle_deltas).mean():.1f}°)")
else:
    print(f"  ✅ Angle predictions reasonable (|mean|={np.abs(angle_deltas).mean():.1f}°)")
