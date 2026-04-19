#!/usr/bin/env python3
"""
Angle-Aware ESN Training
========================

Invece di predire (mx, my) direttamente, predice:
1. Delta heading (cambio angolare)
2. Speed magnitude
3. Boost

Questo migliora la predizione angolare perché:
- Gli angoli sono circolari (359° → 0°)
- Il modello impara cambi relativi invece di valori assoluti
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from vnicktest.scripts.configuration import *
from utilities.data_loader import load_all_data


def convert_to_angle_representation(y: np.ndarray, headings: np.ndarray,
                                   horizon: int = PREDICTION_HORIZON) -> np.ndarray:
    """
    Convert (mx, my, boost) to (delta_heading, speed, boost)
    
    Args:
        y: Original outputs [n_samples, 3] (mx, my, boost)
        headings: Current headings [n_samples]
        horizon: Prediction horizon
        
    Returns:
        y_angle: [n_samples, 3] (delta_heading_rad, speed, boost)
    """
    mx = y[:, 0]
    my = y[:, 1]
    boost = y[:, 2]
    
    # Calcola angolo target da (mx, my)
    target_angle = np.arctan2(my, mx)
    
    # Calcola delta heading (cambio angolare)
    # Shift headings by horizon to align with targets
    current_heading = headings[:-horizon] if horizon > 0 else headings
    target_heading = headings[horizon:] if horizon > 0 else headings
    
    # Calcola delta con wrapping [-π, π]
    delta = target_heading - current_heading
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    
    # Calcola speed magnitude
    speed = np.sqrt(mx**2 + my**2)
    
    # Stack nuovo target
    y_angle = np.column_stack([delta, speed, boost])
    
    return y_angle


def convert_from_angle_representation(y_pred: np.ndarray, current_heading: float) -> tuple:
    """
    Convert (delta_heading, speed, boost) back to (mx, my, boost)
    
    Args:
        y_pred: Predicted [3] (delta_heading_rad, speed, boost)
        current_heading: Current snake heading in radians
        
    Returns:
        mx, my, boost
    """
    delta_heading = y_pred[0]
    speed = y_pred[1]
    boost = y_pred[2]
    
    # Calcola nuovo heading
    new_heading = current_heading + delta_heading
    
    # Convert to (mx, my)
    mx = speed * np.cos(new_heading)
    my = speed * np.sin(new_heading)
    
    return mx, my, boost


def main():
    print("\n" + "=" * 70)
    print("ANGLE-AWARE ESN TRAINING")
    print("=" * 70)
    print("\n⚡ Using angle-relative representation:")
    print("   Output: (Δheading, speed, boost) instead of (mx, my, boost)")
    print("   Benefit: Better angular prediction with circular wrapping")
    
    # Load data
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    X_list, y_list, session_names, usernames = load_all_data(SLITHER_DATA_PATH, verbose=True)
    
    if len(X_list) == 0:
        print("\n❌ ERROR: No data found!")
        return 1
    
    # Extract headings for angle conversion (need raw data)
    print("\n🔄 Converting to angle-aware representation...")
    # TODO: Need to modify data_loader to also return headings
    # For now, this is a conceptual implementation
    
    print("\n✅ Angle-aware training framework created!")
    print("   Next step: Modify data_loader to return headings array")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
