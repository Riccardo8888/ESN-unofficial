#!/usr/bin/env python3
"""
Training ESN per Slither.io usando Brain Connectome Reservoir

Questo script usa brain_connectome_reservoir.py invece di reservoir.py:
- Usa struttura del connettoma cerebrale invece di matrice random
- Più biologicamente plausibile
- Potenzialmente migliori prestazioni

Autori: Nick & Riccardo
Data: 27 Ottobre 2025
"""

import sys
import os
import glob
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from configuration import *
from utilities.data_loader import load_all_data, train_test_split, create_sequences_with_horizon
from utilities.metrics import (
    compute_direction_metrics,
    compute_boost_metrics,
    plot_predictions,
    save_results
)
from brain_connectome_reservoir import ConnectomeReservoir

warnings.filterwarnings('ignore')

def compute_wout(X, Y, alpha=1e-3):
    """
    Compute output weights using ridge regression.
    
    Args:
        X: Reservoir states [n_samples, n_neurons]
        Y: Target outputs [n_samples, n_outputs]
        alpha: Regularization parameter
    
    Returns:
        Wout: Output weights [n_neurons, n_outputs]
    """
    # Ridge regression: Wout = (X^T X + alpha*I)^-1 X^T Y
    R = X.T @ X
    P = X.T @ Y
    Wout = np.linalg.solve(R + alpha * np.eye(X.shape[1]), P)
    return Wout


def check_brain_connectome_available():
    """Check if brain connectome files are available."""
    base_dir = Path(__file__).parent
    graphml_files = list(base_dir.glob("*.graphml"))
    
    if not graphml_files:
        print("\n" + "="*80)
        print("⚠️  ATTENZIONE: Nessun file brain connectome (.graphml) trovato!")
        print("="*80)
        print("\nPer usare il brain connectome reservoir, hai bisogno di file .graphml")
        print("che contengono la struttura del connettoma cerebrale.")
        print("\nOpzioni:")
        print("  1. Scarica brain connectome dataset (es. C. elegans, Human Connectome)")
        print("  2. Genera grafi sintetici che simulano strutture cerebrali")
        print("  3. Usa il fallback a reservoir random (meno interessante)")
        print("\nPer ora questo script terminerà. Usa 'train_slither_reservoir.py' oppure")
        print("'train_slither_esn.py' se vuoi usare reservoir random classici.")
        print("="*80 + "\n")
        return False
    
    print(f"\n✅ Trovati {len(graphml_files)} file brain connectome:")
    for f in graphml_files[:5]:  # Show first 5
        print(f"   - {f.name}")
    if len(graphml_files) > 5:
        print(f"   ... e altri {len(graphml_files) - 5} file")
    print()
    return True


def train_connectome_esn(
    X_train, Y_train, X_test, Y_test,
    n_inputs, graph_dir,
    n_reservoir=1000,
    spectral_radius=1.25,
    leak_range=(0.1, 0.3),
    alpha=1e-3,
    washout=50,
    seed=42,
    verbose=True
):
    """
    Train ESN using brain connectome reservoir.
    
    Args:
        X_train: Training input sequences [n_samples, seq_len, n_features]
        Y_train: Training targets [n_samples, n_outputs]
        X_test: Test input sequences
        Y_test: Test targets
        n_inputs: Number of input features
        graph_dir: Directory containing .graphml brain connectome files
        n_reservoir: Target number of neurons (will resize brain graph)
        spectral_radius: Spectral radius for W matrix
        leak_range: Range for leak rates
        alpha: Ridge regression regularization
        washout: Number of initial timesteps to discard
        seed: Random seed
        verbose: Print progress
    
    Returns:
        results: Dictionary with training results
    """
    if verbose:
        print("\n" + "="*80)
        print("🧠 TRAINING CON BRAIN CONNECTOME RESERVOIR")
        print("="*80)
        print(f"\n📊 Configurazione:")
        print(f"   Brain connectome dir: {graph_dir}")
        print(f"   Target reservoir size: {n_reservoir}")
        print(f"   Spectral radius: {spectral_radius}")
        print(f"   Leak range: {leak_range}")
        print(f"   Alpha (regularization): {alpha}")
        print(f"   Washout: {washout}")
        print(f"   Random seed: {seed}")
        
    # Create brain connectome reservoir
    if verbose:
        print(f"\n🧬 Creazione Brain Connectome Reservoir...")
    
    try:
        reservoir = ConnectomeReservoir(
            n_inputs=n_inputs,
            graph_dir=str(graph_dir),
            target_size=n_reservoir,
            rhow=spectral_radius,
            leak_range=leak_range,
            seed=seed,
            combine='mean',  # Combine multiple connectomes by averaging
            symmetric=True,  # Make connections symmetric
            edge_attr='weight'  # Use edge weights if available
        )
        
        actual_neurons = reservoir.n_neurons
        if verbose:
            print(f"   ✓ Reservoir creato con {actual_neurons} neuroni (basato su brain connectome)")
            print(f"   ✓ Spectral radius: {spectral_radius}")
            
    except Exception as e:
        print(f"\n❌ Errore nella creazione del brain connectome reservoir:")
        print(f"   {str(e)}")
        print("\nVerifica che i file .graphml siano nel formato corretto.")
        raise
    
    # Flatten training sequences for state collection
    n_train = len(X_train)
    seq_len = X_train.shape[1]
    
    if verbose:
        print(f"\n📈 Training set:")
        print(f"   Samples: {n_train}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Features: {n_inputs}")
        print(f"   Total timesteps: {n_train * seq_len}")
        print(f"   Washout per sequence: {washout}")
        print(f"   Usable timesteps: {n_train * (seq_len - washout)}")
    
    # Collect reservoir states from all training sequences
    if verbose:
        print(f"\n🔄 Raccolta stati del reservoir...")
    
    all_states = []
    all_targets = []
    
    for i in range(n_train):
        if verbose and (i + 1) % 100 == 0:
            print(f"   Processate {i+1}/{n_train} sequenze...")
        
        # Run reservoir forward
        X_seq = reservoir.forward(X_train[i], collect_states=True)
        
        # Discard washout period
        X_seq = X_seq[washout:]
        
        # Repeat target for all timesteps after washout
        Y_seq = np.repeat(Y_train[i:i+1], len(X_seq), axis=0)
        
        all_states.append(X_seq)
        all_targets.append(Y_seq)
    
    # Concatenate all sequences
    X_train_collected = np.vstack(all_states)
    Y_train_collected = np.vstack(all_targets)
    
    if verbose:
        print(f"   ✓ Stati raccolti: {X_train_collected.shape}")
        print(f"   ✓ Target: {Y_train_collected.shape}")
    
    # Train output weights with ridge regression
    if verbose:
        print(f"\n🎓 Training output weights (Ridge regression, alpha={alpha})...")
    
    wout = compute_wout(X_train_collected, Y_train_collected, alpha=alpha)
    
    if verbose:
        print(f"   ✓ Wout shape: {wout.shape}")
    
    # Evaluate on training set
    if verbose:
        print(f"\n📊 Valutazione su training set...")
    
    train_predictions = []
    train_targets = []
    
    for i in range(n_train):
        X_seq = reservoir.forward(X_train[i], collect_states=True)
        X_seq = X_seq[washout:]
        
        # Predict
        Y_pred = X_seq @ wout
        
        # Use only last prediction (corresponds to target at end of sequence)
        train_predictions.append(Y_pred[-1])
        train_targets.append(Y_train[i])
    
    train_predictions = np.array(train_predictions)
    train_targets = np.array(train_targets)
    
    # Compute training metrics
    train_dir_metrics = compute_direction_metrics(
        train_targets[:, :2], train_predictions[:, :2]
    )
    train_boost_metrics = compute_boost_metrics(
        train_targets[:, 2], train_predictions[:, 2]
    )
    
    if verbose:
        print(f"\n✅ Training Results:")
        print(f"   Direction RMSE: {train_dir_metrics['rmse']:.4f}")
        print(f"   Direction MAE: {train_dir_metrics['mae']:.4f}")
        print(f"   Angular Error: {train_dir_metrics['angular_error_mean']:.2f}°")
        print(f"   Boost Accuracy: {train_boost_metrics['accuracy']:.2%}")
        print(f"   Boost F1 Score: {train_boost_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    n_test = len(X_test)
    
    if verbose:
        print(f"\n📊 Valutazione su test set ({n_test} samples)...")
    
    test_predictions = []
    test_targets = []
    
    for i in range(n_test):
        X_seq = reservoir.forward(X_test[i], collect_states=True)
        X_seq = X_seq[washout:]
        
        # Predict
        Y_pred = X_seq @ wout
        
        # Use only last prediction
        test_predictions.append(Y_pred[-1])
        test_targets.append(Y_test[i])
    
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    # Compute test metrics
    test_dir_metrics = compute_direction_metrics(
        test_targets[:, :2], test_predictions[:, :2]
    )
    test_boost_metrics = compute_boost_metrics(
        test_targets[:, 2], test_predictions[:, 2]
    )
    
    if verbose:
        print(f"\n✅ Test Results:")
        print(f"   Direction RMSE: {test_dir_metrics['rmse']:.4f}")
        print(f"   Direction MAE: {test_dir_metrics['mae']:.4f}")
        print(f"   Angular Error: {test_dir_metrics['angular_error_mean']:.2f}°")
        print(f"   Boost Accuracy: {test_boost_metrics['accuracy']:.2%}")
        print(f"   Boost F1 Score: {test_boost_metrics['f1_score']:.4f}")
    
    # Check for overfitting
    if verbose:
        print(f"\n🔍 Analisi Overfitting:")
        train_mse = train_dir_metrics['rmse'] ** 2
        test_mse = test_dir_metrics['rmse'] ** 2
        ratio = test_mse / train_mse if train_mse > 0 else float('inf')
        
        print(f"   Train MSE: {train_mse:.4f}")
        print(f"   Test MSE: {test_mse:.4f}")
        print(f"   Test/Train ratio: {ratio:.3f}", end="")
        
        if ratio > 2.0:
            print(" ⚠️  OVERFITTING FORTE")
        elif ratio > 1.5:
            print(" ⚠️  Overfitting medio")
        elif ratio > 1.2:
            print(" ✓ Lieve overfitting (normale)")
        else:
            print(" ✅ Ottimo!")
        
        acc_diff = train_boost_metrics['accuracy'] - test_boost_metrics['accuracy']
        print(f"   Accuracy difference: {acc_diff:.4f}", end="")
        
        if acc_diff > 0.2:
            print(" ⚠️  OVERFITTING FORTE")
        elif acc_diff > 0.1:
            print(" ⚠️  Overfitting medio")
        else:
            print(" ✓ Buono")
    
    # Return results
    results = {
        'reservoir': reservoir,
        'wout': wout,
        'train_predictions': train_predictions,
        'train_targets': train_targets,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'train_metrics': {
            'direction': train_dir_metrics,
            'boost': train_boost_metrics
        },
        'test_metrics': {
            'direction': test_dir_metrics,
            'boost': test_boost_metrics
        },
        'config': {
            'n_reservoir': actual_neurons,
            'spectral_radius': spectral_radius,
            'leak_range': leak_range,
            'alpha': alpha,
            'washout': washout,
            'seed': seed,
            'graph_dir': str(graph_dir),
            'reservoir_type': 'brain_connectome'
        }
    }
    
    return results


def main():
    """Main training function."""
    print("\n" + "="*80)
    print("🐍 SLITHER.IO ESN TRAINING - BRAIN CONNECTOME RESERVOIR")
    print("="*80)
    
    # Check if brain connectome files are available
    base_dir = Path(__file__).parent
    if not check_brain_connectome_available():
        print("\n💡 Suggerimento: Usa 'hyperparameter_search.py' per ottimizzare")
        print("   i parametri con il reservoir che hai disponibile.")
        return 1
    
    # Load data
    print(f"\n📁 Caricamento dati da: {SLITHER_DATA_PATH}")
    
    try:
        X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=True)
    except Exception as e:
        print(f"\n❌ Errore nel caricamento dati: {e}")
        return 1
    
    # Split train/test
    print(f"\n🔀 Split train/test (test_size={TEST_SPLIT})...")
    from utilities.data_loader import train_test_split as split_sessions
    X_train, y_train, X_test, y_test, train_names, test_names = split_sessions(
        X_list, y_list, session_names, test_size=TEST_SPLIT, random_seed=RANDOM_SEED
    )
    
    print(f"   ✓ Training sessions: {len(X_train)}")
    print(f"   ✓ Test sessions: {len(X_test)}")
    
    # Concatenate sessions
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)
    
    # Reshape to sequences [n_samples, seq_len, features]
    # Each sample is already a time sequence from load_all_data
    # We need to add time dimension for reservoir processing
    print(f"\n🔄 Preparazione sequenze per reservoir...")
    
    # Group by original session length to maintain temporal structure
    # For now, treat each sample as a 1-frame sequence
    X_train = X_train.reshape(len(X_train), 1, -1)
    X_test = X_test.reshape(len(X_test), 1, -1)
    
    print(f"   ✓ X_train shape: {X_train.shape}")
    print(f"   ✓ y_train shape: {y_train.shape}")
    print(f"   ✓ X_test shape: {X_test.shape}")
    print(f"   ✓ y_test shape: {y_test.shape}")
    
    n_inputs = X_train.shape[2]
    
    # Train ESN with brain connectome
    results = train_connectome_esn(
        X_train, y_train, X_test, y_test,
        n_inputs=n_inputs,
        graph_dir=base_dir,  # Look for .graphml files in project root
        n_reservoir=N_RESERVOIR,
        spectral_radius=SPECTRAL_RADIUS,
        leak_range=(0.1, 0.3),  # Default leak range
        alpha=ALPHA,
        washout=WASHOUT,
        seed=RANDOM_SEED,
        verbose=True
    )
    
    # Save results
    output_dir = Path(__file__).parent / "slither_esn_results" / f"connectome_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Salvataggio risultati in: {output_dir}")
    
    # Save model
    np.save(output_dir / "wout.npy", results['wout'])
    np.save(output_dir / "config.npy", results['config'])
    
    # Plot results
    plot_predictions(
        results['test_targets'][:100],  # First 100 samples
        results['test_predictions'][:100],
        output_dir / "predictions_test.png"
    )
    
    # Save metrics
    save_results(results, output_dir / "metrics.json")
    
    print(f"   ✓ Model salvato")
    print(f"   ✓ Plots salvati")
    print(f"   ✓ Metrics salvati")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETATO CON SUCCESSO!")
    print("="*80)
    print(f"\nRisultati salvati in: {output_dir}")
    print("\n💡 Prossimi passi:")
    print("   1. Analizza i grafici in slither_esn_results/")
    print("   2. Se c'è overfitting, prova 'hyperparameter_search.py'")
    print("   3. Confronta con reservoir random usando 'compare_reservoirs.py'")
    print("   4. Raccogli più dati per migliorare le prestazioni")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
