#!/usr/bin/env python3
"""
Confronto tra Reservoir Random e Brain Connectome Reservoir

Questo script addestra ESN con entrambi i tipi di reservoir e confronta:
- Prestazioni (accuracy, angular error)
- Overfitting (train vs test metrics)
- Velocità di training
- Stabilità delle predizioni

Autori: Nick & Riccardo
Data: 27 Ottobre 2025
"""

import sys
import os
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime
import json
import time
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from configuration import *
from utilities.data_loader import load_all_data, train_test_split, create_sequences_with_horizon
from utilities.metrics import compute_direction_metrics, compute_boost_metrics
from reservoir import Reservoir
from brain_connectome_reservoir import ConnectomeReservoir

warnings.filterwarnings('ignore')


def compute_wout(X, Y, alpha=1e-3):
    """Ridge regression to compute output weights."""
    R = X.T @ X
    P = X.T @ Y
    Wout = np.linalg.solve(R + alpha * np.eye(X.shape[1]), P)
    return Wout


def train_random_reservoir(
    X_train, Y_train, X_test, Y_test,
    n_inputs, config, verbose=True
):
    """Train ESN with random reservoir."""
    if verbose:
        print("\n" + "="*80)
        print("🎲 RANDOM RESERVOIR")
        print("="*80)
    
    start_time = time.time()
    
    # Create reservoir
    reservoir = Reservoir(
        n_inputs=n_inputs,
        n_neurons=config['n_reservoir'],
        rhow=config['spectral_radius'],
        inp_scaling=1.0,
        leak_range=config['leak_range'],
        verbose=False
    )
    
    # Collect training states
    all_states = []
    all_targets = []
    
    for i in range(len(X_train)):
        X_seq = reservoir.forward(X_train[i], collect_states=True)
        X_seq = X_seq[config['washout']:]
        Y_seq = np.repeat(Y_train[i:i+1], len(X_seq), axis=0)
        all_states.append(X_seq)
        all_targets.append(Y_seq)
    
    X_train_states = np.vstack(all_states)
    Y_train_targets = np.vstack(all_targets)
    
    # Train
    wout = compute_wout(X_train_states, Y_train_targets, alpha=config['alpha'])
    
    train_time = time.time() - start_time
    
    # Evaluate
    train_preds, train_targets = [], []
    for i in range(len(X_train)):
        X_seq = reservoir.forward(X_train[i], collect_states=True)
        if len(X_seq) <= config['washout']:
            continue
        X_seq = X_seq[config['washout']:]
        Y_seq = Y_train[i][config['washout']:]
        Y_pred = X_seq @ wout
        train_preds.append(Y_pred)
        train_targets.append(Y_seq)
    
    test_preds, test_targets = [], []
    for i in range(len(X_test)):
        X_seq = reservoir.forward(X_test[i], collect_states=True)
        if len(X_seq) <= config['washout']:
            continue
        X_seq = X_seq[config['washout']:]
        Y_seq = Y_test[i][config['washout']:]
        Y_pred = X_seq @ wout
        test_preds.append(Y_pred)
        test_targets.append(Y_seq)
    
    if not train_preds or not test_preds:
        raise ValueError("All sessions too short after washout!")
    
    train_preds = np.vstack(train_preds)
    train_targets = np.vstack(train_targets)
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # Metrics
    train_dir = compute_direction_metrics(train_targets[:, :2], train_preds[:, :2])
    train_boost = compute_boost_metrics(train_targets, train_preds)  # Pass full array
    test_dir = compute_direction_metrics(test_targets[:, :2], test_preds[:, :2])
    test_boost = compute_boost_metrics(test_targets, test_preds)  # Pass full array
    
    if verbose:
        print(f"\n⏱️  Training time: {train_time:.2f}s")
        print(f"\n📊 Train: Boost Acc={train_boost['accuracy']:.2%}, Angular={train_dir['angular_error_deg']:.1f}°")
        print(f"📊 Test:  Boost Acc={test_boost['accuracy']:.2%}, Angular={test_dir['angular_error_deg']:.1f}°")
    
    return {
        'reservoir_type': 'random',
        'train_time': train_time,
        'wout': wout,
        'train_metrics': {'direction': train_dir, 'boost': train_boost},
        'test_metrics': {'direction': test_dir, 'boost': test_boost},
        'train_predictions': train_preds,
        'test_predictions': test_preds
    }


def train_connectome_reservoir(
    X_train, Y_train, X_test, Y_test,
    n_inputs, graph_dir, config, verbose=True
):
    """Train ESN with brain connectome reservoir."""
    if verbose:
        print("\n" + "="*80)
        print("🧠 BRAIN CONNECTOME RESERVOIR")
        print("="*80)
    
    start_time = time.time()
    
    try:
        # Create reservoir
        reservoir = ConnectomeReservoir(
            n_inputs=n_inputs,
            graph_dir=str(graph_dir),
            target_size=config['n_reservoir'],
            rhow=config['spectral_radius'],
            leak_range=config['leak_range'],
            seed=config['seed'],
            combine='mean',
            symmetric=True,
            edge_attr='weight'
        )
        
        # Collect training states
        all_states = []
        all_targets = []
        
        for i in range(len(X_train)):
            X_seq = reservoir.forward(X_train[i], collect_states=True)
            if len(X_seq) <= config['washout']:
                continue
            X_seq = X_seq[config['washout']:]
            Y_seq = Y_train[i][config['washout']:]
            all_states.append(X_seq)
            all_targets.append(Y_seq)
        
        X_train_states = np.vstack(all_states)
        Y_train_targets = np.vstack(all_targets)
        
        # Train
        wout = compute_wout(X_train_states, Y_train_targets, alpha=config['alpha'])
        
        train_time = time.time() - start_time
        
        # Evaluate
        train_preds, train_targets = [], []
        for i in range(len(X_train)):
            X_seq = reservoir.forward(X_train[i], collect_states=True)
            if len(X_seq) <= config['washout']:
                continue
            X_seq = X_seq[config['washout']:]
            Y_seq = Y_train[i][config['washout']:]
            Y_pred = X_seq @ wout
            train_preds.append(Y_pred)
            train_targets.append(Y_seq)
        
        test_preds, test_targets = [], []
        for i in range(len(X_test)):
            X_seq = reservoir.forward(X_test[i], collect_states=True)
            if len(X_seq) <= config['washout']:
                continue
            X_seq = X_seq[config['washout']:]
            Y_seq = Y_test[i][config['washout']:]
            Y_pred = X_seq @ wout
            test_preds.append(Y_pred)
            test_targets.append(Y_seq)
        
        if not train_preds or not test_preds:
            raise ValueError("All sessions too short after washout!")
        
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        # Metrics
        train_dir = compute_direction_metrics(train_targets[:, :2], train_preds[:, :2])
        train_boost = compute_boost_metrics(train_targets, train_preds)  # Pass full array
        test_dir = compute_direction_metrics(test_targets[:, :2], test_preds[:, :2])
        test_boost = compute_boost_metrics(test_targets, test_preds)  # Pass full array
        
        if verbose:
            print(f"\n⏱️  Training time: {train_time:.2f}s")
            print(f"\n📊 Train: Boost Acc={train_boost['accuracy']:.2%}, Angular={train_dir['angular_error_deg']:.1f}°")
            print(f"📊 Test:  Boost Acc={test_boost['accuracy']:.2%}, Angular={test_dir['angular_error_deg']:.1f}°")
        
        return {
            'reservoir_type': 'brain_connectome',
            'train_time': train_time,
            'wout': wout,
            'train_metrics': {'direction': train_dir, 'boost': train_boost},
            'test_metrics': {'direction': test_dir, 'boost': test_boost},
            'train_predictions': train_preds,
            'test_predictions': test_preds
        }
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Errore: {str(e)}")
        return None


def plot_comparison(results_random, results_connectome, output_dir):
    """Create comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    metrics = {
        'Random': {
            'train_boost': results_random['train_metrics']['boost']['accuracy'],
            'test_boost': results_random['test_metrics']['boost']['accuracy'],
            'train_angular': results_random['train_metrics']['direction']['angular_error_deg'],
            'test_angular': results_random['test_metrics']['direction']['angular_error_deg'],
            'time': results_random['train_time']
        }
    }
    
    if results_connectome:
        metrics['Connectome'] = {
            'train_boost': results_connectome['train_metrics']['boost']['accuracy'],
            'test_boost': results_connectome['test_metrics']['boost']['accuracy'],
            'train_angular': results_connectome['train_metrics']['direction']['angular_error_deg'],
            'test_angular': results_connectome['test_metrics']['direction']['angular_error_deg'],
            'time': results_connectome['train_time']
        }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Random vs Brain Connectome Reservoir Comparison', fontsize=16, fontweight='bold')
    
    # 1. Boost Accuracy
    ax = axes[0, 0]
    reservoir_types = list(metrics.keys())
    train_accs = [metrics[k]['train_boost'] for k in reservoir_types]
    test_accs = [metrics[k]['test_boost'] for k in reservoir_types]
    
    x = np.arange(len(reservoir_types))
    width = 0.35
    
    ax.bar(x - width/2, train_accs, width, label='Train', color='skyblue', edgecolor='black')
    ax.bar(x + width/2, test_accs, width, label='Test', color='salmon', edgecolor='black')
    
    ax.set_xlabel('Reservoir Type', fontweight='bold')
    ax.set_ylabel('Boost Accuracy', fontweight='bold')
    ax.set_title('Boost Prediction Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(reservoir_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 2. Angular Error
    ax = axes[0, 1]
    train_angular = [metrics[k]['train_angular'] for k in reservoir_types]
    test_angular = [metrics[k]['test_angular'] for k in reservoir_types]
    
    ax.bar(x - width/2, train_angular, width, label='Train', color='lightgreen', edgecolor='black')
    ax.bar(x + width/2, test_angular, width, label='Test', color='orange', edgecolor='black')
    
    ax.set_xlabel('Reservoir Type', fontweight='bold')
    ax.set_ylabel('Angular Error (degrees)', fontweight='bold')
    ax.set_title('Direction Prediction Error')
    ax.set_xticks(x)
    ax.set_xticklabels(reservoir_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Training Time
    ax = axes[1, 0]
    times = [metrics[k]['time'] for k in reservoir_types]
    
    ax.bar(reservoir_types, times, color='mediumpurple', edgecolor='black')
    ax.set_xlabel('Reservoir Type', fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax.set_title('Training Speed')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Overfitting Analysis
    ax = axes[1, 1]
    
    for i, res_type in enumerate(reservoir_types):
        train_acc = metrics[res_type]['train_boost']
        test_acc = metrics[res_type]['test_boost']
        acc_diff = train_acc - test_acc
        
        color = 'skyblue' if res_type == 'Random' else 'lightcoral'
        ax.bar(i, acc_diff, color=color, edgecolor='black', label=res_type)
        
        # Add text
        ax.text(i, acc_diff + 0.01, f"{acc_diff:.2%}", ha='center', fontweight='bold')
    
    ax.set_xlabel('Reservoir Type', fontweight='bold')
    ax.set_ylabel('Train - Test Accuracy', fontweight='bold')
    ax.set_title('Overfitting (lower is better)')
    ax.set_xticks(range(len(reservoir_types)))
    ax.set_xticklabels(reservoir_types)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High overfitting threshold')
    ax.axhline(y=0.2, color='darkred', linestyle='--', alpha=0.5, label='Very high overfitting')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot salvato: {output_dir / 'comparison.png'}")
    
    plt.close()


def create_comparison_report(results_random, results_connectome, output_path):
    """Create detailed comparison report."""
    lines = [
        "="*80,
        "CONFRONTO: RANDOM vs BRAIN CONNECTOME RESERVOIR",
        "="*80,
        "",
        "Data: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
    ]
    
    # Random reservoir
    lines.extend([
        "-"*80,
        "🎲 RANDOM RESERVOIR",
        "-"*80,
        f"Training time: {results_random['train_time']:.2f}s",
        "",
        "Training Metrics:",
        f"  Boost Accuracy: {results_random['train_metrics']['boost']['accuracy']:.2%}",
        f"  Angular Error:  {results_random['train_metrics']['direction']['angular_error_deg']:.2f}°",
        f"  Direction RMSE: {results_random['train_metrics']['direction']['rmse']:.4f}",
        "",
        "Test Metrics:",
        f"  Boost Accuracy: {results_random['test_metrics']['boost']['accuracy']:.2%}",
        f"  Angular Error:  {results_random['test_metrics']['direction']['angular_error_deg']:.2f}°",
        f"  Direction RMSE: {results_random['test_metrics']['direction']['rmse']:.4f}",
        "",
    ])
    
    # Connectome reservoir
    if results_connectome:
        lines.extend([
            "-"*80,
            "🧠 BRAIN CONNECTOME RESERVOIR",
            "-"*80,
            f"Training time: {results_connectome['train_time']:.2f}s",
            "",
            "Training Metrics:",
            f"  Boost Accuracy: {results_connectome['train_metrics']['boost']['accuracy']:.2%}",
            f"  Angular Error:  {results_connectome['train_metrics']['direction']['angular_error_deg']:.2f}°",
            f"  Direction RMSE: {results_connectome['train_metrics']['direction']['rmse']:.4f}",
            "",
            "Test Metrics:",
            f"  Boost Accuracy: {results_connectome['test_metrics']['boost']['accuracy']:.2%}",
            f"  Angular Error:  {results_connectome['test_metrics']['direction']['angular_error_deg']:.2f}°",
            f"  Direction RMSE: {results_connectome['test_metrics']['direction']['rmse']:.4f}",
            "",
        ])
        
        # Comparison
        lines.extend([
            "-"*80,
            "📊 CONFRONTO",
            "-"*80,
            "",
        ])
        
        # Test accuracy difference
        test_acc_diff = results_connectome['test_metrics']['boost']['accuracy'] - results_random['test_metrics']['boost']['accuracy']
        lines.append(f"Test Boost Accuracy difference: {test_acc_diff:+.2%}")
        if test_acc_diff > 0.05:
            lines.append("  ✅ Brain connectome SIGNIFICATIVAMENTE migliore!")
        elif test_acc_diff > 0.01:
            lines.append("  ✓ Brain connectome leggermente migliore")
        elif test_acc_diff < -0.05:
            lines.append("  ⚠️  Random reservoir SIGNIFICATIVAMENTE migliore!")
        elif test_acc_diff < -0.01:
            lines.append("  Random reservoir leggermente migliore")
        else:
            lines.append("  ≈ Prestazioni simili")
        
        lines.append("")
        
        # Angular error difference
        ang_diff = results_connectome['test_metrics']['direction']['angular_error_deg'] - results_random['test_metrics']['direction']['angular_error_deg']
        lines.append(f"Test Angular Error difference: {ang_diff:+.2f}°")
        if ang_diff < -5:
            lines.append("  ✅ Brain connectome SIGNIFICATIVAMENTE più preciso!")
        elif ang_diff < -1:
            lines.append("  ✓ Brain connectome più preciso")
        elif ang_diff > 5:
            lines.append("  ⚠️  Random reservoir SIGNIFICATIVAMENTE più preciso!")
        elif ang_diff > 1:
            lines.append("  Random reservoir più preciso")
        else:
            lines.append("  ≈ Precisione simile")
        
        lines.append("")
        
        # Training time
        time_diff = results_connectome['train_time'] - results_random['train_time']
        lines.append(f"Training time difference: {time_diff:+.2f}s")
        if abs(time_diff) < 1:
            lines.append("  ≈ Velocità simile")
        elif time_diff > 0:
            lines.append(f"  Brain connectome più lento ({results_connectome['train_time']/results_random['train_time']:.1f}x)")
        else:
            lines.append(f"  Brain connectome più veloce ({results_random['train_time']/results_connectome['train_time']:.1f}x)")
        
        lines.append("")
        
        # Overfitting
        random_overfit = results_random['train_metrics']['boost']['accuracy'] - results_random['test_metrics']['boost']['accuracy']
        connectome_overfit = results_connectome['train_metrics']['boost']['accuracy'] - results_connectome['test_metrics']['boost']['accuracy']
        
        lines.extend([
            "Overfitting Analysis:",
            f"  Random:     {random_overfit:.2%}",
            f"  Connectome: {connectome_overfit:.2%}",
        ])
        
        if connectome_overfit < random_overfit - 0.05:
            lines.append("  ✅ Brain connectome generalizza MEGLIO!")
        elif connectome_overfit < random_overfit:
            lines.append("  ✓ Brain connectome generalizza leggermente meglio")
        elif connectome_overfit > random_overfit + 0.05:
            lines.append("  ⚠️  Random reservoir generalizza MEGLIO!")
        else:
            lines.append("  ≈ Generalizzazione simile")
        
    else:
        lines.extend([
            "-"*80,
            "🧠 BRAIN CONNECTOME RESERVOIR",
            "-"*80,
            "❌ Non disponibile (mancano file .graphml)",
            "",
        ])
    
    lines.extend([
        "",
        "="*80,
        "CONCLUSIONI",
        "="*80,
    ])
    
    if results_connectome:
        if test_acc_diff > 0.03 and connectome_overfit < random_overfit:
            lines.extend([
                "",
                "✅ BRAIN CONNECTOME è MIGLIORE:",
                "   - Maggiore accuratezza sul test set",
                "   - Minore overfitting",
                "   → RACCOMANDATO per questo dataset!",
            ])
        elif test_acc_diff < -0.03:
            lines.extend([
                "",
                "⚠️  RANDOM RESERVOIR è migliore:",
                "   - Accuratezza superiore",
                "   → Per questo dataset, la struttura cerebrale non aiuta",
            ])
        else:
            lines.extend([
                "",
                "≈ PRESTAZIONI SIMILI:",
                "   - Non c'è un chiaro vincitore",
                "   - Prova con più dati o altri hyperparameter",
            ])
    else:
        lines.extend([
            "",
            "Per confrontare con brain connectome:",
            "  1. Scarica dataset connectome (es. C. elegans, Human Connectome)",
            "  2. Salva file .graphml nella root del progetto",
            "  3. Ri-esegui questo script",
        ])
    
    lines.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"📄 Report salvato: {output_path}")


def main():
    """Main function."""
    print("\n" + "="*80)
    print("⚖️  CONFRONTO RESERVOIR: RANDOM vs BRAIN CONNECTOME")
    print("="*80)
    
    # Load data
    print(f"\n📁 Caricamento dati da: {SLITHER_DATA_PATH}")
    
    try:
        X_list, y_list, session_names = load_all_data(SLITHER_DATA_PATH, verbose=True)
    except Exception as e:
        print(f"\n❌ Errore nel caricamento dati: {e}")
        return 1
    
    # Split
    print(f"\n🔀 Split train/test (test_size={TEST_SPLIT})...")
    from utilities.data_loader import train_test_split as split_sessions
    X_train, y_train, X_test, y_test, train_names, test_names = split_sessions(
        X_list, y_list, session_names, test_size=TEST_SPLIT, random_seed=RANDOM_SEED
    )
    
    # Concatenate
    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)
    
    # Reshape to sequences [n_samples, seq_len, features]
    X_train = X_train.reshape(len(X_train), 1, -1)
    X_test = X_test.reshape(len(X_test), 1, -1)
    
    n_inputs = X_train.shape[2]
    
    config = {
        'n_reservoir': N_RESERVOIR,
        'spectral_radius': SPECTRAL_RADIUS,
        'leak_range': (0.1, 0.3),
        'alpha': ALPHA,
        'washout': WASHOUT,
        'seed': RANDOM_SEED
    }
    
    # Train random reservoir
    results_random = train_random_reservoir(
        X_train, y_train, X_test, y_test,
        n_inputs, config, verbose=True
    )
    
    # Train brain connectome reservoir (if available)
    base_dir = Path(__file__).parent
    graphml_files = list(base_dir.glob("*.graphml"))
    
    if graphml_files:
        print(f"\n✅ Trovati {len(graphml_files)} file brain connectome")
        results_connectome = train_connectome_reservoir(
            X_train, y_train, X_test, y_test,
            n_inputs, base_dir, config, verbose=True
        )
    else:
        print("\n⚠️  Nessun file brain connectome trovato (.graphml)")
        print("   Solo random reservoir verrà testato.")
        results_connectome = None
    
    # Save results
    output_dir = Path(__file__).parent / "slither_esn_results" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot comparison
    plot_comparison(results_random, results_connectome, output_dir)
    
    # Create report
    create_comparison_report(
        results_random, results_connectome,
        output_dir / "comparison_report.txt"
    )
    
    print(f"\n{'='*80}")
    print("✅ CONFRONTO COMPLETATO!")
    print("="*80)
    print(f"\nRisultati salvati in: {output_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
