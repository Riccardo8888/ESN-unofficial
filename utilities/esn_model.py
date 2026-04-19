"""
Echo State Network Model for Temporal Prediction
=================================================

ESN implementation optimized for predicting future player actions
from current game state in Slither.io.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent))
from vnicktest.scripts.configuration import *


class SlitherESN:
    """
    Echo State Network for predicting future player actions.
    
    This ESN takes the current game state at frame t and predicts
    the player's actions at frame t+horizon.
    """
    
    def __init__(self, 
                 n_inputs: int = INPUT_DIM,
                 n_reservoir: int = N_RESERVOIR,
                 n_outputs: int = OUTPUT_DIM,
                 spectral_radius: float = SPECTRAL_RADIUS,
                 input_scale: float = INPUT_SCALE,
                 leak_rate_range: Tuple[float, float] = (LEAK_RATE_MIN, LEAK_RATE_MAX),
                 sparsity: Optional[float] = SPARSITY,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize Echo State Network.
        
        Args:
            n_inputs: Number of input features
            n_reservoir: Number of reservoir neurons
            n_outputs: Number of output features
            spectral_radius: Spectral radius of reservoir matrix
            input_scale: Scaling factor for input weights
            leak_rate_range: (min, max) leak rates for neurons
            sparsity: Fraction of connections to zero out (None for dense)
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leak_rate_range = leak_rate_range
        self.sparsity = sparsity
        
        # Initialize input weights: [n_reservoir, n_inputs + 1] (for bias)
        self.W_in = np.random.uniform(
            low=-1.0, 
            high=1.0, 
            size=(n_reservoir, n_inputs + 1)
        ) * input_scale
        
        # Initialize reservoir weights: [n_reservoir, n_reservoir]
        self.W_res = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(n_reservoir, n_reservoir)
        )
        
        # Apply sparsity if specified
        if sparsity is not None:
            mask = np.random.rand(n_reservoir, n_reservoir) > sparsity
            self.W_res = self.W_res * mask
        
        # Scale to desired spectral radius
        current_radius = self._compute_spectral_radius(self.W_res)
        self.W_res = self.W_res * (spectral_radius / current_radius)
        
        # Initialize leak rates for each neuron
        leak_min, leak_max = leak_rate_range
        self.leak_rates = np.random.uniform(
            low=leak_min,
            high=leak_max,
            size=(n_reservoir,)
        )
        
        # Output weights (trained via ridge regression)
        self.W_out = None
        
        # Training info
        self.is_trained = False
        
    @staticmethod
    def _compute_spectral_radius(W: np.ndarray) -> float:
        """Compute spectral radius of matrix W"""
        eigenvalues = np.linalg.eigvals(W)
        return np.max(np.abs(eigenvalues))
    
    def _reservoir_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single reservoir update step with leaky integrator neurons.
        
        Args:
            x: Current reservoir state [n_reservoir]
            u: Input vector [n_inputs]
            
        Returns:
            New reservoir state [n_reservoir]
        """
        # Add bias to input
        u_with_bias = np.concatenate([u, [1.0]])
        
        # Compute pre-activation
        x_pre = np.tanh(self.W_in @ u_with_bias + self.W_res @ x)
        
        # Apply leaky integration
        x_new = (1.0 - self.leak_rates) * x + self.leak_rates * x_pre
        
        return x_new
    
    def collect_states(self, U: np.ndarray, washout: int = WASHOUT) -> np.ndarray:
        """
        Collect reservoir states for a sequence of inputs.
        
        Args:
            U: Input sequence [n_timesteps, n_inputs]
            washout: Number of initial timesteps to discard
            
        Returns:
            Reservoir states [n_timesteps - washout, n_reservoir]
        """
        n_timesteps = U.shape[0]
        
        # Initialize reservoir state
        x = np.zeros(self.n_reservoir)
        
        # Collect all states (including washout)
        states = np.zeros((n_timesteps, self.n_reservoir))
        
        for t in range(n_timesteps):
            x = self._reservoir_step(x, U[t])
            states[t] = x
        
        # Return states after washout
        return states[washout:]
    
    def train(self, U: np.ndarray, Y: np.ndarray, 
              alpha: float = ALPHA, washout: int = WASHOUT,
              verbose: bool = True) -> Dict[str, float]:
        """
        Train the ESN using ridge regression.
        
        Args:
            U: Input sequence [n_timesteps, n_inputs]
            Y: Target output sequence [n_timesteps, n_outputs]
            alpha: Ridge regression regularization parameter
            washout: Number of initial timesteps to discard
            verbose: Print training information
            
        Returns:
            Dictionary with training statistics
        """
        if verbose:
            print(f"\nTraining ESN...")
            print(f"  Input shape: {U.shape}")
            print(f"  Output shape: {Y.shape}")
            print(f"  Washout: {washout} frames")
        
        # Collect reservoir states
        X = self.collect_states(U, washout=washout)
        
        # Corresponding outputs (also excluding washout)
        Y_train = Y[washout:]
        
        if verbose:
            print(f"  Reservoir states shape: {X.shape}")
            print(f"  Training with {X.shape[0]} samples")
        
        # Ridge regression: W_out = (X^T X + alpha I)^{-1} X^T Y
        # State correlation matrix
        R = X.T @ X
        
        # State-output cross-correlation
        P = X.T @ Y_train
        
        # Solve with regularization
        self.W_out = np.linalg.solve(
            R + alpha * np.eye(self.n_reservoir),
            P
        )
        
        self.is_trained = True
        
        # Compute training error
        Y_pred = X @ self.W_out
        mse = np.mean((Y_pred - Y_train) ** 2)
        
        if verbose:
            print(f"  ✓ Training complete")
            print(f"  Training MSE: {mse:.6f}")
        
        return {
            'mse': mse,
            'n_samples': X.shape[0]
        }
    
    def predict(self, U: np.ndarray, washout: int = WASHOUT) -> np.ndarray:
        """
        Make predictions using the trained ESN.
        
        Args:
            U: Input sequence [n_timesteps, n_inputs]
            washout: Number of initial timesteps to discard
            
        Returns:
            Predictions [n_timesteps - washout, n_outputs]
        """
        if not self.is_trained:
            raise RuntimeError("ESN must be trained before prediction!")
        
        # Collect reservoir states
        X = self.collect_states(U, washout=washout)
        
        # Generate predictions
        Y_pred = X @ self.W_out
        
        return Y_pred
    
    def save(self, filepath: Path):
        """Save ESN model to file"""
        model_data = {
            'W_in': self.W_in,
            'W_res': self.W_res,
            'W_out': self.W_out,
            'leak_rates': self.leak_rates,
            'n_inputs': self.n_inputs,
            'n_reservoir': self.n_reservoir,
            'n_outputs': self.n_outputs,
            'spectral_radius': self.spectral_radius,
            'input_scale': self.input_scale,
            'leak_rate_range': self.leak_rate_range,
            'sparsity': self.sparsity,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'SlitherESN':
        """Load ESN model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        esn = cls(
            n_inputs=model_data['n_inputs'],
            n_reservoir=model_data['n_reservoir'],
            n_outputs=model_data['n_outputs'],
            spectral_radius=model_data['spectral_radius'],
            input_scale=model_data['input_scale'],
            leak_rate_range=model_data['leak_rate_range'],
            sparsity=model_data['sparsity']
        )
        
        # Restore weights
        esn.W_in = model_data['W_in']
        esn.W_res = model_data['W_res']
        esn.W_out = model_data['W_out']
        esn.leak_rates = model_data['leak_rates']
        esn.is_trained = model_data['is_trained']
        
        print(f"✓ Model loaded from {filepath}")
        
        return esn
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return (f"SlitherESN(inputs={self.n_inputs}, reservoir={self.n_reservoir}, "
                f"outputs={self.n_outputs}, spectral_radius={self.spectral_radius}, "
                f"status={status})")


if __name__ == "__main__":
    """Test ESN creation and basic operations"""
    print_config()
    
    print("\n" + "=" * 60)
    print("TESTING ESN MODEL")
    print("=" * 60)
    
    # Create ESN
    esn = SlitherESN()
    print(f"\n{esn}")
    
    # Test with dummy data
    n_timesteps = 200
    U_dummy = np.random.randn(n_timesteps, INPUT_DIM)
    Y_dummy = np.random.randn(n_timesteps, OUTPUT_DIM)
    
    print(f"\nTraining with dummy data...")
    stats = esn.train(U_dummy, Y_dummy, verbose=True)
    
    print(f"\nMaking predictions...")
    Y_pred = esn.predict(U_dummy)
    print(f"  Prediction shape: {Y_pred.shape}")
    
    print(f"\n{esn}")
    
    # Test save/load
    test_path = OUTPUT_PATH / "test_esn.pkl"
    esn.save(test_path)
    esn_loaded = SlitherESN.load(test_path)
    print(f"\n{esn_loaded}")
