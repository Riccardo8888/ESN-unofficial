"""Task metrics + ridge readout for the slither.io pipeline (Phase 3 extraction)."""
import numpy as np

from .config import NUM_ANGLE_BINS, T_WASHOUT, ALPHA


def angle_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of timesteps where the argmax angle bin matches."""
    yp = y_pred[..., :NUM_ANGLE_BINS].reshape(-1, NUM_ANGLE_BINS).argmax(axis=1)
    yt = y_true[..., :NUM_ANGLE_BINS].reshape(-1, NUM_ANGLE_BINS).argmax(axis=1)
    return float(np.mean(yp == yt))


def boost_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Accuracy of the thresholded (>=0.5) boost channel (last column)."""
    yp = (y_pred[..., -1].reshape(-1) >= 0.5).astype(int)
    yt = (y_true[..., -1].reshape(-1) >= 0.5).astype(int)
    return float(np.mean(yp == yt))


def compute_wout(X_train_states: np.ndarray, y_train: np.ndarray,
                 washout: int = T_WASHOUT, alpha: float = ALPHA) -> np.ndarray:
    """Closed-form ridge readout Wout = (XᵀX + αI)⁻¹ XᵀY.

    X_train_states : [n_windows, T, n_neurons]; y_train : [n_windows, T, n_outputs].
    Washout rows are dropped per-window before fitting.
    """
    X = X_train_states[:, washout:, :].reshape(-1, X_train_states.shape[-1])
    Y = y_train[:, washout:, :].reshape(-1, y_train.shape[-1])
    if X.shape[0] == 0:
        raise ValueError(
            f"compute_wout: no training rows after washout={washout} "
            f"(window length {X_train_states.shape[1]} <= washout). Lower the washout or use longer windows."
        )
    R = X.T @ X
    P = X.T @ Y
    return np.linalg.solve(R + alpha * np.eye(X.shape[1]), P)
