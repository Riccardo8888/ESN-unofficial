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


def leaked_feature_baseline(u_train: np.ndarray, y_train: np.ndarray,
                            u_test: np.ndarray = None, y_test: np.ndarray = None,
                            n_leaked: int = 2, washout: int = T_WASHOUT,
                            alpha: float = ALPHA) -> float:
    """No-reservoir angle accuracy using only the last `n_leaked` raw input features.

    `prepare_features` puts the previous-heading features (`prev_sin`, `prev_cos`) in the last two
    columns, and on the mock data the angle label is essentially a function of them. This baseline
    fits a plain ridge on just those columns, with no reservoir, to show how much of the label is
    solvable from the leak alone. On the committed mock data it scores ~0.6 in-sample (17 classes,
    chance ~0.06), and on the session-grouped split it is comparable to or beats the connectome
    reservoir. That is why mock numbers must not be read as modelling skill (see
    docs/METHODOLOGY.md and docs/AUDIT.md F4). Report it next to any mock reservoir number.

    u_train/u_test : [B, T, n_features]; y_train/y_test : [B, T, n_outputs]. If the test pair is
    omitted, the train pair is reused (the in-sample leak ceiling).
    """
    if u_test is None or y_test is None:
        u_test, y_test = u_train, y_train
    Xtr = u_train[:, :, -n_leaked:]
    Wout = compute_wout(Xtr, y_train, washout=washout, alpha=alpha)
    pred = u_test[:, washout:, -n_leaked:].reshape(-1, n_leaked) @ Wout
    yt = y_test[:, washout:, :].reshape(-1, y_test.shape[-1])
    return angle_accuracy(pred, yt)


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
    # Solve in float64: forming XᵀX squares the condition number and the reservoir states are
    # float32, so a float32 Gram can let rounding noise swamp a small ridge α and corrupt Wout.
    Xd = X.astype(np.float64)
    Yd = Y.astype(np.float64)
    R = Xd.T @ Xd
    P = Xd.T @ Yd
    return np.linalg.solve(R + alpha * np.eye(Xd.shape[1]), P)
