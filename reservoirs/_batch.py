"""Vectorized leaky-integrator state collection across a batch of windows (Phase 5).

The time loop is inherently sequential (x(t) depends on x(t-1)), but the B windows are
independent, so each step is one GEMM over the whole batch instead of B separate matvecs.
The dynamics are identical to the per-window loop (up to float associativity). The speedup is
modest and depends on batch and reservoir size: typically ≈1.5 to 2× in favourable regimes, and
it can be slower than the loop for some B/N (matmul flops dominate at large N). Benchmark before
relying on it. Used by the engines' `collect_states_batch`.
"""
import numpy as np


def leaky_collect_batch(drive: np.ndarray, W: np.ndarray, leak: np.ndarray) -> np.ndarray:
    """Run the leaky-integrator reservoir for a batch.

    Parameters
    drive : [B, T, N]  precomputed input contribution per step (Win @ u + bias).
    W : [N, N]  recurrent weights.   leak : [N]  per-neuron leak.

    Returns states [B, T, N] (state recorded after each update, matching the engines' forward()).
    """
    B, T, N = drive.shape
    WT = W.T
    x = np.zeros((B, N), dtype=drive.dtype)
    out = np.empty((B, T, N), dtype=drive.dtype)
    for t in range(T):
        cand = np.tanh(drive[:, t, :] + x @ WT)
        x = (1.0 - leak) * x + leak * cand
        out[:, t, :] = x
    return out
