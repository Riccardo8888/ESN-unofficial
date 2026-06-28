"""Continual-learning metrics — pure functions of the R accuracy matrix.

R[i, j] = test accuracy on task j after training through task i (row = stage, col = task).
References: Lopez-Paz & Ranzato (2017, GEM) for ACC/BWT/FWT; Chaudhry et al. (2018, RWalk)
for Forgetting/Intransigence.
"""
import numpy as np


def cl_metrics(R, baselines=None, joint=None) -> dict:
    """Compute CL metrics from the accuracy matrix.

    Always returns: acc, bwt, forgetting.
    Returns fwt only if `baselines` (random-init per-task accuracy) is given.
    Returns intransigence only if `joint` (jointly-trained per-task accuracy) is given.
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    out = {"acc": float(np.mean(R[-1, :]))}

    if T > 1:
        out["bwt"] = float(np.mean([R[-1, i] - R[i, i] for i in range(T - 1)]))
        # RWalk forgetting: for each earlier task, best-ever minus final.
        out["forgetting"] = float(np.mean([np.max(R[:T - 1, j]) - R[T - 1, j] for j in range(T - 1)]))
    else:
        out["bwt"] = 0.0
        out["forgetting"] = 0.0

    if baselines is not None and T > 1:
        b = np.asarray(baselines, dtype=float)
        out["fwt"] = float(np.mean([R[i - 1, i] - b[i] for i in range(1, T)]))

    if joint is not None:
        j = np.asarray(joint, dtype=float)
        out["intransigence"] = float(np.mean([j[k] - R[k, k] for k in range(T)]))

    return out
