"""Spectral-radius helper for the reservoirs package.

`dense_spectral_radius` is the exact spectral radius via dense eigenvalues, matching the
legacy random-reservoir engine's `max(abs(eig))` convention. It is used by the random family
(incl. `ErdosRenyiReservoir`).

NOTE: `connectome.py` keeps its OWN inline power-iteration `_spectral_radius` (golden-pinned),
so it is intentionally not shared here. Consolidating the two onto one helper would change the
characterization goldens and is deferred.
"""
import numpy as np


def dense_spectral_radius(A):
    """Exact spectral radius via dense eigenvalues. Matches reservoir.py's convention."""
    A = np.asarray(A)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))
