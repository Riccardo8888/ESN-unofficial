"""Random-connectivity reservoir substrates (frozen recurrent weights).

Ported verbatim from the original ``reservoir.py`` (classes ``Reservoir`` /
``Reservoir2`` / ``Reservoir3``) so the Phase-1 characterization goldens reproduce
bit-for-bit, plus descriptive aliases and a genuine sparse ``ErdosRenyiReservoir``.
"""
import numpy as np

from ._batch import leaky_collect_batch


def dense_spectral_radius(A):
    """Exact spectral radius via dense eigenvalues (matches the legacy `max(abs(eig))` convention).

    Used by the random family (incl. `ErdosRenyiReservoir`). The connectome engine keeps its own
    inline power-iteration `_spectral_radius` (golden-pinned), so the two are intentionally separate.
    (Folded in from the former `reservoirs/_spectral.py`, whose only caller was this module.)
    """
    A = np.asarray(A)
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(A))))


class Reservoir:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1., leak_range=(0.1,0.3),
                 verbose=False, seed=None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range

        # RNG: default to the GLOBAL numpy RNG (seed=None) so the characterization goldens
        # reproduce; pass seed= for a self-contained, reproducible draw that does not depend on
        # external np.random.seed() ordering (AUDIT.md F9).
        rng = np.random if seed is None else np.random.default_rng(seed)

        self.win = rng.uniform(low=-1., high=1., size=(n_neurons, n_inputs+1)) * inp_scaling
        self.w = rng.random((n_neurons, n_neurons)) * 2. - 1.
        leak_low, leak_high = leak_range
        self.leak = rng.uniform(low=leak_low, high=leak_high, size=(n_neurons,))

        # set spectral radius
        rhow_current = self.spectral_radius
        self.w = self.w * rhow / rhow_current
        if verbose:
            print(f'spectral radius: {self.spectral_radius:.3f}')


    @property
    def spectral_radius(self):
        # compute the spectral radius
        return max(abs(np.linalg.eig(self.w)[0]))


    def forward(self, u, wout=None, collect_states=False):
        n_timesteps = u.shape[0]
        # initialize state
        x = np.zeros((self.n_neurons,))
        # setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        # forward pass loop
        for t in range(n_timesteps):
            ut = u[t,:]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
            x = (1. - self.leak) * x + self.leak * x_next
            if collect_states or wout is not None:
                X[t,:] = x
        # compute outputs if desired
        if wout is not None:
            Y = X @ wout
        # return outputs and/or states
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X

    def collect_states_batch(self, U):
        """Vectorized state collection over a batch of windows (Phase 5).

        One GEMM per timestep across the batch instead of B per-window matvecs. Speedup is modest
        and batch-size dependent (≈1.5 to 2×; can be slower for some B/N, see _batch.py).

        U : [B, T, n_inputs] -> states [B, T, n_neurons]. Equivalent (within float tolerance)
        to ``np.stack([self.forward(u, collect_states=True) for u in U])``.
        """
        U = np.asarray(U, dtype=float)
        if U.ndim != 3 or U.shape[2] != self.n_inputs:
            raise ValueError(f"U must be [B, T, n_inputs={self.n_inputs}]; got {U.shape}.")
        B, T, _ = U.shape
        Uaug = np.concatenate([U, np.ones((B, T, 1))], axis=-1)  # implicit bias input
        drive = Uaug @ self.win.T                               # [B, T, N]
        return leaky_collect_batch(drive, self.w, self.leak)



class Reservoir2:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1., leak_range=(0.1, 0.3),
                 verbose=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range

        # Initialize input weight matrix with fixed values
        self.win = np.full((n_neurons, n_inputs + 1), 0.5) * inp_scaling

        # Initialize reservoir weight matrix
        self.w = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            self.w[i, (i + 1) % n_neurons] = 0.5  # Connect to next neuron
            self.w[i, (i - 1) % n_neurons] = -0.5  # Connect to previous neuron

        # Set leak rates as fixed values in the given range
        leak_low, leak_high = leak_range
        self.leak = np.full((n_neurons,), (leak_low + leak_high) / 2)  # Use the midpoint

        # Set spectral radius
        rhow_current = self.spectral_radius
        self.w = self.w * rhow / rhow_current
        if verbose:
            print(f'Spectral radius: {self.spectral_radius:.3f}')

    @property
    def spectral_radius(self):
        # Compute the spectral radius
        return max(abs(np.linalg.eig(self.w)[0]))

    def forward(self, u, wout=None, collect_states=False):
        n_timesteps = u.shape[0]
        x = np.zeros((self.n_neurons,))
        #setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        #forward pass loop
        for t in range(n_timesteps):
            ut = u[t, :]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
            x = (1. - self.leak) * x + self.leak * x_next
            if collect_states or wout is not None:
                X[t, :] = x
        #compute outputs if desired
        if wout is not None:
            Y = X @ wout
        #return outputs and/or states
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X
  
        
class Reservoir3:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1., leak_range=(0.1, 0.3),
                 sigma=1.0, verbose=False, seed=None):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range
        self.sigma = sigma

        # RNG: default to the GLOBAL numpy RNG (seed=None) to preserve goldens; pass seed= for a
        # self-contained reproducible draw (AUDIT.md F9).
        rng = np.random if seed is None else np.random.default_rng(seed)

        # Initialize input weight matrix with random values
        self.win = rng.uniform(low=-1., high=1., size=(n_neurons, n_inputs + 1)) * inp_scaling

        # Initialize reservoir weight matrix with Gaussian distribution
        self.w = rng.normal(loc=0.0, scale=sigma, size=(n_neurons, n_neurons))

        # Set leak rates as random values in the given range
        leak_low, leak_high = leak_range
        self.leak = rng.uniform(low=leak_low, high=leak_high, size=(n_neurons,))

        # Set spectral radius
        rhow_current = self.spectral_radius
        self.w = self.w * rhow / rhow_current
        if verbose:
            print(f'Spectral radius: {self.spectral_radius:.3f}')

    @property
    def spectral_radius(self):
        # Compute the spectral radius
        return max(abs(np.linalg.eig(self.w)[0]))

    def forward(self, u, wout=None, collect_states=False):
        n_timesteps = u.shape[0]
        x = np.zeros((self.n_neurons,))
        # Setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        # Forward pass loop
        for t in range(n_timesteps):
            ut = u[t, :]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
            x = (1. - self.leak) * x + self.leak * x_next
            if collect_states or wout is not None:
                X[t, :] = x
        # Compute outputs if desired
        if wout is not None:
            Y = X @ wout
        # Return outputs and/or states
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X


# Descriptive aliases for the connectivity each legacy class implements
RingReservoir = Reservoir2            # bidirectional ring / cyclic topology
GaussianReservoir = Reservoir3        # dense Gaussian recurrent weights


class ErdosRenyiReservoir:
    """Sparse Erdős-Rényi reservoir (the classic sparse ESN; Jaeger 2001).

    Each ordered pair (i, j), i != j, gets a recurrent connection with probability
    ``density``; present weights are drawn uniformly from [-1, 1]; the matrix is then
    rescaled to spectral radius ``rhow``. This is the genuine Erdős-Rényi connectivity
    the ``handson_erdos_renyi`` experiment was meant to use (it had silently fallen
    back to the dense ``Reservoir``), and it fulfils the intent of the broken
    ``reservoir_ramiro.py`` import ``from reservoirs.ErdosRenyi import ...``.

    Mirrors the rest of the random family, plus two extra arguments. ``density`` is the
    Erdős-Rényi edge probability p, a float in (0, 1]. ``seed`` is an optional int RNG seed
    (it uses ``np.random.default_rng`` for reproducibility; the legacy classes use the global RNG).
    """

    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1.,
                 leak_range=(0.1, 0.3), density=0.1, seed=None, verbose=False):
        if not (0.0 < density <= 1.0):
            raise ValueError("density must be in (0, 1].")
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range
        self.density = density

        rng = np.random.default_rng(seed)

        # input weights (with bias column) and per-neuron leak
        self.win = rng.uniform(-1., 1., size=(n_neurons, n_inputs + 1)) * inp_scaling
        leak_low, leak_high = leak_range
        self.leak = rng.uniform(low=leak_low, high=leak_high, size=(n_neurons,))

        # sparse Erdős-Rényi recurrent matrix
        mask = rng.random((n_neurons, n_neurons)) < density
        weights = rng.uniform(-1., 1., size=(n_neurons, n_neurons))
        w = weights * mask
        np.fill_diagonal(w, 0.0)  # no self-loops

        sr = dense_spectral_radius(w)
        if sr > 0.0:
            w = w * (rhow / sr)
        self.w = w

        if verbose:
            print(f'spectral radius: {self.spectral_radius:.3f}, density~{mask.mean():.3f}')

    @property
    def spectral_radius(self):
        return dense_spectral_radius(self.w)

    def forward(self, u, wout=None, collect_states=False):
        n_timesteps = u.shape[0]
        x = np.zeros((self.n_neurons,))
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        for t in range(n_timesteps):
            ut = u[t, :]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
            x = (1. - self.leak) * x + self.leak * x_next
            if collect_states or wout is not None:
                X[t, :] = x
        if wout is not None:
            Y = X @ wout
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X
