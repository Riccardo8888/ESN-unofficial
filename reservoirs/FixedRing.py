import numpy as np


class FixedRingReservoir:
    """Deterministic +/-0.5 ring (paper Appendix: 'Fixed matrix values').

    Each neuron i is connected to i+1 with weight +0.5 and to i-1 with -0.5
    (modulo N). Eliminates randomness from the recurrent matrix entirely so
    that variation in dynamics comes only from rho_W, leak, and input scaling.
    """

    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1.0,
                 leak_range=(0.1, 0.3), seed=None, verbose=False):
        rng = np.random.default_rng(seed)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.leak_range = leak_range

        # Win is still random with bias column (input drives the otherwise
        # deterministic reservoir).
        self.win = rng.uniform(-inp_scaling, inp_scaling,
                               size=(n_neurons, n_inputs + 1))

        # Deterministic ring with +/-0.5 weights.
        W = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            W[i, (i + 1) % n_neurons] = 0.5
            W[i, (i - 1) % n_neurons] = -0.5
        self.w = W

        a, b = leak_range
        self.leak = rng.uniform(a, b, size=(n_neurons,))

        sr = self.spectral_radius
        if sr > 0 and np.isfinite(sr):
            self.w = self.w * (rhow / sr)
        if verbose:
            print(f"FixedRing spectral radius: {self.spectral_radius:.3f}")

    @property
    def spectral_radius(self):
        try:
            return float(np.max(np.abs(np.linalg.eigvals(self.w))))
        except Exception:
            return 0.0

    def forward(self, u, wout=None, collect_states=False):
        u = np.atleast_2d(u)
        T = u.shape[0]
        x = np.zeros(self.n_neurons)
        need = collect_states or wout is not None
        X = np.zeros((T, self.n_neurons)) if need else None
        for t in range(T):
            ut = u[t]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1.0])) + self.w @ x)
            x = (1.0 - self.leak) * x + self.leak * x_next
            if need:
                X[t] = x
        if wout is not None:
            Y = X @ wout
            return (X, Y) if collect_states else Y
        return X if collect_states else None
