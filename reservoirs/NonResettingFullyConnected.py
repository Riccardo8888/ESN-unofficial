import numpy as np


class NonResettingFullyConnectedReservoir:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1.0,
                 leak_range=(0.1, 0.3), seed=None, verbose=False):
        rng = np.random.default_rng(seed)
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.leak_range = leak_range
        self.x = np.zeros(n_neurons, dtype=float)
        self.win = rng.uniform(-inp_scaling, inp_scaling, size=(n_neurons, n_inputs + 1))
        W = rng.uniform(-1.0, 1.0, size=(n_neurons, n_neurons))
        np.fill_diagonal(W, 0.0)
        self.w = W
        a, b = leak_range
        self.leak = rng.uniform(a, b, size=(n_neurons,))
        sr = self.spectral_radius
        if sr > 0 and np.isfinite(sr):
            self.w = self.w * (rhow / sr)
        if verbose:
            print(f"NRFC spectral radius: {self.spectral_radius:.3f} (non-resetting, fully connected)")

    def reset_state(self):
        self.x = np.zeros(self.n_neurons, dtype=float)

    @property
    def spectral_radius(self):
        try:
            return float(np.max(np.abs(np.linalg.eigvals(self.w))))
        except Exception:
            return 0.0

    def forward(self, u, wout=None, collect_states=False):
        u = np.atleast_2d(u)
        T = u.shape[0]
        need = collect_states or wout is not None
        X = np.zeros((T, self.n_neurons)) if need else None
        for t in range(T):
            ut = u[t]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1.0])) + self.w @ self.x)
            self.x = (1.0 - self.leak) * self.x + self.leak * x_next
            if need:
                X[t] = self.x
        if wout is not None:
            Y = X @ wout
            return (X, Y) if collect_states else Y
        return X if collect_states else None
