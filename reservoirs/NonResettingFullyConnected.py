import numpy as np

class FullyConnectedReservoir:
    def __init__(
        self,
        n_inputs,
        n_neurons,
        rhow=1.25,
        inp_scaling=1.0,
        leak_range=(0.1, 0.3),
        density=0.1,
        seed=None,
        verbose=False,
    ):
        rng = np.random.default_rng(seed)

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range

        # Continuous internal state
        self.x = np.zeros((n_neurons,), dtype=float)

        self.win = rng.uniform(
            low=-inp_scaling,
            high=inp_scaling,
            size=(n_neurons, n_inputs + 1),  # +1 for bias
        )

        W = rng.uniform(-1.0, 1.0, size=(n_neurons, n_neurons))
        mask = rng.random((n_neurons, n_neurons)) < density
        W *= mask

        eigvals = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigvals))
        if sr > 0:
            W *= (rhow / sr)

        self.w = W

        a, b = leak_range
        self.leak = rng.uniform(a, b, size=(n_neurons,))

        if verbose:
            eigvals = np.linalg.eigvals(self.w)
            print(f"Spectral radius: {np.max(np.abs(eigvals)):.3f}")

    @property
    def spectral_radius(self):
        # Compute the spectral radius
        return max(abs(np.linalg.eig(self.w)[0]))

    def forward(self, u: np.ndarray, wout: np.ndarray = None, collect_states: bool = False):
        n_timesteps = u.shape[0]
        #x = np.zeros((self.n_neurons,)) # This the main difference in reservoirs
        #setup matrix for states

        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))

        #forward pass loop
        for t in range(n_timesteps):
            ut = u[t, :]
            x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ self.x)
            self.x = (1. - self.leak) * self.x + self.leak * x_next
            if collect_states or wout is not None:
                X[t, :] = self.x

        #compute outputs if desired
        if wout is not None:
            Y = X @ wout
        else: Y = None # Reference safety

        #return outputs and/or states
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X