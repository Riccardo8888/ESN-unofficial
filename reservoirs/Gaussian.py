
import numpy as np


class GaussianReservoir:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1., leak_range=(0.1, 0.3),
                 sigma=1.0, verbose=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range
        self.sigma = sigma

        # Initialize input weight matrix with random values
        self.win = np.random.uniform(low=-1., high=1., size=(n_neurons, n_inputs + 1)) * inp_scaling

        # Initialize reservoir weight matrix with Gaussian distribution
        self.w = np.random.normal(loc=0.0, scale=sigma, size=(n_neurons, n_neurons))

        # Set leak rates as random values in the given range
        leak_low, leak_high = leak_range
        self.leak = np.random.uniform(low=leak_low, high=leak_high, size=(n_neurons,))

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