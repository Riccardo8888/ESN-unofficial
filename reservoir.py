import numpy as np


class Reservoir:
    def __init__(self, n_inputs, n_neurons, rhow=1.25, inp_scaling=1., leak_range=(0.1,0.3),
                 verbose=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.rhow = rhow
        self.inp_scaling = inp_scaling
        self.leak_range = leak_range

        # initialize weight matrices
        #self.win = np.ones()


        self.win = np.random.uniform(low=-1., high=1., size=(n_neurons, n_inputs+1)) * inp_scaling
        self.w = np.random.random((n_neurons, n_neurons)) * 2. - 1.
        leak_low, leak_high = leak_range
        self.leak = np.random.uniform(low=leak_low, high=leak_high, size=(n_neurons,))

        # set spectral radius
        rhow_current = self.spectral_radius
        self.w = self.w * rhow / rhow_current
        if verbose:
            print(f'spectral radius: {self.spectral_radius:.3f}')


    @property
    def spectral_radius(self):
        # compute the spectral radius
        return max(abs(np.linalg.eig(self.w)[0]))

    def update(self, ut, x):
        """
        Update reservoir state with a single input.
        
        Args:
            ut: Input vector at time t [n_inputs]
            x: Current reservoir state [n_neurons]
            
        Returns:
            Updated reservoir state [n_neurons]
        """
        x_next = np.tanh(self.win @ np.concatenate((ut, [1])) + self.w @ x)
        x = (1. - self.leak) * x + self.leak * x_next
        return x

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