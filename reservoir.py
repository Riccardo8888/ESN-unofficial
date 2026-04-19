import numpy as np

from reservoirs.ErdosRenyi import ErdosRenyiReservoir

# It seems like a lot of this file is broken, idk why. Use the reservoirs in the reservoirs package instead.
class Reservoir(ErdosRenyiReservoir):
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

    def forward(self, u, wout=None, collect_states=False, show_progress=False):
        n_timesteps = u.shape[0]
        # initialize state
        x = np.zeros(self.n_neurons)
        # setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        
        # Pre-compute constants for speed
        leak_factor = self.leak
        inv_leak_factor = 1.0 - leak_factor
        
        # Pre-allocate augmented input vector (avoid concatenate in loop)
        u_aug = np.ones(u.shape[1] + 1)
        
        # Progress tracking
        progress_interval = max(1, n_timesteps // 100)  # Update every 1%
        import time
        start_time = time.time() if show_progress else None
        
        # forward pass loop - OPTIMIZED for macOS Accelerate
        for t in range(n_timesteps):
            # Augment input with bias (faster than concatenate)
            u_aug[:-1] = u[t]
            
            # Compute next state using optimized matmul (uses Accelerate on macOS)
            x_next = np.tanh(np.dot(self.win, u_aug) + np.dot(self.w, x))
            
            # Leaky integration (vectorized)
            x = inv_leak_factor * x + leak_factor * x_next
            
            if collect_states or wout is not None:
                X[t] = x
            
            # Show progress with ETA
            if show_progress and (t % progress_interval == 0 or t == n_timesteps - 1):
                percent = (t + 1) / n_timesteps * 100
                bar_length = 40
                filled = int(bar_length * (t + 1) / n_timesteps)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if t > 0:
                    eta = elapsed / (t + 1) * (n_timesteps - t - 1)
                    eta_str = f"{int(eta//60)}:{int(eta%60):02d}"
                    elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
                    print(f'\r  [{bar}] {percent:.1f}% ({t+1}/{n_timesteps}) | Elapsed: {elapsed_str} | ETA: {eta_str}', end='', flush=True)
                else:
                    print(f'\r  [{bar}] {percent:.1f}% ({t+1}/{n_timesteps})', end='', flush=True)
        
        if show_progress:
            print()  # Newline after progress bar
        
        # compute outputs if desired (optimized matmul)
        if wout is not None:
            Y = np.dot(X, wout)
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

    def forward(self, u, wout=None, collect_states=False, show_progress=False):
        n_timesteps = u.shape[0]
        x = np.zeros(self.n_neurons)
        #setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        
        # Pre-compute constants for speed
        leak_factor = self.leak
        inv_leak_factor = 1.0 - leak_factor
        
        # Pre-allocate augmented input vector
        u_aug = np.ones(u.shape[1] + 1)
        
        # Progress tracking
        progress_interval = max(1, n_timesteps // 100)  # Update every 1%
        import time
        start_time = time.time() if show_progress else None
        
        #forward pass loop - OPTIMIZED for macOS Accelerate
        for t in range(n_timesteps):
            # Augment input with bias
            u_aug[:-1] = u[t]
            
            # Compute next state (uses Accelerate on macOS)
            x_next = np.tanh(np.dot(self.win, u_aug) + np.dot(self.w, x))
            
            # Leaky integration (vectorized)
            x = inv_leak_factor * x + leak_factor * x_next
            
            if collect_states or wout is not None:
                X[t] = x
            
            # Show progress with ETA
            if show_progress and (t % progress_interval == 0 or t == n_timesteps - 1):
                percent = (t + 1) / n_timesteps * 100
                bar_length = 40
                filled = int(bar_length * (t + 1) / n_timesteps)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if t > 0:
                    eta = elapsed / (t + 1) * (n_timesteps - t - 1)
                    eta_str = f"{int(eta//60)}:{int(eta%60):02d}"
                    elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
                    print(f'\r  [{bar}] {percent:.1f}% ({t+1}/{n_timesteps}) | Elapsed: {elapsed_str} | ETA: {eta_str}', end='', flush=True)
                else:
                    print(f'\r  [{bar}] {percent:.1f}% ({t+1}/{n_timesteps})', end='', flush=True)
        
        if show_progress:
            print()  # Newline after progress bar
        
        #compute outputs if desired
        if wout is not None:
            Y = np.dot(X, wout)
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

    def forward(self, u, wout=None, collect_states=False, show_progress=False):
        n_timesteps = u.shape[0]
        x = np.zeros(self.n_neurons)
        # Setup matrix for states
        if collect_states or wout is not None:
            X = np.zeros((n_timesteps, self.n_neurons))
        
        # Pre-compute constants for speed
        leak_factor = self.leak
        inv_leak_factor = 1.0 - leak_factor
        
        # Pre-allocate augmented input vector
        u_aug = np.ones(u.shape[1] + 1)
        
        # Progress tracking
        progress_interval = max(1, n_timesteps // 100)  # Update every 1%
        import time
        start_time = time.time() if show_progress else None
        
        # Forward pass loop - OPTIMIZED for macOS Accelerate
        for t in range(n_timesteps):
            # Augment input with bias
            u_aug[:-1] = u[t]
            
            # Compute next state (uses Accelerate on macOS)
            x_next = np.tanh(np.dot(self.win, u_aug) + np.dot(self.w, x))
            
            # Leaky integration (vectorized)
            x = inv_leak_factor * x + leak_factor * x_next
            
            if collect_states or wout is not None:
                X[t] = x
            
            # Show progress with ETA
            if show_progress and (t % progress_interval == 0 or t == n_timesteps - 1):
                percent = (t + 1) / n_timesteps * 100
                bar_length = 40
                filled = int(bar_length * (t + 1) / n_timesteps)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if t > 0:
                    eta = elapsed / (t + 1) * (n_timesteps - t - 1)
                    eta_str = f"{int(eta//60)}:{int(eta%60):02d}"
                    elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
                    print(f'\r  [{bar}] {percent:.1f}% ({t+1}/{n_timesteps}) | Elapsed: {elapsed_str} | ETA: {eta_str}', end='', flush=True)
                else:
                    print(f'\r  [{bar}] {percent:.1f}% ({t+1}/{n_timesteps})', end='', flush=True)
        
        if show_progress:
            print()  # Newline after progress bar
        
        # Compute outputs if desired
        if wout is not None:
            Y = np.dot(X, wout)
        # Return outputs and/or states
        if wout is not None and collect_states:
            return X, Y
        elif wout is not None:
            return Y
        elif collect_states:
            return X