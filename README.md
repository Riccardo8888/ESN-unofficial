# Project: Reservoir Computing for Connectome Networks I
(automatically transacripted)
## Author
Victor Buendía

## Description
The rate model computes the firing rate (in spikes per second) of each neuron as a continuous variable. The 𝑖-th neuron has a rate 𝑟𝑖(𝑡) at time 𝑡.

The model evolution for 𝑁 neurons follows the differential equation:

![Equation](https://latex.codecogs.com/png.latex?%5Cfrac%7Bd%20r_i(t)%7D%7Bdt%7D%20%3D%20-r_i(t)%20%2B%20%5Cphi%20%5Cleft(%5Csum_%7Bj%3D1%7D%5EN%20A_%7Bij%7D%20r_j(t)%20%5Cright))

where the sigmoidal function:

![Sigmoid](https://latex.codecogs.com/png.latex?%5Cphi(x)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20%5Cexp(-x)%7D%20-%20%5Cfrac%7B1%7D%7B2%7D)

converts the input current into the neuron's firing rate.

Using an Euler integrator, the differential equation can be integrated computationally to analyze system behavior for different network topologies. This involves plotting individual trajectories and analyzing the statistics of the mean firing rate:

![Mean Firing Rate](https://latex.codecogs.com/png.latex?r(t)%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20r_i(t))

For example, the mean and variance of \( r(t) \) can be computed over time after allowing the network to reach a stationary state. The initial condition is chosen randomly within a small interval, e.g., \( r_i \in [0,1] \).

## Tasks

1. **Fully Connected Network**  
   - All links have the same weight:  
     ![Fully Connected](https://latex.codecogs.com/png.latex?A_%7Bij%7D%20%3D%20%5Cfrac%7BJ%7D%7BN%7D%20%5Cquad%20%5Ctext%7Bfor%20all%20%7D%20i%2C%20j.)

2. **Erdős-Rényi Network**  
   - Nodes are randomly connected with probability \( p \).  
   - The number of connections is \( k = pN \), and:  
     ![Erdos-Renyi](https://latex.codecogs.com/png.latex?A_%7Bij%7D%20%3D%20%5Cfrac%7BJ%7D%7Bk%7D%20%5Cquad%20%5Ctext%7Bwith%20probability%20%7D%20p%2C%20%5Ctext%7B%20otherwise%20%7D%200.)

3. **Gaussian Random Network**  
   - Fully connected network where each element is drawn from a Gaussian distribution:  
     ![Gaussian Network](https://latex.codecogs.com/png.latex?A_%7Bij%7D%20%5Csim%20%5Cmathcal%7BN%7D(0%2C%20%5Cfrac%7BJ%5E2%7D%7BN%7D).)

4. **Data Integration**  
   - The network data is incorporated by setting:  
     ![Data Integration](https://latex.codecogs.com/png.latex?A_%7Bij%7D%20%3D%20J%20D_%7Bij%7D%2C)
     where \( D_{ij} \) represents the network data and \( J \) is an arbitrary scaling factor.

Networks 1-3 serve as control reservoirs for comparison with real data.
