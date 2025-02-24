# Project: Reservoir Computing for Connectome Networks I

## Author
Victor Buendía

## Description
The rate model computes the firing rate (in spikes per second) of each neuron as a continuous variable. The 𝑖-th neuron has a rate 𝑟𝑖(𝑡) at time 𝑡.

The model evolution for 𝑁 neurons follows the differential equation:

\[
\frac{d r_i(t)}{dt} = -r_i(t) + \phi \left(\sum_{j=1}^{N} A_{ij} r_j(t) \right)
\]

where the sigmoidal function:

\[
\phi(x) = \frac{1}{1 + \exp(-x)} - \frac{1}{2}
\]

converts the input current into the neuron's firing rate.

Using an Euler integrator, the differential equation can be integrated computationally to analyze system behavior for different network topologies. This involves plotting individual trajectories and analyzing the statistics of the mean firing rate:

\[
r(t) = \frac{1}{N} \sum_{i=1}^{N} r_i(t)
\]

For example, the mean and variance of \( r(t) \) can be computed over time after allowing the network to reach a stationary state. The initial condition is chosen randomly within a small interval, e.g., \( r_i \in [0,1] \).

## Tasks

1. **Fully Connected Network**  
   - All links have the same weight:  
     \[
     A_{ij} = \frac{J}{N} \quad \text{for all } i, j.
     \]

2. **Erdős-Rényi Network**  
   - Nodes are randomly connected with probability \( p \).  
   - The number of connections is \( k = pN \), and:  
     \[
     A_{ij} = \frac{J}{k} \quad \text{with probability } p, \text{ otherwise } 0.
     \]

3. **Gaussian Random Network**  
   - Fully connected network where each element is drawn from a Gaussian distribution:  
     \[
     A_{ij} \sim \mathcal{N}(0, \frac{J^2}{N}).
     \]

4. **Data Integration**  
   - The network data is incorporated by setting:  
     \[
     A_{ij} = J D_{ij},
     \]
     where \( D_{ij} \) represents the network data and \( J \) is an arbitrary scaling factor.

Networks 1-3 serve as control reservoirs for comparison with real data.
