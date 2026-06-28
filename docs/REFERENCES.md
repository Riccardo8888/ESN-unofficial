# References

Citations underpinning the methods in this repo. (Compiled in Phase 4, with primary sources
preferred. The Cossu et al. 2021 decimals quoted in the design doc were verified against the ESANN
proceedings PDF and arXiv:2105.07674: Split-MNIST, class-incremental, ESN column.)

> **Implemented vs planned.** Entries tagged **[planned]** back methods that are designed but not yet
> implemented (see the status table in `CONTINUOUS_LEARNING_DESIGN.md`). What the code actually
> implements: a leaky-integrator ESN; ridge plus online RLS/LMS/NLMS readouts; a per-class conceptor
> bank; degree-preserving (Maslov-Sneppen) nulls; and GEM plus RWalk CL metrics.

## Echo State Networks / reservoir computing
- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148.
- Lukoševičius, M. (2012). *A Practical Guide to Applying Echo State Networks.* In *Neural Networks: Tricks of the Trade*. Covers input scaling, spectral radius, leak, ridge readout, and washout.
- Jaeger, H., Lukoševičius, M., Popovici, D., Siewert, U. (2007). *Optimization and applications of echo state networks with leaky-integrator neurons.* Neural Networks 20(3):335-352.
- Yildiz, I.B., Jaeger, H., Kiebel, S.J. (2012). *Re-visiting the echo state property.* Neural Networks 35:1-9. Shows that ρ<1 is necessary but not sufficient, and that the ESP can hold for ρ>1 under input drive, which justifies a super-critical spectral radius on short driven windows.

## Connectome-based reservoir computing
- Suárez, L.E., Richards, B.A., Lajoie, G., Mišić, B. (2021). *Learning function from structure in neuromorphic networks.* Nature Machine Intelligence 3:771-786.
- Suárez, L.E. et al. (2024). *Connectome-based reservoir computing with the conn2res toolbox.* Nature Communications 15:656. Covers directed and undirected connectomes, ridge readout, null-model (rewired) baselines, and α swept across the critical point.

## Online / incremental readouts
- Jaeger, H. (2003). *Adaptive Nonlinear System Identification with Echo State Networks.* NIPS 15. The RLS readout for ESNs.
- Sussillo, D., Abbott, L.F. (2009). *Generating Coherent Patterns of Activity from Chaotic Neural Networks.* Neuron 63(4):544-557. FORCE. **[planned]** The bundled online readout is plain RLS, which is not FORCE: FORCE feeds the live output back into the recurrent loop during training. Do not cite FORCE for the current RLS readout.
- Widrow, B., Hoff, M.E. (1960). *Adaptive switching circuits.* LMS / Widrow-Hoff (implemented: `method='lms'`/`'nlms'`).
- Haykin, S. (ed.) (2001). *Kalman Filtering and Neural Networks.* Wiley. Kalman as RLS with process noise. **[planned]** (no Kalman readout implemented).

## Continual / lifelong learning
- Cossu, A., Bacciu, D., Carta, A., Gallicchio, C., Lomonaco, V. (2021). *Continual Learning with Echo State Networks.* ESANN 2021, pp. 275-280 (arXiv:2105.07674; DOI:10.14428/esann/2021.ES2021-80). A frozen reservoir does not by itself prevent forgetting; on Split-MNIST class-IL (ESN column), SLDA 0.88 and Replay 0.74 are strong while EWC and Naive collapse to the 0.20 floor. (Distinct from the LSTM study arXiv:2103.07492, "...an Empirical Evaluation", Neural Networks 143.)
- Jaeger, H. (2014). *Controlling Recurrent Neural Networks by Conceptors.* arXiv:1403.3369.
- Jaeger, H. (2017). *Using Conceptors to Manage Neural Long-Term Memories for Temporal Patterns.* JMLR 18(13):1-43.
- He, X., Jaeger, H. (2018). *Overcoming Catastrophic Interference by Conceptors.* arXiv:1707.04853.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS 114(13):3521-3526. EWC. **[planned]**
- Zenke, F., Poole, B., Ganguli, S. (2017). *Continual Learning Through Synaptic Intelligence.* ICML. SI. **[planned]**
- Aljundi, R. et al. (2018). *Memory Aware Synapses.* ECCV. MAS. **[planned]**
- Robins, A. (1995). *Catastrophic Forgetting, Rehearsal and Pseudorehearsal.* Connection Science 7(2):123-146. **[planned]** (replay/pseudo-rehearsal not implemented).

## Continual-learning evaluation
- van de Ven, G.M., Tolias, A.S. (2019). *Three scenarios for continual learning.* arXiv:1904.07734.
- van de Ven, G.M., Tuytelaars, T., Tolias, A.S. (2022). *Three types of incremental learning.* Nature Machine Intelligence 4:1185-1197.
- Lopez-Paz, D., Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning.* NeurIPS. ACC/BWT/FWT.
- Chaudhry, A., Dokania, P., Ajanthan, T., Torr, P. (2018). *Riemannian Walk for Incremental Learning.* ECCV. Forgetting and Intransigence.
- Díaz-Rodríguez, N., Lomonaco, V., Filliat, D., Maltoni, D. (2018). *Don't forget, there is more than forgetting: new metrics for Continual Learning.* arXiv:1810.13166.

## Null models
- Maslov, S., Sneppen, K. (2002). *Specificity and stability in topology of protein networks.* Science 296:910-913. Degree-preserving double-edge-swap randomization.
