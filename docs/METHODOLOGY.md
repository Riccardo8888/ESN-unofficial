# Methodology & Limitations

Paper-grade notes on the modelling choices and their honest limitations. Citations live in
[`REFERENCES.md`](REFERENCES.md). Status reflects Phase 4.

## Modelling choices
- Leaky-integrator ESN: `x(t+1) = (1-α)·x(t) + α·tanh(Win·u + bias + W·x)`, with a per-neuron leak α drawn from a range. This is standard (Jaeger 2007; Lukoševičius 2012).
- Readout: closed-form ridge regression on reservoir states. The continuous-learning layer adds an exact recursive (RLS) online readout and a conceptor continual readout.
- Washout: the leading timesteps of each window are discarded before fitting, per window rather than on the concatenated stream.
- Spectral radius: the engine warns for ρ≥1 but allows it. A super-critical ρ is defensible for short, strongly-driven windows (Yildiz et al. 2012). Pin the exact ρ used for any reported number.

## Connectome handling
- Undirected by data. Structural connectomes from non-invasive imaging (diffusion MRI / tractography) cannot resolve fibre directionality, so the connectivity matrix is symmetric and `symmetric=True` is correct, not a modelling shortcut. The limitation: a symmetric `W` has a real-eigenvalue-only spectrum (no oscillatory modes), which is dynamically weaker than an asymmetric random ESN. Directed connectomes are out of scope (unrecoverable non-invasively); inducing asymmetry via random edge signs is possible future work.
- Edge weight: pass `edge_attr="number_of_fibers"` (the streamline count), which is the biologically meaningful connection strength. This is NOT the engine default. The committed HCP GraphMLs expose `number_of_fibers`/`FA_mean`/`fiber_length_mean` but no `weight`, and the engine default is `edge_attr="weight"`, so a default-built reservoir on these connectomes is binary topology unless you pass `edge_attr="number_of_fibers"` explicitly (example 07 does). When the requested attribute is absent the engine emits a UserWarning and falls back to an unweighted (binary) adjacency rather than silently discarding the weights; that warning now recommends `number_of_fibers` rather than the alphabetically-first `FA_mean`, which is a tissue-anisotropy scalar, not a weight. Do not read a default-built reservoir on these connectomes as fibre-weighted.
- Combining subjects: multiple subject connectomes are averaged (`combine="mean"`). The committed `data/connectomes/scale83` holds 5 representative HCP subjects (83-node Lausanne parcellation); the full multi-scale dataset (83/129/234/463/1015 nodes, ~1000 subjects each) runs from hundreds of MB to GB and is not committed.

## Null-model baseline (the #1 reviewer requirement)
Implemented in `reservoirs/nulls.py` (`rewire_degree_preserving`, Maslov-Sneppen) and demonstrated in the
null-model section of `examples/combined_examples.ipynb`. A connectome-RC claim, that "the brain *topology* helps",
requires the real connectome to beat a degree-preserving rewired null (the same degree sequence and weight multiset).
Honest finding on Iris: real ≈ null within noise (z≈0.5), both > random ESN. In other words the degree/weight
*distribution* helps but the specific wiring does not, as expected for a near-static task. The topology
advantages reported in the literature appear on memory and temporal tasks.

Honest finding on temporal tasks (now tested). `reservoirs/tasks.py` plus the temporal-benchmark section of
`examples/combined_examples.ipynb` add Memory Capacity (MC) and NARMA-10, each scored as the mean over multiple
reservoir seeds × input draws (never a single draw) and compared against a degree-preserving null *ensemble*, with
an empirical percentile and a permutation p on top of the z-score. On the real 83-node connectome
(`number_of_fibers` weights):
- NARMA-10 (nonlinear prediction, lower NRMSE is better): real 0.656 vs null 0.679 ± 0.006, z ≈ +3.67, with the
  real connectome beating all 200 nulls at p = 0.010 (a higher-power rerun: n_null=200, 4 reservoir seeds × 2
  input draws). The effect is consistent: all 5 individual HCP subjects show positive z, and 3/5 are individually
  significant (z = +3.59, +3.71, +2.88; the other two +1.56, +1.22, n.s.). This is the first defensible positive
  topology signal in this repo, though the effect is small (~3.5% relative NRMSE).
- Memory Capacity (linear memory): real 6.80 vs null 7.27 ± 0.15, with z of about negative 3.1, so here the real
  wiring is *worse* than its null.
Important caveats. First, on both tasks the connectome (real and null) trails a generic signed-asymmetric random
ESN by a wide margin (NARMA 0.41; MC 20.8), so the symmetric connectome path is a handicap for memory. Second, the
reservoir genuinely helps on NARMA (0.66 vs 2.29 with no reservoir), unlike static Iris. Third, the NARMA effect,
while it holds up statistically on the mean connectome, is small and not significant on 2/5 individual subjects,
and the degree-preserving null shuffles weights, so it conflates *wiring* with *weight-placement*. A stricter
weight-preserving null and a real temporal dataset are the next steps before a paper-grade claim.

## Train/test protocol (leakage fix)
Slither windows overlap (stride < window length). The split now defaults to session-grouped
(`group_by_session=True`): whole sessions are held out, so overlapping windows cannot straddle the split. The
original window-shuffle, which leaks and optimistically biases test metrics, remains available as
`group_by_session=False` for reproduction. The leakage-free split honestly shows higher test MSE.

## Continual-learning scenario
Domain-incremental: the class/label set is fixed; only the input distribution shifts across sessions and players.
The benchmark is domain-incremental by design, so there is no scenario argument; construct it as
`ContinualBenchmark(readout)`. Report ACC/BWT/FWT (GEM) and forgetting (RWalk) against joint
(upper) and naive-finetuning (lower) baselines, over multiple task orders and seeds.

## Remaining limitations / to-do for publication
- Short windows vs washout. 25-step windows with washout 5 and leak ~0.1 to 0.3 spend much of each window in transient; revisit the window length or carry state across windows within a session.
- Mock data is not evidence. Examples 03/06 run on synthetic mock gameplay. The angle label is essentially the previous-step heading, so the `prev_sin/prev_cos` features predict it with no reservoir: a no-reservoir ridge on those two features alone scores ~0.6 angle accuracy (17 classes, chance ~0.06), and on the leakage-free session-grouped split it is comparable to or beats the connectome reservoir (the 3-session split is also very noisy: the per-split accuracy swings ~0.33↔0.74). The boost label is `velocity > 110` with velocity an input feature, so boost is ~100% recoverable by a threshold. Call `slither.leaked_feature_baseline(...)` to print the no-reservoir baseline next to any mock number. The mock measures pipeline plumbing, not modelling skill, and must NOT be read as a result. Real scraper sessions are needed for any gameplay claim.
- Upscaling caveat. Projecting a connectome of 83 to 234 nodes up to ~1000 neurons (random projection) largely converts it into a random reservoir; question whether large-N sweep points are meaningful.
- Reproducibility. Seed everything; the RNG draw order (`Win`→`leak`→`Win_bias`) is part of the contract (pinned by the characterization goldens). Pin library versions (numpy/networkx), because the PCG64 stream and `read_graphml` node ordering are version-sensitive.
