# Methodology & Limitations

Paper-grade notes on the modelling choices and their honest limitations. Citations in
[`REFERENCES.md`](REFERENCES.md). Status reflects Phase 4.

## Modelling choices
- **Leaky-integrator ESN**: `x(t+1) = (1−α)·x(t) + α·tanh(Win·u + bias + W·x)`, per-neuron leak α drawn from a range — standard (Jaeger 2007; Lukoševičius 2012).
- **Readout**: closed-form **ridge** regression on reservoir states; the continuous-learning layer adds an exact recursive (**RLS**) online readout and a **conceptor** continual readout.
- **Washout**: leading timesteps of each window are discarded before fitting (per-window, not on the concatenated stream).
- **Spectral radius**: the engine warns for ρ≥1 but allows it; super-critical ρ is defensible for short, strongly-driven windows (Yildiz et al. 2012). Pin the exact ρ used for any reported number.

## Connectome handling
- **Undirected by data.** Structural connectomes from non-invasive imaging (diffusion MRI / tractography) cannot resolve fibre directionality, so the connectivity matrix is symmetric and `symmetric=True` is correct — *not* a modelling shortcut. **Limitation:** a symmetric `W` has a real-eigenvalue-only spectrum (no oscillatory modes), which is dynamically weaker than an asymmetric random ESN. Directed connectomes are out of scope (unrecoverable non-invasively); inducing asymmetry via random edge signs is possible future work.
- **Edge weight = `number_of_fibers`** (streamline count) — the biologically meaningful connection strength. (Note: the original notebooks used `edge_attr="weight"`, which is absent in these GraphMLs, so they silently ran *unweighted*; this repo uses the fibre count.)
- **Combining subjects**: multiple subject connectomes are averaged (`combine="mean"`). The committed `data/connectomes/scale83` holds 5 representative HCP subjects (83-node Lausanne parcellation); the full multi-scale dataset (83/129/234/463/1015 nodes, ~1000 subjects each) is hundreds of MB–GB and is not committed.

## Null-model baseline (the #1 reviewer requirement)
Implemented in `reservoirs/nulls.py` (`rewire_degree_preserving`, Maslov–Sneppen) and demonstrated in
`examples/07_null_model_baseline.ipynb`. A connectome-RC claim ("the brain *topology* helps") requires
the real connectome to beat a **degree-preserving rewired null** (same degree sequence + weight multiset).
**Honest finding on Iris:** real ≈ null within noise (z≈0.5), both > random ESN — i.e. the degree/weight
*distribution* helps but the specific wiring does not, as expected for a near-static task. Topology
advantages in the literature appear on **memory/temporal** tasks; re-run there for a topology claim.

## Train/test protocol (leakage fix)
Slither windows overlap (stride < window length). The split now **defaults to session-grouped**
(`group_by_session=True`) — whole sessions are held out, so overlapping windows cannot straddle the
split. The original window-shuffle (which leaks and optimistically biases test metrics) remains available
as `group_by_session=False` for reproduction. The leakage-free split honestly shows higher test MSE.

## Continual-learning scenario
**Domain-incremental**: the class/label set is fixed; only the input distribution shifts across
sessions/players. `ContinualBenchmark(scenario='domain_il')`. Report ACC/BWT/FWT (GEM) + forgetting
(RWalk) against **joint (upper)** and **naive-finetuning (lower)** baselines, over multiple task orders
and seeds.

## Remaining limitations / to-do for publication
- **Short windows vs washout.** 25-step windows with washout 5 and leak ~0.1–0.3 spend much of each window in transient; revisit window length or carry state across windows within a session.
- **Mock data is not evidence.** Examples 03/06 run on synthetic mock gameplay. The mock task is in fact **~93% solvable from a single input feature** (the previous-heading `prev_sin/prev_cos`, which the angle label is essentially a function of) with no reservoir at all — so the mock numbers measure pipeline plumbing, not modelling skill, and must NOT be read as a result. Real scraper sessions are needed for any gameplay claim.
- **Upscaling caveat.** Projecting an 83–234-node connectome up to ~1000 neurons (random projection) largely converts it into a random reservoir; question whether large-N sweep points are meaningful.
- **Reproducibility.** Seed everything; the RNG draw order (`Win`→`leak`→`Win_bias`) is part of the contract (pinned by the characterization goldens). Pin library versions (numpy/networkx) — the PCG64 stream and `read_graphml` node ordering are version-sensitive.
