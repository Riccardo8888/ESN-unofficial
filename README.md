# ESN-unofficial: connectome reservoir computing → continuous learning

A reservoir-computing toolkit built around a brain-connectome Echo State Network. The recurrent
weights are a fixed connectome (or a random graph), and only a linear readout is trained. The project
is being turned into a package for continuous learning, both online and continual; see
[`docs/CONTINUOUS_LEARNING_DESIGN.md`](docs/CONTINUOUS_LEARNING_DESIGN.md).

> **Status:** build complete (phases 0 to 6). That covers the packaged engines, the continuous-learning
> layer (`reservoirs/learning/`: online RLS/LMS/NLMS, conceptor continual readouts, and benchmark/metrics),
> consolidated examples, methodology hardening, and 94 tests. It has been reviewed only with an internal
> adversarial self-review (AI multi-agent, self-scored); the findings were addressed, but this has not had
> an external or third-party audit and is not yet paper-grade (see "Honest status" below).

## Layout

```
reservoirs/     frozen reservoir engines (the package)
  random.py       Reservoir / RingReservoir / GaussianReservoir / ErdosRenyiReservoir
  connectome.py   ConnectomeReservoir (connectome-derived; fit/predict/transform + legacy forward())
slither/        slither.io gameplay pipeline (data, features, metrics, mock-data helpers)
examples/       runnable, executed example notebooks (see below)
tests/          pytest suite incl. characterization "golden" tests that pin engine behavior
data/           committed mock demo sessions (data/mock_user_*/session_*)
generated_artifacts/graphs/mock_connectome.graphml   committed 60-node mock connectome
docs/           design + methodology docs (internal process notes under docs/internal/)
ESN_official_paper.pdf   the project write-up (paper)
test_particolari/        exploratory research notebooks (network architectures, reference/J networks, connectome graphs)
lorenz_testing/          Lorenz-system ESN experiments + data
```

## Quickstart

```bash
pip install -r requirements.txt          # pinned; or: pip install -e ".[dev]"

# run the examples (each finds the repo root automatically and uses the committed mock fixtures)
jupyter notebook examples/combined_examples.ipynb

# run the test suite (fast; no private data needed)
pytest tests/ -q                         # 94 tests

# execute all example notebooks (what CI does)
pytest --nbmake examples/ -q
```

```python
from reservoirs.connectome import ConnectomeReservoir
res = ConnectomeReservoir(n_inputs=4, graph_dir="generated_artifacts/graphs", spectral_radius=0.9, seed=7)
states = res.transform(U, washout=10)     # frozen reservoir -> states; train your own readout
```

## Examples

All demos live in one merged notebook, [`examples/combined_examples.ipynb`](examples/combined_examples.ipynb).
The first code cell holds every import, and each section below is one demo, in order:

| section | what it shows |
|---|---|
| ESN tutorial (Iris) | ESN on Iris with random reservoirs (fully-connected / ring / gaussian / Erdős-Rényi) |
| Connectome (Iris) | the connectome reservoir on the same Iris benchmark |
| Slither pipeline | end-to-end slither.io gameplay prediction on mock data (angle + boost) |
| Continuous-time dynamics | continuous-time (Euler) reservoir dynamics across topologies |
| Online learning | streaming RLS readout (learning curve + concept-drift tracking) |
| Continual benchmark | `ContinualBenchmark` + `cl_metrics`: catastrophic forgetting vs forgetting-free conceptors |
| Null-model baseline | real connectome vs degree-preserving null vs random ESN; no measured topological advantage on Iris |
| Temporal benchmark | Memory Capacity + NARMA-10; topology gives a small NARMA edge (z≈+3.67, p=0.010) but hurts MC (z≈−3.1), and both trail a plain random ESN |
| Hyperparameter tuning | random search: on Iris (tuned 0.90 > default 0.83 > linear 0.70, matches logistic 0.90) and on the temporal benchmark |

Every example runs on the committed mock fixtures, so they validate the pipeline rather than make a
scientific claim. To get real results, drop scraper sessions under `data/<user>/session_*` and a real
connectome `.graphml` under `generated_artifacts/graphs/`.

## Status

The numerical core is correct and well-tested (94 tests), but the project's central hypothesis that
real brain-connectome wiring confers a computational advantage is **not supported by the evidence
gathered here.** The contribution of this repo is the baseline-grounded tooling that makes
that verdict honest, not a positive result. Concretely:

- **Static task (Iris): no topological advantage, and the reservoir does not help.** The
  degree-preserving null is statistically indistinguishable from the real connectome (z≈0.5). The
  reservoir pipeline (~0.70 to 0.73) underperforms even a no-reservoir linear model on the raw features
  (one-hot ridge ~0.80; logistic regression ~0.97), because Iris has no temporal structure for a
  reservoir's memory to exploit.
- **Temporal tasks (Memory Capacity + NARMA-10): mostly negative, one small caveated positive.** On
  Memory Capacity the real wiring is *worse* than its own null (z≈−3.1) topology hurts. On NARMA-10 the
  real connectome beats the null with a small but consistent effect (z≈+3.67, p=0.010 on the mean
  connectome; positive on all 5 HCP subjects, individually significant on 3/5), but the gain is only
  ~3.5% relative NRMSE. On *both* tasks the connectome trails a plain signed-asymmetric random ESN by a
  wide margin (NARMA NRMSE 0.41 vs 0.66; MC 20.8 vs 6.8), because the undirected symmetric matrix is a
  real handicap for memory.
- **The one positive signal is not yet paper-grade.** The degree-preserving null shuffles weights, so it
  conflates *wiring* with *weight-placement*; a stricter weight-preserving null and a real temporal
  dataset are the next steps before any topology claim. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- **Always read accuracy beside a baseline.** Use `reservoirs.baselines` (majority-class, no-reservoir
  linear/ridge, and real-vs-null). Mock-data numbers are plumbing checks, not evidence (the mock labels
  are leaked from input features).

## Notes / limitations

- Structural connectomes from non-invasive imaging are undirected, so the reservoir matrix is symmetric
  (real eigenvalues, no oscillatory modes). That is a stated limitation, not a bug. Directionality is
  not recoverable non-invasively.
- The slither split defaults to a leakage-free session-grouped split (`group_by_session=True`). The
  original window-shuffle, which leaks across overlapping windows and biases test metrics, is still
  available as `group_by_session=False` for reproduction.
- See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full list of modelling choices and limitations,
  and [`docs/REFERENCES.md`](docs/REFERENCES.md) for citations.

## License & data

- Code: MIT (see [`LICENSE`](LICENSE)).
- Data: the connectomes under `data/connectomes/` are a subset of the **braingraph.org** structural
  connectome database, derived from **WU-Minn Human Connectome Project (HCP)** imaging data. They are NOT
  under the MIT license; they are redistributed under the WU-Minn HCP Open Access Data Use Terms (permitted
  for derived data by Term 4). If you use them you must accept those terms, not attempt to identify
  subjects, include the HCP acknowledgement, and cite the braingraph.org papers — see
  [`data/connectomes/DATA_TERMS.md`](data/connectomes/DATA_TERMS.md) and
  [`data/connectomes/README.md`](data/connectomes/README.md).

> **Acknowledgement.** Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium
> (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH
> Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell
> Center for Systems Neuroscience at Washington University.

## CI

`.github/workflows/ci.yml` (Ubuntu, Python 3.11/3.12, BLAS threads pinned for deterministic goldens) runs a
fresh-install smoke check, the portable suite (`pytest -m "not golden"`), the bit-frozen `golden`
characterization tests as a separate single-platform regression gate, and `pytest --nbmake examples/`.

It stays dormant until a GitHub remote exists (the connectome data redistribution terms are addressed —
the data ships under the WU-Minn HCP Open Access Data Use Terms; see **License & data** above).
Until then, run the identical gate locally:

```bash
python scripts/local_ci.py     # imports + portable suite + goldens + notebook execution
```

## Original project background

This repository began as **"Reservoir Computing for Connectome Networks"** (original author: Victor
Buendía). That early work — a rate-model / network-topology study — is preserved in the exploratory
notebooks under `test_particolari/` and `lorenz_testing/`, and the write-up is `ESN_official_paper.pdf`.
The original project description is reproduced below for context.

> ### Description
> The rate model computes the firing rate (in spikes per second) of each neuron as a continuous variable.
> The *i*-th neuron has a rate *rᵢ(t)* at time *t*. For *N* neurons the dynamics follow
> *drᵢ/dt = −rᵢ(t) + φ(Σⱼ Aᵢⱼ rⱼ(t))*, with the sigmoid *φ(x) = 1/(1+e⁻ˣ) − 1/2* converting input
> current to firing rate. An Euler integrator integrates this for different network topologies, plotting
> trajectories and the statistics of the mean firing rate *r(t) = (1/N) Σᵢ rᵢ(t)* after the network
> reaches a stationary state (random initial conditions *rᵢ ∈ [0,1]*).
>
> ### Tasks (network topologies, as control reservoirs vs real data)
> 1. **Fully connected** — all links equal weight, *Aᵢⱼ = J/N*.
> 2. **Erdős–Rényi** — nodes connected with probability *p* (*k = pN*), *Aᵢⱼ = J/k* with prob. *p*, else 0.
> 3. **Gaussian random** — fully connected, *Aᵢⱼ ~ 𝒩(0, J²/N)*.
> 4. **Data integration** — real network data *Dᵢⱼ* via *Aᵢⱼ = J·Dᵢⱼ* (*J* an arbitrary scaling factor).
>
> Networks 1–3 serve as control reservoirs for comparison with real connectome data.
