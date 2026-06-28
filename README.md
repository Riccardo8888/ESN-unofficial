# ESN-unofficial — connectome reservoir computing → continuous learning

A reservoir-computing toolkit built around a **brain-connectome Echo State Network**: the recurrent
weights are a fixed connectome (or a random graph), and only a linear readout is trained. The project
is being turned into a **package for continuous learning** (online + continual) — see
[`docs/CONTINUOUS_LEARNING_DESIGN.md`](docs/CONTINUOUS_LEARNING_DESIGN.md).

> **Status:** active cleanup/build. Phases 0–3 done (packaged engines + consolidated examples + tests);
> the continuous-learning layer (`reservoirs/learning/`) and methodology hardening are next. The full
> plan and current state live in [`CLEANUP_PLAN.md`](CLEANUP_PLAN.md) and [`docs/HANDOFF.md`](docs/HANDOFF.md).

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
archive/        superseded notebooks/engines/exports (kept, not deleted)
docs/           design + handoff docs
```

## Quickstart

```bash
pip install -r requirements.txt          # pinned; or: pip install -e ".[dev]"

# run the examples (each finds the repo root automatically and uses the committed mock fixtures)
jupyter notebook examples/01_esn_tutorial_iris.ipynb

# run the test suite (fast; no private data needed)
pytest tests/ -q                         # 46 tests

# execute all example notebooks (what CI does)
pytest --nbmake examples/ -q
```

```python
from reservoirs.connectome import ConnectomeReservoir
res = ConnectomeReservoir(n_inputs=4, graph_dir="generated_artifacts/graphs", spectral_radius=0.9, seed=7)
states = res.transform(U, washout=10)     # frozen reservoir -> states; train your own readout
```

## Examples

| notebook | what it shows |
|---|---|
| `01_esn_tutorial_iris` | ESN on Iris with **random** reservoirs (fully-connected / ring / gaussian / Erdős–Rényi) |
| `02_connectome_iris`   | the **connectome** reservoir on the same Iris benchmark (compare biological vs random) |
| `03_slither_pipeline`  | end-to-end **slither.io** gameplay prediction on the mock data (angle + boost) |
| `04_continuous_time_dynamics` | continuous-time (Euler) reservoir dynamics across connectivity topologies |
| `05_online_learning` | streaming RLS readout (learning curve + concept-drift tracking via the forgetting factor) |
| `06_continual_benchmark` | `ContinualBenchmark` + `cl_metrics`: catastrophic forgetting (BWT) vs forgetting-free conceptors |
| `07_null_model_baseline` | real connectome vs **degree-preserving rewired null** vs random ESN (the key topology baseline) |

All examples run on the committed **mock** fixtures, so they validate the pipeline rather than make a
scientific claim. To get real results, drop scraper sessions under `data/<user>/session_*` and a real
connectome `.graphml` under `generated_artifacts/graphs/`.

## Notes / limitations

- Structural connectomes from non-invasive imaging are **undirected**, so the reservoir matrix is
  symmetric (real eigenvalues, no oscillatory modes) — a stated limitation, not a bug. Directionality is
  not recoverable non-invasively.
- The slither split **defaults to a leakage-free session-grouped split** (`group_by_session=True`); the
  original window-shuffle (which leaks across overlapping windows and biases test metrics) is available as
  `group_by_session=False` for reproduction.
- See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full list of modelling choices + limitations and
  [`docs/REFERENCES.md`](docs/REFERENCES.md) for citations.

## License & data

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Data:** the connectomes under `data/connectomes/` are derived from third-party neuroimaging data
  (Human Connectome Project) and are **subject to the original provider's terms, not the MIT license**.
  Verify the HCP Open Access Data Use Terms (and add acknowledgement) before redistributing publicly, or
  remove them and point to the source. See [`data/connectomes/README.md`](data/connectomes/README.md).

## CI

`.github/workflows/ci.yml` runs `pytest tests/` + `pytest --nbmake examples/` on push/PR. It activates
once the repo is pushed to a GitHub remote (currently the repo is local-only — pushing is deferred).
