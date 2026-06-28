# File Map: what is in this repo and why

**Generated:** 2026-06-25 (independent audit pass). A guided tour of every tracked file and its
objective. Read it alongside [`AUDIT.md`](AUDIT.md) for the critical findings and
[`HANDOFF.md`](HANDOFF.md) for status and history. Where a file has a known problem, the Note column
flags it and points to `AUDIT.md`.

> **Note:** the flags below reference AUDIT.md findings F1 to F12. All of them were remediated on
> 2026-06-25 (see the remediation table at the end of `AUDIT.md`); the flags are kept as a map of
> where each issue *was* and what to read for context. The suite is now 94 tests green.

The repo is a reservoir-computing / Echo State Network (ESN) project that was refactored from about
15 near-duplicate notebooks into a Python package. The product is two installable packages
(`reservoirs/`, `slither/`), one combined example notebook, a test suite, and docs.

---

## Top level

| File | Objective |
|---|---|
| `pyproject.toml` | Package metadata + build config; makes `pip install -e .` work; pytest config; `dev`/`zarr` extras. |
| `requirements.txt` | Pinned runtime+dev deps. numpy/networkx pins are load-bearing (the PCG64 RNG stream and `read_graphml` node order are version-sensitive; the goldens were frozen against them). Installed env matches (numpy 2.4.2, networkx 3.6.1). |
| `README.md` | Public front page: install, layout, quickstart, license/data notes, and an "Honest status" section (no major scientific claim is yet demonstrated). Current: 94 tests; status reworded to "internal self-review, not paper-grade". |
| `LICENSE` | MIT, holder = BAINSA, with an explicit caveat that the HCP connectome data is not MIT-licensed. |
| `CLEANUP_PLAN.md` | The original audit plus the 6-phase consolidation plan (file-by-file disposition, target structure, methodology fixes). A historical planning doc. |
| `iris.csv` | The classic Iris dataset; input for the tutorial examples (01, 02). |
| `.gitignore` | Excludes caches, `_nb_extracts/`, build artefacts. |
| `.github/workflows/ci.yml` | CI: runs `pytest tests/` and `pytest --nbmake examples/`. Verified locally; never run on a real GitHub runner yet (no remote). |

## `reservoirs/`: the reservoir engines (core library)

| File | Objective | Note |
|---|---|---|
| `__init__.py` | Public API surface: exports `ConnectomeReservoir`, the random family plus descriptive aliases, `ErdosRenyiReservoir`, `rewire_degree_preserving`. | Re-exports `Reservoir2/Reservoir3` (meaningless names outside the archived history). |
| `connectome.py` | The central engine. Builds a recurrent matrix `W` from connectome GraphML files (or a raw `adjacency=`), scales to a target spectral radius, runs leaky-integrator dynamics, and offers `fit`/`predict`/`transform` (ridge readout) plus a legacy `forward()` shim and batched `collect_states_batch`. | Three significant issues, see AUDIT F1 (binary weights on real data), F2 (spectral-radius bug reachable via `symmetric=False`), and F5 (docstring default ≠ code default 1.25). |
| `random.py` | The random-connectivity substrates: `Reservoir` (dense uniform), `Reservoir2`/`RingReservoir` (cyclic), `Reservoir3`/`GaussianReservoir`, and a genuine sparse `ErdosRenyiReservoir`. Also holds `dense_spectral_radius` (folded in from the former `_spectral.py`). Ported verbatim from the original `reservoir.py` so the goldens reproduce. | `Reservoir/3` now take an optional `seed=` (default global RNG → goldens preserved; AUDIT F9 fixed). |
| `_batch.py` | `leaky_collect_batch`: vectorized leaky-integrator state collection across a batch of windows (one GEMM per timestep instead of B matvecs). Shared by both engines. | Speedup is modest and batch-size dependent (≈1.5 to 2×; AUDIT F10 docstrings corrected). Dynamics are correct. |
| `nulls.py` | `rewire_degree_preserving`: a Maslov-Sneppen degree-preserving randomized null connectome (the standard "is it the topology?" baseline). | Verified correct: preserves the degree sequence and weight multiset exactly, randomizes wiring, and fails fast on bad input. |
| `baselines.py` | Honest-baseline harness (added 2026-06): `majority_class_baseline`, `linear_readout_baseline` (no-reservoir one-hot ridge), optional `logistic_readout_baseline` (lazy sklearn), `reservoir_vs_null` (z + percentile + permutation p), `compare_classification` (with-intercept linear bar), `format_comparison`. numpy-only at import. | Built and verified adversarially; prints a baseline beside every reservoir number. |
| `tasks.py` | Temporal benchmark (added 2026-06): `narma10` (resample-on-divergence), `memory_capacity_inputs`, `memory_capacity` (chance-floor), `narma_nrmse`, averaged scorers, `memory_capacity_topology` / `narma_topology` (real vs degree-preserving null, signed-asymmetric random ESN capability ref). | Verified: NARMA hand-check, delay-line MC exact, averaging kills single-draw z noise. |
| `tuning.py` | Random-search hyperparameter tuner (added 2026-06): `random_search`, `default_reservoir_space`, `train_val_test_split`, `build_reservoir`. numpy plus stdlib `math` only at import. | Verified: finds known optima, reproducible, handles failing configs gracefully, no val/test leakage. |

### `reservoirs/learning/`: the continuous-learning layer (the main "product")

| File | Objective | Note |
|---|---|---|
| `__init__.py` | Exports the CL API: `RLSReadout`, `OnlineReadout`, `ConceptorClassifier`, conceptor primitives, `ContinualBenchmark`, `cl_metrics`. | |
| `online.py` | Online learning. `RLSReadout` (exact recursive least squares plus LMS/NLMS) and `OnlineReadout` (sklearn-style one-hot classifier on top). The frozen reservoir plus an adapting readout is online/incremental learning. | Math verified: RLS reproduces closed-form ridge to ~1e-16; LMS/NLMS are real, distinct, and converging. |
| `continual.py` | Continual learning (the central method). Conceptor algebra (`C=R(R+a⁻²I)⁻¹`, NOT/AND/OR) plus `ConceptorClassifier`, a per-class conceptor bank. | Conceptor formula and algebra verified correct. But it is walled off from `ContinualBenchmark` (no `partial_fit`): AUDIT F3, the central structural gap. Plain `inv` is fragile at extreme aperture on rank-deficient input (AUDIT F7). |
| `benchmark.py` | `ContinualBenchmark`: drives a sequence of tasks through a readout, builds the R accuracy matrix, and computes the CL metrics. A domain/class-incremental harness. | Requires `partial_fit`, so the conceptor classifier cannot be run through it (F3). |
| `metrics.py` | `cl_metrics`: pure functions for ACC / BWT / FWT (GEM) plus forgetting / intransigence (RWalk) from the R matrix. | All five formulas independently re-derived as correct; no off-by-one. |

## `slither/`: the slither.io gameplay-prediction application

| File | Objective | Note |
|---|---|---|
| `__init__.py` | Package exports for the pipeline. | |
| `config.py` | Constants: angle binning, window length/stride, washout, ridge α, test ratio. | |
| `data.py` | Data loading, feature engineering, windowing, and train/test split. `train_test_split_windows` defaults to session-grouped, the leakage-free protocol. | The mock task is fully leaked (AUDIT F4); the 2-session split silently ignores `test_ratio` (F8). Mechanics are otherwise correct. |
| `metrics.py` | Task metrics (`angle_accuracy`, `boost_accuracy`) plus `compute_wout` (closed-form ridge readout). | Fail-fast guards verified to fire with clear errors. |
| `mock.py` | Generates schema-compatible mock slither sessions plus a mock connectome so the pipeline runs on a fresh clone without the private scraper data. | The mock generation is the source of the label leakage (F4): labels are a near-deterministic function of an input feature. |

## `data/`: committed datasets

| Path | Objective | Note |
|---|---|---|
| `connectomes/scale83/*.graphml` | 5 representative real HCP connectomes (83-node Lausanne parcellation) so the connectome results are reproducible. | These have edge attrs `number_of_fibers`/`FA_mean`/`fiber_length_mean`, but no `weight`. The engine default discards the fibre weights (AUDIT F1). |
| `connectomes/README.md` | Explains the scale33→83-node subset and provenance. | |
| `mock_user_*/session_*/*.npy` + `metadata.json` | The committed mock gameplay sessions (grids, inputs, headings, velocities, …) used by examples 03/05/06 and the smoke tests. | Synthetic; "not evidence" (correctly stated in METHODOLOGY). |
| `generated_artifacts/graphs/mock_connectome.graphml` | A 60-node Erdős-Rényi mock connectome (it has a real `weight` attr) for examples that need a connectome without the HCP data. | |

## `examples/`: executable notebooks (the "benchmarks")

All execute under `pytest --nbmake`. They exercise the package API and double as documentation.

| Notebook | Objective | Note |
|---|---|---|
| `combined_examples.ipynb` | The single merged example notebook (built from the former notebooks 01 through 09; the originals were removed). The first code cell holds all imports; then there is one section per demo, in order: (1) ESN tutorial on Iris (random reservoirs including a genuine ErdosRenyi); (2) connectome reservoir on Iris; (3) full slither pipeline on mock data; (4) continuous-time Euler dynamics; (5) streaming RLS plus concept drift; (6) `ContinualBenchmark` plus `cl_metrics` (catastrophic forgetting vs forgetting-free conceptors); (7) null-model baseline (real vs degree-preserving null, with no measured topological advantage on Iris); (8) temporal benchmark (Memory Capacity plus NARMA-10; topology helps NARMA at z≈+3.4 and hurts MC at z≈-3.1, and both trail a plain random ESN); (9) hyperparameter tuning (random search; tuned Iris 0.90 > default 0.83 > linear 0.70, matching logistic). | Executes under `pytest --nbmake`; the prose contains no double hyphens or em dashes; numbers are sourced at runtime. |

## `tests/`: the test suite (94 tests, ~3s)

| File | Objective | Note |
|---|---|---|
| `_characterization_common.py` | Shared snapshot helper for the golden tests. | Depends on global-RNG seed ordering (fragile by design but deterministic in-test). |
| `golden/*.npy`, `golden/meta.json` | Frozen reference arrays (W, spectral radius, full state trajectories, RNG draws) the characterization tests compare against. | Compared via `assert_allclose(1e-6)`, not regenerated at test time. |
| `regen_goldens.py` | Standalone script to (re)generate the goldens. Not imported by any test (so the goldens are genuinely pinned). | |
| `test_characterization.py` | Pins the engines' dynamics bit-for-(near)-bit. | Genuinely bites (mutation-tested). |
| `test_legacy_old_engine.py` | Verifies the archived old engine still reproduces its goldens from `archive/`. | |
| `test_batch.py` | Batched state collection == per-window forward. | |
| `test_engines_iris.py` | Accuracy floors for the engines on Iris. | Loose floors (regression canaries). |
| `test_smoke.py` | End-to-end pipeline smoke test. | |
| `test_online_readout.py` | RLS=ridge, LMS/NLMS convergence, P symmetry, `classes` contract. | One bare `pytest.raises(Exception)` that the remediation said it would narrow (F12). |
| `test_conceptors.py` | Conceptor eigenvalue bound, AND/OR algebra, classifier separation. | Exact math checks. |
| `test_cl_metrics.py` | Hand-computed checks of ACC/BWT/FWT/forgetting/intransigence. | Verified. |
| `test_continual_benchmark.py` | Benchmark builds R, detects catastrophic forgetting. | Never run with the conceptor classifier (F3). |
| `test_cl_integration.py` | Reservoir states → CL readouts end-to-end. | |
| `test_connectome_validation.py` | 8 constructor guard tests (`pytest.raises(..., match=)`). | Specific. |
| `test_guards.py` | Fail-fast guards (degenerate split, empty fit, zero windows, nulls). | Verified. |
| `test_nulls.py` | Degree-preserving null correctness. | Verified. |
| `tests/README.md` | Test-suite overview. | |

**Coverage note:** the `ConnectomeReservoir.fit()`/`predict()` ridge readout, which previously had zero
coverage (AUDIT F11), is now exercised by `test_connectome_validation.py` (a fit→predict round-trip
plus a predict-before-fit guard).

## `docs/`

| File | Objective |
|---|---|
| `HANDOFF.md` | The entry point: status, decisions, phase log, quality-checkpoint log. |
| `REVIEW.md` | The prior session's self-review (13 agents, scored 68/100) plus a remediation status section. |
| `AUDIT.md` | This independent second-pass audit (the document to read for current critical findings). |
| `FILE_MAP.md` | This file. |
| `CONTINUOUS_LEARNING_DESIGN.md` | The CL package design/spec, with a spec↔code status table marking unimplemented methods. |
| `METHODOLOGY.md` | Modelling choices and honest limitations (symmetrization, leakage, mock-not-evidence, upscaling). Line 14 claims "this repo uses the fibre count", which is true only if you pass the kwarg (F1). |
| `REFERENCES.md` | Citations, with `[planned]` tags for not-yet-implemented methods. |
| `tools/extract_notebooks.py` | Utility to extract lightweight notebook code/markdown (outputs stripped) for auditing. |

## `archive/`: the safety net (nothing deleted)

Holds the original `reservoir.py`, both connectome engines, `reservoir_ramiro.py`, all 15 original
notebooks, 5 derived `.py` exports, and `results.txt`. Kept per the "archive, not delete" decision so
nothing from the messy original is lost. Not part of the shipping package.
