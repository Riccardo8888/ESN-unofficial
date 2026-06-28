# Repo Cleanup & Consolidation Plan

**Date:** 2026-06-25
**Status:** PROPOSED — not yet executed. No source files have been modified.
**Goal (confirmed):** turn this branch into a *public, paper-grade* reservoir-computing repo whose deliverable is a **Python package for continuous learning** (online + continual) with a brain-connectome reservoir. See [`docs/CONTINUOUS_LEARNING_DESIGN.md`](docs/CONTINUOUS_LEARNING_DESIGN.md) for the package API design.
**Decisions (confirmed):** package is the product, **notebooks become examples + benchmarks** that exercise the API · standardize on the new connectome engine with a back-compat shim · **archive** redundant files (do not hard-delete) · target a **proper Python package** layout · continuous learning means **both** online (RLS readout) and continual (conceptor/SLDA, no forgetting).

This plan was produced after a full audit: the three reservoir engines were read directly, all 15 notebooks were extracted to code+markdown (outputs stripped) and analyzed by parallel subagents, and three quality reviews were run (literature/methodology, efficiency, regressions/tests).

---

## 1. What's actually here (the mess, decoded)

The repo contains **two parallel research tracks** plus tutorials, each forked into many near-identical copies:

- **Tutorial track** — Echo State Network (ESN) on the **iris** dataset, using random reservoirs (`reservoir.py`).
- **Connectome track** — an ESN whose recurrent matrix is built from **brain connectome GraphML** files, applied to (a) "human data" gameplay and (b) a **slither.io** prediction pipeline.

There are **two competing connectome engines** and **two competing notebook APIs**, which is the root cause of most of the duplication and the latent bugs.

### Engine lineage
| File | Role | Verdict |
|---|---|---|
| `reservoir.py` | Random reservoirs: `Reservoir` (fully-connected), `Reservoir2` (ring), `Reservoir3` (gaussian). `forward()` API, no fit/predict. | **Canonical random baseline** — keep, modernize. |
| `reservoir_ramiro.py` | Copy of `reservoir.py` that tries `from reservoirs.ErdosRenyi import ...`. The package does **not exist**; file comment says it's "broken". | **Archive/delete** — broken; but its intent (a `reservoirs/` package) is exactly the target layout. |
| `brain_connectome_reservoir.py` | **OLD** connectome engine: `forward(u, wout, collect_states)`, param `rhow`, **tiling** upscale. | **Archive after migration** — superseded. |
| `brain_connectome_reservoir_v0_1.py` | **NEW** connectome engine: `fit/predict/transform`, ridge + washout, **random-projection** upscale, "FIX 1-6", validation, docstrings. | **Canonical connectome engine.** |

### Notebook families
| Family | Files | Overlap | Notes |
|---|---|---|---|
| **A — handson tutorials** | `handson_tutorial`, `handson_fully_connected`, `handson_gaussian`, `handson_erdos_renyi` | **~95–98%** | Differ only by which `Reservoir*` class. `handson_tutorial` is the superset (TOC, dual-reservoir demo, most markdown). **`handson_erdos_renyi` is code-identical to `fully_connected`** (both use plain `Reservoir`) — the filename is misleading; there is no real Erdős–Rényi class. `gaussian` is the only carrier of `Reservoir3`. |
| **B — humand_data (long form)** | `humand_data`, `humand_data_1015`, `humand_data_1015_copy`, `humand_data_1015_fixed` | **~90%** | `_fixed` is newest, uses the new engine, and contains **real bug fixes** (the leak/ρ/neuron sweeps in the others never actually varied their loop variable; `u_trn` typo). **But** `humand_data_1015` (old engine, ρ=1.5) holds the only complete, trustworthy results (**test 96.67%**); `_fixed`'s displayed outputs are stale. |
| **C — slither pipeline** | `humand_data_1015_slither_copy`, `..._slither_copy_executed`, `slither_copy`, `..._use_brain_connectome_reservoir_executed` | **~75–80%** | `..._use_brain_connectome_reservoir_executed` (newest, Apr 10) sweeps multiple real connectomes and has genuine executed metrics → canonical. `..._slither_copy` embeds `ConnectomeReservoir` **inline** and self-bootstraps mock data → ideal smoke test. `slither_copy` is the only **random-reservoir baseline** (an ablation). |
| **D — standalone** | `project1`, `slither_io_test`, `test_reservoirs` | n/a | `project1` is the **only** continuous-time/Euler dynamics study. `test_reservoirs` is the **only** old-vs-new cross-engine comparison on iris. `slither_io_test` is an early, superseded slither test. |

### Smells found (must be addressed during cleanup)
- `collect_states_batch` defined **twice** in the sweep notebooks.
- Sweep notebooks hard-code absolute Windows paths to private connectome dirs (`...\folder_path_{83,129,234,463,1015}`) that **do not ship**.
- Many displayed outputs are **stale** (sweeps interrupted by `KeyboardInterrupt`) or **hand-pasted** (`results.txt`, trailing markdown tables).
- `__pycache__/` is committed.
- The 5 `*.py` files next to notebooks are **derived `nbconvert` exports**, not real modules.
- `transform()`'s docstring claims it returns `[T, n_neurons+1]` (bias column) but it actually returns `[T, n_neurons]` — a latent doc/behavior mismatch.

---

## 2. Target structure (proper package)

```
esn-connectome/
├── reservoirs/                 # THE PRODUCT: continuous-learning RC package
│   ├── __init__.py
│   ├── random.py               # from reservoir.py: Reservoir / RingReservoir / GaussianReservoir
│   │                           #   + NEW real ErdosRenyiReservoir (sparse, prob. p) — frozen substrates
│   ├── connectome.py           # canonical = v0_1 + back-compat shim (forward(), rhow= alias); frozen
│   ├── _spectral.py            # shared spectral-radius helpers (power iter / eigvalsh)
│   └── learning/               # the continuous-learning layer (see docs/CONTINUOUS_LEARNING_DESIGN.md)
│       ├── online.py           # OnlineReadout: RLS (default), LMS, NLMS, Kalman, FORCE — partial_fit
│       ├── continual.py        # ConceptorReadout (headline), SLDAReadout, ReplayReadout, EWC/SI/MAS
│       ├── benchmark.py        # ContinualBenchmark: task sequence -> R accuracy matrix
│       └── metrics.py          # cl_metrics: ACC, BWT, FWT (GEM); Forgetting, Intransigence (RWalk)
├── slither/                    # the slither.io application, extracted from the notebooks
│   ├── __init__.py
│   ├── data.py                 # discover_sessions, load_session, normalize_grids, prepare_features,
│   │                           #   convert_to_angle_bins, make_windows, train_test_split_windows (SESSION-aware)
│   ├── metrics.py              # angle_accuracy, boost_accuracy (+ angular MAE)
│   └── mock.py                 # ensure_mock_graph, ensure_mock_data
├── examples/                   # consolidated notebooks = examples + benchmarks of the package API
│   ├── 01_esn_tutorial_iris.ipynb          # merged Group A, connectivity selectable by one variable
│   ├── 02_connectome_iris.ipynb            # merged Group B — NOTE: "humand_data" was connectome-on-IRIS, not gameplay
│   ├── 03_slither_pipeline.ipynb           # merged Group C (base = ..._use_brain_connectome..._executed)
│   ├── 04_continuous_time_dynamics.ipynb   # project1, kept as-is + header
│   ├── 05_online_learning.ipynb            # NEW: RLS streaming readout demo (online axis)
│   └── 06_continual_benchmark.ipynb        # NEW: conceptor/SLDA task-sequence + BWT/FWT/forgetting (continual axis)
├── tests/
│   ├── conftest.py
│   ├── fixtures/               # tiny graphml + tiny session
│   ├── golden/                 # characterization snapshots (.npy)
│   ├── test_random_reservoir.py
│   ├── test_connectome_engine.py
│   ├── test_engines_iris.py    # ported from test_reservoirs.ipynb
│   ├── test_characterization.py
│   └── test_smoke.py           # end-to-end on mock data (ported from inline-class slither notebook)
├── data/                       # mock_user_*/session_* fixtures (already present)
├── generated_artifacts/graphs/mock_connectome.graphml
├── archive/                    # ALL superseded notebooks/scripts preserved here (nothing lost)
├── docs/                       # methodology notes + citations
├── iris.csv
├── requirements.txt            # pinned
├── README.md                   # rewritten
├── LICENSE
└── .gitignore                  # __pycache__, *.pyc, .ipynb_checkpoints
```

**Clutter reduction:** 15 notebooks → **4 canonical** (rest archived); 9 loose `.py` → **~3 package modules + a slither/ subpackage + a tests/ suite**; private absolute paths → committed fixtures.

> **Safety note:** you chose *archive* (not git). Recommend ALSO running `git init` + a baseline commit before Phase 0 so the consolidation is diff-able and recoverable — but the plan does not assume it; archiving is the primary safety net.

---

## 3. File-by-file disposition (nothing meaningful is lost)

> **Phase 2 status (DONE):** the 4 engine `.py` rows below are executed — `reservoir.py`→`reservoirs/random.py`, `brain_connectome_reservoir_v0_1.py`→`reservoirs/connectome.py` (+shim), and `brain_connectome_reservoir.py` + `reservoir_ramiro.py` **moved to `archive/`**. Package reproduces the goldens bit-for-bit.

> **Phase 3 status (DONE):** all 15 notebook rows → consolidated to 4 executed `examples/` + `slither/` package; every original notebook moved to `archive/notebooks/`, the 5 derived `.py` exports to `archive/derived_py/`, `results.txt` to `archive/`. `test_reservoirs`→`tests/test_engines_iris.py`; inline-slither smoke→`tests/test_smoke.py`; `slither_io_test` loader→`slither/data.py`. **17 tests green.** (Discovery: Group B "humand_data" is connectome-on-IRIS, not gameplay — example 02 named accordingly.)

> **On "DELETE" vs your archive-not-delete rule:** `DELETE` below is used *only* for mechanically-regenerable, content-free artifacts (the 5 derived `nbconvert` `.py` exports, `__pycache__/`). Everything with any unique content goes to `archive/`. If you prefer zero deletions, route the derived `.py` to `archive/` too — they're just `nbconvert` output and add no information.

| File | Disposition | Unique content preserved → where |
|---|---|---|
| `handson_tutorial.ipynb` | **MERGE-BASE** → `examples/01_esn_tutorial_iris.ipynb` | TOC, dual-reservoir demo, full markdown narrative. |
| `handson_fully_connected.ipynb` | **ARCHIVE** | Pure subset (the `Reservoir` path) of the base. |
| `handson_gaussian.ipynb` | **ARCHIVE** *(after wiring `Reservoir3` into base)* | The gaussian connectivity path → becomes a selector option in 01. |
| `handson_erdos_renyi.ipynb` | **MERGE into `examples/01` as the ER option** (then ARCHIVE the file) | **RESOLVED: a real `ErdosRenyiReservoir` will be implemented** (sparse random graph, prob. `p`) in `reservoirs/random.py` — the notebook's ER experiment becomes genuine (today it secretly uses plain `Reservoir`). Its 97.5/96.7% accuracy is captured as reference expected-output. |
| `humand_data.ipynb` | **ARCHIVE** | Oldest 83-node prototype, broken path, worst results — nothing unique. |
| `humand_data_1015.ipynb` | **KEEP as reference until reproduced, then ARCHIVE** | The only complete, trustworthy run (**test 96.67%**). Its numbers must be reproduced on the canonical engine before archiving. |
| `humand_data_1015_copy.ipynb` | **ARCHIVE** | Buggy predecessor of `_fixed`. |
| `humand_data_1015_fixed.ipynb` | **MERGE-BASE** → `examples/02_connectome_gameplay.ipynb` | Correct sweep fixes; migrate to shim; reconcile its ρ inconsistency (0.9/1.25/0.8). |
| `humand_data_1015_slither_copy.ipynb` | **FOLD → `tests/test_smoke.py`, then ARCHIVE** | Self-contained mock-data pipeline → becomes the dependency-free smoke test. |
| `humand_data_1015_slither_copy_executed.ipynb` | **ARCHIVE** | Stale/interrupted subset of the canonical slither notebook. |
| `..._slither_copy_use_brain_connectome_reservoir_executed.ipynb` | **MERGE-BASE** → `examples/03_slither_pipeline.ipynb` | Multi-connectome sweep + genuine metrics. |
| `slither_copy.ipynb` | **FOLD random-reservoir ablation into 03, then ARCHIVE** | The only non-connectome baseline → optional engine switch in 03. |
| `project1.ipynb` | **KEEP** → `examples/04_continuous_time_dynamics.ipynb` | Only continuous-time Euler / connectivity-topology study. |
| `slither_io_test.ipynb` | **SALVAGE loader+metrics → `slither/`, then ARCHIVE** | `load_slither_data`, documented zarr schema, sklearn RMSE/boost-accuracy. |
| `test_reservoirs.ipynb` | **CONVERT → `tests/test_engines_iris.py`, then ARCHIVE** | Only 4-engine head-to-head + linear baseline; `collect_states` engine-shim. |
| `reservoir.py` | **MOVE → `reservoirs/random.py`** | Keep old class names as aliases. |
| `brain_connectome_reservoir_v0_1.py` | **MOVE → `reservoirs/connectome.py`** (+ shim) | Canonical engine. |
| `brain_connectome_reservoir.py` | **ARCHIVE after migration** | `forward()` API preserved via shim. **Tiling upscale is NOT reproduced by the shim alone** (new engine uses random projection) — to reproduce old upscaled results, port tiling as `resize_method='tile'`; otherwise scope goldens to native (un-upscaled) size. See §4. |
| `reservoir_ramiro.py` | **ARCHIVE/DELETE** | Broken; no unique content. |
| `*_copy.py`, `*_fixed.py`, `slither_copy.py`, `..._executed.py`, `slither_io_test.py` | **DELETE (regenerate on demand)** | Derived `nbconvert` exports — no unique content. |
| `results.txt` | **ARCHIVE then regenerate from a script** | Hand-pasted; numbers must be reproduced programmatically. |
| `__pycache__/` | **DELETE + gitignore** | — |
| `iris.csv`, `data/`, `generated_artifacts/` | **KEEP** (data/ moves under package root) | Fixtures for tests. |
| `README.md` | **REWRITE** | Currently documents only the OLD engine API. |

---

## 4. Engine consolidation (standardize on new + shim)

1. `reservoirs/connectome.py` = today's `brain_connectome_reservoir_v0_1.py`, plus a **compatibility shim** so old notebooks keep running during migration:
   - Accept `rhow=` as a deprecated alias for `spectral_radius=` (emit `DeprecationWarning`).
   - Add a `forward(u, wout=None, collect_states=False)` method delegating to `transform()`/`predict()` so `forward`-style notebooks run unchanged.
   - **Fix the `transform()` docstring** to say `[T-washout, n_neurons]` (no bias column) — or intentionally add the bias column with a shape test; do not change silently. (Note: the bias-column claim is a *triple* inconsistency — both the `transform()` and `_collect_states` docstrings and `fit()`'s comment assume `[N+1]`, but the code returns `[N]`.)
   - **Add a `resize_method={'project','tile'}` parameter** so the old engine's tiling upscale can be reproduced exactly (`'tile'`) when validating old-notebook results; default `'project'` (the new behavior). Without this, upscaled (target_size > N) goldens will NOT match the old engine.
2. `reservoirs/random.py` = today's `reservoir.py`, classes optionally renamed (`Reservoir`→ keep; add `RingReservoir`, `GaussianReservoir` aliases). Replace the O(N³) `np.linalg.eig` spectral-radius with power iteration (see §6).
3. Old engine `brain_connectome_reservoir.py` is **archived only after** all notebooks are migrated and the characterization goldens pass identically through the shim.

---

## 5. Methodology fixes (required for paper-grade)

The ESN fundamentals (leaky integration, spectral-radius scaling, ridge readout, per-window washout, accuracy metric) are **correct and literature-aligned**. The following are the issues a reviewer would flag — addressing them is the difference between "tidy repo" and "publishable":

1. **Null-model baseline (#1 gap).** The central connectome-RC claim requires comparing the empirical connectome against **degree/strength-preserving rewired nulls** and an equal-size random ESN. None exists today. Add it (this is the headline experiment, per Suárez et al. 2024).
2. **Symmetrization — RESOLVED as a stated *limitation*, not a bug.** The source connectomes are **undirected** (confirmed), so `symmetric=True` is correct and data-driven — it is *not* discarding directionality that existed. **However**, an undirected connectome ⇒ symmetric `W` ⇒ a real-eigenvalue-only reservoir (no complex/oscillatory eigenmodes), which is dynamically weaker than a standard asymmetric random ESN. This must be **disclosed as a limitation in the paper**, with the reason directed connectivity was not used: *non-invasive structural imaging (diffusion MRI / tractography) cannot resolve the directional trajectory of nerve fibres, so edge directionality is physically unrecoverable; obtaining a directed connectome is out of scope due to this measurement limitation and the added modelling complexity.* (Optional future work: induce asymmetry via random edge-sign/weight assignment on the undirected topology, or use directionality priors — flag as future work, not required.)
3. **Train/test leakage in slither.** Windows overlap (stride 15 < length 25) and are shuffled before splitting → contamination. Switch to a **session-level (group) split** (the `session_ids` are already tracked but unused).
4. **Disabled neuron sweep.** In the canonical slither notebook `target_size=N_NEURONS` is commented out, so the "neuron sweep" silently builds the same size every time and "best neurons=1000" is meaningless. Wire it back.
5. **Mock vs real results.** The executed slither numbers were produced on **synthetic mock data** whose labels are a function of the inputs → not evidence. Label mock runs explicitly as smoke tests; the paper's numbers must come from the real connectomes.
6. **Spectral-radius narrative.** Reconcile the contradictory cells: the engine *warns* (does not reject) ρ≥1; ρ>1 is **defensible** for short, strongly-driven windows (Yildiz et al. 2012; conn2res). Fix the `list_rhow` print/text mismatch.
7. **Short-window washout.** 25-step windows with washout=5 and leak≈0.1–0.3 spend most of each window in transient. Revisit window length or carry state across windows within a session.
8. **Upscaling caveat.** Upscaling an 83–234-node connectome to 1000 neurons (random projection) largely converts it into a random reservoir — acknowledge this and question whether large-N sweep points are meaningful.
9. **Citations to add:** Jaeger 2001; Lukoševičius 2012 (*Practical Guide*); Jaeger et al. 2007 (leaky-integrator + effective spectral radius); Yildiz et al. 2012 (ESP re-visited); Suárez et al. 2021 & 2024 (connectome RC / conn2res).

---

## 6. Efficiency improvements (the `KeyboardInterrupt` cause)

The code is correct but leaves 1–2 orders of magnitude on the table. Ranked by impact/effort:

| # | Fix | Where | Effort | Expected win |
|---|---|---|---|---|
| 1 | **Hoist reservoir build + state collection OUT of the sweep loops** — they rebuild the *identical* reservoir hundreds of times | sweep cells in Groups B & C | Low | ~35–175× fewer rebuilds; removes the interrupts |
| 2 | Hoist data load/window/split out of the neuron×ρ loops; restore `target_size` | slither sweep | Low | ~40× less I/O |
| 3 | Replace `np.linalg.eig` (O(N³)) with power iteration / `eigsh`; make spectral radius an attribute not a recomputed `@property` | `reservoirs/random.py` | Low | ~20–100× on that step |
| 4 | Cache `XᵀX` / `XᵀY` per washout, re-solve per ridge α | `compute_wout` | Low | ~7× on α sweeps |
| 5 | **Batch the forward pass across windows** ([B,N] state, GEMM); precompute `Win@U+bias` | all engines' forward | Medium | ✅ DONE (Phase 5): **measured 2–9×** (small-N/many-windows), ~1.3× at large N — the 10–50× estimate was optimistic |
| 6 | `eigvalsh` for the symmetric connectome; reuse `‖Ax‖` in power iter | connectome engine | Low | ~2× + stability |
| 7 | Sparse `W` (csr) for connectomes; float32 in `random.py` | engines | Medium | memory + matvec |

**Reproducibility caveats for the refactor:** preserve the RNG draw order (`Win`→`leak`→`Win_bias`); batched GEMM changes results at ~1e-6 (float associativity); switching the spectral-radius method perturbs `W` slightly. All of these are why the **characterization goldens in §7 must be written first**.

---

## 7. Test suite (must exist before publishing)

**Order matters: write characterization goldens against the CURRENT code first**, so every refactor below proves it didn't move the numbers.

1. **Characterization / golden tests (FIRST):** snapshot, for fixed seed on the committed `mock_connectome.graphml`:
   - `Win`, `leak`, `Win_bias` draws (pins RNG order);
   - `W` and its spectral radius for both engines;
   - upscaled `W` for tiling vs random-projection (documents the divergence);
   - predicted `Y`/states for a fixed tiny input;
   - iris end-to-end accuracy (re-derived on the mock graph, recorded with a ±tolerance).
2. **Unit tests per engine:** constructor validation, spectral-radius accuracy, echo-state/contractivity, fit/predict/transform shapes, washout behavior, resize up/down, `edge_attr` fallback, graphml loading, determinism under seed.
3. **Integration / smoke tests (committed fixtures only):** end-to-end connectome-iris and slither-on-mock pipelines that run **without** the private connectome dirs; plus `nbmake` execution of the canonical notebooks in CI.

---

## 8. Pre-publish checklist

- [ ] Remove all absolute Windows paths; repoint notebooks at `generated_artifacts/graphs/` + `data/mock_user_*`.
- [ ] Clear + re-execute notebook outputs top-to-bottom (kill stale/interrupted/hand-pasted outputs).
- [ ] Delete the derived `.py` exports; regenerate `results.txt` from a script (or drop it).
- [ ] `requirements.txt` (pinned: numpy, networkx, pandas, matplotlib, scikit-learn, zarr, jupyter), `.gitignore` (`__pycache__`), `LICENSE`.
- [ ] Seed everything; document that RNG draw order is part of the contract.
- [ ] Rewrite `README.md` for the unified API + run instructions + expected mock-data accuracy.
- [ ] (Recommended) `git init` + CI running `pytest` + `nbmake`.

---

## 9. Sequencing (phased; each phase is independently reviewable)

- **Phase 0 — Safety & inventory.** Create `archive/`; (recommended) `git init` + baseline commit. Move nothing yet; just snapshot.
- **Phase 1 — Characterization goldens.** Write §7.1 against current code. *Nothing else proceeds until these pass.*
- **Phase 2 — Package the frozen substrate.** Create `reservoirs/` (random + connectome + shim); re-run goldens (must match bit-for-bit through the shim). Archive old engine + `reservoir_ramiro.py`.
- **Phase 2b — Build the continuous-learning layer (THE PRODUCT). ✅ DONE (TDD).** Implemented `reservoirs/learning/`: `online.RLSReadout`/`OnlineReadout` (RLS + LMS/NLMS), `continual.ConceptorClassifier` (forgetting-free), `benchmark.ContinualBenchmark`, `metrics.cl_metrics`. 21 tests green incl. reservoir-integration. (SLDA/Replay/EWC + Kalman readout deferred as optional extensions.)
- **Phase 3 — Consolidate notebooks → examples. ✅ DONE.** Built the 4 merged example notebooks; repointed to the committed mock fixtures; re-executed; archived the rest. **Per user instruction, did NOT chase `humand_data_1015`'s 96.67%** — examples just run and report whatever comes out (~0.70–0.90 on the mock 60-node connectome). Reproducing the real number is deferred to Phase 4 (needs the committable real connectomes).
- **Phase 3b — Continuous-learning examples.** Add `examples/05_online_learning.ipynb` (RLS streaming) and `examples/06_continual_benchmark.ipynb` (conceptor/SLDA task sequence + BWT/FWT/forgetting vs joint-upper & naive-lower baselines, multiple task orders/seeds).
- **Phase 4 — Methodology fixes (science). ✅ DONE.** Null-model baseline (`reservoirs/nulls.py` rewired null + random-ESN, `examples/07`, `adjacency=` engine option; honest result: real≈null>random on iris); session-grouped split now the **default** (leakage fix); committed 5 real HCP connectomes; `docs/REFERENCES.md` + `docs/METHODOLOGY.md` (undirected-connectome + other limitations); edge weight = `number_of_fibers`; CL scenario = **domain-incremental**. (Disabled-neuron-sweep is moot — the sweep notebooks were archived; examples use native size.) Goldens still bit-for-bit.
- **Phase 5 — Efficiency refactors. ✅ DONE (golden-guarded).** Added vectorized `collect_states_batch` (GEMM-batched forward) on both engines via `reservoirs/_batch.py` — additive, so the golden-pinned `forward` is untouched and goldens stay bit-for-bit. Honest speedup 2–9× in the small-N/many-windows regime (the slither case), ~1.3× at large N. Wired into the smoke test + examples 03/07. The O(N³) `eig`→power-iteration swap in `random.py` was deferred (would change `rand_*` goldens for marginal current benefit).
- **Phase 6 — Polish & publish prep. ✅ DONE.** Pinned `requirements.txt`; installable `pyproject.toml`; MIT `LICENSE` (+ HCP-data caveat); `.github/workflows/ci.yml` (pytest + nbmake) verified locally (7 notebooks pass). README install/license/data/CI sections. Local git only (no push). Remaining before public push: confirm HCP redistribution terms + set LICENSE holder name.

Phases 0–3 are *reorganization* (results should not change). Phase 2b is *new feature work* (the CL package). Phases 4–5 *change results on purpose* (science + speed) and must be reviewed against the goldens.

---

## 10. Regression risk register (condensed)

| Risk | Mitigation |
|---|---|
| API unification breaks the `forward()`-style notebooks | Shim provides `forward()` + `rhow=` before any deletion; migrate in same step |
| Spectral-radius regime silently changes (old ρ=1.25 vs new ≤1) → accuracy drop | Pin the exact ρ that produced published numbers; golden test |
| Tiling → random-projection upscale changes every upscaled reservoir | Keep both behind `resize_method=` flag; snapshot both |
| RNG draw-order change → every weight changes at fixed seed | Golden test on `Win/leak/Win_bias`; comment guarding draw order |
| Deleting the notebook with the only good result (96.67%) | Reproduce on canonical engine first; keep as tagged reference if not reproducible |
| Losing the random-reservoir baseline | Fold into notebook 03 as an engine switch; keep `reservoirs/random.py` |
| Stale/hand-pasted outputs cited as results | Clear + re-execute; regenerate `results.txt` |
| Notebooks unrunnable on a fresh clone (private connectome paths) | **RESOLVED:** real connectomes will be committed (subset); repoint notebooks to in-repo paths |

---

## 11. Open questions — RESOLVED (user, 2026-06-25)

1. **Erdős–Rényi → implement for real.** Build a genuine sparse `ErdosRenyiReservoir` (connection prob. `p`) in `reservoirs/random.py` as a 4th connectivity option; fulfils the `reservoir_ramiro.py` import intent. The `handson_erdos_renyi` experiment becomes meaningful.
2. **Private connectome data → commit a representative subset** of `folder_path_{83,129,234,463,1015}` so the paper's numbers reproduce on a fresh clone. (Mind dataset license/ethics + size; prefer `.graphml` under `data/connectomes/`.)
3. **Connectomes are undirected** → `symmetric=True` is correct; document the real-eigenvalue-only dynamics as a limitation with the non-invasive-imaging rationale (see §5.2).
4. **CL scenario = domain-incremental** (label set fixed; input distribution shifts across sessions).

**Remaining minor decision (non-blocking):** standardize the `humand_data` → `human_data` typo during consolidation? (Recommend yes.)

---

*Prepared as a planning artifact only. Awaiting approval before any file is moved, merged, or modified.*
