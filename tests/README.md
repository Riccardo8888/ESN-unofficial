# tests/

## Characterization (golden) tests, Phase 1

These pin the current behavior of the three engines so the upcoming refactor
(packaging into `reservoirs/`, the back-compat shim, the efficiency rewrite) can
show it did not change any numbers. They were written and passing against the
pre-refactor code.

### Files
- `_characterization_common.py` builds the kept engines deterministically (seed=7, mock
  60-node connectome, fixed tiny input) and returns the arrays to pin via `snapshot()`.
  As of Phase 2 it imports from the `reservoirs/` package (`reservoirs.connectome`, `reservoirs.random`).
- `regen_goldens.py` is run once against pre-package code to write `golden/*.npy` (already done).
- `test_characterization.py` recomputes the kept-engine snapshots, asserts they match `golden/`
  (which is what shows the package reproduces the pre-package behavior bit for bit), and exercises the shim
  (`rhow=` alias, `forward()`, `resize_method`) plus `ErdosRenyiReservoir`. It also verifies the retired
  old engine's `old_native_Win`/`old_native_leak` goldens directly against the new engine
  (`test_old_new_native_rng_identical_static`).
- `test_engines_iris.py` (Phase 3) is the cross-engine guardrail: each package engine on sine-encoded Iris
  must clear an accuracy floor (it ports the old ad-hoc `test_reservoirs.ipynb`).
- `test_smoke.py` (Phase 3) runs the end-to-end slither.io pipeline on the committed mock fixtures.
- `golden/` holds the committed baseline snapshots (19 `*.npy`: 12 kept-engine plus 7 legacy) and `meta.json`.
- All test files run under `pytest tests/` or standalone (`python tests/<file>.py`). Current count: 92 passed.

### What is pinned (19 goldens)
- RNG draws `Win` / `leak` / bias for both connectome engines and the random engine (this pins draw order).
- `W` and its exact spectral radius for both connectome engines and the random engine.
- Reservoir states for a fixed deterministic input (this pins the leaky-integrator dynamics).
- Upscaled `W` at size 120 for both tiling (old) and random-projection (new).

### Invariants also asserted
- `test_upscale_methods_diverge`: tiling differs from random-projection, which documents why `resize_method` is needed.
- `test_old_new_native_rng_identical_static`: at the golden-file level, the retired old engine and the
  new engine drew identical `Win`/`leak` at native size and seed, which justified the shim. Checked
  statically against the committed `old_*` goldens so it survives the old engine's retirement.

### Workflow
```bash
# baseline (already done in Phase 1):
python tests/regen_goldens.py
# after any refactor, this must stay green WITHOUT regenerating:
python -m pytest tests/ -q          # or: python tests/test_characterization.py
```
Only re-run `regen_goldens.py` when an intended behavior change has been reviewed and the new
numbers deliberately accepted. An example is an efficiency rewrite that legitimately alters float
associativity, in which case you widen tolerances or re-baseline with a documented justification.

### Status / dependencies
- numpy 2.4.2, networkx 3.6.1 (recorded in `golden/meta.json`). The PCG64 RNG stream and `read_graphml`
  node ordering are version-sensitive; if these change, expect to re-baseline.
- pytest is a dev dependency (install: `python -m pip install pytest`). The suite also runs without it.

### Current test-file inventory
Beyond the characterization/golden files above, the suite now covers:
- `test_engines_iris.py`, `test_smoke.py` - cross-engine Iris guardrail and end-to-end slither.io pipeline smoke.
- `test_connectome_validation.py` - connectome engine construction/validation.
- `test_baselines.py`, `test_nulls.py`, `test_tasks.py`, `test_tuning.py` - baseline harness, degree-preserving
  null ensembles, temporal (MC/NARMA) tasks, and hyperparameter tuning.
- `test_batch.py` - batched state collection.
- CL layer: `test_online_readout.py` (RLS/online), `test_conceptors.py` (conceptor algebra),
  `test_cl_metrics.py` (GEM/RWalk metrics), `test_continual_benchmark.py` and `test_cl_integration.py`
  (the `ContinualBenchmark` runner end-to-end).
- `test_guards.py` - session-aware train/test splitting guards.

Still open: a pinned numeric golden for the real-connectome pipeline, which waits on committable real
connectomes (Phase 4).
</content>
</invoke>
