# tests/

## Characterization (golden) tests — Phase 1

These pin the **current** behavior of the three engines so the upcoming refactor
(packaging into `reservoirs/`, the back-compat shim, the efficiency rewrite) can
**prove** it did not change any numbers. Written and passing against the pre-refactor code.

### Files
- `_characterization_common.py` — builds the KEPT engines deterministically (seed=7, mock
  60-node connectome, fixed tiny input) and returns the arrays to pin via `snapshot()`.
  **Phase 2:** now imports from the `reservoirs/` package (`reservoirs.connectome`, `reservoirs.random`).
- `regen_goldens.py` — **run once** against pre-package code to write `golden/*.npy` (already done).
- `test_characterization.py` — recomputes the kept-engine snapshots, asserts they match `golden/`
  (proving the package reproduces the pre-package behavior bit-for-bit), and exercises the shim
  (`rhow=` alias, `forward()`, `resize_method`) + `ErdosRenyiReservoir`.
- `test_legacy_old_engine.py` — verifies the `old_*` goldens against the **archived** old engine
  (imported from `archive/`), keeping that baseline honest after archival.
- `test_engines_iris.py` — (Phase 3) cross-engine guardrail: each package engine on sine-encoded Iris
  must clear an accuracy floor (ports the old ad-hoc `test_reservoirs.ipynb`).
- `test_smoke.py` — (Phase 3) end-to-end slither.io pipeline on the committed mock fixtures.
- `golden/` — committed baseline snapshots (19 `*.npy`: 12 kept-engine + 7 legacy) + `meta.json`.
- All test files run under `pytest tests/` **or** standalone (`python tests/<file>.py`). Current: **17 passed**.

### What is pinned (19 goldens)
- RNG draws `Win` / `leak` / bias for both connectome engines + the random engine (pins draw order).
- `W` and its exact spectral radius for both connectome engines and the random engine.
- Reservoir states for a fixed deterministic input (pins the leaky-integrator dynamics).
- Upscaled `W` at size 120 for **both** tiling (old) and random-projection (new).

### Invariants also asserted
- `test_upscale_methods_diverge` — tiling ≠ random-projection (documents why `resize_method` is needed).
- `test_old_new_native_rng_identical_premigration` — old & new engines draw **identical** `Win`/`leak`
  at native size+seed (justifies the shim; retire after the old engine is archived).

### Workflow
```bash
# baseline (already done in Phase 1):
python tests/regen_goldens.py
# after any refactor — must stay green WITHOUT regenerating:
python -m pytest tests/ -q          # or: python tests/test_characterization.py
```
Only re-run `regen_goldens.py` when an **intended** behavior change has been reviewed and the new
numbers deliberately accepted (e.g. an efficiency rewrite that legitimately alters float associativity —
then widen tolerances or re-baseline with a documented justification).

### Status / dependencies
- numpy 2.4.2, networkx 3.6.1 (recorded in `golden/meta.json`). The PCG64 RNG stream and `read_graphml`
  node ordering are version-sensitive; if these change, expect to re-baseline.
- pytest is a dev dependency (install: `python -m pip install pytest`). The suite also runs without it.

### Not yet covered (later phases)
- Full unit tests per engine (constructor validation, washout edge cases, resize down, edge_attr fallback,
  graphml loading) — to land alongside the Phase-5 engine consolidation.
- CL-layer tests (RLS update, conceptor algebra, cl_metrics) — Phase 2b, TDD against `docs/CONTINUOUS_LEARNING_DESIGN.md`.
- End-to-end iris/slither pipeline **smoke** is now covered (`test_engines_iris.py`, `test_smoke.py`); a pinned
  numeric golden for the real-connectome pipeline waits on the committable real connectomes (Phase 4).
