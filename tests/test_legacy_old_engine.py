"""
Legacy characterization — the ARCHIVED old connectome engine.

`brain_connectome_reservoir.py` (forward()/rhow/tiling) was superseded by
`reservoirs.connectome` + shim and moved to `archive/` in Phase 2. We still verify its
Phase-1 goldens here, importing it from `archive/`, so the old baseline stays honest
and reproducible (e.g. if old-notebook numbers ever need to be regenerated).

If the old engine is eventually deleted outright, delete this file too — its goldens
(`old_*`) then become a frozen historical record only.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
ARCHIVE = os.path.join(REPO_ROOT, "archive")
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
GRAPH_DIR = os.path.join(REPO_ROOT, "generated_artifacts", "graphs")
SEED, N_INPUTS, UPSCALE = 7, 4, 120

# import the archived old engine
if ARCHIVE not in sys.path:
    sys.path.insert(0, ARCHIVE)

try:
    import brain_connectome_reservoir as old_mod  # noqa: E402
    _HAVE_OLD = True
except Exception:  # pragma: no cover - skip if old engine was fully removed
    _HAVE_OLD = False


def _tiny(T=20, n=N_INPUTS):
    return np.sin(np.arange(T * n, dtype=np.float64).reshape(T, n) / 3.0)


def _sr_exact(W):
    return np.array(float(np.max(np.abs(np.linalg.eigvals(np.asarray(W, dtype=np.float64))))))


def _golden(name):
    return np.load(os.path.join(GOLDEN_DIR, name + ".npy"))


def _check(cur, name, atol=1e-6):
    np.testing.assert_allclose(cur, _golden(name), rtol=1e-6, atol=atol,
                               err_msg=f"legacy old-engine mismatch for {name}")


def test_old_connectome_native():
    if not _HAVE_OLD:
        print("SKIP: archived old engine not importable")
        return
    u = _tiny()
    r = old_mod.ConnectomeReservoir(N_INPUTS, graph_dir=GRAPH_DIR, rhow=1.25, seed=SEED)
    _check(np.asarray(r.Win), "old_native_Win")
    _check(np.asarray(r.leak), "old_native_leak")
    _check(np.asarray(r.input_bias), "old_native_input_bias")
    _check(np.asarray(r.W), "old_native_W")
    _check(_sr_exact(r.W), "old_native_sr")
    _check(np.asarray(r.forward(u, collect_states=True)), "old_native_states")


def test_old_connectome_upscale_tiling():
    if not _HAVE_OLD:
        print("SKIP: archived old engine not importable")
        return
    r = old_mod.ConnectomeReservoir(N_INPUTS, graph_dir=GRAPH_DIR, rhow=1.25, seed=SEED, target_size=UPSCALE)
    _check(np.asarray(r.W), "old_up120_W")


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print("FAIL", name, "->", repr(e))
    print(f"\n{failures} failure(s)")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
