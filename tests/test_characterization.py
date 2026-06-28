"""
Characterization (golden) tests for the kept engines, now living in the `reservoirs/` package.

These prove that the Phase-2 move (into the package, plus the shim) changed none of the numbers:
the package reproduces the Phase-1 goldens, generated from the original pre-package code,
bit-for-bit. They also cover the new shim surface (the rhow alias, forward(), resize_method) and
the new ErdosRenyiReservoir. Run them under `pytest tests/` or on their own.

The retired old connectome engine's old_* goldens (old_native_Win/old_native_leak) are checked
directly here, against the new engine, in test_old_new_native_rng_identical_static.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _characterization_common import snapshot, build_new, tiny_input, UPSCALE, GRAPH_DIR, N_INPUTS, SEED  # noqa: E402

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
_SNAPS_CACHE = {}


def _snaps():
    if "v" not in _SNAPS_CACHE:
        _SNAPS_CACHE["v"] = snapshot()
    return _SNAPS_CACHE["v"]


def _golden(name):
    path = os.path.join(GOLDEN_DIR, name + ".npy")
    assert os.path.exists(path), f"missing golden {name}.npy: run `python tests/regen_goldens.py`"
    return np.load(path)


def _check(name, atol=1e-6):
    np.testing.assert_allclose(
        _snaps()[name], _golden(name), rtol=1e-6, atol=atol,
        err_msg=f"characterization mismatch for {name} (behavior changed vs golden)",
    )


# golden reproduction (the package reproduces the original behavior)

@pytest.mark.golden
def test_new_connectome_native():
    for k in ("new_native_Win", "new_native_leak", "new_native_Win_bias",
              "new_native_W", "new_native_sr", "new_native_states"):
        _check(k)


@pytest.mark.golden
def test_random_engine_baseline():
    for k in ("rand_R1_win", "rand_R1_w", "rand_R1_leak", "rand_R1_sr", "rand_R1_states"):
        _check(k)


@pytest.mark.golden
def test_upscaled_W_golden():
    _check("new_up120_W")


# invariants and structural properties

def test_upscale_methods_diverge():
    """Random-projection (default) vs tiling upscale produce different, differently-sparse W."""
    proj = np.asarray(build_new(spectral_radius=0.9, n_neurons=UPSCALE).W)
    tile = np.asarray(build_new(spectral_radius=0.9, n_neurons=UPSCALE, resize_method="tile").W)
    assert proj.shape == tile.shape == (UPSCALE, UPSCALE)
    assert not np.allclose(proj, tile), "tiling and random-projection upscale unexpectedly identical"
    assert np.mean(tile != 0) < np.mean(proj != 0), "tiling should be sparser than random projection"


def test_old_new_native_rng_identical_static():
    """Historical invariant, at the golden-file level: the archived old engine and the new engine
    drew identical Win/leak at the native size and seed, which is what justified the back-compat
    shim. We check it statically against the committed goldens so it survives the old engine's
    archival."""
    np.testing.assert_array_equal(_golden("new_native_Win"), _golden("old_native_Win"))
    np.testing.assert_array_equal(_golden("new_native_leak"), _golden("old_native_leak"))


# back-compat shim

def test_rhow_alias_matches_spectral_radius():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via_alias = build_new(rhow=0.9).W
    np.testing.assert_array_equal(via_alias, _snaps()["new_native_W"])


def test_rhow_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_new(rhow=0.9)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_forward_matches_transform():
    r = build_new(spectral_radius=0.9)
    u = tiny_input()
    np.testing.assert_array_equal(r.forward(u, collect_states=True), r.transform(u, washout=0))


# new ErdosRenyiReservoir

def test_erdos_renyi_reservoir():
    from reservoirs.random import ErdosRenyiReservoir
    er = ErdosRenyiReservoir(N_INPUTS, 200, rhow=1.1, density=0.1, seed=3)
    density = float(np.mean(er.w != 0))
    assert 0.06 < density < 0.14, f"ER density {density} far from target 0.1"
    assert abs(float(er.spectral_radius) - 1.1) < 1e-3, "ER spectral radius not scaled to rhow"
    states = er.forward(tiny_input(), collect_states=True)
    assert states.shape == (20, 200) and np.isfinite(states).all()
    # determinism under seed
    er2 = ErdosRenyiReservoir(N_INPUTS, 200, rhow=1.1, density=0.1, seed=3)
    np.testing.assert_array_equal(er.w, er2.w)


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
