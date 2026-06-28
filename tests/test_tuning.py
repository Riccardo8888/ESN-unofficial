"""Fast, biting tests for reservoirs.tuning (numpy + pytest only).

Total runtime is a couple of seconds: tiny reservoirs, no heavy tasks.
"""
import numpy as np
import pytest

from reservoirs.tuning import (
    random_search,
    default_reservoir_space,
    train_val_test_split,
    build_reservoir,
)


def test_random_search_finds_optimum():
    # Concave objective peaking at x = 3.0. Random search over 200 draws should land close.
    res = random_search(
        lambda p: -((p["x"] - 3.0) ** 2),
        {"x": (0.0, 6.0)},
        n_iter=200,
        seed=0,
        maximize=True,
    )
    assert abs(res["best_params"]["x"] - 3.0) < 0.3
    assert res["best_score"] > -0.1   # close to the optimum value of 0
    assert res["n_failed"] == 0
    assert res["n_iter"] == 200
    # history is sorted best-first (descending score when maximizing)
    scores = [rec["score"] for rec in res["history"]]
    assert scores == sorted(scores, reverse=True)
    assert res["best_score"] == scores[0]


def test_random_search_reproducible():
    obj = lambda p: p["x"] - 2.0 * p["y"]
    space = {"x": (0.0, 1.0), "y": (-1.0, 1.0)}
    a = random_search(obj, space, n_iter=20, seed=123)
    b = random_search(obj, space, n_iter=20, seed=123)
    # identical params AND scores, in identical order
    assert [r["params"] for r in a["history"]] == [r["params"] for r in b["history"]]
    assert [r["score"] for r in a["history"]] == [r["score"] for r in b["history"]]
    # a different seed yields a different search
    c = random_search(obj, space, n_iter=20, seed=999)
    assert [r["params"] for r in c["history"]] != [r["params"] for r in a["history"]]


def test_space_sampling():
    # constant objective -> every draw recorded, no failures; inspect the sampled params.
    space = {
        "lg": (1e-6, 1e-1, "log"),
        "cat": ["a", "b", "c"],
        "uni": (2.0, 5.0),
    }
    res = random_search(lambda p: 0.0, space, n_iter=600, seed=0, maximize=True)
    assert res["n_failed"] == 0
    lg = np.array([r["params"]["lg"] for r in res["history"]], dtype=float)
    cats = [r["params"]["cat"] for r in res["history"]]
    uni = np.array([r["params"]["uni"] for r in res["history"]], dtype=float)

    # log spec: stays inside [lo, hi] and spans several orders of magnitude
    assert lg.min() >= 1e-6 and lg.max() <= 1e-1
    assert lg.max() / lg.min() > 1e3

    # list spec: only ever yields listed values, and (with 600 draws) all of them appear
    assert set(cats) == {"a", "b", "c"}

    # uniform spec: stays inside range and is not degenerate
    assert uni.min() >= 2.0 and uni.max() <= 5.0
    assert uni.max() - uni.min() > 1.0


def test_random_search_handles_failures():
    # objective raises for x < 1 and returns NaN for x > 5 -> both count as failures.
    def flaky(p):
        x = p["x"]
        if x < 1.0:
            raise ValueError("boom")
        if x > 5.0:
            return float("nan")
        return -((x - 3.0) ** 2)

    res = random_search(flaky, {"x": (0.0, 6.0)}, n_iter=80, seed=1, maximize=True)
    assert res["n_failed"] > 0
    assert np.isfinite(res["best_score"])          # a finite best from the successes
    assert 1.0 <= res["best_params"]["x"] <= 5.0   # the best is in the valid region


def test_minimize():
    res = random_search(
        lambda p: (p["x"] - 2.0) ** 2,
        {"x": (0.0, 6.0)},
        n_iter=200,
        seed=0,
        maximize=False,
    )
    assert abs(res["best_params"]["x"] - 2.0) < 0.3
    assert res["best_score"] < 0.1
    # best-first means ASCENDING score when minimizing
    scores = [rec["score"] for rec in res["history"]]
    assert scores == sorted(scores)


def test_default_reservoir_space_keys():
    space = default_reservoir_space()
    assert set(space) == {
        "spectral_radius", "leak_low", "leak_high", "input_scale", "ridge"
    }
    assert space["input_scale"][2] == "log" and space["ridge"][2] == "log"


def test_train_val_test_split_partitions():
    train, val, test = train_val_test_split(100, val=0.2, test=0.2, seed=0)
    # disjoint, exhaustive, correct sizes
    assert len(test) == 20 and len(val) == 20 and len(train) == 60
    allidx = np.concatenate([train, val, test])
    assert np.array_equal(np.sort(allidx), np.arange(100))
    # reproducible; integer dtype
    train2, val2, test2 = train_val_test_split(100, val=0.2, test=0.2, seed=0)
    assert np.array_equal(test, test2) and np.array_equal(val, val2)
    assert np.issubdtype(train.dtype, np.integer)


def test_build_reservoir_respects_params():
    # leak_low > leak_high on purpose -> must be handled (sorted) without error.
    params = {
        "spectral_radius": 0.9,
        "leak_low": 0.45,
        "leak_high": 0.12,
        "input_scale": 0.5,
        "ridge": 1e-4,
    }
    res = build_reservoir(params, n_inputs=3, n_neurons=40, substrate="random", seed=0)
    assert res.n_neurons == 40
    # measured spectral radius matches the requested one (Reservoir rescales w)
    assert abs(res.spectral_radius - 0.9) < 0.05
    # leak bounds were sorted into [0.12, 0.45]
    assert res.leak.min() >= 0.12 - 1e-9
    assert res.leak.max() <= 0.45 + 1e-9


def test_build_reservoir_random_requires_n_neurons():
    params = {"spectral_radius": 0.9, "leak_low": 0.1, "leak_high": 0.3, "input_scale": 0.5}
    with pytest.raises(ValueError):
        build_reservoir(params, n_inputs=1, n_neurons=None, substrate="random")
