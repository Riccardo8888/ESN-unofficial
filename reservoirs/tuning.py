"""reservoirs.tuning: a small, reusable RANDOM-SEARCH hyperparameter tuner.

This module provides :func:`random_search` (a dependency-light, reproducible random
search over a typed search space) plus helpers to turn a sampled parameter dict into
a concrete reservoir: :func:`default_reservoir_space`, :func:`train_val_test_split`
and :func:`build_reservoir`.

Honesty note

Tuning is performed on a validation split only. The chosen configuration is then
reported on a held-out test split, right beside the baselines (chance, linear,
logistic). Searching hyperparameters does not guarantee the reservoir beats a plain
linear model: the test-set numbers and the baselines next to them are the honest
verdict, and the notebook reports whatever is actually true.

Dependency policy

``numpy`` (plus the stdlib ``math``) are the only top-level imports.
``reservoirs.random``, ``reservoirs.connectome``, ``reservoirs.baselines`` and
``reservoirs.tasks`` are imported LAZILY inside functions, both to avoid import
cycles and to keep the core import light. Everything is computed in float64.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "random_search",
    "default_reservoir_space",
    "train_val_test_split",
    "build_reservoir",
]


# 1. Search-space sampling
def _sample_one(rng, spec):
    """Draw a single value from one search-space spec.

    spec takes one of three forms: a list, which is a discrete categorical choice over
    the list; a 2-tuple ``(lo, hi)``, which is a uniform float in [lo, hi]; or a
    3-tuple ``(lo, hi, "log")``, which is a log-uniform float in [lo, hi] (requires
    lo > 0).
    """
    if isinstance(spec, list):
        # categorical choice over the list (uniform). We index with a drawn integer so the
        # ORIGINAL Python object (and its type) is returned verbatim, equivalent to
        # rng.choice over the list, but without numpy's scalar coercion.
        if len(spec) == 0:
            raise ValueError("Categorical search-space spec (list) must be non-empty.")
        return spec[int(rng.integers(len(spec)))]

    if isinstance(spec, tuple):
        if len(spec) == 2:
            lo, hi = float(spec[0]), float(spec[1])
            return float(rng.uniform(lo, hi))
        if len(spec) == 3 and spec[2] == "log":
            lo, hi = float(spec[0]), float(spec[1])
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError(
                    f"log-uniform spec requires lo > 0 and hi > 0; got ({lo}, {hi})."
                )
            return float(np.exp(rng.uniform(math.log(lo), math.log(hi))))

    raise ValueError(
        "Unsupported search-space spec: expected a list (categorical), a 2-tuple "
        f"(lo, hi) (uniform), or a 3-tuple (lo, hi, 'log') (log-uniform); got {spec!r}."
    )


def _sample_params(rng, space):
    """Draw one full parameter dict, sampling each name in the space's insertion order."""
    return {name: _sample_one(rng, spec) for name, spec in space.items()}


# 2. Random search
def random_search(objective, space, n_iter=30, seed=0, maximize=True):
    """Reproducible random search over a typed hyperparameter ``space``.

    Parameters:
        objective : callable. ``objective(params: dict) -> float``; higher is better
            when ``maximize``.
        space : dict. Maps a parameter name to a spec (see :func:`_sample_one`): a
            list (categorical), a 2-tuple ``(lo, hi)`` (uniform), or a 3-tuple
            ``(lo, hi, "log")`` (log-uniform, requires lo > 0).
        n_iter : int. Number of parameter draws (objective evaluations attempted).
        seed : int. Seed for ``np.random.default_rng``; an identical
            ``(objective, space, n_iter, seed)`` yields an identical history.
        maximize : bool. If True keep the largest score, else the smallest.

    Robustness: if ``objective(params)`` raises OR returns a non-finite or
    non-numeric value, that draw is recorded with score ``-inf`` (maximize) or
    ``+inf`` (minimize) and the search CONTINUES; the number of such draws is reported
    as ``n_failed``. The search never crashes because a single evaluation failed.

    Returns a dict with these keys. ``best_params`` is the best-scoring parameter dict
    and ``best_score`` is its score. ``history`` is the list
    ``[{"params": .., "score": ..}, ...]`` sorted best-first. ``n_iter`` is the number
    of draws attempted, and ``n_failed`` is how many draws raised or returned
    non-finite.
    """
    n_iter = int(n_iter)
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    rng = np.random.default_rng(seed)
    fail_score = -np.inf if maximize else np.inf

    history = []
    n_failed = 0
    for _ in range(n_iter):
        params = _sample_params(rng, space)
        try:
            raw = objective(params)
            score = float(raw)
            if not math.isfinite(score):
                raise ValueError("objective returned a non-finite value")
        except Exception:
            score = fail_score
            n_failed += 1
        history.append({"params": params, "score": score})

    # sort best-first (failures, with +/-inf, naturally fall to the end)
    history.sort(key=lambda rec: rec["score"], reverse=maximize)

    best = history[0]
    return {
        "best_params": best["params"],
        "best_score": float(best["score"]),
        "history": history,
        "n_iter": n_iter,
        "n_failed": int(n_failed),
    }


# 3. A sensible default ESN search space
def default_reservoir_space():
    """A sensible default search space for an ESN reservoir + ridge readout."""
    return {
        "spectral_radius": (0.3, 1.25),
        "leak_low": (0.05, 0.5),
        "leak_high": (0.5, 1.0),
        "input_scale": (0.05, 1.0, "log"),
        "ridge": (1e-8, 1e-1, "log"),
    }


# 4. Train / validation / test split
def train_val_test_split(n, val=0.2, test=0.2, seed=0):
    """Shuffle ``range(n)`` and carve TEST, then VAL, then TRAIN index arrays.

    Returns ``(train_idx, val_idx, test_idx)`` as int arrays. The remainder after
    carving the test and validation blocks becomes the training set.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1.")
    if val < 0 or test < 0 or (val + test) >= 1.0:
        raise ValueError("val and test must be >= 0 and sum to < 1.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(test * n))
    n_val = int(round(val * n))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (
        np.asarray(train_idx, dtype=int),
        np.asarray(val_idx, dtype=int),
        np.asarray(test_idx, dtype=int),
    )


# 5. Build a reservoir from a sampled parameter dict
def build_reservoir(params, n_inputs=1, n_neurons=None, adjacency=None,
                    substrate="random", seed=0):
    """Construct a reservoir from a sampled parameter dict.

    ``params`` must contain ``spectral_radius``, ``leak_low``, ``leak_high`` and
    ``input_scale`` (``ridge`` is consumed by the caller's readout, not here). The
    leak bounds are sorted so ``leak_low > leak_high`` is handled gracefully.

    substrate="connectome" (or a non-None ``adjacency``) builds a
    :class:`reservoirs.connectome.ConnectomeReservoir` (symmetric path); otherwise a
    dense random :class:`reservoirs.random.Reservoir` of ``n_neurons`` neurons.

    NOTE: ConnectomeReservoir emits a UserWarning for spectral_radius >= 1; that is
    expected, and callers may filter it (``warnings.catch_warnings``).
    """
    lo, hi = sorted([float(params["leak_low"]), float(params["leak_high"])])
    if hi <= lo:
        hi = lo + 1e-3

    if adjacency is not None or substrate == "connectome":
        from reservoirs.connectome import ConnectomeReservoir

        return ConnectomeReservoir(
            n_inputs=n_inputs,
            adjacency=adjacency,
            symmetric=True,
            seed=seed,
            spectral_radius=float(params["spectral_radius"]),
            leak_range=(lo, hi),
            input_scale=float(params["input_scale"]),
        )

    from reservoirs.random import Reservoir

    if n_neurons is None:
        raise ValueError(
            "n_neurons is required for substrate='random' (got None)."
        )
    return Reservoir(
        n_inputs,
        int(n_neurons),
        rhow=float(params["spectral_radius"]),
        inp_scaling=float(params["input_scale"]),
        leak_range=(lo, hi),
        seed=seed,
    )
