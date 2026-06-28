"""reservoirs.tasks: temporal (memory) benchmarks for reservoir computing.

Why this module exists

The rest of the project keeps evaluating connectome reservoirs on tasks that do
NOT actually require memory (near-static classification such as Iris), and keeps
reporting "significance" from a SINGLE draw. Connectome-topology benefits in the
literature (Suarez et al. 2021/2024) appear on temporal tasks, and a single draw
cannot support a z-score. This module fixes both problems.

First, it uses tasks that genuinely need memory. Memory Capacity (MC) drives the
reservoir with i.i.d. noise u(t) and, for each delay k>=1, trains a linear readout
to reconstruct u(t-k) from the current reservoir state. MC_k is the squared
Pearson correlation (the coefficient of determination of the linear
reconstruction), and the total MC = sum_k MC_k measures how many past inputs the
reservoir linearly remembers. NARMA-10 is a 10th-order non-linear
auto-regressive moving-average system whose next output depends on the last 10
outputs and inputs. Predicting it one step ahead REQUIRES short-term memory and
non-linearity. We score it with the normalized RMSE (NRMSE), where ~1.0 means no
better than always predicting the mean and lower is better.

Second, the reservoir is run in a NEAR-LINEAR, memory-preserving regime
(spectral_radius ~ 0.95, leak ~ 1.0, small input scale): contractive enough for
the echo-state property, slow-fading enough to actually store the recent past. See
TASK_RESERVOIR_KW.

Third, every reported number is AVERAGED over multiple reservoir seeds AND
multiple input draws, never a single draw, and the topology comparison is run
against an ENSEMBLE of degree-preserving rewired nulls, with an empirical
percentile and permutation p-value on top of the z-score.

Two DISTINCT comparisons (do not conflate them)

The TOPOLOGY question asks whether it is the empirical brain wiring that helps, or
just its degree and weight distribution. It pits the real connectome against a
degree-preserving null ensemble. Both sides go through the SAME symmetric
connectome path (``symmetric=True``) so the ONLY thing that differs is which node
pairs are wired. This is what ``memory_capacity_topology`` and ``narma_topology``
test.

The CAPABILITY question asks whether a reservoir can do this task at all, and
whether the connectome even beats a generic reservoir or no reservoir. Here the
reference is a SIGNED, ASYMMETRIC random ESN (``_build_random``), the standard ESN
substrate, NOT routed through the symmetric connectome path, because symmetrising
handicaps the capability baseline. The ``no_reservoir`` control feeds the raw
input as the "state" (expected MC ~ 0, NARMA NRMSE ~ 1).

Important caveat on MC units

MC here is reported BOTH as a raw total and as a FRACTION of N (``mc_fraction =
mc_total / N``). Because chance-level correlation is subtracted per delay and the
readout is regularized, these numbers are NOT directly comparable to the classical
"MC <= N" linear-reservoir results in the literature; treat them as an internal,
relative measure for the real-vs-null and real-vs-random comparisons only.

Everything is computed in float64. ``numpy`` is the only top-level import;
``reservoirs.connectome``, ``reservoirs.random``, ``reservoirs.nulls`` and
``reservoirs.baselines`` are imported lazily inside functions to keep the core
import light and avoid import cycles.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "TASK_RESERVOIR_KW",
    "DEFAULT_WASHOUT",
    "narma10",
    "memory_capacity_inputs",
    "memory_capacity",
    "narma_nrmse",
    "score_mc_adjacency",
    "score_narma_adjacency",
    "score_mc_random",
    "score_narma_random",
    "score_mc_no_reservoir",
    "score_narma_no_reservoir",
    "memory_capacity_topology",
    "narma_topology",
    "format_temporal_report",
]

# Near-linear, memory-preserving regime: contractive (ESP holds) but slow-fading
# so the reservoir actually stores the recent past.
TASK_RESERVOIR_KW = dict(spectral_radius=0.95, leak_range=(0.85, 1.0), input_scale=0.1)
DEFAULT_WASHOUT = 100


# 1. Task generators
def narma10(n_samples, *, seed=0, discard=200, max_resample=50):
    """Generate a NARMA-10 input/target pair.

    s(t) ~ Uniform(0, 0.5); the target obeys the standard 10th-order recurrence

        y[t+1] = 0.3*y[t] + 0.05*y[t]*sum_{i=0..9} y[t-i]
                 + 1.5*s[t-9]*s[t] + 0.1

    NARMA-10 with a uniform(0, 0.5) drive occasionally diverges for an unlucky
    input sequence. We do NOT clip y (clipping would corrupt the target); instead
    we RESAMPLE the input with a fresh seed (``seed + attempt``) and recompute, up
    to ``max_resample`` attempts, raising ``RuntimeError`` if it still diverges.

    Parameters:
        n_samples : int, number of returned timesteps.
        seed : int, base RNG seed for the input sequence.
        discard : int, leading transient timesteps dropped before returning.
        max_resample : int, max divergence-driven resample attempts.

    Returns ``(u, y)``, each ``[n_samples, 1]`` float64, where u is the input and y
    the target.
    """
    n_samples = int(n_samples)
    discard = int(discard)
    L = n_samples + discard

    s = None
    y = None
    for attempt in range(int(max_resample)):
        rng = np.random.default_rng(seed + attempt)
        s_try = rng.uniform(0.0, 0.5, size=L).astype(np.float64)
        y_try = np.zeros(L, dtype=np.float64)
        # y[:10] = 0 by construction; recurrence fills y[10:].
        for t in range(9, L - 1):
            y_try[t + 1] = (
                0.3 * y_try[t]
                + 0.05 * y_try[t] * np.sum(y_try[t - 9:t + 1])
                + 1.5 * s_try[t - 9] * s_try[t]
                + 0.1
            )
        if np.all(np.isfinite(y_try)) and np.max(np.abs(y_try)) <= 1e3:
            s, y = s_try, y_try
            break
    else:
        raise RuntimeError(
            f"narma10 diverged for all {max_resample} resample attempts "
            f"(base seed={seed}); cannot produce a bounded NARMA-10 target."
        )

    u = s[discard:][:, None]
    y = y[discard:][:, None]
    assert u.shape[0] == n_samples, (u.shape, n_samples)
    assert y.shape[0] == n_samples, (y.shape, n_samples)
    return u, y


def memory_capacity_inputs(n_samples, *, seed=0, low=-1.0, high=1.0):
    """i.i.d. uniform input drive for the memory-capacity task.

    Returns ``u`` of shape ``[n_samples, 1]`` (float64), drawn from
    Uniform(low, high) with ``np.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(low, high, size=int(n_samples)).astype(np.float64)
    return u[:, None]


# 2. Readout + metric primitives
def _ridge_readout(Xtr, Ytr, ridge):
    """Closed-form ridge readout in float64:  W = (X^T X + ridge*I)^-1 X^T Y."""
    Xtr = np.asarray(Xtr, dtype=np.float64)
    Ytr = np.asarray(Ytr, dtype=np.float64)
    G = Xtr.T @ Xtr
    G[np.diag_indices_from(G)] += ridge
    return np.linalg.solve(G, Xtr.T @ Ytr)


def _r2(a, b):
    """Squared Pearson correlation between two 1-D arrays; 0.0 on zero variance."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(r * r)


def memory_capacity(states, u, *, max_delay=None, train_frac=0.5, ridge=1e-6,
                    subtract_chance=True, seed=0):
    """Linear memory capacity of a reservoir's state trajectory.

    Parameters:
        states : [T, N], reservoir states, time-aligned to ``u``.
        u : [T] or [T, 1], the input drive (same time index as ``states``).
        max_delay : int or None, max reconstruction delay; default min(2*N, T-10),
            clamped to >= 1.
        train_frac : float, contiguous train fraction (first ``train_frac*len``).
        ridge : float, ridge penalty for the per-delay linear readout.
        subtract_chance : bool, subtract the squared correlation against a SEEDED
            permutation of the test target (a chance floor), clipped at 0.
        seed : int, base seed for the per-delay chance permutation.

    For delay k (>=1) we reconstruct u(t-k) from state(t). ``X = states[k:]`` and
    ``target = u[:-k]`` so state row i reconstructs the input k steps earlier.

    Returns a dict with 'mc_total', 'mc_curve', 'max_delay', 'n_neurons',
    'mc_fraction'.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("states must be 2-D [T, N].")
    u_flat = np.asarray(u, dtype=np.float64).ravel()
    T, N = states.shape
    if u_flat.shape[0] != T:
        raise ValueError(
            f"u (len {u_flat.shape[0]}) must be time-aligned to states (T={T})."
        )
    if max_delay is None:
        max_delay = min(2 * N, T - 10)
    max_delay = max(1, int(max_delay))

    mc_curve = []
    for k in range(1, max_delay + 1):
        X = states[k:]
        target = u_flat[:-k]
        c = int(train_frac * len(X))
        Xtr, ttr = X[:c], target[:c]
        Xte, tte = X[c:], target[c:]
        if Xtr.shape[0] < 1 or Xte.shape[0] < 2:
            mc_curve.append(0.0)
            continue
        W = _ridge_readout(Xtr, ttr[:, None], ridge)
        pred = (Xte @ W).ravel()
        mc_k = _r2(pred, tte)
        if subtract_chance:
            rng = np.random.default_rng(seed + k)
            chance_k = _r2(pred, rng.permutation(tte))
            mc_k = max(0.0, mc_k - chance_k)
        mc_curve.append(float(mc_k))

    mc_total = float(sum(mc_curve))
    return {
        "mc_total": mc_total,
        "mc_curve": mc_curve,
        "max_delay": int(max_delay),
        "n_neurons": int(N),
        "mc_fraction": float(mc_total / N),
    }


def narma_nrmse(states, y, *, train_frac=0.5, ridge=1e-6):
    """Normalized RMSE of a ridge readout predicting ``y`` from ``states``.

    Contiguous train/test split; lower is better. ~1.0 means no better than
    predicting the mean. ``var(y_test) == 0`` returns 0.0 (degenerate target).
    """
    states = np.asarray(states, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    T = states.shape[0]
    if y.shape[0] != T:
        raise ValueError(f"y (len {y.shape[0]}) must be aligned to states (T={T}).")
    c = int(train_frac * T)
    W = _ridge_readout(states[:c], y[:c], ridge)
    pred = (states[c:] @ W).ravel()
    yte = y[c:].ravel()
    denom = float(np.var(yte))
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean((pred - yte) ** 2) / denom))


# 3. State collection + reservoir builders
def _collect_states(reservoir, u, washout):
    """Return post-washout reservoir states ``[T-washout, N]`` for input ``u``."""
    if hasattr(reservoir, "transform"):
        return np.asarray(reservoir.transform(u, washout=washout))
    X = np.asarray(reservoir.forward(u, collect_states=True))
    return X[washout:]


def _build_connectome(adjacency, *, seed, symmetric=True, n_inputs=1, **kw):
    """Build a ConnectomeReservoir from an adjacency in the near-linear regime.

    Both the real connectome and its degree-preserving nulls go through this
    SAME symmetric path, so only the wiring differs (the topology comparison).
    """
    from reservoirs.connectome import ConnectomeReservoir

    regime = {**TASK_RESERVOIR_KW, **kw}
    return ConnectomeReservoir(
        n_inputs=n_inputs, adjacency=adjacency, symmetric=symmetric, seed=seed, **regime
    )


def _build_random(n_inputs, n_neurons, *, seed):
    """Build a SIGNED, ASYMMETRIC random ESN (the capability reference)."""
    from reservoirs.random import Reservoir

    return Reservoir(
        n_inputs, n_neurons, rhow=0.95, inp_scaling=0.1, leak_range=(0.85, 1.0), seed=seed
    )


# 4. Averaged scorers (never score a single draw)
def _input_seed(d):
    """Per-input-draw seed, reused across adjacencies/reservoirs so the comparison is fair."""
    return 1000 * d + 7


def score_mc_adjacency(A, *, n_seeds=5, n_input_draws=3, n_samples=2000,
                       washout=DEFAULT_WASHOUT, seed=0):
    """Mean total memory capacity of a connectome adjacency.

    Averaged over reservoir seeds (0..n_seeds-1) x input draws (0..n_input_draws-1).
    The input seeds (``1000*d + 7``) are reused across adjacencies for fairness.
    """
    vals = []
    for s in range(n_seeds):
        res = _build_connectome(A, seed=s)
        for d in range(n_input_draws):
            u = memory_capacity_inputs(n_samples, seed=_input_seed(d))
            states = _collect_states(res, u, washout)
            vals.append(memory_capacity(states, u[washout:], seed=seed)["mc_total"])
    return float(np.mean(vals))


def score_narma_adjacency(A, *, n_seeds=5, n_input_draws=3, n_samples=2000,
                          washout=DEFAULT_WASHOUT, seed=0):
    """Mean NARMA-10 GOODNESS (= -NRMSE; higher is better) of a connectome adjacency.

    Averaged over the same reservoir-seed x input-draw grid as ``score_mc_adjacency``.
    """
    vals = []
    for s in range(n_seeds):
        res = _build_connectome(A, seed=s)
        for d in range(n_input_draws):
            u, y = narma10(n_samples, seed=_input_seed(d))
            states = _collect_states(res, u, washout)
            vals.append(-narma_nrmse(states, y[washout:]))
    return float(np.mean(vals))


def score_mc_random(n_neurons, *, n_seeds=5, n_input_draws=3, n_samples=2000,
                    washout=DEFAULT_WASHOUT, seed=0):
    """Mean total MC of a signed, asymmetric random ESN of ``n_neurons`` neurons."""
    vals = []
    for s in range(n_seeds):
        res = _build_random(1, n_neurons, seed=s)
        for d in range(n_input_draws):
            u = memory_capacity_inputs(n_samples, seed=_input_seed(d))
            states = _collect_states(res, u, washout)
            vals.append(memory_capacity(states, u[washout:], seed=seed)["mc_total"])
    return float(np.mean(vals))


def score_narma_random(n_neurons, *, n_seeds=5, n_input_draws=3, n_samples=2000,
                       washout=DEFAULT_WASHOUT, seed=0):
    """Mean NARMA-10 NRMSE (positive; lower is better) of a random ESN."""
    vals = []
    for s in range(n_seeds):
        res = _build_random(1, n_neurons, seed=s)
        for d in range(n_input_draws):
            u, y = narma10(n_samples, seed=_input_seed(d))
            states = _collect_states(res, u, washout)
            vals.append(narma_nrmse(states, y[washout:]))
    return float(np.mean(vals))


def score_mc_no_reservoir(*, n_seeds=5, n_input_draws=3, n_samples=2000,
                          washout=DEFAULT_WASHOUT, seed=0):
    """Mean total MC with NO reservoir: the raw input itself is used as the state.

    Expected ~0, because the instantaneous input carries no memory of its own past.
    (``n_seeds`` is accepted for a uniform call signature but unused: there is no
    reservoir randomness to average over.)
    """
    vals = []
    for d in range(n_input_draws):
        u = memory_capacity_inputs(n_samples, seed=_input_seed(d))
        states = u[washout:]
        vals.append(memory_capacity(states, u[washout:], seed=seed)["mc_total"])
    return float(np.mean(vals))


def score_narma_no_reservoir(*, n_seeds=5, n_input_draws=3, n_samples=2000,
                             washout=DEFAULT_WASHOUT, seed=0):
    """Mean NARMA-10 NRMSE (positive) with NO reservoir: raw input as the state.

    Expected ~1, because the instantaneous input cannot predict the NARMA system.
    """
    vals = []
    for d in range(n_input_draws):
        u, y = narma10(n_samples, seed=_input_seed(d))
        states = u[washout:]
        vals.append(narma_nrmse(states, y[washout:]))
    return float(np.mean(vals))


# 5. Topology runners (real connectome vs degree-preserving null ensemble)
def _percentile_pvalue(rvn):
    """Empirical percentile and two-sided permutation p-value from a reservoir_vs_null result.

    ``percentile`` is the fraction of null scores the real score beats; ``p_value`` is the
    standard ``(count + 1) / (n + 1)`` two-sided permutation estimate around ``null_mean``.
    Both use whatever orientation ``rvn`` carries (for NARMA the scores are goodness = -NRMSE,
    so a high percentile / small p still means the real wiring beats the null ensemble).
    """
    ns = np.asarray(rvn["null_scores"], dtype=np.float64)
    percentile = float(np.mean(rvn["real"] > ns))
    p_value = float(
        (np.sum(np.abs(ns - rvn["null_mean"]) >= abs(rvn["real"] - rvn["null_mean"])) + 1)
        / (len(ns) + 1)
    )
    return percentile, p_value


def memory_capacity_topology(adjacency, *, n_null=50, seed=0, **score_kw):
    """Memory-capacity topology test: real connectome vs degree-preserving nulls.

    Uses ``reservoirs.baselines.reservoir_vs_null`` for the null ensemble (z-score)
    and adds an empirical percentile and a two-sided permutation p-value, plus the
    capability baselines (random ESN, no reservoir).

    Returns a dict with: metric, real, null_mean, null_std, z, percentile,
    p_value, random_esn, no_reservoir, n_neurons, n_null. (Higher MC is better, so
    a high percentile / positive z = the real wiring beats the null ensemble.)
    """
    from reservoirs.baselines import reservoir_vs_null

    A = np.asarray(adjacency, dtype=np.float64)
    N = int(A.shape[0])
    rvn = reservoir_vs_null(
        A, score_fn=lambda M: score_mc_adjacency(M, **score_kw), n_null=n_null, seed=seed
    )
    percentile, p_value = _percentile_pvalue(rvn)
    random_mc = score_mc_random(N, **score_kw)
    no_reservoir_mc = score_mc_no_reservoir(**score_kw)
    return {
        "metric": "memory_capacity",
        "real": rvn["real"],
        "null_mean": rvn["null_mean"],
        "null_std": rvn["null_std"],
        "z": rvn["z"],
        "percentile": percentile,
        "p_value": p_value,
        "random_esn": random_mc,
        "no_reservoir": no_reservoir_mc,
        "n_neurons": N,
        "n_null": int(n_null),
    }


def narma_topology(adjacency, *, n_null=50, seed=0, **score_kw):
    """NARMA-10 topology test: real connectome vs degree-preserving nulls.

    Same structure as ``memory_capacity_topology``. For READABILITY the stored
    real/null/random/no_reservoir numbers are the POSITIVE NRMSE (lower is
    better), but z / percentile / p_value are computed on the GOODNESS = -NRMSE,
    so a positive z and a high percentile still mean "the real wiring beats the
    null ensemble".
    """
    from reservoirs.baselines import reservoir_vs_null

    A = np.asarray(adjacency, dtype=np.float64)
    N = int(A.shape[0])
    # score_fn returns goodness = -NRMSE (higher is better) so reservoir_vs_null's
    # z is oriented "positive = real beats null".
    rvn = reservoir_vs_null(
        A, score_fn=lambda M: score_narma_adjacency(M, **score_kw), n_null=n_null, seed=seed
    )
    # rvn scores are goodness = -NRMSE, so a high percentile / small p means real beats null.
    percentile, p_value = _percentile_pvalue(rvn)
    random_nrmse = score_narma_random(N, **score_kw)
    no_reservoir_nrmse = score_narma_no_reservoir(**score_kw)
    return {
        "metric": "narma_nrmse",
        "real": -rvn["real"],            # back to positive NRMSE for readability
        "null_mean": -rvn["null_mean"],  # back to positive NRMSE
        "null_std": rvn["null_std"],     # std is sign-invariant
        "z": rvn["z"],                   # on goodness: positive means real beats null
        "percentile": percentile,        # on goodness: fraction of nulls real beats
        "p_value": p_value,
        "random_esn": random_nrmse,
        "no_reservoir": no_reservoir_nrmse,
        "n_neurons": N,
        "n_null": int(n_null),
    }


# 6. Pretty-printer (numpy / stdlib only)
def format_temporal_report(result: dict) -> str:
    """Render a topology-runner result dict as a compact multi-line table."""
    metric = result.get("metric")
    if metric == "memory_capacity":
        n = result["n_neurons"]
        mc_fraction = (result["real"] / n) if n else 0.0
        lines = [
            "memory capacity (MC): topology test   [higher = better]",
            "========================================================",
            f"real connectome    {result['real']:9.4f}   (MC/N = {mc_fraction:.4f})",
            f"deg-preserv. null  {result['null_mean']:9.4f} +/- {result['null_std']:.4f}  (n={result['n_null']})",
            f"random ESN         {result['random_esn']:9.4f}",
            f"no reservoir       {result['no_reservoir']:9.4f}",
            "========================================================",
            f"z (real vs null)   {result['z']:+9.3f}",
            f"percentile         {result['percentile']:9.3f}   (fraction of nulls beaten)",
            f"empirical p        {result['p_value']:9.4f}",
            f"N neurons          {n:9d}",
        ]
        return "\n".join(lines)
    if metric == "narma_nrmse":
        lines = [
            "NARMA-10 NRMSE: topology test   [LOWER NRMSE = better]",
            "z / percentile computed on goodness = -NRMSE",
            "========================================================",
            f"real connectome    {result['real']:9.4f}",
            f"deg-preserv. null  {result['null_mean']:9.4f} +/- {result['null_std']:.4f}  (n={result['n_null']})",
            f"random ESN         {result['random_esn']:9.4f}",
            f"no reservoir       {result['no_reservoir']:9.4f}",
            "========================================================",
            f"z (goodness)       {result['z']:+9.3f}   (positive means real beats null)",
            f"percentile         {result['percentile']:9.3f}   (fraction of nulls beaten)",
            f"empirical p        {result['p_value']:9.4f}",
            f"N neurons          {result['n_neurons']:9d}",
        ]
        return "\n".join(lines)
    raise ValueError(
        f"format_temporal_report: unrecognized metric {metric!r}; expected a "
        "memory_capacity_topology or narma_topology result."
    )
