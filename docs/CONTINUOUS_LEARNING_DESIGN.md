# Continuous-Learning Reservoir Package — Design

**Date:** 2026-06-25 · **Status:** PARTIALLY IMPLEMENTED — this doc is the design VISION; only a subset has landed. The sketches below (sections 1–7) describe the intended full design, not all of which exists in code. Read the status table first.

> ## ⚠️ Implementation status (spec ↔ code) — authoritative
> | Component | Status in `reservoirs/learning/` |
> |---|---|
> | `RLSReadout` (methods **rls / lms / nlms**) | ✅ implemented (`online.py`) |
> | `OnlineReadout` (one-hot classifier wrapper) | ✅ implemented (`online.py`) |
> | `ConceptorClassifier` | ✅ implemented as a **per-class conceptor bank, positive-evidence only** (`continual.py`) — NOT the free-subspace / negative-evidence scheme sketched below |
> | conceptor boolean algebra (`conceptor_not/_and/_or`) | ✅ implemented + tested, but **not used by the bundled classifier** (primitives for future free-subspace loading) |
> | `ContinualBenchmark` | ⚠️ implemented but signature is `(readout, scenario='domain_il')` — **no `reservoir` arg**, and `scenario` is currently **inert metadata** (not read). Drives readouts via `partial_fit/predict`, so it does NOT yet drive `ConceptorClassifier` (API mismatch — known gap). |
> | `cl_metrics` (ACC/BWT/FWT/forgetting/intransigence) | ✅ implemented + verified (`metrics.py`) |
> | **Kalman readout, FORCE, SLDA, Replay, EWC, SI, MAS** | ❌ **NOT implemented** — design/future work only. Do not assume these exist. |
> | sklearn `BaseEstimator`/`TransformerMixin`/`check_is_fitted` contract | ❌ **not adopted** — the API is sklearn-*style* by convention (`partial_fit`/`predict`/`classes_`) but does not subclass the mixins or pass `check_estimator`. |
> | square-root/UD-factored P, state-noise injection (RLS stability) | ❌ not implemented — only per-step re-symmetrisation. |
>
> The unverified Cossu et al. (2021) decimals quoted in §5 remain "verify before citing".
**Confirmed goal:** the deliverable is a **public, paper-grade Python package for continuous learning** with a brain-connectome reservoir. The notebook cleanup (see [`../CLEANUP_PLAN.md`](../CLEANUP_PLAN.md)) is the means of extracting a reusable engine; the consolidated notebooks become **examples + benchmarks** that exercise this API.

"Continuous learning" is confirmed to mean **BOTH**:
- **Online / incremental** — the readout updates as data streams in (one `partial_fit` per sample/session), not one batch ridge fit.
- **Continual / lifelong** — a *sequence* of tasks/sessions learned over time without catastrophically forgetting earlier ones.

**Why reservoir computing fits:** the recurrent weights (the connectome) are **frozen**; only the *linear readout* adapts. This (a) makes online readout updates cheap and stable, and (b) unlocks closed-form, readout-only continual-learning methods (SLDA, conceptors) that are unavailable to fully-trained RNNs.

> **Load-bearing distinction (a known reviewer trap):** *online ≠ continual.* Online's failure mode is **recency bias** (recent data dominates); continual's failure mode is **catastrophic forgetting** (old tasks lost). Keep them as separate axes in the API and separate sections in the paper.

---

## 1. Architecture — three strictly separated roles

```
            ┌─────────────────────┐   states X    ┌──────────────────────┐
 input u →  │  FROZEN SUBSTRATE    │ ────────────▶ │   ADAPTIVE READOUT    │ → y
            │  ConnectomeReservoir │               │  online OR continual  │
            └─────────────────────┘               └──────────────────────┘
                  never learns                          only this learns
                                                              ▲
                                                   ┌──────────┴───────────┐
                                                   │ CONTINUAL ORCHESTRATOR│
                                                   │  (task seq + metrics) │
                                                   └──────────────────────┘
```

**Invariant:** the reservoir is a `TransformerMixin` that maps input sequences → states and **never updates its weights**. Everything that "learns" lives in the readout or the orchestrator.

---

## 2. Module layout (extends `CLEANUP_PLAN.md`'s package)

```
reservoirs/
├── random.py          # Reservoir / RingReservoir / GaussianReservoir / ErdosRenyiReservoir (frozen)
├── connectome.py      # ConnectomeReservoir (frozen; W_rec = rescaled UNDIRECTED connectome ⇒ symmetric W)
├── _spectral.py       # spectral-radius helpers
└── learning/
    ├── online.py      # OnlineReadout: RLS (default), LMS, NLMS, Kalman, FORCE
    ├── continual.py   # ConceptorReadout (headline), SLDAReadout, ReplayReadout, EWC/SI/MAS wrappers
    ├── benchmark.py   # ContinualBenchmark harness (runs task sequence → R matrix)
    └── metrics.py     # cl_metrics(R, baselines): ACC, BWT, FWT, Forgetting, Intransigence
```

> The application code (`slither/` data + metrics + mock helpers), the `examples/` notebooks, `tests/`, and the archive/cleanup mechanics are defined in [`../CLEANUP_PLAN.md`](../CLEANUP_PLAN.md); this doc scopes itself to the `reservoirs/` package.

---

## 3. API surface (sketch — final signatures TBD in implementation plan)

```python
# ---- Substrate: frozen reservoir (sklearn TransformerMixin) ----------------
class ConnectomeReservoir(TransformerMixin, BaseEstimator):
    def __init__(self, connectome, input_nodes=None, readout_nodes=None,
                 spectral_radius=0.9, leak_rate=0.3, input_scaling=1.0,
                 symmetric=True, seed=None): ...        # undirected connectome ⇒ symmetric; weights frozen forever
    def fit(self, X, y=None): ...                      # fixes geometry only
    def transform(self, X): ...                        # input seq -> states
    def run(self, X, reset=True): ...                  # streaming state generation

# ---- Online readout: per-step adaptation -----------------------------------
class OnlineReadout(ClassifierMixin, BaseEstimator):   # RegressorMixin variant too
    """method ∈ {'rls','lms','nlms','kalman','force'}; default numerically-stable RLS."""
    def __init__(self, method='rls', forgetting=1.0, ridge=1e-2,
                 mu=0.1, process_noise=0.0, classes=None): ...
    def partial_fit(self, X, y, classes=None):         # `classes` REQUIRED on first call
        # RLS:  a = y - W@x;  g = P@x / (forgetting + x@P@x)
        #       W += outer(g, a);  P = (P - outer(g, x@P)) / forgetting
        return self                                    # sklearn contract: returns self
    def fit(self, X, y): ...                            # batch-ridge convenience (offline upper bound)
    def predict(self, X): ...

# ---- Continual readout: conceptor-based (headline CL path) -----------------
class ConceptorReadout(ClassifierMixin, BaseEstimator):
    def __init__(self, aperture=10.0): ...
    def learn_task(self, states, y, task_id):
        # C = R (R + aperture**-2 · I)^-1 ,  R = states.T @ states / n
        # store new pattern ONLY in free subspace: N = C AND NOT(OR(previous Cs))
        return self
    @staticmethod
    def C_not(C):  return I - C
    @staticmethod
    def C_and(A,B): return inv(inv(A) + inv(B) - I)     # use regularized/pinv in singular case
    @classmethod
    def C_or(cls,A,B): return cls.C_not(cls.C_and(cls.C_not(A), cls.C_not(B)))
    def predict(self, X): ...                           # evidence x^T C x (pos) + NOT-conceptor (neg)

# ---- Continual orchestration + metrics -------------------------------------
class ContinualBenchmark:
    def __init__(self, reservoir, readout, scenario='domain_il'): ...  # slither.io = domain-IL (resolved)
    def run(self, tasks):                               # tasks: ordered [(X,y), ...]
        # train task i; after each, eval ALL tasks -> fill accuracy matrix R[i,j]
        return Results(R=self.R_, baselines=self.baselines_)

def cl_metrics(R, baselines=None, joint=None) -> dict:
    """R[i,j] = acc on task j after training through task i. Pure function of R."""
    # ACC = mean(R[-1, :])                                  (GEM)
    # BWT = mean(R[-1, i] - R[i, i] for i < T-1)            (GEM; <0 = forgetting)
    # FWT = mean(R[i-1, i] - baselines[i] for i >= 1)       (GEM; zero-shot transfer)
    # Forgetting_k = mean_j max_{l<k}(R[l,j] - R[k,j])      (RWalk; stricter than BWT)
    # Intransigence_k = joint[k] - R[k,k]                   (RWalk)
    ...
```

### Design rules (from the literature)
- **Separate `forgetting` (RLS λ) from `ridge` (Tikhonov δ).** They are two different knobs that the literature routinely conflates. `P(0)=δ⁻¹I` is exactly the ridge penalty on the early solution.
- **RLS numerical stability:** default to a **square-root / UD-factored `P`** (keeps `P` symmetric positive-definite); optionally inject tiny reservoir state noise (σ≈1e-4) — Jaeger (2003) found this *crucial* to bound readout weight growth.
- **sklearn contract:** mixins left of `BaseEstimator`; thin `__init__` (no logic); validate in `fit`; learned attrs end in `_`; `partial_fit`/`fit` return `self`; require `classes` on first `partial_fit` (mirror `SGDClassifier`); `check_is_fitted` in `predict`.
- **Conceptors are reservoir-native:** `C` depends only on state statistics `R=E[xxᵀ]`, fully agnostic to whether the recurrent matrix is random or a connectome — so it drops onto `ConnectomeReservoir` directly. This is the recommended continual mechanism.

---

## 4. Evaluation protocol (paper-grade)

**Scenarios** (van de Ven & Tolias 2019/2022) — declare which one each experiment is:
| Scenario | Task ID at test? | Output space | Difficulty |
|---|---|---|---|
| Task-IL | yes | within-task (multi-head) | easiest |
| Domain-IL | no | fixed labels, shifting input dist. | medium |
| Class-IL | no (must infer) | grows with classes seen | hardest, most realistic |

For slither.io: **RESOLVED → domain-incremental (domain-IL).** The class/label set is fixed over time; only the input distribution shifts across sessions/players. Configure `ContinualBenchmark(scenario='domain_il')`. (Note: EWC tends to work *better* in domain-IL than in the class-IL setting where Cossu et al. found it failed — but conceptors/SLDA remain the headline; report EWC as a comparison.)

**Mandatory baselines (every CL result):**
- **Joint / offline upper bound** — readout trained on all tasks pooled (also gives `joint[k]` for intransigence).
- **Naive / finetuning lower bound** — sequential readout, no CL (expect collapse toward chance; Cossu et al. 2021 confirms EWC ≈ chance in class-IL for ESNs, so don't lead with EWC there).

**Report mean ± std over multiple random task orders AND reservoir seeds** (CL is task-order-sensitive).

**Metrics** — all pure functions of the `R` matrix: ACC, BWT, FWT (GEM, Lopez-Paz & Ranzato 2017); Forgetting & Intransigence (RWalk, Chaudhry et al. 2018); optionally the matrix-aggregated REM/FWT (Díaz-Rodríguez et al. 2018).

---

## 5. Key empirical finding to anchor the package

**Cossu et al. (2021), "Continual Learning with Echo State Networks" (ESANN):** a frozen reservoir does **not** by itself prevent forgetting — it relocates to the readout. Class-incremental ESN accuracy (Split-MNIST), *approximate — verify against the ESANN PDF before citing in the paper*: Naive ≈ chance; EWC ≈ chance; LwF ≈ 0.47; Replay ≈ 0.74; **SLDA ≈ 0.88**; Joint ≈ 0.97. The *direction* (SLDA/Replay ≫ EWC ≈ Naive) is the load-bearing claim; the decimals are not yet verified. Takeaway: the reservoir advantage is **cheap access to readout-only CL methods (SLDA, conceptors)**, not immunity to forgetting. → Make conceptors/SLDA the headline; offer EWC/SI/MAS for task-IL completeness only.

---

## 6. Pitfalls / reviewer checklist
1. Report joint-upper + naive-lower bounds every time.
2. Keep online vs continual conceptually and structurally separate.
3. Report over multiple task orders + seeds.
4. Address RLS stability explicitly (square-root form; bounded weights).
5. Declare the CL scenario (task/domain/class-IL).
6. Pin each metric to its source paper (BWT vs RWalk-forgetting differ: immediate vs max-over-past).
7. Full reproducibility: seeds, hyperparameters (λ, δ, aperture, spectral radius, leak), library versions; release the connectome matrix + session splits.

---

## 7. Core citations (full list in research notes)
- **Online readout:** Jaeger (2003, NIPS) RLS-ESN; Sussillo & Abbott (2009, *Neuron*) FORCE; Widrow & Hoff (1960) LMS; Haykin ed. (2001) Kalman/NN.
- **Continual + RC:** Cossu et al. (2021, ESANN) CL-with-ESN; **Jaeger (2014, arXiv:1403.3369; 2017 JMLR 18) Conceptors**; He & Jaeger (2018) Conceptor-Aided Backprop; Liu et al. (2019, NAACL) conceptor CL.
- **Regularization CL:** Kirkpatrick et al. (2017, *PNAS*) EWC; Zenke et al. (2017, ICML) SI; Aljundi et al. (2018, ECCV) MAS; Robins (1995) pseudo-rehearsal.
- **CL evaluation:** van de Ven & Tolias (2019/2022) scenarios; Lopez-Paz & Ranzato (2017, NeurIPS) GEM ACC/BWT/FWT; Chaudhry et al. (2018, ECCV) RWalk; Díaz-Rodríguez et al. (2018) CL metrics.
- **Connectome substrate / precedent libs:** Suárez et al. (2021, *Nat. Mach. Intell.*); Suárez et al. (2024, *Nat. Commun.*) **conn2res**; ReservoirPy (RLS/LMS online nodes); PyRCN.

---

## 8. Items still to verify before implementation
- Exact conceptor incremental-loading operator and singular-case Boolean forms (check arXiv:1403.3369 §3 / JMLR 15-449 PDF).
- conn2res / ReservoirPy version-specific online-learning API (`train()` vs `partial_fit()`).
- ~~Whether the source connectomes are directed or undirected.~~ **RESOLVED: undirected** ⇒ `symmetric=True` is correct; the real-eigenvalue-only dynamics is documented as a *limitation* (directionality is unrecoverable via non-invasive imaging), see [`../CLEANUP_PLAN.md`](../CLEANUP_PLAN.md) §5.2. (Note: `0.5*(A+Aᵀ)` is a genuine averaging op for a generic adjacency, not a no-op.)
