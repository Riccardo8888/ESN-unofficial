# Independent Audit, 2026-06-25

**Score: 63/100** (vs the prior self-review's 68/100). A second, independent critical pass that did not
trust the docs or [`REVIEW.md`](REVIEW.md). Five adversarial agents (engines, CL layer,
slither/methodology, test-suite, docs/meta), each backing every claim with code it ran, plus
orchestrator verification of the key findings. The brief was explicitly to be critical of the work and
to critically evaluate the prior review's comments.

## Big-picture verdict

The numerical substrate is genuinely good and trustworthy. Two independent agents plus a direct check
confirm: RLS reproduces closed-form ridge to ~1e-16, the conceptor formula `C=R(R+a⁻²I)⁻¹` and its
boolean algebra are Jaeger-correct, all five GEM/RWalk CL metrics are correctly derived, the leaky-ESN
dynamics are consistent across all three code paths, the degree-preserving null is exact, the
`ErdosRenyiReservoir` is genuine, and the goldens really bite (11/15 key source mutations were caught
by the right test at tight tolerance). The null-model example (07) is the most careful artefact in the
repo: an 8-draw ensemble, z≈0.47, no spin. The remediation pass did real, verifiable work (the
design-doc status table, the `[planned]` citation tags, the dead-code removal, and the honesty-pass
docstrings all check out).

But the "paper-grade continuous-learning package with a brain-connectome reservoir" framing is still
aspirational. The key new result of this pass is that the prior review missed or rationalized away the
three most damaging problems:

1. On the committed real connectomes, the engine silently runs without the fibre-count weights the
   methodology says it uses (the central "brain-connectome" premise is undercut by a default).
2. The spectral-radius bug that the prior review declared "unreachable" is in fact reachable through a
   public parameter, with a demonstrated 60-million-× mis-scaling.
3. The central continual method cannot be evaluated by the package's own benchmark, and the reason
   given for not fixing it is empirically false (a 9-line adapter yields BWT = -1.0).

Net: above-average research code with a solid, well-tested core, wrapped in claims the artefact does
not yet back up. All findings are eminently fixable.

---

## Confirmed findings (severity-ranked)

### [MAJOR] F1: The committed "brain-connectome" reservoir silently discards the real connection weights
**Where:** `reservoirs/connectome.py:78,155-156` (default `edge_attr="weight"` plus silent fallback);
`data/connectomes/scale83/*.graphml`; claim at `docs/METHODOLOGY.md:14`.
**Evidence (orchestrator-verified):** the scale83 GraphMLs expose `number_of_fibers`, `FA_mean`, and
`fiber_length_mean`. There is no `weight` attribute. With the default, `_edge_attr_exists_any` returns
`False`, so `edge_attr` is set to `None` and `nx.to_numpy_array(weight=None)` builds a binary
adjacency; combined across the 5 subjects it yields a 5-level cross-subject edge-frequency matrix
(5 distinct nonzero magnitudes). Passing `edge_attr="number_of_fibers"` instead yields 919 distinct
magnitudes (the actual fibre counts, roughly 1.5 to 10150). METHODOLOGY.md:14 states *"this repo uses
the fibre count"*, but only example 07 passes the kwarg; the default, the quickstart pattern, and
example 02 do not. This was missed by the 13-agent review, the remediation, and the design doc, and it
contradicts a written methodology claim. It is the single most misleading thing in the repo: a reader
pointing the engine at the shipped connectomes gets topology-only wiring, not the biological weighting
the premise rests on.
**Fix:** make the default `edge_attr=None` (force an explicit choice), or warn clearly when the
requested attribute is absent and a named alternative (`number_of_fibers`) exists; update all examples
and METHODOLOGY to match.

### [MAJOR] F2: Spectral-radius mis-scale is reachable via the public `symmetric=False` param (the prior review's "unreachable" downgrade is wrong)
**Where:** `reservoirs/connectome.py:456-491` (`_spectral_radius`, Rayleigh-quotient power iteration);
reached via `symmetric=False` (`connectome.py:88,174`) or a non-symmetric `adjacency=`.
**Evidence (engine agent, end-to-end through the public API):** the convergence test is on the
Rayleigh quotient `λ = xᵀAx`; for a complex-dominant or near-rotation matrix it "converges" in one
iteration to a wrong (~0) value, and the `eigvals` fallback (which only fires on non-finite/non-positive)
never triggers. A rotation with true ρ=2.0 returns 0.347; a cyclic-shift with true ρ=1.0 returns 0.444.
Driven end-to-end, `ConnectomeReservoir(adjacency=complex_dominant_A, spectral_radius=0.9,
symmetric=False)` produces ρ(W) ≈ 5.5e7, a catastrophic echo-state-property violation. The prior
review (REVIEW.md:22-24,81,118) repeatedly asserts "no current path triggers it … the engine always
symmetrises," and deferred the fix on that basis. That is false: `symmetric=False` is a documented
public constructor option. The default symmetric path is correct (verified), so the blast radius is
limited to non-symmetric use, but it is a live MAJOR bug behind a now-corrected docstring, not a
dormant one.
**Fix:** use the iterate magnitude `λ = ‖Ax‖` (correct for any A) or route non-symmetric input to
`np.linalg.eigvals`; do not change the symmetric path's numerics (goldens).

### [MAJOR] F3: The central continual method is walled off from its own benchmark; the deferral's justification is empirically false
**Where:** `reservoirs/learning/benchmark.py:19-33` (calls `partial_fit`/`predict`) vs
`reservoirs/learning/continual.py:59,75` (`ConceptorClassifier` exposes `learn_class`/`predict`, no
`partial_fit`). Deferral recorded at `REVIEW.md:117`.
**Evidence (CL agent):** the remediation deferred wiring conceptors into `ContinualBenchmark` with
*"a per-class bank is forgetting-free by construction (its BWT≈0 is trivial), so the benchmark adds
little."* The CL agent wrote a 9-line `partial_fit` adapter and ran it through the unmodified
`ContinualBenchmark`: on a disjoint-subspace stream → R=I, BWT=0 (the deferral's only valid case); on
a nested/overlapping-subspace class-incremental stream (the scenario the docs call "hardest, most
realistic") → `{'acc': 0.333, 'bwt': -1.0, 'forgetting': 1.0}`, a total decision collapse. So BWT is
not trivially ≈0; the benchmark produces exactly the number a paper reader needs, and the deferral
suppresses it behind an incorrect claim. The representation is forgetting-free (byte-identical
conceptors, verified); the decisions are not.
**Fix:** add the ~10-line `partial_fit(X,y,classes)` adapter (accumulate per-class state correlation,
refit each class's conceptor) so the package's main method can report its central metrics, or stop
calling it the "headline" method. Pick one.

### [MAJOR] F4: The mock slither task is fully leaked; the reservoir is *beaten* by the leaked feature, so mock numbers validate nothing
**Where:** `slither/mock.py:40-48` (mock generation) plus `slither/data.py:96-102` (`prev_sin/prev_cos`)
vs `slither/data.py:106-122` (`convert_to_angle_bins`, the label). Understated at `METHODOLOGY.md:39`,
rated "minor" at `REVIEW.md:30-32`.
**Evidence (slither agent, computed):** the label is the heading-angle bin; `prev_sin/prev_cos` are the
sin/cos of the previous step's input angle of a smooth `cumsum` random walk, so previous ≈ current.
corr(prev-angle, label) = 0.912 (mean over sessions). On the leakage-free session-grouped split, the
full 60-neuron connectome reservoir angle accuracy = 0.693, but a no-reservoir ridge on just the two
leaked features = 0.830: the reservoir is worse than the feature it is fed. The boost label is 100%
recoverable by thresholding the velocity input feature (`boost ≡ velocity>110`), which is undocumented
anywhere. The prior review's "~93% from one feature" is the best single session (the true cross-session
mean is ≈0.68 to 0.83) and points the wrong way: it frames the leak as "mock too easy" rather than
"model loses to the leak."
**Fix:** decouple labels from echoed input features, or report the no-reservoir baseline next to every
mock number. Document the boost leak. (This is correctly a CRITICAL caveat for honesty, not a code bug.)

### [MINOR] F5: `spectral_radius` docstring contradicts the code default and the safe-default advice
**Where:** `reservoirs/connectome.py:56-57` (docstring: *"Must be < 1 … Default: 0.9"*) vs
`connectome.py:86` (`spectral_radius: float = 1.25`).
**Evidence:** the actual default is 1.25, which is super-critical, and it fires the ESP warning on every
default construction. A user trusting the docstring expects a safe contractive reservoir and silently
gets an edge-of-chaos one. (Super-critical ρ is defensible for short driven windows, but the docstring
should not say the opposite of the code.)
**Fix:** make the docstring state the real default and its rationale, or change the default to 0.9.

### [MINOR] F6: Remediation claimed to fix the stale `[N+1]` bias-column comments; three remain verbatim
**Where:** `reservoirs/connectome.py:282` (`# [T-w, N+1]`), `:286` (`# Shape: [N+1, K]`), `:320`
(`# [T-w, N+1]`). Claim at `REVIEW.md:108` ("Fixed … stale FIX-5 'constant-1 column' claim").
**Evidence (docs/meta agent, grep-confirmed):** only the two prose docstrings (`_collect_states`,
`transform`) were updated; the inline comments the review explicitly named were not. There is no bias
column (it is `[T-w, N]`); a maintainer reading `Wout` shape `[N+1, K]` would mis-shape the readout.
This is a small but telling "claimed-but-not-done" inside the section whose entire purpose is honesty.
**Fix:** correct the three comments to `[T-w, N]` / `[N, K]`.

### [MINOR] F7: `conceptor_from_states` uses plain `inv`; invalid eigenvalues / crash at extreme aperture on rank-deficient input
**Where:** `reservoirs/learning/continual.py:28`. (Prior review called plain `inv` "NOT a live bug.")
**Evidence (CL agent):** with `n·T < N` (rank-deficient R) the aperture term is the only regularizer;
aperture=1e6 → eigenvalues escape [0,1] (-1.08e-3 … 1.0002, an invalid conceptor); aperture=1e9 →
`LinAlgError: Singular matrix` (a hard crash). The default aperture 4 to 8 on full-rank states is fine
(orchestrator-verified: eigenvalues in [0.86, 0.97]). Narrow, but it can crash, contrary to the prior
review. Also note: the prior review's prescribed fix (normalize evidence by `xᵀx`) is insufficient for
the shared-subspace scale-bias; only a negative-evidence term fixes that (as the CL agent demonstrated).
**Fix:** `np.linalg.solve(R + a⁻²I, R.T).T` or symmetric eig with eigenvalue clamping; document aperture
as the inversion regularizer.

### [MINOR] F8: Two-session grouped split silently ignores `test_ratio` (always 50/50)
**Where:** `slither/data.py:171`. Documented as deferred in REVIEW.md but still live.
**Evidence:** `n_test=max(1,int(len(uniq)*test_ratio))` → 1 of 2 for any ratio ≤0.5; with the default
3 mock sessions, `test_ratio=0.25` actually holds out 33%, not 25%.
**Fix:** warn when the effective fraction materially deviates from `test_ratio`.

### [MINOR] F9: `Reservoir`/`Reservoir2`/`Reservoir3` reproducibility is genuinely fragile
**Where:** `reservoirs/random.py:26-29,154-161`. **Evidence:** they draw from the bare global
`np.random.*` with no `seed`/`rng`; one stray `np.random.*` call before construction changes `w`
(verified). `ErdosRenyiReservoir`/`ConnectomeReservoir` are seeded and reproducible.
**Fix:** add an optional `seed`/`rng` (default to the global RNG to preserve the goldens).

### [MINOR] F10: `collect_states_batch` "~B× faster" claim is overstated
**Where:** `reservoirs/_batch.py:4-6`, `connectome.py:372`, `random.py:70-73`.
**Evidence (engine agent):** the measured speedup is 1.5 to 1.8× (not B×), with a reproducible cliff
where batched is 3 to 9× slower than the loop at some batch sizes (e.g. B≈64, N=300), unchanged with
single-thread BLAS. Dynamics are correct.
**Fix:** soften the docstrings; note the batch-size caveat.

### [MINOR] F11: `ConnectomeReservoir.fit()`/`predict()` ridge readout has zero test coverage
**Where:** `reservoirs/connectome.py:250-321`. **Evidence (test-suite agent, mutation M13):** zeroing
the entire ridge solve (`Wout = 0`) passes all 61 tests. The iris tests do their own solve and call only
`transform`/`forward`; every `.predict()` in the suite targets the CL readouts. The engine's main
supervised path and its `predict`-before-`fit` guard are unverified. This was missed by the prior review.
**Fix:** add a connectome `fit→predict` round-trip test (asserts that it learns and that
predict-before-fit raises `RuntimeError`).

### [NIT] F12: Residual doc/comment and test-narrowing items
- `connectome.py:421` asserts `P⁺ = Pᵀ when rows are unit-norm`, which is false (unit-norm rows ≠ orthonormal;
  `‖pinv(P)-Pᵀ‖≈0.48`). No runtime bug (W is rescaled after resize) but the justification is wrong and
  compounds the (correctly flagged) topology densification of project-upscaling.
- `test_online_readout.py:94` still uses bare `pytest.raises(Exception)` where the code raises a specific
  `ValueError("classes …")`; the remediation said it would narrow this to `match='classes'` and did not.
- The README status line and the "46 tests" figure are stale (the CL layer is built; 61 tests).

---

## Strengths (verified, for fairness)
- RLS equals closed-form ridge to ~1e-16; LMS/NLMS are real and distinct (the prior NLMS-mislabel is
  fixed, with no lingering duplicate).
- The conceptor formula and the NOT/AND/OR algebra are correct (including the subtle `C AND C ≠ C` for
  soft conceptors).
- All five CL metrics are correct, with no off-by-one.
- The goldens genuinely bite: 11/15 source mutations were caught by the right test at
  `assert_allclose(1e-6)`; the goldens are compared, not regenerated at test time.
- The degree-preserving null is exact; `ErdosRenyiReservoir` density and ρ track requests.
- Example 07 (the null model) is honest and careful (an 8-draw ensemble, z≈0.47, no topology spin), and
  it is the one connectome path that correctly uses fibre weights.
- The remediation did real work: the design-doc status table, the `[planned]` tags, the FORCE-citation
  correction, the dead-code removal, and the honesty-pass docstrings were all independently confirmed true.

---

## Meta-evaluation of the prior review (`REVIEW.md`)
The prior 13-agent review was thorough in its verdict-writing and unusually self-aware. Its central
thesis (the spec oversells the code; the conceptor-benchmark split is the structural defect) is correct,
and most of its adversarial rejections are defensible. But checked against the code, it graded its own
remediation too generously:
- It declared the spectral-radius bug unreachable and deferred it. F2 shows it is reachable via a
  public param with a 6e7× mis-scale. The "500 asymmetric matrices, 0 failures" stress test only sampled
  real-Perron-dominant matrices, which by construction cannot trigger the bug; it does not establish
  unreachability.
- Its deferral of the conceptor-benchmark wiring rests on a false claim ("BWT≈0 trivially"). F3
  disproves it with running code.
- It claimed comment fixes that were not applied (F6).
- Its proposed scale-bias fix (`xᵀx` normalization) is insufficient (F7).
- It entirely missed the binary-weights-on-real-data issue (F1), which is more damaging to the central
  premise than several issues it did catch.
Hence 63/100, slightly below its self-assigned 68. The gap is the three rationalized or missed MAJORs,
offset upward by a genuinely correct and well-tested core.

---

## Process note (transparency)
During this audit the five agents ran in parallel against one shared working tree. The test-suite agent
was (by design) temporarily mutating source files to test whether the suite catches regressions, while
the slither agent independently ran `pytest`. The slither agent consequently observed "58 passed, 3
failed" with conceptor eigenvalues at -2.9, a false positive: it had caught the test-suite agent's
sign-flip mutation mid-flight. Verified after the fact: the working tree was clean, 61/61 pass, and
conceptor eigenvalues sit in [0,1] at default aperture. *Lesson for future multi-agent audits: never run
a tree-mutating agent concurrently with read-only agents that execute the code; isolate it in a worktree.*

---

## Recommended remediation order
1. F1: fix the connectome `edge_attr` default or warn; align METHODOLOGY and examples. (Highest
   priority: it undercuts the central premise on the shipped data.)
2. F3: add the conceptor `partial_fit` adapter, or drop the "headline method" framing. Report its
   real (often catastrophic) BWT honestly.
3. F2: fix `_spectral_radius` for non-symmetric input (`‖Ax‖` or `eigvals` fallback).
4. F4: report the no-reservoir leaked-feature baseline beside every mock number; document the boost leak.
5. F5, F6, F11, F12: docstring and comment honesty, plus the missing connectome `fit/predict` test.
6. F7, F8, F9, F10: robustness, reproducibility, and efficiency polish as time allows.

---

## Remediation applied, 2026-06-25 (all findings fixed)

All twelve findings were remediated in the same session, golden-guarded (the symmetric spectral-radius
path and all characterization goldens are byte-for-byte unchanged). Suite 61 → 67 tests, all green;
all 7 example notebooks still execute under nbmake.

| # | Fix | Evidence |
|---|---|---|
| F1 | `connectome.py` now emits a `UserWarning` when the requested `edge_attr` is absent but real numeric edge attributes exist (instead of silently going binary); a `_numeric_edge_attrs` helper was added. METHODOLOGY.md corrected (the default is not fibre-weighted; pass `edge_attr='number_of_fibers'`). | new test `test_missing_edge_attr_with_real_weights_warns` (asserts the warning on scale83) |
| F2 | `_spectral_radius` routes non-symmetric input to exact `eigvals`; the symmetric (golden-pinned) power-iteration path is unchanged. | rotation ρ=2.0 now returns 2.0 (was 0.347); `symmetric=False` scales W to 0.9 (was ~5e7); 61 goldens still pass |
| F3 | Added `ConceptorReadout` (`partial_fit`/`predict`) so the conceptor method runs through `ContinualBenchmark`; exported it; updated the design-doc status table. | 2 new tests: disjoint → BWT≈0; overlapping class-IL → BWT=-0.49 (refutes the old "trivially ≈0" deferral) |
| F4 | Added `slither.leaked_feature_baseline`; mock.py and METHODOLOGY now state the leak honestly (no-reservoir baseline ~0.6, comparable to or beating the reservoir; boost ~100% leaked; corrected the cherry-picked "93%"). | new test `test_mock_is_leaked_no_reservoir_baseline_is_high` |
| F5 | The `spectral_radius` docstring now states the real default (1.25, super-critical). |  |
| F6 | The three stale `[N+1]`/`[N+1,K]` comments in `fit`/`predict` corrected to `[T-w, N]`/`[N, K]`. | grep clean |
| F7 | `conceptor_from_states` rewritten via symmetric eigendecomposition (`conceptor_from_correlation`): eigenvalues guaranteed ⊂ [0,1], no `inv` crash at extreme aperture. | verified: aperture=1e9 on rank-deficient input no longer raises; eig ∈ [0,1] |
| F8 | Session-grouped split warns when the effective held-out fraction deviates materially from `test_ratio`. | warning observed in `test_guards` 2-session case |
| F9 | `Reservoir`/`Reservoir3` accept an optional `seed=` (default global RNG → goldens preserved). | goldens still pass |
| F10 | "~B× faster" downgraded to an honest "≈1.5 to 2×, batch-size dependent, can be slower" in `_batch.py`, `connectome.py`, `random.py`. |  |
| F11 | Added a connectome `fit→predict` round-trip test plus a predict-before-fit `RuntimeError` test (was 0 coverage). | 2 new tests |
| F12 | False `P⁺=Pᵀ` comment corrected; bare `pytest.raises(Exception)` narrowed to `ValueError, match='classes'`; README de-staled (67 tests, build complete). |  |

**Process false-positive resolved:** the "58/3 failed" was a concurrent-mutation artifact (see the
Process note above); verified clean afterward (67/67 pass).

**Not done (by design, with rationale):** (a) decoupling the mock labels from inputs (this would
regenerate all committed mock data and example outputs for a fixture explicitly labelled "not evidence",
so transparency via the baseline helper is the proportionate fix); (b) the `ConceptorClassifier`
(sequence API) was left as-is and the benchmark gap was closed with a separate `ConceptorReadout`
rather than overloading the existing class's API. With these fixes the score moves from 63 toward the
mid-70s; the residual gap to "paper-grade" is now real-data validation, not code honesty.
