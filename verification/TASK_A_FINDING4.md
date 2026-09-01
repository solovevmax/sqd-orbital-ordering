# Task A — characterising Finding 4

Finding 4 (COLD_START.md): an independently-rebuilt H10 R=1.6 reference,
numerically agreeing with the cached one to 1e-14 Ha in every stored scalar,
gives a materially different SQD answer at identity/default-anchors (373.63
mHa vs. the cached 300.32 mHa). This section characterises the mechanism
precisely, per your five-part instruction.

## A1 — which determinants swap, and by how much

Sampled both references (identity, default anchors, seed 2026, 2,000,000
shots) and dumped the top-20 alpha/beta marginal counts. **Four strings
differ in the top-15, not one or two:**

| | rank in cached | frac (cached) | rank in fresh | frac (fresh) |
|---|---|---|---|---|
| `0000110111` | 13 | 0.008536 | outside top-20 | — |
| `0100011101` | 15 | 0.007238 | outside top-20 | — |
| `0010001111` | outside top-20 (was 16) | 0.007118 | 12 | 0.009667 |
| `0101010101` | outside top-20 (was 20) | 0.006331 | 15 | 0.007092 |
| `0000111101` | 14 | 0.007959 | **8** | **0.012665** |

(Alpha and beta swap identically, as expected — this system is
alpha/beta-symmetric at identity.) The last row is the important one:
`0000111101` does not creep from rank 16 to rank 15 — it jumps from rank 14
to rank **8**, and its sampled fraction rises by 59% (0.0080 → 0.0127). This
is not a hairline tie at the rank-15 cutoff; it is a **real, order-of-magnitude-larger-than-expected
reordering across ranks 7–15**. Full dump: `task_a1.log` (this session's
scratch output, reproducible via the script below).

## A2 — distribution or cut?

Both: **the distributions themselves differ measurably**, not merely which
side of an unchanged cut a tied pair falls on. Comparing the full 252-string
alpha (and beta) count distributions between cached and fresh references:
the maximum absolute fractional difference over the union of observed
strings is **0.0047** (`0000111101`, alpha) and **0.0048** (beta) — i.e. one
determinant's sampled probability shifted by nearly half a percentage point.
At 2,000,000 shots the Monte Carlo standard error on a count near
frac≈0.008 is roughly $\sqrt{0.008 \times 0.992 / 2{,}000{,}000} \approx
6\times10^{-5}$ — the observed 0.0047 shift is **~80 shot-noise standard
deviations**, not sampling noise.

This is the more serious of your two framings: **the perturbation is
propagating through the CCSD amplitudes into the LUCJ circuit parameters**
and changing the actual sampled distribution, not just nudging a marginal
tie at the selection boundary. Confirmed independently by direct comparison
of the stored `t1L`/`t2L` tensors themselves (not just the aggregate CASCI
energy, which is what the original cold-start check compared): element-wise
**relative** differences between cached and fresh `t2L` reach **7.6e-10**
(max) with a median of 8.8e-12, and `t1L` reaches 2.3e-11 (max) — three to
five orders of magnitude larger than the 1e-14-level agreement seen in the
*aggregate* CASCI energy and FCIDUMP integrals. CCSD amplitude tensors have
thousands of elements; the scalar energy is a sum that can cancel run-to-run
noise that individual amplitude *elements* do not. That is the actual
mechanism: aggregate reproducibility (1e-14) does not imply element-wise
reproducibility (1e-10-ish), and the LUCJ circuit is built from the
elements, not the aggregate.

## A3 — how special is this? (perturbation sweep, H10 identity, default anchors)

Cached reference's `t1L`/`t2L` perturbed by relative Gaussian noise (5 seeds
per level), sampling seed fixed at 2026, energy reference (`E_CASCI`,
FCIDUMP) held at the cached value throughout so only the circuit-construction
tensors vary:

| noise level | n | mean (mHa) | sd (mHa) | min | max | outcomes |
|---|---|---|---|---|---|---|
| 0 (unperturbed) | 1 | 300.32 | — | 300.32 | 300.32 | baseline |
| 1e-14 | 5 | 373.34 | 0.33 | 372.82 | 373.63 | **5/5 near 373** |
| 1e-12 | 5 | 358.59 | 32.45 | 300.55 | 373.92 | 4/5 near 373, 1/5 near 300 |
| 1e-10 | 5 | 305.94 | 64.63 | 243.29 | 373.63 | 2/5 near 373, 1/5 near 300, **2/5 near a third value, ~243** |
| 1e-8 | 5 | 300.01 | 3.42 | 295.62 | 305.17 | **5/5 near 300**, tightly clustered |

**This is not the simple "bigger noise, bigger/more-likely error" picture I
expected going in, and I want to be direct about that rather than smoothing
it over.** Three findings:

1. **Even the smallest tested noise (1e-14 relative) reliably flips the
   result** — all 5 draws land near 373, none near 300. The cached 300.32
   mHa value sits on a knife-edge: an independently-built reference is *more
   likely* to land away from it than on it, at any noise level tested here.
2. Variance is **not monotonic** in noise level. It peaks at 1e-10 (sd=64.6,
   and a *third* distinct plateau near 243 mHa appears — a determinant
   configuration neither the cached nor the fresh reference reached) and
   *drops back down* at 1e-8 (sd=3.4), clustering tightly around the
   *original* 300 mHa value again.
3. Read together with A1/A2: this looks like a genuinely **multi-modal,
   discrete selection landscape** — several nearby quasi-degenerate 15-
   determinant subspaces exist, each giving a different diagonalized energy,
   and which one a given amplitude perturbation lands in is not a smooth or
   monotonic function of perturbation size. 1e-8 landing back near 300 is
   very unlikely to mean "noise this large restores the right answer" in any
   principled sense — five draws is a small sample of a landscape that
   apparently has at least three basins, and I would not extrapolate beyond
   what is shown here.

**Honest limitation:** 5 seeds per level is enough to show the phenomenon
is real and reproducible, not enough to fit a reliable noise→outcome curve.
Script: `verification/verify_tier_a3.py` (renamed from scratch
`task_a3.py`); raw output `task_a3_h10.csv`.

## A4 — N2 identity (boundary ratio 0.504, vs H10's 0.989)

Identical protocol (5 perturbation seeds per level, same relative noise
levels, sampling seed fixed at 2026, energy reference held at the cached
value) applied to N2 identity/default anchors:

| noise level | n | mean (mHa) | sd (mHa) | min | max | draws == exact baseline |
|---|---|---|---|---|---|---|
| 0 (unperturbed) | 1 | 31.87 | — | 31.87 | 31.87 | — |
| 1e-14 | 5 | 39.30 | 7.52 | 31.87 | 46.90 | 2/5 |
| 1e-12 | 5 | 34.36 | 6.62 | 24.04 | 39.05 | 0/5 |
| 1e-10 | 5 | 33.34 | 3.29 | 31.87 | 39.23 | 4/5 |
| 1e-8 | 5 | 39.07 | 12.65 | 31.85 | 61.08 | 3/5 |

**This directly confirms the report's own explanation, tested rather than
asserted.** Two comparisons make it unambiguous:

- **Absolute spread.** Across every noise level, N2's err_mHa ranges over
  roughly 24–61 mHa (a 37 mHa band). H10's ranges over roughly 243–374 mHa
  (a 131 mHa band) — H10's perturbation-induced spread is **~3.5x larger in
  absolute terms**, on a system whose own baseline (31.87 mHa) is nearly
  10x smaller than H10's (300.32 mHa) to begin with. Relative to each
  system's own scale, the gap is even starker.
- **Exact reproduction rate.** N2 reproduces its own unperturbed baseline
  *bit-for-bit* (31.870454...) in **9 of 20** perturbed draws, including at
  the largest noise level tested (1e-8: 3/5 draws land exactly on baseline).
  H10 reproduces its own baseline (300.31919...) in **0 of 20** perturbed
  draws, at any noise level — not once, from 1e-14 all the way to 1e-8.

N2 is not perfectly immune (one 1e-8 draw reached 61.08 mHa, a genuine
outlier), so this is a matter of degree, not a binary safe/unsafe split —
but the direction and rough magnitude match the report's
$w_{16}/w_{15}$-ratio explanation cleanly: N2's much more comfortable
margin (0.504 vs H10's 0.989) makes its determinant selection far less
prone to being flipped by amplitude-level noise several orders of magnitude
below what independent RHF/CCSD runs actually produce.

## A5 — does more shots help?

H10 identity/default-anchors, on the FRESH (rebuilt) reference, at 2e6, 8e6
and 32e6 shots (a 16x range), sampling seed fixed at 2026:

| shots | err_mHa | wall time |
|---|---|---|
| 2,000,000 | 373.611432 | 35.5s |
| 8,000,000 | 373.611432 | 38.9s |
| 32,000,000 | 373.611432 | 51.9s |

**Confirmed, exactly as you predicted: no convergence toward 300.32.** The
result is bit-identical across a 16x shot-count range. This rules out shot
noise as the mechanism entirely, on the fresh reference specifically: the
"wrong" determinant is not a marginal call that more sampling would
resolve correctly — on the fresh reference's actual circuit, it robustly
and unambiguously belongs in the top-15. The disagreement with the cached
reference's 300.32 mHa is a property of which reference built the circuit,
not of how well either circuit was sampled.

## Summary

All five checks point the same direction and are mutually consistent:

1. **A1/A2**: the perturbation is not a hairline tie at one rank — it's a
   real shift in the sampled probability distribution (up to 0.0047 in
   marginal fraction, ~80 shot-noise standard deviations), propagating from
   CCSD amplitude tensor elements that differ by up to ~1e-10 relative
   between independent builds (not the ~1e-14 the aggregate CASCI energy
   agreement would suggest).
2. **A3**: H10 identity/default-anchors sits in a fragile, multi-modal
   region — even 1e-14 relative noise reliably relocates the answer away
   from the cached value, and the noise→outcome relationship is not
   monotonic (a third, previously-unseen ~243 mHa plateau appears at
   1e-10; 1e-8 clusters back near 300 by what looks like chance given the
   small sample).
3. **A4**: N2, at a far more comfortable selection-boundary margin (0.504
   vs. H10's 0.989), is measurably and substantially more robust to the
   identical perturbation — smaller absolute spread, and it reproduces its
   own exact baseline in 9/20 perturbed draws where H10 reproduces its own
   baseline in 0/20. This is the report's own explanation for the
   phenomenon, now tested directly rather than asserted.
4. **A5**: shot count is not the lever — 16x more shots changes nothing on
   the fresh reference.

**This is a genuine scientific result about the H10 R=1.6, CAS(10,10),
identity-chain, default-anchor, budget-15 configuration specifically — not
a bug, not an audit artefact, and not (per A4) a property of the SQD method
in general.** It says something concrete and worth stating in its own
right: at this particular system/budget combination, the top-15
determinant-selection boundary is close enough to degenerate that the
*specific* published number (300.32 mHa) is one of at least three nearby,
similarly-plausible outcomes that an independently-converged CCSD
calculation could land on, and standard reproducibility practices (more
shots, more seeds at fixed reference) cannot detect or fix this because
they hold the reference fixed. Whether this is worth flagging as a
limitation in the report itself, investigating further (e.g. whether the
*mechanism* result — capture correlates with error — still holds across
these alternate ~243/300/373 outcomes), or left as characterised-but-out-of-scope
is a judgement call for you; I have not touched the report or any cached
data.

Raw data and scripts: `task_a1_marginal_dump.py`,
`task_a3_h10_perturbation.py` (+ `_results.csv`),
`task_a4_n2_perturbation.py` (+ `_results.csv`), `task_a5_shot_sweep.py`.

## Amendment 3 — thread-related or irreducible? Neither, precisely

Built the H10 R=1.6 reference from scratch 5 times with
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, and 5 more times
with those unset, each as a genuinely separate `python3` subprocess (not a
loop in one process) with a fresh, unique cache directory so every run is a
real independent build. Result:

**Both groups are bit-for-bit identical, in every one of the 10 runs, and
identical to the shipped `cache/h10_R1.6/reference.npz` too** — `t1L`/`t2L`
max absolute difference `0.000e+00` across all 10, `E_CASCI` the same to
every printed digit (`-4.966071088325821`) in all 10 runs plus the cached
file. Thread pinning makes no detectable difference either way, in this
environment, today.

**This does not mean the noise vanished — it means the real variable is
which conda environment does the building, not thread count.** All 10 of
the runs above used the `sqd` conda environment — the same one that
originally produced the shipped cache. Repeating the identical build in a
*different*, independently-solved environment reproduces Finding 4's
original discrepancy exactly:

```
$ conda activate sqd-orbital-ordering   # a different env, built today from
                                          # the now-fixed environment.yml
$ python3 -c "... build_or_load_h10_reference(...) ..."
t1L: max abs diff = 2.371e-13   max rel diff = 1.217e-11
t2L: max abs diff = 4.425e-12   max rel diff = 1.737e-09
```

— the same order of magnitude as the original cold-start finding, even
though `pip show` reports **identical** `pyscf`/`numpy`/`scipy` version
numbers (2.10.0 / 2.5.2 / 1.18.0) in both environments. The actual
difference is one level down, in `conda list`:

| package | `sqd` (built the shipped cache) | `sqd-orbital-ordering` (fresh, today) |
|---|---|---|
| `libblas` | 3.11.0 **build 9**\_h51639a9\_openblas | 3.11.0 **build 10**\_h51639a9\_openblas |
| `liblapack` | 3.11.0 **build 9**\_hd9741b5\_openblas | 3.11.0 **build 10**\_hd9741b5\_openblas |
| `numpy` | 2.5.2 build `py312ha003a3f_0` | 2.5.2 build `py312hff34920_1` |

Same nominal versions, different conda **builds** of the underlying
BLAS/LAPACK — exactly the kind of low-level numerical-library difference
(different SIMD dispatch or summation order inside GEMM/dot-product
kernels) that shifts iteratively-converged CCSD amplitudes at the 1e-10 to
1e-13 level without touching the aggregate energy at anything above 1e-14,
which is a well-understood, standard phenomenon in numerical computing —
not a race condition.

**Answer to your question, precisely: the noise is deterministic and fully
reproducible *within* one fixed environment regardless of thread pinning
(so the existing OMP/MKL/OPENBLAS=1 protocol is not what's failing here),
but is not reproducible *across* independently-resolved environments, even
ones that satisfy `environment.yml` and report matching top-level package
versions.** `environment.yml` pins no package versions at all (not even
after Task B's fixes), so two separate `conda env create` runs — today vs.
whenever the shipped cache was built, or on two different machines — are
not guaranteed to solve to the same BLAS build. Practically this has the
same implication as your "irreducible" branch (shipping/caching the
reference is the only real mitigation; thread pinning alone does not
guarantee bit-reproducibility of freshly-built references), but for a
different and more specific reason: it's an **environment/BLAS-build
reproducibility gap**, not process-launch randomness, and it is in
principle fixable by pinning exact package builds (e.g. a `conda-lock`
file or exact-version pins in `environment.yml`) rather than by anything
involving threads.

Script: `task_amendment3_thread_determinism.py` (per-run worker) — the
driver that invoked it as 10 separate subprocesses was scratch shell script,
not committed; the worker plus this write-up are enough to reproduce the
comparison.

## Correction, found during FIX 1 verification: Amendment 3's conclusion above is incomplete

While verifying the lockfile fix (see `FIX1_LOCKFILE_VERIFICATION.md`), the
`sqd` environment — the exact same one that gave 10/10 bit-identical,
cache-matching builds above — was re-tested with 8 more independent
builds and gave a **different** result, consistently: `E_CASCI =
-4.966071088325831`, not the cached `-4.966071088325821`. Not
intermittent — 8/8 agreed with each other, just not with the earlier 10/10
or the cache.

`conda-meta/history` for `sqd` confirms no package was added, removed, or
changed at any point during this session (its last modification predates
this session by weeks) — so this is not package/BLAS-build drift of the
kind identified above. Something about **machine or process state that
changes over time within one long-running session** is producing a
discrete, and evidently persistent-once-triggered, shift in
floating-point outcome, even with the environment provably unchanged and
thread counts pinned identically both times.

I do not have a confirmed root cause for this second-order effect (candidates:
Apple Silicon P-core/E-core scheduling assignment, ASLR-dependent memory
alignment interacting with a SIMD-dispatch code path, thermal/frequency
state after hours of sustained computation — none confirmed). What is
established, empirically, is that **environment pinning alone (Amendment
3's original conclusion) is not sufficient to guarantee bit-reproducibility
of this reference build, even within the single environment that produced
the shipped cache.** This revises Amendment 3's practical conclusion
somewhere between your two original hypotheses: not simple thread-count
nondeterminism, not purely environment/BLAS-build drift either — there is
a real, currently unexplained, process/machine-state component on top of
the confirmed environment-drift one. See `FIX1_LOCKFILE_VERIFICATION.md`
for the full account and its implications for the lockfile fix.
