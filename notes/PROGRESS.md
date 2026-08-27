# Progress

## 2026-08-20

**Result.** 200 random orderings, N2 CAS(6,10) @ 1.55 A, fixed 225-dim budget,
1e6 shots. Subspace error 23.16 to 155.49 mHa — factor 6.7 at identical cost.
Superseded the earlier 26-ordering figure of 4.6x.

**Robustness.** Deterministic re-run with transpiler seed pinned gives
between/within variance ratio 27.2. Predictor ranking proved unstable at n=26:
retained_J correlation moved -0.560 -> -0.378 under the seed fix alone, and its
selection regret became worse than random.

**Negative result.** Fast evaluator (exact marginals from |psi|^2, skipping
Aer and sbd) does not reproduce sbd's ranking (Spearman -0.08). Internals
verified correct: HF-only returns E_HF exactly, full space returns CASCI
exactly, bitstring->index map verified. Given identical determinant files, sbd
returns energies 12-50 mHa lower than direct diagonalisation, so sbd appears to
expand the subspace internally. Ask Shirakawa.

**Meeting.** Shirakawa confirmed orbital ordering is an open problem for the
group and nobody is working on it. Offered Fugaku access with larger basis sets
if a good ordering approach emerges.

**Next.** Compute target marginals from the CASCI vector; test whether
divergence between sampler and target marginals predicts subspace error across
the 200-ordering dataset. Then construct chemically motivated orderings and
measure their percentile against the random baseline.

## Voided results (25 Aug)

**All prior H10 numbers are withdrawn, pending re-run.** `run_ordering_pipeline.py`'s
mask construction (`same_spin_pairs`/`interaction_pairs_for`) omitted the
same-spin diagonal `(p, p)` Jastrow entries that `unified_run.py`'s N2 mask
always retained. This is not a numerically-small effect: it is a physically
wrong mask (a same-spin diagonal term `J_pp n_p n_p` reduces by fermionic
idempotency to `J_pp n_p`, a single-qubit Z rotation; heavy-hex connectivity
constrains two-qubit gates only, so there is no hardware justification for
masking it out). Withdrawn: `s1_max` (lines 175-218), `identity` (373-374),
`retainedJ_max` (452-458), `physical` (390-393), and the anchor-phase
`253 -> 363` sweep. All of these used the H10 path (mechanism B) and are
suspect.

**N2 results are unaffected.** `unified_run.py`'s `_m_aa` always included
`np.fill_diagonal(_m_aa, True)`, independent of ordering - the N2 mechanism
was correct throughout.

**Why the earlier Spearman +0.993 mask-model validation (`stage1()`'s
convention guard) did not catch this:** the diagonal is retained
identically for every ordering (it is permutation-invariant), so it
contributes an *additive constant* to `retained_J` across the whole dataset.
A rank correlation (Spearman) is invariant to additive constants and is
blind to this by construction - the guard could be satisfied at
spearman > 0.99 while the underlying quantity was still missing 10 of its
31 terms per repetition (1302 vs 882 retained |J| values on N2, confirmed
via `experiments/preflight.py crosscheck`).

**Fix and validation:** mask logic extracted to the single shared
`src/sqd_ordering/mask.py` (commit `b84ff9f`), both pipelines rewired to
import from it. `experiments/preflight.py crosscheck`'s operator-level test
(no sampling, no sbd) now confirms mechanism A and mechanism B build
entrywise-identical operators (`diff_aa = diff_ab = diff_U = 0.0` exactly,
across `diag_coulomb_mats` and `orbital_rotations`) for identity, reverse,
and three random permutations on the cached N2 reference; `pytest
tests/test_mask_equivalence.py` extends this to 20 random permutations. H10
must be re-run against the corrected mask before any of the withdrawn
numbers can be trusted again.

**Methods note: `interaction_pairs` performs post-hoc zeroing, not a
constrained fit.** The operator-level identity `P^T[M .* (P J P^T)]P ==
(P^T M P) .* J` matches mechanism A (un-permuted) and mechanism B exactly,
entrywise, to < 1e-12 - including `n_reps` staying identical (42/42) between
the two constructions. If `ffsim.UCJOpSpinBalanced`'s `interaction_pairs`
argument re-fit the operator to the restricted pair set (a constrained
optimization), the two constructions would generically produce different
`diag_coulomb_mats` values at the *surviving* entries, since a constrained
fit and a free fit followed by zeroing are different operations whenever
the mask is non-trivial. Exact entrywise equality of the surviving entries
is only possible if `interaction_pairs` builds the unrestricted operator
first and zeroes everything outside the requested pairs - i.e. post-hoc
masking, algebraically equivalent to constructing the full operator and
then multiplying by the fixed 0/1 mask `M`. This settles the question
raised when `run_ordering_pipeline.py`'s H10 path and `unified_run.py`'s N2
path were first found to disagree structurally: `interaction_pairs` is not
doing anything a subsequent elementwise mask couldn't equally do.

**Diagnostic pitfall: comparing mechanism A's samples against mechanism
B's requires no relabelling at all - remapping breaks it.** An early
version of `experiments/preflight.py`'s `invariance` and `crosscheck`
subcommands remapped mechanism A's raw sampled determinant strings via
`perm` before comparing them (on the theory that `permute_operator`'s `(P J
P^T, U P^T)` reparametrisation puts qubit `k` in a "permuted frame"
representing original orbital `perm[k]`). This is wrong for both the
unmasked and the masked case, and produced two symptoms that looked like
real bugs but were diagnostic artefacts: (1) `invariance` reported Jaccard
0.03-0.07 between orderings whose energies were bit-identical to 12
decimals; (2) `crosscheck`'s sampling confirmation reported Jaccard 0.0
between `identity` and `reverse`, again despite bit-identical energies. In
both cases, `unified_run.py`'s established convention (`captured_of` /
`retained_J_of`, used for every published N2 result) already treats raw
qubit index `k` as canonical orbital `k` directly - `permute_operator` only
reshuffles which `(J, U)` values are plugged into the fixed masked
positions; it never relabels what a sampled bit means, because
`PrepareHartreeFockJW`'s reference state is never permuted to match. Fixed
by comparing raw determinant sets directly, with no remap step, in both
subcommands (removed `remap_determinant_to_canonical` entirely). After the
fix: `invariance` gives Jaccard = 1.0 for all four orderings (asserted);
`crosscheck`'s sampling confirmation gives Jaccard = 1.0 for 4/5 orderings
and, for the one exception (`rand_seed203`, -2.78 mHa), a clean single
determinant flipping at the budget-15 boundary (beta sets match 15/15;
alpha sets match 14/16) - consistent with ordinary finite-shot noise near a
near-degenerate cutoff, not a real disagreement between mechanisms.