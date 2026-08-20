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