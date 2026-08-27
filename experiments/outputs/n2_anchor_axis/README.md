# N2 anchor-axis transfer

**Script:** `experiments/n2_anchor_axis.py`

**Question.** The anchor-selection result so far rests entirely on H10.
Does it transfer to N2 -- a different molecule, different basis, and a
system near its capture ceiling (0.9866 vs. H10's 0.7554) -- or is it an
H10-specific artefact?

**Protocol.** Cached canonical N2 CAS(6,10) R=1.55 reference, no new
reference data, 1e6 shots seed 2026. All 120 anchor triples at identity;
40 shared triples (rng seed 20260826001) at reverse and at r039 (a
baseline-median-nearest random ordering). Plus (mid-task addition) err_lucj
-- the masked-LUCJ variational energy, no sampling -- for all 120 identity
triples.

**Headline.** Transfers on every axis tested: real anchor effect (24.27-
114.00 mHa range at identity), the capture mechanism holds
(rho(captured,err) = -0.877), the ansatz-level rule replicates and is even
*stronger* than on H10 (rho(err_lucj, retained_J_oppspin) = -0.965 vs.
H10's -0.850), and the smaller raw N2 effect size is fully explained as a
ceiling-proximity artefact once normalised by headroom (N2: 2550 mHa per
unit headroom vs. H10's 1445 -- actually higher, not lower). See
`report.txt`, `identity_120.csv`, `reverse_median_40.csv`,
`metadata.json`.
