# Transpilation audit: circuit resource accounting across layouts

**Script:** `experiments/transpilation_audit.py`

**Question.** Every comparison in this project holds the LUCJ mask size,
shot budget, and SQD determinant budget fixed. Has this project verified
that different orbital layouts produce circuits of *equal cost* once
mapped onto real heavy-hex hardware connectivity? If not, part of any
measured improvement could be bought with hidden circuit resources.

**Protocol.** `CouplingMap.from_heavy_hex(distance=5)` (57 qubits, 128
edges), `GenericBackendV2`, `generate_preset_pass_manager(optimization_
level=1, backend=...)` with `pm.pre_init = ffsim.qiskit.PRE_INIT` (matches
the pipeline's `CFG["use_pre_init"]=True`), 5 SABRE seeds per layout
{11,22,33,44,55}, `initial_layout` not pinned (the pipeline doesn't pin
it either). Circuit construction imports `sqd_ordering.sampling.
build_circuit` directly -- refactored out of `sample_bitstrings()` for
this purpose (see commit "refactor: extract build_circuit()"), so this
audit transpiles the exact circuit the sampling pipeline runs, not a
lookalike. SWAP count captured by splitting the pass manager at the
routing/translation boundary (before SWAPs are decomposed into the basis
gate set) -- verified to reproduce the full-pipeline result exactly
before use.

Four layout sets: A1 (all 120 H10 anchor triples, identity chain), A2 (20
H10 same-spin chains at default anchors), A3 (10 named configurations
quoted elsewhere in this project's reports), A4 (sanity check: the
no-alpha-beta control must have strictly fewer 2-qubit gates than any
retaining configuration -- PASSED, 10568 vs a minimum of 11626 among the
120 A1 triples).

**Headline.**
1. Anchor axis NOT resource-neutral: CV = 5.1% (2Q gates), 11.6% (depth),
   27.4% (swaps) across the 120 triples.
2. Same-spin axis NOT resource-neutral either, and more so as expected:
   CV = 19.0% (2Q gates), 21.7% (depth), 37.5% (swaps) across 20 chains.
3. Resource cost correlates FAVOURABLY with performance on the anchor
   axis: rho(2Q gates, S0)=-0.709, rho(2Q gates, err_sqd)=+0.608 --
   better-scoring, lower-error anchor triples tend to need FEWER gates,
   not more. But one specific quoted comparison is not clean: H10
   physical's default->best improvement (389.71->172.15 mHa) comes with a
   real resource INCREASE (+336.6 2Q gates, +13.8 depth) -- small relative
   to the ~2x gap between identity and physical chains overall (12800 vs
   26500 2Q gates), but real. The other three quoted pairs (H10 identity,
   N2 identity, Cr2 identity) are clean: error falls, resources fall too.
4. Layout-to-layout variation (sd=672.5 2Q gates) exceeds SABRE seed
   noise (sd=484.8), ratio 1.39 -- real signal, not routing noise, though
   the margin is modest.

This audit does not overturn the project's conclusions -- the anchor
effect and the capture mechanism are unaffected by circuit resource cost,
and cost mostly moves in the SAME direction as accuracy (favourable) --
but it finds a genuine, previously-unverified confound at H10/physical
specifically, and confirms same-spin ordering (not anchor selection) is
the dominant driver of hardware resource variation. See `report.txt`,
`a1_by_triple.csv`, `a2_by_chain.csv`, `a3_table.csv`, `v3_table.csv`,
`all_rows.csv` (every (layout, seed) row).

**Follow-up (`experiments/transpilation_audit_followup.py`):**

*Q1 -- where does the anchor axis's 5.1% CV come from?* All 120 triples
retain the same PAIR COUNT (19 same-spin + 3 opposite-spin) by
construction, so gate count should match unless something else varies.
Measured the 2Q gate count of the logical circuit translated to the same
basis gate set but with NO coupling map (no layout, no routing, no
SWAPs): CV=0.27% (3 distinct values, range 10772-10868), versus 5.1% CV
post-routing on the same 120 triples. Logical-circuit sd is only 4.4% of
post-routing sd, so **routing/SWAP insertion (option a) accounts for the
large majority (~96%) of the variation** -- but not all of it: a small,
real basis-gate-decomposition effect survives even with no coupling map
at all (option b -- gate counts depend on the specific retained numerical
parameter values, not just the retained pair count). Circuit-construction
non-invariance (option c) is ruled out -- the same 3 distinct logical
values recur across many triples, consistent with decomposition
sensitivity to parameter values, not a construction bug.

*Q2 -- gates per mHa recovered, A3 default-vs-best pairs* (negative =
resources fell alongside the improvement, a net win on both axes;
positive = a real per-mHa cost):

| Pair | d_err (mHa) | d_2Q gates | gates per mHa |
|---|---|---|---|
| H10 identity | -75.72 | -66.6 | -0.88 (clean) |
| H10 physical | -217.56 | +336.6 | **+1.55** (costs 1.5 gates/mHa) |
| N2 identity | -7.60 | -880.8 | -115.89 (clean) |
| Cr2 identity | -38.84 | -2823.4 | -72.69 (clean) |

H10/physical is the only pair with a positive (cost-bearing) value, and
even there the cost is modest (1.55 2Q gates per mHa of accuracy
recovered, against a >26,000-gate circuit). Full detail in
`q1_logical_rows.csv`, `q1_comparison.csv`, `v3_table.csv` (now with
`gates_per_mHa`/`depth_per_mHa` columns), `report.txt`,
`metadata.json`.
