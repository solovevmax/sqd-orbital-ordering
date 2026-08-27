# E1 -- degenerate floor, and E2 -- budget-transfer screening

**Script:** `experiments/floor_investigation.py` (E1),
`experiments/budget_transfer.py` (E2) -- same output directory

**Question (E1).** Why do 17 of H10-identity's 120 anchor triples share an
identical err_mHa = 458.70 "floor"?

**Question (E2).** Can a cheap 8x8-dimension screen (instead of the full
15x15 budget) cheaply identify a good anchor triple, beating prediction by
retained_J_oppspin on cost or accuracy?

**Protocol (E1).** Direct control: `interaction_pairs_ab = []` (no
opposite-spin coupling at all) at identity/physical/rand007, confirming
the floor is exactly this no-alpha-beta-correlation state, chain-specific
(not a universal constant), and checking floor triples' J_ab weight.

**Protocol (E2).** Sample once per triple, derive results at multiple
budgets {5, 8, 10, 15} from the same sampled distribution; compare
cost/accuracy against `retained_J_oppspin`-based selection.

**Headline.** E1: the floor is the no-alpha-beta state, confirmed by
direct control; "anchors can be worse than nothing" for a real minority of
triples (14.2% at H10 identity). E2: the premise fails on both cost (sbd's
fixed per-call overhead dominates at these small dimensions -- wall time
is flat across budgets) and accuracy (mixed results, no clean win) --
cheap screening does not clearly beat prediction. See `e1_report.txt`,
`e1_metadata.json`, `e2_report.txt`.
