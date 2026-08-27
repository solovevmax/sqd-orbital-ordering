# Part B -- anchor decomposition, and C1/C2 hardening

**Scripts:** `experiments/anchor_decomposition.py` (Part B),
`experiments/anchor_hardening.py` (C1/C2, same output directory)

**Question.** The mask exposes two separable levers: the same-spin chain
ordering, and which orbitals anchor the opposite-spin (on-site) terms
(originally position-based, `p % 4 == 0`). Does free anchor *selection*
matter more than same-spin *ordering* -- and is that finding robust to
selection bias?

**Protocol.**
- B0: symbolic proof that the same-spin mask is reversal-invariant (57/57).
- B1: anchor-offset sweep at 4 fixed same-spin orderings (no free
  selection yet).
- B2: free anchor selection at identity -- all 120 orbital triples ranked
  by retained_J_oppspin (no SQD-quality proxy exists for mechanism B), 30
  sampled (top/bottom 10 + 10 random).
- C1: all 120 triples sampled at identity (removes B2's top/bottom
  selection bias).
- C2: 40 shared triples sampled at physical / physical_reverse / rand007
  (transferability across chains).

**Headline.** Anchor selection dominates same-spin ordering: 120-triple
range at identity = 234.10 mHa vs. the ~163-171 mHa range from offset/
same-spin sweeps alone. rho(captured, err) = -0.840; rho(retained_J_oppspin,
err) = -0.764 at identity but degrades badly at other chains (transfer is
"real but mixed, not clean" -- this is the finding that the whole
`transmission.py` / `chain_aware.py` line of work later explains
mechanistically). See `c1_all120_identity.csv`, `c2_transfer.csv`,
`b1_offset_sweep.csv`, `b2_all120_ranking.csv`, and both metadata.json files.
