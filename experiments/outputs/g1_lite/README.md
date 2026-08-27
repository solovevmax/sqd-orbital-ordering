# G1-lite -- do the two levers interact?

**Script:** `experiments/g1_lite.py`

**Question.** Same-spin ordering and anchor selection were studied mostly
in isolation. After anchor optimisation, does same-spin ordering still
matter, and does an ordering's baseline (default-anchor) rank predict its
post-optimisation rank?

**Protocol.** 8 same-spin orderings (identity, physical, plus the best-2/
median-2/worst-2 of the 50-ordering H10 baseline) x the same 40 anchor
triples (rng seed 20260825003, uniform over the 120), best-of-40 per
ordering compared against each ordering's own baseline and floor.

**Headline.** Yes, same-spin ordering still matters after anchor
optimisation (spread of best-of-40 across the 8 orderings = 59.84 mHa,
not zero), but baseline ranking does NOT reliably predict
post-optimisation ranking (rho = +0.405, p = 0.32, not significant) --
baseline-worst orderings can nearly match baseline-best after anchor
optimisation. This is the n=8 half of the compression-factor figure that
`experiments/chain_aware_phaseB_analysis.py` extends to n=20. See
`g1_report.txt`, `g1_summary.csv`, `g1_metadata.json`.
