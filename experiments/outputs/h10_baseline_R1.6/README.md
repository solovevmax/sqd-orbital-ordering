# H10 baseline

**Script:** `experiments/h10_baseline.py`

**Question.** What is the spread of SQD subspace error across same-spin
orbital orderings on H10 (STO-6G, R=1.6, CAS(10,10)), at the historical
default (position-based, `p % 4 == 0`) anchor convention?

**Protocol.** 7 named orderings (identity, physical, physical_reverse,
s1_max, s2_max, retainedJ_max, reverse) + 50 random orderings (rng seed
20260825), 2 seeds each, 2e6 shots, 15x15 determinant budget. Mechanism B
(`run_ordering_pipeline.py`), corrected same-spin-diagonal mask.

**Headline.** Establishes the same-spin-ordering-only baseline that every
later anchor-selection experiment (Part B onward) compares against. Mean
achieved capture across the 50 random orderings: 0.5934. See
`h10_baseline_results.csv` and `metadata.json` for full per-ordering
figures.
