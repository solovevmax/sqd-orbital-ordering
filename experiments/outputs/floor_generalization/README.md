# F1 -- opposite-spin mask as a net liability

**Script:** `experiments/floor_generalization.py`

**Question.** E1 found "anchors can be worse than nothing" at H10
identity. Does that generalise across ordinary random same-spin orderings,
and is there a cheap decision rule to guard against it?

**Protocol.** Extend the floor-vs-default comparison to all 50 baseline
random orderings: fraction harmed, correlation of the floor-vs-default gap
with overall ordering quality, and a cheap one-extra-evaluation guard
(sample the no-alpha-beta control alongside the default anchor; keep
whichever is better).

**Headline.** The opposite-spin locality mask is a net liability (default
anchor worse than no coupling at all) at 4.0% of the 50 random orderings
-- a real, non-trivial minority effect, and the benefit of opposite-spin
coupling correlates with overall ordering quality (helps good orderings
more). The one-extra-evaluation guard is exactly what
`experiments/chain_aware_phaseB.py`'s "top1_with_guard" configuration
tests out of sample later. See `f1_report.txt` and
`f1c_floor_vs_default_50random.csv`.
