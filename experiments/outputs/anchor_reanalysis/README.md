# D1-D6 -- anchor re-analysis

**Script:** `experiments/anchor_reanalysis.py`

**Question.** A set of follow-up analysis questions on the Part A/B data:
is the capture-vs-error mechanism itself ordering-dependent or just the
cheap proxy; what does proper regret accounting show; does
amplitude-weighted anchor selection do any better; what structural
features separate good triples from bad ones; where exactly is the
bottleneck?

**Protocol.** Pure re-analysis of Part A/B/C1/C2's cached CSVs -- no new
sampling. Extended `score1()` with an `anchor_orbitals` parameter to
support this (also used by later experiments' `retained_J_oppspin`-based
selection).

**Headline.** Confirms the capture mechanism itself is robust
(rho(captured,err) consistently strong); the bottleneck is in translating
a cheap proxy into a reliable selection rule, not in the underlying
physics. This finding is what the `transmission.py` experiment later
localises precisely (link 2 always holds; link 1, ansatz-quality ->
sampling-concentration, is where the proxy's reliability breaks down).
See `report.txt`.
