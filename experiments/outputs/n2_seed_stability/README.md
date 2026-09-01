# N2 seed stability: is the ordering effect signal or sampling noise?

**Data:** `seed_replication.csv` (relocated from `archive/legacy_outputs/`
during the reproducibility audit — see below).

**Question.** The main N2 ordering sweep (`outputs/unified/results.csv`)
uses 2 seeds per layout. Is that enough to say the observed spread across
layouts is signal rather than seed-to-seed sampling noise?

**Protocol.** N2 CAS(6,10), R=1.55 Å, 6-31G, 26 same-spin orderings, 5
independent Aer sampling seeds each (7, 1234, 2026, 31415, 55555), fixed
reference data, budget 15/spin sector.

**Headline.**
- Between-orderings variance / within-ordering (seed-to-seed) variance =
  **27.2** — the spread across orderings is ~27x larger than seed noise.
- Ranking stability across seed pairs: mean pairwise Kendall **τ = 0.83** —
  a layout's relative rank is largely seed-independent.

Verified independently in `verification/verify_tier0.py`
(`abs_between_within_ratio_n2`, `abs_kendall_tau_n2`), recomputing both
statistics directly from this CSV.

## Provenance note

This dataset predates the `experiments/outputs/` one-directory-per-experiment
convention (it uses the pre-restructuring `n2_cas610_155` naming) and was
filed under `archive/legacy_outputs/` during this session's repository
restructure, on the (at-the-time correct) assumption that nothing current
depended on it. It was moved back out here when the reproducibility audit
found that the report's own "five-seed study" figures (variance ratio 27.2,
τ=0.83) have no other source in the repository — this file is the only
place that number can be independently checked. Two byte-identical copies
(`seed_replication_n2_cas610_155_full.csv`,
`seed_replication_n2_cas610_155_seedT2026.csv`) remain in
`archive/legacy_outputs/`; this is the canonical copy going forward.

No `experiments/*.py` script reproduces this run — it predates the current
experiment-script convention entirely. The data is retained as-is because a
published claim rests on it, not because the generating process is
documented or rerunnable.
