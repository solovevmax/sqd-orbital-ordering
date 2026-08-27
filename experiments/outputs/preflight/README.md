# Preflight -- mechanism A/B equivalence, and shot-count/seed variance

**Script:** `experiments/preflight.py` (subcommands: `crosscheck`,
`invariance`, `shotscan`)

**Question.** Do mechanism A (`unified_run.py`, N2's fixed-mask pipeline)
and mechanism B (`run_ordering_pipeline.py`, H10's `interaction_pairs`
pipeline) build entrywise-identical operators after both were rewired
onto the shared `src/sqd_ordering/mask.py`? And how much does H10's
subspace error vary with shot count and simulator seed alone?

**Protocol.** `crosscheck`: operator-level comparison (no sampling, no
sbd) of `diag_coulomb_mats`/`orbital_rotations` for identity, reverse, and
random permutations, plus a sampling-level confirmation. `invariance`:
Jaccard overlap of sampled determinant sets under reversal (compares raw
determinant sets directly -- an earlier version incorrectly remapped
determinants through the permutation first; see `notes/PROGRESS.md`,
"Diagnostic pitfall"). `shotscan`: H10 at shots in
{500k, 2M, 8M} x 5 seeds.

**Headline.** Mechanisms A and B are entrywise identical (diff = 0.0 exactly)
for every ordering tested; `invariance` gives Jaccard = 1.0 for all four
orderings once the remap bug was fixed. This is the validation that
license every other experiment to treat mechanism A and B results as
comparable. See `crosscheck_metadata.json`, `invariance_results.csv`,
`shotscan_results.csv`.
