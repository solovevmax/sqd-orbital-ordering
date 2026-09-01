# Archive

Superseded or one-off work, kept for history. Nothing here is imported by
the live pipeline (`scripts/run_ordering_pipeline.py`,
`scripts/unified_run.py`, `src/sqd_ordering/`, `experiments/*.py`). Moved,
not deleted, per the project's convention — git history still holds every
prior version.

## Negative result

**`fast_evaluator.ipynb`** — an attempt to skip Aer sampling and sbd
entirely by computing exact determinant marginals from `|psi|^2` directly.
The result was negative and is documented in `notes/PROGRESS.md`
(2026-08-20 entry): the fast evaluator does **not** reproduce sbd's
ranking of orderings (Spearman rho = -0.08 against the real pipeline).
Internals were verified correct in isolation — HF-only input returns
`E_HF` exactly, the full determinant space returns `E_CASCI` exactly, and
the bitstring-to-index map was checked directly — so the discrepancy is
not an implementation bug in the evaluator. Given identical determinant
files, sbd consistently returns energies 12-50 mHa lower than direct
diagonalisation against the same files, indicating sbd expands the
subspace internally beyond what is passed to it. This was never resolved
(see notes/PROGRESS.md, "Ask Shirakawa") and the fast-evaluator approach was
abandoned rather than debugged further, in favour of the full Aer+sbd
pipeline used everywhere else in this project. Kept here as the record of
why that shortcut was ruled out, not merely as dead code.

## Superseded pipeline (pre-`run_ordering_pipeline.py`)

- `legacy_stage1.py`, `legacy_stage2.py`, `legacy_stage2b.py` — the
  original driver scripts (Hamiltonian/CCSD/LUCJ construction; Aer
  sampling; alpha/beta determinant export). Superseded by
  `run_ordering_pipeline.py` and `unified_run.py`. Renamed with a
  `legacy_` prefix on archiving: `run_ordering_pipeline.py` has its own,
  unrelated `stage0()`-`stage3()` functions (N2 canonical-orbital
  pipeline), and the original top-level `stage1.py`/`stage2.py` names
  collided with that unrelated naming — keeping the old names in
  `archive/` risked being read as the same thing.
- `SQD_workflow.ipynb` — the original exploratory driver notebook,
  superseded by `scripts/run_ordering_pipeline.py` and
  `scripts/unified_run.py`. Previously left at root pending a final
  interactive check (reading `outputs/scaleup_n2_cas610_155.csv`); that
  check is committed (see git history) and the notebook is archived here
  as a superseded exploratory driver, not maintained further.

## One-off scratch / exploratory scripts (all dead — not imported anywhere)

- `anchor_free_diagnostic.py` — centroid-based anchor-free (translation-
  invariant) opposite-spin mask diagnostic, H10. Precursor to the
  systematic Part B anchor decomposition (`experiments/anchor_decomposition.py`).
- `anchor_phase_diagnostic.py` — physical/reversed anchor-phase sweep on
  H10. Same lineage as above.
- `centered_mask_quickcheck.py` — quick check of a centroid-centered mask
  variant.
- `end_weighted_quickcheck.py` — quick check of an end-weighted anchor
  selection variant.
- `largest_J_quickcheck.py` — quick check of a largest-\|J_ab\| anchor
  selection variant.
- `cheap_target.py` — "can a CISD target replace the exact CASCI target
  for capture?" Superseded by the capture metric used throughout
  `experiments/`.
- `constructed_orderings.py` — early constructed-vs-random-baseline
  comparison. Superseded by `unified_run.py`.
- `diagnose_repro.py` — debugging script chasing an identity-ordering
  reproducibility discrepancy between scripts; resolved (see
  `notes/PROGRESS.md`, "Voided results").
- `direct_optimise.py` — direct permutation hill-climbing over
  retained-J / captured objectives. Superseded by `hill_climb()` inside
  `run_ordering_pipeline.py`.
- `marginal_matching.py` — early marginal-vs-subspace-error correlation
  check. Superseded by `experiments/score_audit.py`.
- `scale_up.py` — the original 200-random-ordering N2 sweep. Explicitly
  superseded by `unified_run.py` ("single consolidated run... one code
  path").
- `self_consistent_target.py` — "can a self-sampled target replace the
  exact wavefunction target?" exploratory question, dead end.
- `test_mutation.py` — early check of whether `ffsim.apply_unitary`
  mutates its input, on N2. The same check is now built into
  `run_ordering_pipeline.py`'s `stage0()`.

## One-shot source-patching scripts (unsafe to re-run)

`patch_end_weighted.py`, `patch_largest_J.py`, `patch_pipeline_centered.py`
+ `.sh`, `patch_remaining.py`, `patch_retained_J.py`, `patch_stage3_Jab.py`
— each mutated `run_ordering_pipeline.py`'s source text in place, already
applied. `run_ordering_pipeline.py` has been rewritten many times since
(most recently 2026-08-25; these patches are all 2026-08-21) — the exact
text each patch searches for may no longer exist, and running one against
the current file could silently no-op or corrupt it. Kept only as a record
of what changed and why.

## Pre-project development artefacts (2026-08-18 - 19)

- `POC.ipynb` — proof of concept: orbital permutation invariance and mask
  sensitivity, H2. The starting point for this whole project.
- `SQD_workflow_dev.ipynb` — development copy of the main driver notebook.
- `budget_sweep.ipynb` — early N2 CAS(6,6) budget-sweep notebook.
- `h2_lucj.ipynb` — first LUCJ-on-H2 smoke test.
- `SQD_practice.ipynb`, `n2_lucj.ipynb`, `test.ipynb` — empty stub
  notebooks, never filled in.
- `N2_FCIDUMP`, `hamiltonian_h2.fcidump` — early hand-generated FCIDUMP
  files (N2 CAS(6,6); H2), superseded by the cached, hash-verified
  references under `cache/`.
- `bitstrings_h2_lucj.txt` — sampled bitstring dump from the H2 smoke
  test.
- `sqd_ordering_results.txt` — early run log, 2026-08-19.
- `outputs_dev/` — generated outputs (FCIDUMPs, circuits, bitstrings)
  from the pre-`run_ordering_pipeline.py` development pipeline.

## `legacy_outputs/` (2026-08-20 - 24, pre-`experiments/outputs/` convention)

Everything that used to live directly under root `outputs/`, except
`outputs/unified/` (still live — read by `scripts/run_ordering_pipeline.py`'s
`stage1()` as `canonical_results`, and by `scripts/unified_run.py`'s own
reference cache). This is early H2/N2 interface validation
(`VALIDATION.md`, `h2_sto3g_*`, `n2_equilibrium_*`, `n2_stretched_*`,
`n2_very_stretched_*`, `n2_cas610_155*`), one-off H10 diagnostic sweeps
(`h10_anchor_free/`, `h10_anchor_phase/`, `h10_centered_quick/`,
`h10_end_weighted_quick/`, `h10_largest_J_quick/`, `diag_repro/`,
`directopt/`, `constructed/`, `scaleup/`), and early figures (`figures/`,
5 PDFs — superseded by `results/figures/`'s 47). All predate the
one-directory-per-experiment convention now used under
`experiments/outputs/`; nothing here is read by any current script.

**Correction:** this note originally also claimed nothing here was
referenced by any number in the report. That was wrong — the
reproducibility audit found the report's "five-seed study" figures
(between/within variance ratio 27.2, Kendall τ=0.83) trace to
`seed_replication_n2_cas610_155.csv`, which was filed here. That file has
been moved to `experiments/outputs/n2_seed_stability/` (see its README);
two identical copies remain here, superseded by the relocated one.

## `legacy_logs/` (2026-08-21)

Console logs from the same early H10 smoke-test / diagnostic-sweep runs
as `legacy_outputs/` above. Not referenced by any script or README.
