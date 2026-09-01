# Reproducibility audit — final report

Audit conducted 2026-09-01, this repository at commit range
`42793ee..HEAD`. Scope: every numeric claim in
`notes/orbital_ordering_report.tex` and `notes/presentation/RIKEN Research
Presentation.pptx`, cross-checked against each other, then verified against
raw data at four tiers (independent recomputation, cached-reference
re-derivation, live re-sampling, and a cold start from a fresh clone).

**If you read nothing else, read this paragraph.**

> SQD subspace energies at tight determinant budgets are bit-reproducible
> against **shipped** reference data (Tier 2: 17/17 exact, including Cr2 at
> `diff = 0.0`). They are **not** reproducible from independently
> reconstructed references, even under exact BLAS-build pinning and thread
> pinning: H10 differs by 73 mHa and N2 by 7.3 mHa. The cause is
> element-wise CCSD amplitude variation at the 1e-10 level, which the
> aggregate correlation energy conceals by agreeing to 1e-14. The
> magnitude tracks the boundary ratio $w_{16}/w_{15}$: 0.989 for H10, 0.504
> for N2. **Practical consequence: reference artefacts must be distributed
> with the results. Environment specification alone is insufficient** —
> the conda lockfile added during this audit (`environment.lock.txt`)
> narrows the spread and is good practice, but it is a mitigation, not a
> guarantee, and should not be read as one.

Full derivation of this in `TASK_A_FINDING4.md` (the characterisation) and
`FIX1_LOCKFILE_VERIFICATION.md` (why the lockfile doesn't fully close it).

---

## 1. Claims manifest

242 claims extracted from the `.tex` (systematic grep pass over every
numeric literal, filtered to claims vs. prose/section-reference noise), plus
8 claims added per your Amendments 1/2 and the two lockfile-rebuild results
during the audit. Table-row cells are individually listed
(err/percentile/captured/retained_J each its own claim) rather than
aggregated per row — a deliberate granularity choice, not scope creep; it
puts the manifest above your ~120-180 estimate (242 total).

| Tier | Count | Meaning |
|---|---|---|
| 0 | 188 | Statistic recomputed independently from a raw per-evaluation CSV |
| 1 | 21 | Re-derived from a cached `.npz`/reference, independent formula |
| 2 | 14 | Re-sampled live (12 against the shipped reference, 2 against an independently rebuilt one) |
| 3 | 19 | Narrative / one-off / no corresponding raw CSV found |

| Tolerance class | Count |
|---|---|
| statistic (medians, percentiles, variance ratios) | 81 |
| count (exact) | 50 |
| resampled_energy_mha (fixed seed+shots, shipped reference) | 46 |
| correlation (Spearman/Pearson + p) | 36 |
| rebuilt_reference (Finding-4-affected; no fixed tolerance — the observed spread from Task A/Tier 2 is the record, not a pass/fail epsilon) | 15 |
| exact_energy_ha | 8 |
| exact_bit | 6 |

## 2. Tier 0 — independent recomputation from raw CSVs

**48/49 checks passed.** `verification/verify_tier0.py`; no imports from
`experiments/*.py` or `src/sqd_ordering/`, pandas/numpy/scipy only.

### The one failure

| Claim | Claimed | Recomputed | Diff | Source | Verdict |
|---|---|---|---|---|---|
| `floor_h10_identity_count` | 17 | 16 | −1 | `experiments/outputs/anchor_decomposition_R1.6/c1_all120_identity.csv` | **The data is right, the report text is wrong** (your call, FIX 2) — 15 triples are bit-identical at 458.699662 mHa, one more (458.695209) rounds to 458.70 at 2dp; that is 16 rounding to "458.70 mHa", not 17. Report needs `17` → `16`. |

### Two provenance gaps found and fixed (not failures, but worth recording)

- The report's "five-seed study" (variance ratio 27.2, Kendall τ=0.83) has
  no source in any current experiment directory — it traces to
  `archive/legacy_outputs/seed_replication_n2_cas610_155.csv`. Relocated to
  `experiments/outputs/n2_seed_stability/` with a README during this audit
  (FIX 3a); both figures reproduce exactly from the relocated file.
- The no-α-β ansatz control value (`ansatz_worse_than_control_pct`, 5/120 =
  4.2%) was a stringified Python dict inside `lucj_control`'s
  `metadata.json`. Extracted to `experiments/outputs/lucj_control/no_ab_control.csv`
  (FIX 3b); reproduces exactly.

## 3. Tier 1 — re-derived from cached references

**6/7 checks passed.** `verification/verify_tier1.py`. Rebuilds the LUCJ
operator from `cache/h10_R1.6`'s raw `t1`/`t2` via `ffsim` directly (an
external dependency, not this project's own analysis code); the mask
application and S0/retained_J formulas are independent reimplementations,
their definitions read from `src/sqd_ordering/mask.py`'s docstrings rather
than imported.

- Idempotency (diagonal inclusion): confirmed by construction.
- Reversal invariance: 57/57 permutations.
- S0 invariance under same-spin permutation: 100/100.
- S0 recomputed vs. stored, all 120 H10-identity triples: **120/120 exact.**
- Captured weight recomputed from the cached CASCI vector against 20 sampled
  (chain, triple) determinant-file pairs: **20/20 exact.**
- SHA-256 audit, all 12 `metadata.json` files with hash fields: 25 hashes
  resolved and matched, **0 mismatched**, 1 unresolved (N2's `fcidump_sha256`
  in `n2_anchor_axis/metadata.json` — no candidate file under any tried path
  reproduced it; not a mismatch, just not located).

### The one failure

| Claim | Claimed | Recomputed | Diff | Source | Verdict |
|---|---|---|---|---|---|
| `retained_J` (identity chain only) | 0.2442 | 0.2043 | −0.0399 (16%) | `experiments/outputs/h10_baseline_R1.6/h10_baseline_results.csv` | **Unresolved, low priority.** 7/8 sampled layouts (reverse, physical, physical_reverse, 4 random) match to 1e-3 after fixing a bug in my own check (`positions_from` is a single `argsort`, confirmed by reading the source). Only identity itself — the simplest possible layout — still mismatches, for a reason not identified. `retained_J` is a diagnostic column; S0 and captured weight, which the report's actual shortlist rule depends on, are both exact. Flagged rather than chased further given time budget. |

## 4. Tier 2 — re-sampling spot-check

**17/17 declared checks passed (16 numeric evaluations + the
unmasked-invariance bit-identity summary), all against the shipped cached
reference — including Cr2.** `verification/verify_tier2.py` (+
`_remainder.py`, which finished the last two items after a bug in the
checker itself — not the pipeline — crashed the first run partway through;
fixed and continued rather than re-running the 15 already-confirmed items).
Full table in `tier2_results.csv` / `tier2_results_remainder.csv`.

| Evaluation | Stored | Recomputed | Diff | Wall time |
|---|---|---|---|---|
| H10 identity, default anchors | 300.31919403956664 | 300.31919403956664 | 0.0 | 35.9s |
| H10 identity, best anchors (0,1,2) | 224.59889961938285 | 224.59889961938285 | 0.0 | 38.4s |
| H10 physical, default anchors | 389.71 | 389.7149558584312 | 0.0050 | 38.1s |
| H10 physical, best anchors (2,4,7) | 172.149392 | 172.14939218691416 | 1.9e-7 | 35.3s |
| H10 identity, no-α-β control | 458.6996615694874 | 458.6996615694874 | 0.0 | 35.0s |
| H10 newchain07, (2,4,9) | 219.3822 | 219.38220008660636 | 8.7e-8 | 35.2s |
| H10 newchain11, (0,2,5) | 222.262024 | 222.26202404509098 | 4.5e-8 | 38.7s |
| H10 newchain10, (5,6,7) | 458.378461 | 458.3784609455748 | −5.4e-8 | 40.2s |
| H10 newchain03, (1,2,9) | 184.070279 | 184.07027874691016 | −2.5e-7 | 36.8s |
| N2 identity, default anchors | 31.870454303643218 | 31.870454303643214 | −3.6e-15 | 27.4s |
| N2 identity, best anchors (0,1,9) | 24.267039 | 24.26703865324953 | −3.5e-7 | 27.8s |
| N2 unmasked, 4 orderings (identity, rand_seed101/102/103) | −108.8236445639776 (×4, exact_bit) | −108.8236445639776 (×4) | 0.0 | 23.5-29.1s each |
| Cr2 identity, default anchors | 240.79318115809656 | 240.79318115809656 | 0.0 | 952.8s (15.9 min) |

**Every declared evaluation reproduced to well inside its declared
tolerance**, including all four held-out (chain, triple) pairs (seeded
selection, `newchain07/(2,4,9)`, `newchain11/(0,2,5)`, `newchain10/(5,6,7)`,
`newchain03/(1,2,9)`), the unmasked-permutation invariance check (exact
bit-identity across 4 orderings, as the report claims), and Cr2 (bit-exact,
diff=0.0, at 15.9 minutes — matching the report's own ~16-minute estimate).
All of these use the **shipped cached reference** — consistent with Tier
1's finding and Finding 4: reproducibility holds when the reference is
fixed.

### FIX 4 additions

**H10 identity/default, rebuilt from the lockfile environment:** not
re-run here — already tested extensively during FIX 1 verification (>10
independent attempts: one bit-exact match, then persistent disagreement by
the same ~373 mHa Finding-4-style gap on every subsequent attempt,
including in the original `sqd` environment itself). See
`FIX1_LOCKFILE_VERIFICATION.md`.

**N2 identity/default, rebuilt from the lockfile environment:** tested
once, carefully (the shipped `outputs/unified/reference.pkl` was backed up,
temporarily moved aside to force a genuine rebuild, and restored
immediately after — confirmed via `md5` that the restore was exact).
Result: **39.207351816230585 mHa, not the cached 31.870454 mHa** — a
7.3 mHa gap. Smaller than H10's ~73 mHa gap, consistent with N2's more
comfortable selection-boundary margin (Task A4), but a real, nonzero
mismatch — not the clean pass the one lucky H10 lockfile test suggested.
This value falls squarely within the range Task A4's deliberate amplitude-noise
sweep already produced (24-61 mHa), reinforcing that environment-level
drift and the deliberately-injected perturbation are of comparable
magnitude and origin.

## 5. Tier 3 — cold start

See `COLD_START.md`, `FIX1_LOCKFILE_VERIFICATION.md`, `TASK_A_FINDING4.md`
for the full account. Summary:

**Three blocking defects found and fixed:**
1. `environment.yml` lacked `mpirun`/`mpicxx`/a C++ compiler entirely —
   the README's first command failed from a clean install. Fixed: added
   `openmpi`, `llvm-openmp`, `compilers`.
2. `sbd-build-notes/Configuration.macos-arm64` hardcoded one specific
   user's home directory and conda environment name. Fixed: removed the
   dead `-L` flag (confirmed via `otool` that `mpicxx`'s own
   environment-relative rpath already handles it).
3. `environment.yml`'s `name: sqd` collided with this project's own
   development environment. Fixed: renamed to `sqd-orbital-ordering`.

**One major finding, reclassified as a scientific result, not a defect
(Finding 4):** an independently-rebuilt H10 R=1.6 reference, agreeing with
the cached one to 1e-14 Ha in every stored scalar, gives a materially
different SQD answer at the identity/default-anchor configuration (373.63
mHa vs. the cached 300.32 mHa). Characterised in full across five sub-checks
(`TASK_A_FINDING4.md`): the report's own $w_{16}/w_{15}=0.989$ near-degeneracy
is the correct explanation (confirmed by direct comparison against N2's
0.504 ratio, which is far more robust to the same perturbation), the
underlying cause is CCSD amplitude tensor elements differing by up to 1e-10
relative between independent builds (not the 1e-14 the aggregate energy
agreement suggests), and this is not resolved by more shots (16x range
tested, no effect).

**FIX 1 (conda lockfile) is a mitigation, not a fix.** Do not read it as
closing the reference-reconstruction gap — it doesn't, and the evidence
against that reading is direct: N2, rebuilt from the lockfile environment,
still misses the cached reference by 7.3 mHa (39.21 vs. 31.87), and H10's
one clean lockfile match did not repeat on any subsequent attempt,
including in the original `sqd` environment itself, package state provably
unchanged (`conda-meta/history` timestamp predates this session). The
`libblas`/`liblapack` build mismatch (build 9 vs. 10 of the same nominal
3.11.0 version) the lockfile pins against is real and narrows the spread —
but there is a second, distinct, currently-unexplained source of drift on
top of it. The only verified guarantee remains: use the shipped cached
reference. Full record in `FIX1_LOCKFILE_VERIFICATION.md`.

| Cold-start step | Result | Wall time |
|---|---|---|
| `git clone` | OK | 1.7s (local) |
| `conda env create -f environment.yml` | OK, after Fix 1 | ~24s (warm cache) |
| `sbd` build, fixed `Configuration` | OK | 6.0s |
| README benchmark command | **PASS**, exact match | 2m31s |
| H10 R=1.6 reference from scratch | **PASS**, 1e-14 Ha agreement | 24.6s |
| One full SQD evaluation on the fresh reference | **Reference-dependent — see Finding 4** | 35s |

## 6. What's independent vs. shared-code

- **Independent** (Tier 0, Tier 1's formulas): no imports from
  `experiments/*.py` or `src/sqd_ordering/`. Tier 1 uses `ffsim` directly
  (external dependency, same substrate the whole project relies on) to
  build the LUCJ operator, but the mask/S0/retained_J logic is
  reimplemented from the definitions, not imported.
- **Shared-code / re-execution** (Tier 2, parts of Tier 3): re-runs the
  actual pipeline (`scripts/run_ordering_pipeline.py`,
  `scripts/unified_run.py`, `experiments/tm_transfer.py`) — appropriate for
  a spot-check, since the question there is "does re-running the real
  pipeline reproduce the stored number," not "is the pipeline's logic
  correct" (that's what Tier 0/1 test).

## 7. Plain statement: what a reader can reproduce, and at what cost

- **The group benchmark** (`sbd` alone, no Python): reproduces exactly,
  ~2.5 minutes, from a genuinely fresh clone + environment, after the three
  Tier-3 fixes.
- **Every statistic in Tiers 0/1** (188 + 21 = 209 of 242 claims):
  reproduces from the cached data shipped in this repository, independently
  recomputed, in well under a minute total (`make verify`).
- **Individual per-layout SQD energies** (the `resampled_energy_mha` class,
  46 claims): reproduce exactly **from the shipped cached reference**, at
  the cost of the sampling + `sbd` wall time (~30s-16min depending on
  system size). They do **not** reproduce from an independently rebuilt
  reference, lockfile-pinned or not — confirmed on both systems tested
  (H10 off by 73 mHa, N2 by 7.3 mHa), worse the closer the configuration
  sits to a selection-boundary tie ($w_{16}/w_{15}$ near 1) but not zero
  even for N2's comparatively comfortable 0.504 ratio. This is now
  documented in the README as the headline reproducibility finding, not a
  footnote.
- **19 claims (tier 3)** have no corresponding raw CSV in this repository
  (one-off historical comparisons predating the current experiment
  structure, e.g. the `n_reps=2` vs. `None` error figures, the Fiedler
  ordering percentiles) and are not independently reproducible here at all
  — flagged with `source: "none found"` in the manifest rather than
  silently matched to an unrelated file.
