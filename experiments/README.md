# Experiments

Each subdirectory under `outputs/` is one experiment: its own README
(question, protocol, headline numbers), CSVs, and metadata.json (git
commit hash + sha256 of every cached input it read). Roughly chronological
order below; later experiments build on earlier ones' cached data.

| Experiment | Script(s) | Question |
|---|---|---|
| [preflight](outputs/preflight/README.md) | `preflight.py` | Do mechanism A (N2) and B (H10) agree entrywise? |
| [h10_baseline](outputs/h10_baseline_R1.6/README.md) | `h10_baseline.py` | Same-spin ordering spread at the default anchor |
| [score_audit (Part A)](outputs/score_audit_R1.6/README.md) | `score_audit.py` | Does any cheap score predict H10 error? |
| [anchor_decomposition (Part B, C1/C2)](outputs/anchor_decomposition_R1.6/README.md) | `anchor_decomposition.py`, `anchor_hardening.py` | Does anchor selection matter more than ordering? |
| [anchor_reanalysis (D1-D6)](outputs/anchor_reanalysis/README.md) | `anchor_reanalysis.py` | Where exactly is the mechanism's bottleneck? |
| [budget_transfer (E1, E2)](outputs/budget_transfer/README.md) | `floor_investigation.py`, `budget_transfer.py` | The degenerate floor; cheap-budget screening |
| [floor_generalization (F1)](outputs/floor_generalization/README.md) | `floor_generalization.py` | Is the mask a net liability at some orderings? |
| [g1_lite](outputs/g1_lite/README.md) | `g1_lite.py` | Do the two levers interact after optimisation? |
| [lucj_control (+ V1-V5)](outputs/lucj_control/README.md) | `lucj_control.py`, `lucj_verification.py` | Is the effect ansatz-level or an SQD artefact? |
| [n2_anchor_axis](outputs/n2_anchor_axis/README.md) | `n2_anchor_axis.py` | Does the anchor effect transfer to N2? |
| [transmission](outputs/transmission/README.md) | `transmission.py` | Which link in the chain actually breaks? |
| [chain_aware (Phase A/B)](outputs/chain_aware/README.md) | `chain_aware.py`, `chain_aware_phaseB.py`, `chain_aware_phaseB_analysis.py` | Can a chain-aware score fix the chain-dependence? |
| [tm_transfer](outputs/tm_transfer/README.md) | `tm_transfer.py`, `tm_transfer_analysis.py` | Does everything survive a transfer to a localised transition-metal active space (Cr2)? |
| [chain_aware_v2](outputs/chain_aware_v2/README.md) | `chain_aware_v2.py` | Can a score that targets capture directly (not variational energy) beat S0? |
| [transpilation_audit](outputs/transpilation_audit/README.md) | `transpilation_audit.py` | Are the fixed-resource comparisons actually resource-neutral on real heavy-hex hardware? |
| [chain_aware_v3](outputs/chain_aware_v3/README.md) | `chain_aware_v3.py` | Tie-break, interface score, and shortlist-recall reframing -- does anything beat S0 chain-by-chain? |
| [n2_seed_stability](outputs/n2_seed_stability/README.md) | none (pre-restructuring dataset) | Is the N2 ordering effect signal or seed noise, at 5 seeds? |

`figures.py` generates the publication figure set in `results/figures/`
from several of the CSVs above; it is not its own experiment.
