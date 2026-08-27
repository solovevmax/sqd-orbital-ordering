# Transition-metal transfer: Cr2 CAS(12,12)

**Scripts:** `experiments/tm_transfer.py` (Stage 0-2),
`experiments/tm_transfer_analysis.py` (Stage 3)

**Question.** Every anchor result so far is on H10 (a hydrogen chain) and
N2 (a first-row diatomic). Does the phenomenon, its mechanism (subspace
capture), and its chain-dependent failure mode survive a move to a
compact localised transition-metal active space, the intended application
domain?

**Protocol.** Cr2 CAS(12,12) at R=1.68 A, cc-pVDZ, AVAS-selected (Cr 3d+4s,
threshold=0.3, robust across 0.25-0.5), occupied/virtual block-Boys
localised, 24 qubits. A declared H12/STO-6G fallback was NOT needed --
CCSD converged cleanly (26 cycles, e_corr=-0.610 Ha). Both reference gates
passed (E_corr match 6.7e-16, CASCI match 7.3e-12). Budget=55 (0.354% of
the 853,776-dim CASCI space, matching H10's fraction exactly). 220 anchor
triples at 3 chains (identity, reverse, one random permutation) for
err_lucj (660 evals, no sampling); 60 shared triples + default anchor +
no-alpha-beta control at each chain for full SQD (186 evals, 2e6 shots --
each evaluation costs ~16 minutes here, entirely from 24-qubit statevector
simulation, not shot count; total wall time 683 minutes / 11.4 hours with
8-way parallelism).

**Headline.** Replicates on every axis tested: capture mechanism holds
(rho(captured,err_sqd) = -0.97, even stronger than H10/N2), the
ansatz-level rule replicates strongly (rho(S0,err_lucj) = -0.96 to -0.97,
at the strong end of the H10/N2 range), the anchor effect is comparable in
headroom-normalised terms (1677 mHa/unit vs. H10's 1445 and N2's 2550),
and the best triple still moves between chains (chain-dependence is not
solved here either). Identity and reverse chains give bit-identical
results, confirming the established same-spin reversal-invariance
(B0, `anchor_decomposition_R1.6/`) holds on this system too. See
`run_report.txt`, `stage3_report.txt`, `stage1_ansatz.csv`,
`stage2_sqd.csv`, `stage3_per_chain_summary.csv`.
