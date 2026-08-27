# Chain-aware anchor score: Phase A (construct/screen) and Phase B (out-of-sample)

**Scripts:** `experiments/chain_aware.py` (Phase A),
`experiments/chain_aware_phaseB.py` (Phase B evaluations),
`experiments/chain_aware_phaseB_analysis.py` (Phase B analysis, B3.1-B3.5)

**Question.** retained_J_oppspin ("S0") is exactly chain-invariant -- it
depends only on which orbitals are anchored, not on the same-spin
permutation -- which is why its correlation with outcome degrades badly
across chains. Can a *chain-aware* score (using same-spin coupling
strength in the anchor's neighbourhood) do better?

**Protocol (Phase A).** Five pre-declared, closed scores (S0 control, plus
S1 reach-weighted / S2 coverage-penalised / S3 amplitude-reach / S4
combined), one declared hyperparameter sweep, selected on H10/N2 identity
only, then screened for worst-case rho(err_sqd) across all 6
`transmission.py` chains -- no new sampling.

**Protocol (Phase B).** 12 new H10 chains never used in any previous
experiment (rng seed 20260827001, verified disjoint from the 57-chain
baseline pool), 40 shared triples + S0's top-1/top-3 + default anchor +
no-alpha-beta floor per chain, 2e6 shots.

**Headline.** Phase A: none of S1-S4 beats S0 on worst-case rho -- a
clean negative result. Phase B validates S0 itself out of sample: link 2
(captured -> err_sqd) holds at 12/12 never-before-examined chains (the
strongest generalisation test the mechanism has had), but S0's own rho
sign-flips at the ansatz level on 8/12 chains, and the 4.8x compression
figure (G1-lite, n=8) becomes 3.15x at n=20. Verdict: the capture
mechanism generalises; S0 as a selection rule does not -- chain-dependence
is not solved. See `phaseA_report.txt`, `phaseB_report.txt`,
`step2_analysis_report.txt`.
