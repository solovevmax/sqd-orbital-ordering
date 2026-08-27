# LUCJ-only control, and V1-V5 verification

**Scripts:** `experiments/lucj_control.py`, `experiments/lucj_verification.py`
(same output directory)

**Question.** Is the anchor effect an SQD artefact (sampling + subspace
selection), or a property of the LUCJ ansatz itself? And is the identity-
chain correlation a floor artefact, or does it survive removing the
degenerate no-alpha-beta triples?

**Protocol.** No sampling, no sbd, no new reference data: build the exact
CAS Hamiltonian from the cached FCIDUMP, evaluate `<psi|H|psi>` of the
masked LUCJ state directly in ffsim's number-conserving statevector
representation. 120 triples at H10 identity; 40 shared at physical/rand007.
V1-V5: sign-convention check; re-run rho with floor triples excluded, at
all three chains; rho(err_lucj, captured); rho(err_lucj,
retained_J_oppspin) to localise where chain-dependence enters.

**Headline.** Both real and uneven: rho(err_lucj, err_sqd) = +0.527 at
identity, survives floor exclusion (not a floor artefact) -- but collapses
at physical (rho = -0.160, not significant), and best-by-LUCJ is NOT
best-by-SQD (ranks 39/120 and 49/120 respectively). This is the
"CASE C" result (mixed, not a clean ansatz-vs-SQD-artefact split) that
`experiments/transmission.py` later localises to link 1 specifically
(ansatz quality does not reliably concentrate the sampling distribution).
See `report.txt`, `verification_report.txt`, `identity_120_lucj.csv`,
`physical_rand007_40_lucj.csv`.
