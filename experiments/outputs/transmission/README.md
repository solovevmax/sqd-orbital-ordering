# Transmission -- where does ansatz quality fail to reach SQD outcome?

**Script:** `experiments/transmission.py`

**Question.** retained_J_oppspin predicts err_lucj strongly (rho = -0.850
H10, -0.965 N2), but err_lucj only weakly predicts err_sqd (+0.475 H10,
+0.432 N2), and that link collapses entirely at H10/physical and N2/r039.
Which link in the chain (retained_J_oppspin -> err_lucj -> captured ->
err_sqd) actually breaks?

**Protocol.** 400 fresh evaluations across 6 chains (H10 identity/
physical/rand007, N2 identity/reverse/r039), cross-validated 400/400 exact
against every previously-cached err_sqd and retained_J_oppspin value
before trusting the new diagnostics (sampling-distribution entropy, Gini,
top-15 mass, w16/w15 boundary ratio -- none of which were ever persisted
by prior scripts, only the post-selection top-15 dets).

**Headline.** The break is link 1 (ansatz quality -> sampling
concentration), specifically at H10/physical and H10/rand007 -- link 2
(subspace capture -> answer) holds at ALL 6 chains without exception
(rho -0.88 to -0.96 everywhere). No single sampling diagnostic explains
the residual well. The over-concentration hypothesis is rejected (the
rule's wrong picks are not more concentrated than the true best in any of
4 disagreeing chains). This is the dataset `experiments/chain_aware.py`
reuses directly for its Phase A score screen. See `report.txt`,
`all_evaluations.csv`.
