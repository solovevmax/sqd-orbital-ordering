# Part A -- score audit

**Script:** `experiments/score_audit.py`

**Question.** Of the 11 no-sampling score variants computable from the
mask and CCSD amplitudes (score1/score2 and their sub-components, plus
retained_J), which one predicts H10 subspace error well enough to select
orderings without running SQD?

**Protocol.** All 11 score variants computed for the 57 baseline orderings
(h10_baseline_R1.6), audited against err_mHa: Spearman rho, an exact
combinatorial null test (rank-uniform-under-exchangeability), and a
same-spin/opposite-spin sector split of retained_J.

**Headline.** None of the 11 variants predicts H10 err_mHa -- a clean
negative result. This is what motivated Part B's anchor decomposition
(does the SAME-SPIN vs OPPOSITE-SPIN split matter more than any single
combined score?) and, later, the whole chain-aware-score line of work
(`experiments/chain_aware.py`), which reached the same negative
conclusion for a differently-constructed score family. See
`all_scores.csv` and `metadata.json`.
