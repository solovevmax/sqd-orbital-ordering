# Chain-aware score, second attempt: target capture directly

**Script:** `experiments/chain_aware_v2.py`

**Question.** S0 (retained_J_oppspin) predicts err_lucj strongly and
consistently everywhere (rho -0.85 to -0.97) -- that link is not broken.
The broken link is err_lucj -> captured (transmission.py's link 1). Can a
score that targets CAPTURE directly, rather than variational energy, do
better than S0 out of sample?

**Protocol.** Two pre-declared, closed scores, no new sampling (captured
already cached for every triple evaluated across the project):
- T1 "anchor-conditioned reachability": amplitude-weighted fraction of the
  top-K |t2| double-excitation channels whose alpha leg (i->a) and beta leg
  (j->b) are within L chain-positions AND at least one of the channel's 4
  orbitals is within D positions of an anchor. Swept K in {20,50}, L in
  {2,3,4}, D in {0,1} on H10/N2 identity only; frozen at K=50, L=4, D=0
  (mean |rho|=0.205 on the dev chains -- notably weak even there).
- T2 "perturbative support overlap": c_pred[i,j,a,b] = t2[i,j,a,b] *
  (Jaa*mask)[i,a] * (Jaa*mask)[j,b] * (retained Jab diagonal weight at the
  channel's orbitals); top-k (k=15, the fixed SQD budget) entries by
  |c_pred|^2 scored by the true CCSD |t2|^2 weight they carry.

Evaluated across 18 chains: H10 identity/physical/rand007 + the 12 chains
from `chain_aware_phaseB.py` + N2 identity/reverse/r039.

**Headline -- distributional result (primary).** Across all 18 chains, S0 is
uniformly strong, not borderline: median rho(S0,captured)=+0.703 (mean
+0.646), correct sign at 18/18 chains and statistically significant
(p<0.05) at 18/18; median regret_frac=0.347, and only 1/18 chains exceed
regret_frac=1.0. T1 and T2 are the opposite of uniform: median
rho(captured) is *negative* for both (T1=-0.149, T2=-0.233), wrong-signed
at 14/18 and 15/18 chains respectively, and regret_frac exceeds 1.0 at
4/18 (T1) and 9/18 (T2) chains. This is a second closed negative result --
targeting capture instead of variational energy does not fix the
chain-dependence problem -- but the shape of the result is that S0 is
robustly good and T1/T2 are robustly unreliable, not a close contest
decided by one hard chain. Full distribution (median/mean/min/max, sign
and significance counts) in `distribution_summary.csv` and `report.txt`.

**Worst case (kept, not the headline).** S0's single weakest chain still
has rho(captured)=+0.361 (N2/r039) -- it never flips sign anywhere. T1's
worst is -0.486 (H10/newchain03), T2's is -0.644 (H10/newchain01) -- for
both, the "worst case" is representative of a generally unreliable score,
not an outlier pulling down an otherwise-good one. This invariance
property (S0 never wrong-signed) is worth keeping visible alongside the
distribution, even though it is not the headline finding.

**S0's one weak chain: newchain11 (regret_frac=1.152, its only >1.0).**
Investigated and characterised as a single anomalous chain, not a
systematic failure. S0's top pick there, `(0,1,9)`, ranks 20/43 by true
err_sqd, 50.78 mHa above the true best `(2,4,8)`. Root cause: S0 is
chain-invariant and ties EXACTLY between `(0,1,9)` and `(0,8,9)`
(identical Jab-diagonal sum, S0=12.151582) at every one of the 13 chains
where both were evaluated -- a genuine numerical degeneracy, not noise --
broken by an uninformative lexicographic rule that happens to prefer the
worse triple at 6/13 chains, including newchain11. But regret_frac only
exceeds 1.0 at newchain11, because that chain independently has the
smallest error range (89.48 mHa, less than half the next-smallest of the
12) and lowest no-alpha-beta floor (248.24 mHa) of the whole held-out set
-- the same ~47-51 mHa absolute tie-driven cost that produces regret_frac
0.46-0.54 at three other affected chains (newchain02/06/08) is normalised
by newchain11's unusually small denominator into the one value over 1.0.
S0's correlation with captured at newchain11 itself is unaffected
(+0.609, correct sign, significant) -- the tie degrades one pick, not the
score's overall ranking quality. Full detail, including the shared-tie
table across all 13 chains and newchain11's context against the other 11,
in `newchain11_tie_analysis.csv`, `newchain11_context.csv`, and
`report.txt`.

See `report.txt`, `comparison_table.csv`, `all_scores.csv`, `t1_sweep.csv`,
`distribution_summary.csv` for full detail.

**Caveat on a self-caught bug**: the first run's verdict logic had the
sign convention backwards for rho(captured) -- it copied the
"worst case = max()" rule from the err_sqd convention (where negative is
good, so the *least*-negative value is the worst case) without re-deriving
it for captured, where positive is good and worst case is therefore the
*smallest* (most negative) value, i.e. `.min()`, not `.max()`.

This affected exactly one derived quantity: `worst_rho_captured` (and
everything downstream of it -- the beats-S0 comparison and the verdict).
It did NOT affect the underlying per-chain rho(captured)/rho(err_sqd)
values themselves (`comparison_table.csv` was always correct), the T1
hyperparameter sweep (its own selection criterion used `mean_abs_rho`,
already sign-independent), `worst_rho_sqd` (err_sqd's convention was
`.max()` in both versions, correctly), or the regret figures.

Before the fix (`.max()` on rho_captured -- silently picked each score's
*best* chain and mislabelled it "worst"):
  - S0 worst_rho_captured reported as **+0.809** (H10/newchain07's actual
    value -- S0's best chain, not its worst)
  - T1 worst_rho_captured reported as **+0.218** (H10/newchain05's value --
    T1's best chain)
  - T2 worst_rho_captured reported as **+0.156** (H10/newchain00's value --
    T2's best chain)
  - Verdict: **"T1 BEATS S0 by a clear margin" and "T2 BEATS S0 by a clear
    margin"** -- both false positives.

After the fix (`.min()`, the true worst chain for each score):
  - S0 worst_rho_captured = **+0.361** (N2/r039)
  - T1 worst_rho_captured = **-0.486** (H10/newchain03)
  - T2 worst_rho_captured = **-0.644** (H10/newchain01)
  - Verdict: both fail to beat S0, and not marginally -- they are
    significantly anti-correlated with captured at their worst chains,
    which the buggy version had exactly inverted into an apparent win.
