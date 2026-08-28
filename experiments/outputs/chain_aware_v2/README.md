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

**Headline.** Both scores FAIL, and not marginally: T1's worst-case
rho(S,captured) = -0.486, T2's = -0.644 -- both *significantly
anti-correlated* with captured at several chains, not merely uninformative
(mean rho across all 18 chains is negative for both: T1=-0.14, T2=-0.21).
S0 stays positive even at its own worst chain (+0.361, N2/r039). A second
closed negative result: targeting capture instead of variational energy
does not fix the chain-dependence problem -- the failure is structural, not
a matter of what the score is built to predict. See `report.txt`,
`comparison_table.csv`, `all_scores.csv`, `t1_sweep.csv`.

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
