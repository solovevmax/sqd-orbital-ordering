# Chain-aware anchor selection, round 3: tie-break, interface score, shortlist recall

**Script:** `experiments/chain_aware_v3.py`

**Question.** Three re-framings of the same problem, all zero/near-zero
free parameters, all screened by an automatic chain-invariance pre-flight
(reject immediately if a candidate's raw value doesn't change when the
same-spin chain is permuted -- the actual failure mode that doomed
chain_aware.py's S1-S4). P0: can a zero-new-parameter tie-break fix S0's
one documented failure (newchain11) without introducing new ones? P1: does
`S_int` (opposite-spin coupling the mask discarded near the anchor, times
same-spin coupling to its chain neighbour -- a genuinely chain-aware,
zero-parameter score) do better than S0? P2: is argmax the wrong target --
does reframing to shortlist recall (best-of-top-k) change the picture?

**Protocol.** 18 chains: 6 for development (H10 identity/physical/
rand007, N2 identity/reverse/r039), 12 strictly held out
(chain_aware_phaseB.py's newchain00-11), evaluated once, numbers final.
Cr2's 3 chains untouched -- reserved for a later transfer test, not part
of this task's scope. `S_int(c,A) = sum_{a in A} sum_{n in same-spin chain
neighbours of a} |J_ab[a,n]| * |J_same[a,n]|`, using the FULL (unmasked)
opposite-spin matrix; endpoint anchors have one neighbour, interior
anchors two. `S0+S_int` normalised to unit variance per chain from the
candidate set alone (never from the outcome). All cached data, no new
sampling.

**Invariance pre-flight: both S_int and S0+S_int PASS** (raw value
changes across every chain tested) -- confirmed genuinely chain-aware,
unlike S1-S4.

## P0 -- tie-break result

16/18 chains have an exact S0 tie at the top (mostly the established
`(0,1,9)`/`(0,8,9)` degeneracy, which is chain-invariant and so recurs
almost everywhere). The tie-break changes the pick at 3/18 chains. At
**newchain11 it works exactly as intended**: regret 1.152 -> 0.088, well
under 1.0. But it is **not free**: at **newchain01 it makes things
markedly worse** (regret 0.000 -> 0.867) by moving off an S0-optimal pick
onto the tie-break's, and newchain10 degrades marginally (0.234 ->
0.235). Net effect: worst-case regret across 18 chains improves (1.152 ->
0.867), but median regret is flat to very slightly worse (0.347 ->
0.348), and it is not a clean win -- **2/18 chains are made worse.** The
honest characterisation is a worst-case improvement traded for a new,
comparably-sized single-chain regression, not a repair with no downside.

## P1/P2 -- shortlist recall, held out (12 chains, final numbers)

| | S0 | S_int | S0+S_int | random (median of 2000 draws) |
|---|---|---|---|---|
| best-of-top-5 regret, median | **0.002** | 0.036 | 0.008 | 0.181 |
| best-of-top-5 regret, p90 | **0.002** | 0.036 | 0.008 | 0.621 |
| top-5 recall | 0.500 | 0.333 | 0.500 | 0.118 |
| beats default anchors (top-5) | 0.917 (11/12) | 0.833 (10/12) | 0.917 (11/12) | -- |

S0's best-of-top-5 regret at held-out chains is already remarkably small
(median 0.002 -- essentially solved once a budget-5 shortlist is allowed,
not just argmax). Neither S_int nor the combined score improves on this;
both are worse on every P2 metric than S0 alone. S_int does, however,
clear the correct-sign and invariance bars cleanly, and comes within a
single chain of clearing "never worse than random": at newchain02, best-
of-top-5 regret is 0.460 vs. random's median 0.397 -- the only miss out of
12 held-out chains, and a narrow one.

## Acceptance criteria (declared in advance)

| Criterion | Result |
|---|---|
| Invariance pre-flight PASS | **YES** |
| Correct-sign rho(S_int, captured), every held-out chain | **YES** (12/12, all p<0.05) |
| Best-of-top-5 regret, median below S0's | NO (0.036 vs 0.002) |
| Best-of-top-5 regret, p90 below S0's | NO (0.036 vs 0.002) |
| No held-out chain worse than random | NO (newchain02 only, 0.460 vs 0.397) |

**Verdict: S_int does not beat S0 on the declared criteria.** This is a
third closed negative result in this line of work, but a more qualified
one than S1-S4 or chain_aware_v2's T1/T2: S_int is legitimately
chain-aware (passes the pre-flight those failed), correctly signed
everywhere tested, and close to clearing "no chain worse than random" --
it simply does not clear the high bar S0 already sets once the objective
is reframed to shortlist recall rather than argmax. The P0 tie-break is a
genuine partial win (fixes the worst point on the map) at a real, quantified
cost elsewhere, not a strict improvement.

See `report.txt`, `p0_tiebreak.csv`, `p2_shortlist.csv`, `all_scores.csv`,
`preflight_S_int.csv`, `preflight_S0_plus_Sint.csv`, `metadata.json`.
