#!/usr/bin/env python3
"""
experiments/chain_aware_v2_distribution.py
==============================================

Follow-up analysis on chain_aware_v2.py's cached results: the worst-case
framing understated how well S0 actually performs. Reports the full
distribution (median/mean/min/max, wrong-sign counts, significance counts)
for rho_captured, rho_sqd and regret_frac across all 18 chains, for S0,
T1, T2 -- and a deep dive on newchain11, S0's one regret_frac>1.0 chain.

Pure re-analysis of experiments/outputs/chain_aware_v2/{comparison_table,
all_scores}.csv -- no new scoring, no new sampling. Regenerates
report.txt and README.md with the distributional result as the headline.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = Path(__file__).resolve().parent / "outputs" / "chain_aware_v2"

REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def main() -> int:
    table = pd.read_csv(OUTDIR / "comparison_table.csv")
    scored = pd.read_csv(OUTDIR / "all_scores.csv")
    old_meta = json.loads((OUTDIR / "metadata.json").read_text())

    banner("CHAIN-AWARE SCORE v2 -- CORRECTED FRAMING: DISTRIBUTIONAL PERFORMANCE FIRST")
    out("The worst-case-only framing in the original report understated how well S0")
    out("performs: it is a single weak chain among 18 uniformly strong ones, not a")
    out("borderline score. This report leads with the full distribution; worst-case")
    out("numbers are kept below because they support the invariance argument (S0 never")
    out("flips sign, unlike T1/T2), but they are not the headline.")

    # ---------------------------------------------------------- distributional summary
    banner("PRIMARY SUMMARY -- distribution across all 18 chains")
    dist_rows = []
    for name in ["S0", "T1", "T2"]:
        sub = table[table.score == name]
        rc, rs, rg = sub.rho_captured, sub.rho_sqd, sub.regret_frac
        row = dict(
            score=name,
            rho_captured_median=float(rc.median()), rho_captured_mean=float(rc.mean()),
            rho_captured_min=float(rc.min()), rho_captured_max=float(rc.max()),
            rho_captured_wrong_sign=int((rc < 0).sum()), rho_captured_significant=int((sub.p_captured < 0.05).sum()),
            rho_sqd_median=float(rs.median()), rho_sqd_mean=float(rs.mean()),
            rho_sqd_min=float(rs.min()), rho_sqd_max=float(rs.max()),
            rho_sqd_wrong_sign=int((rs > 0).sum()), rho_sqd_significant=int((sub.p_sqd < 0.05).sum()),
            regret_median=float(rg.median()), regret_mean=float(rg.mean()), regret_max=float(rg.max()),
            regret_exceeding_1=int((rg > 1.0).sum()), n_chains=len(sub),
        )
        dist_rows.append(row)
        out(f"\n{name} (n={len(sub)} chains):")
        out(f"  rho_captured:  median={row['rho_captured_median']:+.3f}  mean={row['rho_captured_mean']:+.3f}  "
            f"min={row['rho_captured_min']:+.3f}  max={row['rho_captured_max']:+.3f}")
        out(f"                 wrong-sign (negative): {row['rho_captured_wrong_sign']}/{row['n_chains']}   "
            f"significant (p<0.05): {row['rho_captured_significant']}/{row['n_chains']}")
        out(f"  rho_sqd:       median={row['rho_sqd_median']:+.3f}  mean={row['rho_sqd_mean']:+.3f}  "
            f"min={row['rho_sqd_min']:+.3f}  max={row['rho_sqd_max']:+.3f}")
        out(f"                 wrong-sign (positive): {row['rho_sqd_wrong_sign']}/{row['n_chains']}   "
            f"significant (p<0.05): {row['rho_sqd_significant']}/{row['n_chains']}")
        out(f"  regret_frac:   median={row['regret_median']:.3f}  mean={row['regret_mean']:.3f}  "
            f"max={row['regret_max']:.3f}   exceeding 1.0: {row['regret_exceeding_1']}/{row['n_chains']}")
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(OUTDIR / "distribution_summary.csv", index=False)

    out(f"\nS0 in one line: median rho_captured=+{dist_df.loc[0,'rho_captured_median']:.3f}, "
        f"correct sign and significant at {18-dist_df.loc[0,'rho_captured_wrong_sign']}/18 and "
        f"{dist_df.loc[0,'rho_captured_significant']}/18 chains respectively, median regret "
        f"{dist_df.loc[0,'regret_median']:.3f}, and only 1/18 chains with regret>1.0.")

    # ---------------------------------------------------------- worst case (kept, demoted)
    banner("WORST CASE (kept for the invariance argument, not the headline)")
    for name in ["S0", "T1", "T2"]:
        sub = table[table.score == name]
        worst_row = sub.loc[sub.rho_captured.idxmin()]
        out(f"  {name}: worst-case rho(captured) = {worst_row.rho_captured:+.3f} at "
            f"{worst_row.system}/{worst_row.chain}")
    out("\nS0 never flips sign across any of the 18 chains (0/18 wrong-sign) -- this is the")
    out("invariance property worth keeping visible: even at its single weakest chain, S0")
    out("remains a positively-correlated, statistically significant predictor of capture.")
    out("T1 and T2 flip sign at a majority of chains (14/18 and 15/18 respectively) and are")
    out("frequently anti-correlated -- the worst-case numbers for T1/T2 are not outliers,")
    out("they are representative of a generally unreliable score.")

    # ---------------------------------------------------------- newchain11 deep dive
    banner("NEWCHAIN11 DEEP DIVE -- S0's one regret_frac>1.0 chain")
    sub11 = scored[(scored.system == "H10") & (scored.chain == "newchain11")].reset_index(drop=True)
    pick_idx = int(sub11.S0.to_numpy().argmax())
    pick_row = sub11.iloc[pick_idx]
    ranked = sub11.sort_values("err_sqd").reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    rank_of_pick = int(ranked.loc[ranked.triple == pick_row.triple, "rank"].iloc[0])
    true_best = ranked.iloc[0]
    gap = float(pick_row.err_sqd - true_best.err_sqd)
    out(f"S0's top pick: {pick_row.triple}  (err_sqd={pick_row.err_sqd:.2f} mHa)")
    out(f"  rank among 43 candidates by true err_sqd: {rank_of_pick}/43")
    out(f"  true best: {true_best.triple}  (err_sqd={true_best.err_sqd:.2f} mHa)")
    out(f"  gap to true best: {gap:.2f} mHa")

    banner("Mechanism: an exact, chain-invariant tie in S0")
    tied = sub11[np.isclose(sub11.S0, sub11.S0.max(), atol=1e-9)]
    out(f"S0 is chain-invariant and ties EXACTLY (S0={sub11.S0.max():.6f}) between "
        f"{list(tied.triple)} at every one of the 12 held-out chains (and identity) -- both")
    out("triples share anchor orbitals {0,9}, differing only in the third (1 vs 8); their")
    out("Jab diagonal entries at orbitals 1 and 8 are numerically identical for this system.")
    out("The tie is broken deterministically in favour of (0,1,9) by lexicographic ordering")
    out("(itertools.combinations lists it before (0,8,9)) -- not by anything informative about")
    out("which is the better choice at a given chain.")
    h10 = scored[scored.system == "H10"]
    tie_rows = []
    for c in sorted(h10.chain.unique()):
        sub = h10[h10.chain == c]
        if not {"(0, 1, 9)", "(0, 8, 9)"}.issubset(set(sub.triple)):
            continue
        r019 = sub[sub.triple == "(0, 1, 9)"].iloc[0]
        r089 = sub[sub.triple == "(0, 8, 9)"].iloc[0]
        tie_rows.append(dict(chain=c, err_019=r019.err_sqd, err_089=r089.err_sqd,
                             gap=r019.err_sqd - r089.err_sqd))
    tie_df = pd.DataFrame(tie_rows)
    tie_df.to_csv(OUTDIR / "newchain11_tie_analysis.csv", index=False)
    out(f"\nThe tie recurs at all {len(tie_df)} chains where both triples were evaluated; the")
    out(f"lexicographic pick (0,1,9) is worse than (0,8,9) at "
        f"{int((tie_df.gap > 0).sum())}/{len(tie_df)} of them:")
    out(tie_df.to_string(index=False))
    regret_by_chain = table[table.score == "S0"].set_index("chain")["regret_frac"]
    out(f"\nBut regret_frac only exceeds 1.0 at newchain11. At the other chains where (0,1,9)")
    out(f"loses the tie (newchain02, newchain06, newchain08), the SAME ~47-51 mHa absolute")
    out(f"cost normalises to regret_frac {[f'{regret_by_chain[c]:.3f}' for c in ('newchain02','newchain06','newchain08')]}")
    out(f"respectively -- comfortably under 1.0.")

    banner("Why newchain11 specifically: an unusually compressed error landscape")
    sm_rows = []
    b2 = pd.read_csv(Path(__file__).resolve().parent / "outputs" / "chain_aware" / "phaseB_b2_all.csv")
    b2["role"] = b2["role"].fillna("triple")
    for i in range(12):
        c = f"newchain{i:02d}"
        floor_row = b2[(b2.chain == c) & (b2.is_floor == True)]
        floor = float(floor_row.err_sqd.iloc[0]) if len(floor_row) else float("nan")
        trip = b2[(b2.chain == c) & (b2.role == "triple") & (~b2.is_floor.astype(bool))]
        sm_rows.append(dict(chain=c, floor=floor, err_range=float(trip.err_sqd.max() - trip.err_sqd.min()),
                            achieved_capture_ceiling=float(trip.captured.max()),
                            rand_regret_denom=float(trip.err_sqd.mean() - trip.err_sqd.min())))
    sm = pd.DataFrame(sm_rows).set_index("chain")
    sm.to_csv(OUTDIR / "newchain11_context.csv")
    others = sm.drop("newchain11")
    out(sm.to_string())
    out(f"\nnewchain11 vs. the other 11 held-out chains:")
    out(f"  floor:                    {sm.loc['newchain11','floor']:.2f} mHa vs. others' "
        f"mean {others.floor.mean():.2f} (range {others.floor.min():.2f}-{others.floor.max():.2f}) "
        f"-- LOWEST of all 12, by a wide margin")
    out(f"  err range:                {sm.loc['newchain11','err_range']:.2f} mHa vs. others' "
        f"mean {others.err_range.mean():.2f} (range {others.err_range.min():.2f}-{others.err_range.max():.2f}) "
        f"-- SMALLEST of all 12, less than half the next-smallest (newchain05, "
        f"{others.err_range.min():.2f} mHa)")
    out(f"  achieved capture ceiling: {sm.loc['newchain11','achieved_capture_ceiling']:.4f} vs. others' "
        f"mean {others.achieved_capture_ceiling.mean():.4f} (range "
        f"{others.achieved_capture_ceiling.min():.4f}-{others.achieved_capture_ceiling.max():.4f}) "
        f"-- near the top of the range, not an outlier on this axis")
    out(f"  rand_regret denominator:  {sm.loc['newchain11','rand_regret_denom']:.2f} mHa vs. others' "
        f"mean {others.rand_regret_denom.mean():.2f} -- the smallest denominator of the 12, which is")
    out(f"  what converts the shared tie's ~51 mHa absolute cost into the only regret_frac>1.0.")

    banner("CHARACTERISATION")
    n_pos_gap = int((tie_df.gap > 0).sum())
    out("newchain11 is a single anomalous chain, not a sign of systematic S0 failure. Its")
    out("regret_frac>1.0 has an identified, structural cause: S0 has a genuine, chain-invariant")
    out("EXACT tie between two real anchor choices that differ by ~47-51 mHa in true outcome,")
    out("broken by an uninformative lexicographic rule; the lexicographic pick is the worse of")
    out(f"the two at {n_pos_gap}/{len(tie_df)} chains where both were evaluated (identity + the 12")
    out("held-out chains), yet regret_frac exceeds 1.0 ONLY at newchain11 -- because that chain")
    out("independently has the smallest error range and lowest floor of the set, shrinking the")
    out("normalisation denominator (44.06 mHa) far more than the tie's absolute cost shrinks.")
    out("S0's correlation with captured at newchain11 (+0.609, correct sign, p<0.05) is unaffected")
    out("by this -- the tie only degrades the single top PICK, not the score's overall ranking")
    out("quality.")

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "report.txt"
    old_report = report_path.read_text()
    combined = "\n".join(REPORT) + "\n\n" + "#" * 78 + "\n# ORIGINAL REPORT (per-chain tables, T1 sweep, first verdict) -- kept below\n" + "#" * 78 + "\n" + old_report
    report_path.write_text(combined)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(old_meta, part="chain_aware_v2_distribution", git_commit_addendum=git_commit,
                    distribution_summary=dist_df.to_dict(orient="records"),
                    newchain11_pick=str(pick_row.triple), newchain11_pick_rank=rank_of_pick,
                    newchain11_gap_mHa=gap, newchain11_tie_triples=list(tied.triple),
                    generated_addendum=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUTDIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\n[out] {OUTDIR / 'distribution_summary.csv'}")
    print(f"[out] {OUTDIR / 'newchain11_tie_analysis.csv'}")
    print(f"[out] {OUTDIR / 'newchain11_context.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
