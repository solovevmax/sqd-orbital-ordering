#!/usr/bin/env python3
"""
experiments/chain_aware.py
=============================

PHASE A -- construct and screen a chain-aware anchor score against the
chain-invariant control S0 = sum_{p in A} |Jab[p,p]|. No sampling: reuses
experiments/outputs/transmission/all_evaluations.csv (err_lucj and err_sqd
for all 400 (chain, triple) rows across H10 identity/physical/rand007 and
N2 identity/reverse/r039, already cross-validated 400/400 exact against the
sbd cache).

Note on S0 vs the previously-reported retained_J_oppspin: mask.py's
retained_J_split normalises sum((J_ab*mask)**2)/sum(J_ab**2) -- a SUM OF
SQUARES. The score family declared here uses sum_{p in A} |Jab[p,p]| -- a
SUM OF ABSOLUTE VALUES. These are different monotonic transforms of the
anchor set and are not guaranteed to be rank-identical, only highly
correlated in practice (both are strictly increasing in each |Jab[p,p]|
with no cross terms). S0 is computed fresh from the Jab diagonal here, not
copied from the cached retained_J_oppspin column.

R_d(p) note: "restricted to retained (nearest-neighbour) pairs along the
connecting path" -- since same_spin_pairs(pos) retains exactly the
consecutive-position pairs (a simple path graph over positions 0..norb-1),
reachability within d hops of p via retained edges is exactly the set of
orbitals q with |pos[p]-pos[q]| <= d. Implemented directly as a positional
distance cutoff, which is exact for this mask (no separate graph-search
needed).
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "chain_aware"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRANSMISSION_CSV = Path(__file__).resolve().parent / "outputs" / "transmission" / "all_evaluations.csv"
H10_BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"
H10_CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"

D_GRID = [1, 2, 3]
LAMBDA_GRID = [0.25, 0.5, 1.0]
DEV_CHAINS = [("H10", "identity"), ("N2", "identity")]
ALL_CHAINS = [("H10", "identity"), ("H10", "physical"), ("H10", "rand007"),
              ("N2", "identity"), ("N2", "reverse"), ("N2", "r039")]
SIG_RHO, SIG_P = 0.3, 0.05

REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_triple(s):
    if isinstance(s, tuple):
        return tuple(int(x) for x in s)
    s = str(s).strip()
    if s.startswith("("):
        return tuple(int(x) for x in ast.literal_eval(s))
    return tuple(int(c) for c in s.zfill(3))


# ==========================================================================
# score family -- PRE-DECLARED, CLOSED. Do not add variants here.
# ==========================================================================
def _R_d(p, pos, Jaa, d, weight=None):
    posp = int(pos[p])
    total = 0.0
    for q in range(len(pos)):
        if q == p:
            continue
        if abs(int(pos[q]) - posp) <= d:
            w = weight[q] if weight is not None else 1.0
            total += abs(Jaa[p, q]) * w
    return total


def score_S0(A, Jab):
    return sum(abs(Jab[p, p]) for p in A)


def score_S1(A, Jab, pos, Jaa, d):
    return sum(abs(Jab[p, p]) * _R_d(p, pos, Jaa, d) for p in A)


def score_S2(A, Jab, pos, d, lam):
    s0 = score_S0(A, Jab)
    pen = sum(math.exp(-abs(int(pos[p]) - int(pos[q])) / d)
              for p, q in itertools.combinations(A, 2))
    return s0 - lam * pen


def score_S3(A, Jab, pos, Jaa, A_os_site, d):
    return sum(abs(Jab[p, p]) * _R_d(p, pos, Jaa, d, weight=A_os_site) for p in A)


def score_S4(A, Jab, pos, Jaa, d, lam):
    return score_S1(A, Jab, pos, Jaa, d) * score_S2(A, Jab, pos, d, lam)


SCORE_NAMES = ["S0", "S1", "S2", "S3", "S4"]


def compute_all_scores(A, Jab, pos, Jaa, A_os_site, d, lam):
    return dict(
        S0=score_S0(A, Jab),
        S1=score_S1(A, Jab, pos, Jaa, d),
        S2=score_S2(A, Jab, pos, d, lam),
        S3=score_S3(A, Jab, pos, Jaa, A_os_site, d),
        S4=score_S4(A, Jab, pos, Jaa, d, lam),
    )


# ==========================================================================
# system / chain setup
# ==========================================================================
def build_system_data():
    import run_ordering_pipeline as R
    import unified_run as U

    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(H10_CACHEDIR))
    norb_h, nocc_h = ref["norb"], ref["nocc"]
    t1_h, t2_h = ref["t1L"], ref["t2L"]
    Jaa_h, Jab_h = R.diag_coulomb(R.build_ucj(t2_h, t1_h))
    # diag_coulomb_mats is (n_reps, norb, norb); "Jaa[p,q] ... summed over
    # repetitions" (per the task spec) -> collapse via abs-sum over reps,
    # matching score1()'s established M_ss/M_os convention exactly.
    Jaa_h = np.abs(Jaa_h).sum(axis=0)
    Jab_h = np.abs(Jab_h).sum(axis=0)
    amp_h = R.Amplitudes(t1=np.asarray(t1_h), t2=np.asarray(t2_h), nocc=nocc_h, norb=norb_h)

    norb_n, nocc_n = U.NORB, U.NELEC[0]
    t1_n, t2_n = U.ref_data["t1"], U.ref_data["t2"]
    Jaa_n, Jab_n = R.diag_coulomb(R.build_ucj(t2_n, t1_n))
    Jaa_n = np.abs(Jaa_n).sum(axis=0)
    Jab_n = np.abs(Jab_n).sum(axis=0)
    amp_n = R.Amplitudes(t1=np.asarray(t1_n), t2=np.asarray(t2_n), nocc=nocc_n, norb=norb_n)

    base = pd.read_csv(H10_BASELINE_CSV)
    perm_by_ordering = base.groupby("ordering")["permutation"].first()
    norb = 10
    pos = {
        ("H10", "identity"): R.positions_from(np.arange(norb)),
        ("H10", "physical"): R.positions_from(R.parse_permutation(perm_by_ordering["physical"], norb)),
        ("H10", "rand007"): R.positions_from(R.parse_permutation(perm_by_ordering["rand007"], norb)),
        ("N2", "identity"): R.positions_from(np.arange(norb)),
        ("N2", "reverse"): R.positions_from(np.arange(norb)[::-1]),
        ("N2", "r039"): R.positions_from(R.parse_permutation("0914723658", norb)),
    }
    sysdata = {
        "H10": dict(Jaa=Jaa_h, Jab=Jab_h, A_os_site=amp_h.A_os_site),
        "N2": dict(Jaa=Jaa_n, Jab=Jab_n, A_os_site=amp_n.A_os_site),
    }
    return sysdata, pos


def regret_fraction(scores: np.ndarray, err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    rand_regret = err.mean() - err.min()
    pick = int(np.asarray(scores).argmax())
    regret = err[pick] - err.min()
    return float(regret / rand_regret) if rand_regret > 0 else float("nan")


def main() -> int:
    banner("CHAIN-AWARE ANCHOR SCORE -- PHASE A (construct and screen, no sampling)")

    if not TRANSMISSION_CSV.exists():
        sys.exit(f"FATAL: {TRANSMISSION_CSV} not found -- Phase A depends on the "
                  f"cross-validated transmission dataset.")
    df = pd.read_csv(TRANSMISSION_CSV)
    df["triple"] = df.triple.apply(parse_triple)
    df = df[df.status == "OK"].reset_index(drop=True)
    out(f"Loaded {len(df)} cross-validated (chain, triple) rows from {TRANSMISSION_CSV}")
    out(f"  sha256={sha256_of(TRANSMISSION_CSV)[:16]}")

    sysdata, positions = build_system_data()

    # ---------------------------------------------------------- hyperparameter sweep
    banner("Hyperparameter sweep: d in {1,2,3} x lambda in {0.25,0.5,1.0}, "
           "selection = mean |rho(S,err_lucj)| over {S1,S2,S3,S4} x {H10/identity, N2/identity} ONLY")
    dev_data = {}
    for sysname, chain in DEV_CHAINS:
        sub = df[(df.system == sysname) & (df.chain == chain)].copy()
        dev_data[(sysname, chain)] = sub

    sweep_rows = []
    for d in D_GRID:
        for lam in LAMBDA_GRID:
            abs_rhos = []
            for (sysname, chain), sub in dev_data.items():
                sd = sysdata[sysname]
                pos = positions[(sysname, chain)]
                s1 = sub.triple.apply(lambda A: score_S1(A, sd["Jab"], pos, sd["Jaa"], d))
                s2 = sub.triple.apply(lambda A: score_S2(A, sd["Jab"], pos, d, lam))
                s3 = sub.triple.apply(lambda A: score_S3(A, sd["Jab"], pos, sd["Jaa"], sd["A_os_site"], d))
                s4 = s1 * s2
                for s in (s1, s2, s3, s4):
                    r = spearmanr(s, sub.err_lucj)
                    abs_rhos.append(abs(r.statistic))
            sweep_rows.append(dict(d=d, lam=lam, mean_abs_rho=float(np.mean(abs_rhos))))
    sweep_df = pd.DataFrame(sweep_rows).sort_values("mean_abs_rho", ascending=False).reset_index(drop=True)
    sweep_df.to_csv(OUTDIR / "phaseA_hyperparameter_sweep.csv", index=False)
    out(sweep_df.to_string(index=False))
    D_FROZEN = int(sweep_df.iloc[0]["d"])
    LAM_FROZEN = float(sweep_df.iloc[0]["lam"])
    out(f"\nFROZEN: d={D_FROZEN}, lambda={LAM_FROZEN}  (mean |rho|={sweep_df.iloc[0]['mean_abs_rho']:.4f})")
    out("This (d, lambda) pair is now fixed for every remaining computation in this experiment "
        "(Phase A table below, and Phase B if it proceeds). Not revisited.")

    # ---------------------------------------------------------- score all rows at frozen params
    banner(f"Scoring all {len(df)} rows at frozen d={D_FROZEN}, lambda={LAM_FROZEN}")
    for name in SCORE_NAMES:
        df[name] = np.nan
    for (sysname, chain), pos in positions.items():
        sd = sysdata[sysname]
        mask = (df.system == sysname) & (df.chain == chain)
        for idx in df[mask].index:
            A = df.at[idx, "triple"]
            scores = compute_all_scores(A, sd["Jab"], pos, sd["Jaa"], sd["A_os_site"], D_FROZEN, LAM_FROZEN)
            for name, val in scores.items():
                df.at[idx, name] = val
    df.to_csv(OUTDIR / "phaseA_all_scores.csv", index=False)

    # ---------------------------------------------------------- per-chain, per-score table
    banner("Per-chain, per-score: rho(S, err_lucj), rho(S, err_sqd), normalised selection regret "
           "(against err_sqd, argmax(S) vs true best, over the same triple set already evaluated "
           "at that chain)")
    table_rows = []
    for name in SCORE_NAMES:
        for sysname, chain in ALL_CHAINS:
            sub = df[(df.system == sysname) & (df.chain == chain)]
            r_lucj = spearmanr(sub[name], sub.err_lucj)
            r_sqd = spearmanr(sub[name], sub.err_sqd)
            regret = regret_fraction(sub[name].to_numpy(), sub.err_sqd.to_numpy())
            table_rows.append(dict(score=name, system=sysname, chain=chain, n=len(sub),
                                    rho_lucj=r_lucj.statistic, p_lucj=r_lucj.pvalue,
                                    rho_sqd=r_sqd.statistic, p_sqd=r_sqd.pvalue,
                                    regret_frac=regret))
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUTDIR / "phaseA_score_comparison.csv", index=False)

    for name in SCORE_NAMES:
        sub = table_df[table_df.score == name]
        out(f"\n{name}:")
        for _, row in sub.iterrows():
            out(f"  {row.system}/{row.chain:<10} n={row.n:<4.0f}  "
                f"rho(.,err_lucj)={row.rho_lucj:+.3f} (p={row.p_lucj:.2e})   "
                f"rho(.,err_sqd)={row.rho_sqd:+.3f} (p={row.p_sqd:.2e})   "
                f"regret_frac={row.regret_frac:.3f}")
        worst_rho_lucj = sub.loc[sub.rho_lucj.idxmax()]   # least-negative = worst (all expected negative)
        worst_rho_sqd = sub.loc[sub.rho_sqd.idxmax()]
        worst_regret = sub.loc[sub.regret_frac.idxmax()]
        out(f"  WORST-CASE rho(.,err_lucj): {worst_rho_lucj.rho_lucj:+.3f} at {worst_rho_lucj.system}/{worst_rho_lucj.chain}")
        out(f"  WORST-CASE rho(.,err_sqd):  {worst_rho_sqd.rho_sqd:+.3f} at {worst_rho_sqd.system}/{worst_rho_sqd.chain}")
        out(f"  WORST-CASE regret_frac:     {worst_regret.regret_frac:.3f} at {worst_regret.system}/{worst_regret.chain}")

    # ---------------------------------------------------------- decisive comparison
    banner("DECISIVE COMPARISON -- worst-case rho(err_sqd) and worst-case regret, S0 first")
    summary_rows = []
    for name in SCORE_NAMES:
        sub = table_df[table_df.score == name]
        summary_rows.append(dict(
            score=name,
            worst_rho_sqd=float(sub.rho_sqd.max()),
            worst_rho_lucj=float(sub.rho_lucj.max()),
            worst_regret=float(sub.regret_frac.max()),
            mean_rho_sqd=float(sub.rho_sqd.mean()),
        ))
    summary_df = pd.DataFrame(summary_rows)
    out(summary_df.to_string(index=False))

    s0_worst = summary_df[summary_df.score == "S0"].iloc[0]
    challengers = summary_df[summary_df.score != "S0"].copy()
    challengers["beats_S0_worst_case"] = challengers.worst_rho_sqd < s0_worst.worst_rho_sqd  # more negative = better
    out(f"\nS0 worst-case rho(err_sqd) = {s0_worst.worst_rho_sqd:+.3f}")
    for _, row in challengers.iterrows():
        out(f"  {row.score} worst-case rho(err_sqd) = {row.worst_rho_sqd:+.3f}  "
            f"{'BEATS S0' if row.beats_S0_worst_case else 'does not beat S0'}")

    winners = challengers[challengers.beats_S0_worst_case].sort_values("worst_rho_sqd")
    if len(winners) > 0:
        s_best = winners.iloc[0]["score"]
        out(f"\nBest chain-aware score on worst-case rho(err_sqd): {s_best} "
            f"(worst-case rho={winners.iloc[0]['worst_rho_sqd']:+.3f} vs S0's {s0_worst.worst_rho_sqd:+.3f})")
        phase_b_score = s_best
    else:
        out(f"\nNo chain-aware score (S1-S4) beats S0 on worst-case rho(err_sqd). "
            f"Per protocol: Phase B proceeds with S0 and reports this as a negative result.")
        phase_b_score = "S0"

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "phaseA_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="chain_aware_phaseA", git_commit=git_commit,
        transmission_csv_sha256=sha256_of(TRANSMISSION_CSV),
        d_grid=D_GRID, lambda_grid=LAMBDA_GRID, d_frozen=D_FROZEN, lambda_frozen=LAM_FROZEN,
        sweep=sweep_df.to_dict(orient="records"),
        summary=summary_df.to_dict(orient="records"),
        phase_b_score=phase_b_score,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "phaseA_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"\n[out] {OUTDIR / 'phaseA_all_scores.csv'}")
    print(f"[out] {OUTDIR / 'phaseA_score_comparison.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'phaseA_metadata.json'}")
    print(f"\n[PHASE B] frozen score = {phase_b_score}, d={D_FROZEN}, lambda={LAM_FROZEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
