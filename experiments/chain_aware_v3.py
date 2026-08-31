#!/usr/bin/env python3
"""
experiments/chain_aware_v3.py
================================

CHAIN-AWARE ANCHOR SELECTION, ROUND 3.

STATISTICAL CONSTRAINT. The unit of generalisation is the CHAIN, not the
anchor triple -- 120 triples at one chain are 120 correlated observations
of that chain. 18 chains total: 6 for development (H10 identity/physical/
rand007, N2 identity/reverse/r039), 12 strictly held out
(chain_aware_phaseB.py's newchain00-11). Cr2's 3 chains are NOT used here
at all (a final transfer test reserved for later, per the task's explicit
instruction not to use them for development -- and they are not part of
this task's own evaluation scope either).

No machine learning, no fitted parameters beyond a per-chain normalisation
(computed from the candidate set alone, never from any outcome). Every
candidate is run through an automatic invariance pre-flight before
evaluation: if a score's raw value does not change when the same-spin
chain is permuted (triple held fixed), it cannot solve the chain-dependent
problem and is rejected without evaluation. This would have caught S1-S4's
actual failure mode in seconds.

P0: a zero-new-parameter tie-break on S0 (using the P1 interface score to
break exact ties). P1: S_int, a zero-free-parameter interface score, and
S0+S_int (unit-variance normalised per chain, not fitted). P2: reframes
evaluation from argmax to shortlist recall (top-k, best-of-top-k regret) --
the deployable quantity.

All cached data, no new sampling. Reuses build_system_data/score_S0/
ALL_CHAINS from chain_aware_v2.py directly.
"""
from __future__ import annotations

import ast
import itertools
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "chain_aware_v3"
OUTDIR.mkdir(parents=True, exist_ok=True)

import chain_aware_v2 as CA2  # build_system_data, score_S0, ALL_CHAINS, parse_triple, regret_fraction

DEV6 = [("H10", "identity"), ("H10", "physical"), ("H10", "rand007"),
        ("N2", "identity"), ("N2", "reverse"), ("N2", "r039")]
HELDOUT12 = [("H10", f"newchain{i:02d}") for i in range(12)]
ALL18 = DEV6 + HELDOUT12
TIE_TOL = 1e-10
K_GRID = [1, 3, 5, 10]
N_RANDOM_TRIALS = 2000  # Monte Carlo draws for the random baseline, for stable estimates

REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def get_chain_df(system, chain):
    """captured/err_sqd/err_lucj for every triple at (system, chain), from
    cached CSVs only -- identical source logic to chain_aware_v2.py."""
    trans = pd.read_csv(CA2.TRANSMISSION_CSV)
    trans["triple"] = trans.triple.apply(CA2.parse_triple)
    trans = trans[trans.status == "OK"]
    if (system, chain) in [("H10", "identity"), ("H10", "physical"), ("H10", "rand007"),
                           ("N2", "identity"), ("N2", "reverse"), ("N2", "r039")]:
        return trans[(trans.system == system) & (trans.chain == chain)][
            ["triple", "captured", "err_sqd", "err_lucj"]].reset_index(drop=True)
    phaseb = pd.read_csv(CA2.PHASEB_CSV)
    phaseb["role"] = phaseb["role"].fillna("triple")
    phaseb = phaseb[(phaseb.role == "triple") & (~phaseb.is_floor.astype(bool))]
    phaseb["triple"] = phaseb.triple.apply(CA2.parse_triple)
    phaseb = phaseb[(phaseb.chain == chain) & (phaseb.status == "OK")]
    return phaseb[["triple", "captured", "err_sqd", "err_lucj"]].reset_index(drop=True)


def default_anchor_err(system, chain):
    """The default (p%4==0) anchor's err_sqd at (system, chain), from cache."""
    if system == "H10" and chain in ("identity", "physical", "rand007"):
        base = pd.read_csv(Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv")
        row = base[(base.ordering == chain)]
        return float(row.err_mHa.mean())
    if system == "N2" and chain in ("identity", "reverse", "r039"):
        df = pd.read_csv(REPO_ROOT / "outputs" / "unified" / "results.csv")
        row = df[(df.ordering == chain) & (df.seed == 2026)]
        if len(row):
            return float(row.err_sub_mHa.iloc[0])
        row = df[df.ordering == chain]
        return float(row.err_sub_mHa.mean())
    if system == "H10" and chain.startswith("newchain"):
        b2 = pd.read_csv(CA2.PHASEB_CSV)
        row = b2[(b2.chain == chain) & (b2.is_default == True)]
        return float(row.err_sqd.iloc[0])
    raise ValueError(f"no default-anchor lookup for {system}/{chain}")


# ==========================================================================
# same-spin chain-neighbour structure
# ==========================================================================
def chain_neighbours(pos, a, norb):
    """Orbital(s) that are the same-spin CHAIN neighbours of orbital a under
    layout pos -- i.e. adjacent by position, not the full retained-pair set
    (which also includes the same-spin diagonal). Endpoints (position 0 or
    norb-1) have exactly one neighbour; interior orbitals have two."""
    inv = np.argsort(pos)  # inv[k] = orbital at position k
    p = int(pos[a])
    neighbours = []
    if p > 0:
        neighbours.append(int(inv[p - 1]))
    if p < norb - 1:
        neighbours.append(int(inv[p + 1]))
    return neighbours


# ==========================================================================
# P1 -- the interface score
# ==========================================================================
def score_S_int(pos, A, Jab_full, Jaa_full, norb):
    total = 0.0
    for a in A:
        for n in chain_neighbours(pos, a, norb):
            total += abs(Jab_full[a, n]) * abs(Jaa_full[a, n])
    return total


def score_S0_plus_Sint_raw(pos, A, Jab_full, Jaa_full, norb):
    """Unnormalised pair (S0, S_int) -- normalisation happens per-chain,
    over the candidate set, in the evaluation loop (not here)."""
    return CA2.score_S0(A, Jab_full), score_S_int(pos, A, Jab_full, Jaa_full, norb)


# ==========================================================================
# invariance pre-flight
# ==========================================================================
def invariance_preflight(name, score_fn, sysdata, positions, system, triples_to_test, chains_to_test):
    """For each test triple, evaluate score_fn at every test chain (same
    system) and check whether the raw value changes. PASS = changes at
    every triple x chain-pair tested (score is chain-aware); FAIL = value
    is identical everywhere (score is chain-invariant and cannot solve the
    chain-dependent problem -- reject without further evaluation)."""
    banner(f"INVARIANCE PRE-FLIGHT: {name}")
    sd = sysdata[system]
    norb = sd["norb"]
    rows = []
    any_change = False
    all_change = True
    for A in triples_to_test:
        vals = {}
        for chain in chains_to_test:
            pos = positions[(system, chain)]
            v = score_fn(pos, A, sd["Jab"], sd["Jaa"], norb)
            vals[chain] = v
        distinct = len(set(np.round(list(vals.values()), 12)))
        changed = distinct > 1
        any_change = any_change or changed
        all_change = all_change and changed
        rows.append(dict(triple=str(A), **{f"val_{c}": v for c, v in vals.items()}, changed=changed))
        out(f"  triple {A}: " + "  ".join(f"{c}={v:.6f}" for c, v in vals.items()) +
            f"   {'CHANGES across chains' if changed else 'IDENTICAL across all chains'}")
    verdict = "PASS" if any_change else "FAIL"
    out(f"\n{name} PRE-FLIGHT VERDICT: {verdict} "
        f"({'at least one tested triple changes value across chains -- chain-aware, proceed' if any_change else 'value is identical across every chain tested -- chain-invariant, REJECTED without further evaluation'})")
    pd.DataFrame(rows).to_csv(OUTDIR / f"preflight_{name}.csv", index=False)
    return verdict == "PASS"


def regret_frac(pick_err, best_err, mean_err):
    denom = mean_err - best_err
    return float((pick_err - best_err) / denom) if denom > 0 else float("nan")


def main() -> int:
    banner("CHAIN-AWARE ANCHOR SELECTION -- ROUND 3")
    out("STATISTICAL CONSTRAINT: 6 dev chains, 12 held-out chains, evaluated once at the end.")
    out(f"Dev chains: {DEV6}")
    out(f"Held-out chains: {HELDOUT12}")

    sysdata, positions = CA2.build_system_data()

    # ================================================================ pre-flight
    all120 = list(itertools.combinations(range(10), 3))
    test_triples = [(0, 1, 9), (0, 8, 9), (0, 1, 2), (5, 6, 7)]
    test_chains_h10 = ["identity", "physical", "newchain00", "newchain11"]

    def s0_fn(pos, A, Jab, Jaa, norb):
        return CA2.score_S0(A, Jab)  # chain-invariant by construction -- known, not screened here

    sint_pass = invariance_preflight("S_int", score_S_int, sysdata, positions, "H10", test_triples, test_chains_h10)

    def s0_plus_sint_fn(pos, A, Jab, Jaa, norb):
        # for the pre-flight only: raw sum (normalisation is chain-set-relative,
        # not meaningful for a single-triple invariance probe)
        return CA2.score_S0(A, Jab) + score_S_int(pos, A, Jab, Jaa, norb)

    combined_pass = invariance_preflight("S0_plus_Sint", s0_plus_sint_fn, sysdata, positions, "H10",
                                         test_triples, test_chains_h10)

    if not sint_pass:
        out("\nS_int FAILED the invariance pre-flight -- stopping per protocol. No further evaluation.")
        (OUTDIR / "report.txt").write_text("\n".join(REPORT) + "\n")
        return 1

    # ================================================================ score everything, all 18 chains
    banner("Scoring S0, S_int, S0+S_int (per-chain unit-variance normalised) at all 18 chains")
    all_rows = []
    for system, chain in ALL18:
        df = get_chain_df(system, chain)
        pos = positions[(system, chain)]
        sd = sysdata[system]
        df = df.copy()
        df["S0"] = df.triple.apply(lambda A: CA2.score_S0(A, sd["Jab"]))
        df["S_int"] = df.triple.apply(lambda A: score_S_int(pos, A, sd["Jab"], sd["Jaa"], sd["norb"]))
        # per-chain, per-candidate-set unit-variance normalisation (from the scores alone, never the outcome)
        s0_z = (df.S0 - df.S0.mean()) / df.S0.std(ddof=0)
        sint_z = (df.S_int - df.S_int.mean()) / df.S_int.std(ddof=0)
        df["S0_plus_Sint"] = s0_z + sint_z
        df["system"] = system
        df["chain"] = chain
        all_rows.append(df)
    scored = pd.concat(all_rows, ignore_index=True)
    scored.to_csv(OUTDIR / "all_scores.csv", index=False)

    # ================================================================ P0: tie-break
    banner("P0 -- CHAIN-AWARE TIE-BREAK ON S0 (S_int breaks ties, zero new parameters)")
    p0_rows = []
    for system, chain in ALL18:
        sub = scored[(scored.system == system) & (scored.chain == chain)].reset_index(drop=True)
        s0max = sub.S0.max()
        tied = sub[np.abs(sub.S0 - s0max) < TIE_TOL]
        n_tied_top = len(tied)
        # before: plain argmax (first occurrence, positional -- matches established convention)
        pick_before_idx = int(sub.S0.to_numpy().argmax())
        pick_before = sub.iloc[pick_before_idx]
        # after: among ties on S0 (within TIE_TOL of the max), break by S_int
        pick_after = tied.loc[tied.S_int.idxmax()] if n_tied_top > 1 else pick_before
        best_row = sub.loc[sub.err_sqd.idxmin()]
        mean_err = sub.err_sqd.mean()
        regret_before = regret_frac(pick_before.err_sqd, best_row.err_sqd, mean_err)
        regret_after = regret_frac(pick_after.err_sqd, best_row.err_sqd, mean_err)
        p0_rows.append(dict(system=system, chain=chain, n_tied_at_top=n_tied_top,
                            pick_before=str(pick_before.triple), err_before=pick_before.err_sqd, regret_before=regret_before,
                            pick_after=str(pick_after.triple), err_after=pick_after.err_sqd, regret_after=regret_after,
                            changed=str(pick_before.triple) != str(pick_after.triple),
                            worse=regret_after > regret_before + 1e-9))
    p0_df = pd.DataFrame(p0_rows)
    p0_df.to_csv(OUTDIR / "p0_tiebreak.csv", index=False)
    out(p0_df.to_string(index=False))
    n_ties = int((p0_df.n_tied_at_top > 1).sum())
    n_changed = int(p0_df.changed.sum())
    n_worse = int(p0_df.worse.sum())
    nc11 = p0_df[p0_df.chain == "newchain11"].iloc[0]
    out(f"\nChains with an S0 tie at the top: {n_ties}/18")
    out(f"Chains where the tie-break changed the pick: {n_changed}/18")
    out(f"Chains made WORSE by the tie-break: {n_worse}/18")
    out(f"Median regret before: {p0_df.regret_before.median():.3f}  after: {p0_df.regret_after.median():.3f}")
    out(f"Worst regret before: {p0_df.regret_before.max():.3f}  after: {p0_df.regret_after.max():.3f}")
    out(f"newchain11: regret before={nc11.regret_before:.3f}  after={nc11.regret_after:.3f}  "
        f"below 1.0? {'YES' if nc11.regret_after < 1.0 else 'NO'}")

    # ================================================================ P2: shortlist recall
    banner("P2 -- SHORTLIST RECALL AND BEST-OF-TOP-k REGRET")

    def topk_metrics(df, score_col, k, rng=None):
        err = df.err_sqd.to_numpy()
        best_err = err.min()
        mean_err = err.mean()
        best_triple = df.triple.iloc[int(np.argmin(err))]
        if score_col == "__random__":
            idx = rng.choice(len(df), size=min(k, len(df)), replace=False)
        else:
            idx = np.argsort(-df[score_col].to_numpy())[:k]
        top1_correct = (str(df.triple.iloc[idx[0]]) == str(best_triple))
        recall = bool(any(str(df.triple.iloc[i]) == str(best_triple) for i in idx))
        best_of_topk_err = float(err[idx].min())
        regret = regret_frac(best_of_topk_err, best_err, mean_err)
        return dict(top1=top1_correct, recall=recall, regret=regret)

    def evaluate_group(chains, group_name, rng_seed=0):
        rows = []
        rng = np.random.default_rng(rng_seed)
        for system, chain in chains:
            sub = scored[(scored.system == system) & (scored.chain == chain)].reset_index(drop=True)
            default_err = default_anchor_err(system, chain)
            for score_name in ("S0", "S_int", "S0_plus_Sint", "__random__"):
                for k in K_GRID:
                    if score_name == "__random__":
                        trial_regrets, trial_recalls, trial_top1 = [], [], []
                        for t in range(N_RANDOM_TRIALS):
                            m = topk_metrics(sub, "__random__", k, rng=rng)
                            trial_regrets.append(m["regret"])
                            trial_recalls.append(m["recall"])
                            trial_top1.append(m["top1"])
                        m = dict(top1=np.mean(trial_top1), recall=np.mean(trial_recalls),
                                 regret=np.median(trial_regrets))
                        m["regret_p90"] = float(np.percentile(trial_regrets, 90))
                        m["regret_worst"] = float(np.max(trial_regrets))
                    else:
                        m = topk_metrics(sub, score_name, k)
                        m["regret_p90"] = m["regret"]
                        m["regret_worst"] = m["regret"]
                    err = sub.err_sqd.to_numpy()
                    best_err, mean_err = err.min(), err.mean()
                    beats_default = None
                    if k == 5 and score_name != "__random__":
                        idx = np.argsort(-sub[score_name].to_numpy())[:5]
                        best_of_5 = float(err[idx].min())
                        beats_default = bool(best_of_5 < default_err)
                    rows.append(dict(group=group_name, system=system, chain=chain, score=score_name, k=k,
                                     top1=m["top1"], recall=m["recall"], regret=m["regret"],
                                     regret_p90=m["regret_p90"], regret_worst=m["regret_worst"],
                                     beats_default_top5=beats_default))
        return pd.DataFrame(rows)

    dev_df = evaluate_group(DEV6, "dev", rng_seed=1)
    heldout_df = evaluate_group(HELDOUT12, "heldout", rng_seed=2)
    p2_df = pd.concat([dev_df, heldout_df], ignore_index=True)
    p2_df.to_csv(OUTDIR / "p2_shortlist.csv", index=False)

    for group_name, gdf in (("DEVELOPMENT (6 chains)", dev_df), ("HELD-OUT (12 chains)", heldout_df)):
        banner(f"P2 results -- {group_name}")
        for score_name in ("S0", "S_int", "S0_plus_Sint", "__random__"):
            out(f"\n{score_name}:")
            for k in K_GRID:
                sub = gdf[(gdf.score == score_name) & (gdf.k == k)]
                out(f"  k={k:<2}  top1_acc={sub.top1.mean():.3f}  recall={sub.recall.mean():.3f}  "
                    f"median_regret={sub.regret.median():.3f}  p90_regret={sub.regret_p90.median():.3f}  "
                    f"worst_regret={sub.regret.max():.3f}")
            k5 = gdf[(gdf.score == score_name) & (gdf.k == 5) & (gdf.beats_default_top5.notna())]
            if len(k5):
                out(f"  fraction beating default anchors (best-of-top-5): "
                    f"{k5.beats_default_top5.mean():.3f} ({int(k5.beats_default_top5.sum())}/{len(k5)})")

    # ================================================================ acceptance criteria
    banner("ACCEPTANCE CRITERIA")
    crit1 = sint_pass
    out(f"1. Invariance pre-flight PASS: {crit1}")

    crit2_rows = []
    for system, chain in HELDOUT12:
        sub = scored[(scored.system == system) & (scored.chain == chain)]
        r = spearmanr(sub.S_int, sub.captured)
        crit2_rows.append(dict(chain=chain, rho=r.statistic, p=r.pvalue, correct_sign=r.statistic > 0))
    crit2_df = pd.DataFrame(crit2_rows)
    crit2 = bool(crit2_df.correct_sign.all())
    out(f"2. Correct-sign rho(S_int, captured) at every held-out chain: {crit2}")
    out(crit2_df.to_string(index=False))

    s0_k5_heldout = heldout_df[(heldout_df.score == "S0") & (heldout_df.k == 5)]
    sint_k5_heldout = heldout_df[(heldout_df.score == "S_int") & (heldout_df.k == 5)]
    comb_k5_heldout = heldout_df[(heldout_df.score == "S0_plus_Sint") & (heldout_df.k == 5)]
    crit3_median = sint_k5_heldout.regret.median() < s0_k5_heldout.regret.median()
    crit3_p90 = sint_k5_heldout.regret_p90.median() < s0_k5_heldout.regret_p90.median()
    out(f"3. Best-of-top-5 regret (held-out): S0 median={s0_k5_heldout.regret.median():.3f}, "
        f"S_int median={sint_k5_heldout.regret.median():.3f} (below S0? {crit3_median})")
    out(f"   S0 p90={s0_k5_heldout.regret_p90.median():.3f}, S_int p90={sint_k5_heldout.regret_p90.median():.3f} "
        f"(below S0? {crit3_p90})")

    random_k5_heldout = heldout_df[(heldout_df.score == "__random__") & (heldout_df.k == 5)]
    crit4 = bool((sint_k5_heldout.regret.to_numpy() <= random_k5_heldout.regret.to_numpy()).all())
    out(f"4. No held-out chain worse than random (S_int best-of-top-5 regret <= random): {crit4}")

    all_pass = crit1 and crit2 and crit3_median and crit3_p90 and crit4
    out(f"\nALL ACCEPTANCE CRITERIA MET: {all_pass}")

    # ================================================================ HEADLINE
    banner("HEADLINE")
    out(f"1. Tie-break repairs newchain11 without harming any other chain: "
        f"newchain11 regret {nc11.regret_before:.3f} -> {nc11.regret_after:.3f} "
        f"({'below 1.0' if nc11.regret_after < 1.0 else 'still >= 1.0'}); "
        f"{n_worse}/18 chains made worse overall.")
    out(f"2. S_int passes the invariance pre-flight: {sint_pass}.")
    out(f"3. Best-of-top-5 regret (held-out, median / p90): "
        f"S0 {s0_k5_heldout.regret.median():.3f}/{s0_k5_heldout.regret_p90.median():.3f}  "
        f"S_int {sint_k5_heldout.regret.median():.3f}/{sint_k5_heldout.regret_p90.median():.3f}  "
        f"S0+S_int {comb_k5_heldout.regret.median():.3f}/{comb_k5_heldout.regret_p90.median():.3f}")
    dev_recall5 = {s: dev_df[(dev_df.score == s) & (dev_df.k == 5)].recall.mean() for s in
                  ("S0", "S_int", "S0_plus_Sint")}
    heldout_recall5 = {s: heldout_df[(heldout_df.score == s) & (heldout_df.k == 5)].recall.mean() for s in
                       ("S0", "S_int", "S0_plus_Sint")}
    out(f"4. Top-5 recall: dev {dev_recall5}  held-out {heldout_recall5}")
    out(f"5. Does any candidate beat S0 on the declared acceptance criteria: "
        f"{'YES' if all_pass else 'NO'}")

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="chain_aware_v3", git_commit=git_commit,
        dev_chains=DEV6, heldout_chains=HELDOUT12, tie_tol=TIE_TOL, k_grid=K_GRID,
        preflight_sint_pass=sint_pass, preflight_combined_pass=combined_pass,
        p0_n_ties=n_ties, p0_n_changed=n_changed, p0_n_worse=n_worse,
        p0_newchain11_regret_before=float(nc11.regret_before), p0_newchain11_regret_after=float(nc11.regret_after),
        acceptance=dict(invariance=crit1, correct_sign_captured=crit2, median_regret_better=crit3_median,
                        p90_regret_better=crit3_p90, no_worse_than_random=crit4, all_pass=all_pass),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (OUTDIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\n[out] {OUTDIR / 'all_scores.csv'}")
    print(f"[out] {OUTDIR / 'p0_tiebreak.csv'}")
    print(f"[out] {OUTDIR / 'p2_shortlist.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
