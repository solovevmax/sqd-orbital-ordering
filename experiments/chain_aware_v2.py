#!/usr/bin/env python3
"""
experiments/chain_aware_v2.py
================================

CHAIN-AWARE SCORE, second attempt -- target CAPTURE, not variational energy.

Established (chain_aware.py Phase A/B, transmission.py): S0 predicts
err_lucj strongly and consistently (rho -0.85 to -0.97 across every system
tested) -- that link is not broken. The broken link is err_lucj -> captured
(link 1 in transmission.py), which fails at some chains. A useful score
must therefore predict CAPTURE directly.

Two pre-declared, closed scores. No new sampling -- captured is already
cached for every (chain, triple) evaluated across the whole project.

T1 "anchor-conditioned reachability": for the top-K |t2| double-excitation
channels (i,j->a,b), a channel counts if BOTH excitation legs are within L
chain-positions (i.e. |pos[i]-pos[a]|<=L and |pos[j]-pos[b]|<=L -- purely
positional, testing whether "effective reach" extends beyond the mask's
literal L=1 nearest-neighbour retention) AND at least one of {i,j,a,b} is
within D chain-positions of an anchor orbital. Score = amplitude-weighted
fraction of the top-K channels satisfying all three. Declared sweep:
K in {20,50}, L in {2,3,4}, D in {0,1} -- 12 combinations, selected by mean
|rho(T1,captured)| on H10-identity and N2-identity ONLY, then frozen.

T2 "perturbative support overlap": a first-order-in-retained-J estimate of
the masked LUCJ state's amplitude on each double excitation,
    c_pred[i,j,a,b] = t2[i,j,a,b] * (Jaa*Maa)[i,a] * (Jaa*Maa)[j,b]
                       * sum_{p in {i,j,a,b}} (Jab*Mab)[p,p]
-- product of the true CCSD seed amplitude with the retained (masked)
same-spin coupling along each excitation leg, times the retained
opposite-spin (anchor) coupling at the channel's own orbitals. This uses
BOTH retained sectors, unlike T1's positional-only construction. Top-k
(k = the determinant budget, an already-fixed external quantity, not a
new free parameter) entries by |c_pred|^2 are taken; score = the TRUE CCSD
|t2|^2 weight those determinants carry, divided by total |t2|^2 weight --
a polynomial surrogate for captured itself.

Development: H10-identity and N2-identity ONLY (T1's sweep). Evaluation:
every chain with cached captured data -- H10 identity/physical/rand007 +
the 12 held-out chains (chain_aware_phaseB.py), N2 identity/reverse/r039 --
18 chains total. S0 reported first for direct comparison.
"""
from __future__ import annotations

import ast
import hashlib
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

OUTDIR = Path(__file__).resolve().parent / "outputs" / "chain_aware_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRANSMISSION_CSV = Path(__file__).resolve().parent / "outputs" / "transmission" / "all_evaluations.csv"
PHASEB_CSV = Path(__file__).resolve().parent / "outputs" / "chain_aware" / "phaseB_b2_all.csv"
PHASEB_META = Path(__file__).resolve().parent / "outputs" / "chain_aware" / "phaseB_metadata.json"
H10_BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"
H10_CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"

K_GRID = [20, 50]
L_GRID = [2, 3, 4]
D_GRID = [0, 1]
BUDGET = 15  # SQD determinant budget used throughout H10/N2 -- T2's k, not a new parameter
DEV_CHAINS = [("H10", "identity"), ("N2", "identity")]
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


def regret_fraction(scores: np.ndarray, err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    rand_regret = err.mean() - err.min()
    pick = int(np.asarray(scores).argmax())
    regret = err[pick] - err.min()
    return float(regret / rand_regret) if rand_regret > 0 else float("nan")


# ==========================================================================
# system / chain setup
# ==========================================================================
def build_system_data():
    import run_ordering_pipeline as R
    import unified_run as U

    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(H10_CACHEDIR))
    norb_h, nocc_h = ref["norb"], ref["nocc"]
    t1_h, t2_h = np.asarray(ref["t1L"]), np.asarray(ref["t2L"])
    Jaa_h, Jab_h = R.diag_coulomb(R.build_ucj(t2_h, t1_h))
    Jaa_h = np.abs(Jaa_h).sum(axis=0)
    Jab_h = np.abs(Jab_h).sum(axis=0)

    norb_n, nocc_n = U.NORB, U.NELEC[0]
    t1_n, t2_n = np.asarray(U.ref_data["t1"]), np.asarray(U.ref_data["t2"])
    Jaa_n, Jab_n = R.diag_coulomb(R.build_ucj(t2_n, t1_n))
    Jaa_n = np.abs(Jaa_n).sum(axis=0)
    Jab_n = np.abs(Jab_n).sum(axis=0)

    base = pd.read_csv(H10_BASELINE_CSV, dtype={"permutation": str})
    perm_by_ordering = base.groupby("ordering")["permutation"].first()
    phaseB_meta = json.loads(PHASEB_META.read_text())
    norb = 10
    positions = {
        ("H10", "identity"): R.positions_from(np.arange(norb)),
        ("H10", "physical"): R.positions_from(R.parse_permutation(perm_by_ordering["physical"], norb)),
        ("H10", "rand007"): R.positions_from(R.parse_permutation(perm_by_ordering["rand007"], norb)),
        ("N2", "identity"): R.positions_from(np.arange(norb)),
        ("N2", "reverse"): R.positions_from(np.arange(norb)[::-1]),
        ("N2", "r039"): R.positions_from(R.parse_permutation("0914723658", norb)),
    }
    for name, permstr in phaseB_meta["new_chains"].items():
        positions[("H10", name)] = R.positions_from(R.parse_permutation(permstr, norb))

    sysdata = {
        "H10": dict(t1=t1_h, t2=t2_h, norb=norb_h, nocc=nocc_h, Jaa=Jaa_h, Jab=Jab_h),
        "N2": dict(t1=t1_n, t2=t2_n, norb=norb_n, nocc=nocc_n, Jaa=Jaa_n, Jab=Jab_n),
    }
    return sysdata, positions


ALL_CHAINS = [("H10", "identity"), ("H10", "physical"), ("H10", "rand007")] + \
             [("H10", f"newchain{i:02d}") for i in range(12)] + \
             [("N2", "identity"), ("N2", "reverse"), ("N2", "r039")]


# ==========================================================================
# S0 -- chain-invariant control (same construction as chain_aware.py)
# ==========================================================================
def score_S0(A, Jab):
    return sum(abs(Jab[p, p]) for p in A)


# ==========================================================================
# T1 -- anchor-conditioned reachability
# ==========================================================================
def top_k_channels(t2, k):
    """Top-k (i,j,a,b) [a,b local virtual index] flattened entries of t2 by
    |t2|, as (weight, i, j, a, b) tuples, descending."""
    nocc = t2.shape[0]
    flat = np.abs(t2).ravel()
    idx = np.argsort(flat)[::-1][:k]
    out_ = []
    shp = t2.shape
    for lin in idx:
        i, j, a, b = np.unravel_index(lin, shp)
        out_.append((float(abs(t2[i, j, a, b])), int(i), int(j), int(a), int(b)))
    return out_


def score_T1(pos, A, channels, nocc, L, D):
    """channels: list of (weight, i, j, a, b) [a,b LOCAL virtual index]."""
    num, den = 0.0, 0.0
    for w, i, j, a, b in channels:
        ga, gb = nocc + a, nocc + b
        den += w
        alpha_reach = abs(int(pos[i]) - int(pos[ga])) <= L
        beta_reach = abs(int(pos[j]) - int(pos[gb])) <= L
        anchor_ok = False
        for orb in (i, j, ga, gb):
            for p in A:
                if abs(int(pos[orb]) - int(pos[p])) <= D:
                    anchor_ok = True
                    break
            if anchor_ok:
                break
        if alpha_reach and beta_reach and anchor_ok:
            num += w
    return num / den if den > 0 else 0.0


# ==========================================================================
# T2 -- perturbative support overlap
# ==========================================================================
def score_T2(pos, A, t2, Jaa, Jab, nocc, norb, budget):
    import sqd_ordering.mask as mask
    m_aa, m_ab = mask.mask_matrices(pos, norb, anchor_orbitals=A)
    Jaa_masked = Jaa * m_aa
    Jab_diag_masked = np.diagonal(Jab * m_ab).copy()  # nonzero only at anchor orbitals

    nvir = norb - nocc
    t2 = np.asarray(t2)
    c_pred = np.zeros_like(t2)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    ga, gb = nocc + a, nocc + b
                    leg1 = Jaa_masked[i, ga]
                    leg2 = Jaa_masked[j, gb]
                    if leg1 == 0.0 or leg2 == 0.0:
                        continue
                    anchor_w = Jab_diag_masked[i] + Jab_diag_masked[j] + Jab_diag_masked[ga] + Jab_diag_masked[gb]
                    c_pred[i, j, a, b] = t2[i, j, a, b] * leg1 * leg2 * anchor_w

    flat_c = np.abs(c_pred).ravel() ** 2
    flat_t2sq = (np.abs(t2).ravel()) ** 2
    total = flat_t2sq.sum()
    if total <= 0:
        return 0.0
    top_idx = np.argsort(flat_c)[::-1][:budget]
    return float(flat_t2sq[top_idx].sum() / total)


def main() -> int:
    banner("CHAIN-AWARE SCORE v2 -- targeting CAPTURE, not variational energy")

    if not TRANSMISSION_CSV.exists() or not PHASEB_CSV.exists():
        sys.exit("FATAL: required cached CSVs not found.")

    trans = pd.read_csv(TRANSMISSION_CSV)
    trans["triple"] = trans.triple.apply(parse_triple)
    trans = trans[trans.status == "OK"]

    phaseb = pd.read_csv(PHASEB_CSV)
    phaseb["role"] = phaseb["role"].fillna("triple")
    phaseb = phaseb[(phaseb.role == "triple") & (~phaseb.is_floor.astype(bool))]
    phaseb["triple"] = phaseb.triple.apply(parse_triple)
    phaseb["system"] = "H10"
    phaseb = phaseb[phaseb.status == "OK"]

    def get_chain_df(system, chain):
        if (system, chain) in [("H10", "identity"), ("H10", "physical"), ("H10", "rand007"),
                               ("N2", "identity"), ("N2", "reverse"), ("N2", "r039")]:
            return trans[(trans.system == system) & (trans.chain == chain)][
                ["triple", "captured", "err_sqd"]].reset_index(drop=True)
        else:
            return phaseb[(phaseb.system == system) & (phaseb.chain == chain)][
                ["triple", "captured", "err_sqd"]].reset_index(drop=True)

    out(f"Loaded {len(trans)} transmission rows + {len(phaseb)} phaseB rows "
        f"(sha256 transmission={sha256_of(TRANSMISSION_CSV)[:12]}, phaseB={sha256_of(PHASEB_CSV)[:12]})")
    out(f"Chains to evaluate: {len(ALL_CHAINS)}")

    sysdata, positions = build_system_data()

    # ---------------------------------------------------------- T1 hyperparameter sweep
    banner("T1 hyperparameter sweep: K in {20,50} x L in {2,3,4} x D in {0,1}, "
           "selection = mean |rho(T1,captured)| on H10/identity + N2/identity ONLY")
    dev_data = {}
    dev_channels = {}
    for sysname, chain in DEV_CHAINS:
        df = get_chain_df(sysname, chain)
        dev_data[(sysname, chain)] = df
        sd = sysdata[sysname]
        dev_channels[sysname] = {K: top_k_channels(sd["t2"], K) for K in K_GRID}

    sweep_rows = []
    for K in K_GRID:
        for L in L_GRID:
            for D in D_GRID:
                abs_rhos = []
                for sysname, chain in DEV_CHAINS:
                    df = dev_data[(sysname, chain)]
                    pos = positions[(sysname, chain)]
                    nocc = sysdata[sysname]["nocc"]
                    channels = dev_channels[sysname][K]
                    scores = df.triple.apply(lambda A: score_T1(pos, A, channels, nocc, L, D))
                    r = spearmanr(scores, df.captured)
                    abs_rhos.append(abs(r.statistic) if not np.isnan(r.statistic) else 0.0)
                sweep_rows.append(dict(K=K, L=L, D=D, mean_abs_rho=float(np.mean(abs_rhos))))
    sweep_df = pd.DataFrame(sweep_rows).sort_values("mean_abs_rho", ascending=False).reset_index(drop=True)
    sweep_df.to_csv(OUTDIR / "t1_sweep.csv", index=False)
    out(sweep_df.to_string(index=False))
    K_FROZEN, L_FROZEN, D_FROZEN = int(sweep_df.iloc[0].K), int(sweep_df.iloc[0].L), int(sweep_df.iloc[0].D)
    out(f"\nFROZEN: K={K_FROZEN}, L={L_FROZEN}, D={D_FROZEN}  (mean |rho|={sweep_df.iloc[0].mean_abs_rho:.4f})")

    # ---------------------------------------------------------- score all chains
    banner(f"Scoring S0, T1(K={K_FROZEN},L={L_FROZEN},D={D_FROZEN}), T2(k={BUDGET}) "
           f"at all {len(ALL_CHAINS)} chains")
    channels_frozen = {sysname: top_k_channels(sysdata[sysname]["t2"], K_FROZEN) for sysname in ("H10", "N2")}

    all_rows = []
    for sysname, chain in ALL_CHAINS:
        df = get_chain_df(sysname, chain)
        if len(df) == 0:
            out(f"  WARNING: no cached data for {sysname}/{chain}, skipping")
            continue
        pos = positions[(sysname, chain)]
        sd = sysdata[sysname]
        nocc, norb = sd["nocc"], sd["norb"]
        t0 = time.time()
        df = df.copy()
        df["S0"] = df.triple.apply(lambda A: score_S0(A, sd["Jab"]))
        df["T1"] = df.triple.apply(lambda A: score_T1(pos, A, channels_frozen[sysname], nocc, L_FROZEN, D_FROZEN))
        df["T2"] = df.triple.apply(lambda A: score_T2(pos, A, sd["t2"], sd["Jaa"], sd["Jab"], nocc, norb, BUDGET))
        df["system"] = sysname
        df["chain"] = chain
        all_rows.append(df)
        print(f"  {sysname}/{chain}: n={len(df)}  [{time.time()-t0:.1f}s]", flush=True)
    scored = pd.concat(all_rows, ignore_index=True)
    scored.to_csv(OUTDIR / "all_scores.csv", index=False)

    # ---------------------------------------------------------- comparison table
    banner("Per-chain, per-score: rho(score, captured), rho(score, err_sqd), regret_frac")
    SCORES = ["S0", "T1", "T2"]
    table_rows = []
    for name in SCORES:
        for sysname, chain in ALL_CHAINS:
            sub = scored[(scored.system == sysname) & (scored.chain == chain)]
            if len(sub) < 3:
                continue
            r_cap = spearmanr(sub[name], sub.captured)
            r_sqd = spearmanr(sub[name], sub.err_sqd)
            regret = regret_fraction(sub[name].to_numpy(), sub.err_sqd.to_numpy())
            table_rows.append(dict(score=name, system=sysname, chain=chain, n=len(sub),
                                   rho_captured=r_cap.statistic, p_captured=r_cap.pvalue,
                                   rho_sqd=r_sqd.statistic, p_sqd=r_sqd.pvalue, regret_frac=regret))
    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUTDIR / "comparison_table.csv", index=False)

    for name in SCORES:
        sub = table_df[table_df.score == name]
        out(f"\n{name}:")
        for _, row in sub.iterrows():
            out(f"  {row.system}/{row.chain:<12} n={row.n:<3.0f}  "
                f"rho(.,captured)={row.rho_captured:+.3f} (p={row.p_captured:.2e})   "
                f"rho(.,err_sqd)={row.rho_sqd:+.3f} (p={row.p_sqd:.2e})   regret={row.regret_frac:.3f}")

    # ---------------------------------------------------------- decisive summary
    # NOTE ON SIGN CONVENTION: captured is a "higher is better" quantity, so a
    # useful score must correlate POSITIVELY with it -- worst case is therefore
    # the SMALLEST (most negative) rho across chains, i.e. .min(), the mirror
    # image of the err_sqd convention used throughout this project (where
    # negative rho is good and worst case is the LEAST negative, .max()).
    # Getting this backwards would silently manufacture a false "beats S0"
    # verdict from scores that are actually anti-correlated with captured.
    banner("DECISIVE SUMMARY -- worst-case rho(captured), S0 first")
    summary_rows = []
    for name in SCORES:
        sub = table_df[table_df.score == name]
        summary_rows.append(dict(
            score=name,
            worst_rho_captured=float(sub.rho_captured.min()),   # most-negative = worst (captured: higher=better)
            mean_rho_captured=float(sub.rho_captured.mean()),
            worst_rho_sqd=float(sub.rho_sqd.max()),             # least-negative = worst (err_sqd: lower=better)
            mean_rho_sqd=float(sub.rho_sqd.mean()),
            median_regret=float(sub.regret_frac.median()),
            worst_regret=float(sub.regret_frac.max()),
        ))
    summary_df = pd.DataFrame(summary_rows)
    out(summary_df.to_string(index=False))

    s0_row = summary_df[summary_df.score == "S0"].iloc[0]
    MARGIN = 0.05  # a "clear margin" beyond S0's worst case, in rho units
    verdicts = {}
    for name in ("T1", "T2"):
        row = summary_df[summary_df.score == name].iloc[0]
        beats = row.worst_rho_captured > (s0_row.worst_rho_captured + MARGIN)
        verdicts[name] = beats
        out(f"\n{name} worst-case rho(captured) = {row.worst_rho_captured:+.3f}  vs  "
            f"S0 worst-case = {s0_row.worst_rho_captured:+.3f}  "
            f"(margin={MARGIN}) -> {'BEATS S0 by a clear margin' if beats else 'does NOT beat S0'}")

    banner("VERDICT")
    if not any(verdicts.values()):
        out("Both T1 and T2 FAIL to beat S0's worst-case rho(captured) by a clear margin.")
        out("This is a SECOND closed negative result, strengthening the structural conclusion: "
            "no score constructed so far -- variational-energy-targeted (S1-S4) or "
            "capture-targeted (T1, T2) -- resolves the chain-dependence of the anchor-selection "
            "problem. The failure mode is not in what the score targets (energy vs. capture); it "
            "is structural.")
    else:
        winners = [n for n, v in verdicts.items() if v]
        out(f"Beats S0 by a clear margin: {winners}")

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="chain_aware_v2", git_commit=git_commit,
        transmission_csv_sha256=sha256_of(TRANSMISSION_CSV), phaseb_csv_sha256=sha256_of(PHASEB_CSV),
        k_grid=K_GRID, l_grid=L_GRID, d_grid=D_GRID, k_frozen=K_FROZEN, l_frozen=L_FROZEN, d_frozen=D_FROZEN,
        budget_t2=BUDGET, margin=MARGIN,
        sweep=sweep_df.to_dict(orient="records"), summary=summary_df.to_dict(orient="records"),
        verdicts=verdicts, n_chains=len(ALL_CHAINS),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (OUTDIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\n[out] {OUTDIR / 'all_scores.csv'}")
    print(f"[out] {OUTDIR / 'comparison_table.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
