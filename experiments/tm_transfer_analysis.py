#!/usr/bin/env python3
"""
experiments/tm_transfer_analysis.py
======================================

STAGE 3 -- analysis of the transition-metal transfer experiment
(experiments/tm_transfer.py). Pure re-analysis of stage1_ansatz.csv /
stage2_sqd.csv / run_metadata.json -- no new sampling.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTDIR = Path(__file__).resolve().parent / "outputs" / "tm_transfer"
SIG_RHO, SIG_P = 0.3, 0.05

H10_RANGE_PER_HEADROOM = 1445.1
N2_RANGE_PER_HEADROOM = 2550.1
H10_RHO_S0_LUCJ = -0.850
N2_RHO_S0_LUCJ = -0.965

REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def parse_triple(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if s.startswith("("):
        return tuple(int(x) for x in ast.literal_eval(s))
    return None  # "default" / "no_ab"


def regret_fraction(scores: np.ndarray, err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    rand_regret = err.mean() - err.min()
    pick = int(np.asarray(scores).argmax())
    regret = err[pick] - err.min()
    return float(regret / rand_regret) if rand_regret > 0 else float("nan")


def sig(r) -> bool:
    return abs(r.statistic) >= SIG_RHO and r.pvalue < SIG_P


def main() -> int:
    banner("STAGE 3 -- TRANSITION-METAL TRANSFER: ANALYSIS")

    meta = json.loads((OUTDIR / "run_metadata.json").read_text())
    out(f"system_used = {meta['system_used']}   fallback_fired = {meta['fallback_fired']}")
    out(f"norb={meta['norb']}  nocc={meta['nocc']}  E_CASCI={meta['E_CASCI']:.10f}")
    out(f"budget={meta['budget']}  ideal_ceiling={meta['ceiling']:.6f}  shots={meta['shots']}  "
        f"stability_sd={meta['stability_sd']:.4f} mHa  boundary_ratio={meta['boundary_ratio']}")

    s1 = pd.read_csv(OUTDIR / "stage1_ansatz.csv")
    s1["triple"] = s1.triple.apply(lambda x: tuple(int(c) for c in ast.literal_eval(x)))
    s2 = pd.read_csv(OUTDIR / "stage2_sqd.csv")
    s2["triple_parsed"] = s2.triple.apply(parse_triple)
    ok = s2[s2.status == "OK"].copy()

    chains = sorted(s1.chain.unique())
    out(f"\nchains: {chains}")

    per_chain = {}
    for chain in chains:
        banner(f"Chain: {chain}")
        sub_triples = ok[(ok.chain == chain) & (ok.role == "triple")]
        default_row = ok[(ok.chain == chain) & (ok.role == "default")]
        noab_row = ok[(ok.chain == chain) & (ok.role == "no_ab")]

        err_range = float(sub_triples.err_mHa.max() - sub_triples.err_mHa.min())
        default_captured = float(default_row.captured.iloc[0]) if len(default_row) else float("nan")
        headroom = meta["ceiling"] - default_captured
        range_per_headroom = err_range / headroom if headroom > 0 else float("nan")
        out(f"  err range across {len(sub_triples)} triples: "
            f"{sub_triples.err_mHa.min():.2f} - {sub_triples.err_mHa.max():.2f} ({err_range:.2f} mHa)")
        out(f"  default-anchor captured={default_captured:.4f}  headroom={headroom:.4f}  "
            f"range/headroom={range_per_headroom:.1f} mHa/unit  "
            f"(H10={H10_RANGE_PER_HEADROOM}, N2={N2_RANGE_PER_HEADROOM})")

        r_cap = spearmanr(sub_triples.captured, sub_triples.err_mHa)
        out(f"  rho(captured, err_sqd) = {r_cap.statistic:+.3f} (p={r_cap.pvalue:.2e})  "
            f"{'HOLDS' if sig(r_cap) else 'FAILS'}  [mechanism, expected to hold]")

        s1_chain = s1[s1.chain == chain]
        lucj_by_triple = dict(zip(s1_chain.triple, s1_chain.err_lucj))
        sub_triples = sub_triples.copy()
        sub_triples["err_lucj"] = sub_triples.triple_parsed.map(lucj_by_triple)
        r_s0_lucj = spearmanr(sub_triples.S0, sub_triples.err_lucj)
        out(f"  rho(S0, err_lucj) = {r_s0_lucj.statistic:+.3f} (p={r_s0_lucj.pvalue:.2e})  "
            f"(H10={H10_RHO_S0_LUCJ}, N2={N2_RHO_S0_LUCJ})")

        r_s0_sqd = spearmanr(sub_triples.S0, sub_triples.err_mHa)
        regret = regret_fraction(sub_triples.S0.to_numpy(), sub_triples.err_mHa.to_numpy())
        out(f"  rho(S0, err_sqd) = {r_s0_sqd.statistic:+.3f} (p={r_s0_sqd.pvalue:.2e})  "
            f"regret_frac={regret:.3f}")

        floor = float(noab_row.err_mHa.iloc[0]) if len(noab_row) else float("nan")
        n_worse = int((sub_triples.err_mHa > floor).sum())
        out(f"  no-alpha-beta floor = {floor:.2f} mHa  worse-than-floor = {n_worse}/{len(sub_triples)} "
            f"({100*n_worse/len(sub_triples):.1f}%)")

        best_row = sub_triples.loc[sub_triples.err_mHa.idxmin()]
        out(f"  best triple = {best_row.triple_parsed}  err={best_row.err_mHa:.2f} mHa")

        per_chain[chain] = dict(err_range=err_range, range_per_headroom=range_per_headroom,
                                rho_captured=r_cap.statistic, p_captured=r_cap.pvalue,
                                rho_s0_lucj=r_s0_lucj.statistic, p_s0_lucj=r_s0_lucj.pvalue,
                                rho_s0_sqd=r_s0_sqd.statistic, p_s0_sqd=r_s0_sqd.pvalue,
                                regret_frac=regret, floor=floor, n_worse_than_floor=n_worse,
                                n_triples=len(sub_triples), best_triple=str(best_row.triple_parsed),
                                best_err=float(best_row.err_mHa))

    banner("Cross-chain summary")
    summary_df = pd.DataFrame(per_chain).T
    summary_df.to_csv(OUTDIR / "stage3_per_chain_summary.csv")
    out(summary_df.to_string())

    best_triples = {c: v["best_triple"] for c, v in per_chain.items()}
    all_same = len(set(best_triples.values())) == 1
    out(f"\nbest triple per chain: {best_triples}")
    out(f"same across all chains? {all_same}")

    mech_holds = all(sig_r for sig_r in
                     [abs(v["rho_captured"]) >= SIG_RHO and v["p_captured"] < SIG_P for v in per_chain.values()])
    lucj_rhos = [v["rho_s0_lucj"] for v in per_chain.values()]
    replicates = all(r <= -0.5 for r in lucj_rhos)  # same-sign, comparable magnitude test
    mean_range_per_headroom = float(np.mean([v["range_per_headroom"] for v in per_chain.values()]))

    banner("HEADLINE")
    out(f"1. System used: {meta['system_used']}. Fallback fired: {meta['fallback_fired']}.")
    out(f"2. Capture mechanism holds on the transition-metal active space: "
        f"{'YES' if mech_holds else 'NO'} -- rho(captured,err_sqd) "
        f"{[f'{v['rho_captured']:+.3f}' for v in per_chain.values()]} across chains {chains}.")
    out(f"3. rho(S0, err_lucj) replicates at the -0.85 to -0.97 level: "
        f"{'YES' if replicates else 'NO/PARTIALLY'} -- values {[f'{r:+.3f}' for r in lucj_rhos]} "
        f"across chains {chains} (H10={H10_RHO_S0_LUCJ}, N2={N2_RHO_S0_LUCJ}).")
    out(f"4. Anchor effect in headroom-normalised terms: {mean_range_per_headroom:.1f} mHa/unit headroom "
        f"(mean across chains) vs. H10={H10_RANGE_PER_HEADROOM}, N2={N2_RANGE_PER_HEADROOM}.")
    out(f"5. Best triple moves between chains: {'NO -- same triple at all chains' if all_same else 'YES, differs across chains'}.")

    report_path = OUTDIR / "stage3_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    out_meta = dict(part="tm_transfer_stage3", git_commit=git_commit, per_chain=per_chain,
                    best_triples=best_triples, all_same_best_triple=all_same,
                    mechanism_holds=mech_holds, s0_lucj_replicates=replicates,
                    mean_range_per_headroom=mean_range_per_headroom,
                    generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUTDIR / "stage3_metadata.json").write_text(json.dumps(out_meta, indent=2, default=str))
    print(f"\n[out] {report_path}")
    print(f"[out] {OUTDIR / 'stage3_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
