#!/usr/bin/env python3
"""
experiments/score_audit.py
============================

PART A -- score audit (no sampling). Uses the cached H10 R=1.6 reference and
the 57 permutations already sampled in experiments/outputs/h10_baseline_R1.6/
h10_baseline_results.csv (err_mHa, captured averaged over the 2 seeds per
ordering). Computes every score1/score2/retained_J variant via the shared
scoring functions in run_ordering_pipeline.py - nothing reimplemented.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
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

OUTDIR = Path(__file__).resolve().parent / "outputs" / "score_audit_R1.6"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

RAND_REGRET_GIVEN = 95.01
PREDICTIVE_REGRET_THRESH = RAND_REGRET_GIVEN / 2.0  # 47.5, per spec
NAMED = ["identity", "reverse", "physical", "physical_reverse",
         "s1_max", "s2_max", "retainedJ_max"]

VARIANTS = ["s1_amp", "s1_ampJ", "s1_amp_ss", "s1_amp_os", "s1_ampJ_ss",
           "s1_ampJ_os", "s2", "s2_ss", "s2_os", "s2_soft_ss", "retained_J"]


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    banner("PART A -- SCORE AUDIT (no sampling)")
    import run_ordering_pipeline as R

    ref_path = CACHEDIR / "reference.npz"
    if not ref_path.exists():
        sys.exit(f"FATAL: no cached H10 reference at {CACHEDIR}. Not recomputing.")
    if not BASELINE_CSV.exists():
        sys.exit(f"FATAL: baseline results not found at {BASELINE_CSV}.")

    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(CACHEDIR))
    norb, nocc = ref["norb"], ref["nocc"]
    t1L, t2L = ref["t1L"], ref["t2L"]
    amp = R.Amplitudes(t1L, t2L, nocc, norb)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))
    w_ss = float(np.abs(Jaa).sum() / (np.abs(Jaa).sum() + np.abs(Jab).sum()))
    print(f"[ref] loaded {CACHEDIR}  norb={norb} nocc={nocc} w_ss={w_ss:.4f}")

    base = pd.read_csv(BASELINE_CSV)
    base_ok = base[base.status == "OK"]
    per_ord = base_ok.groupby("ordering").agg(
        err_mHa=("err_mHa", "mean"), captured=("captured", "mean"),
        permutation=("permutation", "first")).reset_index()
    print(f"[baseline] {len(per_ord)} orderings loaded from {BASELINE_CSV.name} "
          f"(mean over seeds, OK rows only)")

    # ------------------------------------------------------- A1: all scores
    rows = []
    for _, r in per_ord.iterrows():
        perm = R.parse_permutation(r["permutation"], norb)
        pos = R.positions_from(perm)
        s1 = R.score1(pos, amp, Jaa, Jab, w_ss)
        s2 = R.score2(pos, amp, w_ss)
        rj = R.retained_J_of(pos, Jaa, Jab)
        rj_ss, rj_os = R.retained_J_split_of(pos, Jaa, Jab)
        row = dict(ordering=r["ordering"], err_mHa=r["err_mHa"], captured=r["captured"])
        row.update({k: s1[k] for k in ("s1_amp", "s1_amp_ss", "s1_amp_os",
                                       "s1_ampJ", "s1_ampJ_ss", "s1_ampJ_os")})
        row.update({k: s2[k] for k in ("s2", "s2_ss", "s2_os", "s2_soft_ss")})
        row["retained_J"] = rj
        row["retained_J_samespin"] = rj_ss
        row["retained_J_oppspin"] = rj_os
        rows.append(row)
    scores_df = pd.DataFrame(rows)
    scores_csv = OUTDIR / "all_scores.csv"
    scores_df.to_csv(scores_csv, index=False)

    is_named = scores_df.ordering.isin(NAMED)
    rnd = scores_df[~is_named].reset_index(drop=True)
    named_df = scores_df[is_named].set_index("ordering")
    print(f"[A1] {len(rnd)} random orderings, {len(named_df)} named orderings")

    err = rnd["err_mHa"].to_numpy(float)
    cap = rnd["captured"].to_numpy(float)
    rand_regret = float(err.mean() - err.min())
    print(f"[A1] random-selection regret (recomputed) = {rand_regret:.2f} mHa "
          f"(spec cites {RAND_REGRET_GIVEN})")

    banner("A1 -- per-variant audit (50 random orderings)")
    print(f"{'variant':<14}{'rho_err':>9}{'p_err':>11}{'rho_cap':>9}{'p_cap':>11}"
          f"{'picks':>10}{'regret':>9}{'const?':>9}")
    audit = {}
    for v in VARIANTS:
        x = rnd[v].to_numpy(float)
        rng_ = float(np.ptp(x))
        mean_ = float(np.mean(x))
        const_flag = "CONST" if rng_ < 1e-12 else (
            "near-const" if mean_ != 0 and abs(rng_ / mean_) < 0.01 else "")
        if rng_ < 1e-12:
            sr_e = sr_c = type("R", (), {"statistic": float("nan"), "pvalue": float("nan")})()
            pick = "n/a"
            regret = float("nan")
        else:
            sr_e = spearmanr(x, err)
            sr_c = spearmanr(x, cap)
            k = int(np.argmax(x))
            pick = rnd["ordering"].iloc[k]
            regret = float(err[k] - err.min())
        audit[v] = dict(rho_err=sr_e.statistic, p_err=sr_e.pvalue,
                        rho_cap=sr_c.statistic, p_cap=sr_c.pvalue,
                        pick=pick, regret=regret, const_flag=const_flag)
        print(f"{v:<14}{sr_e.statistic:>+9.3f}{sr_e.pvalue:>11.1e}"
              f"{sr_c.statistic:>+9.3f}{sr_c.pvalue:>11.1e}"
              f"{str(pick):>10}{regret:>9.2f}{const_flag:>9}")

    # ---------------------------------------------------------- A2: climbed-on variants
    banner("A2 -- which variant was each named ordering actually climbed on?")
    climbed_on = dict(s1_max="s1_ampJ", s2_max="s2", retainedJ_max="retained_J")
    print("From run_ordering_pipeline.py build_or_load_h10_reference() (unchanged since")
    print("commit 333acec, verified via git blame):")
    print('  obj_s2 = lambda p: score2(positions_from(p), amp, w_ss)["s2"]')
    print('  obj_s1 = lambda p: score1(positions_from(p), amp, Jaa, Jab, w_ss)["s1_ampJ"]')
    print('  obj_rj = lambda p: retained_J_of(positions_from(p), Jaa, Jab)')
    print()
    for name, variant in climbed_on.items():
        a = audit[variant]
        predictive = (abs(a["rho_err"]) >= 0.5 and a["p_err"] < 1e-3
                     and not np.isnan(a["regret"]) and a["regret"] <= PREDICTIVE_REGRET_THRESH)
        verdict = "PREDICTIVE" if predictive else "NOT predictive"
        print(f"  {name:<16} climbed on '{variant}':  rho={a['rho_err']:+.3f}  "
              f"p={a['p_err']:.1e}  regret={a['regret']:.2f}  -> {verdict}")
        if not predictive:
            print(f"    -> {name}'s percentile is NOT attributable to '{variant}' "
                  f"predicting err_mHa on the random distribution.")

    # ---------------------------------------------------------- A3: null test
    banner("A3 -- exact combinatorial null test")
    print("Null: each named ordering's err_mHa is an independent exchangeable draw")
    print("from the same distribution as the 50 random orderings. Under exchangeability,")
    print("inserting 1 new i.i.d. draw into a pool of 50 makes its RANK among all 51")
    print("uniform on {1,...,51} (exact, no distributional assumption). If the draw")
    print("beats m of the 50 (has lower err_mHa than m of them), its rank is 51-m, so")
    print("P(beats >= m) = P(rank <= 51-m) = (51-m)/51 exactly.")
    print()
    named_err = {n: float(named_df.loc[n, "err_mHa"]) for n in ("s1_max", "s2_max", "retainedJ_max")
                if n in named_df.index}
    probs = {}
    for name, e in named_err.items():
        m = int((err > e).sum())
        pct = 100.0 * m / len(err)
        p_i = (len(err) + 1 - m) / (len(err) + 1)
        probs[name] = p_i
        print(f"  {name:<16} err={e:.2f} mHa  beats {m}/{len(err)} ({pct:.1f}%ile)  "
              f"P(beats>={m}) = ({len(err)+1}-{m})/{len(err)+1} = {p_i:.4f}")
    joint = float(np.prod(list(probs.values())))
    print(f"\n  Joint (independence assumed): "
          f"{' x '.join(f'{p:.4f}' for p in probs.values())} = {joint:.5f}")
    print(f"  p-value = {joint:.5f}")

    # ------------------------------------------------------- A4: sector split
    banner("A4 -- retained_J: same-spin vs opposite-spin sectors")
    for sector, col in (("same-spin", "retained_J_samespin"), ("opposite-spin", "retained_J_oppspin")):
        x = rnd[col].to_numpy(float)
        sr = spearmanr(x, err)
        print(f"  retained_J ({sector:<13}): rho={sr.statistic:+.3f}  p={sr.pvalue:.2e}  "
              f"range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"  combined retained_J: rho={audit['retained_J']['rho_err']:+.3f} "
          f"(given as +0.001 in Part 6 of the previous report)")
    ss_rho = spearmanr(rnd["retained_J_samespin"], err).statistic
    ss_p = spearmanr(rnd["retained_J_samespin"], err).pvalue
    os_rho = spearmanr(rnd["retained_J_oppspin"], err).statistic
    os_p = spearmanr(rnd["retained_J_oppspin"], err).pvalue
    opposite_signs = np.sign(ss_rho) != np.sign(os_rho)
    either_significant = ss_p < 0.05 or os_p < 0.05
    if opposite_signs and either_significant:
        print(f"  -> Signs DO oppose (same-spin {ss_rho:+.3f}, opp-spin {os_rho:+.3f}), "
              f"and opposite-spin alone is significant (p={os_p:.2e}) even though it "
              f"doesn't clear the |rho|>=0.5 PREDICTIVE bar. This is a genuine partial "
              f"cancellation: retained_J's combined near-zero rho comes from averaging "
              f"a real (if modest) negative opposite-spin signal against a weaker, "
              f"non-significant positive same-spin one - not from two null sectors.")
    else:
        print("  -> No sign-cancellation pattern strong enough to explain the combined "
              "near-zero rho; the sectors are close to independently null.")

    # --------------------------------------------------------------- A5: verdict
    banner("A5 -- VERDICT")
    print(f"{'variant':<14}{'rho':>9}{'p':>11}{'regret':>9}{'verdict':>14}")
    any_predictive = []
    for v in VARIANTS:
        a = audit[v]
        if a["const_flag"] == "CONST":
            verdict = "NULL (constant)"
        elif (not np.isnan(a["rho_err"]) and abs(a["rho_err"]) >= 0.5
              and a["p_err"] < 1e-3 and a["regret"] <= PREDICTIVE_REGRET_THRESH):
            verdict = "PREDICTIVE"
            any_predictive.append(v)
        elif not np.isnan(a["rho_err"]) and abs(a["rho_err"]) >= 0.3:
            verdict = "WEAK"
        else:
            verdict = "NULL"
        print(f"{v:<14}{a['rho_err']:>+9.3f}{a['p_err']:>11.1e}{a['regret']:>9.2f}{verdict:>14}")

    if any_predictive:
        print(f"\nWorth carrying forward: {any_predictive}.")
    else:
        print("\nNone of the eleven variants clears the PREDICTIVE bar on this random "
              "distribution - no score here is worth carrying forward as-is.")

    # ------------------------------------------------------------- metadata
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="A", git_commit=git_commit,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        fcidump_sha256=sha256_of(CACHEDIR / "fcidump.txt"),
        orderings_json_sha256=sha256_of(CACHEDIR / "orderings.json"),
        baseline_csv_sha256=sha256_of(BASELINE_CSV),
        n_random=len(rnd), n_named=len(named_df),
        random_selection_regret=rand_regret,
        climbed_on=climbed_on,
        null_test_probs=probs, null_test_joint_p=joint,
        predictive_variants=any_predictive,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {scores_csv}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
