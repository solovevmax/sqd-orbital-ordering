#!/usr/bin/env python3
"""
experiments/anchor_reanalysis.py
===================================

D1-D6 -- pure re-analysis of experiments/outputs/anchor_decomposition_R1.6/
c1_all120_identity.csv and c2_transfer.csv, plus the cached H10 R=1.6
reference (for amp/J_ab, needed to compute the new amplitude-weighted score
variants). NO sampling, NO sbd, NO reference recomputation.
"""
from __future__ import annotations

import ast
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "anchor_reanalysis"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
ANCHOR_DIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
C1_CSV = ANCHOR_DIR / "c1_all120_identity.csv"
C2_CSV = ANCHOR_DIR / "c2_transfer.csv"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

C2_ORDERINGS_INDEPENDENT = ["identity", "physical", "rand007"]  # physical_reverse dropped (duplicate of physical)
REPORT_LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    REPORT_LINES.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_triple(s):
    if isinstance(s, tuple):
        return s
    s = str(s).strip()
    if s.startswith("("):
        return ast.literal_eval(s)
    return tuple(int(c) for c in s.zfill(3))


def main() -> int:
    banner("D1-D6 -- ANCHOR RE-ANALYSIS (no sampling)")
    import run_ordering_pipeline as R

    for p in (C1_CSV, C2_CSV, BASELINE_CSV):
        if not p.exists():
            sys.exit(f"FATAL: required input missing: {p}")
    if not (CACHEDIR / "reference.npz").exists():
        sys.exit(f"FATAL: no cached H10 reference at {CACHEDIR}.")

    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(CACHEDIR))
    norb, nocc = ref["norb"], ref["nocc"]
    t1L, t2L = ref["t1L"], ref["t2L"]
    amp = R.Amplitudes(t1L, t2L, nocc, norb)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))
    w_ss = float(np.abs(Jaa).sum() / (np.abs(Jaa).sum() + np.abs(Jab).sum()))
    pos_id = R.positions_from(np.arange(norb))
    pos_phys = R.positions_from(R.parse_permutation(
        pd.read_csv(BASELINE_CSV).groupby("ordering")["permutation"].first()["physical"], norb))
    pos_rand007 = R.positions_from(R.parse_permutation(
        pd.read_csv(BASELINE_CSV).groupby("ordering")["permutation"].first()["rand007"], norb))
    pos_by_ordering = dict(identity=pos_id, physical=pos_phys, rand007=pos_rand007)

    c1 = pd.read_csv(C1_CSV)
    c1["triple"] = c1["triple"].apply(parse_triple)
    c2 = pd.read_csv(C2_CSV)
    c2["triple"] = c2["triple"].apply(parse_triple)
    c2 = c2[c2.ordering != "physical_reverse"].reset_index(drop=True)
    print(f"[data] C1: {len(c1)} rows (identity, full 120). "
          f"C2: {len(c2)} rows, orderings {sorted(c2.ordering.unique())} "
          f"(physical_reverse dropped - bit-identical to physical, see prior commit)")

    shared_triples = sorted(set(c2.triple))
    assert len(shared_triples) == 40, f"expected 40 shared triples, got {len(shared_triples)}"
    c1_lookup = c1.set_index(c1.triple.apply(str))
    identity_on_shared = pd.DataFrame([
        dict(ordering="identity", triple=t,
            err_mHa=c1_lookup.loc[str(t), "err_mHa"],
            captured=c1_lookup.loc[str(t), "captured"],
            retained_J_oppspin=c1_lookup.loc[str(t), "retained_J_oppspin"])
        for t in shared_triples])
    per_ordering_40 = pd.concat([identity_on_shared, c2[["ordering", "triple", "err_mHa",
                                                          "captured", "retained_J_oppspin"]]],
                                ignore_index=True)

    # =========================================================== D1
    banner("D1 -- is the MECHANISM ordering-dependent, or only the PROXY?")
    d1_rows = []
    for name in C2_ORDERINGS_INDEPENDENT:
        sub = per_ordering_40[per_ordering_40.ordering == name]
        r_cap = spearmanr(sub.captured, sub.err_mHa)
        r_os = spearmanr(sub.retained_J_oppspin, sub.err_mHa)
        rng = sub.err_mHa.max() - sub.err_mHa.min()
        d1_rows.append(dict(ordering=name, rho_captured=r_cap.statistic, p_captured=r_cap.pvalue,
                            rho_oppspin=r_os.statistic, p_oppspin=r_os.pvalue, range_mHa=rng))
        out(f"  {name:<12} rho(captured,err)={r_cap.statistic:+.3f} (p={r_cap.pvalue:.2e})   "
            f"rho(retained_J_oppspin,err)={r_os.statistic:+.3f} (p={r_os.pvalue:.2e})   "
            f"range={rng:.2f} mHa")
    all_cap_predictive = all(abs(d["rho_captured"]) >= 0.5 and d["p_captured"] < 0.05 for d in d1_rows)
    out(f"\n  captured predicts at all three orderings? {all_cap_predictive}")
    if all_cap_predictive:
        out("  -> MECHANISM HOLDS at all three: capture always predicts err_mHa. "
            "retained_J_oppspin's weakness at physical is a PROXY failure (it doesn't "
            "track capture well there), not a breakdown of the capture-error mechanism.")
    else:
        weak = [d["ordering"] for d in d1_rows if not (abs(d["rho_captured"]) >= 0.5 and d["p_captured"] < 0.05)]
        out(f"  -> Capture itself degrades at: {weak}. Something STRUCTURAL changes there, "
            f"not just a poor proxy.")

    # =========================================================== D2
    banner("D2 -- regret accounting per ordering")
    d2_rows = []
    for name in C2_ORDERINGS_INDEPENDENT:
        sub = per_ordering_40[per_ordering_40.ordering == name].reset_index(drop=True)
        err = sub.err_mHa.to_numpy()
        rand_regret = float(err.mean() - err.min())
        pick_idx = int(sub.retained_J_oppspin.to_numpy().argmax())
        pick_err = err[pick_idx]
        regret = float(pick_err - err.min())
        frac = regret / rand_regret if rand_regret > 0 else float("nan")
        rank = int((err < pick_err).sum() + 1)
        pct = 100.0 * (err > pick_err).mean()
        d2_rows.append(dict(scope=f"{name} (40 shared)", rand_regret=rand_regret, rule_regret=regret,
                            frac=frac, rank=rank, n=len(sub), percentile=pct))
        out(f"  {name:<20} (n=40)  random_regret={rand_regret:6.2f}  rule_regret={regret:6.2f}  "
            f"frac={frac:.3f}  rank={rank}/{len(sub)}  percentile={pct:.1f}%")

    err120 = c1.err_mHa.to_numpy()
    rand_regret120 = float(err120.mean() - err120.min())
    pick_idx120 = int(c1.retained_J_oppspin.to_numpy().argmax())
    pick_err120 = err120[pick_idx120]
    regret120 = float(pick_err120 - err120.min())
    frac120 = regret120 / rand_regret120
    rank120 = int((err120 < pick_err120).sum() + 1)
    pct120 = 100.0 * (err120 > pick_err120).mean()
    d2_rows.append(dict(scope="identity (full 120)", rand_regret=rand_regret120, rule_regret=regret120,
                        frac=frac120, rank=rank120, n=120, percentile=pct120))
    out(f"  {'identity (full 120)':<20} (n=120) random_regret={rand_regret120:6.2f}  "
        f"rule_regret={regret120:6.2f}  frac={frac120:.3f}  rank={rank120}/120  "
        f"percentile={pct120:.1f}%")
    out(f"\n  fractional reduction vs random selection: "
        + ", ".join(f"{d['scope']}={100*(1-d['frac']):.1f}%" for d in d2_rows))

    # =========================================================== D3
    banner("D3 -- amplitude-weighted anchor selection")
    M_os = np.abs(Jab).sum(axis=0).diagonal() * amp.A_os_site
    tot_A = amp.A_os_site.sum()
    tot_M = M_os.sum()

    def t2_participation_raw(orbital):
        """Independent re-derivation of 'total |t2| amplitude ORBITAL p participates
        in', straight from t2L, NOT via amp.A_os_site - a due-diligence cross-check
        on the PER-ORBITAL quantity (the right level to check: A_os_site[p] itself,
        not a triple-summed quantity, which would double-count terms that touch more
        than one anchor orbital and so is not comparable to a naive re-derivation)."""
        total = 0.0
        nvir = norb - nocc
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        w = abs(float(t2L[i, j, a, b]))
                        if w == 0:
                            continue
                        if orbital in {i, j, nocc + a, nocc + b}:
                            total += w
        return total

    def scores_for_triple(triple):
        s1_amp_os = float(sum(amp.A_os_site[p] for p in triple) / tot_A) if tot_A > 0 else 0.0
        s1_ampJ_os = float(sum(M_os[p] for p in triple) / tot_M) if tot_M > 0 else 0.0
        return s1_amp_os, s1_ampJ_os

    all_triples_c1 = list(c1.triple)
    d3_cache = {t: scores_for_triple(t) for t in set(all_triples_c1) | set(shared_triples)}

    # due-diligence: confirm A_os_site[p] (the basis of s1_amp_os, and of the "sum over
    # anchor orbitals of the total |t2| amplitude that orbital participates in" variant
    # requested - the same formula) matches an independent re-derivation, per orbital.
    raw_per_orbital = [t2_participation_raw(p) for p in range(norb)]
    coincide = all(abs(raw_per_orbital[p] - amp.A_os_site[p]) < 1e-9 for p in range(norb))
    out(f"  Due-diligence check (per-orbital, all {norb} orbitals): independently-summed "
        f"raw |t2| participation vs amp.A_os_site: {'MATCH' if coincide else 'MISMATCH'}")
    if coincide:
        out("  -> The 't2-magnitude variant' and 's1_amp_os' are the SAME quantity up to "
            "normalisation - amp.A_os_site is already defined as the total |t2| amplitude "
            "each orbital participates in (see run_ordering_pipeline.py's Amplitudes class). "
            "They will show IDENTICAL rho vs err_mHa; not reported as a third independent test.")

    c1["s1_amp_os"] = c1.triple.map(lambda t: d3_cache[t][0])
    c1["s1_ampJ_os"] = c1.triple.map(lambda t: d3_cache[t][1])
    per_ordering_40["s1_amp_os"] = per_ordering_40.triple.map(lambda t: d3_cache[t][0])
    per_ordering_40["s1_ampJ_os"] = per_ordering_40.triple.map(lambda t: d3_cache[t][1])

    out(f"\n  {'ordering':<14}{'variant':<16}{'rho':>9}{'p':>11}{'regret':>9}{'frac':>7}")
    d3_rows = []
    for name in C2_ORDERINGS_INDEPENDENT:
        sub = per_ordering_40[per_ordering_40.ordering == name].reset_index(drop=True)
        err = sub.err_mHa.to_numpy()
        rand_regret = float(err.mean() - err.min())
        for variant in ("retained_J_oppspin", "s1_amp_os", "s1_ampJ_os"):
            x = sub[variant].to_numpy(float)
            sr = spearmanr(x, err)
            pick = int(x.argmax())
            reg = float(err[pick] - err.min())
            frac = reg / rand_regret if rand_regret > 0 else float("nan")
            d3_rows.append(dict(ordering=name, variant=variant, rho=sr.statistic, p=sr.pvalue,
                                regret=reg, frac=frac))
            out(f"  {name:<14}{variant:<16}{sr.statistic:>+9.3f}{sr.pvalue:>11.1e}"
                f"{reg:>9.2f}{frac:>7.3f}")
    phys_rows = {d["variant"]: d for d in d3_rows if d["ordering"] == "physical"}
    improved = any(abs(phys_rows[v]["rho"]) > abs(phys_rows["retained_J_oppspin"]["rho"]) + 0.1
                  and phys_rows[v]["p"] < 0.05 for v in ("s1_amp_os", "s1_ampJ_os"))
    out(f"\n  Does amplitude weighting fix physical's breakdown? "
        f"retained_J_oppspin rho={phys_rows['retained_J_oppspin']['rho']:+.3f}, "
        f"s1_amp_os rho={phys_rows['s1_amp_os']['rho']:+.3f}, "
        f"s1_ampJ_os rho={phys_rows['s1_ampJ_os']['rho']:+.3f}")
    out(f"  -> {'YES' if improved else 'NO'}: amplitude weighting does "
        f"{'materially improve' if improved else 'NOT fix'} physical's ordering-dependence.")

    # =========================================================== D4
    banner("D4 -- structural features of good/bad triples at identity (all 120)")
    def features(triple, pos):
        occ = sum(1 for p in triple if p < nocc)
        vir = 3 - occ
        idx_seps = [abs(a - b) for a, b in
                   [(triple[0], triple[1]), (triple[0], triple[2]), (triple[1], triple[2])]]
        pos_seps = [abs(int(pos[triple[0]]) - int(pos[triple[1]])),
                   abs(int(pos[triple[0]]) - int(pos[triple[2]])),
                   abs(int(pos[triple[1]]) - int(pos[triple[2]]))]
        return dict(n_occ=occ, n_vir=vir, min_idx_sep=min(idx_seps), max_idx_sep=max(idx_seps),
                   min_pos_sep=min(pos_seps), max_pos_sep=max(pos_seps))

    feat_rows = []
    for _, r in c1.iterrows():
        f = features(r.triple, pos_id)
        f.update(triple=r.triple, err_mHa=r.err_mHa)
        feat_rows.append(f)
    feat_df = pd.DataFrame(feat_rows)
    idx_pos_identical = (feat_df.min_idx_sep == feat_df.min_pos_sep).all() and \
        (feat_df.max_idx_sep == feat_df.max_pos_sep).all()
    out(f"  Index separation == position separation for ALL 120 (expected under identity, "
        f"since pos[p]=p): {idx_pos_identical}")

    out(f"\n  {'feature':<14}{'rho':>9}{'p':>11}")
    for feat in ("n_occ", "min_idx_sep", "max_idx_sep"):
        sr = spearmanr(feat_df[feat], feat_df.err_mHa)
        out(f"  {feat:<14}{sr.statistic:>+9.3f}{sr.pvalue:>11.1e}")

    feat_df_sorted = feat_df.sort_values("err_mHa")
    out(f"\n  10 BEST (lowest err_mHa):")
    out(f"  {'triple':<12}{'err_mHa':>9}{'n_occ':>7}{'min_sep':>9}{'max_sep':>9}")
    for _, r in feat_df_sorted.head(10).iterrows():
        out(f"  {str(r.triple):<12}{r.err_mHa:>9.2f}{r.n_occ:>7}{r.min_idx_sep:>9}{r.max_idx_sep:>9}")
    out(f"\n  10 WORST (highest err_mHa):")
    out(f"  {'triple':<12}{'err_mHa':>9}{'n_occ':>7}{'min_sep':>9}{'max_sep':>9}")
    for _, r in feat_df_sorted.tail(10).iterrows():
        out(f"  {str(r.triple):<12}{r.err_mHa:>9.2f}{r.n_occ:>7}{r.min_idx_sep:>9}{r.max_idx_sep:>9}")
    feat_df.to_csv(OUTDIR / "d4_features_identity120.csv", index=False)

    # =========================================================== D5
    banner("D5 -- bottleneck hypothesis")
    base = pd.read_csv(BASELINE_CSV)
    base_ok = base[base.status == "OK"]
    base_by_ord = base_ok.groupby("ordering")["err_mHa"].mean()
    baseline_err = {name: float(base_by_ord[name]) for name in C2_ORDERINGS_INDEPENDENT}
    order_by_baseline = sorted(baseline_err, key=lambda n: baseline_err[n])
    rho_by_ord = {d["ordering"]: d["rho_oppspin"] for d in d1_rows}
    out(f"  orderings sorted by their OWN default-anchor baseline err_mHa (best to worst):")
    for name in order_by_baseline:
        out(f"    {name:<12} baseline_err={baseline_err[name]:.2f}  "
            f"rho(retained_J_oppspin,err)={rho_by_ord[name]:+.3f}")
    rhos_in_baseline_order = [abs(rho_by_ord[n]) for n in order_by_baseline]
    monotonic = all(rhos_in_baseline_order[i] >= rhos_in_baseline_order[i + 1]
                    for i in range(len(rhos_in_baseline_order) - 1))
    worst_is_weakest = (order_by_baseline[-1] == min(rho_by_ord, key=lambda n: abs(rho_by_ord[n])))
    out(f"\n  strictly monotonic (|rho| decreasing as baseline err worsens)? {monotonic}")
    out(f"  worst-baseline ordering has the weakest |rho|? {worst_is_weakest}")
    if worst_is_weakest and not monotonic:
        out("  -> Direction PARTIALLY consistent with the bottleneck hypothesis: the worst "
            "same-spin ordering (physical) does have the weakest anchor-predictive power, "
            "but the relationship is not monotonic across all three (identity has a "
            "stronger rho than rand007 despite a worse baseline). Three points is not a "
            "test; this is suggestive, not confirmatory.")
    elif monotonic:
        out("  -> Direction fully consistent with the bottleneck hypothesis across all three "
            "points, though n=3 remains too small to call this confirmed.")
    else:
        out("  -> Direction NOT consistent with the bottleneck hypothesis as stated.")

    # =========================================================== D6
    banner("D6 -- VERDICT")
    candidates = ["retained_J_oppspin", "s1_amp_os", "s1_ampJ_os"]
    worst_case = {}
    for v in candidates:
        fracs = [d["frac"] for d in d3_rows if d["variant"] == v]
        worst_case[v] = max(fracs)
    out("  Candidate PRE-SAMPLING anchor-selection rules (captured excluded - it requires "
        "the sampling it would be used to avoid, so it is diagnostic only, not a rule):")
    for v in candidates:
        out(f"    {v:<18} worst-case fractional regret across 3 orderings = {worst_case[v]:.3f}")
    best_rule = min(worst_case, key=worst_case.get)
    if best_rule == "retained_J_oppspin":
        out(f"\n  VERDICT: no variant tested beats retained_J_oppspin "
            f"(worst-case fractional regret {worst_case['retained_J_oppspin']:.3f}).")
    else:
        out(f"\n  VERDICT: {best_rule} beats retained_J_oppspin "
            f"({worst_case[best_rule]:.3f} vs {worst_case['retained_J_oppspin']:.3f} "
            f"worst-case fractional regret).")

    # ------------------------------------------------------------- save
    combined_csv = OUTDIR / "anchor_reanalysis.csv"
    per_ordering_40.to_csv(combined_csv, index=False)
    report_path = OUTDIR / "report.txt"
    report_path.write_text("\n".join(REPORT_LINES) + "\n")

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="D1_D6_reanalysis", git_commit=git_commit,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        c1_csv_sha256=sha256_of(C1_CSV), c2_csv_sha256=sha256_of(C2_CSV),
        baseline_csv_sha256=sha256_of(BASELINE_CSV),
        orderings_used=C2_ORDERINGS_INDEPENDENT,
        physical_reverse_dropped_reason="bit-identical to physical (verified prior commit)",
        d3_t2_magnitude_coincides_with_s1_amp_os=coincide,
        d6_best_rule=best_rule, d6_worst_case_fractions=worst_case,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {combined_csv}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'd4_features_identity120.csv'}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
