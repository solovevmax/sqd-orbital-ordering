#!/usr/bin/env python3
"""
experiments/anchor_hardening.py
==================================

C1 -- remove B2's selection bias: sample all C(10,3)=120 anchor triples at
fixed identity same-spin ordering (30 already done in b2_sampled.csv, reused
here, NOT re-run; 90 new evaluations).

C2 -- transferability: repeat with 40 triples (shared across orderings,
drawn uniformly at random, not by ranking) for physical, physical_reverse,
and one random baseline ordering near the median err_mHa.

Reuses run_ordering_pipeline.py + anchor_decomposition.py's evaluate()/
circuit_stats() (same operator construction, sampling, determinant writing,
sbd call, circuit-stats probe) - nothing reimplemented.
"""
from __future__ import annotations

import ast
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"
B2_SAMPLED_CSV = OUTDIR / "b2_sampled.csv"
B2_RANKING_CSV = OUTDIR / "b2_all120_ranking.csv"

SHOTS = 2_000_000
C2_RNG_SEED = 20260825002
C2_N_SHARED = 40
C2_ORDERINGS = ["physical", "physical_reverse", "rand007"]  # + identity from C1


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    banner("C1/C2 -- ANCHOR RESULT HARDENING")
    import unified_run as U
    import run_ordering_pipeline as R
    from anchor_decomposition import evaluate  # reuse: sampling/det-write/sbd/circuit-stats

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at {R.CFG['sbd_bin']}")
    if not (CACHEDIR / "reference.npz").exists():
        sys.exit(f"FATAL: no cached H10 reference at {CACHEDIR}. Not recomputing.")
    for p in (B2_SAMPLED_CSV, B2_RANKING_CSV, BASELINE_CSV):
        if not p.exists():
            sys.exit(f"FATAL: required prior artefact missing: {p}")

    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(CACHEDIR))
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    fcidump_path = Path(ref["fcidump_path"])
    E_CASCI = ref["E_CASCI"]
    hf = R.hf_bitstring(norb, nocc)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))

    from pyscf.fci import cistring
    strs = cistring.make_strings(range(norb), nocc)
    dim_full = len(strs)
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
    W = np.asarray(ref["ci"]).reshape(dim_full, dim_full) ** 2
    W /= W.sum()

    pos_id = R.positions_from(np.arange(norb))

    # =========================================================== C1
    banner("C1 -- all 120 anchor triples at fixed identity ordering (remove selection bias)")

    ranking = pd.read_csv(B2_RANKING_CSV)
    ranking["triple"] = ranking["triple"].apply(lambda s: ast.literal_eval(s))
    all_triples = list(itertools.combinations(range(norb), 3))
    assert len(all_triples) == 120 == len(ranking)

    prev = pd.read_csv(B2_SAMPLED_CSV)
    prev = prev[prev.status == "OK"].copy()
    prev["triple_tuple"] = prev["triple"].apply(lambda s: tuple(int(c) for c in str(s).zfill(3)))
    done_triples = {t: row for t, row in zip(prev["triple_tuple"], prev.to_dict("records"))}
    print(f"  {len(done_triples)} triples already sampled (B2), reusing as-is")

    todo = [t for t in all_triples if t not in done_triples]
    print(f"  {len(todo)} new evaluations to run")

    c1_rows = []
    for t, rec in done_triples.items():
        c1_rows.append(dict(triple=t, err_mHa=rec["err_mHa"], captured=rec["captured"],
                            retained_J_oppspin=rec["retained_J_oppspin"], source="B2_reused"))

    t0 = time.time()
    for i, triple in enumerate(todo, 1):
        tag = f"c1_{'-'.join(map(str, triple))}"
        row = evaluate(R, t1L, t2L, pos_id, norb, nelec, nocc, hf, fcidump_path,
                      E_CASCI, b2i, W, seed=2026, anchor_orbitals=triple, tag=tag)
        rj_ss, rj_os = R.retained_J_split_of(pos_id, Jaa, Jab, anchor_orbitals=triple)
        c1_rows.append(dict(triple=triple, err_mHa=row["err_mHa"], captured=row["captured"],
                            retained_J_oppspin=rj_os, source="C1_new", status=row["status"]))
        pd.DataFrame(c1_rows).to_csv(OUTDIR / "c1_all120_identity.csv", index=False)
        el = time.time() - t0
        print(f"[C1 {i}/{len(todo)}] {triple}  err={row['err_mHa']:.2f} mHa  "
              f"status={row['status']}  eta={el/i*(len(todo)-i)/60:.1f}m")

    c1_df = pd.DataFrame(c1_rows)
    c1_df.to_csv(OUTDIR / "c1_all120_identity.csv", index=False)
    c1_ok = c1_df[c1_df.err_mHa.notna()].reset_index(drop=True)
    assert len(c1_ok) == 120, f"expected 120 OK triples, got {len(c1_ok)}"

    banner("C1 -- results")
    print(f"  err_mHa range over all 120: {c1_ok.err_mHa.min():.2f} - {c1_ok.err_mHa.max():.2f} mHa "
          f"(range {c1_ok.err_mHa.max()-c1_ok.err_mHa.min():.2f})")

    rho_os_full = spearmanr(c1_ok.retained_J_oppspin, c1_ok.err_mHa)
    rho_cap_full = spearmanr(c1_ok.captured, c1_ok.err_mHa)
    print(f"  rho(retained_J_oppspin, err_mHa), full 120: {rho_os_full.statistic:+.3f}  "
          f"p={rho_os_full.pvalue:.2e}")
    print(f"  rho(captured, err_mHa), full 120:           {rho_cap_full.statistic:+.3f}  "
          f"p={rho_cap_full.pvalue:.2e}")

    biased = c1_ok[c1_ok.source == "B2_reused"]
    rho_os_biased = spearmanr(biased.retained_J_oppspin, biased.err_mHa)
    rho_cap_biased = spearmanr(biased.captured, biased.err_mHa)
    print(f"\n  Same rhos on the ORIGINAL biased 30 (top10/bottom10/random10 by ranking):")
    print(f"  rho(retained_J_oppspin, err_mHa), biased-30: {rho_os_biased.statistic:+.3f}  "
          f"p={rho_os_biased.pvalue:.2e}")
    print(f"  rho(captured, err_mHa), biased-30:           {rho_cap_biased.statistic:+.3f}  "
          f"p={rho_cap_biased.pvalue:.2e}")
    print(f"\n  inflation: rho(oppspin) {rho_os_biased.statistic:+.3f} -> {rho_os_full.statistic:+.3f} "
          f"({'inflated' if abs(rho_os_biased.statistic) > abs(rho_os_full.statistic) else 'not inflated'})")
    print(f"  inflation: rho(captured) {rho_cap_biased.statistic:+.3f} -> {rho_cap_full.statistic:+.3f} "
          f"({'inflated' if abs(rho_cap_biased.statistic) > abs(rho_cap_full.statistic) else 'not inflated'})")

    c1_by_err = c1_ok.sort_values("err_mHa").reset_index(drop=True)
    c1_by_err["rank_err"] = np.arange(1, len(c1_by_err) + 1)
    top_pick_idx = c1_ok["retained_J_oppspin"].idxmax()
    top_pick_triple = c1_ok.loc[top_pick_idx, "triple"]
    top_pick_err = c1_ok.loc[top_pick_idx, "err_mHa"]
    actual_rank = int(c1_by_err[c1_by_err.triple.apply(lambda t: t == top_pick_triple)]["rank_err"].iloc[0])
    true_best_err = c1_ok.err_mHa.min()
    regret = float(top_pick_err - true_best_err)
    print(f"\n  best triple by retained_J_oppspin: {top_pick_triple}  err_mHa={top_pick_err:.2f}")
    print(f"  its ACTUAL rank by err_mHa: {actual_rank} / 120")
    print(f"  true best of 120: err_mHa={true_best_err:.2f}")
    print(f"  selection regret (picking by retained_J_oppspin): {regret:.2f} mHa")

    # =========================================================== C2
    banner("C2 -- transferability across same-spin orderings")
    perm_by_ordering = pd.read_csv(BASELINE_CSV).groupby("ordering")["permutation"].first()
    orderings = {}
    for name in C2_ORDERINGS:
        perm = R.parse_permutation(perm_by_ordering[name], norb)
        orderings[name] = perm
    print(f"  orderings: {list(orderings.keys())} (+ identity from C1)")
    for name, perm in orderings.items():
        print(f"    {name:<16} perm={''.join(map(str, perm))}")

    rng = np.random.default_rng(C2_RNG_SEED)
    idx = rng.choice(len(all_triples), size=C2_N_SHARED, replace=False)
    shared_triples = [all_triples[i] for i in idx]
    print(f"  {len(shared_triples)} triples drawn uniformly (rng seed {C2_RNG_SEED}), "
          f"shared across all 3 new orderings")

    c2_rows = []
    n_total = len(orderings) * len(shared_triples)
    i_done = 0
    t0 = time.time()
    for name, perm in orderings.items():
        pos = R.positions_from(perm)
        for triple in shared_triples:
            tag = f"c2_{name}_{'-'.join(map(str, triple))}"
            row = evaluate(R, t1L, t2L, pos, norb, nelec, nocc, hf, fcidump_path,
                          E_CASCI, b2i, W, seed=2026, anchor_orbitals=triple, tag=tag)
            rj_ss, rj_os = R.retained_J_split_of(pos, Jaa, Jab, anchor_orbitals=triple)
            c2_rows.append(dict(ordering=name, triple=triple, err_mHa=row["err_mHa"],
                                captured=row["captured"], retained_J_oppspin=rj_os,
                                status=row["status"]))
            pd.DataFrame(c2_rows).to_csv(OUTDIR / "c2_transfer.csv", index=False)
            i_done += 1
            el = time.time() - t0
            print(f"[C2 {i_done}/{n_total}] {name:<16} {triple}  err={row['err_mHa']:.2f} mHa  "
                  f"eta={el/i_done*(n_total-i_done)/60:.1f}m")

    c2_df = pd.DataFrame(c2_rows)
    c2_df.to_csv(OUTDIR / "c2_transfer.csv", index=False)

    # identity's values on the SAME shared triples, from C1 (no new evals)
    c1_lookup = {t: e for t, e in zip(c1_ok.triple, c1_ok.err_mHa)}
    identity_shared = pd.DataFrame(
        dict(ordering="identity", triple=t, err_mHa=c1_lookup[t]) for t in shared_triples)

    banner("C2 -- per-ordering results")
    best_triples = {}
    for name in C2_ORDERINGS:
        sub = c2_df[(c2_df.ordering == name) & (c2_df.status == "OK")]
        rho = spearmanr(sub.retained_J_oppspin, sub.err_mHa)
        pick_idx = sub["retained_J_oppspin"].idxmax()
        pick_err = sub.loc[pick_idx, "err_mHa"]
        pick_triple = sub.loc[pick_idx, "triple"]
        best_idx = sub["err_mHa"].idxmin()
        best_err = sub.loc[best_idx, "err_mHa"]
        best_triple = sub.loc[best_idx, "triple"]
        reg = float(pick_err - best_err)
        best_triples[name] = best_triple
        print(f"  {name:<16} range=[{sub.err_mHa.min():.2f}, {sub.err_mHa.max():.2f}] "
              f"({sub.err_mHa.max()-sub.err_mHa.min():.2f})  "
              f"rho(oppspin,err)={rho.statistic:+.3f} p={rho.pvalue:.2e}  "
              f"pick={pick_triple} regret={reg:.2f}  best={best_triple}")
    id_sub = identity_shared
    id_best_idx = id_sub["err_mHa"].idxmin()
    best_triples["identity"] = id_sub.loc[id_best_idx, "triple"]
    print(f"  {'identity':<16} (from C1, restricted to shared 40)  best={best_triples['identity']}")

    print(f"\n  best triple per ordering: {best_triples}")
    all_same = len(set(best_triples.values())) == 1
    print(f"  same best triple for all four? {all_same}")

    banner("C2 -- cross-ordering Spearman correlation (err_mHa over shared triples)")
    pivot = {}
    for name in C2_ORDERINGS:
        sub = c2_df[(c2_df.ordering == name)].set_index(
            c2_df[(c2_df.ordering == name)]["triple"].apply(str))["err_mHa"]
        pivot[name] = sub
    pivot["identity"] = identity_shared.set_index(identity_shared["triple"].apply(str))["err_mHa"]
    pivot_df = pd.DataFrame(pivot)
    names4 = ["identity"] + C2_ORDERINGS
    print(f"{'':<18}" + "".join(f"{n:>18}" for n in names4))
    corr_pairs = {}
    for a in names4:
        line = f"{a:<18}"
        for b in names4:
            r = spearmanr(pivot_df[a], pivot_df[b]).statistic
            line += f"{r:>+18.3f}"
            if a < b:
                corr_pairs[f"{a}_vs_{b}"] = float(r)
        print(line)

    # --------------------------------------------------------------- HEADLINE
    banner("HEADLINE")
    mean_pairwise = float(np.mean(list(corr_pairs.values())))
    generalizes = mean_pairwise > 0.3
    print(f"C1: rho(retained_J_oppspin, err) full-120 = {rho_os_full.statistic:+.3f} "
          f"(biased-30 was {rho_os_biased.statistic:+.3f})")
    print(f"C2: mean pairwise cross-ordering rho of err_mHa over shared triples = {mean_pairwise:+.3f}")
    if generalizes:
        print("VERDICT: retained_J_oppspin's predictive signal and the anchor effect itself "
              "APPEAR TO GENERALIZE beyond identity - same-spin orderings largely agree on "
              "which anchors are good/bad.")
    else:
        print("VERDICT: retained_J_oppspin does NOT clearly generalize beyond identity - "
              "cross-ordering agreement on which anchors are good/bad is weak, so it is "
              "ordering-specific, not a general anchor-selection rule.")

    # ------------------------------------------------------------- metadata
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="C1_C2_hardening", git_commit=git_commit, shots=SHOTS, seed=2026,
        c2_rng_seed=C2_RNG_SEED, c2_n_shared=C2_N_SHARED, c2_orderings=C2_ORDERINGS,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        b2_sampled_csv_sha256=sha256_of(B2_SAMPLED_CSV),
        c1_n_reused=len(done_triples), c1_n_new=len(todo),
        c1_rho_oppspin_full120=float(rho_os_full.statistic), c1_p_oppspin_full120=float(rho_os_full.pvalue),
        c1_rho_captured_full120=float(rho_cap_full.statistic), c1_p_captured_full120=float(rho_cap_full.pvalue),
        c1_rho_oppspin_biased30=float(rho_os_biased.statistic), c1_p_oppspin_biased30=float(rho_os_biased.pvalue),
        c1_rho_captured_biased30=float(rho_cap_biased.statistic), c1_p_captured_biased30=float(rho_cap_biased.pvalue),
        c1_top_pick_triple=str(top_pick_triple), c1_top_pick_actual_rank=actual_rank,
        c1_selection_regret=regret,
        c2_best_triples={k: str(v) for k, v in best_triples.items()},
        c2_all_same_best=all_same, c2_pairwise_rho=corr_pairs,
        c2_mean_pairwise_rho=mean_pairwise, generalizes=generalizes,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "c1_c2_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'c1_all120_identity.csv'}")
    print(f"[out] {OUTDIR / 'c2_transfer.csv'}")
    print(f"[out] {OUTDIR / 'c1_c2_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
