#!/usr/bin/env python3
"""
experiments/budget_transfer.py
=================================

E2 -- budget transfer screening. At fixed identity same-spin ordering,
evaluate all 120 anchor triples at reduced subspace budgets (n_dets in
{5, 8, 10}) and compare their ranking against the existing 15-dim reference
(c1_all120_identity.csv). Then repeat n_dets=8 screening for physical and
rand007's 40 sampled triples (reusing c2_transfer.csv as the 15-dim
reference).

Each triple is sampled ONCE per (ordering, seed) - the same 2e6-shot Aer
run gives the full alpha/beta count distribution, from which top-k
selection at any budget is a cheap downstream step. Re-sampling per budget
would be wasteful and unrealistic: a real screening workflow reuses one
sample and only re-runs the classical top_dets+sbd step at each candidate
budget. This reuses run_ordering_pipeline.py's sample_bitstrings/top_dets/
run_sbd directly - nothing reimplemented, just re-invoked at multiple
budgets from one sample.
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

OUTDIR = Path(__file__).resolve().parent / "outputs" / "budget_transfer"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
ANCHOR_DIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
C1_CSV = ANCHOR_DIR / "c1_all120_identity.csv"
C2_CSV = ANCHOR_DIR / "c2_transfer.csv"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

SHOTS = 2_000_000
SEED = 2026
BUDGETS_IDENTITY = [5, 8, 10, 15]  # 15 re-sampled here too, for a same-machine timing comparison
BUDGETS_TRANSFER = [8]
TRUE_BEST_15DIM = 224.60
REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s)
    REPORT.append(s)


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


def evaluate_multi_budget(R, op, norb, nelec, nocc, hf, fcidump_path, E_CASCI, b2i, W,
                          seed, budgets, tag):
    """Sample once, derive results at every requested budget from that one sample."""
    a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, SHOTS, seed)
    results = {}
    for budget in budgets:
        t0 = time.time()
        a_sel, n_uniq_a = R.top_dets(a_c, budget, hf)
        b_sel, n_uniq_b = R.top_dets(b_c, budget, hf)
        dim_a, dim_b = len(a_sel), len(b_sel)
        if dim_a < budget or dim_b < budget:
            results[budget] = dict(status="SUPPORT_COLLAPSE", err_mHa=float("nan"),
                                   captured=float("nan"), wall_s=time.time() - t0)
            continue
        adet_path = OUTDIR / f"_{tag}_b{budget}_a.txt"
        bdet_path = OUTDIR / f"_{tag}_b{budget}_b.txt"
        adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
        bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
        energy = R.run_sbd(str(fcidump_path), str(adet_path), str(bdet_path), norb)
        wall_s = time.time() - t0
        err_mha = (energy - E_CASCI) * 1000.0
        ia = [b2i[d] for d in a_sel]
        ib = [b2i[d] for d in b_sel]
        captured = float(W[np.ix_(ia, ib)].sum())
        results[budget] = dict(status="OK", err_mHa=err_mha, captured=captured, wall_s=wall_s,
                               dim=dim_a * dim_b)
    return results, depth


def regret_report(name, ranking_col, ref_err_col, df, true_best):
    """Top-1/3/5 by ranking_col (ascending err -> lower is better, so we rank by err
    ASCENDING for the reduced-budget quantity itself), report best REFERENCE err among
    them minus the true best reference err."""
    ordered = df.sort_values(ranking_col)
    lines = []
    for k in (1, 3, 5):
        top_k = ordered.head(k)
        best_ref_among = top_k[ref_err_col].min()
        regret = float(best_ref_among - true_best)
        lines.append((k, best_ref_among, regret))
    return lines


def main() -> int:
    banner("E2 -- BUDGET TRANSFER SCREENING")
    import unified_run as U
    import run_ordering_pipeline as R

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at {R.CFG['sbd_bin']}")
    for p in (C1_CSV, C2_CSV, BASELINE_CSV):
        if not p.exists():
            sys.exit(f"FATAL: required input missing: {p}")

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

    c1 = pd.read_csv(C1_CSV)
    c1["triple"] = c1["triple"].apply(parse_triple)
    c2 = pd.read_csv(C2_CSV)
    c2["triple"] = c2["triple"].apply(parse_triple)
    c2 = c2[c2.ordering != "physical_reverse"].reset_index(drop=True)
    perm_by_ordering = pd.read_csv(BASELINE_CSV).groupby("ordering")["permutation"].first()

    pos_id = R.positions_from(np.arange(norb))

    # =========================================================== identity: 120 x {5,8,10,15}
    banner("Identity: all 120 triples at budgets {5, 8, 10, 15}")
    all_triples = list(itertools.combinations(range(norb), 3))
    id_rows = []
    t0 = time.time()
    for i, triple in enumerate(all_triples, 1):
        pairs = R.interaction_pairs_for(pos_id, anchor_orbitals=triple)
        op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
        results, depth = evaluate_multi_budget(R, op, norb, nelec, nocc, hf, fcidump_path,
                                               E_CASCI, b2i, W, SEED, BUDGETS_IDENTITY,
                                               tag=f"id_{'-'.join(map(str, triple))}")
        for budget, res in results.items():
            id_rows.append(dict(triple=triple, budget=budget, **res))
        pd.DataFrame(id_rows).to_csv(OUTDIR / "e2_identity_multibudget.csv", index=False)
        el = time.time() - t0
        b15 = results[15]["err_mHa"] if results[15]["status"] == "OK" else float("nan")
        print(f"[{i}/120] {triple}  err@15={b15:.2f}  eta={el/i*(120-i)/60:.1f}m")

    id_df = pd.DataFrame(id_rows)
    id_df.to_csv(OUTDIR / "e2_identity_multibudget.csv", index=False)

    # merge in the ORIGINAL 15-dim reference from C1 (independently sampled earlier)
    c1_ref = c1.set_index(c1.triple.apply(str))["err_mHa"]

    banner("Identity: rho vs 15-dim ranking, regret, timing")
    d_rows = []
    for budget in BUDGETS_IDENTITY:
        sub = id_df[(id_df.budget == budget) & (id_df.status == "OK")].copy()
        sub["ref_err_15dim"] = sub.triple.apply(lambda t: c1_ref.loc[str(t)])
        rho = spearmanr(sub.err_mHa, sub.ref_err_15dim)
        mean_wall = sub.wall_s.mean()
        out(f"\n  budget={budget}  n_ok={len(sub)}  "
            f"rho(budget-err, 15dim-ref-err)={rho.statistic:+.3f} (p={rho.pvalue:.2e})  "
            f"mean_wall_s={mean_wall:.2f}")
        for k, best_ref, regret in regret_report(f"b{budget}", "err_mHa", "ref_err_15dim",
                                                 sub, TRUE_BEST_15DIM):
            out(f"    top-{k} by budget-{budget} ranking: best 15-dim err among them = "
                f"{best_ref:.2f}, regret = {regret:.2f} mHa")
        d_rows.append(dict(budget=budget, n_ok=len(sub), rho=rho.statistic, p=rho.pvalue,
                           mean_wall_s=mean_wall))

    banner("Wall time per evaluation, budget vs 15-dim")
    wall_by_budget = {d["budget"]: d["mean_wall_s"] for d in d_rows}
    wall15 = wall_by_budget.get(15, float("nan"))
    for budget in BUDGETS_IDENTITY:
        w = wall_by_budget[budget]
        out(f"  budget={budget:<3}  mean_wall_s={w:.3f}  "
            f"ratio_to_15dim={w/wall15:.3f}" if wall15 else "")

    # =========================================================== transfer: physical, rand007 @ 8
    banner("Transfer screening: n_dets=8 for physical and rand007 (40 shared triples)")
    tr_rows = []
    for name in ("physical", "rand007"):
        perm = R.parse_permutation(perm_by_ordering[name], norb)
        pos = R.positions_from(perm)
        triples_this = list(c2[c2.ordering == name].triple)
        t0 = time.time()
        for i, triple in enumerate(triples_this, 1):
            pairs = R.interaction_pairs_for(pos, anchor_orbitals=triple)
            op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
            results, depth = evaluate_multi_budget(R, op, norb, nelec, nocc, hf, fcidump_path,
                                                   E_CASCI, b2i, W, SEED, BUDGETS_TRANSFER,
                                                   tag=f"{name}_{'-'.join(map(str, triple))}")
            res8 = results[8]
            tr_rows.append(dict(ordering=name, triple=triple, budget=8, **res8))
            pd.DataFrame(tr_rows).to_csv(OUTDIR / "e2_transfer_budget8.csv", index=False)
            print(f"[{name} {i}/{len(triples_this)}] {triple}  err@8={res8.get('err_mHa', float('nan')):.2f}")

    tr_df = pd.DataFrame(tr_rows)
    tr_df.to_csv(OUTDIR / "e2_transfer_budget8.csv", index=False)

    banner("Transfer screening results")
    top3_regrets = {}
    for name in ("physical", "rand007"):
        sub = tr_df[(tr_df.ordering == name) & (tr_df.status == "OK")].copy()
        ref15 = c2[c2.ordering == name].set_index(c2[c2.ordering == name].triple.apply(str))["err_mHa"]
        sub["ref_err_15dim"] = sub.triple.apply(lambda t: ref15.loc[str(t)])
        rho = spearmanr(sub.err_mHa, sub.ref_err_15dim)
        true_best_ord = ref15.min()
        out(f"\n  {name}: n_ok={len(sub)}  rho(budget8-err, 15dim-ref-err)={rho.statistic:+.3f} "
            f"(p={rho.pvalue:.2e})  true_best_15dim={true_best_ord:.2f}")
        for k, best_ref, regret in regret_report(name, "err_mHa", "ref_err_15dim", sub, true_best_ord):
            out(f"    top-{k} by budget-8 ranking: best 15-dim err among them = {best_ref:.2f}, "
                f"regret = {regret:.2f} mHa")
            if k == 3:
                top3_regrets[name] = regret

    # =========================================================== HEADLINE
    banner("HEADLINE")
    id_sub8 = id_df[(id_df.budget == 8) & (id_df.status == "OK")].copy()
    id_sub8["ref_err_15dim"] = id_sub8.triple.apply(lambda t: c1_ref.loc[str(t)])
    id_regret_table = {k: r for k, _, r in
                       regret_report("id8", "err_mHa", "ref_err_15dim", id_sub8, TRUE_BEST_15DIM)}
    id_top3_regret = id_regret_table[3]
    recovers_top3 = id_top3_regret < 20.0 and all(v < 20.0 for v in top3_regrets.values())
    out(f"8x8 screen top-3 regret: identity={id_top3_regret:.2f} mHa, "
        f"physical={top3_regrets.get('physical', float('nan')):.2f} mHa, "
        f"rand007={top3_regrets.get('rand007', float('nan')):.2f} mHa")
    out(f"Does an 8x8 screen recover a top-3-at-15x15 triple? "
        f"{'YES (all regrets small)' if recovers_top3 else 'PARTIAL/NO - see per-ordering numbers above'}")
    works_all_three = id_top3_regret < 50 and all(v < 50 for v in top3_regrets.values())
    out(f"Works at all three orderings (unlike retained_J_oppspin, which failed at physical)? "
        f"{works_all_three}")
    cost_ratio = wall_by_budget[15] / wall_by_budget[8] if wall_by_budget.get(8) else float("nan")
    out(f"Cost ratio (wall time per sbd call, 15-dim / 8-dim): {cost_ratio:.2f}x cheaper at 8x8")
    out(f"(Sampling cost is shared/amortised across budgets - see module docstring; this ratio "
        f"is specifically the classical diagonalisation cost, which is what actually scales "
        f"with budget in a real screen.)")

    # ------------------------------------------------------------- save
    report_path = OUTDIR / "e2_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="E2_budget_transfer", git_commit=git_commit, shots=SHOTS, seed=SEED,
        budgets_identity=BUDGETS_IDENTITY, budgets_transfer=BUDGETS_TRANSFER,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        c1_csv_sha256=sha256_of(C1_CSV), c2_csv_sha256=sha256_of(C2_CSV),
        d_rows=d_rows, top3_regrets=top3_regrets, id_top3_regret=id_top3_regret,
        cost_ratio_15_over_8=cost_ratio,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "e2_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'e2_identity_multibudget.csv'}")
    print(f"[out] {OUTDIR / 'e2_transfer_budget8.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'e2_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
