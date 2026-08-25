#!/usr/bin/env python3
"""
experiments/g1_lite.py
=========================

G1-lite -- do the same-spin-ordering lever and the anchor-selection lever
interact? For 8 same-spin orderings (best-2/median-2/worst-2 from the H10
baseline, plus identity and physical), sample the SAME 40 anchor triples
(rng seed 20260825003, uniform over the 120) and compare best-of-40 against
each ordering's own baseline and floor.

Reuses cached evaluations wherever the exact (ordering, triple) pair was
already sampled: identity's C1 covers all 120 triples (full reuse);
physical/rand007's C2 sampled a DIFFERENT 40-triple set (rng seed
20260825002) with partial overlap against this run's 40 (computed exactly,
not assumed); floor values for all orderings already exist in E1/F1's
outputs. Reuses run_ordering_pipeline.py + anchor_decomposition.py's
evaluate() for anything not already covered - nothing reimplemented.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "g1_lite"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
ANCHOR_DIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
C1_CSV = ANCHOR_DIR / "c1_all120_identity.csv"
C2_CSV = ANCHOR_DIR / "c2_transfer.csv"
E1_META = Path(__file__).resolve().parent / "outputs" / "budget_transfer" / "e1_metadata.json"
F1C_CSV = Path(__file__).resolve().parent / "outputs" / "floor_generalization" / "f1c_floor_vs_default_50random.csv"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

SHOTS = 2_000_000
SEED = 2026
RNG_SEED = 20260825003
N_TRIPLES = 40
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


def main() -> int:
    banner("G1-LITE -- DO THE LEVERS INTERACT?")
    import unified_run as U
    import run_ordering_pipeline as R
    from anchor_decomposition import evaluate

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at {R.CFG['sbd_bin']}")
    for p in (C1_CSV, C2_CSV, E1_META, F1C_CSV, BASELINE_CSV):
        if not p.exists():
            sys.exit(f"FATAL: required input missing: {p}")

    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(CACHEDIR))
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    fcidump_path = Path(ref["fcidump_path"])
    E_CASCI = ref["E_CASCI"]
    hf = R.hf_bitstring(norb, nocc)

    from pyscf.fci import cistring
    strs = cistring.make_strings(range(norb), nocc)
    dim_full = len(strs)
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
    W = np.asarray(ref["ci"]).reshape(dim_full, dim_full) ** 2
    W /= W.sum()

    base = pd.read_csv(BASELINE_CSV)
    base_ok = base[base.status == "OK"]
    perm_by_ordering = base.groupby("ordering")["permutation"].first()
    default_mean = base_ok.groupby("ordering")["err_mHa"].mean()

    # ------------------------------------------------------------- pick the 8 orderings
    rnd = base_ok[base_ok.ordering.str.match(r"^rand\d+$")]
    per_ord = rnd.groupby("ordering")["err_mHa"].mean().sort_values()
    best2 = list(per_ord.index[:2])
    worst2 = list(per_ord.index[-2:])
    median = per_ord.median()
    dist = (per_ord - median).abs().sort_values()
    median2 = list(dist.index[:2])
    orderings = ["identity", "physical"] + best2 + median2 + worst2
    out(f"8 orderings: {orderings}")
    out(f"  best2={best2}  median2={median2}  worst2={worst2}")
    for name in orderings:
        out(f"  {name:<10} baseline_err={default_mean[name]:.2f}")

    # ------------------------------------------------------------- shared 40 triples
    all_triples = list(itertools.combinations(range(norb), 3))
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(all_triples), size=N_TRIPLES, replace=False)
    shared40 = [all_triples[i] for i in idx]
    out(f"\n{len(shared40)} shared anchor triples drawn (rng seed {RNG_SEED})")

    # ------------------------------------------------------------- existing coverage
    c1 = pd.read_csv(C1_CSV)
    c1["triple"] = c1.triple.apply(parse_triple)
    c1_lookup = c1.set_index(c1.triple.apply(str))["err_mHa"]

    c2 = pd.read_csv(C2_CSV)
    c2["triple"] = c2.triple.apply(parse_triple)
    c2_lookup = {name: c2[c2.ordering == name].set_index(c2[c2.ordering == name].triple.apply(str))["err_mHa"]
                for name in ("physical", "rand007")}

    e1_meta = json.loads(E1_META.read_text())
    floor_named = dict(e1_meta["floor_by_ordering"])
    f1c = pd.read_csv(F1C_CSV)
    floor_rand = f1c.set_index("ordering")["floor_err"].to_dict()
    floor_by_ordering = {**floor_named, **floor_rand}
    missing_floor = [n for n in orderings if n not in floor_by_ordering]
    out(f"floors available for all 8 from E1/F1c reuse: {not missing_floor} "
        f"(missing: {missing_floor})")

    # ------------------------------------------------------------- run (with reuse)
    all_rows = []
    t0 = time.time()
    n_new_total = 0
    for name in orderings:
        perm = np.arange(norb) if name == "identity" else R.parse_permutation(perm_by_ordering[name], norb)
        pos = R.positions_from(perm)
        reused = 0
        for triple in shared40:
            key = str(triple)
            if name == "identity" and key in c1_lookup.index:
                all_rows.append(dict(ordering=name, triple=triple, err_mHa=c1_lookup.loc[key],
                                     source="reused_c1"))
                reused += 1
                continue
            if name in c2_lookup and key in c2_lookup[name].index:
                all_rows.append(dict(ordering=name, triple=triple, err_mHa=c2_lookup[name].loc[key],
                                     source="reused_c2"))
                reused += 1
                continue
            tag = f"g1_{name}_{'-'.join(map(str, triple))}"
            row = evaluate(R, t1L, t2L, pos, norb, nelec, nocc, hf, fcidump_path, E_CASCI,
                          b2i, W, seed=SEED, anchor_orbitals=triple, tag=tag)
            all_rows.append(dict(ordering=name, triple=triple, err_mHa=row["err_mHa"],
                                 source="new", status=row["status"]))
            n_new_total += 1
            pd.DataFrame(all_rows).to_csv(OUTDIR / "g1_all.csv", index=False)
            el = time.time() - t0
            print(f"[{name}] new eval {n_new_total} total  triple={triple}  "
                  f"err={row['err_mHa']:.2f}  elapsed={el/60:.1f}m")
        print(f"  {name}: {reused}/40 reused, {40-reused}/40 newly sampled")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTDIR / "g1_all.csv", index=False)
    df["err_mHa"] = df["err_mHa"].astype(float)

    # =========================================================== per-ordering report
    banner("Per-ordering report")
    summary_rows = []
    for name in orderings:
        sub = df[df.ordering == name]
        best_idx = sub.err_mHa.idxmin()
        best_err = sub.loc[best_idx, "err_mHa"]
        best_triple = sub.loc[best_idx, "triple"]
        summary_rows.append(dict(ordering=name, baseline=float(default_mean[name]),
                                 floor=float(floor_by_ordering[name]), best_of_40=float(best_err),
                                 best_triple=str(best_triple)))
        out(f"  {name:<10} baseline={default_mean[name]:7.2f}  floor={floor_by_ordering[name]:7.2f}  "
            f"best_of_40={best_err:7.2f}  winning_triple={best_triple}")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(OUTDIR / "g1_summary.csv", index=False)

    # =========================================================== analysis
    banner("Analysis")
    rho = spearmanr(summ.baseline, summ.best_of_40)
    out(f"  rho(baseline_err, best_of_40_err) across 8 orderings = {rho.statistic:+.3f}  "
        f"p={rho.pvalue:.2e}")

    global_best_idx = summ.best_of_40.idxmin()
    global_best = summ.loc[global_best_idx]
    baseline_rank_pct = 100.0 * (summ.baseline > global_best.baseline).mean()
    out(f"\n  global best (ordering, anchor) = ({global_best.ordering}, {global_best.best_triple}), "
        f"err_mHa={global_best.best_of_40:.2f}")
    out(f"  that ordering's baseline percentile among the 8: {baseline_rank_pct:.1f}% "
        f"(baseline_err={global_best.baseline:.2f})")

    spread_40 = float(summ.best_of_40.max() - summ.best_of_40.min())
    out(f"\n  spread of best-of-40 across the 8 orderings: {spread_40:.2f} mHa "
        f"(vs 286.23 mHa baseline spread across all 50 random orderings)")

    win_counts = {}
    for _, r in summ.iterrows():
        win_counts.setdefault(r.best_triple, []).append(r.ordering)
    shared_wins = {t: os for t, os in win_counts.items() if len(os) > 1}
    out(f"\n  triples that win (best-of-40) for more than one ordering: {len(shared_wins)}")
    for t, os in shared_wins.items():
        out(f"    {t}: wins for {os}")

    # =========================================================== HEADLINE
    banner("HEADLINE")
    matters = spread_40 > 20.0
    out(f"After anchor optimisation, does same-spin ordering still matter? "
        f"spread of best-of-40 = {spread_40:.2f} mHa -> {'YES' if matters else 'little residual effect'}")
    predicts = abs(rho.statistic) >= 0.5 and rho.pvalue < 0.1
    out(f"Does baseline ranking predict post-optimisation ranking? "
        f"rho={rho.statistic:+.3f} (p={rho.pvalue:.2e}) -> {'YES' if predicts else 'NOT clearly'}")

    # ------------------------------------------------------------- save
    report_path = OUTDIR / "g1_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="G1_lite", git_commit=git_commit, shots=SHOTS, seed=SEED, rng_seed=RNG_SEED,
        n_triples=N_TRIPLES, orderings=orderings, n_new_evaluations=n_new_total,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        fcidump_sha256=sha256_of(fcidump_path),
        rho_baseline_best40=float(rho.statistic), p_rho=float(rho.pvalue),
        spread_best40=spread_40, global_best_ordering=str(global_best.ordering),
        global_best_triple=str(global_best.best_triple), global_best_err=float(global_best.best_of_40),
        shared_wins={str(k): v for k, v in shared_wins.items()},
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "g1_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'g1_all.csv'}")
    print(f"[out] {OUTDIR / 'g1_summary.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'g1_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
