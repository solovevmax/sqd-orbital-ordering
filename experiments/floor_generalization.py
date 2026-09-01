#!/usr/bin/env python3
"""
experiments/floor_generalization.py
======================================

F1 -- is the "no opposite-spin coupling" floor sometimes BETTER than the
default p%4==0 anchor mask? Established in E1 for identity/physical/rand007;
here extended to all 50 baseline random orderings (F1c/F1d), plus a
like-for-like re-verification (F1a) and a per-ordering breakdown (F1b).

Reuses run_ordering_pipeline.py + anchor_decomposition.py's evaluate() -
nothing reimplemented. The "no opposite-spin" operator is built the same way
as E1's control: interaction_pairs_for(pos, anchor_orbitals=()).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "floor_generalization"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
ANCHOR_DIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
C1_CSV = ANCHOR_DIR / "c1_all120_identity.csv"
C2_CSV = ANCHOR_DIR / "c2_transfer.csv"
E1_META = Path(__file__).resolve().parent / "outputs" / "budget_transfer" / "e1_metadata.json"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

SHOTS = 2_000_000
SEED = 2026
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
    banner("F1 -- IS THE OPPOSITE-SPIN MASK A NET LIABILITY AT SOME ORDERINGS?")
    import unified_run as U
    import run_ordering_pipeline as R
    from anchor_decomposition import evaluate

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at {R.CFG['sbd_bin']}")
    for p in (C1_CSV, C2_CSV, E1_META, BASELINE_CSV):
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

    c1 = pd.read_csv(C1_CSV)
    c1["triple"] = c1["triple"].apply(parse_triple)
    c2 = pd.read_csv(C2_CSV)
    c2["triple"] = c2["triple"].apply(parse_triple)
    c2 = c2[c2.ordering != "physical_reverse"].reset_index(drop=True)
    e1_meta = json.loads(E1_META.read_text())
    floor_named = e1_meta["floor_by_ordering"]  # identity/physical/rand007

    base = pd.read_csv(BASELINE_CSV)
    base_ok = base[base.status == "OK"]
    perm_by_ordering = base.groupby("ordering")["permutation"].first()
    default_seed2026 = base_ok[base_ok.seed == 2026].set_index("ordering")["err_mHa"]
    default_mean = base_ok.groupby("ordering")["err_mHa"].mean()

    # =========================================================== F1a
    banner("F1a -- like-for-like re-verification at physical")
    pos_phys = R.positions_from(R.parse_permutation(perm_by_ordering["physical"], norb))
    row_floor = evaluate(R, t1L, t2L, pos_phys, norb, nelec, nocc, hf, fcidump_path, E_CASCI,
                        b2i, W, seed=SEED, anchor_orbitals=(), tag="f1a_physical_floor_recheck")
    row_default = evaluate(R, t1L, t2L, pos_phys, norb, nelec, nocc, hf, fcidump_path, E_CASCI,
                          b2i, W, seed=SEED, anchor_offset=0, tag="f1a_physical_default_recheck")
    out(f"  floor (re-run):   err_mHa = {row_floor['err_mHa']:.2f}  "
        f"(E1's stored value: {floor_named['physical']:.2f})")
    out(f"  default (re-run): err_mHa = {row_default['err_mHa']:.2f}  "
        f"(baseline CSV seed=2026: {default_seed2026['physical']:.2f})")
    floor_matches = abs(row_floor["err_mHa"] - floor_named["physical"]) < 1e-6
    default_matches = abs(row_default["err_mHa"] - default_seed2026["physical"]) < 1e-6
    out(f"  floor reproduces exactly: {floor_matches}   default reproduces exactly: {default_matches}")
    gap = row_default["err_mHa"] - row_floor["err_mHa"]
    out(f"  default MINUS floor at physical = {gap:+.2f} mHa "
        f"({'default is WORSE than no-opposite-spin' if gap > 0 else 'default is better'})")

    # =========================================================== F1b
    banner("F1b -- per-ordering breakdown (identity, physical, rand007)")
    f1b_rows = []
    for name, df_, n_total in (("identity", c1, 120), ("physical", c2[c2.ordering == "physical"], 40),
                               ("rand007", c2[c2.ordering == "rand007"], 40)):
        floor_e = floor_named[name]
        default_e = default_seed2026[name]
        best_e = df_.err_mHa.min()
        n_worse_than_floor = int((df_.err_mHa > floor_e).sum())
        frac = n_worse_than_floor / n_total
        f1b_rows.append(dict(ordering=name, floor=floor_e, default=default_e, best=best_e,
                             n_total=n_total, n_worse_than_floor=n_worse_than_floor, frac=frac))
        out(f"  {name:<12} floor={floor_e:7.2f}  default={default_e:7.2f}  best={best_e:7.2f}  "
            f"worse-than-floor={n_worse_than_floor}/{n_total} ({100*frac:.1f}%)")

    # =========================================================== F1c
    banner("F1c -- floor for all 50 random baseline permutations (seed 2026, 15-dim)")
    rand_names = sorted(base_ok[base_ok.ordering.str.match(r"^rand\d+$")].ordering.unique())
    assert len(rand_names) == 50, f"expected 50 random orderings, got {len(rand_names)}"
    rows = []
    t0 = time.time()
    for i, name in enumerate(rand_names, 1):
        perm = R.parse_permutation(perm_by_ordering[name], norb)
        pos = R.positions_from(perm)
        row = evaluate(R, t1L, t2L, pos, norb, nelec, nocc, hf, fcidump_path, E_CASCI,
                      b2i, W, seed=SEED, anchor_orbitals=(), tag=f"f1c_{name}_floor")
        default_e = float(default_seed2026[name])
        rows.append(dict(ordering=name, floor_err=row["err_mHa"], default_err=default_e,
                         status=row["status"]))
        pd.DataFrame(rows).to_csv(OUTDIR / "f1c_floor_vs_default_50random.csv", index=False)
        el = time.time() - t0
        print(f"[{i}/50] {name}  floor={row['err_mHa']:.2f}  default={default_e:.2f}  "
              f"eta={el/i*(50-i)/60:.1f}m")

    f1c_df = pd.DataFrame(rows)
    f1c_ok = f1c_df[f1c_df.status == "OK"].copy()
    f1c_ok["gap"] = f1c_ok.floor_err - f1c_ok.default_err  # positive: default beats floor (ab HELPS)
    n_worse = int((f1c_ok.default_err > f1c_ok.floor_err).sum())
    out(f"\n  {len(f1c_ok)}/50 evaluated OK")
    out(f"  default-anchor error WORSE than own floor: {n_worse}/{len(f1c_ok)} "
        f"({100*n_worse/len(f1c_ok):.1f}%)")
    rho_fd = spearmanr(f1c_ok.floor_err, f1c_ok.default_err)
    out(f"  rho(floor_err, default_err) across 50 = {rho_fd.statistic:+.3f}  p={rho_fd.pvalue:.2e}")

    base_mean_rand = default_mean.reindex(f1c_ok.ordering).to_numpy()
    rho_gap = spearmanr(base_mean_rand, f1c_ok.gap)
    out(f"  rho(baseline err_mHa, floor-minus-default gap) = {rho_gap.statistic:+.3f}  "
        f"p={rho_gap.pvalue:.2e}")
    out(f"  (gap = floor_err - default_err; positive means opposite-spin HELPS at that ordering; "
        f"negative means it HURTS. If rho > 0: opposite-spin helps MORE at ALREADY-BAD orderings "
        f"(err and gap both large together, i.e. worse orderings get more benefit from ab-coupling); "
        f"if rho < 0: opposite-spin helps more at GOOD orderings.)")
    if rho_gap.pvalue < 0.05:
        direction = "helps more at bad orderings (larger err_mHa)" if rho_gap.statistic > 0 \
            else "helps more at good orderings (smaller err_mHa)"
        out(f"  -> Significant: opposite-spin coupling {direction}.")
    else:
        out(f"  -> Not significant: no clear relationship between ordering quality and how much "
            f"opposite-spin coupling helps.")

    # =========================================================== F1d
    banner("F1d -- the decision rule: switch opposite-spin OFF when it hurts")
    would_improve = f1c_ok[f1c_ok.gap < 0].copy()  # floor < default means switching off improves
    frac_improve = len(would_improve) / len(f1c_ok)
    mean_improvement = float((-would_improve.gap).mean()) if len(would_improve) else float("nan")
    out(f"  {len(would_improve)}/{len(f1c_ok)} random orderings ({100*frac_improve:.1f}%) would "
        f"improve by switching opposite-spin OFF")
    out(f"  mean improvement among those: {mean_improvement:.2f} mHa")
    if len(would_improve):
        out(f"  range of improvement: {(-would_improve.gap).min():.2f} - {(-would_improve.gap).max():.2f} mHa")

    # =========================================================== HEADLINE
    banner("HEADLINE")
    liability_frac = frac_improve
    out(f"Opposite-spin locality mask is a net liability (default worse than no-ab-at-all) at "
        f"{100*liability_frac:.1f}% of the 50 random baseline orderings.")
    if liability_frac >= 0.10:
        out("VERDICT: YES - this is a meaningful fraction, not a rare edge case. The default "
            "p%4==0 anchor placement should not be assumed beneficial without checking the "
            "cheap floor control first.")
    else:
        out("VERDICT: a minority effect - the opposite-spin mask helps at most orderings, with "
            "occasional exceptions.")

    # ------------------------------------------------------------- save
    report_path = OUTDIR / "f1_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    pd.DataFrame(f1b_rows).to_csv(OUTDIR / "f1b_named_orderings.csv", index=False)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="F1_floor_generalization", git_commit=git_commit, shots=SHOTS, seed=SEED,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        fcidump_sha256=sha256_of(fcidump_path),
        f1a_floor_matches=bool(floor_matches), f1a_default_matches=bool(default_matches),
        f1a_gap_physical=float(gap),
        f1b=f1b_rows,
        f1c_n_worse_than_floor=n_worse, f1c_n_total=len(f1c_ok),
        f1c_rho_floor_default=float(rho_fd.statistic), f1c_p_floor_default=float(rho_fd.pvalue),
        f1c_rho_baseline_gap=float(rho_gap.statistic), f1c_p_baseline_gap=float(rho_gap.pvalue),
        f1d_frac_would_improve=frac_improve, f1d_mean_improvement=mean_improvement,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "f1_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'f1c_floor_vs_default_50random.csv'}")
    print(f"[out] {OUTDIR / 'f1b_named_orderings.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'f1_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
