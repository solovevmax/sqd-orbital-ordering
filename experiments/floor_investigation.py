#!/usr/bin/env python3
"""
experiments/floor_investigation.py
=====================================

E1 -- investigate the ten triples at identity that share err_mHa=458.70
exactly. Pure re-analysis of experiments/outputs/anchor_decomposition_R1.6/
c1_all120_identity.csv and c2_transfer.csv, plus ONE new sampling
evaluation: the interaction_pairs_ab=[] control (no opposite-spin terms at
all) at identity, to test whether the floor is the "no alpha-beta
correlation" state.
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

OUTDIR = Path(__file__).resolve().parent / "outputs" / "budget_transfer"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
ANCHOR_DIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
C1_CSV = ANCHOR_DIR / "c1_all120_identity.csv"
C2_CSV = ANCHOR_DIR / "c2_transfer.csv"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

FLOOR_ERR = 458.70
FLOOR_TOL = 0.1
SHOTS = 2_000_000
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
    banner("E1 -- THE DEGENERATE FLOOR")
    import unified_run as U
    import run_ordering_pipeline as R
    from anchor_decomposition import evaluate

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
    pos_id = R.positions_from(np.arange(norb))

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

    # =========================================================== J_ab weight per triple
    banner("J_ab on-site weight per triple (120, identity)")
    Jab_diag_abs_sum = np.abs(Jab).sum(axis=0).diagonal()  # sum over reps, |J_ab[p,p]|
    all_triples = list(itertools.combinations(range(norb), 3))
    jw = {t: float(sum(Jab_diag_abs_sum[p] for p in t)) for t in all_triples}
    c1["jab_weight"] = c1.triple.map(jw)
    c1.to_csv(OUTDIR / "e1_c1_with_jab_weight.csv", index=False)

    at_floor = c1[np.abs(c1.err_mHa - FLOOR_ERR) < FLOOR_TOL].copy()
    out(f"  triples at floor (|err_mHa - {FLOOR_ERR}| < {FLOOR_TOL}): {len(at_floor)}")
    out(f"  floor triples: {list(at_floor.triple)}")

    best10 = c1.sort_values("err_mHa").head(10)
    out(f"\n  J_ab weight: floor triples (n={len(at_floor)}): "
        f"mean={at_floor.jab_weight.mean():.4f}  range=[{at_floor.jab_weight.min():.4f}, "
        f"{at_floor.jab_weight.max():.4f}]")
    out(f"  J_ab weight: best 10 triples:                 "
        f"mean={best10.jab_weight.mean():.4f}  range=[{best10.jab_weight.min():.4f}, "
        f"{best10.jab_weight.max():.4f}]")
    out(f"  J_ab weight: ALL 120 triples:                  "
        f"mean={c1.jab_weight.mean():.4f}  range=[{c1.jab_weight.min():.4f}, "
        f"{c1.jab_weight.max():.4f}]")
    floor_is_low_jab = at_floor.jab_weight.max() < c1.jab_weight.median()
    out(f"\n  are floor triples exactly the lowest-J_ab ones? "
        f"floor max jab_weight ({at_floor.jab_weight.max():.4f}) vs median of all 120 "
        f"({c1.jab_weight.median():.4f}): {'YES, all below median' if floor_is_low_jab else 'NOT simply the lowest-J_ab set'}")
    n_below_floor_max_jab = int((c1.jab_weight <= at_floor.jab_weight.max()).sum())
    out(f"  {n_below_floor_max_jab} of 120 triples have jab_weight <= the floor group's max - "
        f"{'exactly matches' if n_below_floor_max_jab == len(at_floor) else 'does NOT exactly match'} "
        f"the floor group size ({len(at_floor)})")

    # =========================================================== control: no ab terms at all
    banner("Control: interaction_pairs_ab = [] (no opposite-spin terms), each ordering's own same-spin mask")
    perm_by_ordering = pd.read_csv(BASELINE_CSV).groupby("ordering")["permutation"].first()
    pos_by_ordering = dict(identity=pos_id)
    for name in ("physical", "rand007"):
        pos_by_ordering[name] = R.positions_from(R.parse_permutation(perm_by_ordering[name], norb))

    floor_by_ordering = {}
    row = evaluate(R, t1L, t2L, pos_id, norb, nelec, nocc, hf, fcidump_path, E_CASCI,
                  b2i, W, seed=2026, anchor_orbitals=(), tag="e1_no_ab_control_identity")
    floor_by_ordering["identity"] = row["err_mHa"]
    out(f"  identity: err_mHa (no ab terms) = {row['err_mHa']:.4f}   status={row['status']}")
    matches_floor = abs(row["err_mHa"] - FLOOR_ERR) < 0.5
    out(f"  matches the observed floor ({FLOOR_ERR})? {matches_floor}")
    if matches_floor:
        out("  -> CONFIRMED: the floor is the 'no alpha-beta correlation' state. Any triple "
            "whose anchors carry negligible on-site J_ab weight effectively samples the "
            "same-spin-only ansatz, regardless of which specific orbitals were nominally "
            "chosen as anchors.")
    else:
        out("  -> NOT confirmed: the floor is something else, not simply 'zero opposite-spin "
            "correlation'. Needs further investigation.")

    out("\n  The floor is a property of the SAME-SPIN mask (position-dependent), so it is "
        "generally ordering-specific - checking physical and rand007's own floors too:")
    for name in ("physical", "rand007"):
        row_o = evaluate(R, t1L, t2L, pos_by_ordering[name], norb, nelec, nocc, hf, fcidump_path,
                        E_CASCI, b2i, W, seed=2026, anchor_orbitals=(), tag=f"e1_no_ab_control_{name}")
        floor_by_ordering[name] = row_o["err_mHa"]
        out(f"  {name}: err_mHa (no ab terms) = {row_o['err_mHa']:.4f}   status={row_o['status']}")

    # =========================================================== floor count per ordering
    banner("Floor count per ordering (each ordering's OWN floor value), and rho excluding it")
    d_rows = []
    for name, df_, n_total in (("identity", c1, 120), ("physical", c2[c2.ordering == "physical"], 40),
                               ("rand007", c2[c2.ordering == "rand007"], 40)):
        this_floor = floor_by_ordering[name]
        floor_mask = np.abs(df_.err_mHa - this_floor) < FLOOR_TOL
        n_floor = int(floor_mask.sum())
        above = df_[~floor_mask]
        rho_all = spearmanr(df_.retained_J_oppspin, df_.err_mHa)
        rho_above = spearmanr(above.retained_J_oppspin, above.err_mHa) if len(above) > 2 else None
        rng_above = float(above.err_mHa.max() - above.err_mHa.min()) if len(above) else float("nan")
        d_rows.append(dict(ordering=name, n_total=n_total, n_floor=n_floor, floor_err_mHa=this_floor,
                           rho_all=rho_all.statistic, p_all=rho_all.pvalue,
                           rho_above=(rho_above.statistic if rho_above else float("nan")),
                           p_above=(rho_above.pvalue if rho_above else float("nan")),
                           range_above=rng_above))
        out(f"  {name:<12} n={n_total:<4} floor={this_floor:.2f} n_floor={n_floor:<3}  "
            f"rho(all)={rho_all.statistic:+.3f} (p={rho_all.pvalue:.2e})  "
            f"rho(above floor)={'n/a' if rho_above is None else f'{rho_above.statistic:+.3f}'} "
            f"(p={'n/a' if rho_above is None else f'{rho_above.pvalue:.2e}'})  "
            f"range_above_floor={rng_above:.2f} mHa")

    banner("Does the proxy fail at physical, or is there just less spread to predict?")
    phys = next(d for d in d_rows if d["ordering"] == "physical")
    others = [d for d in d_rows if d["ordering"] != "physical"]
    less_spread = phys["range_above"] < min(o["range_above"] for o in others)
    still_weak_above_floor = abs(phys["rho_above"]) < 0.5
    out(f"  physical: n_floor={phys['n_floor']}, range above floor={phys['range_above']:.2f} mHa, "
        f"rho above floor={phys['rho_above']:+.3f}")
    if still_weak_above_floor:
        out("  -> Excluding the floor does NOT rescue the proxy at physical: rho above the "
            "floor is still weak. This is a genuine proxy failure, not merely an artefact "
            "of floor-triples adding unpredictable noise to the correlation.")
    else:
        out("  -> Excluding the floor DOES rescue the proxy at physical: the earlier weak rho "
            "was dominated by floor-triple noise, not a real breakdown of the signal.")
    if less_spread:
        out(f"  Also true: physical has less non-degenerate spread ({phys['range_above']:.2f} mHa) "
            f"than the other orderings above their own floors - part of the story, but not "
            f"the whole explanation given the rho result above.")

    # ------------------------------------------------------------- save
    report_path = OUTDIR / "e1_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    pd.DataFrame(d_rows).to_csv(OUTDIR / "e1_floor_by_ordering.csv", index=False)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="E1_floor", git_commit=git_commit, shots=SHOTS, seed=2026,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        c1_csv_sha256=sha256_of(C1_CSV), c2_csv_sha256=sha256_of(C2_CSV),
        floor_err_mHa=FLOOR_ERR, n_floor_identity=len(at_floor),
        no_ab_control_err_mHa=float(row["err_mHa"]), matches_floor=bool(matches_floor),
        floor_by_ordering=floor_by_ordering,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "e1_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'e1_c1_with_jab_weight.csv'}")
    print(f"[out] {OUTDIR / 'e1_floor_by_ordering.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'e1_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
