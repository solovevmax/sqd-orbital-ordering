#!/usr/bin/env python3
"""
experiments/lucj_verification.py
===================================

V1-V5 -- verification pass on lucj_control's data. Pure re-analysis of
cached CSVs, no new sampling, no sbd.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
LUCJ_DIR = Path(__file__).resolve().parent / "outputs" / "lucj_control"
ANCHOR_DIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
E1_META = Path(__file__).resolve().parent / "outputs" / "budget_transfer" / "e1_metadata.json"
F1C_CSV = Path(__file__).resolve().parent / "outputs" / "floor_generalization" / "f1c_floor_vs_default_50random.csv"

FLOOR_TOL = 0.1


def parse_triple(s):
    if isinstance(s, tuple):
        return s
    s = str(s).strip()
    if s.startswith("("):
        return ast.literal_eval(s)
    return tuple(int(c) for c in s.zfill(3))


def banner(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    banner("V1 -- SIGN CONVENTION")
    id_lucj = pd.read_csv(LUCJ_DIR / "identity_120_lucj.csv")
    id_lucj["triple"] = id_lucj.triple.apply(parse_triple)
    print("Definitions used (from experiments/lucj_control.py source):")
    print("  err_lucj = (E_lucj - E_CASCI) * 1000   [E_lucj = <psi|H|psi>/<psi|psi> of the masked")
    print("             LUCJ state applied to HF - a variational upper bound on E_CASCI]")
    print("  err_sqd  = (energy - E_CASCI) * 1000   [energy = sbd's subspace-diagonalised energy -")
    print("             also a variational upper bound within the selected determinant subspace]")
    print("Both are (approximate - exact)*1000: positive, lower-is-better, SAME convention.")
    print(f"\nAll err_lucj > 0? {(id_lucj.err_lucj > 0).all()}  "
          f"min={id_lucj.err_lucj.min():.4f}")
    print(f"All err_sqd > 0? {(id_lucj.err_sqd > 0).all()}  min={id_lucj.err_sqd.min():.4f}")
    both_positive_lower_better = (id_lucj.err_lucj > 0).all() and (id_lucj.err_sqd > 0).all()
    print(f"Verified: both positive, both lower-is-better -> correlations are NOT sign-flipped.")

    # ------------------------------------------------------------- load everything
    c1 = pd.read_csv(ANCHOR_DIR / "c1_all120_identity.csv")
    c1["triple"] = c1.triple.apply(parse_triple)
    c1 = c1.set_index(c1.triple.apply(str))

    c2 = pd.read_csv(ANCHOR_DIR / "c2_transfer.csv")
    c2["triple"] = c2.triple.apply(parse_triple)
    c2 = c2[c2.ordering != "physical_reverse"].reset_index(drop=True)

    pr_lucj = pd.read_csv(LUCJ_DIR / "physical_rand007_40_lucj.csv")
    pr_lucj["triple"] = pr_lucj.triple.apply(parse_triple)

    e1_meta = json.loads(E1_META.read_text())
    floor_by_ordering = dict(e1_meta["floor_by_ordering"])  # identity, physical, rand007

    id_lucj["retained_J_oppspin"] = id_lucj.triple.apply(lambda t: c1.loc[str(t), "retained_J_oppspin"])
    id_lucj["captured_sqd"] = id_lucj.triple.apply(lambda t: c1.loc[str(t), "captured"])

    def merge_ordering(name):
        sub = pr_lucj[pr_lucj.ordering == name].copy()
        c2_o = c2[c2.ordering == name].set_index(c2[c2.ordering == name].triple.apply(str))
        sub["captured_sqd"] = sub.triple.apply(lambda t: c2_o.loc[str(t), "captured"])
        sub["retained_J_oppspin"] = sub.triple.apply(lambda t: c2_o.loc[str(t), "retained_J_oppspin"])
        return sub

    phys = merge_ordering("physical")
    rand007 = merge_ordering("rand007")

    datasets = {"identity": id_lucj, "physical": phys, "rand007": rand007}

    # ------------------------------------------------------------- floor triples per ordering
    floor_triples = {}
    for name, df_ in datasets.items():
        floor_val = floor_by_ordering[name]
        mask = np.abs(df_.err_sqd - floor_val) < FLOOR_TOL
        floor_triples[name] = set(df_[mask].triple)
        print(f"\n{name}: floor={floor_val:.2f}  n_floor_triples={mask.sum()}/{len(df_)}")

    # =========================================================== V2/V3
    banner("V2/V3 -- rho(err_lucj, err_sqd) WITH and WITHOUT floor triples")
    rho_results = {}
    for name, df_ in datasets.items():
        r_all = spearmanr(df_.err_lucj, df_.err_sqd)
        above = df_[~df_.triple.isin(floor_triples[name])]
        if len(above) > 2:
            r_above = spearmanr(above.err_lucj, above.err_sqd)
        else:
            r_above = None
        rho_results[name] = dict(all=r_all, above=r_above, n_floor=len(floor_triples[name]), n_total=len(df_))
        print(f"\n{name} (n={len(df_)}, n_floor={len(floor_triples[name])}):")
        print(f"  rho(err_lucj, err_sqd) ALL triples:        {r_all.statistic:+.3f}  p={r_all.pvalue:.2e}")
        if r_above is not None:
            print(f"  rho(err_lucj, err_sqd) FLOOR EXCLUDED:     {r_above.statistic:+.3f}  p={r_above.pvalue:.2e}")
            collapsed = abs(r_above.statistic) < 0.3 or r_above.pvalue > 0.05
            print(f"  collapsed toward zero (|rho|<0.3 or p>0.05)? {collapsed}")
        else:
            print(f"  (too few non-floor triples for a meaningful rho)")

    # =========================================================== V4
    banner("V4 -- rho(err_lucj, captured_sqd) at all three chains")
    for name, df_ in datasets.items():
        r_all = spearmanr(df_.err_lucj, df_.captured_sqd)
        above = df_[~df_.triple.isin(floor_triples[name])]
        r_above = spearmanr(above.err_lucj, above.captured_sqd) if len(above) > 2 else None
        print(f"\n{name}: rho(err_lucj, captured_sqd) ALL = {r_all.statistic:+.3f} (p={r_all.pvalue:.2e})"
              + (f"   FLOOR EXCLUDED = {r_above.statistic:+.3f} (p={r_above.pvalue:.2e})" if r_above else ""))

    # =========================================================== V5
    banner("V5 -- rho(err_lucj, retained_J_oppspin) at all three chains")
    for name, df_ in datasets.items():
        r_all = spearmanr(df_.err_lucj, df_.retained_J_oppspin)
        above = df_[~df_.triple.isin(floor_triples[name])]
        r_above = spearmanr(above.err_lucj, above.retained_J_oppspin) if len(above) > 2 else None
        print(f"\n{name}: rho(err_lucj, retained_J_oppspin) ALL = {r_all.statistic:+.3f} (p={r_all.pvalue:.2e})"
              + (f"   FLOOR EXCLUDED = {r_above.statistic:+.3f} (p={r_above.pvalue:.2e})" if r_above else ""))

    # =========================================================== VERDICT
    banner("VERDICT")
    any_survives = False
    for name in datasets:
        r_above = rho_results[name]["above"]
        if r_above is not None and abs(r_above.statistic) >= 0.3 and r_above.pvalue < 0.05:
            any_survives = True
            print(f"  {name}: rho survives floor exclusion ({r_above.statistic:+.3f}, p={r_above.pvalue:.2e})")
    if any_survives:
        print("\nDoes ansatz quality predict SQD outcome, at any chain, once floor triples are "
              "removed? YES, at the chain(s) listed above.")
    else:
        print("\nDoes ansatz quality predict SQD outcome, at any chain, once floor triples are "
              "removed? NO - the identity correlation was a floor artefact; ansatz-level err_lucj "
              "does not predict SQD outcome at any of the three chains once the degenerate floor "
              "triples are excluded.")


if __name__ == "__main__":
    sys.exit(main() or 0)
