#!/usr/bin/env python3
"""
Marginal matching: does the true-wavefunction weight captured by an ordering's
product subspace explain its SQD subspace error?

Uses the determinant files actually fed to sbd during the 200-ordering sweep,
so no re-simulation is needed.

Run:  cd ~/sqd-project && python marginal_matching.py
"""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pyscf, pyscf.mcscf
from pyscf.fci import cistring
from scipy.stats import spearmanr

BOND, BASIS = 1.55, "6-31g"
N_FROZEN, NORB, NELEC = 4, 10, (3, 3)

ROOT  = Path.home() / "sqd-project"
OUT   = ROOT / "outputs"
WORK  = OUT / "scaleup"
CSV   = OUT / "scaleup_n2_cas610_155.csv"

DIM_A, DIM_B = comb(NORB, NELEC[0]), comb(NORB, NELEC[1])

# ---- exact wavefunction -------------------------------------------------
mol = pyscf.gto.Mole()
mol.build(atom=[["N", (0, 0, 0)], ["N", (0, 0, BOND)]],
          basis=BASIS, symmetry=False, verbose=0)
mf = pyscf.scf.RHF(mol).run(verbose=0)
cas = pyscf.mcscf.CASCI(mf, ncas=NORB, nelecas=NELEC); cas.ncore = N_FROZEN
cas.run(verbose=0)

C  = np.asarray(cas.ci).reshape(DIM_A, DIM_B)
W  = C ** 2
W /= W.sum()
target_a, target_b = W.sum(axis=1), W.sum(axis=0)

print(f"CASCI energy       = {cas.e_tot:.8f}")
print(f"CI matrix shape    = {C.shape}")
print(f"target_a top-5     = {np.sort(target_a)[::-1][:5].round(5)}")
print(f"target_b top-5     = {np.sort(target_b)[::-1][:5].round(5)}")

# best possible capture at budget 15, for reference
ia_best = np.argsort(target_a)[::-1][:15]
ib_best = np.argsort(target_b)[::-1][:15]
print(f"ideal captured weight at budget 15 = {W[np.ix_(ia_best, ib_best)].sum():.6f}\n")

# ---- bitstring -> index -------------------------------------------------
_strs = cistring.make_strings(range(NORB), NELEC[0])
b2i = {format(s, f"0{NORB}b"): i for i, s in enumerate(_strs)}

def read_idx(path):
    return [b2i[b] for b in Path(path).read_text().split()]

# ---- per-ordering capture ----------------------------------------------
df = pd.read_csv(CSV)
print(f"Loaded {len(df)} orderings\n")

rows = []
for name in df["ordering"]:
    fa, fb = WORK / f"{name}_a.txt", WORK / f"{name}_b.txt"
    if not (fa.exists() and fb.exists()):
        continue
    ia, ib = read_idx(fa), read_idx(fb)

    captured   = W[np.ix_(ia, ib)].sum()          # true weight inside the subspace
    cap_a      = target_a[ia].sum()               # marginal coverage, alpha
    cap_b      = target_b[ib].sum()               # marginal coverage, beta
    rows.append({"ordering": name, "captured": captured,
                 "cap_alpha": cap_a, "cap_beta": cap_b,
                 "cap_product": cap_a * cap_b,
                 "missed_mHa_proxy": (1 - captured)})

cap = pd.DataFrame(rows).merge(df[["ordering", "err_sub_mHa", "err_var_mHa",
                                   "retained_J", "top1", "n_unique"]],
                               on="ordering")
cap.to_csv(OUT / "marginal_matching_n2_cas610_155.csv", index=False)
print(f"Matched {len(cap)} orderings with determinant files\n")

# ---- the question -------------------------------------------------------
print("=" * 62)
print("Spearman correlation with SQD subspace error")
print("=" * 62)
for col in ["captured", "cap_alpha", "cap_beta", "cap_product",
            "retained_J", "err_var_mHa", "top1", "n_unique"]:
    r = spearmanr(cap[col], cap["err_sub_mHa"])
    print(f"  {col:18s} rho = {r.statistic:+.3f}   p = {r.pvalue:.2e}")

print("\n" + "=" * 62)
print("Selector regret (lower is better)")
print("=" * 62)
best   = cap["err_sub_mHa"].min()
spread = cap["err_sub_mHa"].max() - best
rand   = cap["err_sub_mHa"].mean() - best
print(f"  random baseline    {rand:6.2f} mHa ({100*rand/spread:.1f}% of spread)")

for col, direction in [("captured", "max"), ("cap_product", "max"),
                       ("retained_J", "max"), ("err_var_mHa", "min")]:
    pick = cap[col].idxmax() if direction == "max" else cap[col].idxmin()
    row = cap.loc[pick]
    reg = row["err_sub_mHa"] - best
    print(f"  {col:18s} picks {row['ordering']:8s} -> {row['err_sub_mHa']:6.2f} mHa"
          f" | regret {reg:6.2f} ({100*reg/spread:4.1f}%)")

print("\n" + "=" * 62)
print("Top 10 orderings by captured weight vs by actual error")
print("=" * 62)
print(cap.nlargest(10, "captured")[
    ["ordering", "captured", "cap_alpha", "cap_beta", "err_sub_mHa"]
].round(5).to_string(index=False))