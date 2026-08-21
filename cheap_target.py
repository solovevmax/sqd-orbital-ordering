#!/usr/bin/env python3
"""
Can a CHEAP approximate wavefunction replace the exact CASCI target?

If capture computed against a CISD target predicts SQD error nearly as well as
capture against the exact CASCI target, marginal matching becomes a usable
selection criterion rather than a post-hoc diagnostic.

Run:  cd ~/sqd-project && python cheap_target.py
"""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pyscf, pyscf.ci, pyscf.mcscf, pyscf.mp
from pyscf.fci import cistring
from scipy.stats import spearmanr

BOND, BASIS = 1.55, "6-31g"
N_FROZEN, NORB, NELEC = 4, 10, (3, 3)

ROOT, OUT = Path.home() / "sqd-project", Path.home() / "sqd-project/outputs"
WORK, CSV = OUT / "scaleup", OUT / "scaleup_n2_cas610_155.csv"
DIM_A, DIM_B = comb(NORB, NELEC[0]), comb(NORB, NELEC[1])

frozen = None  # set below

mol = pyscf.gto.Mole()
mol.build(atom=[["N", (0, 0, 0)], ["N", (0, 0, BOND)]],
          basis=BASIS, symmetry=False, verbose=0)
mf = pyscf.scf.RHF(mol).run(verbose=0)

active = range(N_FROZEN, N_FROZEN + NORB)
frozen = [i for i in range(mol.nao_nr()) if i not in active]

# ---- exact target (reference) ------------------------------------------
cas = pyscf.mcscf.CASCI(mf, ncas=NORB, nelecas=NELEC); cas.ncore = N_FROZEN
cas.run(verbose=0)
W_exact = np.asarray(cas.ci).reshape(DIM_A, DIM_B) ** 2
W_exact /= W_exact.sum()
print(f"exact  CASCI energy = {cas.e_tot:.8f}")

# ---- cheap target: CISD ------------------------------------------------
myci = pyscf.ci.CISD(mf, frozen=frozen).run(verbose=0)
print(f"cheap  CISD  energy = {myci.e_tot:.8f}")

civec = pyscf.ci.cisd.to_fcivec(myci.ci, NORB, NELEC)
W_cisd = np.asarray(civec).reshape(DIM_A, DIM_B) ** 2
W_cisd /= W_cisd.sum()

# how similar are the two targets?
ta_e, tb_e = W_exact.sum(1), W_exact.sum(0)
ta_c, tb_c = W_cisd.sum(1),  W_cisd.sum(0)
print(f"\ntarget overlap, alpha marginals: "
      f"{np.sum(np.sqrt(ta_e*ta_c))**2:.6f}   (1 = identical)")
print(f"rank agreement of top-15 alpha strings: "
      f"{len(set(np.argsort(ta_e)[::-1][:15]) & set(np.argsort(ta_c)[::-1][:15]))}/15\n")

# ---- per-ordering capture under both targets ---------------------------
_strs = cistring.make_strings(range(NORB), NELEC[0])
b2i = {format(s, f"0{NORB}b"): i for i, s in enumerate(_strs)}
read_idx = lambda p: [b2i[b] for b in Path(p).read_text().split()]

df = pd.read_csv(CSV)
rows = []
for name in df["ordering"]:
    fa, fb = WORK / f"{name}_a.txt", WORK / f"{name}_b.txt"
    if not (fa.exists() and fb.exists()):
        continue
    ia, ib = read_idx(fa), read_idx(fb)
    rows.append({
        "ordering":       name,
        "cap_exact":      W_exact[np.ix_(ia, ib)].sum(),
        "cap_cisd":       W_cisd[np.ix_(ia, ib)].sum(),
        "cap_cisd_alpha": ta_c[ia].sum(),
        "cap_cisd_beta":  tb_c[ib].sum(),
    })

cap = pd.DataFrame(rows).merge(df[["ordering", "err_sub_mHa"]], on="ordering")
cap["cap_cisd_product"] = cap.cap_cisd_alpha * cap.cap_cisd_beta
cap.to_csv(OUT / "cheap_target_n2_cas610_155.csv", index=False)

print("=" * 62)
print("Spearman vs SQD subspace error")
print("=" * 62)
for col in ["cap_exact", "cap_cisd", "cap_cisd_product",
            "cap_cisd_alpha", "cap_cisd_beta"]:
    r = spearmanr(cap[col], cap["err_sub_mHa"])
    print(f"  {col:18s} rho = {r.statistic:+.3f}   p = {r.pvalue:.2e}")

print(f"\n  agreement between exact and CISD capture: "
      f"rho = {spearmanr(cap.cap_exact, cap.cap_cisd).statistic:+.3f}")

print("\n" + "=" * 62)
print("Selector regret")
print("=" * 62)
best   = cap["err_sub_mHa"].min()
spread = cap["err_sub_mHa"].max() - best
print(f"  random               {cap['err_sub_mHa'].mean()-best:6.2f} mHa")
for col in ["cap_exact", "cap_cisd", "cap_cisd_product"]:
    row = cap.loc[cap[col].idxmax()]
    reg = row["err_sub_mHa"] - best
    print(f"  {col:18s}   picks {row['ordering']:8s} -> "
          f"{row['err_sub_mHa']:6.2f} mHa | regret {reg:5.2f} "
          f"({100*reg/spread:4.1f}%)")