#!/usr/bin/env python3
"""
Can a target derived from a single SQD run replace the exact wavefunction?

Uses the identity ordering's own sampled distribution |psi|^2 as the target,
then asks whether capture measured against it predicts subspace error across
the 200 orderings. This is the crudest version of a self-consistent scheme:
run once, re-order, run again.

Run:  cd ~/sqd-project && python self_consistent_target.py
"""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pyscf, pyscf.cc, pyscf.mcscf
import ffsim
from pyscf.fci import cistring
from scipy.stats import spearmanr

BOND, BASIS = 1.55, "6-31g"
N_FROZEN, NORB, NELEC = 4, 10, (3, 3)
OUT  = Path.home() / "sqd-project/outputs"
WORK, CSV = OUT / "scaleup", OUT / "scaleup_n2_cas610_155.csv"
DIM_A, DIM_B = comb(NORB, NELEC[0]), comb(NORB, NELEC[1])

mol = pyscf.gto.Mole()
mol.build(atom=[["N", (0, 0, 0)], ["N", (0, 0, BOND)]],
          basis=BASIS, symmetry=False, verbose=0)
mf = pyscf.scf.RHF(mol).run(verbose=0)
active = range(N_FROZEN, N_FROZEN + NORB)
mol_data = ffsim.MolecularData.from_scf(mf, active_space=active)
cc = pyscf.cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr())
                               if i not in active]).run(verbose=0)
cas = pyscf.mcscf.CASCI(mf, ncas=NORB, nelecas=NELEC); cas.ncore = N_FROZEN
cas.run(verbose=0)

W_exact = np.asarray(cas.ci).reshape(DIM_A, DIM_B) ** 2
W_exact /= W_exact.sum()
ref = ffsim.hartree_fock_state(NORB, NELEC)

# ---- mask + permutation machinery -------------------------------------
op_full = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=cc.t2, t1=cc.t1, n_reps=None)
_m_aa = np.zeros((NORB, NORB), bool)
for p in range(NORB - 1): _m_aa[p, p+1] = _m_aa[p+1, p] = True
np.fill_diagonal(_m_aa, True)
_m_ab = np.zeros((NORB, NORB), bool)
for p in range(0, NORB, 4): _m_ab[p, p] = True

def permute_operator(op, perm):
    P = np.eye(op.norb)[list(perm)]
    J = np.asarray(op.diag_coulomb_mats); U = np.asarray(op.orbital_rotations)
    kw = dict(diag_coulomb_mats=np.einsum('ij,rsjk,lk->rsil', P, J, P),
              orbital_rotations=np.einsum('rij,kj->rik', U, P))
    if op.final_orbital_rotation is not None:
        kw['final_orbital_rotation'] = op.final_orbital_rotation
    return ffsim.UCJOpSpinBalanced(**kw)

def apply_mask(op):
    J = np.asarray(op.diag_coulomb_mats).copy()
    J[:, 0] *= _m_aa; J[:, 1] *= _m_ab
    kw = dict(diag_coulomb_mats=J,
              orbital_rotations=np.asarray(op.orbital_rotations))
    if op.final_orbital_rotation is not None:
        kw['final_orbital_rotation'] = op.final_orbital_rotation
    return ffsim.UCJOpSpinBalanced(**kw)

# ---- target from the identity ordering's own distribution --------------
op_id = apply_mask(permute_operator(op_full, np.arange(NORB)))
psi   = ffsim.apply_unitary(ref, op_id, norb=NORB, nelec=NELEC)
W_self = (np.abs(psi) ** 2).reshape(DIM_A, DIM_B)
W_self /= W_self.sum()

ta_e, ta_s = W_exact.sum(1), W_self.sum(1)
print(f"alpha-marginal overlap with exact: {np.sum(np.sqrt(ta_e*ta_s))**2:.6f}")
print(f"top-15 alpha agreement: "
      f"{len(set(np.argsort(ta_e)[::-1][:15]) & set(np.argsort(ta_s)[::-1][:15]))}/15\n")

# ---- capture under both targets ---------------------------------------
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
    rows.append({"ordering": name,
                 "cap_exact": W_exact[np.ix_(ia, ib)].sum(),
                 "cap_self":  W_self[np.ix_(ia, ib)].sum()})

cap = pd.DataFrame(rows).merge(df[["ordering", "err_sub_mHa"]], on="ordering")
cap.to_csv(OUT / "self_consistent_target.csv", index=False)

print("=" * 62)
print("Spearman vs SQD subspace error")
print("=" * 62)
for col in ["cap_exact", "cap_self"]:
    r = spearmanr(cap[col], cap["err_sub_mHa"])
    print(f"  {col:12s} rho = {r.statistic:+.3f}   p = {r.pvalue:.2e}")

best   = cap["err_sub_mHa"].min()
spread = cap["err_sub_mHa"].max() - best
print(f"\n  random regret       {cap['err_sub_mHa'].mean()-best:6.2f} mHa")
for col in ["cap_exact", "cap_self"]:
    row = cap.loc[cap[col].idxmax()]
    print(f"  {col:12s} picks {row['ordering']:8s} -> {row['err_sub_mHa']:6.2f} mHa"
          f" | regret {row['err_sub_mHa']-best:5.2f} "
          f"({100*(row['err_sub_mHa']-best)/spread:4.1f}%)")