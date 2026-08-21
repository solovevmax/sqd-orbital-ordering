#!/usr/bin/env python3
"""
Direct optimisation of two objectives over permutations:
  (a) retained Jastrow weight  -- computable WITHOUT the exact wavefunction
  (b) captured weight          -- ORACLE, establishes the achievable upper bound

Then evaluates both through the full pipeline against the 200-random baseline.

Run:  cd ~/sqd-project && python direct_optimise.py
"""
import re, subprocess, time
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np, pandas as pd
import pyscf, pyscf.cc, pyscf.mcscf
import ffsim
from pyscf.fci import cistring
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

BOND, BASIS = 1.55, "6-31g"
N_FROZEN, NORB, NELEC = 4, 10, (3, 3)
BUDGET, SHOTS = 15, 1_000_000
SEEDS = [2026, 7, 1234, 55555, 31415]
SEED_TRANS = 2026
ROOT = Path.home()/"sqd-project"; OUT = ROOT/"outputs"
WORK = OUT/"directopt"; WORK.mkdir(parents=True, exist_ok=True)
SBD = ROOT/"sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
DIM_A, DIM_B = comb(NORB, NELEC[0]), comb(NORB, NELEC[1])
rng = np.random.default_rng(11)

mol = pyscf.gto.Mole()
mol.build(atom=[["N",(0,0,0)],["N",(0,0,BOND)]], basis=BASIS,
          symmetry=False, verbose=0)
mf = pyscf.scf.RHF(mol).run(verbose=0)
active = range(N_FROZEN, N_FROZEN+NORB)
mol_data = ffsim.MolecularData.from_scf(mf, active_space=active)
cc = pyscf.cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr())
                               if i not in active]).run(verbose=0)
cas = pyscf.mcscf.CASCI(mf, ncas=NORB, nelecas=NELEC); cas.ncore=N_FROZEN
cas.run(verbose=0)
E_CASCI = cas.e_tot
ref = ffsim.hartree_fock_state(NORB, NELEC)
FCIDUMP = WORK/"n2.fcidump"; mol_data.to_fcidump(str(FCIDUMP))

W = np.asarray(cas.ci).reshape(DIM_A, DIM_B)**2; W /= W.sum()
sa = cistring.make_strings(range(NORB), NELEC[0])
b2i = {format(s, f"0{NORB}b"): i for i, s in enumerate(sa)}

op_full = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=cc.t2, t1=cc.t1, n_reps=None)
AB = list(range(0, NORB, 4))
_m_aa = np.zeros((NORB,NORB),bool)
for p in range(NORB-1): _m_aa[p,p+1]=_m_aa[p+1,p]=True
np.fill_diagonal(_m_aa, True)
_m_ab = np.zeros((NORB,NORB),bool)
for p in AB: _m_ab[p,p]=True

def permute_operator(op, perm):
    P = np.eye(op.norb)[list(perm)]
    J = np.asarray(op.diag_coulomb_mats); U = np.asarray(op.orbital_rotations)
    kw = dict(diag_coulomb_mats=np.einsum('ij,rsjk,lk->rsil',P,J,P),
              orbital_rotations=np.einsum('rij,kj->rik',U,P))
    if op.final_orbital_rotation is not None:
        kw['final_orbital_rotation']=op.final_orbital_rotation
    return ffsim.UCJOpSpinBalanced(**kw)

def apply_mask(op):
    J = np.asarray(op.diag_coulomb_mats).copy()
    J[:,0]*=_m_aa; J[:,1]*=_m_ab
    kw = dict(diag_coulomb_mats=J,
              orbital_rotations=np.asarray(op.orbital_rotations))
    if op.final_orbital_rotation is not None:
        kw['final_orbital_rotation']=op.final_orbital_rotation
    return ffsim.UCJOpSpinBalanced(**kw)

def retained_J_of(perm):
    J = np.asarray(permute_operator(op_full, perm).diag_coulomb_mats)
    tot  = np.sum(J[:,0]**2)+np.sum(J[:,1]**2)
    kept = np.sum((J[:,0]*_m_aa)**2)+np.sum((J[:,1]*_m_ab)**2)
    return float(kept/tot)

def captured_of(perm):
    """ORACLE: exact-wavefunction capture from the infinite-shot marginals."""
    psi = ffsim.apply_unitary(ref, apply_mask(permute_operator(op_full, perm)),
                              norb=NORB, nelec=NELEC)
    p = (np.abs(psi)**2).reshape(DIM_A, DIM_B); p /= p.sum()
    ia = np.argsort(p.sum(1))[::-1][:BUDGET]
    ib = np.argsort(p.sum(0))[::-1][:BUDGET]
    return float(W[np.ix_(ia, ib)].sum())

def hill_climb(score, n_restarts=30, verbose=""):
    best, best_s = None, -np.inf
    for r in range(n_restarts):
        cur = np.arange(NORB) if r==0 else rng.permutation(NORB)
        cur_s = score(cur); improved=True
        while improved:
            improved=False
            for i,j in combinations(range(NORB),2):
                t = cur.copy(); t[[i,j]] = t[[j,i]]
                s = score(t)
                if s > cur_s + 1e-12: cur,cur_s,improved = t,s,True
        if cur_s > best_s: best,best_s = cur,cur_s
    print(f"  {verbose}: {''.join(map(str,best))}  score {best_s:.5f}")
    return best

print(f"identity retained_J {retained_J_of(np.arange(NORB)):.5f}, "
      f"captured {captured_of(np.arange(NORB)):.5f}\n")
print("optimising...")
t0=time.time()
best_J   = hill_climb(retained_J_of, 30, "max retained_J")
best_cap = hill_climb(captured_of,   15, "max captured (ORACLE)")
print(f"  ({time.time()-t0:.0f}s)\n")

orderings = {"identity": np.arange(NORB),
             "max_retainedJ": best_J,
             "max_captured_ORACLE": best_cap}

# ---- evaluate through the full pipeline ----
HF_DET = "0"*(NORB-NELEC[0]) + "1"*NELEC[0]
def run_sbd(a,b):
    r = subprocess.run(["mpirun","-np","1",str(SBD),"--fcidump",str(FCIDUMP),
        "--adetfile",str(a),"--bdetfile",str(b),"--method","0",
        "--iteration","30","--block","100","--tolerance","1e-8",
        "--adet_comm_size","1","--bdet_comm_size","1","--task_comm_size","1",
        "--init","0","--shuffle","0","--carryover_type","0","--rdm","0"],
        capture_output=True, text=True, cwd=SBD.parent)
    m = re.search(r"Sample-based diagonalization: Energy = ([-\d.]+)", r.stdout)
    if not m: raise RuntimeError((r.stdout+r.stderr)[-1000:])
    return float(m.group(1))

rows=[]
for name, perm in orderings.items():
    op_m = apply_mask(permute_operator(op_full, perm))
    for seed in SEEDS:
        q=QuantumRegister(2*NORB,"q"); qc=QuantumCircuit(q)
        qc.append(ffsim.qiskit.PrepareHartreeFockJW(NORB,NELEC),q)
        qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op_m),q); qc.measure_all()
        sim=AerSimulator(seed_simulator=seed)
        counts=sim.run(transpile(qc,sim,optimization_level=3,
                       seed_transpiler=SEED_TRANS),shots=SHOTS).result().get_counts()
        a_c,b_c=Counter(),Counter()
        for bits,n in counts.items():
            bits=bits.replace(" ",""); a_c[bits[-NORB:]]+=n; b_c[bits[:NORB]]+=n
        fa,fb = WORK/f"{name}_{seed}_a.txt", WORK/f"{name}_{seed}_b.txt"
        for c,f in ((a_c,fa),(b_c,fb)):
            d=sorted(x for x,_ in c.most_common(BUDGET))
            if HF_DET not in d: d=sorted(d+[HF_DET])
            f.write_text("\n".join(d)+"\n")
        rows.append({"ordering":name,"seed":seed,
                     "err_sub_mHa":(run_sbd(fa,fb)-E_CASCI)*1000,
                     "retained_J":retained_J_of(perm),
                     "captured":captured_of(perm)})
    print(f"  {name} evaluated", flush=True)

res=pd.DataFrame(rows); res.to_csv(OUT/"direct_optimise.csv", index=False)
base=pd.read_csv(OUT/"scaleup_n2_cas610_155.csv")["err_sub_mHa"]
summ=res.groupby("ordering").agg(mean_err=("err_sub_mHa","mean"),
     sd=("err_sub_mHa","std"), retained_J=("retained_J","first"),
     captured=("captured","first")).sort_values("mean_err")
summ["percentile"]=[100*(base<e).mean() for e in summ.mean_err]
summ["rank_of_200"]=[(base<e).sum()+1 for e in summ.mean_err]
print("\n"+"="*70)
print(f"baseline: best {base.min():.2f}, median {base.median():.2f} mHa | "
      f"top-decile threshold {base.quantile(0.10):.2f}")
print("="*70); print(summ.round(4).to_string())