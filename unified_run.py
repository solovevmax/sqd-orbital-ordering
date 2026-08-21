#!/usr/bin/env python3
"""
Single consolidated run: baseline + constructed + optimised orderings,
one code path, cached reference data, fully reproducible.

Run:  cd ~/sqd-project && OMP_NUM_THREADS=1 python unified_run.py
Output: outputs/unified/results.csv
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import re, subprocess, time, hashlib, pickle
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

# ----------------------------------------------------------------- config
BOND, BASIS = 1.55, "6-31g"
N_FROZEN, NORB, NELEC = 4, 10, (3, 3)
BUDGET, SHOTS = 15, 1_000_000
N_RANDOM, SEEDS, SEED_TRANS = 150, [2026, 7], 2026
SEED_ORD = 90210

ROOT = Path.home()/"sqd-project"; OUT = ROOT/"outputs"
WORK = OUT/"unified"; WORK.mkdir(parents=True, exist_ok=True)
CACHE = WORK/"reference.pkl"
FCIDUMP = WORK/"reference.fcidump"
SBD = ROOT/"sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
DIM_A, DIM_B = comb(NORB,NELEC[0]), comb(NORB,NELEC[1])
AB = list(range(0, NORB, 4))

# ------------------------------------------------- reference data (cached)
if CACHE.exists():
    print("Loading cached reference data")
    ref_data = pickle.loads(CACHE.read_bytes())
else:
    print("Computing reference data (once)")
    mol = pyscf.gto.Mole()
    mol.build(atom=[["N",(0,0,0)],["N",(0,0,BOND)]], basis=BASIS,
              symmetry=False, verbose=0)
    mf = pyscf.scf.RHF(mol).run(verbose=0)
    active = range(N_FROZEN, N_FROZEN+NORB)
    md = ffsim.MolecularData.from_scf(mf, active_space=active)
    cc = pyscf.cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr())
                                   if i not in active]).run(verbose=0)
    cas = pyscf.mcscf.CASCI(mf, ncas=NORB, nelecas=NELEC)
    cas.ncore = N_FROZEN; cas.run(verbose=0)
    md.to_fcidump(str(FCIDUMP))
    ref_data = dict(t1=cc.t1, t2=cc.t2, e_ccsd=cc.e_tot,
                    e_casci=cas.e_tot, e_hf=md.hf_energy,
                    ci=np.asarray(cas.ci), hamiltonian=md.hamiltonian)
    CACHE.write_bytes(pickle.dumps(ref_data))

E_HF, E_CASCI = ref_data["e_hf"], ref_data["e_casci"]
ham = ffsim.linear_operator(ref_data["hamiltonian"], norb=NORB, nelec=NELEC)
ref = ffsim.hartree_fock_state(NORB, NELEC)
W = ref_data["ci"].reshape(DIM_A, DIM_B)**2; W /= W.sum()

print(f"HF {E_HF:.10f} | CCSD {ref_data['e_ccsd']:.10f} | CASCI {E_CASCI:.10f}")
print(f"FCIDUMP md5 {hashlib.md5(FCIDUMP.read_bytes()).hexdigest()[:12]}")
print(f"t2 md5      {hashlib.md5(np.ascontiguousarray(ref_data['t2'])).hexdigest()[:12]}\n")

# ------------------------------------------------------------- machinery
op_full = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=ref_data["t2"], t1=ref_data["t1"], n_reps=None)

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
    psi = ffsim.apply_unitary(ref, apply_mask(permute_operator(op_full, perm)),
                              norb=NORB, nelec=NELEC)
    p = (np.abs(psi)**2).reshape(DIM_A,DIM_B); p/=p.sum()
    ia = np.argsort(p.sum(1))[::-1][:BUDGET]
    ib = np.argsort(p.sum(0))[::-1][:BUDGET]
    return float(W[np.ix_(ia,ib)].sum())

# ----------------------------------------------------------- constructions
rng = np.random.default_rng(SEED_ORD)

def hill_climb(score, n_restarts=25):
    best, best_s = None, -np.inf
    for r in range(n_restarts):
        cur = np.arange(NORB) if r==0 else rng.permutation(NORB)
        cur_s = score(cur); improved=True
        while improved:
            improved=False
            for i,j in combinations(range(NORB),2):
                t=cur.copy(); t[[i,j]]=t[[j,i]]
                s=score(t)
                if s>cur_s+1e-12: cur,cur_s,improved=t,s,True
        if cur_s>best_s: best,best_s=cur,cur_s
    return best

print("Optimising...")
t0=time.time()
best_J   = hill_climb(retained_J_of, 25)
best_cap = hill_climb(captured_of, 12)
print(f"  max_retainedJ       {''.join(map(str,best_J))}")
print(f"  max_captured_ORACLE {''.join(map(str,best_cap))}   ({time.time()-t0:.0f}s)\n")

named = {"identity": np.arange(NORB),
         "reverse": np.arange(NORB)[::-1],
         "max_retainedJ": best_J,
         "max_captured_ORACLE": best_cap}

randoms = {}
seen = {tuple(range(NORB))}
while len(randoms) < N_RANDOM:
    p = rng.permutation(NORB)
    if tuple(p) not in seen:
        seen.add(tuple(p)); randoms[f"r{len(randoms):03d}"] = p

# ------------------------------------------------------------- evaluation
HF_DET = "0"*(NORB-NELEC[0]) + "1"*NELEC[0]

def run_sbd(a,b):
    r = subprocess.run(["mpirun","-np","1",str(SBD),"--fcidump",str(FCIDUMP),
        "--adetfile",str(a),"--bdetfile",str(b),"--method","0",
        "--iteration","30","--block","100","--tolerance","1e-8",
        "--adet_comm_size","1","--bdet_comm_size","1","--task_comm_size","1",
        "--init","0","--shuffle","0","--carryover_type","0","--rdm","0"],
        capture_output=True, text=True, cwd=SBD.parent)
    m = re.search(r"diagonalization: Energy = ([-\d.]+)", r.stdout)
    if not m: raise RuntimeError((r.stdout+r.stderr)[-1000:])
    return float(m.group(1))

def evaluate(name, perm, kind):
    op_m = apply_mask(permute_operator(op_full, perm))
    out=[]
    for seed in SEEDS:
        q=QuantumRegister(2*NORB,"q"); qc=QuantumCircuit(q)
        qc.append(ffsim.qiskit.PrepareHartreeFockJW(NORB,NELEC),q)
        qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op_m),q); qc.measure_all()
        sim=AerSimulator(seed_simulator=seed)
        tqc=transpile(qc,sim,optimization_level=3,seed_transpiler=SEED_TRANS)
        counts=sim.run(tqc,shots=SHOTS).result().get_counts()
        a_c,b_c=Counter(),Counter()
        for bits,n in counts.items():
            bits=bits.replace(" ",""); a_c[bits[-NORB:]]+=n; b_c[bits[:NORB]]+=n
        fa,fb=WORK/f"{name}_{seed}_a.txt", WORK/f"{name}_{seed}_b.txt"
        na=nb=0
        for c,f in ((a_c,fa),(b_c,fb)):
            d=sorted(x for x,_ in c.most_common(BUDGET))
            if HF_DET not in d: d=sorted(d+[HF_DET])
            f.write_text("\n".join(d)+"\n")
            if f is fa: na=len(d)
            else: nb=len(d)
        out.append({"ordering":name,"kind":kind,"perm":"".join(map(str,perm)),
                    "seed":seed,"dim":na*nb,
                    "err_sub_mHa":(run_sbd(fa,fb)-E_CASCI)*1000,
                    "retained_J":retained_J_of(perm),
                    "captured":captured_of(perm),
                    "n_unique":len(counts),
                    "top1":max(counts.values())/SHOTS})
    return out

rows=[]; t0=time.time(); todo=list(named.items())+list(randoms.items())
for i,(name,perm) in enumerate(todo,1):
    kind = "named" if name in named else "random"
    rows += evaluate(name, perm, kind)
    if i%10==0 or i<=4:
        el=time.time()-t0
        print(f"[{i:3d}/{len(todo)}] {name:20s} "
              f"{rows[-1]['err_sub_mHa']:7.2f} mHa  "
              f"eta {el/i*(len(todo)-i)/60:.0f}m", flush=True)
        pd.DataFrame(rows).to_csv(WORK/"results.csv", index=False)

df=pd.DataFrame(rows); df.to_csv(WORK/"results.csv", index=False)
assert df.dim.nunique()==1, f"budget varied: {sorted(df.dim.unique())}"
print(f"\nDone in {(time.time()-t0)/60:.1f} min, dim constant at {df.dim.iloc[0]}")

# --------------------------------------------------------------- analysis
summ = df.groupby(["kind","ordering"]).agg(
    mean_err=("err_sub_mHa","mean"), sd=("err_sub_mHa","std"),
    retained_J=("retained_J","first"), captured=("captured","first")).reset_index()
base = summ[summ.kind=="random"]["mean_err"]

print("\n"+"="*76)
print(f"random baseline (n={len(base)}): best {base.min():.2f}, "
      f"median {base.median():.2f}, worst {base.max():.2f} | "
      f"top-decile {base.quantile(0.10):.2f} mHa")
print("="*76)
named_s = summ[summ.kind=="named"].copy()
named_s["percentile"]  = [100*(base<e).mean() for e in named_s.mean_err]
named_s["rank_of_base"]= [(base<e).sum()+1 for e in named_s.mean_err]
print(named_s.sort_values("mean_err").round(4).to_string(index=False))

from scipy.stats import spearmanr
print("\nAcross random orderings, Spearman vs subspace error:")
rnd = summ[summ.kind=="random"]
for c in ["captured","retained_J"]:
    r=spearmanr(rnd[c], rnd.mean_err)
    print(f"  {c:12s} rho = {r.statistic:+.3f}  p = {r.pvalue:.2e}")