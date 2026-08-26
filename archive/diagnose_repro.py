#!/usr/bin/env python3
"""
Why does the identity ordering give different energies in different scripts?
Runs the identity ordering 3x with fully fresh state and prints every
intermediate, so we can see where the divergence enters.

Run:  cd ~/sqd-project && python diagnose_repro.py
"""
import re, subprocess, hashlib
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import pyscf, pyscf.cc, pyscf.mcscf
import ffsim
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

BOND, BASIS = 1.55, "6-31g"
N_FROZEN, NORB, NELEC = 4, 10, (3, 3)
BUDGET, SHOTS, SEED_TRANS = 15, 1_000_000, 2026
ROOT = Path.home()/"sqd-project"; OUT = ROOT/"outputs"
WORK = OUT/"diag_repro"; WORK.mkdir(parents=True, exist_ok=True)
SBD = ROOT/"sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
DIM_A, DIM_B = comb(NORB,NELEC[0]), comb(NORB,NELEC[1])

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
print(f"CASCI          {E_CASCI:.10f}")
print(f"CCSD           {cc.e_tot:.10f}")
print(f"t2 checksum    {hashlib.md5(np.ascontiguousarray(cc.t2)).hexdigest()[:12]}")
print(f"FCIDUMP md5    {hashlib.md5(FCIDUMP.read_bytes()).hexdigest()[:12]}\n")

op_full = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=cc.t2, t1=cc.t1, n_reps=None)
print(f"n_reps         {op_full.n_reps}")
J = np.asarray(op_full.diag_coulomb_mats)
print(f"J checksum     {hashlib.md5(np.ascontiguousarray(J)).hexdigest()[:12]}")
print(f"J norm         {np.linalg.norm(J):.10f}\n")

_m_aa = np.zeros((NORB,NORB),bool)
for p in range(NORB-1): _m_aa[p,p+1]=_m_aa[p+1,p]=True
np.fill_diagonal(_m_aa, True)
_m_ab = np.zeros((NORB,NORB),bool)
for p in range(0,NORB,4): _m_ab[p,p]=True

Jm = J.copy(); Jm[:,0]*=_m_aa; Jm[:,1]*=_m_ab
kw = dict(diag_coulomb_mats=Jm,
          orbital_rotations=np.asarray(op_full.orbital_rotations))
if op_full.final_orbital_rotation is not None:
    kw['final_orbital_rotation']=op_full.final_orbital_rotation
op_m = ffsim.UCJOpSpinBalanced(**kw)

psi = ffsim.apply_unitary(ref, op_m, norb=NORB, nelec=NELEC)
p = (np.abs(psi)**2).reshape(DIM_A,DIM_B); p/=p.sum()
print(f"exact marg_a top-3  {np.sort(p.sum(1))[::-1][:3].round(8)}")
print(f"exact top1          {p.max():.10f}\n")

HF_DET = "0"*(NORB-NELEC[0]) + "1"*NELEC[0]
def run_sbd(a,b):
    r = subprocess.run(["mpirun","-np","1",str(SBD),"--fcidump",str(FCIDUMP),
        "--adetfile",str(a),"--bdetfile",str(b),"--method","0",
        "--iteration","30","--block","100","--tolerance","1e-8",
        "--adet_comm_size","1","--bdet_comm_size","1","--task_comm_size","1",
        "--init","0","--shuffle","0","--carryover_type","0","--rdm","0"],
        capture_output=True, text=True, cwd=SBD.parent)
    return float(re.search(r"diagonalization: Energy = ([-\d.]+)", r.stdout).group(1))

print("repeat  seed   depth  2q     uniq   top1        det_md5      err_mHa")
print("-"*76)
for rep in range(3):
    for seed in [2026, 7]:
        q=QuantumRegister(2*NORB,"q"); qc=QuantumCircuit(q)
        qc.append(ffsim.qiskit.PrepareHartreeFockJW(NORB,NELEC),q)
        qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op_m),q); qc.measure_all()
        sim=AerSimulator(seed_simulator=seed)
        tqc=transpile(qc,sim,optimization_level=3,seed_transpiler=SEED_TRANS)
        counts=sim.run(tqc,shots=SHOTS).result().get_counts()

        a_c,b_c=Counter(),Counter()
        for bits,n in counts.items():
            bits=bits.replace(" ",""); a_c[bits[-NORB:]]+=n; b_c[bits[:NORB]]+=n
        fa,fb=WORK/f"r{rep}_s{seed}_a.txt", WORK/f"r{rep}_s{seed}_b.txt"
        for c,f in ((a_c,fa),(b_c,fb)):
            d=sorted(x for x,_ in c.most_common(BUDGET))
            if HF_DET not in d: d=sorted(d+[HF_DET])
            f.write_text("\n".join(d)+"\n")
        md5=hashlib.md5(fa.read_bytes()+fb.read_bytes()).hexdigest()[:10]
        n2q=sum(v for k,v in tqc.count_ops().items() if k in ("cz","cx","ecr"))
        print(f"  {rep}    {seed:5d}  {tqc.depth():5d}  {n2q:5d}  "
              f"{len(counts):5d}  {max(counts.values())/SHOTS:.8f}  {md5}  "
              f"{(run_sbd(fa,fb)-E_CASCI)*1000:8.3f}")