#!/usr/bin/env python3
"""
Scale-up: 200 random orbital orderings, single seed, full sbd pipeline.
Standalone -- depends on no notebook state.

Run:  cd ~/sqd-project && python scale_up.py
Output: outputs/scaleup_n2_cas610_155.csv  (written incrementally)
"""

import re, subprocess, sys, time
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pyscf, pyscf.cc, pyscf.mcscf
import ffsim
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------- settings
BOND, BASIS   = 1.55, "6-31g"
N_FROZEN      = 4
NORB, NELEC   = 10, (3, 3)
BUDGET        = 15
SHOTS         = 1_000_000
N_ORDERINGS   = 200
SEED_ORDER    = 4242          # different from 2026: a fresh independent draw
SEED_SIM      = 2026
SEED_TRANS    = 2026

ROOT = Path.home() / "sqd-project"
OUT  = ROOT / "outputs"
WORK = OUT / "scaleup"; WORK.mkdir(parents=True, exist_ok=True)
SBD  = ROOT / "sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
CSV  = OUT / "scaleup_n2_cas610_155.csv"

assert SBD.exists(), f"sbd executable not found at {SBD}"

rng = np.random.default_rng(SEED_ORDER)
DIM_A, DIM_B = comb(NORB, NELEC[0]), comb(NORB, NELEC[1])

print(f"N2 CAS({sum(NELEC)},{NORB}) @ {BOND} A | CI dim {DIM_A*DIM_B}")
print(f"{N_ORDERINGS} orderings | budget {BUDGET}/spin | {SHOTS:,} shots")
print(f"output -> {CSV}\n", flush=True)

# ----------------------------------------------------------------- system
mol = pyscf.gto.Mole()
mol.build(atom=[["N", (0, 0, 0)], ["N", (0, 0, BOND)]],
          basis=BASIS, symmetry=False, verbose=0)
mf = pyscf.scf.RHF(mol).run(verbose=0)

active   = range(N_FROZEN, N_FROZEN + NORB)
mol_data = ffsim.MolecularData.from_scf(mf, active_space=active)
cc = pyscf.cc.CCSD(mf,
        frozen=[i for i in range(mol.nao_nr()) if i not in active]).run(verbose=0)
cas = pyscf.mcscf.CASCI(mf, ncas=NORB, nelecas=NELEC); cas.ncore = N_FROZEN
cas.run(verbose=0)

E_HF, E_CASCI = mol_data.hf_energy, cas.e_tot
ham = ffsim.linear_operator(mol_data.hamiltonian, norb=NORB, nelec=NELEC)
ref = ffsim.hartree_fock_state(NORB, NELEC)

FCIDUMP = WORK / "n2_cas610_155.fcidump"
mol_data.to_fcidump(str(FCIDUMP))

print(f"HF    = {E_HF:.8f}")
print(f"CCSD  = {cc.e_tot:.8f}")
print(f"CASCI = {E_CASCI:.8f}\n", flush=True)

# ----------------------------------------------------------------- operator
op_full = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=cc.t2, t1=cc.t1, n_reps=None)

alpha_alpha = [(p, p + 1) for p in range(NORB - 1)]
alpha_beta  = [(p, p) for p in range(0, NORB, 4)]

_m_aa = np.zeros((NORB, NORB), bool)
for p, q in alpha_alpha: _m_aa[p, q] = _m_aa[q, p] = True
np.fill_diagonal(_m_aa, True)
_m_ab = np.zeros((NORB, NORB), bool)
for p, q in alpha_beta: _m_ab[p, q] = _m_ab[q, p] = True


def permute_operator(op, perm):
    P = np.eye(op.norb)[list(perm)]
    J = np.asarray(op.diag_coulomb_mats)
    U = np.asarray(op.orbital_rotations)
    kw = dict(diag_coulomb_mats=np.einsum('ij,rsjk,lk->rsil', P, J, P),
              orbital_rotations=np.einsum('rij,kj->rik', U, P))
    if op.final_orbital_rotation is not None:
        kw['final_orbital_rotation'] = op.final_orbital_rotation
    return ffsim.UCJOpSpinBalanced(**kw)


def apply_mask(op):
    J = np.asarray(op.diag_coulomb_mats).copy()
    J[:, 0] *= _m_aa
    J[:, 1] *= _m_ab
    kw = dict(diag_coulomb_mats=J,
              orbital_rotations=np.asarray(op.orbital_rotations))
    if op.final_orbital_rotation is not None:
        kw['final_orbital_rotation'] = op.final_orbital_rotation
    return ffsim.UCJOpSpinBalanced(**kw)


def variational_energy(op):
    s = ffsim.apply_unitary(ref, op, norb=NORB, nelec=NELEC)
    sr = np.ascontiguousarray(s.real, dtype=np.float64)
    si = np.ascontiguousarray(s.imag, dtype=np.float64)
    return float(np.dot(sr, ham @ sr) + np.dot(si, ham @ si))


def retained_J(op):
    J = np.asarray(op.diag_coulomb_mats)
    tot  = np.sum(J[:, 0] ** 2) + np.sum(J[:, 1] ** 2)
    kept = np.sum((J[:, 0] * _m_aa) ** 2) + np.sum((J[:, 1] * _m_ab) ** 2)
    return float(kept / tot)


def gini(counts):
    p = np.sort(np.asarray(list(counts), dtype=float))
    p = p / p.sum()
    n = len(p)
    return float(1.0 - 2.0 * np.sum(p * (n - np.arange(1, n + 1) + 0.5) / n))


# ----------------------------------------------------------------- gate
e0 = variational_energy(op_full)
dev = max(abs(variational_energy(permute_operator(op_full, rng.permutation(NORB))) - e0)
          for _ in range(5))
print(f"Permutation invariance gate: max dev {dev:.2e}", flush=True)
assert dev < 1e-8, "permutation is not a symmetry"
print("GATE PASSED\n", flush=True)

# ----------------------------------------------------------------- sampling
sim = AerSimulator(seed_simulator=SEED_SIM)
HF_DET = "0" * (NORB - NELEC[0]) + "1" * NELEC[0]


def sample(op):
    q  = QuantumRegister(2 * NORB, "q")
    qc = QuantumCircuit(q)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(NORB, NELEC), q)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), q)
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=3, seed_transpiler=SEED_TRANS)
    return sim.run(tqc, shots=SHOTS).result().get_counts()


def marginals(counts):
    a, b = Counter(), Counter()
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        a[bits[-NORB:]] += n
        b[bits[:NORB]]  += n
    return a, b


def write_dets(counter, path):
    dets = sorted(d for d, _ in counter.most_common(BUDGET))
    if HF_DET not in dets:
        dets = sorted(dets + [HF_DET])
    Path(path).write_text("\n".join(dets) + "\n")
    return len(dets)


def run_sbd(adet, bdet):
    r = subprocess.run(
        ["mpirun", "-np", "1", str(SBD), "--fcidump", str(FCIDUMP),
         "--adetfile", str(adet), "--bdetfile", str(bdet),
         "--method", "0", "--iteration", "30",
         "--block", "100", "--tolerance", "1e-8"],
        capture_output=True, text=True, cwd=SBD.parent)
    m = re.search(r"Sample-based diagonalization: Energy = ([-\d.]+)", r.stdout)
    if not m:
        raise RuntimeError((r.stdout + r.stderr)[-1200:])
    return float(m.group(1))


# ----------------------------------------------------------------- loop
orderings = {"identity": np.arange(NORB)}
seen = {tuple(range(NORB))}
while len(orderings) < N_ORDERINGS:
    p = rng.permutation(NORB)
    if tuple(p) not in seen:
        seen.add(tuple(p))
        orderings[f"q{len(orderings):03d}"] = p

rows, t0 = [], time.time()
for i, (name, perm) in enumerate(orderings.items(), 1):
    try:
        op_p = permute_operator(op_full, perm)
        op_m = apply_mask(op_p)

        counts = sample(op_m)
        a_c, b_c = marginals(counts)
        adet, bdet = WORK / f"{name}_a.txt", WORK / f"{name}_b.txt"
        n_a, n_b = write_dets(a_c, adet), write_dets(b_c, bdet)
        E_sub = run_sbd(adet, bdet)
        E_var = variational_energy(op_m)

        rows.append({
            "ordering": name, "perm": "".join(map(str, perm)),
            "E_var": E_var, "err_var_mHa": (E_var - E_CASCI) * 1000,
            "retained_J": retained_J(op_p),
            "n_unique": len(counts),
            "n_alpha_tot": len(a_c), "n_beta_tot": len(b_c),
            "top1": max(counts.values()) / SHOTS,
            "gini_alpha": gini(a_c.values()), "gini_beta": gini(b_c.values()),
            "dps_alpha": len(a_c) / SHOTS, "dps_beta": len(b_c) / SHOTS,
            "dim": n_a * n_b,
            "E_sub": E_sub, "err_sub_mHa": (E_sub - E_CASCI) * 1000,
            "pct_corr": 100 * (E_sub - E_HF) / (E_CASCI - E_HF),
        })
    except Exception as e:
        print(f"[{i:3d}/{N_ORDERINGS}] {name:8s} FAILED: {e}", flush=True)
        continue

    if i % 5 == 0 or i == 1:
        el = time.time() - t0
        eta = el / i * (N_ORDERINGS - i)
        print(f"[{i:3d}/{N_ORDERINGS}] {name:8s} "
              f"sub_err={rows[-1]['err_sub_mHa']:7.2f} mHa  "
              f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
        pd.DataFrame(rows).to_csv(CSV, index=False)

df = pd.DataFrame(rows)
df.to_csv(CSV, index=False)

print(f"\nDone: {len(df)}/{N_ORDERINGS} succeeded in {(time.time()-t0)/60:.1f} min")
print(f"dims all equal: {df['dim'].nunique() == 1}  ({sorted(df['dim'].unique())})")
print(f"subspace error: {df.err_sub_mHa.min():.2f} to {df.err_sub_mHa.max():.2f} mHa")
print(f"saved -> {CSV}")