#!/usr/bin/env python3
"""
Constructed orbital orderings vs the 200-random baseline.

Builds: identity, reverse, MI-Fiedler, heavy-hex-aware entanglement.
Evaluates each through the full Aer + sbd pipeline at 5 seeds, budget 15x15.

Run:  cd ~/sqd-project && python constructed_orderings.py
"""

import re, subprocess, time
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
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

ROOT = Path.home() / "sqd-project"
OUT  = ROOT / "outputs"
WORK = OUT / "constructed"; WORK.mkdir(parents=True, exist_ok=True)
SBD  = ROOT / "sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
DIM_A, DIM_B = comb(NORB, NELEC[0]), comb(NORB, NELEC[1])

# ---------------------------------------------------------------- system
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

E_HF, E_CASCI = mol_data.hf_energy, cas.e_tot
ham = ffsim.linear_operator(mol_data.hamiltonian, norb=NORB, nelec=NELEC)
ref = ffsim.hartree_fock_state(NORB, NELEC)
FCIDUMP = WORK / "n2_cas610_155.fcidump"; mol_data.to_fcidump(str(FCIDUMP))

print(f"HF {E_HF:.8f} | CCSD {cc.e_tot:.8f} | CASCI {E_CASCI:.8f}\n")

# ------------------------------------- entanglement from the CASCI vector
W = np.asarray(cas.ci).reshape(DIM_A, DIM_B) ** 2
W /= W.sum()

sa = cistring.make_strings(range(NORB), NELEC[0])
occ_a = np.array([[(s >> p) & 1 for p in range(NORB)] for s in sa])  # (120,10)
occ_b = occ_a  # closed shell, same string list

def shannon(w):
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log(w)))

# one-orbital entropies: states (n_up, n_dn) -> 4 outcomes
s1 = np.zeros(NORB)
for p in range(NORB):
    code = occ_a[:, p][:, None] * 2 + occ_b[:, p][None, :]
    s1[p] = shannon(np.bincount(code.ravel(), weights=W.ravel(), minlength=4))

# two-orbital entropies -> mutual information
s2 = np.zeros((NORB, NORB))
for p, q in combinations(range(NORB), 2):
    ca = occ_a[:, p] * 8 + occ_a[:, q] * 2
    cb = occ_b[:, p] * 4 + occ_b[:, q] * 1
    code = ca[:, None] + cb[None, :]
    s2[p, q] = s2[q, p] = shannon(
        np.bincount(code.ravel(), weights=W.ravel(), minlength=16))

MI = np.zeros((NORB, NORB))
for p, q in combinations(range(NORB), 2):
    MI[p, q] = MI[q, p] = max(s1[p] + s1[q] - s2[p, q], 0.0)

print("one-orbital entropies:", s1.round(4))
print("largest MI pairs:")
for p, q in sorted(combinations(range(NORB), 2), key=lambda x: -MI[x])[:6]:
    print(f"   ({p},{q})  I = {MI[p,q]:.4f}")
print()

# ---------------------------------------------------------- constructions
def fiedler_order(Wmat):
    L = np.diag(Wmat.sum(1)) - Wmat
    vals, vecs = np.linalg.eigh(L)
    return np.argsort(vecs[:, 1])          # 2nd smallest eigenvector

AB_SITES = list(range(0, NORB, 4))         # 0, 4, 8
MI_N = MI / MI.max()
S1_N = s1 / s1.max()

def hh_score(perm, w_chain=1.0, w_ab=1.0):
    """Heavy-hex-aware objective: strongly entangled pairs adjacent on the
    same-spin chain, highly entangled orbitals at the alpha-beta sites."""
    chain = sum(MI_N[perm[p], perm[p + 1]] for p in range(NORB - 1))
    ab    = sum(S1_N[perm[p]] for p in AB_SITES)
    return w_chain * chain + w_ab * ab

def hill_climb(start, score_fn, n_restarts=20, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_s = np.array(start), score_fn(start)
    for r in range(n_restarts):
        cur = np.array(start) if r == 0 else rng.permutation(NORB)
        cur_s = score_fn(cur)
        improved = True
        while improved:
            improved = False
            for i, j in combinations(range(NORB), 2):
                trial = cur.copy(); trial[[i, j]] = trial[[j, i]]
                s = score_fn(trial)
                if s > cur_s + 1e-12:
                    cur, cur_s, improved = trial, s, True
        if cur_s > best_s:
            best, best_s = cur, cur_s
    return best, best_s

fied = fiedler_order(MI)
hh, hh_s = hill_climb(fied, hh_score)

orderings = {
    "identity":   np.arange(NORB),
    "reverse":    np.arange(NORB)[::-1],
    "fiedler":    fied,
    "heavy_hex":  hh,
}
for name, p in orderings.items():
    print(f"{name:12s} {''.join(map(str,p))}   hh_score {hh_score(p):.4f}")
print()

# ------------------------------------------------------------ machinery
op_full = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=cc.t2, t1=cc.t1, n_reps=None)
_m_aa = np.zeros((NORB, NORB), bool)
for p in range(NORB - 1): _m_aa[p, p+1] = _m_aa[p+1, p] = True
np.fill_diagonal(_m_aa, True)
_m_ab = np.zeros((NORB, NORB), bool)
for p in AB_SITES: _m_ab[p, p] = True

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

def retained_J(op):
    J = np.asarray(op.diag_coulomb_mats)
    tot  = np.sum(J[:, 0]**2) + np.sum(J[:, 1]**2)
    kept = np.sum((J[:, 0]*_m_aa)**2) + np.sum((J[:, 1]*_m_ab)**2)
    return float(kept / tot)

HF_DET = "0"*(NORB - NELEC[0]) + "1"*NELEC[0]
b2i = {format(s, f"0{NORB}b"): i for i, s in enumerate(sa)}

def sample(op, seed):
    q = QuantumRegister(2*NORB, "q"); qc = QuantumCircuit(q)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(NORB, NELEC), q)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), q)
    qc.measure_all()
    sim = AerSimulator(seed_simulator=seed)
    tqc = transpile(qc, sim, optimization_level=3, seed_transpiler=SEED_TRANS)
    return sim.run(tqc, shots=SHOTS).result().get_counts()

def write_dets(counter, path):
    dets = sorted(d for d, _ in counter.most_common(BUDGET))
    if HF_DET not in dets: dets = sorted(dets + [HF_DET])
    Path(path).write_text("\n".join(dets) + "\n"); return len(dets)

def run_sbd(a, b):
    r = subprocess.run(
        ["mpirun", "-np", "1", str(SBD), "--fcidump", str(FCIDUMP),
         "--adetfile", str(a), "--bdetfile", str(b),
         "--method", "0", "--iteration", "30", "--block", "100",
         "--tolerance", "1e-8", "--adet_comm_size", "1",
         "--bdet_comm_size", "1", "--task_comm_size", "1",
         "--init", "0", "--shuffle", "0", "--carryover_type", "0", "--rdm", "0"],
        capture_output=True, text=True, cwd=SBD.parent)
    m = re.search(r"Sample-based diagonalization: Energy = ([-\d.]+)", r.stdout)
    if not m: raise RuntimeError((r.stdout + r.stderr)[-1200:])
    return float(m.group(1))

# ------------------------------------------------------------- evaluate
rows, t0 = [], time.time()
for name, perm in orderings.items():
    op_p = permute_operator(op_full, perm); op_m = apply_mask(op_p)
    for seed in SEEDS:
        counts = sample(op_m, seed)
        a_c, b_c = Counter(), Counter()
        for bits, n in counts.items():
            bits = bits.replace(" ", "")
            a_c[bits[-NORB:]] += n; b_c[bits[:NORB]] += n
        fa, fb = WORK / f"{name}_{seed}_a.txt", WORK / f"{name}_{seed}_b.txt"
        na, nb = write_dets(a_c, fa), write_dets(b_c, fb)
        E = run_sbd(fa, fb)
        ia = [b2i[x] for x in fa.read_text().split()]
        ib = [b2i[x] for x in fb.read_text().split()]
        rows.append({"ordering": name, "perm": "".join(map(str, perm)),
                     "seed": seed, "dim": na*nb,
                     "err_sub_mHa": (E - E_CASCI)*1000,
                     "captured": W[np.ix_(ia, ib)].sum(),
                     "retained_J": retained_J(op_p)})
    print(f"{name:12s} done ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "constructed_orderings.csv", index=False)
assert res.dim.nunique() == 1, f"budget not constant: {sorted(res.dim.unique())}"

# ------------------------------------------------------ percentile report
base = pd.read_csv(OUT / "scaleup_n2_cas610_155.csv")["err_sub_mHa"]
summ = res.groupby("ordering").agg(
    mean_err=("err_sub_mHa", "mean"), sd=("err_sub_mHa", "std"),
    captured=("captured", "mean"), retained_J=("retained_J", "first")
).sort_values("mean_err")
summ["percentile"] = [100*(base < e).mean() for e in summ.mean_err]
summ["rank_of_200"] = [(base < e).sum() + 1 for e in summ.mean_err]

print("\n" + "="*72)
print(f"200-random baseline: best {base.min():.2f}, "
      f"median {base.median():.2f}, worst {base.max():.2f} mHa")
print("="*72)
print(summ.round(3).to_string())
print("\nTop decile threshold:", f"{base.quantile(0.10):.2f} mHa")