import numpy as np, ffsim, pyscf, pyscf.cc
from math import comb

NORB, NELEC = 10, (3, 3)
mol = pyscf.gto.Mole()
mol.build(atom=[["N",(0,0,0)],["N",(0,0,1.55)]], basis="6-31g",
          symmetry=False, verbose=0)
mf = pyscf.scf.RHF(mol).run(verbose=0)
active = range(4, 14)
cc = pyscf.cc.CCSD(mf, frozen=[i for i in range(mol.nao_nr())
                               if i not in active]).run(verbose=0)
op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=cc.t2, t1=cc.t1, n_reps=None)

ref = ffsim.hartree_fock_state(NORB, NELEC)
before = ref.copy()
print(f"ref before: nonzero={np.count_nonzero(ref)}, norm={np.linalg.norm(ref):.10f}")

_ = ffsim.apply_unitary(ref, op, norb=NORB, nelec=NELEC)

print(f"ref after : nonzero={np.count_nonzero(ref)}, norm={np.linalg.norm(ref):.10f}")
print(f"UNCHANGED : {np.allclose(ref, before)}")
if not np.allclose(ref, before):
    print(">>> apply_unitary MUTATES its input. This is the bug.")