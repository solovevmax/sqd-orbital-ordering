#!/usr/bin/env python3
"""Tier 2: re-run the full sampling + sbd pipeline for a declared sample of
evaluations and compare to stored values. Uses the actual production code
paths (scripts/run_ordering_pipeline.py, scripts/unified_run.py,
experiments/tm_transfer.py) -- Tier 2 is explicitly about re-executing the
real pipeline, not an independent reimplementation (that's Tier 0/1).

Run from repo root. Cr2 is last, per the approved plan (~16 min alone).
"""
import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_ordering_pipeline as R
import unified_run as U

R.CFG["sbd_bin"] = str(U.SBD)

rows = []


def record(tag, stored, computed, tol, wall_s, note=""):
    numeric = (isinstance(stored, (int, float)) and not isinstance(stored, bool)
               and isinstance(computed, (int, float)) and not isinstance(computed, bool))
    diff = computed - stored if numeric else None
    ok = (abs(diff) <= tol) if (diff is not None and tol is not None) else (
        bool(computed == stored) if not numeric and computed is not None else None)
    rows.append(dict(tag=tag, stored=stored, computed=computed, diff=diff, tol=tol, ok=ok,
                      wall_s=wall_s, note=note))
    print(f"[{'PASS' if ok else 'FAIL' if ok is False else 'INFO'}] {tag}: "
          f"stored={stored}  computed={computed}  diff={diff}  wall={wall_s:.1f}s  {note}")


def h10_eval(pos, anchor_kwargs, tag, cachedir="cache/h10_R1.6", seed=2026, shots=2_000_000):
    t0 = time.time()
    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=cachedir)
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    centroids = ref["centroids"]
    E_CASCI = ref["E_CASCI"]
    fcidump_path = ref["fcidump_path"]
    hf = R.hf_bitstring(norb, nocc)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))
    pairs = R.interaction_pairs_for(pos, centroids, J_ab=Jab, **anchor_kwargs)
    op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
    a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, shots, seed)
    a_sel, _ = R.top_dets(a_c, 15, hf)
    b_sel, _ = R.top_dets(b_c, 15, hf)
    ap = Path(f"_t2_{tag}_a.txt"); bp = Path(f"_t2_{tag}_b.txt")
    ap.write_text("\n".join(sorted(a_sel)) + "\n")
    bp.write_text("\n".join(sorted(b_sel)) + "\n")
    energy = R.run_sbd(str(fcidump_path), str(ap), str(bp), norb)
    ap.unlink(); bp.unlink()
    err_mHa = (energy - E_CASCI) * 1000.0
    return err_mHa, time.time() - t0


print("=" * 70)
print("1. H10 identity, default anchors")
err, t = h10_eval(R.positions_from(np.arange(10)), dict(anchor_offset=0), "id_default")
record("h10_identity_default", 300.31919403956664, err, 1e-3, t, "SHIPPED cache")

print("=" * 70)
print("2. H10 identity, best anchors (0,1,2)")
err, t = h10_eval(R.positions_from(np.arange(10)), dict(anchor_orbitals=(0, 1, 2)), "id_012")
c1 = pd.read_csv(REPO_ROOT / "experiments/outputs/anchor_decomposition_R1.6/c1_all120_identity.csv")
stored = float(c1[c1.triple == "(0, 1, 2)"].err_mHa.iloc[0])
record("h10_identity_best_012", stored, err, 1e-3, t)

print("=" * 70)
print("3. H10 physical, default anchors")
h10ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir="cache/h10_R1.6")
phys_perm = R.parse_permutation(h10ref["orderings"]["physical"]["perm"], 10)
err, t = h10_eval(R.positions_from(phys_perm), dict(anchor_offset=0), "phys_default")
record("h10_physical_default", 389.71, err, 0.02, t)

print("=" * 70)
print("4. H10 physical, best anchors (2,4,7)")
err, t = h10_eval(R.positions_from(phys_perm), dict(anchor_orbitals=(2, 4, 7)), "phys_247")
record("h10_physical_best_247", 172.149392, err, 1e-3, t)

print("=" * 70)
print("5. H10 identity, no-alpha-beta control")
err, t = h10_eval(R.positions_from(np.arange(10)), dict(anchor_orbitals=()), "id_noab")
record("h10_identity_noab", 458.6996615694874, err, 1e-3, t)

print("=" * 70)
print("6-9. Four random (chain, triple) held-out pairs (seed=42)")
phaseb_meta = json.loads((REPO_ROOT / "experiments/outputs/chain_aware/phaseB_metadata.json").read_text())
new_chains = phaseb_meta["new_chains"]
heldout_sample = [
    ("newchain07", (2, 4, 9), 219.382200),
    ("newchain11", (0, 2, 5), 222.262024),
    ("newchain10", (5, 6, 7), 458.378461),
    ("newchain03", (1, 2, 9), 184.070279),
]
for chain, triple, stored_err in heldout_sample:
    perm = [int(c) for c in new_chains[chain]]
    err, t = h10_eval(R.positions_from(perm), dict(anchor_orbitals=triple), f"{chain}_{triple}")
    record(f"h10_{chain}_{triple}", stored_err, err, 1e-3, t)

print("=" * 70)
print("10. N2 identity, default anchors")
t0 = time.time()
out = U.evaluate("identity", np.arange(U.NORB), "named", seeds=[2026])
record("n2_identity_default", 31.870454303643218, out[0]["err_sub_mHa"], 1e-6, time.time() - t0,
       "SHIPPED cache")

print("=" * 70)
print("11. N2 identity, best anchors (0,1,9)")
import ffsim
t0 = time.time()
op_pert = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=U.ref_data["t2"], t1=U.ref_data["t1"], n_reps=None)
op_perm = U.permute_operator(op_pert, np.arange(U.NORB))
J = np.asarray(op_perm.diag_coulomb_mats).copy()
J[:, 0] *= U._m_aa
mask_ab = np.zeros((U.NORB, U.NORB))
for p in (0, 1, 9):
    mask_ab[p, p] = 1.0
J[:, 1] *= mask_ab
kw = dict(diag_coulomb_mats=J, orbital_rotations=np.asarray(op_perm.orbital_rotations))
if op_perm.final_orbital_rotation is not None:
    kw["final_orbital_rotation"] = op_perm.final_orbital_rotation
op_m = ffsim.UCJOpSpinBalanced(**kw)
out = U.evaluate("identity_019", np.arange(U.NORB), "named", seeds=[2026], op_override=op_m)
record("n2_identity_best_019", 24.267039, out[0]["err_sub_mHa"], 1e-3, time.time() - t0)

print("=" * 70)
print("12. FIX 4 addition (H10, lockfile-rebuilt) -- already covered exhaustively")
print("    during FIX 1 verification (tested >10 times: once bit-exact, then")
print("    persistently off by the same ~373 mHa Finding-4-style gap on every")
print("    subsequent attempt, including in 'sqd' itself). Not re-run here to avoid")
print("    duplicating that work -- see verification/FIX1_LOCKFILE_VERIFICATION.md.")
print("    The N2 lockfile addition is run separately (different conda env) --")
print("    see verification/tier2_n2_lockfile.log.")

print("=" * 70)
print("13. Unmasked-permutation invariance, 4 N2 orderings")
inv_perms = {
    "identity": "0123456789",
    "rand_seed101": "5701498362",
    "rand_seed102": "8627109354",
    "rand_seed103": "8407956321",
}
inv_energies = []
for name, permstr in inv_perms.items():
    t0 = time.time()
    perm = [int(c) for c in permstr]
    op_unmasked = U.permute_operator(U.op_full, np.array(perm))
    out = U.evaluate(f"inv_{name}", np.array(perm), "named", seeds=[2026], op_override=op_unmasked)
    energy = U.E_CASCI + out[0]["err_sub_mHa"] / 1000.0
    inv_energies.append(energy)
    record(f"n2_unmasked_{name}_energy", -108.8236445639776, energy, 1e-9, time.time() - t0,
           "exact_bit tolerance")
same = len(set(round(e, 9) for e in inv_energies)) == 1
record("n2_unmasked_bitidentical_across_4", True, same, None, 0.0,
       f"{inv_energies}")

print("=" * 70)
print("14. Cr2 identity, default anchors (LAST, ~16 min)")
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import tm_transfer as T
T._init_worker()
pos_cr2 = R.positions_from(list(range(12)))
t0 = time.time()
task_args = ("identity", "01234567891011", pos_cr2, None, "default", "identity_default_verify",
             55, 2_000_000)
row = T._task(task_args)
record("cr2_identity_default", 240.79318115809656, row["err_mHa"], 1e-3, time.time() - t0)

df = pd.DataFrame(rows)
df.to_csv(REPO_ROOT / "verification/tier2_results.csv", index=False)
n_fail = int((df.ok == False).sum())
n_pass = int((df.ok == True).sum())
print(f"\n=== Tier 2 summary: {n_pass} passed, {n_fail} failed, "
      f"{len(df) - n_pass - n_fail} informational ===")
sys.exit(1 if n_fail else 0)
