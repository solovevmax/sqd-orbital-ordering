"""Task A5: does raising the shot count on the FRESH (rebuilt) H10 reference
converge err_mHa toward the cached value (300.32), or not?
"""
import sys, time
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import unified_run as U
import run_ordering_pipeline as R

R.CFG["sbd_bin"] = str(U.SBD)

FRESH_CACHE = "/private/tmp/claude-502/-Users-maxim-sqd-project/fe23625e-57cc-474d-8afb-d2c2063922ce/scratchpad/cold_start/sqd-project/cache/h10_R1.6"
ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=FRESH_CACHE)
norb, nocc = ref["norb"], ref["nocc"]
nelec = (nocc, nocc)
t1L, t2L = ref["t1L"], ref["t2L"]
centroids = ref["centroids"]
E_CASCI = ref["E_CASCI"]
fcidump_path = ref["fcidump_path"]
hf = R.hf_bitstring(norb, nocc)

Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))
pos = R.positions_from(np.arange(norb))
pairs = R.interaction_pairs_for(pos, centroids, J_ab=Jab)
op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)

rows = []
for shots in (2_000_000, 8_000_000, 32_000_000):
    t0 = time.time()
    a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, shots, 2026)
    a_sel, _ = R.top_dets(a_c, 15, hf)
    b_sel, _ = R.top_dets(b_c, 15, hf)
    import pathlib
    ap = pathlib.Path(f"_a5_{shots}_a.txt")
    bp = pathlib.Path(f"_a5_{shots}_b.txt")
    ap.write_text("\n".join(sorted(a_sel)) + "\n")
    bp.write_text("\n".join(sorted(b_sel)) + "\n")
    energy = R.run_sbd(str(fcidump_path), str(ap), str(bp), norb)
    err_mHa = (energy - E_CASCI) * 1000.0
    ap.unlink(); bp.unlink()
    rows.append(dict(shots=shots, err_mHa=err_mHa, wall_s=time.time() - t0))
    print(f"shots={shots:>10d}  err_mHa={err_mHa:.6f}  wall={time.time()-t0:.1f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/private/tmp/claude-502/-Users-maxim-sqd-project/fe23625e-57cc-474d-8afb-d2c2063922ce/scratchpad/cold_start/task_a5.csv", index=False)
print(f"\nreference (cached, 2e6 shots) value: 300.31919403956664 mHa")
