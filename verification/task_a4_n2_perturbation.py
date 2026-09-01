"""Task A4: same amplitude-perturbation sweep as A3, on N2 identity
(canonical orbitals, boundary ratio w16/w15=0.504 per the report, vs H10's
0.989), using unified_run.py's own evaluate() with op_override so the
per-evaluation machinery matches production exactly.
"""
import sys, time
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import ffsim
import unified_run as U

rows = []
t0 = time.time()

def run_one(t1p, t2p, label):
    op_pert = ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=t2p, t1=t1p, n_reps=None)
    op_m = U.apply_mask(U.permute_operator(op_pert, np.arange(U.NORB)))
    out = U.evaluate(f"identity_{label}", np.arange(U.NORB), "named", seeds=[2026], op_override=op_m)
    return out[0]["err_sub_mHa"]

t1_base, t2_base = U.ref_data["t1"], U.ref_data["t2"]

err0 = run_one(t1_base, t2_base, "base")
rows.append(dict(system="N2", noise_level=0.0, pert_seed=-1, err_mHa=err0))
print(f"[N2] noise=0 (baseline) err_mHa={err0:.6f}  ({time.time()-t0:.0f}s)", flush=True)

for noise in (1e-14, 1e-12, 1e-10, 1e-8):
    for pseed in range(5):
        rng = np.random.default_rng(1000 * int(-np.log10(noise)) + pseed)
        t1p = t1_base * (1 + rng.normal(0, noise, size=t1_base.shape))
        t2p = t2_base * (1 + rng.normal(0, noise, size=t2_base.shape))
        err = run_one(t1p, t2p, f"{noise:.0e}_{pseed}")
        rows.append(dict(system="N2", noise_level=noise, pert_seed=pseed, err_mHa=err))
        print(f"[N2] noise={noise:.0e} seed={pseed} err_mHa={err:.6f}  "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv("/private/tmp/claude-502/-Users-maxim-sqd-project/fe23625e-57cc-474d-8afb-d2c2063922ce/scratchpad/cold_start/task_a4_n2.csv", index=False)
print("\n=== A4 summary (N2) ===")
print(df.groupby("noise_level")["err_mHa"].agg(["count", "mean", "std", "min", "max"]))
