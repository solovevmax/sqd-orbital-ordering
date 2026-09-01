"""Task A3/A4: perturb the CCSD t1/t2 amplitude tensors (the "one-body" /
"two-body" objects that parameterize the LUCJ circuit -- t1 has 2 indices,
t2 has 4) by relative Gaussian noise at several levels, holding the exact
energy reference (E_CASCI, FCIDUMP) fixed at the cached value, and see how
often/how badly the determinant-selection boundary flips.

Sampling seed is held fixed at 2026 throughout (matching the report's own
protocol) so the only thing varying is the perturbation itself; 5
independent noise draws per level via 5 perturbation seeds.
"""
import sys, time
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import unified_run as U
import run_ordering_pipeline as R

R.CFG["sbd_bin"] = str(U.SBD)

def evaluate_system(system, ref, t1_pert, t2_pert, adet_prefix):
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    centroids = ref["centroids"]
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2_pert, t1_pert))
    pos = R.positions_from(np.arange(norb))
    pairs = R.interaction_pairs_for(pos, centroids, J_ab=Jab)
    op = R.build_ucj(t2_pert, t1_pert, interaction_pairs=pairs)
    BUDGET, SHOTS, SEED = 15, 2_000_000, 2026
    a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, SHOTS, SEED)
    hf = R.hf_bitstring(norb, nocc)
    a_sel, _ = R.top_dets(a_c, BUDGET, hf)
    b_sel, _ = R.top_dets(b_c, BUDGET, hf)
    fcidump_path = ref["fcidump_path"]
    E_CASCI = ref["E_CASCI"]
    import pathlib
    ap = pathlib.Path(f"_{adet_prefix}_a.txt")
    bp = pathlib.Path(f"_{adet_prefix}_b.txt")
    ap.write_text("\n".join(sorted(a_sel)) + "\n")
    bp.write_text("\n".join(sorted(b_sel)) + "\n")
    energy = R.run_sbd(str(fcidump_path), str(ap), str(bp), norb)
    err_mHa = (energy - E_CASCI) * 1000.0
    ap.unlink(); bp.unlink()
    return err_mHa

def run_sweep(system, cachedir, build_ref_kwargs, label):
    ref = R.build_or_load_h10_reference(*build_ref_kwargs, cachedir=cachedir) if system == "H10" else None
    t1_base, t2_base = ref["t1L"], ref["t2L"]

    rows = []
    t0 = time.time()
    # noise level 0 (unperturbed) baseline, once
    err0 = evaluate_system(system, ref, t1_base, t2_base, f"{label}_base")
    rows.append(dict(system=system, noise_level=0.0, pert_seed=-1, err_mHa=err0))
    print(f"[{label}] noise=0 (baseline) err_mHa={err0:.6f}  ({time.time()-t0:.0f}s)", flush=True)

    for noise in (1e-14, 1e-12, 1e-10, 1e-8):
        for pseed in range(5):
            rng = np.random.default_rng(1000 * int(-np.log10(noise)) + pseed)
            t1p = t1_base * (1 + rng.normal(0, noise, size=t1_base.shape))
            t2p = t2_base * (1 + rng.normal(0, noise, size=t2_base.shape))
            err = evaluate_system(system, ref, t1p, t2p, f"{label}_{noise:.0e}_{pseed}")
            rows.append(dict(system=system, noise_level=noise, pert_seed=pseed, err_mHa=err))
            print(f"[{label}] noise={noise:.0e} seed={pseed} err_mHa={err:.6f}  "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)

print("=== A3: H10 identity, default anchors, cached reference ===")
df_h10 = run_sweep("H10", "cache/h10_R1.6", (1.6, 10, "sto-6g"), "h10")
df_h10.to_csv("/private/tmp/claude-502/-Users-maxim-sqd-project/fe23625e-57cc-474d-8afb-d2c2063922ce/scratchpad/cold_start/task_a3_h10.csv", index=False)

print("\n=== A3 summary (H10) ===")
print(df_h10.groupby("noise_level")["err_mHa"].agg(["count", "mean", "std", "min", "max"]))
