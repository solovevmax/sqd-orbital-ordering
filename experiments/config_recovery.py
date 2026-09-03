#!/usr/bin/env python3
"""experiments/config_recovery.py

Does the layout effect survive iterative SQD (self-consistent configuration
recovery), or is it a first-iteration artefact? See
experiments/outputs/config_recovery/README.md for the full protocol,
mechanism (R0), and pre-registered prediction (Amendment 4).

System: cached H10 R=1.6. 2e6 shots, seed 2026, budget 15/sector at
iteration 0. Recovery via sbd --carryover_type 3 (literature-standard
self-consistent recovery: amplitude-based dominant-determinant selection +
full singles expansion), threshold 1e-2 (calibrated: the strictest of
{1e-2,1e-3,1e-4} that keeps iteration-1 dimension under ~5000; the real
5-iteration trajectory on identity converges to a stable 9216-dim fixed
point by iteration 2 and never approaches the 63,504 full-CI ceiling).

Two arms:
  A. Anchor axis at identity: default anchors, best anchors (0,1,2), the
     no-alpha-beta control -- 3 trajectories, run FIRST (Amendment 5).
  B. Permutation axis: 8 layouts at default anchors (identity, physical,
     physical_reverse, rand030, rand029, and 3 randomly-seeded held-out
     chains) -- run second, truncated if time demands.
"""
import sys, time, json, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from pyscf.fci import cistring

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_ordering_pipeline as R
import unified_run as U

SBD_BIN = str(U.SBD)
MPIRUN = "mpirun"
R.CFG["sbd_bin"] = SBD_BIN

OUTDIR = REPO_ROOT / "experiments" / "outputs" / "config_recovery"
OUTDIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTDIR / "recovery_results.csv"

N_ITER = 5
THRESHOLD = "1e-2"
BUDGET0 = 15
SHOTS = 2_000_000
SEED = 2026


def parse_energy(text):
    m = re.search(r"diagonalization: Energy = ([-\d.]+)", text)
    return float(m.group(1)) if m else None


def sbd_carryover(fcidump, adet, bdet, norb, threshold, co_adet_path, co_bdet_path, timeout=1800):
    cmd = [MPIRUN, "-n", "1", SBD_BIN,
           "--fcidump", fcidump, "--adetfile", adet, "--bdetfile", bdet,
           "--bit_length", str(max(20, norb)),
           "--method", "0", "--iteration", "200", "--tolerance", "1e-10",
           "--carryover_type", "3", "--carryover_threshold", str(threshold),
           "--carryover_adetfile", str(co_adet_path), "--carryover_bdetfile", str(co_bdet_path),
           "--shuffle", "0", "--init", "0",
           "--adet_comm_size", "1", "--bdet_comm_size", "1", "--task_comm_size", "1"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    energy = parse_energy(p.stdout + p.stderr)
    if energy is None:
        print("STDOUT TAIL:", "\n".join((p.stdout + p.stderr).splitlines()[-30:]))
    return energy, wall


# ---------------------------------------------------------------- H10 ref --
ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir="cache/h10_R1.6")
norb, nocc = ref["norb"], ref["nocc"]
nelec = (nocc, nocc)
t1L, t2L = ref["t1L"], ref["t2L"]
centroids = ref["centroids"]
E_CASCI = ref["E_CASCI"]
fcidump_path = str(ref["fcidump_path"])
hf = R.hf_bitstring(norb, nocc)
Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))

strs = cistring.make_strings(range(norb), nocc)
b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
dim_full = len(strs)
W = np.abs(np.asarray(ref["ci"])).reshape(dim_full, dim_full) ** 2
W = W / W.sum()


def captured_of(a_list, b_list):
    ia = [b2i[d] for d in a_list if d in b2i]
    ib = [b2i[d] for d in b_list if d in b2i]
    if not ia or not ib:
        return float("nan")
    return float(W[np.ix_(ia, ib)].sum())


def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    u = len(set1 | set2)
    return len(set1 & set2) / u if u else 1.0


rows = []


def write_csv():
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)


def run_recovery(label, arm, pos, anchor_kwargs, n_iter=N_ITER):
    """Iteration 0: Aer-sampled top-15/sector at this layout/anchor choice.
    Iterations 1..n_iter: sbd carryover_type=3 self-consistent recovery."""
    print(f"\n===== {label} ({arm}) =====", flush=True)
    pairs = R.interaction_pairs_for(pos, centroids, J_ab=Jab, **anchor_kwargs)
    op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
    a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, SHOTS, SEED)
    a0, _ = R.top_dets(a_c, BUDGET0, hf)
    b0, _ = R.top_dets(b_c, BUDGET0, hf)
    a0_set, b0_set = set(a0), set(b0)

    cur_a, cur_b = sorted(a0), sorted(b0)
    prev_a_set, prev_b_set = set(cur_a), set(cur_b)

    for it in range(n_iter + 1):
        t0 = time.time()
        adet_path = OUTDIR / f"_{label}_{it}_a.txt"
        bdet_path = OUTDIR / f"_{label}_{it}_b.txt"
        adet_path.write_text("\n".join(cur_a) + "\n")
        bdet_path.write_text("\n".join(cur_b) + "\n")

        co_a_path = OUTDIR / f"_{label}_{it+1}_co_a.txt"
        co_b_path = OUTDIR / f"_{label}_{it+1}_co_b.txt"
        energy, wall = sbd_carryover(fcidump_path, str(adet_path), str(bdet_path), norb,
                                      THRESHOLD, co_a_path, co_b_path)
        err_mHa = (energy - E_CASCI) * 1000.0 if energy is not None else float("nan")
        cur_a_set, cur_b_set = set(cur_a), set(cur_b)
        cap = captured_of(cur_a, cur_b)
        jac_prev_a = jaccard(cur_a_set, prev_a_set)
        jac_prev_b = jaccard(cur_b_set, prev_b_set)
        jac_iter0_a = jaccard(cur_a_set, a0_set)
        jac_iter0_b = jaccard(cur_b_set, b0_set)
        frac_traced_a = len(cur_a_set & a0_set) / len(cur_a_set) if cur_a_set else float("nan")
        frac_traced_b = len(cur_b_set & b0_set) / len(cur_b_set) if cur_b_set else float("nan")

        row = dict(label=label, arm=arm, iteration=it, err_mHa=err_mHa, captured=cap,
                   dim_a=len(cur_a), dim_b=len(cur_b), dim=len(cur_a) * len(cur_b),
                   jaccard_prev_a=jac_prev_a, jaccard_prev_b=jac_prev_b,
                   jaccard_iter0_a=jac_iter0_a, jaccard_iter0_b=jac_iter0_b,
                   frac_traced_a=frac_traced_a, frac_traced_b=frac_traced_b,
                   wall_s=wall)
        rows.append(row)
        write_csv()
        print(f"  it={it}: err_mHa={err_mHa:.4f} dim={len(cur_a)*len(cur_b)} "
              f"(dim_a={len(cur_a)},dim_b={len(cur_b)}) captured={cap:.6f} "
              f"traced_a={frac_traced_a:.3f} traced_b={frac_traced_b:.3f} "
              f"wall={wall:.1f}s  [{time.time()-t0:.1f}s total]", flush=True)

        adet_path.unlink(); bdet_path.unlink()

        prev_a_set, prev_b_set = cur_a_set, cur_b_set
        if it < n_iter:
            if not co_a_path.exists() or not co_b_path.exists():
                print(f"  WARNING: carryover files missing at iteration {it}, stopping early")
                break
            cur_a = sorted(set(co_a_path.read_text().split()))
            cur_b = sorted(set(co_b_path.read_text().split()))
    return rows


def main():
    t_start = time.time()
    pos_id = R.positions_from(np.arange(norb))

    print("=" * 70)
    print("ARM A: anchor axis at identity (Amendment 5 -- run first)")
    print("=" * 70)
    run_recovery("anchor_default", "anchor_axis", pos_id, dict(anchor_offset=0))
    run_recovery("anchor_best_012", "anchor_axis", pos_id, dict(anchor_orbitals=(0, 1, 2)))
    run_recovery("anchor_noab", "anchor_axis", pos_id, dict(anchor_orbitals=()))
    print(f"\nArm A wall time so far: {(time.time()-t_start)/60:.1f} min")

    print("=" * 70)
    print("ARM B: permutation axis, 8 layouts, default anchors")
    print("=" * 70)
    phys_perm = R.parse_permutation(ref["orderings"]["physical"]["perm"], norb)
    phaseb_meta = json.loads((REPO_ROOT / "experiments/outputs/chain_aware/phaseB_metadata.json").read_text())
    new_chains = phaseb_meta["new_chains"]
    rng = np.random.default_rng(101)
    held_out_idx = sorted(rng.choice(12, 3, replace=False).tolist())
    held_out = [f"newchain{i:02d}" for i in held_out_idx]
    print(f"randomly selected held-out chains (seed=101): {held_out}")

    h10_results = pd.read_csv(REPO_ROOT / "experiments/outputs/h10_baseline_R1.6/h10_baseline_results.csv",
                               dtype={"permutation": str})
    def perm_of(name):
        return [int(c) for c in h10_results[h10_results.ordering == name].iloc[0].permutation]

    layouts = [
        ("identity", pos_id),
        ("physical", R.positions_from(phys_perm)),
        ("physical_reverse", R.positions_from(phys_perm[::-1].copy())),
        ("rand030", R.positions_from(perm_of("rand030"))),  # best baseline, 168.67
        ("rand029", R.positions_from(perm_of("rand029"))),  # worst baseline, 454.89
    ]
    for name in held_out:
        perm = [int(c) for c in new_chains[name]]
        layouts.append((name, R.positions_from(perm)))

    for name, pos in layouts:
        if pos is None:
            print(f"  SKIP {name}: same-spin permutation not resolved (see README note)")
            continue
        run_recovery(name, "permutation", pos, dict(anchor_offset=0))
        print(f"  cumulative wall time: {(time.time()-t_start)/60:.1f} min", flush=True)

    write_csv()
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")
    print(f"Rows written: {len(rows)} -> {CSV_PATH}")


if __name__ == "__main__":
    sys.exit(main())
