#!/usr/bin/env python3
"""
experiments/h10_baseline.py
============================

H10 random-ordering baseline against the cached R=1.6 reference. Reuses
run_ordering_pipeline.py (mechanism B: operator never rotated, permutation
selects which orbital pairs enter ffsim's interaction_pairs) for operator
construction, sampling, determinant writing, and the sbd call - nothing
reimplemented except a depth/two-qubit-count probe (transpile only, no
sampling - sample_bitstrings doesn't expose the transpiled circuit).

Does NOT recompute the H10 reference: fails loudly if cache/h10_R1.6 is
missing rather than building it silently.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"

BUDGET = 15
SHOTS = 2_000_000
SEEDS = (2026, 7)
RNG_SEED = 20260825
N_RANDOM = 50

N2_RANDOM_RANGE = (21.65, 173.23)
N2_BETWEEN_WITHIN = 27.2
N2_CAPTURED_RHO = -0.880


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def circuit_stats(op, norb: int, nelec: tuple, R) -> tuple[int, int]:
    """(depth, two_qubit_gate_count) for the transpiled circuit - same
    construction as R.sample_bitstrings up to (not including) sim.run(),
    since that function doesn't return the transpiled circuit object.
    """
    import ffsim
    import ffsim.qiskit as fq
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit_aer import AerSimulator

    qr = QuantumRegister(2 * norb, "q")
    qc = QuantumCircuit(qr)
    qc.append(fq.PrepareHartreeFockJW(norb, nelec), qr)
    qc.append(fq.UCJOpSpinBalancedJW(op), qr)
    qc.measure_all()

    sim = AerSimulator(seed_simulator=0)
    tkw = dict(seed_transpiler=R.CFG["seed_transpiler"], optimization_level=1)
    if R.CFG["use_pre_init"]:
        tkw["pre_init"] = fq.PRE_INIT
    try:
        tqc = transpile(qc, sim, **tkw)
    except TypeError:
        tkw.pop("pre_init", None)
        tqc = transpile(qc, sim, **tkw)
    two_q = sum(1 for instr in tqc.data if len(instr.qubits) == 2)
    return int(tqc.depth()), two_q


def main() -> int:
    banner("H10 RANDOM BASELINE -- R=1.6, CAS(10,10), STO-6G (mechanism B)")
    import unified_run as U  # only for the shared sbd binary path
    import run_ordering_pipeline as R
    from pyscf.fci import cistring

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at expected path {R.CFG['sbd_bin']}")

    ref_path = CACHEDIR / "reference.npz"
    if not ref_path.exists():
        sys.exit(f"FATAL: no cached H10 reference at {CACHEDIR} (expected {ref_path}). "
                 f"Not recomputing - build it explicitly first if this is intentional.")
    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(CACHEDIR))
    print(f"[ref] loaded {CACHEDIR}  E_CASCI={ref['E_CASCI']:.12f}")

    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    fcidump_path = Path(ref["fcidump_path"])
    centroids = ref["centroids"]
    E_CASCI = ref["E_CASCI"]
    hf = R.hf_bitstring(norb, nocc)

    amp = R.Amplitudes(t1L, t2L, nocc, norb)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))
    w_ss = float(np.abs(Jaa).sum() / (np.abs(Jaa).sum() + np.abs(Jab).sum()))

    strs = cistring.make_strings(range(norb), nocc)
    dim_full = len(strs)
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
    W = np.asarray(ref["ci"]).reshape(dim_full, dim_full) ** 2
    W /= W.sum()

    print(f"H10 CAS({2*nocc},{norb}) @ R=1.6 A, sto-6g")
    print(f"FCIDUMP: {fcidump_path}  sha256={sha256_of(fcidump_path)[:16]}")

    # --------------------------------------------------------------- orderings
    phys = R.parse_permutation(ref["orderings"]["physical"]["perm"], norb)
    orderings: dict[str, np.ndarray] = {
        "identity": np.arange(norb),
        "reverse": np.arange(norb)[::-1].copy(),
        "physical": phys,
        "physical_reverse": phys[::-1].copy(),
        "s1_max": R.parse_permutation(ref["orderings"]["s1_max"]["perm"], norb),
        "s2_max": R.parse_permutation(ref["orderings"]["s2_max"]["perm"], norb),
        "retainedJ_max": R.parse_permutation(ref["orderings"]["retainedJ_max"]["perm"], norb),
    }
    named_names = set(orderings.keys())

    rng = np.random.default_rng(RNG_SEED)
    seen = {tuple(p.tolist()) for p in orderings.values()}
    randoms = {}
    while len(randoms) < N_RANDOM:
        p = rng.permutation(norb)
        if tuple(p.tolist()) not in seen:
            seen.add(tuple(p.tolist()))
            randoms[f"rand{len(randoms):03d}"] = p
    orderings.update(randoms)
    random_names = set(randoms.keys())

    print(f"{len(orderings)} orderings ({len(named_names)} named + {len(randoms)} random) "
          f"x {len(SEEDS)} seeds = {len(orderings) * len(SEEDS)} evaluations")

    # ---------------------------------------------------------- main sweep
    csv_path = OUTDIR / "h10_baseline_results.csv"
    rows: list[dict] = []
    t0 = time.time()
    n_total = len(orderings) * len(SEEDS)
    n_done = 0
    n_collapse = 0

    for name, perm in orderings.items():
        pos = R.positions_from(perm)
        pairs = R.interaction_pairs_for(pos, centroids, J_ab=Jab)
        op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)

        retained_J = R.retained_J_of(pos, Jaa, Jab)
        s1_amp = R.score1(pos, amp, Jaa, Jab, w_ss)["s1_amp"]
        s2 = R.score2(pos, amp, w_ss)["s2"]
        depth_ref, two_q_ref = circuit_stats(op, norb, nelec, R)

        depths_seen = set()
        for seed in SEEDS:
            a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, SHOTS, seed)
            depths_seen.add(depth)
            assert depth == depth_ref, (
                f"{name}: transpiled depth {depth} (seed {seed}) != {depth_ref} "
                f"(depth-probe circuit) for the SAME operator - transpilation is "
                f"not deterministic under a fixed seed_transpiler."
            )

            a_sel, n_uniq_a = R.top_dets(a_c, BUDGET, hf)
            b_sel, n_uniq_b = R.top_dets(b_c, BUDGET, hf)
            dim_a, dim_b = len(a_sel), len(b_sel)

            row = dict(ordering=name, permutation="".join(map(str, perm)), seed=seed,
                      dim_alpha=dim_a, dim_beta=dim_b, dim=dim_a * dim_b,
                      n_unique_alpha=n_uniq_a, n_unique_beta=n_uniq_b,
                      depth=depth, two_qubit_count=two_q_ref,
                      retained_J=retained_J, s1_amp=s1_amp, s2=s2)

            if dim_a < BUDGET or dim_b < BUDGET:
                row.update(energy=float("nan"), err_mHa=float("nan"),
                          captured=float("nan"), status="SUPPORT_COLLAPSE")
                n_collapse += 1
            else:
                adet_path = OUTDIR / f"_{name}_{seed}_a.txt"
                bdet_path = OUTDIR / f"_{name}_{seed}_b.txt"
                adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
                bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
                energy = R.run_sbd(str(fcidump_path), str(adet_path), str(bdet_path), norb)
                err_mha = (energy - E_CASCI) * 1000.0
                ia = [b2i[d] for d in a_sel]
                ib = [b2i[d] for d in b_sel]
                captured = float(W[np.ix_(ia, ib)].sum())
                row.update(energy=energy, err_mHa=err_mha, captured=captured, status="OK")

            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            n_done += 1
            el = time.time() - t0
            err_str = "nan" if row["status"] != "OK" else f"{row['err_mHa']:7.2f}"
            print(f"[{n_done:3d}/{n_total}] {name:<18} seed={seed:<5} "
                  f"status={row['status']:<16} err={err_str} mHa  "
                  f"eta={el/n_done*(n_total-n_done)/60:.1f}m", flush=True)

        assert len(depths_seen) == 1, (
            f"{name}: transpiled depth varied across seeds {sorted(depths_seen)} "
            f"for the SAME permutation - only the sampling seed should differ."
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nCircuit depth/two-qubit-count logged per ordering; "
          f"depth asserted identical across seeds for every ordering.")

    # =========================================================== ANALYSIS
    banner("ANALYSIS")
    ok = df[df.status == "OK"].copy()
    ok["err_mHa"] = ok["err_mHa"].astype(float)

    print(f"\n1. Random baseline")
    print("-" * 78)
    rnd_ok = ok[ok.ordering.isin(random_names)]
    rnd_by_ord = rnd_ok.groupby("ordering")["err_mHa"].mean()
    n_random_kept = rnd_by_ord.shape[0]
    print(f"  n kept = {n_random_kept} / {N_RANDOM}")
    if n_random_kept:
        print(f"  best={rnd_by_ord.min():.2f}  median={rnd_by_ord.median():.2f}  "
              f"mean={rnd_by_ord.mean():.2f}  worst={rnd_by_ord.max():.2f} mHa")
        print(f"  spread: {rnd_by_ord.max() - rnd_by_ord.min():.2f} mHa "
              f"(range {rnd_by_ord.min():.2f} - {rnd_by_ord.max():.2f})")
    print(f"  N2 comparison: {N2_RANDOM_RANGE[0]:.2f} - {N2_RANDOM_RANGE[1]:.2f} mHa")

    print(f"\n2. Between/within variance ratio (all orderings, non-collapsed)")
    print("-" * 78)
    piv = ok.pivot_table(index="ordering", columns="seed", values="err_mHa")
    piv = piv.dropna()
    between = piv.mean(axis=1).var()
    within = piv.var(axis=1).mean()
    ratio = between / within if within > 0 else float("inf")
    print(f"  n orderings with both seeds present: {len(piv)}")
    print(f"  variance BETWEEN orderings (signal): {between:.2f} mHa^2")
    print(f"  variance WITHIN orderings (noise):   {within:.2f} mHa^2")
    print(f"  ratio: {ratio:.2f}   (N2 gave {N2_BETWEEN_WITHIN})")
    if ratio < 5:
        print(f"  *** RATIO BELOW 5: percentile claims below are NOT reliably "
              f"distinguishable from seed noise. ***")

    print(f"\n3. Named orderings")
    print("-" * 78)
    rnd_err = rnd_by_ord.to_numpy()
    print(f"{'ordering':<18}{'mean_err_mHa':>14}{'sd':>10}{'percentile_in_random':>22}")
    for name in ["identity", "reverse", "physical", "physical_reverse",
                "s1_max", "s2_max", "retainedJ_max"]:
        sub = ok[ok.ordering == name]["err_mHa"]
        if len(sub) == 0:
            print(f"{name:<18}{'SUPPORT_COLLAPSE (all seeds)':>46}")
            continue
        mean_e, sd_e = sub.mean(), sub.std()
        pct = 100.0 * (rnd_err > mean_e).mean() if len(rnd_err) else float("nan")
        print(f"{name:<18}{mean_e:>14.2f}{sd_e:>10.2f}{pct:>21.1f}%")

    print(f"\n4. Ideal capture vs achieved capture")
    print("-" * 78)
    IDEAL_CAPTURE = 0.7554
    cap_rnd = rnd_ok.groupby("ordering")["captured"].mean()
    if len(cap_rnd):
        print(f"  ideal (oracle) capture: {IDEAL_CAPTURE:.4f}")
        print(f"  mean achieved capture (random): {cap_rnd.mean():.4f}")
        best_cap = cap_rnd.max()
        print(f"  best achieved capture (random): {best_cap:.4f}  "
              f"({100*best_cap/IDEAL_CAPTURE:.1f}% of ceiling)")

    print(f"\n5. Spearman rho: captured vs err_mHa (random orderings)")
    print("-" * 78)
    merged = rnd_by_ord.to_frame("err_mHa").join(cap_rnd.to_frame("captured"))
    if len(merged) > 2:
        sr = spearmanr(merged["captured"], merged["err_mHa"])
        print(f"  rho = {sr.statistic:+.3f}   p = {sr.pvalue:.2e}   (N2 gave {N2_CAPTURED_RHO})")
    else:
        print("  not enough non-collapsed random orderings for a meaningful rho")

    print(f"\n6. Score predictors: rho, regret, vs random-selection regret")
    print("-" * 78)
    rnd_scores = ok[ok.ordering.isin(random_names)].groupby("ordering").agg(
        err_mHa=("err_mHa", "mean"), retained_J=("retained_J", "first"),
        s1_amp=("s1_amp", "first"), s2=("s2", "first"))
    err = rnd_scores["err_mHa"].to_numpy()
    rand_regret = float(np.nanmean(err) - np.nanmin(err)) if len(err) else float("nan")
    print(f"  random-selection regret (mean - best) = {rand_regret:.2f} mHa")
    print(f"{'score':<14}{'rho':>9}{'p':>11}{'picks':>18}{'regret_mHa':>12}{'vs_random':>11}")
    for col in ("s1_amp", "s2", "retained_J"):
        x = rnd_scores[col].to_numpy(float)
        if np.allclose(x, x[0]) or len(x) < 3:
            print(f"{col:<14}   insufficient variation / data")
            continue
        sr = spearmanr(x, err)
        k = int(np.nanargmax(x))
        pick = rnd_scores.index[k]
        reg = float(err[k] - np.nanmin(err))
        print(f"{col:<14}{sr.statistic:>+9.3f}{sr.pvalue:>11.1e}{str(pick):>18}"
              f"{reg:>12.2f}{reg/rand_regret if rand_regret else float('nan'):>11.2f}")

    print(f"\n7. Kendall tau: s1_max permutation vs physical permutation")
    print("-" * 78)
    p_s1 = orderings["s1_max"]
    p_phys = orderings["physical"]
    kt = kendalltau(p_s1, p_phys)
    print(f"  s1_max = {''.join(map(str, p_s1))}")
    print(f"  physical = {''.join(map(str, p_phys))}")
    print(f"  tau = {kt.statistic:+.3f}   p = {kt.pvalue:.2e}")

    # --------------------------------------------------------------- HEADLINE
    banner("HEADLINE")
    s1max_err = ok[ok.ordering == "s1_max"]["err_mHa"].mean()
    s1max_pct = 100.0 * (rnd_err > s1max_err).mean() if len(rnd_err) and not np.isnan(s1max_err) else float("nan")
    cap_rho_val = spearmanr(merged["captured"], merged["err_mHa"]).statistic if len(merged) > 2 else float("nan")
    cap_rho_p = spearmanr(merged["captured"], merged["err_mHa"]).pvalue if len(merged) > 2 else float("nan")
    print(f"s1_max percentile in random distribution: {s1max_pct:.1f}%")
    print(f"capture still predicts error on H10: rho={cap_rho_val:+.3f}, p={cap_rho_p:.2e}")
    print(f"between/within variance ratio: {ratio:.2f}")
    print(f"support collapses: {n_collapse}")

    # ------------------------------------------------------------- metadata
    metadata = dict(
        subcommand="h10_baseline", R=1.6, shots=SHOTS, seeds=list(SEEDS),
        rng_seed=RNG_SEED, n_named=len(named_names), n_random=N_RANDOM,
        git_commit=__import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip(),
        fcidump_sha256=sha256_of(fcidump_path),
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        orderings_json_sha256=sha256_of(CACHEDIR / "orderings.json"),
        metadata_json_sha256=sha256_of(CACHEDIR / "metadata.json"),
        n_support_collapse=n_collapse,
        random_ordering_names=sorted(random_names),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        between_within_ratio=float(ratio),
        s1_max_percentile=float(s1max_pct),
        captured_rho=float(cap_rho_val), captured_p=float(cap_rho_p),
    )
    with open(OUTDIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {csv_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
