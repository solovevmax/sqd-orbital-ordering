#!/usr/bin/env python3
"""
experiments/anchor_decomposition.py
=====================================

PART B -- anchor decomposition. Separates the same-spin-ordering lever from
the opposite-spin anchor-selection lever, using the cached H10 R=1.6
reference. Reuses run_ordering_pipeline.py (operator construction, sampling,
determinant writing, sbd) and src/sqd_ordering/mask.py (now extended with
anchor_offset / anchor_orbitals params, backward-compatible default anchor
at offset 0 - see mask.py) - nothing reimplemented except the same
depth/two-qubit-count transpile-only probe used in h10_baseline.py and
score_audit.py's reuse pattern.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "anchor_decomposition_R1.6"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "h10_R1.6"
BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"

BUDGET = 15
SHOTS = 2_000_000
B1_ORDERINGS = ["identity", "physical", "physical_reverse", "s2_max"]
B2_RNG_SEED = 20260825001
BASELINE_SPREAD = 286.23
BASELINE_IDENTITY_ERR = 300.32


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def circuit_stats(op, norb: int, nelec: tuple, R) -> tuple[int, int]:
    """(depth, two_qubit_gate_count) - transpile only, no sampling."""
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


def evaluate(R, t1L, t2L, pos, norb, nelec, nocc, hf, fcidump_path, E_CASCI, b2i, W,
            seed, anchor_offset=0, anchor_orbitals=None, tag=""):
    pairs = R.interaction_pairs_for(pos, anchor_offset=anchor_offset, anchor_orbitals=anchor_orbitals)
    op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
    depth, two_q = circuit_stats(op, norb, nelec, R)
    a_c, b_c, depth_seed = R.sample_bitstrings(op, norb, nelec, SHOTS, seed)
    assert depth_seed == depth, f"{tag}: depth mismatch {depth_seed} != {depth}"
    a_sel, n_uniq_a = R.top_dets(a_c, BUDGET, hf)
    b_sel, n_uniq_b = R.top_dets(b_c, BUDGET, hf)
    dim_a, dim_b = len(a_sel), len(b_sel)
    row = dict(dim_alpha=dim_a, dim_beta=dim_b, n_unique_alpha=n_uniq_a,
              n_unique_beta=n_uniq_b, depth=depth, two_qubit_count=two_q)
    if dim_a < BUDGET or dim_b < BUDGET:
        row.update(energy=float("nan"), err_mHa=float("nan"), captured=float("nan"),
                  status="SUPPORT_COLLAPSE")
    else:
        adet_path = OUTDIR / f"_{tag}_a.txt"
        bdet_path = OUTDIR / f"_{tag}_b.txt"
        adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
        bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
        energy = R.run_sbd(str(fcidump_path), str(adet_path), str(bdet_path), norb)
        err_mha = (energy - E_CASCI) * 1000.0
        ia = [b2i[d] for d in a_sel]
        ib = [b2i[d] for d in b_sel]
        captured = float(W[np.ix_(ia, ib)].sum())
        row.update(energy=energy, err_mHa=err_mha, captured=captured, status="OK")
    return row


def main() -> int:
    banner("PART B -- ANCHOR DECOMPOSITION")
    import unified_run as U
    import run_ordering_pipeline as R
    from pyscf.fci import cistring
    from sqd_ordering import mask

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at {R.CFG['sbd_bin']}")

    ref_path = CACHEDIR / "reference.npz"
    if not ref_path.exists():
        sys.exit(f"FATAL: no cached H10 reference at {CACHEDIR}. Not recomputing.")
    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(CACHEDIR))
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    fcidump_path = Path(ref["fcidump_path"])
    E_CASCI = ref["E_CASCI"]
    hf = R.hf_bitstring(norb, nocc)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))

    strs = cistring.make_strings(range(norb), nocc)
    dim_full = len(strs)
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
    W = np.asarray(ref["ci"]).reshape(dim_full, dim_full) ** 2
    W /= W.sum()

    if not BASELINE_CSV.exists():
        sys.exit(f"FATAL: baseline results not found at {BASELINE_CSV}.")
    base = pd.read_csv(BASELINE_CSV)

    # =========================================================== B0
    banner("B0 -- reversal-invariance of the same-spin pair set (symbolic, no sampling)")
    perm_by_ordering = base.groupby("ordering")["permutation"].first()
    n_match = 0
    n_total = 0
    for name, permstr in perm_by_ordering.items():
        perm = R.parse_permutation(permstr, norb)
        perm_rev = perm[::-1].copy()
        pos = R.positions_from(perm)
        pos_rev = R.positions_from(perm_rev)
        s1 = mask.same_spin_pairs(pos, norb)
        s2 = mask.same_spin_pairs(pos_rev, norb)
        n_total += 1
        if s1 == s2:
            n_match += 1
        else:
            print(f"  MISMATCH: {name}  |s1|={len(s1)} |s2|={len(s2)}  "
                  f"diff={s1 ^ s2}")
    print(f"  {n_match}/{n_total} permutations: same-spin pair set is reversal-invariant")
    if n_match != n_total:
        sys.exit(f"STOP: reversal-invariance premise FAILS for {n_total - n_match} "
                 f"permutation(s). Do not proceed to sampling - the B1/B2 rationale "
                 f"assumes this holds for all baseline orderings.")

    # =========================================================== B1
    banner("B1 -- anchor offset sweep at fixed same-spin ordering")
    perms = {name: R.parse_permutation(perm_by_ordering[name], norb) for name in B1_ORDERINGS}
    b1_rows = []
    t0 = time.time()
    n_b1 = len(B1_ORDERINGS) * 4
    i_done = 0
    for name in B1_ORDERINGS:
        perm = perms[name]
        pos = R.positions_from(perm)
        for offset in (0, 1, 2, 3):
            tag = f"b1_{name}_off{offset}"
            row = evaluate(R, t1L, t2L, pos, norb, nelec, nocc, hf, fcidump_path,
                          E_CASCI, b2i, W, seed=2026, anchor_offset=offset, tag=tag)
            rj = R.retained_J_of(pos, Jaa, Jab, anchor_offset=offset)
            rj_ss, rj_os = R.retained_J_split_of(pos, Jaa, Jab, anchor_offset=offset)
            row.update(ordering=name, permutation="".join(map(str, perm)), seed=2026,
                      anchor_offset=offset, retained_J=rj, retained_J_samespin=rj_ss,
                      retained_J_oppspin=rj_os)
            b1_rows.append(row)
            pd.DataFrame(b1_rows).to_csv(OUTDIR / "b1_offset_sweep.csv", index=False)
            i_done += 1
            print(f"[B1 {i_done}/{n_b1}] {name:<18} offset={offset}  "
                  f"err={row['err_mHa']:.2f} mHa  status={row['status']}")

    print("\nRegression check: offset=0, seed=7, compare against baseline CSV")
    n_repro_ok = 0
    for name in B1_ORDERINGS:
        perm = perms[name]
        pos = R.positions_from(perm)
        tag = f"b1_{name}_off0_seed7"
        row = evaluate(R, t1L, t2L, pos, norb, nelec, nocc, hf, fcidump_path,
                      E_CASCI, b2i, W, seed=7, anchor_offset=0, tag=tag)
        base_row = base[(base.ordering == name) & (base.seed == 7)].iloc[0]
        match = (row["status"] == "OK" and base_row["status"] == "OK"
                and abs(row["err_mHa"] - base_row["err_mHa"]) < 1e-6)
        print(f"  {name:<18} new_err={row['err_mHa']:.6f}  baseline_err={base_row['err_mHa']:.6f}  "
              f"{'MATCH' if match else 'MISMATCH'}")
        if match:
            n_repro_ok += 1
    if n_repro_ok != len(B1_ORDERINGS):
        sys.exit(f"STOP: only {n_repro_ok}/{len(B1_ORDERINGS)} orderings reproduced the "
                 f"baseline exactly at offset=0. The anchor-offset parameter is not "
                 f"cleanly separable from what the baseline ran.")
    print(f"  All {n_repro_ok}/{len(B1_ORDERINGS)} reproduced exactly.")

    b1_df = pd.DataFrame(b1_rows)
    banner("B1 -- key numbers")
    b1_ranges = {}
    for name in B1_ORDERINGS:
        sub = b1_df[(b1_df.ordering == name) & (b1_df.status == "OK")]
        rng = sub.err_mHa.max() - sub.err_mHa.min()
        b1_ranges[name] = rng
        best_off = sub.loc[sub.err_mHa.idxmin(), "anchor_offset"]
        print(f"  {name:<18} err range over offsets: {sub.err_mHa.min():.2f} - "
              f"{sub.err_mHa.max():.2f} mHa (range {rng:.2f}), best offset={best_off}")
    max_b1_range = max(b1_ranges.values())
    print(f"\n  physical/physical_reverse baseline gap: 171.07 mHa")
    print(f"  max err range from offset alone (any of the 4 orderings): {max_b1_range:.2f} mHa")
    print(f"  fraction of the 171.07 mHa gap reproduced by offset alone: "
          f"{100*max_b1_range/171.07:.1f}%")
    best_offsets = {name: int(b1_df[(b1_df.ordering == name) & (b1_df.status == "OK")]
                              .loc[lambda d: d.err_mHa.idxmin(), "anchor_offset"])
                    for name in B1_ORDERINGS}
    print(f"  best offset per ordering: {best_offsets}")
    print(f"  same offset best for all four? {len(set(best_offsets.values())) == 1}")

    # =========================================================== B2
    banner("B2 -- free anchor selection at fixed identity ordering (no sampling: all 120)")
    pos_id = R.positions_from(np.arange(norb))
    has_capture_proxy = False
    print(f"  Searched run_ordering_pipeline.py for a no-sampling capture proxy for "
          f"mechanism B: none exists (only unified_run.py's captured_of, mechanism A, "
          f"has one). Ranking by retained_J_oppspin only, as specified.")
    all120 = []
    for triple in itertools.combinations(range(norb), 3):
        rj_ss, rj_os = R.retained_J_split_of(pos_id, Jaa, Jab, anchor_orbitals=triple)
        all120.append(dict(triple=triple, retained_J_oppspin=rj_os))
    all120_df = pd.DataFrame(all120).sort_values("retained_J_oppspin", ascending=False).reset_index(drop=True)
    all120_df.to_csv(OUTDIR / "b2_all120_ranking.csv", index=False)
    print(f"  120/120 triples ranked by retained_J_oppspin "
          f"(range [{all120_df.retained_J_oppspin.min():.4f}, {all120_df.retained_J_oppspin.max():.4f}])")

    top10 = list(all120_df.head(10)["triple"])
    bottom10 = list(all120_df.tail(10)["triple"])
    chosen = set(top10) | set(bottom10)
    rng_b2 = np.random.default_rng(B2_RNG_SEED)
    remaining = [t for t in itertools.combinations(range(norb), 3) if t not in chosen]
    rand10_idx = rng_b2.choice(len(remaining), size=10, replace=False)
    rand10 = [remaining[i] for i in rand10_idx]

    to_sample = ([("top", t) for t in top10] + [("bottom", t) for t in bottom10]
                + [("random", t) for t in rand10])
    print(f"  sampling {len(to_sample)} triples: 10 top + 10 bottom + 10 random "
          f"(rng seed {B2_RNG_SEED})")

    b2_rows = []
    for i, (bucket, triple) in enumerate(to_sample, 1):
        tag = f"b2_{bucket}_{'-'.join(map(str, triple))}"
        row = evaluate(R, t1L, t2L, pos_id, norb, nelec, nocc, hf, fcidump_path,
                      E_CASCI, b2i, W, seed=2026, anchor_orbitals=triple, tag=tag)
        rj = R.retained_J_of(pos_id, Jaa, Jab, anchor_orbitals=triple)
        rj_ss, rj_os = R.retained_J_split_of(pos_id, Jaa, Jab, anchor_orbitals=triple)
        row.update(bucket=bucket, triple="".join(map(str, triple)), seed=2026,
                  retained_J=rj, retained_J_samespin=rj_ss, retained_J_oppspin=rj_os)
        b2_rows.append(row)
        pd.DataFrame(b2_rows).to_csv(OUTDIR / "b2_sampled.csv", index=False)
        print(f"[B2 {i}/{len(to_sample)}] {bucket:<7} {triple}  "
              f"err={row['err_mHa']:.2f} mHa  status={row['status']}")

    b2_df = pd.DataFrame(b2_rows)
    b2_ok = b2_df[b2_df.status == "OK"]
    banner("B2 -- key numbers")
    print(f"  err_mHa range over 30 sampled: {b2_ok.err_mHa.min():.2f} - "
          f"{b2_ok.err_mHa.max():.2f} mHa (range {b2_ok.err_mHa.max()-b2_ok.err_mHa.min():.2f})")
    rho_cap = spearmanr(b2_ok.captured, b2_ok.err_mHa)
    print(f"  rho(captured, err_mHa) over 30 = {rho_cap.statistic:+.3f}  p={rho_cap.pvalue:.2e}")
    rho_rank = spearmanr(b2_ok.retained_J_oppspin, b2_ok.err_mHa)
    print(f"  rho(retained_J_oppspin, err_mHa) over 30 = {rho_rank.statistic:+.3f}  "
          f"p={rho_rank.pvalue:.2e}")
    best_row = b2_ok.loc[b2_ok.err_mHa.idxmin()]
    print(f"  best anchor triple found: {best_row['triple']}  err_mHa={best_row['err_mHa']:.2f}  "
          f"(bucket={best_row['bucket']})")
    print(f"  vs identity baseline (300.32 mHa): "
          f"{'BETTER by ' + f'{300.32 - best_row.err_mHa:.2f}' if best_row.err_mHa < 300.32 else 'WORSE by ' + f'{best_row.err_mHa - 300.32:.2f}'} mHa")

    b2_range = b2_ok.err_mHa.max() - b2_ok.err_mHa.min()

    # =========================================================== B3
    banner("B3 -- HEADLINE")
    print(f"err_mHa range, same-spin ordering alone (baseline): {BASELINE_SPREAD}")
    print(f"err_mHa range, anchor offset alone at fixed ordering (B1, max over 4): {max_b1_range:.2f}")
    print(f"err_mHa range, free anchor selection at fixed ordering (B2, 30 sampled): {b2_range:.2f}")
    larger = "free anchor selection (B2)" if b2_range > max_b1_range else "anchor offset (B1)"
    print(f"larger lever: {larger}")
    print(f"cheap-quantity prediction within the anchor lever: "
          f"rho(retained_J_oppspin, err_mHa)={rho_rank.statistic:+.3f} (p={rho_rank.pvalue:.2e}), "
          f"rho(captured, err_mHa)={rho_cap.statistic:+.3f} (p={rho_cap.pvalue:.2e})")

    # =========================================================== B4
    banner("B4")
    print("Part A found no PREDICTIVE variant (all 11 scores NULL or WEAK). Skipping B4.")

    # ------------------------------------------------------------- metadata
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="B", git_commit=git_commit, shots=SHOTS,
        b1_seeds=[2026, 7], b2_seed=2026, b2_rng_seed=B2_RNG_SEED,
        reference_npz_sha256=sha256_of(CACHEDIR / "reference.npz"),
        fcidump_sha256=sha256_of(CACHEDIR / "fcidump.txt"),
        baseline_csv_sha256=sha256_of(BASELINE_CSV),
        b0_reversal_invariant=f"{n_match}/{n_total}",
        b1_orderings=B1_ORDERINGS, b1_ranges=b1_ranges, b1_best_offsets=best_offsets,
        b2_n_triples_total=120, b2_n_sampled=len(to_sample),
        b2_range=float(b2_range), b2_rho_captured=float(rho_cap.statistic),
        b2_rho_captured_p=float(rho_cap.pvalue),
        b2_rho_retained_J_oppspin=float(rho_rank.statistic),
        b2_rho_retained_J_oppspin_p=float(rho_rank.pvalue),
        b2_best_triple=str(best_row["triple"]), b2_best_err_mHa=float(best_row["err_mHa"]),
        larger_lever=larger, b4_skipped=True,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'b1_offset_sweep.csv'}")
    print(f"[out] {OUTDIR / 'b2_all120_ranking.csv'}")
    print(f"[out] {OUTDIR / 'b2_sampled.csv'}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
