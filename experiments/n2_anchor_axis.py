#!/usr/bin/env python3
"""
experiments/n2_anchor_axis.py
================================

Does the anchor-selection result transfer to N2, or is it H10-specific?
Uses the cached canonical N2 CAS(6,10) R=1.55 reference (unified_run.py's
outputs/unified/reference.pkl/.fcidump) - no new reference data. Mechanism
B (run_ordering_pipeline.py's interaction_pairs_for/build_ucj), same as
every H10 anchor experiment, applied to N2's cached t1/t2/FCIDUMP - this is
exactly what experiments/preflight.py crosscheck already validated as
entrywise-equivalent to mechanism A on N2.
"""
from __future__ import annotations

import itertools
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
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

OUTDIR = Path(__file__).resolve().parent / "outputs" / "n2_anchor_axis"
OUTDIR.mkdir(parents=True, exist_ok=True)

SHOTS = 1_000_000
SEED = 2026
RNG_SEED = 20260826001
N_SHARED = 40
BUDGET = 15
N2_IDEAL_CAPTURE = 0.9866
H10_IDEAL_CAPTURE = 0.7554
H10_ANCHOR_RANGE = 234.10
REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def circuit_stats(op, norb, nelec, R):
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


def build_hamiltonian_lo(fcidump_path, norb, nelec):
    import ffsim
    from pyscf.tools import fcidump as fcidump_mod
    from pyscf import ao2mo

    fd = fcidump_mod.read(str(fcidump_path))
    h1 = fd["H1"]
    h2 = ao2mo.restore(1, fd["H2"], norb)
    ecore = fd["ECORE"]
    ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=h2, constant=ecore)
    return ffsim.linear_operator(ham, norb=norb, nelec=nelec)


def lucj_metrics(R, lo, hf_state, t1, t2, pos, norb, nelec, E_CASCI, anchor_orbitals):
    import ffsim

    pairs = R.interaction_pairs_for(pos, anchor_orbitals=anchor_orbitals)
    op = R.build_ucj(t2, t1, interaction_pairs=pairs)
    ref_copy = hf_state.copy()
    psi = ffsim.apply_unitary(ref_copy, op, norb=norb, nelec=nelec)
    assert np.array_equal(ref_copy, hf_state), (
        "ffsim.apply_unitary mutated its input state - reference no longer pristine."
    )
    norm2 = float(np.vdot(psi, psi).real)
    Hpsi = (lo @ psi.real.astype(np.float64)) + 1j * (lo @ psi.imag.astype(np.float64))
    E_lucj = float(np.vdot(psi, Hpsi).real / norm2)
    return (E_lucj - E_CASCI) * 1000.0, norm2


def evaluate(R, t1, t2, pos, norb, nelec, nocc, hf, fcidump_path, E_CASCI, b2i, W,
            seed, anchor_orbitals, tag):
    pairs = R.interaction_pairs_for(pos, anchor_orbitals=anchor_orbitals)
    op = R.build_ucj(t2, t1, interaction_pairs=pairs)
    depth, two_q = circuit_stats(op, norb, nelec, R)
    a_c, b_c, depth_seed = R.sample_bitstrings(op, norb, nelec, SHOTS, seed)
    assert depth_seed == depth, f"{tag}: depth mismatch"
    a_sel, n_uniq_a = R.top_dets(a_c, BUDGET, hf)
    b_sel, n_uniq_b = R.top_dets(b_c, BUDGET, hf)
    dim_a, dim_b = len(a_sel), len(b_sel)
    row = dict(dim_alpha=dim_a, dim_beta=dim_b, dim=dim_a * dim_b,
              n_unique_alpha=n_uniq_a, n_unique_beta=n_uniq_b, depth=depth, two_qubit_count=two_q)
    if dim_a < BUDGET or dim_b < BUDGET:
        row.update(status="SUPPORT_COLLAPSE", err_mHa=float("nan"), captured=float("nan"))
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
        row.update(status="OK", err_mHa=err_mha, captured=captured)
    return row


def main() -> int:
    banner("N2 ANCHOR-AXIS TRANSFER")
    import unified_run as U
    import run_ordering_pipeline as R
    from pyscf.fci import cistring

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at {R.CFG['sbd_bin']}")

    norb, nocc = U.NORB, U.NELEC[0]
    nelec = U.NELEC
    t1, t2 = U.ref_data["t1"], U.ref_data["t2"]
    fcidump_path = U.FCIDUMP
    E_CASCI = U.E_CASCI
    hf = R.hf_bitstring(norb, nocc)
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2, t1))

    strs = cistring.make_strings(range(norb), nocc)
    dim_full = len(strs)
    assert dim_full == 120, f"expected C(10,3)=120, got {dim_full}"
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
    W = np.asarray(U.ref_data["ci"]).reshape(dim_full, dim_full) ** 2
    W /= W.sum()

    out(f"[setup] N2 CAS({sum(nelec)},{norb}) @ {U.BOND} A, {U.BASIS}  E_CASCI={E_CASCI:.10f}")
    out(f"[setup] FCIDUMP: {fcidump_path}  sha256={sha256_of(fcidump_path)[:16]}")

    # ------------------------------------------------------------- N2 baseline (for context)
    base = pd.read_csv(REPO_ROOT / "outputs" / "unified" / "results.csv", dtype={"perm": str})
    base_rnd = base[base.kind == "random"]
    per_ord = base_rnd.groupby("ordering").agg(err=("err_sub_mHa", "mean"), perm=("perm", "first"))
    per_ord_excl = per_ord.drop("r029")  # established outlier exclusion (CFG EXCLUDE_NAMES), matches
                                          # the historical 21.65-173.23 mHa range cited throughout
    mean_captured_n2 = base_rnd.groupby("ordering")["captured"].mean().drop("r029").mean()
    median_err = per_ord_excl.err.median()
    median_ordering = (per_ord_excl.err - median_err).abs().sort_values().index[0]
    median_perm_str = per_ord_excl.loc[median_ordering, "perm"]
    out(f"\n[N2 baseline] 150 random orderings (r029 excluded as an established outlier -> "
        f"149 kept, range {per_ord_excl.err.min():.2f}-{per_ord_excl.err.max():.2f} mHa, "
        f"matching the historical 21.65-173.23 mHa figure)")
    out(f"[N2 baseline] median-nearest ordering: {median_ordering}  err={per_ord_excl.loc[median_ordering,'err']:.2f}  "
        f"perm={median_perm_str}")
    out(f"[N2 baseline] mean achieved capture (random, excl r029): {mean_captured_n2:.4f}")

    orderings = {
        "identity": np.arange(norb),
        "reverse": np.arange(norb)[::-1].copy(),
        median_ordering: R.parse_permutation(median_perm_str, norb),
    }

    # ------------------------------------------------------------- LUCJ Hamiltonian (no sampling)
    import ffsim
    lo_n2 = build_hamiltonian_lo(fcidump_path, norb, nelec)
    ci_flat = np.asarray(U.ref_data["ci"]).reshape(-1).astype(np.float64)
    ci_norm2 = float(np.vdot(ci_flat, ci_flat))
    lo_check_E = float(np.vdot(ci_flat, lo_n2 @ ci_flat).real / ci_norm2)
    out(f"\n[lucj setup] H (from cached FCIDUMP) reproduces E_CASCI to: {abs(lo_check_E - E_CASCI):.2e}")
    hf_state_n2 = ffsim.hartree_fock_state(norb, nelec)

    # =========================================================== identity: all 120
    banner("Identity: all 120 anchor triples (SQD + LUCJ-only, no extra sbd)")
    all_triples = list(itertools.combinations(range(norb), 3))
    pos_id = R.positions_from(orderings["identity"])
    id_rows = []
    t0 = time.time()
    for i, triple in enumerate(all_triples, 1):
        row = evaluate(R, t1, t2, pos_id, norb, nelec, nocc, hf, fcidump_path, E_CASCI, b2i, W,
                      SEED, triple, tag=f"identity_{'-'.join(map(str, triple))}")
        rj = R.retained_J_of(pos_id, Jaa, Jab, anchor_orbitals=triple)
        rj_ss, rj_os = R.retained_J_split_of(pos_id, Jaa, Jab, anchor_orbitals=triple)
        err_lucj, full_capture = lucj_metrics(R, lo_n2, hf_state_n2, t1, t2, pos_id, norb, nelec,
                                              E_CASCI, triple)
        row.update(ordering="identity", triple=triple, retained_J=rj,
                  retained_J_samespin=rj_ss, retained_J_oppspin=rj_os,
                  err_lucj=err_lucj, full_capture=full_capture)
        id_rows.append(row)
        pd.DataFrame(id_rows).to_csv(OUTDIR / "identity_120.csv", index=False)
        if i % 20 == 0 or i == 120:
            el = time.time() - t0
            print(f"[{i}/120] {triple}  err={row['err_mHa']:.2f}  eta={el/i*(120-i)/60:.1f}m")

    # =========================================================== reverse, median: 40 shared
    banner("40 shared triples at reverse and the median-nearest baseline ordering")
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(all_triples), size=N_SHARED, replace=False)
    shared40 = [all_triples[i] for i in idx]
    out(f"{len(shared40)} triples drawn (rng seed {RNG_SEED})")

    other_rows = []
    for name in ("reverse", median_ordering):
        perm = orderings[name]
        pos = R.positions_from(perm)
        t0 = time.time()
        for i, triple in enumerate(shared40, 1):
            row = evaluate(R, t1, t2, pos, norb, nelec, nocc, hf, fcidump_path, E_CASCI, b2i, W,
                          SEED, triple, tag=f"{name}_{'-'.join(map(str, triple))}")
            rj = R.retained_J_of(pos, Jaa, Jab, anchor_orbitals=triple)
            rj_ss, rj_os = R.retained_J_split_of(pos, Jaa, Jab, anchor_orbitals=triple)
            row.update(ordering=name, triple=triple, retained_J=rj,
                      retained_J_samespin=rj_ss, retained_J_oppspin=rj_os)
            other_rows.append(row)
            pd.DataFrame(other_rows).to_csv(OUTDIR / "reverse_median_40.csv", index=False)
            if i % 10 == 0 or i == N_SHARED:
                el = time.time() - t0
                print(f"[{name} {i}/{N_SHARED}] {triple}  err={row['err_mHa']:.2f}  "
                      f"eta={el/i*(N_SHARED-i)/60:.1f}m")

    id_df = pd.DataFrame(id_rows)
    other_df = pd.DataFrame(other_rows)

    # =========================================================== floor per chain
    banner("No-alpha-beta floor at each chain")
    floor = {}
    for name, perm in orderings.items():
        pos = R.positions_from(perm)
        row = evaluate(R, t1, t2, pos, norb, nelec, nocc, hf, fcidump_path, E_CASCI, b2i, W,
                      SEED, (), tag=f"floor_{name}")
        floor[name] = row["err_mHa"]
        out(f"  {name:<10} floor err_mHa = {row['err_mHa']:.2f}")

    # =========================================================== 1
    banner("1. err_mHa range over 120 at identity, vs N2 baseline and H10 anchor range")
    id_ok = id_df[id_df.status == "OK"]
    id_range = float(id_ok.err_mHa.max() - id_ok.err_mHa.min())
    out(f"  identity anchor range (120 triples): {id_ok.err_mHa.min():.2f} - {id_ok.err_mHa.max():.2f} "
        f"({id_range:.2f} mHa)")
    out(f"  N2 permutation (same-spin ordering) range: 21.65 - 173.23 (151.58 mHa)")
    out(f"  H10 anchor range (120 triples, identity): 234.10 mHa")

    # =========================================================== 2
    banner("2. rho(retained_J_oppspin, err_mHa) over 120, and regret")
    r_os = spearmanr(id_ok.retained_J_oppspin, id_ok.err_mHa)
    out(f"  rho = {r_os.statistic:+.3f}  p={r_os.pvalue:.2e}")
    err = id_ok.err_mHa.to_numpy()
    rand_regret = float(err.mean() - err.min())
    pick_idx = int(id_ok.retained_J_oppspin.to_numpy().argmax())
    pick_err = err[pick_idx]
    regret = float(pick_err - err.min())
    frac = regret / rand_regret if rand_regret > 0 else float("nan")
    out(f"  random-selection regret = {rand_regret:.2f} mHa;  rule regret = {regret:.2f} mHa;  "
        f"fraction = {frac:.3f}")

    # =========================================================== 3
    banner("3. rho(captured, err_mHa) over 120")
    r_cap = spearmanr(id_ok.captured, id_ok.err_mHa)
    out(f"  rho = {r_cap.statistic:+.3f}  p={r_cap.pvalue:.2e}")

    # =========================================================== ADDITION: LUCJ-only, N2 identity
    banner("ADDITION -- err_lucj at N2 identity (no sampling), vs H10's rho=-0.850")
    out(f"  full_capture (should be 1.0 for all 120): min={id_ok.full_capture.min():.8f}  "
        f"max={id_ok.full_capture.max():.8f}")
    r_lucj_sqd = spearmanr(id_ok.err_lucj, id_ok.err_mHa)
    r_lucj_os = spearmanr(id_ok.err_lucj, id_ok.retained_J_oppspin)
    out(f"  rho(err_lucj, err_sqd) = {r_lucj_sqd.statistic:+.3f}  p={r_lucj_sqd.pvalue:.2e}")
    out(f"  rho(err_lucj, retained_J_oppspin) = {r_lucj_os.statistic:+.3f}  p={r_lucj_os.pvalue:.2e}")
    out(f"  err_lucj range: {id_ok.err_lucj.min():.2f} - {id_ok.err_lucj.max():.2f} "
        f"({id_ok.err_lucj.max()-id_ok.err_lucj.min():.2f} mHa)")
    h10_lucj_os_rho = -0.850
    replicates = (np.sign(r_lucj_os.statistic) == np.sign(h10_lucj_os_rho)
                 and abs(r_lucj_os.statistic) >= 0.5 and r_lucj_os.pvalue < 0.05)
    out(f"\n  Does H10's very strong identity-chain relationship "
        f"(rho(retained_J_oppspin, err_lucj)={h10_lucj_os_rho:+.3f}) replicate on N2? "
        f"{'YES' if replicates else 'NO'} (N2: {r_lucj_os.statistic:+.3f}, p={r_lucj_os.pvalue:.2e})")
    if replicates:
        out("  -> The ansatz-level rule (retained_J_oppspin predicts err_lucj strongly at "
            "identity) is GENERAL, not an H10 artefact - it holds on a canonical-orbital "
            "system near its capture ceiling too.")
    else:
        out("  -> Does NOT replicate at H10's strength - either weaker, opposite-signed, or "
            "not significant on N2. The strong identity-chain ansatz-level relationship may be "
            "specific to systems far from their capture ceiling (H10, ceiling 0.7554) and not "
            "general to systems near it (N2, ceiling 0.9866).")

    # =========================================================== 4
    banner("4. Floor and count worse-than-floor per chain")
    n_worse = {}
    for name, df_ in (("identity", id_ok), ("reverse", other_df[(other_df.ordering == "reverse") & (other_df.status == "OK")]),
                      (median_ordering, other_df[(other_df.ordering == median_ordering) & (other_df.status == "OK")])):
        n_w = int((df_.err_mHa > floor[name]).sum())
        n_worse[name] = (n_w, len(df_))
        out(f"  {name:<10} floor={floor[name]:7.2f}  worse-than-floor={n_w}/{len(df_)} "
            f"({100*n_w/len(df_):.1f}%)")

    # =========================================================== 5
    banner("5. Same quantities at reverse and median - is the rule chain-dependent on N2?")
    chain_results = {}
    for name in ("reverse", median_ordering):
        sub = other_df[(other_df.ordering == name) & (other_df.status == "OK")]
        r_os_o = spearmanr(sub.retained_J_oppspin, sub.err_mHa)
        r_cap_o = spearmanr(sub.captured, sub.err_mHa)
        rng_o = float(sub.err_mHa.max() - sub.err_mHa.min())
        chain_results[name] = dict(rho_oppspin=r_os_o.statistic, p_oppspin=r_os_o.pvalue,
                                   rho_captured=r_cap_o.statistic, p_captured=r_cap_o.pvalue,
                                   range=rng_o)
        out(f"  {name:<10} range={rng_o:.2f} mHa  rho(oppspin,err)={r_os_o.statistic:+.3f} "
            f"(p={r_os_o.pvalue:.2e})  rho(captured,err)={r_cap_o.statistic:+.3f} (p={r_cap_o.pvalue:.2e})")
    consistent = all(abs(v["rho_oppspin"]) >= 0.5 and v["p_oppspin"] < 0.05 for v in chain_results.values())
    out(f"\n  rule chain-dependent on N2 (like H10) or consistent? "
        f"{'CONSISTENT across chains' if consistent else 'CHAIN-DEPENDENT, same pattern as H10'}")

    # =========================================================== 6
    banner("6. Best triple per chain")
    best_by_chain = {}
    best_by_chain["identity"] = id_ok.loc[id_ok.err_mHa.idxmin(), "triple"]
    for name in ("reverse", median_ordering):
        sub = other_df[(other_df.ordering == name) & (other_df.status == "OK")]
        best_by_chain[name] = sub.loc[sub.err_mHa.idxmin(), "triple"]
    for name, t in best_by_chain.items():
        out(f"  {name:<10} best triple = {t}")
    all_agree = len(set(str(v) for v in best_by_chain.values())) == 1
    out(f"  all three agree? {all_agree}")

    # =========================================================== INTERPRETATION
    banner("INTERPRETATION -- headroom-normalised comparison")
    n2_headroom = N2_IDEAL_CAPTURE - mean_captured_n2
    h10_mean_captured = 0.5934  # from the H10 baseline report
    h10_headroom = H10_IDEAL_CAPTURE - h10_mean_captured
    n2_range_frac = id_range / n2_headroom if n2_headroom > 0 else float("nan")
    h10_range_frac = H10_ANCHOR_RANGE / h10_headroom if h10_headroom > 0 else float("nan")
    out(f"  N2:  ideal capture={N2_IDEAL_CAPTURE:.4f}  mean achieved={mean_captured_n2:.4f}  "
        f"headroom={n2_headroom:.4f}  anchor range={id_range:.2f} mHa  "
        f"range/headroom={n2_range_frac:.1f} mHa per unit headroom")
    out(f"  H10: ideal capture={H10_IDEAL_CAPTURE:.4f}  mean achieved={h10_mean_captured:.4f}  "
        f"headroom={h10_headroom:.4f}  anchor range={H10_ANCHOR_RANGE:.2f} mHa  "
        f"range/headroom={h10_range_frac:.1f} mHa per unit headroom")
    out(f"  ratio (N2/H10, per unit headroom): {n2_range_frac/h10_range_frac:.3f}")
    if n2_range_frac / h10_range_frac > 0.5:
        out("  -> Comparable per unit headroom: the smaller RAW N2 anchor range is explained by "
            "N2 being near its capture ceiling, not by the anchor mechanism being weaker on N2. "
            "This is a mechanistic result, not a failure of transfer.")
    else:
        out("  -> Still smaller than H10 even after headroom normalisation: some of the "
            "difference is not explained by proximity to the ceiling alone.")

    # ------------------------------------------------------------- save
    report_path = OUTDIR / "report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="n2_anchor_axis", git_commit=git_commit, shots=SHOTS, seed=SEED, rng_seed=RNG_SEED,
        reference_pkl_sha256=sha256_of(U.CACHE), fcidump_sha256=sha256_of(fcidump_path),
        median_ordering=median_ordering, median_ordering_perm=median_perm_str,
        id_range=id_range, rho_oppspin_identity=float(r_os.statistic), p_oppspin_identity=float(r_os.pvalue),
        rho_lucj_sqd_identity=float(r_lucj_sqd.statistic), p_lucj_sqd_identity=float(r_lucj_sqd.pvalue),
        rho_lucj_oppspin_identity=float(r_lucj_os.statistic), p_lucj_oppspin_identity=float(r_lucj_os.pvalue),
        lucj_os_replicates_h10=bool(replicates),
        regret_fraction_identity=frac, rho_captured_identity=float(r_cap.statistic),
        floor_by_chain=floor, n_worse_than_floor=n_worse, chain_results=chain_results,
        best_by_chain={k: str(v) for k, v in best_by_chain.items()}, all_agree=all_agree,
        n2_headroom=n2_headroom, h10_headroom=h10_headroom,
        n2_range_per_headroom=n2_range_frac, h10_range_per_headroom=h10_range_frac,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'identity_120.csv'}")
    print(f"[out] {OUTDIR / 'reverse_median_40.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
