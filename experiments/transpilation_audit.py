#!/usr/bin/env python3
"""
experiments/transpilation_audit.py
=====================================

TRANSPILATION AUDIT -- closes the "fixed quantum resources" claim.

Every comparison in this project holds fixed the LUCJ mask size, shot
budget and SQD determinant budget, but has never verified that different
orbital layouts produce circuits of EQUAL cost once mapped onto real
heavy-hex hardware connectivity. This audit transpiles the exact circuits
the sampling pipeline runs (via sqd_ordering.sampling.build_circuit,
imported directly -- no separate circuit constructor) against a heavy-hex
CouplingMap and reports two-qubit gate count, depth, two-qubit depth, and
SWAP count, across the anchor axis, the same-spin axis, and the specific
configurations quoted in the report. No sampling, no sbd, no new
reference data -- transpilation only.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "transpilation_audit"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
OPT_LEVEL = 1  # matches CFG["seed_transpiler"]-adjacent convention: optimization_level=1 everywhere in this project
HEAVY_HEX_DISTANCE = 5
BACKEND_SEED = 42

REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==========================================================================
# SETUP -- heavy-hex backend, held fixed for the whole audit
# ==========================================================================
def build_backend():
    from qiskit.transpiler import CouplingMap
    from qiskit.providers.fake_provider import GenericBackendV2

    cm = CouplingMap.from_heavy_hex(distance=HEAVY_HEX_DISTANCE)
    n_qubits = cm.size()
    n_edges = len(cm.get_edges())
    backend = GenericBackendV2(num_qubits=n_qubits, coupling_map=cm, seed=BACKEND_SEED)
    return backend, n_qubits, n_edges


# ==========================================================================
# system references -- CACHED ONLY, nothing recomputed
# ==========================================================================
def load_systems():
    import run_ordering_pipeline as R
    import unified_run as U

    R.CFG["sbd_bin"] = str(REPO_ROOT / "sbd" / "apps" /
                           "chemistry_tpb_selected_basis_diagonalization" / "diag")

    h10_ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=str(REPO_ROOT / "cache" / "h10_R1.6"))
    h10 = dict(t1=np.asarray(h10_ref["t1L"]), t2=np.asarray(h10_ref["t2L"]),
              norb=h10_ref["norb"], nocc=h10_ref["nocc"])

    n2 = dict(t1=np.asarray(U.ref_data["t1"]), t2=np.asarray(U.ref_data["t2"]),
             norb=U.NORB, nocc=U.NELEC[0])

    cr2_data = np.load(REPO_ROOT / "cache" / "tm_transfer" / "reference.npz")
    cr2 = dict(t1=cr2_data["t1L"], t2=cr2_data["t2L"],
              norb=int(cr2_data["norb"]), nocc=int(cr2_data["nocc"]))

    return dict(H10=h10, N2=n2, Cr2=cr2), R


# ==========================================================================
# core measurement -- one (system, pos, anchor_kwargs, seed) -> metrics dict
# ==========================================================================
def transpile_one(R, backend, sysdata, pos, seed, *, anchor_offset=None, anchor_orbitals=None):
    import ffsim.qiskit as fq
    from sqd_ordering.sampling import build_circuit
    from qiskit.transpiler import generate_preset_pass_manager, PassManager

    norb, nocc = sysdata["norb"], sysdata["nocc"]
    nelec = (nocc, nocc)
    if anchor_orbitals is not None:
        pairs = R.interaction_pairs_for(pos, anchor_orbitals=anchor_orbitals)
    else:
        pairs = R.interaction_pairs_for(pos, anchor_offset=anchor_offset if anchor_offset is not None else 0)
    pairs_aa, pairs_ab = pairs
    op = R.build_ucj(sysdata["t2"], sysdata["t1"], interaction_pairs=pairs)
    n_reps = int(np.asarray(op.diag_coulomb_mats).shape[0])

    qc = build_circuit(op, norb, nelec)

    t0 = time.time()
    pm = generate_preset_pass_manager(optimization_level=OPT_LEVEL, backend=backend, seed_transpiler=seed)
    pm.pre_init = fq.PRE_INIT  # matches CFG["use_pre_init"]=True, the pipeline default, for every layout

    part1 = PassManager()
    for stage_name in ("pre_init", "init", "layout", "routing"):
        stage = getattr(pm, stage_name)
        if stage is not None:
            part1 += stage
    routed = part1.run(qc)
    swap_count = int(routed.count_ops().get("swap", 0))

    part2 = PassManager()
    for stage_name in ("translation", "optimization", "scheduling"):
        stage = getattr(pm, stage_name)
        if stage is not None:
            part2 += stage
    final = part2.run(routed)
    elapsed = time.time() - t0

    two_q = sum(1 for instr in final.data if len(instr.qubits) == 2)
    single_q = sum(1 for instr in final.data if len(instr.qubits) == 1)
    depth = int(final.depth())
    two_q_depth = int(final.depth(filter_function=lambda instr: len(instr.qubits) == 2))

    return dict(
        two_q_gates=two_q, depth=depth, two_q_depth=two_q_depth, swap_count=swap_count,
        single_q_gates=single_q, n_reps=n_reps, n_pairs_aa=len(pairs_aa), n_pairs_ab=len(pairs_ab),
        n_free_params=n_reps * (len(pairs_aa) + len(pairs_ab)),
        transpile_wall_s=elapsed,
    )


def evaluate(R, backend, sysdata, pos, tag, **anchor_kwargs):
    rows = []
    for seed in SEEDS:
        m = transpile_one(R, backend, sysdata, pos, seed, **anchor_kwargs)
        m.update(tag=tag, seed=seed)
        rows.append(m)
    return rows


def cv(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / x.mean()) if x.mean() != 0 else float("nan")


def main() -> int:
    banner("TRANSPILATION AUDIT")

    backend, n_qubits, n_edges = build_backend()
    out(f"Heavy-hex CouplingMap(distance={HEAVY_HEX_DISTANCE}): {n_qubits} qubits, {n_edges} edges")
    out(f"Backend: GenericBackendV2(num_qubits={n_qubits}, coupling_map=<above>, seed={BACKEND_SEED})")
    out(f"Transpiler policy: generate_preset_pass_manager(optimization_level={OPT_LEVEL}, backend=backend, "
        f"seed_transpiler=<one of {SEEDS}>), pre_init=ffsim.qiskit.PRE_INIT (matches CFG['use_pre_init']=True, "
        f"the pipeline default). initial_layout NOT pinned -- the pipeline does not pin it either "
        f"(sample_bitstrings/circuit_stats never pass initial_layout), so this audit matches that policy.")
    out("Circuit construction: sqd_ordering.sampling.build_circuit(op, norb, nelec) -- imported directly, "
        "the exact function sample_bitstrings() calls (refactored out of it for this purpose; see commit "
        "'refactor: extract build_circuit()').")

    systems, R = load_systems()
    for name, sd in systems.items():
        out(f"  {name}: norb={sd['norb']}  nocc={sd['nocc']}")

    base = pd.read_csv(Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv",
                       dtype={"permutation": str})
    perm_by_ordering = base.groupby("ordering")["permutation"].first()

    def h10_pos(name):
        if name == "identity":
            return R.positions_from(np.arange(10))
        return R.positions_from(R.parse_permutation(perm_by_ordering[name], 10))

    all_rows = []

    # ---------------------------------------------------------- A1: anchor axis
    banner("A1 -- anchor axis: all 120 triples at H10 identity")
    pos_id_h10 = h10_pos("identity")
    t0 = time.time()
    all_triples = list(itertools.combinations(range(10), 3))
    for i, A in enumerate(all_triples, 1):
        rows = evaluate(R, backend, systems["H10"], pos_id_h10, f"A1_{'-'.join(map(str,A))}", anchor_orbitals=A)
        for r in rows:
            r.update(set="A1", system="H10", chain="identity", triple=str(A))
        all_rows.extend(rows)
        if i % 20 == 0 or i == 120:
            pd.DataFrame(all_rows).to_csv(OUTDIR / "all_rows.csv", index=False)
            print(f"[A1 {i}/120] elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    # ---------------------------------------------------------- A2: same-spin axis
    banner("A2 -- same-spin axis: 20 H10 layouts at default anchors")
    rnd = base[base.ordering.str.match(r"^rand")].groupby("ordering").agg(
        err=("err_mHa", "mean"), perm=("permutation", "first")).sort_values("err")
    idx16 = [int(round(x)) for x in np.linspace(0, len(rnd) - 1, 16)]
    rand16 = list(rnd.index[idx16])
    out(f"16 random orderings spanning the baseline range: {rand16}")
    a2_names = ["identity", "reverse", "physical", "physical_reverse"] + rand16
    for name in a2_names:
        pos = h10_pos(name)
        rows = evaluate(R, backend, systems["H10"], pos, f"A2_{name}", anchor_offset=0)
        for r in rows:
            r.update(set="A2", system="H10", chain=name, triple="default")
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(OUTDIR / "all_rows.csv", index=False)
    out(f"A2 done: {len(a2_names)} layouts x {len(SEEDS)} seeds")

    # ---------------------------------------------------------- A3: key comparisons
    banner("A3 -- key comparisons quoted in the report")
    pos_phys_h10 = h10_pos("physical")
    pos_n2_id = R.positions_from(np.arange(systems["N2"]["norb"]))
    pos_cr2_id = R.positions_from(np.arange(systems["Cr2"]["norb"]))

    a3_configs = [
        ("H10_identity_default", systems["H10"], pos_id_h10, dict(anchor_offset=0), "300.32 mHa"),
        ("H10_identity_best_012", systems["H10"], pos_id_h10, dict(anchor_orbitals=(0, 1, 2)), "224.60 mHa"),
        ("H10_identity_S0_top_019", systems["H10"], pos_id_h10, dict(anchor_orbitals=(0, 1, 9)), "n/a (S0 top pick)"),
        ("H10_physical_default", systems["H10"], pos_phys_h10, dict(anchor_offset=0), "389.71 mHa"),
        ("H10_physical_best_247", systems["H10"], pos_phys_h10, dict(anchor_orbitals=(2, 4, 7)), "172.15 mHa"),
        ("H10_identity_no_ab", systems["H10"], pos_id_h10, dict(anchor_orbitals=()), "458.70 mHa"),
        ("N2_identity_default", systems["N2"], pos_n2_id, dict(anchor_offset=0), "31.87 mHa"),
        ("N2_identity_best_019", systems["N2"], pos_n2_id, dict(anchor_orbitals=(0, 1, 9)), "24.27 mHa"),
        ("Cr2_identity_default", systems["Cr2"], pos_cr2_id, dict(anchor_offset=0), "240.79 mHa"),
        ("Cr2_identity_best_01_11", systems["Cr2"], pos_cr2_id, dict(anchor_orbitals=(0, 1, 11)), "201.95 mHa"),
    ]
    a3_rows = []
    for tag, sysdata, pos, kwargs, sqd_err_label in a3_configs:
        rows = evaluate(R, backend, sysdata, pos, tag, **kwargs)
        for r in rows:
            r.update(set="A3", sqd_err_label=sqd_err_label)
        all_rows.extend(rows)
        a3_rows.extend([dict(r, config=tag, sqd_err_label=sqd_err_label) for r in rows])
        out(f"  {tag}: done")
    pd.DataFrame(all_rows).to_csv(OUTDIR / "all_rows.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTDIR / "all_rows.csv", index=False)

    # ---------------------------------------------------------------- A3 table
    banner("A3 TABLE -- configuration, SQD error, 2Q gates, depth, SWAPs, parameters (mean +/- sd over 5 seeds)")
    a3_summary = []
    for tag, sysdata, pos, kwargs, sqd_err_label in a3_configs:
        sub = df[(df.set == "A3") & (df.tag == tag)]
        row = dict(config=tag, sqd_err=sqd_err_label,
                  two_q_gates_mean=sub.two_q_gates.mean(), two_q_gates_sd=sub.two_q_gates.std(),
                  depth_mean=sub.depth.mean(), depth_sd=sub.depth.std(),
                  two_q_depth_mean=sub.two_q_depth.mean(), swap_mean=sub.swap_count.mean(),
                  swap_sd=sub.swap_count.std(), n_free_params=int(sub.n_free_params.iloc[0]),
                  n_pairs_aa=int(sub.n_pairs_aa.iloc[0]), n_pairs_ab=int(sub.n_pairs_ab.iloc[0]))
        a3_summary.append(row)
        out(f"  {tag:<28} err={sqd_err_label:<16} 2Q={row['two_q_gates_mean']:.1f}+/-{row['two_q_gates_sd']:.1f}  "
            f"depth={row['depth_mean']:.1f}+/-{row['depth_sd']:.1f}  swaps={row['swap_mean']:.1f}+/-{row['swap_sd']:.1f}  "
            f"params={row['n_free_params']}  pairs_aa={row['n_pairs_aa']}  pairs_ab={row['n_pairs_ab']}")
    a3_df = pd.DataFrame(a3_summary)
    a3_df.to_csv(OUTDIR / "a3_table.csv", index=False)

    # ---------------------------------------------------------------- A4: sanity check
    banner("A4 -- sanity check: no-alpha-beta control must have strictly fewer 2Q gates than any retaining config")
    noab_2q = df[(df.tag == "H10_identity_no_ab")].two_q_gates.mean()
    a1_min_2q = df[df.set == "A1"].groupby("tag").two_q_gates.mean().min()
    out(f"  no-alpha-beta control mean 2Q gates: {noab_2q:.1f}")
    out(f"  minimum mean 2Q gates among all 120 A1 anchor triples (which all retain ab pairs): {a1_min_2q:.1f}")
    a4_pass = noab_2q < a1_min_2q
    out(f"  SANITY CHECK: {'PASS' if a4_pass else 'FAIL'} -- "
        f"{'no-ab has strictly fewer 2Q gates than every retaining configuration' if a4_pass else 'no-ab does NOT have fewer 2Q gates -- STOPPING, something is wrong'}")
    if not a4_pass:
        out("A4 FAILED. Per protocol, stopping analysis here rather than proceeding on a broken measurement.")
        report_path = OUTDIR / "report.txt"
        report_path.write_text("\n".join(REPORT) + "\n")
        return 1

    # ---------------------------------------------------------------- V1: anchor axis resource-neutrality
    banner("V1 -- is anchor selection resource-neutral? (A1, 120 triples)")
    a1 = df[df.set == "A1"]
    a1_by_triple = a1.groupby("triple").agg(two_q_gates=("two_q_gates", "mean"), depth=("depth", "mean"),
                                            swap_count=("swap_count", "mean")).reset_index()
    cv_2q, cv_depth, cv_swap = cv(a1_by_triple.two_q_gates), cv(a1_by_triple.depth), cv(a1_by_triple.swap_count)
    out(f"  2Q gate count: mean={a1_by_triple.two_q_gates.mean():.1f}  sd={a1_by_triple.two_q_gates.std():.2f}  "
        f"CV={cv_2q*100:.3f}%  range=[{a1_by_triple.two_q_gates.min():.0f},{a1_by_triple.two_q_gates.max():.0f}]")
    out(f"  depth:         mean={a1_by_triple.depth.mean():.1f}  sd={a1_by_triple.depth.std():.2f}  "
        f"CV={cv_depth*100:.3f}%  range=[{a1_by_triple.depth.min():.0f},{a1_by_triple.depth.max():.0f}]")
    out(f"  swap count:    mean={a1_by_triple.swap_count.mean():.1f}  sd={a1_by_triple.swap_count.std():.2f}  "
        f"CV={cv_swap*100:.3f}%  range=[{a1_by_triple.swap_count.min():.0f},{a1_by_triple.swap_count.max():.0f}]")
    v1_neutral = all(c < 0.01 for c in (cv_2q, cv_depth, cv_swap))
    out(f"  V1 VERDICT: {'resource-neutral (all CV < 1%)' if v1_neutral else 'NOT resource-neutral -- CV exceeds 1% for at least one metric'}")

    # correlation with S0 and err_sqd
    from sqd_ordering import mask as maskmod
    Jaa_h, Jab_h = R.diag_coulomb(R.build_ucj(systems["H10"]["t2"], systems["H10"]["t1"]))
    Jab_h = np.abs(Jab_h).sum(axis=0)
    a1_by_triple["S0"] = a1_by_triple.triple.apply(lambda s: sum(abs(Jab_h[p, p]) for p in ast.literal_eval(s)))
    trans_csv = Path(__file__).resolve().parent / "outputs" / "transmission" / "all_evaluations.csv"
    err_lookup = {}
    if trans_csv.exists():
        t = pd.read_csv(trans_csv)
        t = t[(t.system == "H10") & (t.chain == "identity") & (t.status == "OK")]
        err_lookup = dict(zip(t.triple, t.err_sqd))
    a1_by_triple["err_sqd"] = a1_by_triple.triple.apply(lambda s: err_lookup.get(s, np.nan))
    have_err = a1_by_triple.dropna(subset=["err_sqd"])
    for metric in ("two_q_gates", "depth", "swap_count"):
        r_s0 = spearmanr(a1_by_triple[metric], a1_by_triple.S0)
        out(f"  rho({metric}, S0) = {r_s0.statistic:+.3f} (p={r_s0.pvalue:.2e})")
        if len(have_err) > 3:
            r_err = spearmanr(have_err[metric], have_err.err_sqd)
            out(f"  rho({metric}, err_sqd) = {r_err.statistic:+.3f} (p={r_err.pvalue:.2e})  [n={len(have_err)}]")
    a1_by_triple.to_csv(OUTDIR / "a1_by_triple.csv", index=False)

    # ---------------------------------------------------------------- V2: same-spin axis
    banner("V2 -- is same-spin ordering resource-neutral? (A2, 20 layouts)")
    a2 = df[df.set == "A2"]
    a2_by_chain = a2.groupby("chain").agg(two_q_gates=("two_q_gates", "mean"), depth=("depth", "mean"),
                                          swap_count=("swap_count", "mean")).reset_index()
    cv_2q2, cv_depth2, cv_swap2 = cv(a2_by_chain.two_q_gates), cv(a2_by_chain.depth), cv(a2_by_chain.swap_count)
    out(f"  2Q gate count: mean={a2_by_chain.two_q_gates.mean():.1f}  sd={a2_by_chain.two_q_gates.std():.2f}  "
        f"CV={cv_2q2*100:.3f}%  range=[{a2_by_chain.two_q_gates.min():.0f},{a2_by_chain.two_q_gates.max():.0f}]")
    out(f"  depth:         mean={a2_by_chain.depth.mean():.1f}  sd={a2_by_chain.depth.std():.2f}  "
        f"CV={cv_depth2*100:.3f}%  range=[{a2_by_chain.depth.min():.0f},{a2_by_chain.depth.max():.0f}]")
    out(f"  swap count:    mean={a2_by_chain.swap_count.mean():.1f}  sd={a2_by_chain.swap_count.std():.2f}  "
        f"CV={cv_swap2*100:.3f}%  range=[{a2_by_chain.swap_count.min():.0f},{a2_by_chain.swap_count.max():.0f}]")
    v2_neutral = all(c < 0.01 for c in (cv_2q2, cv_depth2, cv_swap2))
    out(f"  V2 VERDICT: {'resource-neutral (all CV < 1%)' if v2_neutral else 'NOT resource-neutral -- CV exceeds 1% for at least one metric'}")
    a2_by_chain.to_csv(OUTDIR / "a2_by_chain.csv", index=False)

    # ---------------------------------------------------------------- V3: quoted-improvement resource check
    banner("V3 -- do quoted improvements (default vs best anchors) come with a resource increase?")
    pairs_to_check = [
        ("H10 identity", "H10_identity_default", "H10_identity_best_012", 300.32, 224.60),
        ("H10 physical", "H10_physical_default", "H10_physical_best_247", 389.71, 172.15),
        ("N2 identity", "N2_identity_default", "N2_identity_best_019", 31.87, 24.27),
        ("Cr2 identity", "Cr2_identity_default", "Cr2_identity_best_01_11", 240.79, 201.95),
    ]
    v3_rows = []
    for label, tag_def, tag_best, err_def, err_best in pairs_to_check:
        row_def = a3_df[a3_df.config == tag_def].iloc[0]
        row_best = a3_df[a3_df.config == tag_best].iloc[0]
        d_err = err_best - err_def
        d_2q = row_best.two_q_gates_mean - row_def.two_q_gates_mean
        d_depth = row_best.depth_mean - row_def.depth_mean
        clean = d_err < 0 and d_2q <= 0 and d_depth <= 0
        v3_rows.append(dict(pair=label, d_err_mHa=d_err, d_2q_gates=d_2q, d_depth=d_depth, clean=clean))
        out(f"  {label:<15} err {err_def:.2f}->{err_best:.2f} ({d_err:+.2f} mHa)   "
            f"2Q {row_def.two_q_gates_mean:.1f}->{row_best.two_q_gates_mean:.1f} ({d_2q:+.1f})   "
            f"depth {row_def.depth_mean:.1f}->{row_best.depth_mean:.1f} ({d_depth:+.1f})   "
            f"{'CLEAN' if clean else 'RESOURCE CHANGE ACCOMPANIES THE IMPROVEMENT'}")
    v3_df = pd.DataFrame(v3_rows)
    v3_df.to_csv(OUTDIR / "v3_table.csv", index=False)
    v3_any_confound = not all(v3_df.clean)

    # ---------------------------------------------------------------- V4: seed vs layout variation
    banner("V4 -- SABRE seed variation vs layout variation")
    seed_var = a1.groupby("seed").two_q_gates.mean()
    within_triple_seed_sd = a1.groupby("triple").two_q_gates.std().mean()
    layout_sd = a1_by_triple.two_q_gates.std()
    out(f"  mean within-triple seed-to-seed sd (2Q gates): {within_triple_seed_sd:.3f}")
    out(f"  layout-to-layout sd (2Q gates, across 120 triples): {layout_sd:.3f}")
    ratio = layout_sd / within_triple_seed_sd if within_triple_seed_sd > 0 else float("inf")
    out(f"  ratio layout_sd / seed_sd = {ratio:.2f}  "
        f"-> layout variation is {'ABOVE' if ratio > 1 else 'BELOW'} SABRE seed noise")

    # ---------------------------------------------------------------- HEADLINE
    banner("HEADLINE")
    out(f"1. Anchor axis resource-neutral: {'YES' if v1_neutral else 'NO'} "
        f"(CV: 2Q={cv_2q*100:.3f}%, depth={cv_depth*100:.3f}%, swap={cv_swap*100:.3f}%)")
    out(f"2. Same-spin axis resource-neutral: {'YES' if v2_neutral else 'NO'} "
        f"(CV: 2Q={cv_2q2*100:.3f}%, depth={cv_depth2*100:.3f}%, swap={cv_swap2*100:.3f}%)")
    out(f"3. Any quoted improvement comes with extra circuit cost: {'YES' if v3_any_confound else 'NO'}")
    out(f"4. Layout variation vs SABRE seed noise: "
        f"{'ABOVE' if ratio > 1 else 'BELOW'} (ratio={ratio:.2f})")

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    import qiskit
    metadata = dict(
        part="transpilation_audit", git_commit=git_commit, qiskit_version=qiskit.__version__,
        heavy_hex_distance=HEAVY_HEX_DISTANCE, n_qubits=n_qubits, n_edges=n_edges,
        backend_seed=BACKEND_SEED, optimization_level=OPT_LEVEL, seeds=SEEDS,
        v1_resource_neutral=v1_neutral, v1_cv=dict(two_q=cv_2q, depth=cv_depth, swap=cv_swap),
        v2_resource_neutral=v2_neutral, v2_cv=dict(two_q=cv_2q2, depth=cv_depth2, swap=cv_swap2),
        v3_any_confound=v3_any_confound, v4_ratio_layout_to_seed=ratio,
        a4_sanity_pass=a4_pass,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (OUTDIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\n[out] {OUTDIR / 'all_rows.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
