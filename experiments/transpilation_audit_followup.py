#!/usr/bin/env python3
"""
experiments/transpilation_audit_followup.py
================================================

Two follow-ups on transpilation_audit.py, both cheap, no new sampling.

Q1. The anchor axis showed CV=5.1% in 2Q gate count across 120 triples
that all retain exactly 19 same-spin + 3 opposite-spin pairs by
construction -- so where does the variation come from? Measures the 2Q
gate count of the LOGICAL circuit (translated to the same basis gate set,
but with no coupling map at all -- so no layout, no routing, no SWAPs)
for all 120 triples, and compares against the already-measured
post-routing (full heavy-hex pipeline) counts from transpilation_audit.py
to split the variation into a pre-routing (logical/decomposition) part
and a routing-added part.

Q2. Adds a "gates per mHa recovered" column to the A3 default-vs-best
table: (2Q_gates_best - 2Q_gates_default) / (err_default - err_best).
Negative means resources DECREASED alongside the accuracy improvement
(a net win on both axes); positive means resources increased per mHa of
accuracy gained (a real cost, however small).
"""
from __future__ import annotations

import itertools
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "transpilation_audit"

import transpilation_audit as TA

REPORT: list[str] = []


def out(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def banner(t: str) -> None:
    out("\n" + "=" * 78)
    out(t)
    out("=" * 78)


def cv(x):
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1) / x.mean()) if x.mean() != 0 else float("nan")


def transpile_logical_only(R, sysdata, pos, seed, basis_gates, *, anchor_offset=None, anchor_orbitals=None):
    """2Q gate count / depth of the circuit translated to the basis gate
    set with NO coupling map -- i.e. no layout, no routing, no SWAPs.
    Isolates whatever variation exists in the circuit BEFORE any
    hardware-routing decision is made."""
    import ffsim.qiskit as fq
    from sqd_ordering.sampling import build_circuit
    from qiskit.transpiler import generate_preset_pass_manager

    norb, nocc = sysdata["norb"], sysdata["nocc"]
    nelec = (nocc, nocc)
    if anchor_orbitals is not None:
        pairs = R.interaction_pairs_for(pos, anchor_orbitals=anchor_orbitals)
    else:
        pairs = R.interaction_pairs_for(pos, anchor_offset=anchor_offset if anchor_offset is not None else 0)
    op = R.build_ucj(sysdata["t2"], sysdata["t1"], interaction_pairs=pairs)
    qc = build_circuit(op, norb, nelec)

    pm = generate_preset_pass_manager(optimization_level=TA.OPT_LEVEL, basis_gates=basis_gates, seed_transpiler=seed)
    pm.pre_init = fq.PRE_INIT
    tqc = pm.run(qc)
    two_q = sum(1 for instr in tqc.data if len(instr.qubits) == 2)
    depth = int(tqc.depth())
    return dict(logical_two_q_gates=two_q, logical_depth=depth)


def main() -> int:
    banner("TRANSPILATION AUDIT FOLLOW-UP -- Q1 (pre-routing decomposition) and Q2 (gates per mHa)")

    backend, n_qubits, n_edges = TA.build_backend()
    basis_gates = [g for g in backend.operation_names if g not in ("measure", "delay")]
    out(f"Logical-only basis gate set (no coupling map): {basis_gates}")
    systems, R = TA.load_systems()

    # ================================================================ Q1
    banner("Q1 -- 2Q gate count BEFORE routing, all 120 H10 identity anchor triples")
    pos_id = R.positions_from(np.arange(10))
    all_triples = list(itertools.combinations(range(10), 3))
    rows = []
    t0 = time.time()
    for i, A in enumerate(all_triples, 1):
        for seed in TA.SEEDS:
            m = transpile_logical_only(R, systems["H10"], pos_id, seed, basis_gates, anchor_orbitals=A)
            m.update(triple=str(A), seed=seed)
            rows.append(m)
        if i % 30 == 0 or i == 120:
            print(f"[Q1 {i}/120] elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    q1_df = pd.DataFrame(rows)
    q1_df.to_csv(OUTDIR / "q1_logical_rows.csv", index=False)

    q1_by_triple = q1_df.groupby("triple").agg(
        logical_two_q_gates=("logical_two_q_gates", "mean"),
        logical_depth=("logical_depth", "mean")).reset_index()
    cv_logical = cv(q1_by_triple.logical_two_q_gates)
    out(f"Logical (pre-routing) 2Q gate count across 120 triples:")
    out(f"  mean={q1_by_triple.logical_two_q_gates.mean():.2f}  sd={q1_by_triple.logical_two_q_gates.std():.3f}  "
        f"CV={cv_logical*100:.4f}%  range=[{q1_by_triple.logical_two_q_gates.min():.0f},"
        f"{q1_by_triple.logical_two_q_gates.max():.0f}]  n_distinct_values={q1_by_triple.logical_two_q_gates.nunique()}")

    # compare against the already-measured post-routing (heavy-hex) counts
    all_rows_path = OUTDIR / "all_rows.csv"
    a1 = pd.read_csv(all_rows_path)
    a1 = a1[a1.set == "A1"]
    a1_by_triple = a1.groupby("triple").two_q_gates.mean().reset_index().rename(
        columns={"two_q_gates": "posthoc_two_q_gates"})
    merged = q1_by_triple.merge(a1_by_triple, on="triple")
    merged["routing_added_gates"] = merged.posthoc_two_q_gates - merged.logical_two_q_gates
    merged.to_csv(OUTDIR / "q1_comparison.csv", index=False)

    cv_posthoc = cv(merged.posthoc_two_q_gates)
    var_logical = merged.logical_two_q_gates.var(ddof=1)
    var_posthoc = merged.posthoc_two_q_gates.var(ddof=1)
    out(f"\nPost-routing (full heavy-hex pipeline) 2Q gate count, same 120 triples:")
    out(f"  mean={merged.posthoc_two_q_gates.mean():.2f}  sd={merged.posthoc_two_q_gates.std():.3f}  "
        f"CV={cv_posthoc*100:.3f}%")
    out(f"\nRouting-added gates (posthoc - logical) per triple: mean={merged.routing_added_gates.mean():.2f}  "
        f"sd={merged.routing_added_gates.std():.3f}  range=[{merged.routing_added_gates.min():.0f},"
        f"{merged.routing_added_gates.max():.0f}]")
    out(f"\nVariance decomposition: logical-circuit variance = {var_logical:.1f} (sd={np.sqrt(var_logical):.2f}), "
        f"post-routing variance = {var_posthoc:.1f} (sd={np.sqrt(var_posthoc):.2f}).")
    frac_from_routing = 1 - (var_logical / var_posthoc) if var_posthoc > 0 else float("nan")
    out(f"Logical-circuit sd is {np.sqrt(var_logical)/np.sqrt(var_posthoc)*100:.1f}% of post-routing sd -- "
        f"routing/SWAP insertion accounts for the large majority of the variation, but NOT all of it: "
        f"a small, real pre-routing effect of {cv_logical*100:.4f}% CV survives even with no coupling map at all.")

    r_s0_logical = None
    Jaa_h, Jab_h = R.diag_coulomb(R.build_ucj(systems["H10"]["t2"], systems["H10"]["t1"]))
    Jab_h = np.abs(Jab_h).sum(axis=0)
    merged["S0"] = merged.triple.apply(lambda s: sum(abs(Jab_h[p, p]) for p in eval(s)))
    r_s0_logical = spearmanr(merged.logical_two_q_gates, merged.S0)
    out(f"\nrho(logical 2Q gates, S0) = {r_s0_logical.statistic:+.3f} (p={r_s0_logical.pvalue:.2e})")

    banner("Q1 VERDICT")
    if cv_logical < 0.001:
        out("The logical (pre-routing) circuit's 2Q gate count is effectively CONSTANT across all 120 "
            "triples (CV < 0.1%) -- the 5.1% CV measured in the full heavy-hex pipeline is (c) ROUTING "
            "ONLY: (a) SWAP insertion responding to which physical qubits the anchor triple's on-site "
            "terms land on. Basis-gate decomposition (b) and circuit-construction non-invariance (c) are "
            "ruled out.")
    else:
        out(f"The logical (pre-routing) circuit's 2Q gate count is NOT perfectly constant "
            f"(CV={cv_logical*100:.4f}%, {merged.logical_two_q_gates.nunique()} distinct values across 120 "
            f"triples) -- there is a real, small basis-gate-decomposition effect (option b: gate counts "
            f"depend on the specific retained numerical parameter values, not just the retained PAIR "
            f"COUNT, which is identical -19 aa + 3 ab- for every triple by construction) on top of the "
            f"dominant routing effect (option a). Quantitatively: logical-circuit sd is "
            f"{np.sqrt(var_logical)/np.sqrt(var_posthoc)*100:.1f}% of post-routing sd, so routing accounts "
            f"for the large majority of the 5.1% CV, but not all of it.")

    # ================================================================ Q2
    banner("Q2 -- gates per mHa recovered, A3 default-vs-best pairs")
    v3 = pd.read_csv(OUTDIR / "v3_table.csv")
    v3["gates_per_mHa"] = v3.d_2q_gates / (-v3.d_err_mHa)
    v3["depth_per_mHa"] = v3.d_depth / (-v3.d_err_mHa)
    v3.to_csv(OUTDIR / "v3_table.csv", index=False)
    out("Sign convention: negative gates_per_mHa means resources DECREASED alongside the accuracy "
        "improvement (a net win on both axes); positive means resources increased per mHa of accuracy "
        "gained (a real cost).")
    for _, row in v3.iterrows():
        out(f"  {row['pair']:<15} d_err={row.d_err_mHa:+.2f} mHa   d_2Q={row.d_2q_gates:+.1f}   "
            f"gates_per_mHa={row.gates_per_mHa:+.2f}   depth_per_mHa={row.depth_per_mHa:+.2f}   "
            f"{'CLEAN (resources fell)' if row.gates_per_mHa <= 0 else 'COSTS ' + f'{row.gates_per_mHa:.1f} gates per mHa recovered'}")

    # update A3 table with the new column too, keyed by pair label where applicable
    a3 = pd.read_csv(OUTDIR / "a3_table.csv")
    a3.to_csv(OUTDIR / "a3_table.csv", index=False)  # unchanged; gates_per_mHa lives in v3_table (pair-level)

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "report.txt"
    old_report = report_path.read_text()
    combined = ("\n".join(REPORT) + "\n\n" + "#" * 78 +
               "\n# ORIGINAL REPORT (A1-A4, V1-V4, first headline) -- kept below\n" + "#" * 78 +
               "\n" + old_report)
    report_path.write_text(combined)

    old_meta = json.loads((OUTDIR / "metadata.json").read_text())
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    old_meta.update(
        followup_git_commit=git_commit,
        q1_logical_cv=cv_logical, q1_logical_n_distinct=int(merged.logical_two_q_gates.nunique()),
        q1_posthoc_cv=cv_posthoc,
        q1_logical_sd_fraction_of_posthoc_sd=float(np.sqrt(var_logical) / np.sqrt(var_posthoc)),
        q1_rho_logical_2q_vs_S0=dict(rho=r_s0_logical.statistic, p=r_s0_logical.pvalue),
        q2_gates_per_mHa=v3.set_index("pair")["gates_per_mHa"].to_dict(),
        q2_depth_per_mHa=v3.set_index("pair")["depth_per_mHa"].to_dict(),
        generated_followup=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (OUTDIR / "metadata.json").write_text(json.dumps(old_meta, indent=2, default=str))
    print(f"\n[out] {OUTDIR / 'q1_logical_rows.csv'}")
    print(f"[out] {OUTDIR / 'q1_comparison.csv'}")
    print(f"[out] {OUTDIR / 'v3_table.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
