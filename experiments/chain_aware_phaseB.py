#!/usr/bin/env python3
"""
experiments/chain_aware_phaseB.py
====================================

PHASE B -- out-of-sample validation. Phase A found no chain-aware score
(S1-S4) beats the chain-invariant control S0 on worst-case rho(err_sqd), so
per the pre-declared protocol S_best = S0 here (a negative result reported
in its own right, not a bug). Twelve NEW H10 same-spin chains, never used
in any previous experiment, drawn with rng seed 20260827001.

Reuses transmission.py's worker pool (_init_worker, _task) for the actual
sampling -- system="H10" only here -- and chain_aware.py's score functions
for S0 (chain-invariant: since S0 = sum_{p in A} |Jab[p,p]| depends only on
the anchor orbitals, its ranking over the 120 triples -- and hence "top-1
by S0" / "top-3 by S0" -- is the SAME fixed triple(s) at every chain; this
is not a simplification, it is the exact phenomenon under test).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "chain_aware"
OUTDIR.mkdir(parents=True, exist_ok=True)

import transmission as T
import chain_aware as CA

H10_BASELINE_CSV = Path(__file__).resolve().parent / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"
PHASEA_META = OUTDIR / "phaseA_metadata.json"

RNG_CHAINS_SEED = 20260827001
RNG_TRIPLES_SEED = 20260827002
N_NEW_CHAINS = 12
N_SHARED = 40
N_WORKERS = 8
SIG_RHO, SIG_P = 0.3, 0.05

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


def sig(r) -> bool:
    return abs(r.statistic) >= SIG_RHO and r.pvalue < SIG_P


def load_known_h10_perms() -> set[str]:
    df = pd.read_csv(H10_BASELINE_CSV, dtype={"permutation": str})
    perms = df.groupby("ordering")["permutation"].first()
    return set(p.zfill(10) for p in perms)


def draw_new_chains():
    known = load_known_h10_perms()
    rng = np.random.default_rng(RNG_CHAINS_SEED)
    chosen = []
    seen = set()
    draws = 0
    while len(chosen) < N_NEW_CHAINS:
        perm = rng.permutation(10)
        draws += 1
        s = "".join(str(int(x)) for x in perm)
        if s in known or s in seen:
            continue
        seen.add(s)
        chosen.append(perm.copy())
    names = [f"newchain{i:02d}" for i in range(N_NEW_CHAINS)]
    return names, chosen, known, draws


def _task_default(args):
    """Mirror of transmission._task, but for the historical position-based
    default anchor convention (anchor_offset=0, no explicit triple) rather
    than an explicit anchor-orbital triple."""
    system, chain, pos, tag = args
    R = T._W["R"]
    d = T._W[system]
    import ffsim

    pairs = R.interaction_pairs_for(pos, anchor_offset=0)
    op = R.build_ucj(d["t2"], d["t1"], interaction_pairs=pairs)

    ref_copy = d["hf_state"].copy()
    psi = ffsim.apply_unitary(ref_copy, op, norb=d["norb"], nelec=d["nelec"])
    assert np.array_equal(ref_copy, d["hf_state"]), f"{tag}: apply_unitary mutated its input"
    norm2 = float(np.vdot(psi, psi).real)
    Hpsi = (d["lo"] @ psi.real.astype(np.float64)) + 1j * (d["lo"] @ psi.imag.astype(np.float64))
    E_lucj = float(np.vdot(psi, Hpsi).real / norm2)
    err_lucj = (E_lucj - d["E_CASCI"]) * 1000.0

    a_c, b_c, depth = R.sample_bitstrings(op, d["norb"], d["nelec"], d["shots"], T.SEED)
    a_sel, n_uniq_a = R.top_dets(a_c, T.BUDGET, d["hf"])
    b_sel, n_uniq_b = R.top_dets(b_c, T.BUDGET, d["hf"])

    row = dict(system=system, chain=chain, role="default_anchor", tag=tag,
               err_lucj=err_lucj, full_capture=norm2,
               n_unique_alpha=n_uniq_a, n_unique_beta=n_uniq_b, depth=depth)
    if len(a_sel) < T.BUDGET or len(b_sel) < T.BUDGET:
        row.update(status="SUPPORT_COLLAPSE", err_sqd=float("nan"), captured=float("nan"))
    else:
        adet_path = OUTDIR / f"_{system}_{tag}_a.txt"
        bdet_path = OUTDIR / f"_{system}_{tag}_b.txt"
        adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
        bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
        energy = R.run_sbd(str(d["fcidump"]), str(adet_path), str(bdet_path), d["norb"])
        err_sqd = (energy - d["E_CASCI"]) * 1000.0
        ia = [d["b2i"][s2] for s2 in a_sel]
        ib = [d["b2i"][s2] for s2 in b_sel]
        captured = float(d["W"][np.ix_(ia, ib)].sum())
        row.update(status="OK", err_sqd=err_sqd, captured=captured)
    rj_ss, rj_os = R.retained_J_split_of(pos, d["Jaa"], d["Jab"], anchor_offset=0)
    row.update(retained_J_samespin=rj_ss, retained_J_oppspin=rj_os)
    return row


def main() -> int:
    banner("CHAIN-AWARE ANCHOR SCORE -- PHASE B (out-of-sample validation)")

    if not PHASEA_META.exists():
        sys.exit("FATAL: Phase A metadata not found -- run experiments/chain_aware.py first.")
    phaseA = json.loads(PHASEA_META.read_text())
    s_best_name = phaseA["phase_b_score"]
    out(f"Frozen from Phase A: S_best = {s_best_name} (d={phaseA['d_frozen']}, "
        f"lambda={phaseA['lambda_frozen']})")
    if s_best_name == "S0":
        out("Phase A found no chain-aware challenger beat S0 -- this run validates S0 "
            "out-of-sample and reports that as the result, per protocol.")

    # ---------------------------------------------------------- new chains
    banner("Drawing 12 new H10 chains (rng seed 20260827001), verified never used before")
    names, perms, known, n_draws = draw_new_chains()
    out(f"Known previously-used H10 permutations: {len(known)} (from h10_baseline_R1.6)")
    out(f"Draws needed to find {N_NEW_CHAINS} unique, never-before-used permutations: {n_draws}")
    for name, perm in zip(names, perms):
        s = "".join(str(int(x)) for x in perm)
        out(f"  {name}: {s}  (novel: {s not in known})")
    assert all("".join(str(int(x)) for x in p) not in known for p in perms), "collision with known chains"
    assert len({"".join(str(int(x)) for x in p) for p in perms}) == N_NEW_CHAINS, "internal duplicate"
    out("VERIFIED: all 12 are novel and mutually distinct.")

    import run_ordering_pipeline as R
    positions = {name: R.positions_from(perm) for name, perm in zip(names, perms)}
    chains = list(zip(names, perms))

    # ---------------------------------------------------------- S0 ranking (chain-invariant)
    sysdata, _ = CA.build_system_data()
    Jab_h = sysdata["H10"]["Jab"]
    all120 = list(itertools.combinations(range(10), 3))
    s0_by_triple = {A: CA.score_S0(A, Jab_h) for A in all120}
    ranked = sorted(all120, key=lambda A: -s0_by_triple[A])
    top1_triple = ranked[0]
    top3_triples = ranked[:3]
    out(f"\nS0 ranking is chain-invariant (established fact, re-confirmed by construction): "
        f"top-1 = {top1_triple}, top-3 = {top3_triples} -- IDENTICAL at all 12 new chains.")

    # ---------------------------------------------------------- shared 40 triples
    rng_t = np.random.default_rng(RNG_TRIPLES_SEED)
    idx = rng_t.choice(len(all120), size=N_SHARED, replace=False)
    shared40 = [all120[i] for i in idx]
    out(f"{N_SHARED} shared triples drawn (rng seed {RNG_TRIPLES_SEED})")

    # dedupe: union of shared40 + top1 + top3, tag roles
    role_map: dict[tuple, set[str]] = {}
    for t in shared40:
        role_map.setdefault(t, set()).add("shared40")
    role_map.setdefault(top1_triple, set()).add("top1")
    for t in top3_triples:
        role_map.setdefault(t, set()).add("top3")
    unique_triples = list(role_map.keys())
    out(f"Unique triples needing SQD sampling per chain (after dedup): {len(unique_triples)}")

    # ---------------------------------------------------------- B1: ansatz-level, no sampling
    banner("B1 -- ansatz-level sweep, no sampling: all 120 triples x 12 chains = 1440 err_lucj evaluations")
    import ffsim
    d_h10 = None  # filled after a single-process init below for B1 (cheap, serial is fine)
    T._init_worker()
    d_h10 = T._W["H10"]
    Rm = T._W["R"]

    def err_lucj_of(pos, triple):
        pairs = Rm.interaction_pairs_for(pos, anchor_orbitals=triple)
        op = Rm.build_ucj(d_h10["t2"], d_h10["t1"], interaction_pairs=pairs)
        ref_copy = d_h10["hf_state"].copy()
        psi = ffsim.apply_unitary(ref_copy, op, norb=d_h10["norb"], nelec=d_h10["nelec"])
        assert np.array_equal(ref_copy, d_h10["hf_state"])
        norm2 = float(np.vdot(psi, psi).real)
        Hpsi = (d_h10["lo"] @ psi.real.astype(np.float64)) + 1j * (d_h10["lo"] @ psi.imag.astype(np.float64))
        E_lucj = float(np.vdot(psi, Hpsi).real / norm2)
        return (E_lucj - d_h10["E_CASCI"]) * 1000.0

    b1_rows = []
    t0 = time.time()
    for ci, (name, perm) in enumerate(chains, 1):
        pos = positions[name]
        no_ab_lucj = err_lucj_of(pos, ())
        for A in all120:
            el = err_lucj_of(pos, A)
            b1_rows.append(dict(chain=name, triple=str(A), err_lucj=el, S0=s0_by_triple[A]))
        b1_df_chain = pd.DataFrame([r for r in b1_rows if r["chain"] == name])
        n_beat_floor = int((b1_df_chain.err_lucj < no_ab_lucj).sum())
        r_s0 = spearmanr(b1_df_chain.S0, b1_df_chain.err_lucj)
        out(f"[{ci}/12] {name}: err_lucj range {b1_df_chain.err_lucj.min():.2f}-{b1_df_chain.err_lucj.max():.2f}  "
            f"rho(S0,err_lucj)={r_s0.statistic:+.3f} (p={r_s0.pvalue:.2e})  "
            f"no-ab err_lucj={no_ab_lucj:.2f}  beat-floor={n_beat_floor}/120  "
            f"elapsed={(time.time()-t0)/60:.1f}m")
        pd.DataFrame(b1_rows).to_csv(OUTDIR / "phaseB_b1_ansatz_sweep.csv", index=False)

    b1_df = pd.DataFrame(b1_rows)
    rhos_s0 = []
    for name in names:
        sub = b1_df[b1_df.chain == name]
        r = spearmanr(sub.S0, sub.err_lucj)
        rhos_s0.append(r.statistic)
    out(f"\nrho(S0, err_lucj) across 12 chains: min={min(rhos_s0):+.3f}  max={max(rhos_s0):+.3f}  "
        f"mean={np.mean(rhos_s0):+.3f}")
    out(f"(S_best == S0 here per Phase A's negative result, so this IS the S_best distribution -- "
        f"no separate S_best column to compare against.)")

    # ---------------------------------------------------------- B2: SQD sampling
    banner(f"B2 -- SQD sampling: {len(unique_triples)} unique triples + default-anchor + floor, "
           f"x 12 chains, {T.H10_SHOTS} shots, seed {T.SEED}")
    tasks = []
    for name in names:
        pos = positions[name]
        for t in unique_triples:
            tag = f"{name}_{'-'.join(map(str, t))}"
            tasks.append(("H10", name, pos, t, tag))
        tasks.append(("H10", name, pos, (), f"{name}_floor"))
    default_tasks = [("H10", name, positions[name], f"{name}_default") for name in names]

    out(f"Total SQD evaluations: {len(tasks)} (triples incl. floor) + {len(default_tasks)} (default-anchor) "
        f"= {len(tasks) + len(default_tasks)}")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=T._init_worker) as ex:
        futs = {ex.submit(T._task, a): a for a in tasks}
        futs.update({ex.submit(_task_default, a): a for a in default_tasks})
        done = 0
        total = len(futs)
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            done += 1
            if done % 30 == 0 or done == total:
                pd.DataFrame(rows).to_csv(OUTDIR / "phaseB_b2_all.csv", index=False)
            if done % 20 == 0 or done == total:
                el = time.time() - t0
                print(f"[{done}/{total}] elapsed={el/60:.1f}m eta={el/done*(total-done)/60:.1f}m", flush=True)

    b2_df = pd.DataFrame(rows)
    b2_df.to_csv(OUTDIR / "phaseB_b2_all.csv", index=False)
    out(f"\n[timing] {len(rows)} SQD evaluations in {(time.time()-t0)/60:.1f} minutes ({N_WORKERS} workers)")

    # attach roles / S0
    def roles_of(row):
        if row.get("role") == "default_anchor":
            return "default_anchor"
        t = row.get("triple")
        if t is None:
            return "floor" if row["tag"].endswith("_floor") else "unknown"
        return t
    b2_df["triple_parsed"] = b2_df["triple"].apply(lambda s: None if pd.isna(s) else T.parse_triple(s))
    b2_df["S0"] = b2_df["triple_parsed"].apply(lambda t: s0_by_triple.get(t) if t else np.nan)
    b2_df["is_floor"] = b2_df.tag.str.endswith("_floor")
    b2_df["is_default"] = b2_df.role.fillna("") == "default_anchor" if "role" in b2_df.columns else False
    b2_df.to_csv(OUTDIR / "phaseB_b2_all.csv", index=False)

    # save intermediate metadata checkpoint (B1+B2 done, B3 to follow)
    report_path = OUTDIR / "phaseB_report_partial.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    print(f"\n[checkpoint] B1+B2 complete. {report_path}")

    # ---------------------------------------------------------------- B3: analysis
    banner("B3 -- analysis")
    ok = b2_df[b2_df.status == "OK"].copy()

    banner("B3a -- rho(S0, err_sqd) per chain, distribution and worst case")
    rho_rows = []
    for name in names:
        sub = ok[(ok.chain == name) & (ok.tag.str.contains("shared40", na=False) == False) &
                  (~ok.is_floor) & (~ok.is_default)]
        # use the union-triple set (shared40 ∪ top1 ∪ top3) for this chain, all tagged with a real triple
        sub = ok[(ok.chain == name) & (ok.triple_parsed.notna()) & (~ok.is_default) & (~ok.is_floor)]
        if len(sub) < 3:
            continue
        r = spearmanr(sub.S0, sub.err_sqd)
        rho_rows.append(dict(chain=name, n=len(sub), rho_s0_sqd=r.statistic, p=r.pvalue))
        out(f"  {name}: n={len(sub)}  rho(S0,err_sqd)={r.statistic:+.3f} (p={r.pvalue:.2e})")
    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(OUTDIR / "phaseB_b3a_rho.csv", index=False)
    out(f"\n  distribution: min={rho_df.rho_s0_sqd.min():+.3f}  max={rho_df.rho_s0_sqd.max():+.3f}  "
        f"mean={rho_df.rho_s0_sqd.mean():+.3f}")
    out(f"  worst-case rho(S0,err_sqd) across 12 new chains: {rho_df.rho_s0_sqd.max():+.3f} "
        f"at {rho_df.loc[rho_df.rho_s0_sqd.idxmax(), 'chain']}")
    out(f"  (compare Phase A's worst case across the original 6 chains: {phaseA['summary'][0]['worst_rho_sqd']:+.3f})")

    banner("B3b -- normalised regret (argmax S0 vs true best, over the union-triple set) per chain")
    regret_rows = []
    for name in names:
        sub = ok[(ok.chain == name) & (ok.triple_parsed.notna()) & (~ok.is_default) & (~ok.is_floor)]
        if len(sub) < 2:
            continue
        rf = CA.regret_fraction(sub.S0.to_numpy(), sub.err_sqd.to_numpy())
        regret_rows.append(dict(chain=name, regret_frac=rf))
        out(f"  {name}: regret_frac={rf:.3f}")
    regret_df = pd.DataFrame(regret_rows)
    regret_df.to_csv(OUTDIR / "phaseB_b3b_regret.csv", index=False)
    out(f"\n  median regret_frac: {regret_df.regret_frac.median():.3f}")
    out(f"  worst-case regret_frac: {regret_df.regret_frac.max():.3f} at "
        f"{regret_df.loc[regret_df.regret_frac.idxmax(), 'chain']}")

    banner("B3c -- frozen protocol's five configurations: top-1 by S0, top-3 by S0 (best-of-3), "
           "default anchors, random, no-ab control")
    guard_fires = 0
    config_rows = []
    for name in names:
        chain_ok = ok[ok.chain == name]
        floor_row = chain_ok[chain_ok.is_floor]
        default_row = chain_ok[chain_ok.is_default]
        top1_row = chain_ok[chain_ok.triple_parsed == top1_triple]
        top3_rows = chain_ok[chain_ok.triple_parsed.isin(top3_triples)]
        shared_rows = chain_ok[(chain_ok.triple_parsed.notna()) & (~chain_ok.is_default) & (~chain_ok.is_floor)]
        floor_err = float(floor_row.err_sqd.iloc[0]) if len(floor_row) else float("nan")
        top1_err = float(top1_row.err_sqd.iloc[0]) if len(top1_row) else float("nan")
        top3_err = float(top3_rows.err_sqd.min()) if len(top3_rows) else float("nan")
        default_err = float(default_row.err_sqd.iloc[0]) if len(default_row) else float("nan")
        random_triple = shared40[0]
        random_row = chain_ok[chain_ok.triple_parsed == random_triple]
        random_err = float(random_row.err_sqd.iloc[0]) if len(random_row) else float("nan")
        best_err = float(shared_rows.err_sqd.min()) if len(shared_rows) else float("nan")
        rand_regret_denom = float(shared_rows.err_sqd.mean() - best_err) if len(shared_rows) else float("nan")

        guard_fired = (not np.isnan(top1_err)) and (not np.isnan(floor_err)) and (top1_err > floor_err)
        guard_fires += int(guard_fired)
        effective_top1 = floor_err if guard_fired else top1_err

        for cfg, err in (("top1_S0", top1_err), ("top3_S0_best_of", top3_err),
                         ("default_anchor", default_err), ("random", random_err),
                         ("no_ab_floor", floor_err), ("top1_with_guard", effective_top1)):
            rf = (err - best_err) / rand_regret_denom if (rand_regret_denom and rand_regret_denom > 0) else float("nan")
            config_rows.append(dict(chain=name, config=cfg, err_sqd=err, regret_frac=rf))
    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(OUTDIR / "phaseB_b3c_configs.csv", index=False)
    for cfg in config_df.config.unique():
        sub = config_df[config_df.config == cfg]
        out(f"  {cfg:<18} median regret_frac={sub.regret_frac.median():.3f}  "
            f"worst-case regret_frac={sub.regret_frac.max():.3f}")
    out(f"\n  guard (top1_S0 worse than no-ab floor) fires in {guard_fires}/12 chains")

    banner("B3d -- link decomposition per chain: rho(err_lucj,captured), rho(captured,err_sqd)")
    link_rows = []
    for name in names:
        sub = ok[(ok.chain == name) & (ok.triple_parsed.notna()) & (~ok.is_default) & (~ok.is_floor)]
        if len(sub) < 3:
            continue
        r1 = spearmanr(sub.err_lucj, sub.captured)
        r2 = spearmanr(sub.captured, sub.err_sqd)
        link_rows.append(dict(chain=name, rho_link1=r1.statistic, p_link1=r1.pvalue,
                              rho_link2=r2.statistic, p_link2=r2.pvalue,
                              link1_holds=sig(r1), link2_holds=sig(r2)))
        out(f"  {name}: link1={r1.statistic:+.3f} (p={r1.pvalue:.2e}) {'HOLDS' if sig(r1) else 'FAILS'}   "
            f"link2={r2.statistic:+.3f} (p={r2.pvalue:.2e}) {'HOLDS' if sig(r2) else 'FAILS'}")
    link_df = pd.DataFrame(link_rows)
    link_df.to_csv(OUTDIR / "phaseB_b3d_links.csv", index=False)
    n_link2_holds = int(link_df.link2_holds.sum())
    out(f"\n  link2 holds at {n_link2_holds}/{len(link_df)} of the 12 new chains "
        f"(previously: 6/6 of the original chains in the transmission experiment)")

    banner("B3e -- baseline vs best-anchor spread across the 12 new chains (compression factor)")
    spread_rows = []
    for name in names:
        sub = ok[(ok.chain == name) & (ok.triple_parsed.notna()) & (~ok.is_default) & (~ok.is_floor)]
        default_row = ok[(ok.chain == name) & (ok.is_default)]
        if len(sub) < 2 or len(default_row) == 0:
            continue
        best = float(sub.err_sqd.min())
        base = float(default_row.err_sqd.iloc[0])
        spread_rows.append(dict(chain=name, default_err=base, best_anchor_err=best,
                                compression=base / best if best > 0 else float("nan")))
    spread_df = pd.DataFrame(spread_rows)
    spread_df.to_csv(OUTDIR / "phaseB_b3e_spread.csv", index=False)
    out(spread_df.to_string(index=False))
    if len(spread_df):
        out(f"\n  mean compression factor across {len(spread_df)} new chains: {spread_df.compression.mean():.2f}x "
            f"(prior result at n=8: 4.8x)")

    # ---------------------------------------------------------------- HEADLINE
    banner("HEADLINE")
    out(f"1. Worst-case rho(S_best=S0, err_sqd) at ansatz level: {min(rhos_s0):+.3f} (12 new chains); "
        f"at SQD level: {rho_df.rho_s0_sqd.max():+.3f}. Compare Phase A's original worst case: "
        f"{phaseA['summary'][0]['worst_rho_sqd']:+.3f}.")
    top1_guard_sub = config_df[config_df.config == "top1_with_guard"]
    out(f"2. Frozen protocol (top1_S0 + floor guard): median regret_frac={top1_guard_sub.regret_frac.median():.3f}, "
        f"worst-case={top1_guard_sub.regret_frac.max():.3f}.")
    out(f"3. Link 2 (captured->err_sqd) holds at {n_link2_holds}/{len(link_df)} of the 12 new chains.")
    if len(spread_df):
        out(f"4. Compression factor at n=20 (8 original + 12 new): "
            f"mean {spread_df.compression.mean():.2f}x on the new 12 alone.")
    out(f"5. Does a chain-aware score solve the chain-dependence: NO -- Phase A found none of S1-S4 "
        f"beat S0's worst-case rho, and this out-of-sample run validates S0 itself (not a chain-aware "
        f"alternative), confirming the chain-dependence is unresolved by this score family.")

    # ---------------------------------------------------------------- save
    report_path = OUTDIR / "phaseB_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.strip()
    metadata = dict(
        part="chain_aware_phaseB", git_commit=git_commit,
        s_best=s_best_name, new_chains={n: "".join(str(int(x)) for x in p) for n, p in chains},
        n_draws_to_find_novel=n_draws, top1_triple=str(top1_triple), top3_triples=[str(t) for t in top3_triples],
        rho_s0_lucj_range=[float(min(rhos_s0)), float(max(rhos_s0))],
        rho_s0_sqd_worst=float(rho_df.rho_s0_sqd.max()) if len(rho_df) else None,
        regret_median=float(regret_df.regret_frac.median()) if len(regret_df) else None,
        regret_worst=float(regret_df.regret_frac.max()) if len(regret_df) else None,
        guard_fires=guard_fires, link2_holds_count=n_link2_holds, link2_total=len(link_df),
        compression_mean=float(spread_df.compression.mean()) if len(spread_df) else None,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(OUTDIR / "phaseB_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"\n[out] {OUTDIR / 'phaseB_b2_all.csv'}")
    print(f"[out] {report_path}")
    print(f"[out] {OUTDIR / 'phaseB_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
