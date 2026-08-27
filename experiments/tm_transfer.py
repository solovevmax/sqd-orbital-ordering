#!/usr/bin/env python3
"""
experiments/tm_transfer.py
=============================

FINAL TRANSFER EXPERIMENT -- does the anchor-selection phenomenon, its
mechanism (subspace capture), and its chain-dependent failure mode survive
a move from H10/N2 to a compact localised transition-metal active space?

SYSTEM (attempted first, per the declared protocol): Cr2 CAS(12,12) at
R=1.68 A, cc-pVDZ, occupied/virtual block-Boys localised. Active space
selected via AVAS targeting the Cr 3d+4s manifold (threshold=0.3, robust
across 0.25-0.5 -- not a knife-edge selection); frozen-core CCSD restricted
to that 12-orbital window.

FALLBACK (declared in advance, only taken if Cr2 CCSD fails to converge to
a sensible E_corr within 300 cycles): H12 linear chain, STO-6G, CAS(12,12),
R=1.6 A, block-Boys localised -- same 24 qubits, same 220 triples.
build_h10() already generalises to any natoms, so this reuses it directly
with natoms=12.

Reuses run_ordering_pipeline.py's mask/score/sampling/sbd machinery
throughout (block_boys, rotate_amplitudes, Amplitudes, score1/score2,
interaction_pairs_for, sample_bitstrings, top_dets, run_sbd) -- nothing
reimplemented except the Cr2/H12-specific reference construction itself,
which mirrors build_or_load_h10_reference's two validation gates exactly.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

OUTDIR = Path(__file__).resolve().parent / "outputs" / "tm_transfer"
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHEDIR = REPO_ROOT / "cache" / "tm_transfer"

# ---- declared system ----
CR2_R, CR2_BASIS = 1.68, "cc-pvdz"
AVAS_LABELS = ["Cr 3d", "Cr 4s"]
AVAS_THRESHOLD = 0.3
NCAS, NELECAS = 12, 12          # CAS(12,12) either way
FALLBACK_R, FALLBACK_BASIS, FALLBACK_NATOMS = 1.6, "sto-6g", 12

SEED = 2026
N_WORKERS = 8
RNG_CHAIN_SEED = 20260828001
RNG_TRIPLE_SEED = 20260828002
N_SHARED = 60
H10_RANGE_PER_HEADROOM = 1445.1
N2_RANGE_PER_HEADROOM = 2550.1
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


def parse_triple(s):
    if isinstance(s, tuple):
        return tuple(int(x) for x in s)
    s = str(s).strip()
    if s.startswith("("):
        return tuple(int(x) for x in ast.literal_eval(s))
    return tuple(int(c) for c in s)


# ==========================================================================
# STAGE 0 -- reference build (Cr2, with declared H12 fallback)
# ==========================================================================
def _try_cr2_ccsd():
    """Attempt Cr2 CAS(12,12): RHF -> AVAS active-space selection ->
    frozen-core CCSD restricted to the 12-orbital window. Returns a dict
    with the trace and mol/mf/mo_coeff/ncore on success, or a dict with
    ok=False and the reason on failure. Never raises -- the caller decides
    whether to fall back."""
    import pyscf.gto, pyscf.scf, pyscf.cc
    from pyscf.mcscf import avas

    trace = {}
    mol = pyscf.gto.M(atom=[["Cr", (0, 0, 0)], ["Cr", (0, 0, CR2_R)]], basis=CR2_BASIS,
                      symmetry=False, verbose=0, spin=0, charge=0)
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    trace["rhf_converged"] = bool(mf.converged)
    trace["E_RHF"] = float(mf.e_tot)
    trace["nao"] = int(mol.nao_nr())
    out(f"[cr2] RHF converged={mf.converged}  E={mf.e_tot:.10f}  nao={mol.nao_nr()}")
    if not mf.converged:
        return dict(ok=False, reason="RHF did not converge", trace=trace)

    ncas, nelecas, mo_coeff = avas.avas(mf, AVAS_LABELS, canonicalize=False,
                                        threshold=AVAS_THRESHOLD, verbose=0)
    trace["avas_ncas"] = int(ncas)
    trace["avas_nelecas"] = int(nelecas)
    out(f"[cr2] AVAS({AVAS_LABELS}, threshold={AVAS_THRESHOLD}): ncas={ncas} nelecas={nelecas}")
    if ncas != NCAS or nelecas != NELECAS:
        return dict(ok=False, reason=f"AVAS gave CAS({nelecas},{ncas}), not the declared "
                                     f"CAS({NELECAS},{NCAS})", trace=trace)

    ncore = (mol.nelectron - nelecas) // 2
    active = list(range(ncore, ncore + ncas))
    frozen = [i for i in range(mol.nao_nr()) if i not in active]
    mf.mo_coeff = mo_coeff
    t0 = time.time()
    cc = pyscf.cc.CCSD(mf, frozen=frozen)
    cc.conv_tol = 1e-8
    cc.max_cycle = 300
    cc.diis_space = 12
    cc.kernel()
    trace["ccsd_converged"] = bool(cc.converged)
    trace["ccsd_cycles"] = int(cc.cycles)
    trace["e_corr"] = float(cc.e_corr)
    trace["ccsd_elapsed_s"] = time.time() - t0
    out(f"[cr2] CCSD (CAS({nelecas},{ncas}) window, frozen={len(frozen)}): "
        f"converged={cc.converged}  e_corr={cc.e_corr:.10f}  cycles={cc.cycles}  "
        f"time={trace['ccsd_elapsed_s']:.2f}s")
    sensible = cc.converged and abs(cc.e_corr) < 5.0 and not math.isnan(cc.e_corr)
    trace["sensible"] = bool(sensible)
    if not sensible:
        return dict(ok=False, reason=f"CCSD did not converge to a sensible E_corr "
                                     f"(converged={cc.converged}, e_corr={cc.e_corr})",
                    trace=trace)
    return dict(ok=True, trace=trace, mol=mol, mf=mf, cc=cc, ncas=ncas, nelecas=nelecas,
               ncore=ncore, active=active, mo_coeff=mo_coeff)


def build_or_load_tm_reference(cachedir=CACHEDIR):
    """Cr2 CAS(12,12), block-Boys localised, with the declared H12 fallback.
    Mirrors run_ordering_pipeline.py's build_or_load_h10_reference: same two
    gates (amplitude/integral E_corr match < 1e-9; localised/canonical CASCI
    match < 1e-10), fails loudly (sys.exit) on either, nothing cached if so.
    """
    import run_ordering_pipeline as R
    import ffsim
    from pyscf import mcscf
    from pyscf.tools import fcidump as fcidump_mod

    cachedir = Path(cachedir)
    ref_path = cachedir / "reference.npz"
    fci_path = cachedir / "fcidump.txt"
    meta_path = cachedir / "metadata.json"

    if ref_path.exists() and fci_path.exists() and meta_path.exists():
        banner("STAGE 0 -- reference (loading from cache)")
        out(f"[tm-ref] loading cached reference from {cachedir}")
        data = np.load(ref_path)
        metadata = json.loads(meta_path.read_text())
        return dict(t1L=data["t1L"], t2L=data["t2L"], norb=int(data["norb"]),
                    nocc=int(data["nocc"]), E_CASCI=float(data["E_CASCI"]), ci=data["ci"],
                    fcidump_path=str(fci_path), metadata=metadata,
                    system_used=metadata["system_used"], fallback_fired=metadata["fallback_fired"])

    cachedir.mkdir(parents=True, exist_ok=True)
    banner("STAGE 0 -- reference build and validation")

    t0 = time.time()
    attempt = _try_cr2_ccsd()
    cr2_trace = attempt["trace"]
    if attempt["ok"]:
        out(f"[tm-ref] Cr2 CCSD converged with a sensible E_corr -- proceeding with the "
            f"REAL Cr2 system. Fallback NOT triggered.")
        system_used = "Cr2_CAS(12,12)_cc-pVDZ"
        fallback_fired = False
        mol, mf, cc = attempt["mol"], attempt["mf"], attempt["cc"]
        ncas, nelecas, ncore, active = attempt["ncas"], attempt["nelecas"], attempt["ncore"], attempt["active"]
        mo_coeff = attempt["mo_coeff"]
        norb, nocc = ncas, nelecas // 2
        t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
    else:
        out(f"[tm-ref] Cr2 attempt FAILED: {attempt['reason']}")
        out(f"[tm-ref] FALLBACK per declared protocol: H12 linear chain, STO-6G, "
            f"CAS(12,12), R={FALLBACK_R} A")
        system_used = f"H12_CAS(12,12)_STO-6G_fallback"
        fallback_fired = True
        mol, mf, cc, norb, nocc = R.build_h10(FALLBACK_R, FALLBACK_NATOMS, FALLBACK_BASIS)
        ncas, nelecas, ncore = norb, 2 * nocc, 0
        active = list(range(norb))
        mo_coeff = np.asarray(mf.mo_coeff)
        t1, t2 = np.asarray(cc.t1), np.asarray(cc.t2)
        if not cc.converged:
            sys.exit("FATAL: fallback H12 CCSD also did not converge. Stopping per protocol "
                     "(no further fallback declared).")

    out(f"[tm-ref] system_used={system_used}  norb={norb}  nocc={nocc}  nelec=({nocc},{nocc})")

    # --- block-Boys localization + exact amplitude rotation, active window only
    C_loc, Uo, Uv = R.block_boys(mol, mf, active, nocc, tag="tm-")
    t1L, t2L = R.rotate_amplitudes(t1, t2, Uo, Uv)

    import copy as _copy
    md_can = ffsim.MolecularData.from_scf(mf, active_space=active)
    mf_loc = _copy.copy(mf)
    # block_boys returns a full-width mo_coeff array with only the active
    # occ/vir columns replaced by their localized versions.
    mf_loc.mo_coeff = C_loc
    md_loc = ffsim.MolecularData.from_scf(mf_loc, active_space=active)

    def ecorr(t1_, t2_, md):
        eri = np.asarray(md.hamiltonian.two_body_tensor)
        o, v = slice(0, nocc), slice(nocc, norb)
        e = eri[o, v, o, v]
        tau = (2 * t2_ - t2_.transpose(0, 1, 3, 2)
               + 2 * np.einsum("ia,jb->ijab", t1_, t1_, optimize=True)
               - np.einsum("ib,ja->ijab", t1_, t1_, optimize=True))
        return float(np.einsum("ijab,iajb->", tau, e, optimize=True))

    e_can = ecorr(t1, t2, md_can)
    e_loc = ecorr(t1L, t2L, md_loc)
    gate1_diff = abs(e_can - e_loc)
    out(f"[tm-ref] GATE 1 -- E_corr from amplitudes: canonical {e_can:.12f}  "
        f"localized {e_loc:.12f}  diff {gate1_diff:.2e}  (threshold 1e-9)")
    if gate1_diff > 1e-9:
        sys.exit("FATAL: amplitude rotation inconsistent with rotated integrals "
                 "(GATE 1 failed). Refusing to cache. Stopping per 'do not proceed "
                 "past any failure'.")
    out("[tm-ref] GATE 1 PASSED.")

    def casci(C):
        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.verbose = 0
        mc.fcisolver.conv_tol = 1e-10
        mc.fcisolver.max_cycle = 300
        e, _, ci_vec, *_ = mc.kernel(C)
        return float(e), np.asarray(ci_vec)

    t_casci = time.time()
    e_casci_can, _ = casci(np.asarray(mf.mo_coeff))
    e_casci_loc, ci_loc = casci(C_loc)
    out(f"[tm-ref] GATE 2 -- CASCI canonical {e_casci_can:.12f}  localized {e_casci_loc:.12f}  "
        f"diff {abs(e_casci_can-e_casci_loc):.2e}  (threshold 1e-10)  "
        f"[{time.time()-t_casci:.1f}s for both diagonalisations]")
    if abs(e_casci_can - e_casci_loc) > 1e-10:
        sys.exit("FATAL: localized vs canonical CASCI energies differ (GATE 2 failed) "
                 "-- not the same active space. Refusing to cache.")
    out("[tm-ref] GATE 2 PASSED.")
    E_CASCI = e_casci_loc

    h1 = np.asarray(md_loc.hamiltonian.one_body_tensor)
    h2 = np.asarray(md_loc.hamiltonian.two_body_tensor)
    fcidump_mod.from_integrals(str(fci_path), h1, h2, norb, nelecas,
                               nuc=float(md_loc.core_energy), ms=0)
    out(f"[tm-ref] FCIDUMP: {fci_path}  sha256={sha256_of(fci_path)[:16]}")

    dim_full = math.comb(norb, nocc) ** 2
    out(f"[tm-ref] CASCI dimension C({norb},{nocc})^2 = {dim_full}")

    np.savez(ref_path, t1L=t1L, t2L=t2L, norb=norb, nocc=nocc, E_CASCI=E_CASCI, ci=ci_loc)
    metadata = dict(
        system_used=system_used, fallback_fired=fallback_fired,
        cr2_attempt_trace=cr2_trace,
        norb=norb, nocc=nocc, nelec=[nocc, nocc], E_CASCI=E_CASCI,
        dim_full=dim_full, gate1_diff=gate1_diff, gate2_diff=abs(e_casci_can - e_casci_loc),
        fcidump_sha256=sha256_of(fci_path),
        elapsed_min=round((time.time() - t0) / 60, 2),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    out(f"[tm-ref] cached to {cachedir}  ({metadata['elapsed_min']} min)")

    return dict(t1L=t1L, t2L=t2L, norb=norb, nocc=nocc, E_CASCI=E_CASCI, ci=ci_loc,
                fcidump_path=str(fci_path), metadata=metadata,
                system_used=system_used, fallback_fired=fallback_fired)


# ==========================================================================
# STAGE 0 continued -- budget selection, ideal ceiling, shot-count stability
# ==========================================================================
H10_BUDGET_FRACTION = 225 / 63504


def choose_budget(dim_full):
    b_float = math.sqrt(dim_full * H10_BUDGET_FRACTION)
    b = round(b_float)
    out(f"[budget] target fraction (H10's 225/63504 = {H10_BUDGET_FRACTION:.6f}) applied to "
        f"dim={dim_full}: b_float={b_float:.4f} -> b={b}  "
        f"(actual fraction at b={b}: {b*b/dim_full:.6f})")
    return b


def ideal_ceiling(ci, dim_a, dim_b, budget):
    W = np.asarray(ci).reshape(dim_a, dim_b) ** 2
    W /= W.sum()
    ia = np.argsort(W.sum(axis=1))[::-1][:budget]
    ib = np.argsort(W.sum(axis=0))[::-1][:budget]
    ceiling = float(W[np.ix_(ia, ib)].sum())
    out(f"[budget] ideal capture ceiling at budget={budget}: {ceiling:.6f}")
    return ceiling, W


def shot_stability_check(R, ref, budget):
    """5 seeds at 2e6 shots, identity chain, default (p%4==0) anchor. If
    seed sd of err_sqd exceeds 5 mHa, double shots and repeat. Cached --
    deterministic given (op, seeds, shots), and each seed costs ~16 minutes
    on this system, so a re-run must not repeat it."""
    cache_path = OUTDIR / "stability_check.json"
    if cache_path.exists():
        banner("STAGE 0 -- shot-count stability check (loaded from cache)")
        d = json.loads(cache_path.read_text())
        out(f"  cached result: shots={d['shots']}  sd={d['sd']:.4f} mHa  "
            f"boundary_ratio={d['boundary_ratio']}")
        return d["shots"], d["sd"], d["boundary_ratio"]

    banner("STAGE 0 -- shot-count stability check (identity chain, default anchor)")
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    fcidump_path = ref["fcidump_path"]
    E_CASCI = ref["E_CASCI"]
    hf = R.hf_bitstring(norb, nocc)
    pos = R.positions_from(np.arange(norb))
    pairs = R.interaction_pairs_for(pos, anchor_offset=0)
    op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)

    dim_a = math.comb(norb, nocc)
    W = np.asarray(ref["ci"]).reshape(dim_a, dim_a) ** 2
    W /= W.sum()

    shots = 2_000_000
    boundary_ratio = None
    while True:
        errs = []
        for seed in (2026, 7, 41, 97, 13):
            a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, shots, seed)
            a_sel, _ = R.top_dets(a_c, budget, hf)
            b_sel, _ = R.top_dets(b_c, budget, hf)
            adet_path = OUTDIR / f"_stability_{shots}_{seed}_a.txt"
            bdet_path = OUTDIR / f"_stability_{shots}_{seed}_b.txt"
            adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
            bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
            energy = R.run_sbd(str(fcidump_path), str(adet_path), str(bdet_path), norb)
            err = (energy - E_CASCI) * 1000.0
            errs.append(err)
            if seed == 2026:
                counts = np.array(sorted(a_c.values(), reverse=True), dtype=float)
                if len(counts) > budget:
                    boundary_ratio = float(counts[budget] / counts[budget - 1])
            out(f"  shots={shots}  seed={seed}  err_sqd={err:.4f} mHa")
        errs = np.asarray(errs)
        sd = float(errs.std(ddof=1))
        out(f"  shots={shots}: mean={errs.mean():.4f}  sd={sd:.4f} mHa  "
            f"(threshold 5 mHa)  w(n+1)/w(n) at boundary={boundary_ratio}")
        if sd <= 5.0:
            out(f"  sd within threshold -- using shots={shots} for the sweep.")
            cache_path.write_text(json.dumps(dict(shots=shots, sd=sd,
                                                   boundary_ratio=boundary_ratio), indent=2))
            return shots, sd, boundary_ratio
        shots *= 2
        out(f"  sd exceeds threshold -- doubling shots to {shots} and repeating.")


SBD_BIN = str(REPO_ROOT / "sbd" / "apps" / "chemistry_tpb_selected_basis_diagonalization" / "diag")


def score_S0(A, Jab):
    return sum(abs(Jab[p, p]) for p in A)


# ==========================================================================
# STAGE 1 -- ansatz level, no sampling
# ==========================================================================
def run_stage1(R, ref, chains):
    banner(f"STAGE 1 -- ansatz level, no sampling: {len(chains)} chains x "
           f"{math.comb(ref['norb'], 3)} triples")
    import ffsim
    from pyscf.tools import fcidump as fcidump_mod
    from pyscf import ao2mo

    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    E_CASCI = ref["E_CASCI"]

    fd = fcidump_mod.read(ref["fcidump_path"])
    h1 = fd["H1"]
    h2 = ao2mo.restore(1, fd["H2"], norb)
    ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=h2, constant=fd["ECORE"])
    lo = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
    hf_state = ffsim.hartree_fock_state(norb, nelec)

    all_triples = list(itertools.combinations(range(norb), 3))
    out(f"  {len(all_triples)} triples per chain")

    csv_path = OUTDIR / "stage1_ansatz.csv"
    rows = []
    t0 = time.time()
    n_total = len(chains) * len(all_triples)
    n_done = 0
    for chain_name, pos in chains:
        for A in all_triples:
            pairs = R.interaction_pairs_for(pos, anchor_orbitals=A)
            op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
            ref_copy = hf_state.copy()
            psi = ffsim.apply_unitary(ref_copy, op, norb=norb, nelec=nelec)
            assert np.array_equal(ref_copy, hf_state), "apply_unitary mutated its input"
            norm2 = float(np.vdot(psi, psi).real)
            Hpsi = (lo @ psi.real.astype(np.float64)) + 1j * (lo @ psi.imag.astype(np.float64))
            E_lucj = float(np.vdot(psi, Hpsi).real / norm2)
            err_lucj = (E_lucj - E_CASCI) * 1000.0
            rows.append(dict(chain=chain_name, triple=str(A), err_lucj=err_lucj, full_capture=norm2))
            n_done += 1
            if n_done % 40 == 0 or n_done == n_total:
                pd.DataFrame(rows).to_csv(csv_path, index=False)
            if n_done % 110 == 0 or n_done == n_total:
                el = time.time() - t0
                print(f"[stage1 {n_done}/{n_total}] elapsed={el/60:.1f}m "
                      f"eta={el/n_done*(n_total-n_done)/60:.1f}m", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    out(f"  stage1 done: {len(df)} rows -> {csv_path}")
    return df


# ==========================================================================
# STAGE 2 -- SQD, multiprocessing pool
# ==========================================================================
_W: dict = {}


def _circuit_two_qubit_count(op, norb, nelec, R):
    """Transpile-only probe (no simulation) for the two-qubit gate count,
    matching the circuit_stats() pattern used throughout this project's
    earlier experiments."""
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
    return sum(1 for instr in tqc.data if len(instr.qubits) == 2)


def _init_worker():
    import run_ordering_pipeline as R
    import ffsim

    R.CFG["sbd_bin"] = SBD_BIN
    data = np.load(CACHEDIR / "reference.npz")
    t1L, t2L = data["t1L"], data["t2L"]
    norb, nocc = int(data["norb"]), int(data["nocc"])
    nelec = (nocc, nocc)
    fcidump_path = str(CACHEDIR / "fcidump.txt")
    E_CASCI = float(data["E_CASCI"])
    hf = R.hf_bitstring(norb, nocc)

    dim_a = math.comb(norb, nocc)
    from pyscf.fci import cistring
    strs_list = cistring.make_strings(range(norb), nocc)
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs_list)}
    W = np.asarray(data["ci"]).reshape(dim_a, dim_a) ** 2
    W /= W.sum()

    Jaa_raw, Jab_raw = R.diag_coulomb(R.build_ucj(t2L, t1L))
    Jaa = np.abs(Jaa_raw).sum(axis=0)
    Jab = np.abs(Jab_raw).sum(axis=0)

    from pyscf.tools import fcidump as fcidump_mod
    from pyscf import ao2mo
    fd = fcidump_mod.read(fcidump_path)
    h1 = fd["H1"]
    h2 = ao2mo.restore(1, fd["H2"], norb)
    ham = ffsim.MolecularHamiltonian(one_body_tensor=h1, two_body_tensor=h2, constant=fd["ECORE"])
    lo = ffsim.linear_operator(ham, norb=norb, nelec=nelec)
    hf_state = ffsim.hartree_fock_state(norb, nelec)

    _W["R"] = R
    _W["data"] = dict(t1L=t1L, t2L=t2L, norb=norb, nocc=nocc, nelec=nelec,
                      fcidump_path=fcidump_path, E_CASCI=E_CASCI, hf=hf,
                      b2i=b2i, W=W, Jaa=Jaa, Jab=Jab, lo=lo, hf_state=hf_state)


def _task(args):
    chain, perm_str, pos, triple, role, tag, budget, shots = args
    R = _W["R"]
    d = _W["data"]
    import ffsim

    if role == "default":
        pairs = R.interaction_pairs_for(pos, anchor_offset=0)
        A_for_S0 = tuple(sorted(p for p, _ in R.mask.opp_spin_pairs(
            pos, d["norb"], anchor_mod=4, anchor_offset=0)))
    elif role == "no_ab":
        pairs = R.interaction_pairs_for(pos, anchor_orbitals=())
        A_for_S0 = ()
    else:
        pairs = R.interaction_pairs_for(pos, anchor_orbitals=triple)
        A_for_S0 = triple

    op = R.build_ucj(d["t2L"], d["t1L"], interaction_pairs=pairs)
    two_qubit_count = _circuit_two_qubit_count(op, d["norb"], d["nelec"], R)

    ref_copy = d["hf_state"].copy()
    psi = ffsim.apply_unitary(ref_copy, op, norb=d["norb"], nelec=d["nelec"])
    assert np.array_equal(ref_copy, d["hf_state"]), f"{tag}: apply_unitary mutated its input"
    norm2 = float(np.vdot(psi, psi).real)
    Hpsi = (d["lo"] @ psi.real.astype(np.float64)) + 1j * (d["lo"] @ psi.imag.astype(np.float64))
    E_lucj = float(np.vdot(psi, Hpsi).real / norm2)
    err_lucj = (E_lucj - d["E_CASCI"]) * 1000.0

    a_c, b_c, depth = R.sample_bitstrings(op, d["norb"], d["nelec"], shots, SEED)
    a_sel, n_uniq_a = R.top_dets(a_c, budget, d["hf"])
    b_sel, n_uniq_b = R.top_dets(b_c, budget, d["hf"])
    dim_a_sel, dim_b_sel = len(a_sel), len(b_sel)

    row = dict(chain=chain, permutation=perm_str, triple=str(triple) if role == "triple" else role,
               role=role, tag=tag, err_lucj=err_lucj, full_capture=norm2, depth=depth,
               two_qubit_count=two_qubit_count,
               n_unique_alpha=n_uniq_a, n_unique_beta=n_uniq_b, dim=dim_a_sel * dim_b_sel)
    if dim_a_sel < budget or dim_b_sel < budget:
        row.update(status="SUPPORT_COLLAPSE", err_mHa=float("nan"), captured=float("nan"))
    else:
        adet_path = OUTDIR / f"_{tag}_a.txt"
        bdet_path = OUTDIR / f"_{tag}_b.txt"
        adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
        bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
        energy = R.run_sbd(d["fcidump_path"], str(adet_path), str(bdet_path), d["norb"])
        err_mHa = (energy - d["E_CASCI"]) * 1000.0
        ia = [d["b2i"][s] for s in a_sel]
        ib = [d["b2i"][s] for s in b_sel]
        captured = float(d["W"][np.ix_(ia, ib)].sum())
        row.update(status="OK", err_mHa=err_mHa, captured=captured)

    # retained_J_samespin depends only on pos, never on anchor_orbitals --
    # safe to pass A_for_S0 (possibly empty) directly.
    rj_ss, _ = R.retained_J_split_of(pos, d["Jaa"], d["Jab"], anchor_orbitals=A_for_S0)
    row["retained_J_samespin"] = rj_ss
    row["S0"] = score_S0(A_for_S0, d["Jab"]) if A_for_S0 else 0.0
    return row


def build_stage2_tasks(chains, shared_triples, budget, shots):
    tasks = []
    for chain_name, perm, pos in chains:
        perm_str = "".join(str(int(x)) for x in perm)
        for A in shared_triples:
            tag = f"{chain_name}_{'-'.join(map(str, A))}"
            tasks.append((chain_name, perm_str, pos, A, "triple", tag, budget, shots))
        tasks.append((chain_name, perm_str, pos, None, "default", f"{chain_name}_default", budget, shots))
        tasks.append((chain_name, perm_str, pos, (), "no_ab", f"{chain_name}_noab", budget, shots))
    return tasks


def run_stage2(chains, shared_triples, budget, shots):
    banner(f"STAGE 2 -- SQD: {len(shared_triples)} shared triples x {len(chains)} chains "
           f"+ default + no-ab at each")
    tasks = build_stage2_tasks(chains, shared_triples, budget, shots)
    out(f"  total evaluations: {len(tasks)}")
    csv_path = OUTDIR / "stage2_sqd.csv"

    rows = []
    done_tags = set()
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict(orient="records")
        done_tags = set(prev.tag)
        out(f"  resuming: {len(done_tags)} evaluations already in {csv_path}")
    tasks = [t for t in tasks if t[5] not in done_tags]
    out(f"  remaining to compute: {len(tasks)}")

    t0 = time.time()
    if tasks:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker) as ex:
            futs = {ex.submit(_task, a): a for a in tasks}
            done = 0
            total = len(tasks)
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done += 1
                if done % 2 == 0 or done == total:
                    pd.DataFrame(rows).to_csv(csv_path, index=False)
                el = time.time() - t0
                print(f"[stage2 {done}/{total}] elapsed={el/60:.1f}m "
                      f"eta={el/done*(total-done)/60:.1f}m  last_tag={futs[fut][5]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    out(f"  stage2 done: {len(df)} rows -> {csv_path}")
    return df


def main():
    import run_ordering_pipeline as R
    R.CFG["sbd_bin"] = SBD_BIN

    ref = build_or_load_tm_reference()
    banner("SYSTEM SUMMARY")
    out(f"  system_used = {ref['system_used']}")
    out(f"  fallback_fired = {ref['fallback_fired']}")
    out(f"  norb={ref['norb']}  nocc={ref['nocc']}  E_CASCI={ref['E_CASCI']:.10f}")

    dim_full = ref["metadata"]["dim_full"]
    budget = choose_budget(dim_full)
    dim_a = math.comb(ref["norb"], ref["nocc"])
    ceiling, _ = ideal_ceiling(ref["ci"], dim_a, dim_a, budget)
    shots, sd, boundary_ratio = shot_stability_check(R, ref, budget)

    norb = ref["norb"]
    rng_c = np.random.default_rng(RNG_CHAIN_SEED)
    rand_perm = rng_c.permutation(norb)
    chains_named = [
        ("identity", np.arange(norb)),
        ("reverse", np.arange(norb)[::-1]),
        ("random", rand_perm),
    ]
    chains_pos = [(name, R.positions_from(perm)) for name, perm in chains_named]
    chains_full = [(name, perm, R.positions_from(perm)) for name, perm in chains_named]
    out(f"\n[chains] identity, reverse, random (rng seed {RNG_CHAIN_SEED}) = "
        f"{''.join(str(int(x)) for x in rand_perm)}")

    stage1_df = run_stage1(R, ref, chains_pos)

    all_triples = list(itertools.combinations(range(norb), 3))
    rng_t = np.random.default_rng(RNG_TRIPLE_SEED)
    idx = rng_t.choice(len(all_triples), size=N_SHARED, replace=False)
    shared_triples = [all_triples[i] for i in idx]
    out(f"[triples] {N_SHARED} shared triples drawn (rng seed {RNG_TRIPLE_SEED})")

    stage2_df = run_stage2(chains_full, shared_triples, budget, shots)

    banner("STAGE 0/1/2 COMPLETE -- see stage3 script for analysis")
    meta = dict(
        system_used=ref["system_used"], fallback_fired=ref["fallback_fired"],
        norb=ref["norb"], nocc=ref["nocc"], E_CASCI=ref["E_CASCI"], dim_full=dim_full,
        budget=budget, ceiling=ceiling, shots=shots, stability_sd=sd, boundary_ratio=boundary_ratio,
        rng_chain_seed=RNG_CHAIN_SEED, rng_triple_seed=RNG_TRIPLE_SEED,
        random_chain_perm="".join(str(int(x)) for x in rand_perm),
        n_stage1=len(stage1_df), n_stage2=len(stage2_df),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    (OUTDIR / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    report_path = OUTDIR / "run_report.txt"
    report_path.write_text("\n".join(REPORT) + "\n")
    print(f"\n[out] {report_path}")
    print(f"[out] {OUTDIR / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
