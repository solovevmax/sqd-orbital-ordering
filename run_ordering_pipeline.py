#!/usr/bin/env python3
"""
run_ordering_pipeline.py
========================

One file, four stages, run in order. Each stage gates the next.

  stage0  environment + API inspection, ffsim.apply_unitary mutation test
  stage1  permutation-convention validation + amplitude-weighted /
          reachability scores on the EXISTING canonical N2 results
          (zero new sampling; minutes)
  stage2  block-Boys covariance check -- does localization change
          diag_coulomb_mats, or is it absorbed into orbital_rotations?
          (minutes; decides whether localized N2 is a real experiment)
  stage3  H10 linear chain, STO-6G, CAS(10e,10o), R = 1.8 A -- the primary
          ordering experiment, with a GROUND-TRUTH physical ordering from
          the nuclear sequence along the chain (~1-2 h)

Design commitments
------------------
* The orbital permutation enters ONLY through the LUCJ interaction_pairs
  mask. Orbital labels, the FCIDUMP, and the determinant labelling are
  identical across every ordering. One FCIDUMP per system, hashed once.
  This removes the cross-script FCIDUMP divergence class of bug entirely.
* Block-Boys only (occ-occ and virt-virt separately). Amplitudes are
  ROTATED, never recomputed -- pyscf CCSD assumes a diagonal Fock which
  localized orbitals do not have.
* Everything is cached to disk once and reloaded. Threads pinned before
  numpy imports BLAS. Fresh AerSimulator per evaluation. Fixed seeds.
* Budget violations are recorded and excluded, never silently run at a
  smaller dimension and never allowed to abort the sweep.

Adapt only the CONFIG block and, if your sbd prints a different final
line, parse_sbd_energy().
"""

from __future__ import annotations

# Thread pinning MUST precede any numpy/pyscf import.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau

HARTREE_TO_MHA = 1000.0

# ==========================================================================
# CONFIG
# ==========================================================================
CFG = dict(
    # --- stage 1 inputs
    canonical_results="outputs/unified/results.csv",
    canonical_reference="cache/canonical/partA_reference.npz",

    # --- canonical N2 system (stages 1-2)
    n2_bond=1.55, n2_basis="6-31g", n2_frozen=4, n2_norb=10, n2_nelec=6,

    # --- H10 system (stage 3)
    h10_natoms=10, h10_R=1.6, h10_basis="sto-6g",

    # --- ansatz / mask
    n_reps=None,              # None lets ffsim choose; LOGGED either way
    anchor_mod=4,             # opposite-spin on-site mask: position % 4 == 0
    mask_mode="centered",  # "anchor" or "centered"
    k_os=4,                # number of opposite-spin on-site terms for centered mask

    # --- sampling / SQD
    shots=500_000,
    n_dets=15,                # per spin sector -> 225-dim product space
    seeds=(2026, 7),
    seed_transpiler=1234,
    use_pre_init=True,        # ffsim.qiskit.PRE_INIT passes

    # --- sweep size
    n_random=100,

    # --- sbd
    sbd_bin=os.environ.get("SBD_BIN", ""),
    mpirun=os.environ.get("MPIRUN", "mpirun"),
    sbd_extra=["--carryover_type", "0", "--shuffle", "0", "--init", "0",
               "--adet_comm_size", "1", "--bdet_comm_size", "1",
               "--task_comm_size", "1"],
    sbd_method_args=["--method", "0", "--iteration", "200",
                     "--tolerance", "1e-10"],
    sbd_timeout=900,
)

# --- pre-registered score parameters (fixed before any correlation seen) ---
K_CHANNELS = 20
L_SPAN_SS = 5
D_ANCHOR_OS = 1
SENSITIVITY_L = (3, 4, 5, 6, 7)
GO_RHO, GO_P, GO_REGRET_FRAC = 0.50, 1e-4, 0.50

IDENTITY_NAMES = {"identity", "canonical", "id"}
EXCLUDE_NAMES = {"r029"}
COLUMN_CANDIDATES = {
    "ordering": ["ordering", "name", "label", "ordering_name"],
    "permutation": ["permutation", "perm", "order", "ordering_string",
                    "orbital_order", "perm_str"],
    "error": [
    "err_sub_mHa",
    "subspace_error_mHa",
    "subspace_error_mha",
    "err_mHa",
    "error_mHa",
    "subspace_error",
    "sqd_error_mHa",
],
    "dim": ["dim", "subspace_dim", "dimension"],
    "dim_alpha": ["dim_alpha", "n_alpha", "dima"],
    "dim_beta": ["dim_beta", "n_beta", "dimb"],
    "retained_J": ["retained_J", "retainedJ", "retained_j"],
    "captured": ["captured", "cap", "captured_weight"],
}


def sha(x) -> str:
    if isinstance(x, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    if isinstance(x, (str, os.PathLike)) and os.path.exists(x):
        with open(x, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


# ==========================================================================
# MASK MODEL  (the permutation enters here and nowhere else)
# ==========================================================================
def positions_from(perm, convention="layout"):
    perm = np.asarray(perm, dtype=int)
    return np.argsort(perm) if convention == "layout" else perm.copy()


def same_spin_pairs(pos):
    inv = np.argsort(pos)                       # inv[k] = orbital at position k
    return [(int(inv[k]), int(inv[k + 1])) for k in range(len(pos) - 1)]


def opp_spin_sites(pos, centroids=None, mode="pos", J_ab=None):
    # positional mask (default, used in stage1 and as fallback)
    if centroids is None or mode == "pos":
        return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]

    # centered mask path (only if centroids provided and mode != "pos")
    # J_ab is ignored here but kept for API compatibility
    assert hasattr(centroids, "__len__"), "centroids must be array-like"
    # TODO: implement your centered-mask logic using centroids here
    # For now, fall back to positional to avoid breaking stage3:
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]



def _centered_opp_spin_sites(pos, centroids, k):
    """Select k orbitals closest to the centroid of all active orbitals."""
    center = float(np.mean(centroids))
    dists = [(abs(float(centroids[int(orb)]) - center), int(orb))
             for orb in range(len(pos))]
    dists.sort()
    return [int(orb) for _, orb in dists[:k]]



def _end_weighted_opp_spin_sites(pos, centroids, k):
    """
    Select k orbitals closest to the chain ends (end-weighted mask).
    Translation-invariant and permutation-sensitive.
    """
    import numpy as np
    z = np.array([float(centroids[int(orb)]) for orb in range(len(pos))])
    zmin, zmax = z.min(), z.max()
    if zmax - zmin < 1e-12:
        # Degenerate case: all centroids identical; fall back to first k orbitals
        return list(range(k))
    x = (z - zmin) / (zmax - zmin)  # normalized to [0,1]
    s = np.minimum(x, 1.0 - x)       # small near ends
    idx = np.argsort(s)
    return [int(i) for i in idx[:k]]



def _largest_J_ab_opp_spin_sites(pos, J_ab, k):
    """
    Select k orbitals with largest |J_ab(p,p)| (largest onsite opposite-spin coupling).
    Translation-invariant and permutation-sensitive.
    """
    import numpy as np
    diag = np.abs(np.diag(J_ab))
    idx = np.argsort(-diag)  # descending
    return [int(i) for i in idx[:k]]


def interaction_pairs_for(pos, centroids=None, J_ab=None):
    """(pairs_aa, pairs_ab) for ffsim, normalised to p <= q and deduped."""
    aa = sorted({tuple(sorted(pq)) for pq in same_spin_pairs(pos)})
    ab = sorted((p, p) for p in opp_spin_sites(pos, centroids=centroids, J_ab=J_ab))
    return list(aa), list(ab)


def _opp_spin_sites_anchor(pos):
    """Legacy anchor-based opposite-spin sites, used for retained_J scalar."""
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]


def retained_J_of(pos, J_aa, J_ab):
    tot = np.abs(J_aa).sum() + np.abs(J_ab).sum()
    if tot <= 0:
        return 0.0
    keep = sum(2.0 * np.abs(J_aa[:, p, q]).sum() for p, q in same_spin_pairs(pos))
    keep += sum(np.abs(J_ab[:, p, p]).sum() for p in _opp_spin_sites_anchor(pos))
    return float(keep / tot)


def parse_permutation(value, norb):
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=int)
    else:
        s = str(value).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.startswith("["):
            arr = np.asarray(json.loads(s), dtype=int)
        elif "," in s or " " in s:
            arr = np.asarray([int(x) for x in s.replace(",", " ").split()], int)
        else:
            if len(s) < norb and s.isdigit() and norb <= 10:
                s = s.zfill(norb)          # pandas ate the leading zeros
            arr = np.asarray([int(c) for c in s], dtype=int)
    if sorted(arr.tolist()) != list(range(norb)):
        raise ValueError(f"not a permutation of 0..{norb-1}: {value!r}")
    return arr


def perm_to_str(perm):
    return "".join(str(int(x)) for x in perm)


# ==========================================================================
# AMPLITUDE-DERIVED SCORES  (non-oracle: RCCSD + mask only)
# ==========================================================================
@dataclass
class Amplitudes:
    t1: np.ndarray
    t2: np.ndarray
    nocc: int
    norb: int
    A_ss: np.ndarray = field(init=False)
    A_os_site: np.ndarray = field(init=False)
    channels_ss: list = field(init=False)
    channels_os: list = field(init=False)

    def __post_init__(self):
        nocc, norb = self.nocc, self.norb
        nvir = norb - nocc
        t2 = np.asarray(self.t2)
        assert t2.shape == (nocc, nocc, nvir, nvir), t2.shape
        A_ss = np.zeros((norb, norb))
        A_os_site = np.zeros(norb)
        css, cos = [], []
        for i, j, a, b in itertools.product(range(nocc), range(nocc),
                                            range(nvir), range(nvir)):
            ga, gb = nocc + a, nocc + b
            w_os = abs(float(t2[i, j, a, b]))
            if w_os > 0:
                uniq = sorted({i, j, ga, gb})
                for p in uniq:
                    A_os_site[p] += w_os
                cos.append((w_os, tuple(uniq)))
            if i < j and a < b:
                w_ss = abs(float(t2[i, j, a, b] - t2[i, j, b, a]))
                if w_ss > 0:
                    uniq = sorted({i, j, ga, gb})
                    for p, q in itertools.combinations(uniq, 2):
                        A_ss[p, q] += w_ss
                        A_ss[q, p] += w_ss
                    css.append((w_ss, tuple(uniq)))
        for i in range(nocc):
            for a in range(nvir):
                w = abs(float(self.t1[i, a]))
                A_ss[i, nocc + a] += w
                A_ss[nocc + a, i] += w
        self.A_ss, self.A_os_site = A_ss, A_os_site
        self.channels_ss = sorted(css, key=lambda c: -c[0])[:K_CHANNELS]
        self.channels_os = sorted(cos, key=lambda c: -c[0])[:K_CHANNELS]


def score1(pos, amp, J_aa, J_ab, w_ss):
    ssp, oss = same_spin_pairs(pos), opp_spin_sites(pos)
    iu = np.triu_indices(amp.norb, k=1)

    tot = amp.A_ss[iu].sum()
    s_ss = (sum(amp.A_ss[p, q] for p, q in ssp) / tot) if tot > 0 else 0.0
    tot = amp.A_os_site.sum()
    s_os = (sum(amp.A_os_site[p] for p in oss) / tot) if tot > 0 else 0.0

    M_ss = np.abs(J_aa).sum(axis=0) * amp.A_ss
    M_os = np.abs(J_ab).sum(axis=0).diagonal() * amp.A_os_site
    tot2 = M_ss[iu].sum()
    s_ss2 = (sum(M_ss[p, q] for p, q in ssp) / tot2) if tot2 > 0 else 0.0
    tot2 = M_os.sum()
    s_os2 = (sum(M_os[p] for p in oss) / tot2) if tot2 > 0 else 0.0

    return dict(s1_amp=w_ss * s_ss + (1 - w_ss) * s_os,
                s1_amp_ss=s_ss, s1_amp_os=s_os,
                s1_ampJ=w_ss * s_ss2 + (1 - w_ss) * s_os2,
                s1_ampJ_ss=s_ss2, s1_ampJ_os=s_os2)


def _span(pos, orbs):
    ps = [pos[o] for o in orbs]
    return int(max(ps) - min(ps))


def _anchor_dist(pos, orbs):
    m = CFG["anchor_mod"]
    return min(min(abs(int(pos[o]) - a) for a in range(0, len(pos), m))
               for o in orbs)


def score2(pos, amp, w_ss, L=L_SPAN_SS, D=D_ANCHOR_OS):
    def frac(ch, test):
        tot = sum(w for w, _ in ch)
        return (sum(w for w, o in ch if test(o)) / tot) if tot > 0 else 0.0
    r_ss = frac(amp.channels_ss, lambda o: _span(pos, o) <= L)
    r_os = frac(amp.channels_os, lambda o: _anchor_dist(pos, o) <= D)
    tot = sum(w for w, _ in amp.channels_ss) or 1.0
    soft = sum(w * np.exp(-max(0, _span(pos, o) - 3) / 2.0)
               for w, o in amp.channels_ss) / tot
    return dict(s2=w_ss * r_ss + (1 - w_ss) * r_os,
                s2_ss=r_ss, s2_os=r_os, s2_soft_ss=soft)


def hill_climb(objective, norb, seed=0, restarts=12, max_sweeps=200):
    """Maximise a permutation objective by pairwise swaps. Non-oracle by
    construction: `objective` may only see amplitudes and the mask."""
    rng = np.random.default_rng(seed)
    best_perm, best_val = None, -np.inf
    for r in range(restarts):
        perm = np.arange(norb) if r == 0 else rng.permutation(norb)
        val = objective(perm)
        for _ in range(max_sweeps):
            improved = False
            for i, j in itertools.combinations(range(norb), 2):
                cand = perm.copy()
                cand[i], cand[j] = cand[j], cand[i]
                v = objective(cand)
                if v > val + 1e-12:
                    perm, val, improved = cand, v, True
            if not improved:
                break
        if val > best_val:
            best_perm, best_val = perm.copy(), val
    return best_perm, best_val


# ==========================================================================
# STAGE 0 -- inspection
# ==========================================================================
def stage0():
    banner("STAGE 0 -- environment, API signatures, apply_unitary mutation test")
    import inspect
    import ffsim, pyscf, qiskit
    from importlib.metadata import version as package_version

    print(f"ffsim {package_version('ffsim')} | pyscf {package_version('pyscf')} | "
          f"qiskit {package_version('qiskit')} | numpy {package_version('numpy')}")
    for name in ("UCJOpSpinBalanced", "UCJOperator"):
        cls = getattr(ffsim, name, None)
        if cls is not None:
            print(f"  {name}.from_t_amplitudes"
                  f"{inspect.signature(cls.from_t_amplitudes)}")
    print(f"  MolecularData.from_scf{inspect.signature(ffsim.MolecularData.from_scf)}")
    from pyscf import lo
    print(f"  lo.Boys{inspect.signature(lo.Boys.__init__)}")
    print(f"  sbd binary: {CFG['sbd_bin'] or '(unset -- export SBD_BIN)'}")

    # --- apply_unitary mutation test ------------------------------------
    import pyscf.gto, pyscf.scf, pyscf.cc
    norb, nelec = CFG["n2_norb"], (3, 3)
    mol = pyscf.gto.M(atom=[["N", (0, 0, 0)], ["N", (0, 0, CFG["n2_bond"])]],
                      basis=CFG["n2_basis"], symmetry=False, verbose=0)
    mf = pyscf.scf.RHF(mol).run()
    active = range(CFG["n2_frozen"], CFG["n2_frozen"] + norb)
    frozen = [i for i in range(mol.nao_nr()) if i not in active]
    mycc = pyscf.cc.CCSD(mf, frozen=frozen).run()
    op = build_ucj(mycc.t2, mycc.t1)

    ref = ffsim.hartree_fock_state(norb, nelec)
    before = ref.copy()
    print(f"\n  ref before: nonzero={np.count_nonzero(ref)}, "
          f"norm={np.linalg.norm(ref):.10f}")
    _ = ffsim.apply_unitary(ref, op, norb=norb, nelec=nelec)
    print(f"  ref after : nonzero={np.count_nonzero(ref)}, "
          f"norm={np.linalg.norm(ref):.10f}")
    ok = np.allclose(ref, before)
    print(f"  UNCHANGED : {ok}")
    if not ok:
        print("  >>> apply_unitary MUTATES its input. Every call site must pass "
              "a copy. This is consistent with the direct_optimise.py anomaly "
              "(identity 47.19 mHa, seed sd 1.52).")
    else:
        print("  >>> apply_unitary does NOT mutate. The direct_optimise anomaly "
              "was the FCIDUMP divergence alone; the mutation hypothesis is "
              "disproved and should be recorded as such.")
    return ok


# ==========================================================================
# shared ffsim helpers
# ==========================================================================
def build_ucj(t2, t1=None, interaction_pairs=None):
    import ffsim
    cls = getattr(ffsim, "UCJOpSpinBalanced", None) or ffsim.UCJOperator
    kw = {}
    if t1 is not None:
        kw["t1"] = np.asarray(t1)
    if CFG["n_reps"] is not None:
        kw["n_reps"] = CFG["n_reps"]
    if interaction_pairs is not None:
        kw["interaction_pairs"] = interaction_pairs
    return cls.from_t_amplitudes(np.asarray(t2), **kw)


def diag_coulomb(op):
    dcm = np.asarray(op.diag_coulomb_mats)
    if dcm.ndim == 4 and dcm.shape[1] == 2:
        return dcm[:, 0], dcm[:, 1]
    if dcm.ndim == 3:
        return dcm, dcm
    raise RuntimeError(f"unexpected diag_coulomb_mats shape {dcm.shape}")


# ==========================================================================
# STAGE 1 -- canonical N2 score analysis, zero new sampling
# ==========================================================================
def stage1(outdir="outputs/stage1"):
    banner("STAGE 1 -- non-oracle scores on existing canonical N2 data")
    os.makedirs(outdir, exist_ok=True)

    ref_path = CFG["canonical_reference"]
    if not os.path.exists(ref_path):
        print(f"[ref] {ref_path} missing -- rebuilding from cached canonical run")
        os.makedirs(os.path.dirname(ref_path) or ".", exist_ok=True)
        import pyscf.gto, pyscf.scf, pyscf.cc
        norb = CFG["n2_norb"]
        mol = pyscf.gto.M(atom=[["N", (0, 0, 0)],
                                ["N", (0, 0, CFG["n2_bond"])]],
                          basis=CFG["n2_basis"], symmetry=False, verbose=0)
        mf = pyscf.scf.RHF(mol).run()
        active = range(CFG["n2_frozen"], CFG["n2_frozen"] + norb)
        frozen = [i for i in range(mol.nao_nr()) if i not in active]
        cc = pyscf.cc.CCSD(mf, frozen=frozen).run()
        Jaa, Jab = diag_coulomb(build_ucj(cc.t2, cc.t1))
        np.savez(ref_path, t1=cc.t1, t2=cc.t2, J_aa=Jaa, J_ab=Jab,
                 nocc=cc.t1.shape[0], norb=norb)

    R = np.load(ref_path)
    J_aa, J_ab = R["J_aa"], R["J_ab"]
    norb, nocc = int(R["norb"]), int(R["nocc"])
    amp = Amplitudes(R["t1"], R["t2"], nocc, norb)
    w_ss = float(np.abs(J_aa).sum() / (np.abs(J_aa).sum() + np.abs(J_ab).sum()))

    df = pd.read_csv(CFG["canonical_results"])
    cols = {c.lower(): c for c in df.columns}
    C = {}
    for k, cands in COLUMN_CANDIDATES.items():
        for c in cands:
            if c.lower() in cols:
                C[k] = cols[c.lower()]
                break

    for req in ("ordering", "permutation"):
        if req not in C:
            sys.exit(
                f"FATAL: no {req!r} column in {CFG['canonical_results']}. "
                f"Columns: {list(df.columns)}"
            )

    if "error" not in C:
        sys.exit(
            f"FATAL: no recognized error column in {CFG['canonical_results']}. "
            f"Expected one of {COLUMN_CANDIDATES['error']}; "
            f"columns: {list(df.columns)}"
        )

    n0 = len(df)
    dim_req = CFG["n_dets"] ** 2
    if "dim" in C:
        df = df[df[C["dim"]] == dim_req]
    elif "dim_alpha" in C and "dim_beta" in C:
        df = df[(df[C["dim_alpha"]] * df[C["dim_beta"]]) == dim_req]
    df = df[~df[C["ordering"]].astype(str).str.lower().isin(EXCLUDE_NAMES)]
    print(f"[filter] {n0} -> {len(df)} rows at dim={dim_req}")

    agg = {C["error"]: "mean"}
    for k in ("retained_J", "captured"):
        if k in C:
            agg[C[k]] = "mean"
    g = df.groupby([C["ordering"], C["permutation"]], as_index=False).agg(agg)
    ren = {C["ordering"]: "ordering", C["permutation"]: "permutation",
           C["error"]: "err_mHa"}
    for k in ("retained_J", "captured"):
        if k in C:
            ren[C[k]] = k
    g = g.rename(columns=ren)
    print(f"[agg] {len(g)} unique orderings")

    perms = {r.ordering: parse_permutation(r.permutation, norb)
             for r in g.itertuples()}

    # --- CONVENTION GUARD (Spearman > 0.99 required) --------------------
    conv = "layout"
    if "retained_J" in g.columns:
        best = (-2.0, None)
        for c in ("layout", "position"):
            rj = np.array([retained_J_of(positions_from(perms[o], c), J_aa, J_ab)
                           for o in g.ordering])
            rho = spearmanr(rj, g.retained_J.to_numpy()).statistic
            r = pearsonr(rj, g.retained_J.to_numpy()).statistic
            print(f"[guard] {c:9s} spearman={rho:+.5f}  pearson={r:+.5f}")
            if rho > best[0]:
                best = (rho, c)
        if best[0] <= 0.99:
            sys.exit(
                f"FATAL: cannot reproduce retained_J from op_full + mask "
                f"(best spearman {best[0]:+.5f} <= 0.99).\n"
                "My mask model does not match yours. Every score below would be "
                "scoring the wrong permutation. Fix same_spin_pairs / "
                "opp_spin_sites / retained_J_of to match your implementation "
                "before trusting anything.")
        conv = best[1]
        print(f"[guard] PASS -- convention '{conv}' (spearman {best[0]:+.5f})")
    else:
        print("[guard] SKIPPED (no retained_J column). Scores are UNVALIDATED.")

    rows = []
    for o in g.ordering:
        pos = positions_from(perms[o], conv)
        row = {"ordering": o, "retained_J_recomputed": retained_J_of(pos, J_aa, J_ab)}
        row.update(score1(pos, amp, J_aa, J_ab, w_ss))
        row.update(score2(pos, amp, w_ss))
        for L in SENSITIVITY_L:
            row[f"s2_ss_L{L}"] = score2(pos, amp, w_ss, L=L)["s2_ss"]
        rows.append(row)
    out = g.merge(pd.DataFrame(rows), on="ordering")

    named = ~out.ordering.astype(str).str.match(r"^[qr]\d+$")
    rnd = out[~named].reset_index(drop=True)
    err = rnd.err_mHa.to_numpy()
    rand_regret = float(err.mean() - err.min())
    print(f"\n[base] {len(rnd)} random orderings | best {err.min():.2f} "
          f"mean {err.mean():.2f} worst {err.max():.2f} mHa")
    print(f"[base] random-selection regret = {rand_regret:.2f} mHa")

    score_cols = [c for c in ("retained_J", "retained_J_recomputed", "s1_amp",
                              "s1_amp_ss", "s1_amp_os", "s1_ampJ", "s1_ampJ_ss",
                              "s1_ampJ_os", "s2", "s2_ss", "s2_os", "s2_soft_ss")
                  if c in rnd.columns]
    print(f"\n{'score':<24}{'rho':>9}{'p':>11}{'regret':>10}{'vs rand':>9}{'picks':>10}")
    res = {}
    for c in score_cols:
        x = rnd[c].to_numpy(float)
        if np.allclose(x, x[0]):
            print(f"{c:<24}   constant across orderings -- carries no signal")
            continue
        sr = spearmanr(x, err)
        k = int(np.argmax(x))
        reg = float(err[k] - err.min())
        res[c] = dict(rho=sr.statistic, p=sr.pvalue, regret=reg,
                      pick=rnd.ordering.iloc[k])
        print(f"{c:<24}{sr.statistic:>+9.3f}{sr.pvalue:>11.1e}{reg:>10.2f}"
              f"{reg/rand_regret:>9.2f}{str(rnd.ordering.iloc[k]):>10}")

    if "captured" in rnd.columns:
        print("\n[oracle diagnostic -- CASCI captured, not part of any score]")
        print(f"  rho(captured, error) = "
              f"{spearmanr(rnd.captured.to_numpy(float), err).statistic:+.3f}"
              "   [ceiling]")
        for c in ("s1_amp", "s1_ampJ", "s2"):
            if c in rnd.columns and not np.allclose(rnd[c], rnd[c].iloc[0]):
                print(f"  rho({c}, captured) = "
                      f"{spearmanr(rnd[c].to_numpy(float), rnd.captured.to_numpy(float)).statistic:+.3f}")

    print("\n[named orderings: error percentile vs score percentile]")
    print(f"{'ordering':<22}{'err_mHa':>9}{'err_pct':>9}"
          f"{'s1_amp':>9}{'s1_ampJ':>9}{'s2':>9}")
    for r in out[named].itertuples():
        line = (f"{str(r.ordering):<22}{r.err_mHa:>9.2f}"
                f"{100.0*(err > r.err_mHa).mean():>9.1f}")
        for c in ("s1_amp", "s1_ampJ", "s2"):
            line += f"{100.0*(rnd[c].to_numpy() < getattr(r, c)).mean():>9.1f}"
        print(line)

    print("\n[L sensitivity -- EXPLORATORY, not pre-registered]")
    for L in SENSITIVITY_L:
        x = rnd[f"s2_ss_L{L}"].to_numpy(float)
        if np.allclose(x, x[0]):
            print(f"  L={L}: degenerate")
            continue
        sr = spearmanr(x, err)
        print(f"  L={L}: rho={sr.statistic:+.3f} p={sr.pvalue:.1e} "
              f"regret={err[int(np.argmax(x))]-err.min():.2f} mHa")

    banner("STAGE 1 VERDICT")
    go = []
    for c in ("s1_amp", "s1_ampJ", "s2"):
        if c not in res:
            continue
        r = res[c]
        ok = (abs(r["rho"]) >= GO_RHO and r["p"] < GO_P
              and r["regret"] <= GO_REGRET_FRAC * rand_regret)
        tag = "GO" if ok else ("CONDITIONAL" if abs(r["rho"]) >= 0.35 else "NO-GO")
        print(f"  {c:<10} rho={r['rho']:+.3f} p={r['p']:.1e} "
              f"regret={r['regret']:.2f} -> {tag}")
        if ok:
            go.append(c)
    print(f"\n  criteria: |rho|>={GO_RHO}, p<{GO_P}, regret<="
          f"{GO_REGRET_FRAC*rand_regret:.2f} mHa")
    if go:
        print(f"  CLEARED: {go}. Carry unchanged to H10. Do not re-tune here.")
    else:
        print("  Nothing cleared. This does not kill H10 -- canonical N2 is the")
        print("  atypical case by hypothesis -- but you may not claim a")
        print("  predictor on this evidence. H10 still runs: the physical-order")
        print("  recovery test does not need a score that correlates here.")

    out.to_csv(f"{outdir}/nonoracle_scores.csv", index=False)
    print(f"\n[out] {outdir}/nonoracle_scores.csv")
    return go


# ==========================================================================
# STAGE 2 -- block-Boys covariance check
# ==========================================================================
def block_boys(mol, mf, active, n_act_occ, tag=""):
    from pyscf import lo
    C = np.asarray(mf.mo_coeff)
    S = mf.get_ovlp()
    occ_idx, vir_idx = active[:n_act_occ], active[n_act_occ:]
    np.random.seed(0)
    outs = {}
    for name, idx in (("occ", occ_idx), ("vir", vir_idx)):
        loc = lo.Boys(mol, C[:, idx])
        loc.init_guess = "atomic"
        loc.conv_tol = 1e-10
        loc.max_cycle = 500
        Cl = loc.kernel()
        U = C[:, idx].T @ S @ Cl
        dev = np.abs(U.T @ U - np.eye(U.shape[0])).max()
        span = np.abs(C[:, idx] @ C[:, idx].T @ S - Cl @ Cl.T @ S).max()
        if dev > 1e-9 or span > 1e-8:
            sys.exit(f"FATAL: {tag}{name} block-Boys broke unitarity/span "
                     f"(dev={dev:.2e}, span={span:.2e}).")
        print(f"[boys] {tag}{name}: {len(idx)} orbs, unitary {dev:.1e}, "
              f"span {span:.1e}")
        outs[name] = (Cl, U)
    Cn = C.copy()
    Cn[:, occ_idx] = outs["occ"][0]
    Cn[:, vir_idx] = outs["vir"][0]
    return Cn, outs["occ"][1], outs["vir"][1]


def rotate_amplitudes(t1, t2, Uo, Uv):
    t1L = np.einsum("iI,aA,ia->IA", Uo, Uv, t1, optimize=True)
    t2L = np.einsum("iI,jJ,aA,bB,ijab->IJAB", Uo, Uo, Uv, Uv, t2, optimize=True)
    return t1L, t2L


def stage2(n_perms=200, seed=2026):
    banner("STAGE 2 -- block-Boys covariance check (is localized N2 a real test?)")
    import pyscf.gto, pyscf.scf, pyscf.cc
    norb = CFG["n2_norb"]
    mol = pyscf.gto.M(atom=[["N", (0, 0, 0)], ["N", (0, 0, CFG["n2_bond"])]],
                      basis=CFG["n2_basis"], symmetry=False, verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    active = list(range(CFG["n2_frozen"], CFG["n2_frozen"] + norb))
    frozen = [i for i in range(mol.nao_nr()) if i not in active]
    n_act_occ = int((mf.mo_occ > 0).sum()) - CFG["n2_frozen"]
    cc = pyscf.cc.CCSD(mf, frozen=frozen)
    cc.conv_tol = 1e-10
    cc.kernel()
    print(f"[scf ] E_RHF={mf.e_tot:.12f}  E_corr={cc.e_corr:.12f}  "
          f"n_act_occ={n_act_occ}")

    _, Uo, Uv = block_boys(mol, mf, active, n_act_occ)
    t1L, t2L = rotate_amplitudes(np.asarray(cc.t1), np.asarray(cc.t2), Uo, Uv)
    print(f"[amp ] ||t2|| canonical {np.linalg.norm(cc.t2):.10f}  "
          f"localized {np.linalg.norm(t2L):.10f}")

    Jc = diag_coulomb(build_ucj(cc.t2, cc.t1))
    Jl = diag_coulomb(build_ucj(t2L, t1L))
    spec = lambda J: np.sort(np.abs(np.asarray(J)).ravel())[::-1]
    d_aa = float(np.abs(spec(Jc[0]) - spec(Jl[0])).max())
    d_ab = float(np.abs(spec(Jc[1]) - spec(Jl[1])).max())
    print(f"[J   ] sorted-|J| spectra max diff: aa {d_aa:.3e}  ab {d_ab:.3e}  "
          f"(scale {np.abs(Jc[0]).max():.3e})")

    rng = np.random.default_rng(seed)
    rc, rl = [], []
    for _ in range(n_perms):
        pos = positions_from(rng.permutation(norb))
        rc.append(retained_J_of(pos, *Jc))
        rl.append(retained_J_of(pos, *Jl))
    rc, rl = np.asarray(rc), np.asarray(rl)
    rho = float(spearmanr(rc, rl).statistic)
    md = float(np.abs(rc - rl).max())
    print(f"[mask] retained_J over {n_perms} perms: spearman={rho:+.4f}  "
          f"max|diff|={md:.3e}")
    print(f"       canonical [{rc.min():.4f},{rc.max():.4f}]  "
          f"localized [{rl.min():.4f},{rl.max():.4f}]")

    banner("STAGE 2 VERDICT")
    if md < 1e-8 and d_aa < 1e-8:
        v = "VACUOUS"
        print("VACUOUS: double factorization absorbs the block-Boys rotation")
        print("into orbital_rotations. J is invariant, so the mask retains the")
        print("same entries and the localized ordering landscape is the")
        print("canonical one relabelled. Localized N2 would test determinant-")
        print("basis / finite-budget effects, NOT the ordering hypothesis.")
        print("-> Skip localized N2 as an ordering test. Go straight to H10.")
    elif rho > 0.98:
        v = "NEARLY VACUOUS"
        print("NEARLY VACUOUS: J differs numerically but the retention ordering")
        print("is essentially unchanged. Weak independent evidence. Prefer H10.")
    else:
        v = "MEANINGFUL"
        print("MEANINGFUL: genuinely different mask-retention landscape.")
        print("Localized N2 is worth running as a SECONDARY control after H10.")
    return dict(verdict=v, rho=rho, maxdiff=md, d_aa=d_aa, d_ab=d_ab)


# ==========================================================================
# STAGE 3 -- H10 primary experiment
# ==========================================================================
def parse_sbd_energy(text):
    """Extract the final total energy from sbd stdout.

    ADAPT HERE if your build prints a different final line. Returns None on
    failure so the caller can dump the tail rather than guess.
    """
    pats = [
        r"diagonalization:\s*Energy\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"[Ff]inal\s+[Ee]nergy\s*[:=]\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
            r"[Tt]otal\s+[Ee]nergy\s*[:=]\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
            r"\bE\s*=\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
            r"[Ee]nergy\s*[:=]?\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)"]
    for p in pats:
        m = re.findall(p, text)
        if m:
            return float(m[-1])
    return None


def build_h10(R, natoms, basis):
    import pyscf.gto, pyscf.scf, pyscf.cc, pyscf.fci
    atom = [["H", (0.0, 0.0, i * R)] for i in range(natoms)]
    mol = pyscf.gto.M(atom=atom, basis=basis, symmetry=False, verbose=0,
                      spin=0, charge=0)
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        sys.exit("FATAL: H10 RHF did not converge.")
    norb = mol.nao_nr()
    nocc = mol.nelectron // 2
    cc = pyscf.cc.CCSD(mf)          # full space: CAS(10e,10o), nothing frozen
    cc.conv_tol = 1e-9
    cc.max_cycle = 300
    cc.diis_space = 12
    cc.kernel()
    if not cc.converged:
        raise RuntimeError(
            "FATAL: H10 CCSD did not converge. Refusing to initialize LUCJ "
            "from unstable amplitudes. Try a shorter H–H separation."
        )
    return mol, mf, cc, norb, nocc


def orbital_centroids(mol, C, idx):
    """<phi|z|phi> per orbital -- gives the nuclear sequence along the chain."""
    z = mol.intor("int1e_r")[2]
    return np.array([float(C[:, i] @ z @ C[:, i]) for i in idx])


def physical_ordering(centroids, n_act_occ):
    """Sort all active orbitals by chain position; break ties occ-before-virt."""
    norb = len(centroids)
    key = [(centroids[p], 0 if p < n_act_occ else 1, p) for p in range(norb)]
    order = [p for _, _, p in sorted(key)]
    return np.asarray(order, dtype=int)     # 'layout': order[k] = orbital at k


def hf_bitstring(norb, nocc):
    return "0" * (norb - nocc) + "1" * nocc      # rightmost bit = orbital 1


def sample_bitstrings(op, norb, nelec, shots, seed):
    """Fresh AerSimulator per call. Returns (alpha_counts, beta_counts)."""
    import ffsim
    import ffsim.qiskit as fq
    from qiskit import QuantumCircuit, QuantumRegister, transpile
    from qiskit_aer import AerSimulator

    qr = QuantumRegister(2 * norb, "q")
    qc = QuantumCircuit(qr)
    qc.append(fq.PrepareHartreeFockJW(norb, nelec), qr)
    qc.append(fq.UCJOpSpinBalancedJW(op), qr)
    qc.measure_all()

    sim = AerSimulator(seed_simulator=seed)
    tkw = dict(seed_transpiler=CFG["seed_transpiler"], optimization_level=1)
    if CFG["use_pre_init"]:
        tkw["pre_init"] = fq.PRE_INIT
    try:
        tqc = transpile(qc, sim, **tkw)
    except TypeError:
        tkw.pop("pre_init", None)
        tqc = transpile(qc, sim, **tkw)
    counts = sim.run(tqc, shots=shots).result().get_counts()

    a, b = {}, {}
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        # validated 'split' transform: alpha = rightmost norb, beta = leftmost
        a[bits[norb:]] = a.get(bits[norb:], 0) + n
        b[bits[:norb]] = b.get(bits[:norb], 0) + n
    return a, b, int(tqc.depth())


def top_dets(counts, k, hf):
    """Top-k by marginal count, HF forced in (sbd needs it present)."""
    ranked = [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    if hf in ranked:
        ranked.remove(hf)
    sel = [hf] + ranked[: k - 1]
    return sel, len(counts)


def run_sbd(fcidump, adet, bdet, norb):
    cmd = [CFG["mpirun"], "-n", "1", CFG["sbd_bin"],
           "--fcidump", fcidump, "--adetfile", adet, "--bdetfile", bdet,
           "--bit_length", str(max(20, norb))]
    cmd += CFG["sbd_method_args"] + CFG["sbd_extra"]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=CFG["sbd_timeout"])
    e = parse_sbd_energy(p.stdout + "\n" + p.stderr)
    if e is None:
        print("---- sbd stdout tail ----")
        print("\n".join((p.stdout + p.stderr).splitlines()[-25:]))
        sys.exit("FATAL: could not parse an energy from sbd output. "
                 "Adapt parse_sbd_energy().")
    return e


def stage3(outdir="outputs/h10", cachedir="cache/h10", carry_scores=()):
    banner(
        f"STAGE 3 -- H10 chain, STO-6G, CAS(10e,10o), "
        f"R = {CFG['h10_R']} A"
    )
    if not CFG["sbd_bin"] or not shutil.which(CFG["sbd_bin"]):
        sys.exit("FATAL: set SBD_BIN to your sbd chemistry_tpb_selected_basis_"
                 "diagonalization binary (export SBD_BIN=/path/to/main).")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(cachedir, exist_ok=True)
    import ffsim
    from pyscf import mcscf
    from pyscf.tools import fcidump as fcidump_mod

    t0 = time.time()
    mol, mf, cc, norb, nocc = build_h10(CFG["h10_R"], CFG["h10_natoms"],
                                        CFG["h10_basis"])
    nelec = (nocc, nocc)
    active = list(range(norb))
    print(f"[h10 ] norb={norb} nelec={nelec} E_RHF={mf.e_tot:.12f} "
          f"E_corr={cc.e_corr:.12f}")

    # --- block-Boys localization + exact amplitude rotation --------------
    C_loc, Uo, Uv = block_boys(mol, mf, active, nocc, tag="h10-")
    t1L, t2L = rotate_amplitudes(np.asarray(cc.t1), np.asarray(cc.t2), Uo, Uv)

    md_can = ffsim.MolecularData.from_scf(mf, active_space=active)
    import copy as _copy
    mf_loc = _copy.copy(mf)
    mf_loc.mo_coeff = C_loc
    md_loc = ffsim.MolecularData.from_scf(mf_loc, active_space=active)

    def ecorr(t1, t2, md):
        eri = np.asarray(md.hamiltonian.two_body_tensor)
        o, v = slice(0, nocc), slice(nocc, norb)
        e = eri[o, v, o, v]
        tau = (2 * t2 - t2.transpose(0, 1, 3, 2)
               + 2 * np.einsum("ia,jb->ijab", t1, t1, optimize=True)
               - np.einsum("ib,ja->ijab", t1, t1, optimize=True))
        return float(np.einsum("ijab,iajb->", tau, e, optimize=True))

    e_can = ecorr(np.asarray(cc.t1), np.asarray(cc.t2), md_can)
    e_loc = ecorr(t1L, t2L, md_loc)
    print(f"[chk ] E_corr from amplitudes: canonical {e_can:.12f}  "
          f"localized {e_loc:.12f}  diff {abs(e_can-e_loc):.2e}")
    if abs(e_can - e_loc) > 1e-9:
        sys.exit("FATAL: amplitude rotation inconsistent with rotated "
                 "integrals. Do not proceed.")
# CASCI is active-space rotation invariant; permit small numerical noise only.
    def casci(C):
        mc = mcscf.CASCI(mf, norb, mol.nelectron)
        mc.verbose = 0
        return float(mc.kernel(C)[0])
    e_casci_can, e_casci_loc = casci(np.asarray(mf.mo_coeff)), casci(C_loc)
    print(f"[chk ] CASCI canonical {e_casci_can:.12f}  localized "
          f"{e_casci_loc:.12f}  diff {abs(e_casci_can-e_casci_loc):.2e}")
    if abs(e_casci_can - e_casci_loc) > 1e-7:
        sys.exit("FATAL: CASCI energies differ -- not the same active space.")
    E_EXACT = e_casci_loc

    # --- one FCIDUMP for every ordering ---------------------------------
    h1 = np.asarray(md_loc.hamiltonian.one_body_tensor)
    h2 = np.asarray(md_loc.hamiltonian.two_body_tensor)
    fci_path = os.path.join(cachedir, "fcidump.txt")
    fcidump_mod.from_integrals(fci_path, h1, h2, norb, mol.nelectron,
                               nuc=float(md_loc.core_energy), ms=0)
    print(f"[fcid] {fci_path}  sha={sha(fci_path)}")

    # --- ground truth + candidate orderings ------------------------------
    cent = orbital_centroids(mol, C_loc, active)
    phys = physical_ordering(cent, nocc)
    print(f"[phys] centroids (a.u.): "
          f"{np.array2string(cent, precision=2, suppress_small=True)}")
    print(f"[phys] physical (chain) ordering: {perm_to_str(phys)}")

    amp = Amplitudes(t1L, t2L, nocc, norb)
    Jaa, Jab = diag_coulomb(build_ucj(t2L, t1L))
    w_ss = float(np.abs(Jaa).sum() / (np.abs(Jaa).sum() + np.abs(Jab).sum()))

    obj_s2 = lambda p: score2(positions_from(p), amp, w_ss)["s2"]
    obj_s1 = lambda p: score1(
        positions_from(p), amp, Jaa, Jab, w_ss
    )["s1_ampJ"]
    obj_rj = lambda p: retained_J_of(positions_from(p), Jaa, Jab)
    print("[opt ] hill-climbing non-oracle objectives ...")
    p_s2, v_s2 = hill_climb(obj_s2, norb, seed=11)
    p_s1, v_s1 = hill_climb(obj_s1, norb, seed=12)
    p_rj, v_rj = hill_climb(obj_rj, norb, seed=13)
    print(f"[opt ] s2_max={perm_to_str(p_s2)} ({v_s2:.4f})  "
          f"s1_max={perm_to_str(p_s1)} ({v_s1:.4f})  "
          f"rJ_max={perm_to_str(p_rj)} ({v_rj:.4f})")

    tau = lambda a, b: float(kendalltau(positions_from(a),
                                        positions_from(b)).statistic)
    print(f"[test] Kendall tau vs PHYSICAL order: "
          f"s2_max {tau(p_s2, phys):+.3f}  s1_max {tau(p_s1, phys):+.3f}  "
          f"rJ_max {tau(p_rj, phys):+.3f}")

    named = {
        "physical": phys,
        "physical_reverse": phys[::-1].copy(),
        "identity": np.arange(norb),
        "reverse": np.arange(norb)[::-1].copy(),
        "s2_max": p_s2, "s1_max": p_s1, "retainedJ_max": p_rj,
    }
    rng = np.random.default_rng(20260821)
    orderings = list(named.items())
    orderings += [(f"h{i:03d}", rng.permutation(norb))
                  for i in range(CFG["n_random"])]

    # --- sweep ------------------------------------------------------------
    hf = hf_bitstring(norb, nocc)
    csv_path = os.path.join(outdir, "h10_results.csv")
    rows, wrote_header = [], False
    total = len(orderings) * len(CFG["seeds"])
    done, t_sweep = 0, time.time()
    print(f"\n[sweep] {len(orderings)} orderings x {len(CFG['seeds'])} seeds "
          f"= {total} evaluations")

    for name, perm in orderings:
        pos = positions_from(perm)
        centroids = orbital_centroids(mol, C_loc, active)
        pairs = interaction_pairs_for(pos, centroids, J_ab=Jab)
        try:
            op = build_ucj(t2L, t1L, interaction_pairs=pairs)
        except TypeError:
            sys.exit("FATAL: your ffsim from_t_amplitudes does not accept "
                     "interaction_pairs. Use the masked-construction helper "
                     "from your unified runner instead (edit build_ucj).")
        sc = {}
        sc.update(score1(pos, amp, Jaa, Jab, w_ss))
        sc.update(score2(pos, amp, w_ss))
        sc["retained_J"] = retained_J_of(pos, Jaa, Jab)

        for seed in CFG["seeds"]:
            ac, bc, depth = sample_bitstrings(op, norb, nelec, CFG["shots"], seed)
            asel, na = top_dets(ac, CFG["n_dets"], hf)
            bsel, nb = top_dets(bc, CFG["n_dets"], hf)
            row = dict(ordering=name, permutation=perm_to_str(perm), seed=seed,
                       depth=depth, n_unique_alpha=na, n_unique_beta=nb,
                       dim_alpha=len(asel), dim_beta=len(bsel),
                       dim=len(asel) * len(bsel), **sc)
            if len(asel) < CFG["n_dets"] or len(bsel) < CFG["n_dets"]:
                row.update(energy=np.nan, err_mHa=np.nan, status="SUPPORT_COLLAPSE")
                print(f"  {name:<16} seed={seed} SUPPORT COLLAPSE "
                      f"({len(asel)}a/{len(bsel)}b) -- recorded, excluded")
            else:
                ap = os.path.join(outdir, f"_a_{name}_{seed}.txt")
                bp = os.path.join(outdir, f"_b_{name}_{seed}.txt")
                open(ap, "w").write("\n".join(asel) + "\n")
                open(bp, "w").write("\n".join(bsel) + "\n")
                E = run_sbd(fci_path, ap, bp, norb)
                row.update(energy=E,
                           err_mHa=(E - E_EXACT) * HARTREE_TO_MHA,
                           status="OK", adet_sha=sha(ap), bdet_sha=sha(bp))
                os.remove(ap); os.remove(bp)
            rows.append(row)
            done += 1
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            if done == 2:
                eta = (time.time() - t_sweep) / done * (total - done) / 60
                print(f"  [eta] ~{eta:.0f} min remaining")
            if done % 25 == 0:
                print(f"  [{done}/{total}] {(time.time()-t_sweep)/60:.1f} min "
                      f"elapsed", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # --- analysis ---------------------------------------------------------
    banner("STAGE 3 ANALYSIS")
    ok = df[(df.status == "OK") & (df.dim == CFG["n_dets"] ** 2)]
    g = ok.groupby("ordering", as_index=False).agg(
        err_mHa=("err_mHa", "mean"), err_sd=("err_mHa", "std"),
        s2=("s2", "mean"), s1_amp=("s1_amp", "mean"),
        retained_J=("retained_J", "mean"))
    rnd = g[g.ordering.str.match(r"^h\d+$")]
    err = rnd.err_mHa.to_numpy()
    n_collapse = int((df.status == "SUPPORT_COLLAPSE").sum())
    print(f"exact CASCI E = {E_EXACT:.12f} Ha")
    print(f"random orderings kept: {len(rnd)}/{CFG['n_random']}  "
          f"(support collapses: {n_collapse} evaluations)")
    print(f"random error: best {err.min():.2f}  median {np.median(err):.2f}  "
          f"mean {err.mean():.2f}  worst {err.max():.2f} mHa")
    rand_regret = float(err.mean() - err.min())

    print(f"\n{'ordering':<18}{'err_mHa':>10}{'sd':>8}{'pctile':>9}"
          f"{'tau_vs_phys':>13}")
    for r in g[~g.ordering.str.match(r"^h\d+$")].itertuples():
        p = dict(orderings)[r.ordering]
        print(f"{r.ordering:<18}{r.err_mHa:>10.2f}"
              f"{(0.0 if np.isnan(r.err_sd) else r.err_sd):>8.2f}"
              f"{100.0*(err > r.err_mHa).mean():>9.1f}"
              f"{tau(p, phys):>13.3f}")

    print(f"\n{'score':<14}{'rho':>9}{'p':>11}{'regret':>10}{'vs rand':>9}")
    for c in ("s2", "s1_amp", "retained_J"):
        x = rnd[c].to_numpy(float)
        if np.allclose(x, x[0]):
            print(f"{c:<14}   constant")
            continue
        sr = spearmanr(x, err)
        reg = float(err[int(np.argmax(x))] - err.min())
        print(f"{c:<14}{sr.statistic:>+9.3f}{sr.pvalue:>11.1e}{reg:>10.2f}"
              f"{reg/rand_regret:>9.2f}")

    e_phys = float(g.loc[g.ordering == "physical", "err_mHa"].iloc[0])
    e_s2 = float(g.loc[g.ordering == "s2_max", "err_mHa"].iloc[0])
    banner("STAGE 3 VERDICT")
    print(f"physical-order error   : {e_phys:.2f} mHa "
          f"({100.0*(err > e_phys).mean():.1f}th pctile)")
    print(f"s2_max (non-oracle)    : {e_s2:.2f} mHa "
          f"({100.0*(err > e_s2).mean():.1f}th pctile)")
    print(f"Kendall tau(s2_max, physical) = {tau(p_s2, phys):+.3f}")
    if tau(p_s2, phys) > 0.6 and 100.0 * (err > e_s2).mean() > 80:
        print("\n-> The non-oracle rule RECOVERS the geometric ordering and")
        print("   lands in the top quintile. This is the publishable result:")
        print("   a chemically motivated rule, not a scalar that happens to")
        print("   correlate. Next: block-localized N2 control, then Cr2/FeS.")
    elif 100.0 * (err > e_phys).mean() < 60:
        print("\n-> The PHYSICAL ordering is not itself good. The premise that")
        print("   geometric locality is the right ordering principle for LUCJ")
        print("   is wrong, or the mask/ansatz does not exploit it. Do not")
        print("   scale up; diagnose this first.")
    else:
        print("\n-> Physical ordering is good but the non-oracle rule does not")
        print("   recover it. The target exists and is reachable; the rule is")
        print("   the missing piece. Iterate on the construction, not on scale.")

    meta = dict(system="H10", R=CFG["h10_R"], basis=CFG["h10_basis"],
                norb=norb, nelec=list(nelec), e_rhf=float(mf.e_tot),
                e_corr=float(cc.e_corr), cc_converged=bool(cc.converged),
                e_casci=E_EXACT, n_reps=CFG["n_reps"], shots=CFG["shots"],
                n_dets=CFG["n_dets"], seeds=list(CFG["seeds"]),
                use_pre_init=CFG["use_pre_init"],
                fcidump_sha=sha(fci_path), mo_loc_sha=sha(C_loc),
                t2_loc_sha=sha(t2L), physical_order=perm_to_str(phys),
                s2_max=perm_to_str(p_s2), s1_max=perm_to_str(p_s1),
                retainedJ_max=perm_to_str(p_rj),
                tau_s2_phys=tau(p_s2, phys), n_support_collapse=n_collapse,
                elapsed_min=round((time.time() - t0) / 60, 2))
    with open(os.path.join(outdir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n[out] {csv_path}\n[out] {outdir}/metadata.json")
    print(f"[done] {meta['elapsed_min']} min")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["stage0", "stage1", "stage2", "stage3", "all"])
    ap.add_argument("--results", default=CFG["canonical_results"])
    ap.add_argument("--n-random", type=int, default=CFG["n_random"])
    ap.add_argument("--shots", type=int, default=CFG["shots"])
    ap.add_argument("--n-reps", type=int, default=None)
    ap.add_argument("--sbd-bin", default=CFG["sbd_bin"])
    ap.add_argument("--force", action="store_true",
                    help="run stage3 even if earlier stages advise against it")
    a = ap.parse_args()
    CFG["canonical_results"] = a.results
    CFG["n_random"] = a.n_random
    CFG["shots"] = a.shots
    CFG["sbd_bin"] = a.sbd_bin
    if a.n_reps is not None:
        CFG["n_reps"] = a.n_reps

    if a.stage == "stage0":
        stage0()
    elif a.stage == "stage1":
        stage1()
    elif a.stage == "stage2":
        stage2()
    elif a.stage == "stage3":
        stage3()
    else:
        stage0()
        carry = stage1()
        v = stage2()
        print(f"\n[gate] stage2 verdict: {v['verdict']}  "
              f"(localized N2 is {'a real' if v['verdict']=='MEANINGFUL' else 'NOT a valid'} ordering test)")
        stage3(carry_scores=carry)


if __name__ == "__main__":
    main()