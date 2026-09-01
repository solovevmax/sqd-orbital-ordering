#!/usr/bin/env python3
"""
experiments/preflight.py
=========================

Validation gates for the SQD orbital-ordering pipeline. Every operator
construction, sampling, determinant-writing, and sbd-invocation call here
goes through the existing pipeline modules - nothing is reimplemented:

  unified_run.py            N2 CAS(6,10) @ 1.55 A, mechanism A (P J P^T
                             absorbed into orbital_rotations, fixed mask
                             applied after). Reference cached once in
                             outputs/unified/reference.pkl.
  run_ordering_pipeline.py  H10 (and, for crosscheck, N2) mechanism B
                             (operator never rotated; the permutation
                             selects which physical orbital pairs enter
                             ffsim's interaction_pairs).
  src/sqd_ordering/mask.py  the single shared mask model both of the above
                             now import from.

Subcommands
-----------
  crosscheck    Do mechanism A and mechanism B agree on N2, using the SAME
                cached reference and the SAME FCIDUMP for both? All H10
                results use mechanism B; all validated N2 results use
                mechanism A.

                Primary test: OPERATOR-LEVEL equivalence, no sampling, no
                sbd. If ffsim's interaction_pairs performs post-hoc zeroing
                (not a constrained fit), then for a permutation matrix P
                (P^-1 = P^T):

                    P^T [ M_fixed .* (P J P^T) ] P == (P^T M_fixed P) .* J
                                                    == M_B .* J

                so un-permuting mechanism A's masked operator must equal
                mechanism B's operator exactly, entrywise, with no sampling
                noise involved. Secondary test (gated on the primary
                passing): end-to-end sampling + sbd at 1e6 shots, reported
                as confirmation only - finite-shot energies are NOT expected
                to match exactly since A's orbital_rotations are permuted
                and B's are not.

Thread pinning MUST precede any numpy/pyscf import - matches the existing
pipeline's discipline (see run_ordering_pipeline.py's module docstring):
recomputing reference data across processes with unpinned threads has
previously produced different energies for the same ordering because
thread-dependent floating point flips which determinants make the budget
cut.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

OUTDIR = Path(__file__).resolve().parent / "outputs" / "preflight"
OUTDIR.mkdir(parents=True, exist_ok=True)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def git_commit_hash() -> str:
    """Current git commit hash of the repository (short + dirty flag)."""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"(unavailable: {exc!r})"


def sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==========================================================================
# crosscheck: does mechanism A (unified_run.py) agree with mechanism B
# (run_ordering_pipeline.py) on the SAME N2 reference and FCIDUMP?
# ==========================================================================

def unpermute_diag_coulomb_and_rotations(
    diag_coulomb_mats: np.ndarray, orbital_rotations: np.ndarray, perm: np.ndarray, norb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Invert unified_run.permute_operator's P J P^T / U P^T relabelling.

    P is a permutation matrix, so P^-1 = P^T; applying the same einsum
    formulas permute_operator uses, with P replaced by P^T, computes
    P^T J P and U P (round-trip-verified: permute then unpermute reproduces
    the pre-permutation operator to machine precision).
    """
    P_T = np.eye(norb)[list(perm)].T
    J_unpermuted = np.einsum("ij,rsjk,lk->rsil", P_T, diag_coulomb_mats, P_T)
    U_unpermuted = np.einsum("rij,kj->rik", orbital_rotations, P_T)
    return J_unpermuted, U_unpermuted


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def operator_level_check(U: Any, R: Any, perm: np.ndarray, t1, t2, norb: int) -> dict[str, Any]:
    """Compare mechanism A (un-permuted) against mechanism B, entrywise, on
    diag_coulomb_mats (aa, ab separately) and orbital_rotations. No sampling,
    no sbd.
    """
    op_a = U.apply_mask(U.permute_operator(U.op_full, perm))
    pos = R.positions_from(perm)
    pairs = R.interaction_pairs_for(pos)
    op_b = R.build_ucj(t2, t1, interaction_pairs=pairs)

    J_a = np.asarray(op_a.diag_coulomb_mats)
    U_a = np.asarray(op_a.orbital_rotations)
    J_a_unpermuted, U_a_unpermuted = unpermute_diag_coulomb_and_rotations(J_a, U_a, perm, norb)

    J_b = np.asarray(op_b.diag_coulomb_mats)
    U_b = np.asarray(op_b.orbital_rotations)

    n_pairs_aa_b, n_pairs_ab_b = len(pairs[0]), len(pairs[1])

    shape_match = (J_a_unpermuted.shape == J_b.shape) and (U_a_unpermuted.shape == U_b.shape)
    result: dict[str, Any] = dict(
        perm="".join(map(str, perm)),
        n_reps_A=J_a_unpermuted.shape[0], n_reps_B=J_b.shape[0],
        shape_match=shape_match,
        n_pairs_aa_B=n_pairs_aa_b, n_pairs_ab_B=n_pairs_ab_b,
    )
    if not shape_match:
        result.update(diff_aa=float("nan"), diff_ab=float("nan"), diff_U=float("nan"),
                      equivalent=False)
        return result

    diff_aa = float(np.max(np.abs(J_a_unpermuted[:, 0] - J_b[:, 0])))
    diff_ab = float(np.max(np.abs(J_a_unpermuted[:, 1] - J_b[:, 1])))
    diff_U = float(np.max(np.abs(U_a_unpermuted - U_b)))
    result.update(diff_aa=diff_aa, diff_ab=diff_ab, diff_U=diff_U,
                  equivalent=(diff_aa < 1e-12 and diff_ab < 1e-12 and diff_U < 1e-12))
    return result


def pair_set_difference(R: Any, perm: np.ndarray, norb: int) -> dict[str, Any]:
    """Diagnostic only: which (orbital-index) pairs mechanism B requests for
    this ordering, for inspection when pair counts look wrong.
    """
    pos = R.positions_from(perm)
    aa, ab = R.interaction_pairs_for(pos)
    return dict(aa_pairs_B=aa, ab_pairs_B=ab)


def run_operator_level_crosscheck(U: Any, R: Any, orderings: dict[str, np.ndarray],
                                  t1, t2, norb: int) -> tuple[bool, list[dict]]:
    banner("CROSSCHECK (PRIMARY) -- operator-level equivalence, no sampling, no sbd")
    rows = []
    for name, perm in orderings.items():
        r = operator_level_check(U, R, perm, t1, t2, norb)
        r["ordering"] = name
        rows.append(r)
        print(f"{name:<14} perm={r['perm']}  n_reps A/B={r['n_reps_A']}/{r['n_reps_B']}  "
              f"diff_aa={r['diff_aa']:.3e}  diff_ab={r['diff_ab']:.3e}  "
              f"diff_U={r['diff_U']:.3e}  equivalent={r['equivalent']}")

    all_shape_match = all(r["shape_match"] for r in rows)
    all_equivalent = all(r["equivalent"] for r in rows)
    return all_equivalent, rows


def run_sampling_confirmation(U: Any, R: Any, orderings: dict[str, np.ndarray],
                              t1, t2, norb: int, nelec: tuple, fcidump_path: Path,
                              shots: int) -> list[dict]:
    banner("CROSSCHECK (SECONDARY, CONFIRMATION ONLY) -- end-to-end sampling + sbd")
    print("Operator-level equivalence already established above; this checks that")
    print("finite-shot sampling + sbd agree in practice too (not expected to be exact -")
    print("A's orbital rotations are permuted, B's are not - only within N2 seed noise).")
    nocc = nelec[0]
    hf = R.hf_bitstring(norb, nocc)
    seed_sim = 2026
    rows = []
    for name, perm in orderings.items():
        print(f"\n--- ordering: {name}  perm={''.join(map(str, perm))} ---")

        out_a = U.evaluate(name, perm, "crosscheck_A", seeds=[seed_sim])[0]
        energy_a = out_a["err_sub_mHa"] / 1000.0 + U.E_CASCI
        a_det = (U.WORK / f"{name}_{seed_sim}_a.txt").read_text().split()
        b_det = (U.WORK / f"{name}_{seed_sim}_b.txt").read_text().split()
        # NOT remapped via perm: mechanism A's own established convention
        # (captured_of/retained_J_of, used for every published N2 result)
        # already treats raw qubit index k as canonical orbital k directly -
        # permute_operator only reshuffles which (J,U) VALUES are plugged
        # into the fixed masked positions, it does not relabel what a
        # sampled bit means. See run_invariance's note: applying perm here
        # scrambles two already-comparable sets instead of aligning them.
        a_canon = set(a_det)
        b_canon = set(b_det)

        pos = R.positions_from(perm)
        pairs = R.interaction_pairs_for(pos)
        op_b = R.build_ucj(t2, t1, interaction_pairs=pairs)
        ac_b, bc_b, _depth_b = R.sample_bitstrings(op_b, norb, nelec, shots, seed_sim)
        asel_b, na_b = R.top_dets(ac_b, U.BUDGET, hf)
        bsel_b, nb_b = R.top_dets(bc_b, U.BUDGET, hf)
        adet_path = OUTDIR / f"_crosscheck_{name}_a.txt"
        bdet_path = OUTDIR / f"_crosscheck_{name}_b.txt"
        adet_path.write_text("\n".join(asel_b) + "\n")
        bdet_path.write_text("\n".join(bsel_b) + "\n")
        energy_b = R.run_sbd(str(fcidump_path), str(adet_path), str(bdet_path), norb)

        delta_mha = (energy_a - energy_b) * 1000.0
        jacc_a = jaccard(a_canon, set(asel_b))
        jacc_b = jaccard(b_canon, set(bsel_b))
        row = dict(ordering=name, perm="".join(map(str, perm)),
                  energy_A=energy_a, energy_B=energy_b, delta_mHa=delta_mha,
                  jaccard_alpha=jacc_a, jaccard_beta=jacc_b)
        rows.append(row)
        print(f"  energy_A={energy_a:.12f}  energy_B={energy_b:.12f}  delta={delta_mha:.4f} mHa")
        print(f"  Jaccard (alpha)={jacc_a:.4f}  Jaccard (beta)={jacc_b:.4f}")
        if jacc_a < 0.9 or jacc_b < 0.9:
            print(f"  NOTE: Jaccard below 0.9 for {name!r} (alpha={jacc_a:.4f}, beta={jacc_b:.4f})")
    return rows


def run_crosscheck(shots: int) -> int:
    banner("CROSSCHECK -- mechanism A (unified_run) vs mechanism B (run_ordering_pipeline)")
    print("Loading mechanism A module (unified_run.py) ...")
    import unified_run as U
    print("Loading mechanism B module (run_ordering_pipeline.py) ...")
    import run_ordering_pipeline as R

    if not U.SBD.exists():
        sys.exit(f"FATAL: sbd binary not found at expected path {U.SBD}")
    R.CFG["sbd_bin"] = str(U.SBD)

    norb, nelec = U.NORB, U.NELEC
    t1, t2 = U.ref_data["t1"], U.ref_data["t2"]
    fcidump_path = U.FCIDUMP
    print("Shared cached reference (identical t1/t2/FCIDUMP fed to both mechanisms):")
    print(f"  N2 CAS({sum(nelec)},{norb}) @ {U.BOND} A, {U.BASIS}")
    print(f"  E_CASCI (cached) = {U.E_CASCI:.12f}")
    print(f"  shared FCIDUMP   = {fcidump_path}  sha256={sha256_of(fcidump_path)[:16]}")

    orderings: dict[str, np.ndarray] = {
        "identity": np.arange(norb),
        "reverse": np.arange(norb)[::-1].copy(),
    }
    for s in (201, 202, 203):
        orderings[f"rand_seed{s}"] = np.random.default_rng(s).permutation(norb)

    all_equivalent, op_rows = run_operator_level_crosscheck(U, R, orderings, t1, t2, norb)

    banner("CROSSCHECK DECISION (operator-level, primary)")
    metadata: dict[str, Any] = dict(
        subcommand="crosscheck", shots=shots,
        orderings=list(orderings.keys()),
        git_commit=git_commit_hash(),
        fcidump_sha256=sha256_of(fcidump_path),
        reference_pkl_sha256=sha256_of(U.CACHE),
        n_reps=None, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        operator_level_rows=op_rows,
    )

    all_shape_match = all(r["shape_match"] for r in op_rows)
    all_pair_counts_equal = all(
        r["n_pairs_aa_B"] + r["n_pairs_ab_B"] == op_rows[0]["n_pairs_aa_B"] + op_rows[0]["n_pairs_ab_B"]
        for r in op_rows
    )  # sanity: pair counts should be ordering-independent (same mask.py on every perm)

    if not all_shape_match:
        verdict = "SHAPE_MISMATCH"
        print("STOP: mechanism A and B produce different n_reps / operator shapes for at")
        print("least one ordering - not directly comparable entrywise.")
        for r in op_rows:
            if not r["shape_match"]:
                print(f"  {r['ordering']}: n_reps_A={r['n_reps_A']} n_reps_B={r['n_reps_B']}")
        exit_code = 1
    elif all_equivalent:
        verdict = "EQUIVALENT"
        print("EQUIVALENT: un-permuted mechanism-A operators match mechanism-B operators")
        print("entrywise (diag_coulomb_mats aa/ab, orbital_rotations) to < 1e-12, with no")
        print("sampling involved. H10 (mechanism B) now inherits the N2 (mechanism A)")
        print("validation.")
        exit_code = 0
    else:
        # distinguish "pairs differ" from "pairs match but values differ (constrained fit)"
        first_bad = next(r for r in op_rows if not r["equivalent"])
        if first_bad["diff_aa"] < 1e-12 and first_bad["diff_ab"] < 1e-12 and not all_pair_counts_equal:
            verdict = "MASK_MODELS_DISAGREE"
            print("STOP: the mask models genuinely disagree (retained-pair counts differ).")
            print(f"First failing ordering: {first_bad['ordering']!r}")
            print(json.dumps(pair_set_difference(R, orderings[first_bad["ordering"]], norb), indent=2))
        else:
            verdict = "NOT_EQUIVALENT_FIT"
            print("STOP: mechanisms are not equivalent; H10 results use an untested ansatz")
            print("construction. Un-permuted operators disagree entrywise even though pair")
            print("counts match - ffsim's interaction_pairs is doing a constrained FIT, not")
            print("a zeroing, when building the operator directly from a pair list.")
            print(f"First failing ordering: {first_bad['ordering']!r}  "
                  f"diff_aa={first_bad['diff_aa']:.3e}  diff_ab={first_bad['diff_ab']:.3e}  "
                  f"diff_U={first_bad['diff_U']:.3e}")
        exit_code = 1

    metadata["verdict"] = verdict

    sampling_rows = None
    if exit_code == 0:
        sampling_rows = run_sampling_confirmation(U, R, orderings, t1, t2, norb, nelec,
                                                  fcidump_path, shots)
        metadata["sampling_confirmation_rows"] = sampling_rows
        low_jaccard = [r for r in sampling_rows if r["jaccard_alpha"] < 0.9 or r["jaccard_beta"] < 0.9]
        banner("SAMPLING CONFIRMATION SUMMARY")
        for r in sampling_rows:
            print(f"{r['ordering']:<14} delta={r['delta_mHa']:>9.4f} mHa  "
                  f"jaccard_a={r['jaccard_alpha']:.4f}  jaccard_b={r['jaccard_beta']:.4f}")
        if low_jaccard:
            print(f"\n{len(low_jaccard)} ordering(s) with Jaccard < 0.9: "
                  f"{[r['ordering'] for r in low_jaccard]}")
        else:
            print("\nAll orderings: Jaccard >= 0.9 for both alpha and beta.")
    else:
        print("\nSkipping sampling confirmation (secondary): primary operator-level test")
        print("did not pass - no point spending sbd time confirming a foregone conclusion.")

    import pandas as pd
    pd.DataFrame(op_rows).to_csv(OUTDIR / "crosscheck_operator_level.csv", index=False)
    if sampling_rows is not None:
        pd.DataFrame(sampling_rows).to_csv(OUTDIR / "crosscheck_sampling_confirmation.csv", index=False)
    with open(OUTDIR / "crosscheck_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'crosscheck_operator_level.csv'}")
    if sampling_rows is not None:
        print(f"[out] {OUTDIR / 'crosscheck_sampling_confirmation.csv'}")
    print(f"[out] {OUTDIR / 'crosscheck_metadata.json'}")

    return exit_code


# ==========================================================================
# invariance: is the UNMASKED operator's SQD energy invariant under orbital
# relabelling? Mechanism A (unified_run.py) only - under mechanism B an
# unmasked permutation is a no-op (interaction_pairs=None ignores position
# entirely), which would make this test vacuous. This exercises the whole
# chain a mask-model check cannot: P J P^T, orbital rotations, bitstring ->
# determinant mapping, FCIDUMP orbital labels.
# ==========================================================================

def run_invariance(shots: int) -> int:
    banner("INVARIANCE -- unmasked operator SQD energy under orbital relabelling (mechanism A)")
    print("Loading mechanism A module (unified_run.py) ...")
    import unified_run as U

    if not U.SBD.exists():
        sys.exit(f"FATAL: sbd binary not found at expected path {U.SBD}")

    norb = U.NORB
    seed_sim = 2026
    print(f"N2 CAS({sum(U.NELEC)},{norb}) @ {U.BOND} A, {U.BASIS}  "
          f"shots={shots}  seed_simulator={seed_sim}")
    print(f"Shared FCIDUMP: {U.FCIDUMP}  sha256={sha256_of(U.FCIDUMP)[:16]}")

    orderings: dict[str, np.ndarray] = {"identity": np.arange(norb)}
    for s in (101, 102, 103):
        orderings[f"rand_seed{s}"] = np.random.default_rng(s).permutation(norb)

    # NOTE ON FRAMING: raw sampled bits are already in canonical orbital
    # labelling, mask or no mask - see run_sampling_confirmation's comment.
    # For this UNMASKED case there is additionally an exact algebraic
    # reason the sets end up bit-identical across orderings: J dense means
    # permute_operator's (P J P^T, U P^T) reparametrisation is a pure gauge
    # redundancy (P^T[(P J P^T)]P == J is the M=all-ones case of the
    # crosscheck identity), and empirically the ACTION on |HF> (not just
    # the un-permuted operator) is bit-identical - overlap 1.0 with
    # op_full|HF> for arbitrary perm, verified directly for random perms.
    rows = []
    canon_dets: dict[str, tuple[set, set]] = {}
    for name, perm in orderings.items():
        op_unmasked = U.permute_operator(U.op_full, perm)
        out = U.evaluate(name, perm, "invariance", seeds=[seed_sim], op_override=op_unmasked)[0]
        energy = out["err_sub_mHa"] / 1000.0 + U.E_CASCI
        a_det = (U.WORK / f"{name}_{seed_sim}_a.txt").read_text().split()
        b_det = (U.WORK / f"{name}_{seed_sim}_b.txt").read_text().split()
        canon_dets[name] = (set(a_det), set(b_det))
        rows.append(dict(ordering=name, perm="".join(map(str, perm)), energy=energy))
        print(f"{name:<14} perm={''.join(map(str, perm))}  energy={energy:.12f}")

    id_a, id_b = canon_dets["identity"]
    e_id = next(r["energy"] for r in rows if r["ordering"] == "identity")
    for r in rows:
        r["delta_Ha"] = r["energy"] - e_id
        a_c, b_c = canon_dets[r["ordering"]]
        r["jaccard_alpha"] = jaccard(a_c, id_a)
        r["jaccard_beta"] = jaccard(b_c, id_b)

    banner("INVARIANCE RESULTS")
    print(f"{'ordering':<14}{'energy':>18}{'delta_Ha':>14}{'jaccard_a':>11}{'jaccard_b':>11}")
    for r in rows:
        print(f"{r['ordering']:<14}{r['energy']:>18.12f}{r['delta_Ha']:>14.3e}"
              f"{r['jaccard_alpha']:>11.4f}{r['jaccard_beta']:>11.4f}")

    max_abs_delta = max(abs(r["delta_Ha"]) for r in rows)

    banner("INVARIANCE DECISION")
    metadata: dict[str, Any] = dict(
        subcommand="invariance", shots=shots, seed_simulator=seed_sim,
        orderings=list(orderings.keys()), git_commit=git_commit_hash(),
        fcidump_sha256=sha256_of(U.FCIDUMP), reference_pkl_sha256=sha256_of(U.CACHE),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), rows=rows,
        max_abs_delta_Ha=max_abs_delta,
    )

    if max_abs_delta < 1e-9:
        min_jaccard = min(min(r["jaccard_alpha"], r["jaccard_beta"]) for r in rows)
        assert min_jaccard == 1.0, (
            f"bit-identical energies (max|delta|={max_abs_delta:.3e}) but "
            f"min Jaccard={min_jaccard:.4f} < 1.0 - determinant sets disagree "
            f"despite identical energies, which is impossible for disjoint "
            f"225-dim subspaces. The Jaccard diagnostic is broken again."
        )
        verdict = "PASS"
        print(f"PASS: max|delta| = {max_abs_delta:.3e} Ha < 1e-9. Plumbing is correct.")
        print(f"Jaccard = 1.0000 for all orderings (asserted) - raw determinant sets "
              f"are identical, consistent with bit-identical energies.")
        print("This is the control: differences between orderings under the MASKED")
        print("operator (crosscheck / future sweeps) come from the mask, not from an")
        print("inconsistent permutation/relabelling chain.")
        exit_code = 0
    elif max_abs_delta > 1e-6:
        verdict = "FAIL"
        print(f"FAIL: max|delta| = {max_abs_delta:.3e} Ha > 1e-6.")
        print("STOP: permutation plumbing is inconsistent. All H10 results are suspect.")
        exit_code = 1
    else:
        verdict = "WARN"
        print(f"WARN: max|delta| = {max_abs_delta:.3e} Ha (between 1e-9 and 1e-6).")
        print("Jaccard overlaps vs identity (alpha/beta), after inverse-permutation mapping:")
        for r in rows:
            print(f"  {r['ordering']:<14} alpha={r['jaccard_alpha']:.4f}  beta={r['jaccard_beta']:.4f}")
        print("Determinant selection differs at the cut, so the test is inconclusive.")
        exit_code = 1

    metadata["verdict"] = verdict

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUTDIR / "invariance_results.csv", index=False)
    with open(OUTDIR / "invariance_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {OUTDIR / 'invariance_results.csv'}")
    print(f"[out] {OUTDIR / 'invariance_metadata.json'}")

    return exit_code


def run_shotscan() -> int:
    banner("SHOTSCAN -- H10 shot-count / seed variance, ordering s1_max (mechanism B)")
    import unified_run as U  # only for the shared sbd binary path
    import run_ordering_pipeline as R
    from pyscf.fci import cistring

    R.CFG["sbd_bin"] = str(U.SBD)
    if not Path(R.CFG["sbd_bin"]).exists():
        sys.exit(f"FATAL: sbd binary not found at expected path {R.CFG['sbd_bin']}")

    natoms, basis = 10, "sto-6g"
    R_TARGET = 1.8
    R_FALLBACK = 1.6
    print(f"Attempting H10 reference at R={R_TARGET} (cached only - build_or_load_h10_reference "
          f"will build it if not cached).")
    try:
        ref = R.build_or_load_h10_reference(R_TARGET, natoms, basis, cachedir=f"cache/h10_R{R_TARGET}")
        R_used = R_TARGET
    except (SystemExit, RuntimeError) as exc:
        print(f"R={R_TARGET}: {exc}")
        print(f"Falling back to R={R_FALLBACK} per protocol (CCSD does not converge at R={R_TARGET} - "
              f"see notes/PROGRESS.md / Step 4d report for the diverging E_corr trace).")
        ref = R.build_or_load_h10_reference(R_FALLBACK, natoms, basis, cachedir=f"cache/h10_R{R_FALLBACK}")
        R_used = R_FALLBACK
    print(f"Using R={R_used} for the shotscan.")

    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    fcidump_path = Path(ref["fcidump_path"])
    perm = R.parse_permutation(ref["orderings"]["s1_max"]["perm"], norb)
    pos = R.positions_from(perm)
    # match stage3()'s own call exactly (centroids, J_ab) rather than the
    # bare interaction_pairs_for(pos) crosscheck uses for N2's "anchor" mode
    Jaa_full, Jab_full = R.diag_coulomb(R.build_ucj(t2L, t1L))
    pairs = R.interaction_pairs_for(pos, ref["centroids"], J_ab=Jab_full)
    op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
    hf = R.hf_bitstring(norb, nocc)
    BUDGET = 15

    strs = cistring.make_strings(range(norb), nocc)
    b2i = {format(s, f"0{norb}b"): i for i, s in enumerate(strs)}
    W = np.asarray(ref["ci"]).reshape(len(strs), len(strs)) ** 2
    W /= W.sum()

    print(f"H10 CAS({2*nocc},{norb}) @ R={R_used} A, {basis}  ordering=s1_max  perm={''.join(map(str, perm))}")
    print(f"FCIDUMP: {fcidump_path}  sha256={sha256_of(fcidump_path)[:16]}")

    shot_counts = [500_000, 2_000_000, 8_000_000]
    seeds = [2026, 7, 13, 41, 97]
    csv_path = OUTDIR / "shotscan_results.csv"
    rows: list[dict[str, Any]] = []
    depths: set[int] = set()
    sel_by_shots: dict[int, list[tuple[set, set]]] = {s: [] for s in shot_counts}

    import pandas as pd
    t0 = time.time()
    n_total = len(shot_counts) * len(seeds)
    n_done = 0
    for shots in shot_counts:
        for seed in seeds:
            a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, shots, seed)
            depths.add(depth)
            a_sel, na = R.top_dets(a_c, BUDGET, hf)
            b_sel, nb = R.top_dets(b_c, BUDGET, hf)
            ia = [b2i[d] for d in a_sel]
            ib = [b2i[d] for d in b_sel]
            captured = float(W[np.ix_(ia, ib)].sum())

            adet_path = OUTDIR / f"_shotscan_{shots}_{seed}_a.txt"
            bdet_path = OUTDIR / f"_shotscan_{shots}_{seed}_b.txt"
            adet_path.write_text("\n".join(sorted(a_sel)) + "\n")
            bdet_path.write_text("\n".join(sorted(b_sel)) + "\n")
            energy = R.run_sbd(str(fcidump_path), str(adet_path), str(bdet_path), norb)
            err_mha = (energy - ref["E_CASCI"]) * 1000.0

            row = dict(shots=shots, seed=seed, energy=energy, err_mHa=err_mha,
                      n_unique_alpha=na, n_unique_beta=nb, depth=depth, captured=captured,
                      top15_alpha="|".join(sorted(a_sel)), top15_beta="|".join(sorted(b_sel)))
            rows.append(row)
            sel_by_shots[shots].append((set(a_sel), set(b_sel)))

            pd.DataFrame(rows).to_csv(csv_path, index=False)
            n_done += 1
            el = time.time() - t0
            print(f"[{n_done:2d}/{n_total}] shots={shots:>9,} seed={seed:<5} "
                  f"err={err_mha:8.3f} mHa  captured={captured:.4f}  "
                  f"n_uniq(a/b)={na}/{nb}  eta={el/n_done*(n_total-n_done)/60:.1f}m")

    assert len(depths) == 1, (
        f"circuit depth varied across runs ({sorted(depths)}) - only the sampling "
        f"seed should differ between evaluations for a fixed ordering."
    )
    circuit_depth = depths.pop()
    print(f"\nCircuit depth constant at {circuit_depth} across all {n_total} evaluations (asserted).")

    banner("SHOTSCAN SUMMARY")
    df = pd.DataFrame(rows)
    summary = {}
    for shots in shot_counts:
        sub = df[df.shots == shots]
        mean_err, sd_err = sub.err_mHa.mean(), sub.err_mHa.std()
        mean_cap = sub.captured.mean()
        sets = sel_by_shots[shots]
        pw_a = [jaccard(sets[i][0], sets[j][0]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
        pw_b = [jaccard(sets[i][1], sets[j][1]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
        summary[shots] = dict(mean_err_mHa=mean_err, sd_err_mHa=sd_err, mean_captured=mean_cap,
                              mean_jaccard_alpha=float(np.mean(pw_a)), mean_jaccard_beta=float(np.mean(pw_b)))
        print(f"shots={shots:>9,}  mean_err={mean_err:8.3f} mHa  sd={sd_err:7.3f} mHa  "
              f"mean_captured={mean_cap:.4f}  mean_pairwise_jaccard(a/b)="
              f"{summary[shots]['mean_jaccard_alpha']:.3f}/{summary[shots]['mean_jaccard_beta']:.3f}")

    sd_ratio = summary[8_000_000]["sd_err_mHa"] / summary[500_000]["sd_err_mHa"]
    print(f"\nsd(8e6)/sd(5e5) = {sd_ratio:.4f}")

    banner("SHOTSCAN DECISION")
    if sd_ratio < 0.45:
        verdict = "SHOT_NOISE_DOMINATED"
        print(f"SHOT-NOISE DOMINATED: sd ratio {sd_ratio:.4f} < 0.45.")
        print("Increasing shots materially reduces variance - selection converges to a")
        print("stable top-15 set. Recommendation: use the highest affordable shot count;")
        print("no need to average over multiple seeds at fixed shots for production runs.")
    elif sd_ratio > 0.70:
        verdict = "INTRINSIC_NEAR_DEGENERACY"
        print(f"INTRINSIC NEAR-DEGENERACY: sd ratio {sd_ratio:.4f} > 0.70.")
        print("Variance does not shrink materially with more shots - the spread reflects")
        print("near-degenerate marginal weights at the budget boundary, not sampling noise.")
        print("Recommendation: more shots will not stabilize the top-15 selection; either")
        print("increase BUDGET past the degenerate cluster or report energy as a")
        print("seed-averaged distribution rather than a single number.")
    else:
        verdict = "PARTIAL"
        print(f"PARTIAL: sd ratio {sd_ratio:.4f} in [0.45, 0.70] - mixed shot-noise and")
        print("intrinsic-degeneracy contributions. Recommendation: increase shots AND")
        print("consider whether BUDGET should be widened; re-run this scan after either")
        print("change to see which one moves the sd ratio.")

    print(f"\nVerdict: {verdict}")

    banner("ANALYTIC CROSS-CHECK (Step 4d prediction vs empirical trend)")
    oa = np.argsort(W.sum(1))[::-1]
    ob = np.argsort(W.sum(0))[::-1]
    wa16_15 = W.sum(1)[oa[15]] / W.sum(1)[oa[14]]
    wb16_15 = W.sum(0)[ob[15]] / W.sum(0)[ob[14]]
    w_pred = max(wa16_15, wb16_15)
    print(f"H10 R={R_used} analytic w16/w15: alpha={wa16_15:.4f}  beta={wb16_15:.4f}")
    pred_verdict = ("INTRINSIC_NEAR_DEGENERACY" if w_pred > 0.70 else
                    "SHOT_NOISE_DOMINATED" if w_pred < 0.45 else "PARTIAL")
    print(f"Analytic prediction: {pred_verdict}   Empirical (sd-ratio) verdict: {verdict}")
    if pred_verdict != verdict:
        print("MISMATCH: the analytic marginal-weight prediction and the empirical "
              "sd-ratio trend disagree. Something in the sampling or selection path "
              "may be wrong - do not trust either verdict without investigating further.")
    else:
        print("Analytic prediction and empirical trend agree.")

    mismatch = pred_verdict != verdict
    metadata = dict(
        subcommand="shotscan", R_used=R_used, R_target=R_TARGET, R_fallback_used=(R_used != R_TARGET),
        ordering="s1_max", perm="".join(map(str, perm)), norb=norb, nelec=list(nelec),
        shot_counts=shot_counts, seeds=seeds, git_commit=git_commit_hash(),
        fcidump_sha256=sha256_of(fcidump_path), circuit_depth=circuit_depth,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"), summary=summary, sd_ratio_8e6_over_5e5=sd_ratio,
        verdict=verdict, analytic_w16_w15_alpha=float(wa16_15), analytic_w16_w15_beta=float(wb16_15),
        analytic_prediction=pred_verdict, analytic_empirical_mismatch=mismatch,
    )
    with open(OUTDIR / "shotscan_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"\n[out] {csv_path}")
    print(f"[out] {OUTDIR / 'shotscan_metadata.json'}")

    return 1 if mismatch else 0




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subcommand", choices=["crosscheck", "invariance", "shotscan"])
    ap.add_argument("--shots", type=int, default=1_000_000)
    args = ap.parse_args()

    if args.subcommand == "crosscheck":
        sys.exit(run_crosscheck(args.shots))
    elif args.subcommand == "invariance":
        sys.exit(run_invariance(args.shots))
    elif args.subcommand == "shotscan":
        sys.exit(run_shotscan())


if __name__ == "__main__":
    main()
