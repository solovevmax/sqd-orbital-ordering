#!/usr/bin/env python3
"""
Anchor-phase diagnostic for the existing H10 R=1.6 pipeline.

This file intentionally reuses the validated pipeline implementation rather
than duplicating PySCF/ffsim/SBD logic. It evaluates physical and reversed
chain orderings for anchor offsets 0,1,2,3.
"""

from pathlib import Path
import csv
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np

import run_ordering_pipeline as p


OUT = Path("outputs/h10_anchor_phase")
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "results.csv"

SHOTS = int(os.environ.get("ANCHOR_SHOTS", p.CFG["shots"]))
SEEDS = tuple(p.CFG["seeds"])
N_DETS = p.CFG["n_dets"]


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def anchor_sites(pos, offset):
    return [
        int(orb)
        for orb in range(len(pos))
        if (int(pos[orb]) - offset) % p.CFG["anchor_mod"] == 0
    ]


def interaction_pairs(pos, offset):
    aa = sorted({
        tuple(sorted(pair))
        for pair in p.same_spin_pairs(pos)
    })
    ab = sorted((orb, orb) for orb in anchor_sites(pos, offset))
    return aa, ab


def main():
    print("ANCHOR-PHASE DIAGNOSTIC")
    print(f"H10 R={p.CFG['h10_R']} Å | shots={SHOTS} | seeds={SEEDS}")

    import ffsim
    from pyscf import mcscf
    from pyscf.tools import fcidump as fcidump_mod

    mol, mf, cc, norb, nocc = p.build_h10(
        p.CFG["h10_R"],
        p.CFG["h10_natoms"],
        p.CFG["h10_basis"],
    )
    nelec = (nocc, nocc)
    active = list(range(norb))

    C_loc, Uo, Uv = p.block_boys(
        mol, mf, active, nocc, tag="anchor-"
    )
    t1_loc, t2_loc = p.rotate_amplitudes(
        np.asarray(cc.t1),
        np.asarray(cc.t2),
        Uo,
        Uv,
    )

    mf_loc = __import__("copy").copy(mf)
    mf_loc.mo_coeff = C_loc

    md_loc = ffsim.MolecularData.from_scf(
        mf_loc,
        active_space=active,
    )

    cas = mcscf.CASCI(mf, norb, mol.nelectron)
    cas.verbose = 0
    e_exact = float(cas.kernel(C_loc)[0])

    fcidump = OUT / "fcidump.txt"
    fcidump_mod.from_integrals(
        str(fcidump),
        np.asarray(md_loc.hamiltonian.one_body_tensor),
        np.asarray(md_loc.hamiltonian.two_body_tensor),
        norb,
        mol.nelectron,
        nuc=float(md_loc.core_energy),
        ms=0,
    )

    centroids = p.orbital_centroids(mol, C_loc, active)
    physical = p.physical_ordering(centroids, nocc)
    orderings = {
        "physical": physical,
        "physical_reverse": physical[::-1].copy(),
    }

    Jaa, Jab = p.diag_coulomb(
        p.build_ucj(t2_loc, t1_loc)
    )
    amp = p.Amplitudes(t1_loc, t2_loc, nocc, norb)
    w_ss = float(
        np.abs(Jaa).sum()
        / (np.abs(Jaa).sum() + np.abs(Jab).sum())
    )
    hf = p.hf_bitstring(norb, nocc)

    rows = []
    if CSV.exists():
        CSV.unlink()

    for offset in range(p.CFG["anchor_mod"]):
        for name, perm in orderings.items():
            pos = p.positions_from(perm)

            pairs = interaction_pairs(pos, offset)
            op = p.build_ucj(
                t2_loc,
                t1_loc,
                interaction_pairs=pairs,
            )

            retained = p.retained_J_of(pos, Jaa, Jab)

            for seed in SEEDS:
                alpha, beta, depth = p.sample_bitstrings(
                    op,
                    norb,
                    nelec,
                    SHOTS,
                    seed,
                )
                alpha_sel, alpha_unique = p.top_dets(
                    alpha, N_DETS, hf
                )
                beta_sel, beta_unique = p.top_dets(
                    beta, N_DETS, hf
                )

                row = {
                    "ordering": name,
                    "anchor_offset": offset,
                    "permutation": p.perm_to_str(perm),
                    "seed": seed,
                    "depth": depth,
                    "n_unique_alpha": alpha_unique,
                    "n_unique_beta": beta_unique,
                    "dim_alpha": len(alpha_sel),
                    "dim_beta": len(beta_sel),
                    "dim": len(alpha_sel) * len(beta_sel),
                    "retained_J": retained,
                    "status": "OK",
                    "energy": np.nan,
                    "err_mHa": np.nan,
                }

                if (
                    len(alpha_sel) < N_DETS
                    or len(beta_sel) < N_DETS
                ):
                    row["status"] = "SUPPORT_COLLAPSE"
                    rows.append(row)
                    pd = __import__("pandas")
                    pd.DataFrame(rows).to_csv(CSV, index=False)
                    continue

                afile = OUT / f"_a_{name}_{offset}_{seed}.txt"
                bfile = OUT / f"_b_{name}_{offset}_{seed}.txt"

                afile.write_text("\n".join(alpha_sel) + "\n")
                bfile.write_text("\n".join(beta_sel) + "\n")

                energy = p.run_sbd(
                    str(fcidump),
                    str(afile),
                    str(bfile),
                    norb,
                )

                row["energy"] = energy
                row["err_mHa"] = (
                    energy - e_exact
                ) * p.HARTREE_TO_MHA
                row["adet_sha"] = sha_file(afile)
                row["bdet_sha"] = sha_file(bfile)

                afile.unlink(missing_ok=True)
                bfile.unlink(missing_ok=True)

                rows.append(row)
                __import__("pandas").DataFrame(rows).to_csv(
                    CSV, index=False
                )

                print(
                    f"{name:18s} offset={offset} seed={seed} "
                    f"err={row['err_mHa']:.2f} mHa",
                    flush=True,
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    ok = df[
        (df["status"] == "OK")
        & (df["dim"] == N_DETS ** 2)
    ]

    summary = (
        ok.groupby(["ordering", "anchor_offset"], as_index=False)
        .agg(
            mean_err_mHa=("err_mHa", "mean"),
            sd_err_mHa=("err_mHa", "std"),
            mean_retained_J=("retained_J", "mean"),
        )
    )

    summary.to_csv(OUT / "summary.csv", index=False)

    print("\nSUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print(f"\nResults: {CSV}")
    print(f"Summary: {OUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
