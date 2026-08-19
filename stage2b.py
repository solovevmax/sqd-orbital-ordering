"""Stage 2B: convert Stage 2's sampled bitstrings into sbd-format input.

Entry point: run_stage2b(config). Reads the Stage 2 bitstring text file and
writes the alpha/beta determinant files expected by the downstream C++ app
(https://github.com/r-ccs-cms/sbd,
apps/chemistry_tpb_selected_basis_diagonalization). Does not implement any
diagonalisation itself - sbd does that.

=== WHAT SBD EXPECTS (from main.cc and the repository README) ===
  --fcidump <file>   standard FCIDUMP (Stage 1's output is compatible as-is)
  --adetfile <file>  alpha determinants: ONE BITSTRING PER LINE, NO COUNTS,
                     NO HEADER
  --bdetfile <file>  beta determinants, same format; if omitted, beta=alpha
Bitstring convention: rightmost bit = orbital 1 (matches FCIDUMP labelling).
The Hilbert space sbd solves in is the TENSOR PRODUCT of the alpha and beta
determinant sets (confirmed in the sbd README - not paired combinations).
sbd sorts the input determinants and treats the first one after sorting as
the Hartree-Fock initial state for Davidson, so the HF determinant MUST be
present in both files.

=== BIT ORDERING: CONFIRMED, NOT ASSUMED ===
Qiskit writes qubit 0 as the rightmost character of the measurement string;
ffsim maps alpha spin-orbitals to qubits 0..norb-1 and beta to
qubits norb..2*norb-1. This was checked against the sbd repository's example
file AlphaDets.txt: its first (HF, since sorted ascending) line is
36 alpha orbitals wide with 27 trailing ones under the "rightmost bit =
orbital 1" convention, matching Qiskit's little-endian output directly with
NO reversal. 'split' (below) is therefore the CONFIRMED default transform.
The alternative transforms remain available via CONFIG['sbd_bit_transform']
as a fallback should a different sbd build/version disagree.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import stage2

BIT_TRANSFORM_MODES: tuple[str, ...] = ("split", "split_reverse", "split_swap")


# --------------------------------------------------------------------------
# Bit-order transform
# --------------------------------------------------------------------------


def sbd_transform(full_bitstring: str, norb: int, mode: str) -> tuple[str, str]:
    """Split one 2*norb Qiskit bitstring into (alpha, beta) sbd determinant strings.

    This is a SPLIT, not a reversal: within each norb-wide half, the
    rightmost character is already orbital 1 in sbd's convention.

    Modes:
        'split' (CONFIRMED default - verified against the sbd repo's
            AlphaDets.txt example): alpha = full[-norb:], beta = full[:norb].
        'split_reverse': as 'split', then each half is reversed end-to-end.
        'split_swap': as 'split', but the two halves are swapped (fallback
            in case alpha/beta were mapped to qubits the other way around).
    """
    if len(full_bitstring) != 2 * norb:
        raise ValueError(f"bitstring length {len(full_bitstring)} != 2*norb={2 * norb}")

    alpha = full_bitstring[-norb:]
    beta = full_bitstring[:norb]

    if mode == "split":
        return alpha, beta
    elif mode == "split_reverse":
        return alpha[::-1], beta[::-1]
    elif mode == "split_swap":
        return beta, alpha
    else:
        raise ValueError(
            f"Unknown sbd_bit_transform {mode!r}; expected one of {BIT_TRANSFORM_MODES}"
        )


def hf_determinant_string(norb: int, n_electrons: int) -> str:
    """Hartree-Fock determinant in sbd convention: n_electrons ones on the right, zeros on the left."""
    if not (0 <= n_electrons <= norb):
        raise ValueError(f"n_electrons={n_electrons} out of range for norb={norb}")
    return "0" * (norb - n_electrons) + "1" * n_electrons


# --------------------------------------------------------------------------
# Parsing Stage 2 output
# --------------------------------------------------------------------------


def _parse_bitstring_file(path: Path, separator: str) -> dict[str, int]:
    """Parse a Stage 2 '<bitstring><sep><count>' file, skipping '#' comment lines."""
    counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bitstring, count_str = line.rsplit(separator, 1)
            counts[bitstring] = int(count_str)
    return counts


def _aggregate_determinants(
    full_counts: dict[str, int], norb: int, mode: str
) -> tuple[dict[str, int], dict[str, int]]:
    """Split every sampled full bitstring into (alpha, beta) and sum counts per determinant."""
    alpha_weights: dict[str, int] = {}
    beta_weights: dict[str, int] = {}
    for full_bitstring, count in full_counts.items():
        alpha, beta = sbd_transform(full_bitstring, norb, mode)
        alpha_weights[alpha] = alpha_weights.get(alpha, 0) + count
        beta_weights[beta] = beta_weights.get(beta, 0) + count
    return alpha_weights, beta_weights


# --------------------------------------------------------------------------
# Determinant selection + HF presence
# --------------------------------------------------------------------------


def _select_top_n(
    det_weights: dict[str, int], max_determinants: int | None
) -> tuple[list[str], dict[str, Any]]:
    """Keep the max_determinants most frequent determinants by summed sampled weight."""
    total_weight = sum(det_weights.values())
    ranked = sorted(det_weights.items(), key=lambda kv: kv[1], reverse=True)
    kept = ranked if max_determinants is None or max_determinants >= len(ranked) else ranked[:max_determinants]
    kept_weight = sum(c for _, c in kept)
    stats = {
        "n_available": len(ranked),
        "n_kept": len(kept),
        "kept_weight_fraction": (kept_weight / total_weight) if total_weight else 0.0,
    }
    return [d for d, _ in kept], stats


def _ensure_hf_present(determinants: list[str], hf_string: str, label: str) -> list[str]:
    """Guarantee the HF determinant is present; sbd uses it as the Davidson initial state.

    Must run AFTER max_determinants truncation, since truncation could
    otherwise drop it.
    """
    if hf_string in determinants:
        return determinants
    print("!" * 70)
    print(f"WARNING: HF determinant {hf_string!r} is MISSING from the {label} determinant list.")
    print(
        "sbd sorts determinants and uses the first one (after sorting) as the "
        "Hartree-Fock initial state for Davidson; prepending it now."
    )
    print("!" * 70)
    return [hf_string] + determinants


def write_determinant_file(determinants: list[str], output_path: Path) -> None:
    """Write determinants sorted ascending as fixed-width binary strings, one per line.

    No header, no comments, no counts - sbd's --adetfile/--bdetfile format.
    Ascending sort places the all-zeros-then-ones HF determinant first,
    matching the repository's example file; sbd re-sorts internally anyway,
    but sorting here keeps the file deterministic and readable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(sorted(determinants)) + "\n")
    print(f"Determinant file written to {output_path} ({len(determinants)} determinants)")


# --------------------------------------------------------------------------
# Product-space diagnostics
# --------------------------------------------------------------------------


def compute_product_space_diagnostics(
    n_unique_full: int, n_alpha_dets: int, n_beta_dets: int, norb: int, n_alpha: int, n_beta: int
) -> dict[str, Any]:
    """Report the sbd tensor-product Hilbert-space size implied by the exported determinants.

    sbd forms its Hilbert space as the TENSOR PRODUCT of the alpha and beta
    determinant sets (confirmed in the sbd README), not paired combinations,
    so the product-space dimension usually far exceeds the number of unique
    full bitstrings actually sampled.
    """
    full_ci_dim = math.comb(norb, n_alpha) * math.comb(norb, n_beta)
    product_dim = n_alpha_dets * n_beta_dets
    ratio = (product_dim / n_unique_full) if n_unique_full else float("inf")
    pct_of_full_ci = (100 * product_dim / full_ci_dim) if full_ci_dim else float("nan")

    print("Product-space diagnostics (tensor product of alpha x beta sectors):")
    print(f"  unique full bitstrings sampled  : {n_unique_full}")
    print(f"  unique alpha determinants       : {n_alpha_dets}")
    print(f"  unique beta determinants        : {n_beta_dets}")
    print(f"  implied product-space dimension : {product_dim} (= {n_alpha_dets} x {n_beta_dets})")
    print(f"  product_dim / unique_full_bitstrings ratio: {ratio:.2f}x")
    print(f"  full CI space dimension (C(norb,na)*C(norb,nb)): {full_ci_dim}")
    print(f"  product space as % of full CI space: {pct_of_full_ci:.4f}%")

    return {
        "n_unique_full_bitstrings": n_unique_full,
        "n_alpha_determinants": n_alpha_dets,
        "n_beta_determinants": n_beta_dets,
        "product_space_dimension": product_dim,
        "product_dim_over_unique_full_ratio": ratio,
        "full_ci_space_dimension": full_ci_dim,
        "product_space_pct_of_full_ci": pct_of_full_ci,
    }


# --------------------------------------------------------------------------
# Naming helper
# --------------------------------------------------------------------------


def _sbd_base_name(bitstring_filename: str) -> str:
    """Derive the sbd export base name from Stage 2's bitstring filename."""
    stem = Path(bitstring_filename).stem
    if stem.endswith("_bitstrings"):
        stem = stem[: -len("_bitstrings")]
    return stem


# --------------------------------------------------------------------------
# Per-transform export
# --------------------------------------------------------------------------


def _run_one_transform(
    full_counts: dict[str, int],
    norb: int,
    nelec: tuple[int, int],
    mode: str,
    config: dict[str, Any],
    base_name: str,
    is_only_variant: bool,
) -> dict[str, Any]:
    """Aggregate, truncate, HF-check, and write the adet/bdet files for one transform mode."""
    n_alpha, n_beta = nelec
    print(f"--- sbd export: sbd_bit_transform={mode!r} ---")
    alpha_weights, beta_weights = _aggregate_determinants(full_counts, norb, mode)

    max_determinants = config.get("max_determinants")
    alpha_kept, alpha_stats = _select_top_n(alpha_weights, max_determinants)
    beta_kept, beta_stats = _select_top_n(beta_weights, max_determinants)
    print(
        f"  alpha determinants kept: {alpha_stats['n_kept']}/{alpha_stats['n_available']} "
        f"({100 * alpha_stats['kept_weight_fraction']:.4f}% of sampled weight)"
    )
    print(
        f"  beta determinants kept:  {beta_stats['n_kept']}/{beta_stats['n_available']} "
        f"({100 * beta_stats['kept_weight_fraction']:.4f}% of sampled weight)"
    )

    hf_alpha = hf_determinant_string(norb, n_alpha)
    hf_beta = hf_determinant_string(norb, n_beta)
    alpha_kept = _ensure_hf_present(alpha_kept, hf_alpha, "alpha")

    write_bdetfile = config.get("write_bdetfile", True)
    if write_bdetfile:
        beta_kept = _ensure_hf_present(beta_kept, hf_beta, "beta")

    suffix = "" if is_only_variant else f"_{mode}"
    output_dir = Path(config["output_dir"])
    adet_path = output_dir / f"{base_name}{suffix}_adets.txt"
    write_determinant_file(alpha_kept, adet_path)

    bdet_path: Path | None = None
    if write_bdetfile:
        bdet_path = output_dir / f"{base_name}{suffix}_bdets.txt"
        write_determinant_file(beta_kept, bdet_path)
    else:
        print("  write_bdetfile=False: beta file skipped (sbd defaults to beta=alpha).")

    diagnostics = compute_product_space_diagnostics(
        n_unique_full=len(full_counts),
        n_alpha_dets=len(alpha_kept),
        n_beta_dets=len(beta_kept) if write_bdetfile else len(alpha_kept),
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
    )

    return {
        "mode": mode,
        "adet_path": str(adet_path),
        "bdet_path": str(bdet_path) if bdet_path else None,
        "alpha_selection": alpha_stats,
        "beta_selection": beta_stats,
        "diagnostics": diagnostics,
    }


# --------------------------------------------------------------------------
# HF-only calibration file
# --------------------------------------------------------------------------


def write_hf_only_determinant(config: dict[str, Any]) -> dict[str, Any]:
    """Write a determinant file containing ONLY the Hartree-Fock determinant.

    Calibration test: running sbd against this single-determinant file must
    return exactly the Hartree-Fock energy printed here. If it returns
    anything else, the bit-ordering convention is wrong - try a different
    CONFIG['sbd_bit_transform'].
    """
    qpy_path = Path(config["qpy_path"])
    metadata = stage2.load_sibling_metadata(qpy_path)
    norb = metadata["norb"]
    n_alpha, n_beta = metadata["nelec"]

    hf_alpha = hf_determinant_string(norb, n_alpha)
    hf_beta = hf_determinant_string(norb, n_beta)

    output_dir = Path(config["output_dir"])
    base_name = _sbd_base_name(config["bitstring_filename"])

    adet_path = output_dir / f"{base_name}_hf_only_adets.txt"
    write_determinant_file([hf_alpha], adet_path)

    bdet_path: Path | None = None
    if config.get("write_bdetfile", True):
        bdet_path = output_dir / f"{base_name}_hf_only_bdets.txt"
        write_determinant_file([hf_beta], bdet_path)

    hf_energy = metadata.get("hf_energy")
    casci_energy = metadata.get("casci_reference_energy")

    print()
    print("HF-only calibration file written.")
    print(
        "Running sbd with this single-determinant file must return exactly the "
        "Hartree-Fock energy (printed below). If it returns anything else, the "
        "bit-ordering convention is wrong -- try a different sbd_bit_transform."
    )
    print(f"  Expected HF energy (sbd should reproduce this)      = {hf_energy}")
    print(f"  For reference, FCI/CASCI energy (sbd should NOT return this) = {casci_energy}")
    print(f"  adet file: {adet_path}")
    if bdet_path:
        print(f"  bdet file: {bdet_path}")

    return {
        "adet_path": str(adet_path),
        "bdet_path": str(bdet_path) if bdet_path else None,
        "expected_hf_energy": hf_energy,
        "expected_casci_energy": casci_energy,
    }


# --------------------------------------------------------------------------
# Run command generator
# --------------------------------------------------------------------------


def print_sbd_command(config: dict[str, Any], sbd_export: dict[str, Any] | None = None) -> None:
    """Print a ready-to-paste mpirun command line for the sbd C++ app.

    method: 0 = Davidson matrix-free, 1 = Davidson stored,
            2 = Lanczos matrix-free, 3 = Lanczos stored.
    --bit_length is only included when norb > 20 (the app's default is 20).
    See apps/chemistry_tpb_selected_basis_diagonalization/run.sh in the sbd
    repository for the group's own usage examples.
    """
    qpy_path = Path(config["qpy_path"])
    metadata = stage2.load_sibling_metadata(qpy_path)
    norb = metadata["norb"]

    fcidump_path = (Path(config["output_dir"]) / config["fcidump_filename"]).resolve()

    default_mode = config.get("sbd_bit_transform", "split")
    if sbd_export is not None:
        variant = sbd_export["variants"][default_mode]
        adet_path = Path(variant["adet_path"]).resolve()
        bdet_path = Path(variant["bdet_path"]).resolve() if variant["bdet_path"] else None
    else:
        base_name = _sbd_base_name(config["bitstring_filename"])
        adet_path = (Path(config["output_dir"]) / f"{base_name}_adets.txt").resolve()
        bdet_path = (
            (Path(config["output_dir"]) / f"{base_name}_bdets.txt").resolve()
            if config.get("write_bdetfile", True)
            else None
        )

    lines = [
        "mpirun -np 4 ./sbd_chemistry_tpb \\",
        f"    --fcidump {fcidump_path} \\",
        f"    --adetfile {adet_path} \\",
    ]
    if bdet_path is not None:
        lines.append(f"    --bdetfile {bdet_path} \\")
    tail = "    --method 0 --iteration 10 --block 50 --tolerance 1e-8"
    if norb > 20:
        lines.append(tail + " \\")
        lines.append(f"    --bit_length {norb}")
    else:
        lines.append(tail)

    print()
    print("Ready-to-paste sbd run command (edit -np / method / iteration / block / tolerance as needed):")
    print()
    print("\n".join(lines))
    print()
    print(
        "method values: 0=Davidson matrix-free, 1=Davidson stored, "
        "2=Lanczos matrix-free, 3=Lanczos stored"
    )
    print(
        f"--bit_length omitted: norb={norb} <= 20 (the app default); only needed above that."
        if norb <= 20
        else f"--bit_length {norb} included: norb exceeds the app default of 20."
    )
    print(
        "See apps/chemistry_tpb_selected_basis_diagonalization/run.sh in the "
        "sbd repository for the group's own usage examples."
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_stage2b(config: dict[str, Any]) -> dict[str, Any]:
    """Run the Stage 2B sbd export: Stage 2 bitstrings -> sbd adet/bdet files.

    Reads the Stage 2 bitstring text file at
    <output_dir>/<bitstring_filename>, which must have been written with
    bit_order_mode='qiskit' (the CONFIG default) and include_counts=True,
    since the alpha/beta split (sbd_transform) is defined on that raw
    representation. Writes one adet/bdet file pair per
    CONFIG['sbd_bit_transform'] variant requested (all four if
    CONFIG['emit_all_sbd_variants'] is True), plus a JSON diagnostics summary.
    """
    print("=== Stage 2B: sbd export ===")

    if config.get("bit_order_mode", "qiskit") != "qiskit":
        raise ValueError(
            "run_stage2b requires the Stage 2 bitstring file to have been "
            "written with bit_order_mode='qiskit' (the CONFIG default), "
            "since the alpha/beta split is defined on that raw "
            "representation. Set CONFIG['bit_order_mode']='qiskit' before "
            "running Stage 2, or point CONFIG['bitstring_filename'] at a "
            "'_qiskit' variant written via emit_all_bit_order_variants."
        )
    if not config.get("include_counts", True):
        raise ValueError("run_stage2b requires CONFIG['include_counts']=True on the Stage 2 output.")

    bitstring_path = Path(config["output_dir"]) / config["bitstring_filename"]
    separator = config.get("output_separator", " ")
    full_counts = _parse_bitstring_file(bitstring_path, separator)
    print(f"Loaded {len(full_counts)} unique full bitstrings from {bitstring_path}")

    qpy_path = Path(config["qpy_path"])
    metadata = stage2.load_sibling_metadata(qpy_path)
    norb = metadata["norb"]
    nelec = tuple(metadata["nelec"])

    base_name = _sbd_base_name(config["bitstring_filename"])
    default_mode = config.get("sbd_bit_transform", "split")
    modes = BIT_TRANSFORM_MODES if config.get("emit_all_sbd_variants", False) else (default_mode,)

    variant_results: dict[str, Any] = {}
    for mode in modes:
        variant_results[mode] = _run_one_transform(
            full_counts, norb, nelec, mode, config, base_name, is_only_variant=(len(modes) == 1)
        )

    export_summary = {
        "norb": norb,
        "nelec": list(nelec),
        "source_bitstring_file": str(bitstring_path),
        "default_transform": default_mode,
        "max_determinants": config.get("max_determinants"),
        "write_bdetfile": config.get("write_bdetfile", True),
        "variants": variant_results,
    }

    summary_path = Path(config["output_dir"]) / f"{base_name}_sbd_export.json"
    with open(summary_path, "w") as f:
        json.dump(export_summary, f, indent=2)
    print(f"sbd export diagnostics written to {summary_path}")

    print("=== Stage 2B complete ===")
    return export_summary
