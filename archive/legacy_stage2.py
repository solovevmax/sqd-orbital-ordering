"""Stage 2: load a Stage 1 .qpy circuit, sample it, write bitstrings + counts.

Entry point: run_stage2(config). Consumes the .qpy file (and its sibling
.json metadata) produced by stage1.py. Does not compute any sampling-quality
metrics (Gini, compactness, subspace diagonalisation) - that work is
deferred to a separate stage.

=== BIT ORDERING (READ BEFORE TOUCHING THE DOWNSTREAM FORMAT) ===
Qiskit is little-endian: qubit 0 is the RIGHTMOST character of the
measurement string produced by AerSimulator / Circuit.measure_all(). ffsim's
UCJOpSpinBalancedJW + PrepareHartreeFockJW map alpha spin-orbitals to qubits
0..norb-1 and beta spin-orbitals to qubits norb..2*norb-1. Consequently, for
a bitstring `b` of length 2*norb:
    alpha block = b[-norb:]   (rightmost norb characters)
    beta block  = b[:-norb]   (leftmost norb characters)
This is the 'qiskit' convention below and is what this module produces by
default.

The four conventions exposed by convert_bit_order:
    'qiskit'              - unchanged, as produced by Aer (DEFAULT).
    'reversed'             - the whole string reversed end-to-end.
    'alpha_beta_swapped'   - alpha and beta blocks swapped as whole chunks.
    'interleaved'          - bits regrouped orbital-major: a0 b0 a1 b1 ...

*** The downstream C++ code (r-ccs-cms/sbd, input parsing in
*** sample/selected_basis_diagonalization - NOT the header-only algorithm
*** library in include/sbd) has NOT yet had its expected bit-ordering
*** convention confirmed against a known-good example file from the RIKEN
*** team. DO NOT assume 'qiskit' (or any other mode) is correct until that
*** verification has happened. Use
*** CONFIG["emit_all_bit_order_variants"]=True to write all four variants at
*** once so they can be tried against the C++ parser without resampling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import qiskit.qpy
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

BIT_ORDER_MODES: tuple[str, ...] = ("qiskit", "reversed", "alpha_beta_swapped", "interleaved")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_circuit(qpy_path: Path) -> QuantumCircuit:
    """Load a transpiled circuit from .qpy."""
    print(f"Loading circuit from {qpy_path} ...")
    with open(qpy_path, "rb") as f:
        (circuit,) = qiskit.qpy.load(f)
    print(f"  qubits = {circuit.num_qubits}, depth = {circuit.depth()}")
    return circuit


def load_sibling_metadata(qpy_path: Path) -> dict[str, Any]:
    """Load the Stage 1 JSON metadata sitting alongside the .qpy file."""
    metadata_path = Path(qpy_path).with_suffix(".json")
    print(f"Loading metadata from {metadata_path} ...")
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"  norb = {metadata['norb']}, nelec = {metadata['nelec']}")
    return metadata


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def sample_circuit(circuit: QuantumCircuit, shots: int, seed: int) -> dict[str, int]:
    """Sample the circuit with AerSimulator and return {bitstring: count}.

    Prints wall-clock sampling time, since shot counts in the 10^5-10^7
    range (as used in the SQD literature) make the sampling cost visible.
    """
    print(f"Sampling with AerSimulator: shots={shots}, seed={seed} ...")
    simulator = AerSimulator(seed_simulator=seed)
    start = time.perf_counter()
    job = simulator.run(circuit, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    elapsed = time.perf_counter() - start
    print(f"  Got {len(counts)} unique bitstrings from {shots} shots in {elapsed:.2f}s.")
    return dict(counts)


# --------------------------------------------------------------------------
# Bit order helper
# --------------------------------------------------------------------------


def convert_bit_order(bitstring: str, norb: int, mode: str = "qiskit") -> str:
    """Convert a sampled bitstring between bit-ordering conventions.

    Args:
        bitstring: measurement string as produced by Qiskit (little-endian,
            length 2*norb, alpha block = rightmost norb chars, beta block =
            leftmost norb chars).
        norb: number of spatial orbitals.
        mode: one of:
            'qiskit'              - unchanged, as produced (DEFAULT). This is
                                     the convention actually emitted by this
                                     pipeline; it has NOT been verified
                                     against the downstream C++ code's
                                     expected convention. Verify against a
                                     known-good example file before assuming
                                     any other mode is needed.
            'reversed'             - the whole string reversed end-to-end.
            'alpha_beta_swapped'   - the alpha and beta blocks swapped as
                                     whole chunks (each block's internal
                                     little-endian order preserved).
            'interleaved'          - bits regrouped orbital-major: for
                                     orbital p = 0..norb-1, emit (alpha_p,
                                     beta_p), i.e. "a0 b0 a1 b1 ... a{n-1}
                                     b{n-1}". This is one reasonable
                                     convention among several and is not
                                     guaranteed to match any particular
                                     downstream expectation.

    Returns:
        The bitstring re-expressed in the requested convention.
    """
    if len(bitstring) != 2 * norb:
        raise ValueError(
            f"bitstring length {len(bitstring)} does not match 2*norb = {2 * norb}"
        )

    if mode == "qiskit":
        return bitstring
    elif mode == "reversed":
        return bitstring[::-1]
    elif mode == "alpha_beta_swapped":
        beta_block = bitstring[:-norb]
        alpha_block = bitstring[-norb:]
        return alpha_block + beta_block
    elif mode == "interleaved":
        alpha_block = bitstring[-norb:]
        beta_block = bitstring[:-norb]
        chars = []
        for p in range(norb):
            idx = norb - 1 - p
            chars.append(alpha_block[idx])
            chars.append(beta_block[idx])
        return "".join(chars)
    else:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of "
            "'qiskit', 'reversed', 'alpha_beta_swapped', 'interleaved'."
        )


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def run_diagnostics(
    counts: dict[str, int], norb: int, nelec: tuple[int, int], top_k: int = 10
) -> dict[str, Any]:
    """Compute and print sampling diagnostics. No sampling-quality metrics here."""
    n_alpha, n_beta = nelec
    total_shots = sum(counts.values())
    unique_bitstrings = len(counts)

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[:top_k]

    print(f"Total shots: {total_shots}")
    print(f"Unique bitstrings: {unique_bitstrings}")
    print(f"Top {len(top)} bitstrings:")
    top_report = []
    for bs, cnt in top:
        prob = cnt / total_shots
        print(f"  {bs}  count={cnt}  prob={prob:.6f}")
        top_report.append({"bitstring": bs, "count": cnt, "probability": prob})

    top1_bitstring, top1_count = ranked[0]
    top1_fraction = top1_count / total_shots
    print(f"Fraction of shots on the single most frequent bitstring: {top1_fraction:.6f}")

    n_pass_total = 0
    n_pass_alpha = 0
    n_pass_beta = 0
    target_total = n_alpha + n_beta
    for bs, cnt in counts.items():
        alpha_block = bs[-norb:]
        beta_block = bs[:-norb]
        total_ones = bs.count("1")
        alpha_ones = alpha_block.count("1")
        beta_ones = beta_block.count("1")
        if total_ones == target_total:
            n_pass_total += cnt
        if alpha_ones == n_alpha:
            n_pass_alpha += cnt
        if beta_ones == n_beta:
            n_pass_beta += cnt

    pct_total = 100 * n_pass_total / total_shots
    pct_alpha = 100 * n_pass_alpha / total_shots
    pct_beta = 100 * n_pass_beta / total_shots

    print("Particle-number check (fraction of shots, by weight):")
    print(f"  total set bits == {target_total}: {pct_total:.4f}%")
    print(f"  alpha set bits == {n_alpha}: {pct_alpha:.4f}%")
    print(f"  beta set bits == {n_beta}: {pct_beta:.4f}%")

    if pct_total < 100.0 or pct_alpha < 100.0 or pct_beta < 100.0:
        print("!" * 70)
        print(
            "WARNING: particle-number check failed for some shots on a "
            "noiseless simulator. This indicates a bit-ordering or circuit "
            "construction bug."
        )
        print("!" * 70)

    return {
        "total_shots": total_shots,
        "unique_bitstrings": unique_bitstrings,
        "top_bitstrings": top_report,
        "top1_bitstring": top1_bitstring,
        "top1_fraction": top1_fraction,
        "particle_number_check": {
            "pct_total_correct": pct_total,
            "pct_alpha_correct": pct_alpha,
            "pct_beta_correct": pct_beta,
        },
    }


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------


def write_bitstring_file(
    counts: dict[str, int],
    output_path: Path,
    norb: int,
    nelec: tuple[int, int],
    total_shots: int,
    source_qpy_path: Path,
    separator: str = " ",
    include_header: bool = True,
    include_counts: bool = True,
    bit_order_mode: str = "qiskit",
) -> None:
    """Write bitstrings (optionally with counts) to a text file.

    Format is deliberately configurable: the exact format expected by the
    downstream C++ code has not yet been confirmed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    lines = []
    if include_header:
        lines.append(f"# norb={norb} nelec={list(nelec)} total_shots={total_shots}")
        lines.append(f"# source_qpy={source_qpy_path}")
        lines.append(f"# bit_order_mode={bit_order_mode}")
        lines.append(
            "# convention: qiskit little-endian (qubit 0 = rightmost char); "
            "alpha=qubits[0:norb], beta=qubits[norb:2*norb] "
            "(see stage2.convert_bit_order for other modes)"
        )
        if include_counts:
            lines.append(f"# <bitstring>{separator}<count>")
        else:
            lines.append("# <bitstring>")

    for bs, cnt in ranked:
        converted = convert_bit_order(bs, norb, bit_order_mode)
        if include_counts:
            lines.append(f"{converted}{separator}{cnt}")
        else:
            lines.append(converted)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Bitstring file written to {output_path} ({len(ranked)} unique bitstrings)")


def write_json_summary(diagnostics: dict[str, Any], output_path: Path) -> None:
    """Write the diagnostics dict as a JSON summary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"JSON summary written to {output_path}")


def _add_filename_suffix(filename: str, suffix: str) -> str:
    """Insert `suffix` before the file extension, e.g. ('a.txt', '_x') -> 'a_x.txt'."""
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix}"


def write_bitstring_outputs(
    counts: dict[str, int],
    output_dir: Path,
    bitstring_filename: str,
    norb: int,
    nelec: tuple[int, int],
    total_shots: int,
    source_qpy_path: Path,
    config: dict[str, Any],
) -> Path:
    """Write the bitstring file(s) for one sampled circuit.

    By default writes a single file in config['bit_order_mode']. If
    config['emit_all_bit_order_variants'] is True, instead writes one file
    per mode in BIT_ORDER_MODES (mode name embedded in the filename), so all
    four conventions can be tried against the downstream C++ parser without
    resampling.

    Returns:
        Path of the file written in config['bit_order_mode'] (the canonical
        file used to anchor the JSON summary).
    """
    output_dir = Path(output_dir)
    separator = config.get("output_separator", " ")
    include_header = config.get("include_header", True)
    include_counts = config.get("include_counts", True)
    default_mode = config.get("bit_order_mode", "qiskit")

    modes = BIT_ORDER_MODES if config.get("emit_all_bit_order_variants", False) else (default_mode,)

    primary_path: Path | None = None
    for mode in modes:
        suffix = f"_{mode}" if len(modes) > 1 else ""
        path = output_dir / _add_filename_suffix(bitstring_filename, suffix)
        write_bitstring_file(
            counts,
            path,
            norb=norb,
            nelec=nelec,
            total_shots=total_shots,
            source_qpy_path=source_qpy_path,
            separator=separator,
            include_header=include_header,
            include_counts=include_counts,
            bit_order_mode=mode,
        )
        if mode == default_mode:
            primary_path = path

    assert primary_path is not None
    return primary_path


def print_diversity_comparison(masked: dict[str, Any], unmasked: dict[str, Any]) -> dict[str, Any]:
    """Print and return a side-by-side masked-vs-unmasked sampling diversity comparison."""
    print()
    print("=== Masked vs unmasked sampling diversity ===")
    print(f"{'':24s}{'masked':>14s}{'unmasked':>14s}")
    print(
        f"{'unique bitstrings':24s}"
        f"{masked['unique_bitstrings']:>14d}{unmasked['unique_bitstrings']:>14d}"
    )
    print(
        f"{'top-1 fraction':24s}"
        f"{masked['top1_fraction']:>14.4f}{unmasked['top1_fraction']:>14.4f}"
    )
    print("=" * 52)
    print()
    return {
        "masked_unique_bitstrings": masked["unique_bitstrings"],
        "unmasked_unique_bitstrings": unmasked["unique_bitstrings"],
        "masked_top1_fraction": masked["top1_fraction"],
        "unmasked_top1_fraction": unmasked["top1_fraction"],
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _sample_and_write(
    qpy_path: Path,
    norb: int,
    nelec: tuple[int, int],
    config: dict[str, Any],
    bitstring_filename: str,
    label: str,
) -> dict[str, Any]:
    """Load, sample, diagnose, and write outputs for a single circuit variant."""
    print(f"--- Sampling {label} circuit ---")
    circuit = load_circuit(qpy_path)
    counts = sample_circuit(circuit, shots=config["shots"], seed=config["seed"])
    diagnostics = run_diagnostics(counts, norb, nelec, top_k=config.get("top_k_report", 10))

    output_dir = Path(config["output_dir"])
    bitstring_path = write_bitstring_outputs(
        counts, output_dir, bitstring_filename, norb, nelec,
        diagnostics["total_shots"], qpy_path, config,
    )
    summary_path = bitstring_path.with_suffix(".json")

    diagnostics["norb"] = norb
    diagnostics["nelec"] = list(nelec)
    diagnostics["source_qpy_path"] = str(qpy_path)
    diagnostics["bitstring_file_path"] = str(bitstring_path)
    diagnostics["bit_order_mode"] = config.get("bit_order_mode", "qiskit")
    write_json_summary(diagnostics, summary_path)
    return diagnostics


def run_stage2(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full Stage 2 pipeline: load .qpy -> sample -> diagnostics -> write files.

    Always samples the masked circuit at config['qpy_path']. If Stage 1
    recorded an unmasked comparison circuit in that circuit's metadata
    (key 'unmasked_qpy_path') and the file exists, it is sampled too, and a
    side-by-side sampling-diversity comparison is printed.

    Returns the masked circuit's diagnostics dict (flat, for backward
    compatibility) plus 'masked', 'unmasked', and (when both were sampled)
    'diversity_comparison' keys.
    """
    qpy_path = Path(config["qpy_path"])
    metadata = load_sibling_metadata(qpy_path)
    norb = metadata["norb"]
    nelec = tuple(metadata["nelec"])

    masked_diag = _sample_and_write(
        qpy_path, norb, nelec, config, config["bitstring_filename"], label="MASKED"
    )
    result = dict(masked_diag)
    result["masked"] = masked_diag
    result["unmasked"] = None

    unmasked_qpy_str = metadata.get("unmasked_qpy_path")
    if unmasked_qpy_str and Path(unmasked_qpy_str).exists():
        unmasked_qpy_path = Path(unmasked_qpy_str)
        unmasked_filename = _add_filename_suffix(config["bitstring_filename"], "_unmasked")
        unmasked_diag = _sample_and_write(
            unmasked_qpy_path, norb, nelec, config, unmasked_filename, label="UNMASKED"
        )
        result["unmasked"] = unmasked_diag
        result["diversity_comparison"] = print_diversity_comparison(masked_diag, unmasked_diag)
    elif unmasked_qpy_str:
        print(
            f"Unmasked qpy referenced in metadata not found on disk: "
            f"{unmasked_qpy_str}; skipping."
        )
    else:
        print("No unmasked circuit recorded in metadata; sampling masked circuit only.")

    print("=== Stage 2 complete ===")
    return result
