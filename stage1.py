"""Stage 1: molecular Hamiltonian -> LUCJ circuit construction.

Entry point: run_stage1(config). Produces a .qpy circuit file and a sibling
.json metadata file. Does not sample the circuit (see stage2.py) and does not
implement any downstream SQD subspace-diagonalisation logic.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import ffsim
import ffsim.qiskit
import numpy as np
import pyscf.cc
import pyscf.gto
import pyscf.mcscf
import pyscf.scf
import qiskit
import qiskit.qpy
from pyscf.tools import fcidump
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Benign: the FCIDUMP-reload trick in load_active_space_scf() overrides
# mol.energy_nuc / mol.intor_symmetric with plain lambdas so that CCSD/CASCI
# can run directly on FCIDUMP-derived integrals. PySCF's internal mol.dumps()
# (used for chkfile bookkeeping) warns that it cannot JSON-serialise those
# lambdas. Nothing is actually written to a chkfile here, so this is inert.
warnings.filterwarnings(
    "ignore", message="Function mol.dumps drops attribute", category=UserWarning
)


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ActiveSpaceSystem:
    """The canonical downstream representation shared by both input paths."""

    mf: Any  # a converged pyscf SCF-like object living entirely in the active space
    norb: int
    nelec: tuple[int, int]
    core_energy: float
    fcidump_path: Path


@dataclasses.dataclass
class CCSDResult:
    t1: np.ndarray
    t2: np.ndarray
    e_ccsd: float
    e_corr: float
    e_casci: float | None
    converged: bool


@dataclasses.dataclass
class ExactWavefunctionAnalysis:
    """Concentration statistics of the exact (CASCI) wavefunction.

    Physical baseline for the masked-vs-unmasked ansatz comparison: if the
    exact wavefunction already puts most of its weight on one determinant,
    a concentrated sampler reflects the molecule's physics rather than an
    artifact of the LUCJ locality mask.
    """

    ci_space_dimension: int
    weight_on_top_determinant: float
    n_det_90pct: int
    n_det_99pct: int
    n_det_999pct: int
    casci_energy: float


@dataclasses.dataclass
class AnsatzStats:
    """Circuit + energy metrics for one LUCJ ansatz (masked or unmasked)."""

    n_reps: int
    circuit_depth: int
    two_qubit_gate_count: int
    variational_energy: float
    qpy_path: Path
    metadata_path: Path


@dataclasses.dataclass
class Stage1Result:
    qpy_path: Path
    metadata_path: Path
    fcidump_path: Path
    circuit: QuantumCircuit
    metadata: dict[str, Any]
    exact_wavefunction: ExactWavefunctionAnalysis | None = None
    masked_stats: AnsatzStats | None = None
    unmasked_stats: AnsatzStats | None = None


# --------------------------------------------------------------------------
# System presets
# --------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "n2_equilibrium": {
        "atom": [["N", (0.0, 0.0, 0.0)], ["N", (0.0, 0.0, 1.098)]],
        "basis": "6-31g",
        "n_frozen_core": 4,
        "n_active_orbitals": 6,
        "n_active_electrons": 6,
    },
    "n2_stretched": {
        "atom": [["N", (0.0, 0.0, 0.0)], ["N", (0.0, 0.0, 1.55)]],
        "basis": "6-31g",
        "n_frozen_core": 4,
        "n_active_orbitals": 6,
        "n_active_electrons": 6,
    },
    "n2_very_stretched": {
        "atom": [["N", (0.0, 0.0, 0.0)], ["N", (0.0, 0.0, 2.00)]],
        "basis": "6-31g",
        "n_frozen_core": 4,
        "n_active_orbitals": 6,
        "n_active_electrons": 6,
    },
    "h2_sto3g": {
        # Smallest possible end-to-end test: norb=2, nelec=(1,1). Used to
        # calibrate the sbd bit-ordering convention (see stage2b.py).
        "atom": [["H", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, 0.74)]],
        "basis": "sto-3g",
        "n_frozen_core": 0,
        "n_active_orbitals": 2,
        "n_active_electrons": 2,
    },
}


def apply_preset(config: dict[str, Any]) -> dict[str, Any]:
    """Fill atom/basis/active-space fields from config['preset'], if set.

    Only fields whose config value is currently None are populated from the
    preset - any field the caller has already set explicitly (non-None)
    overrides the preset default. Mutates and returns `config`.
    """
    preset_name = config.get("preset")
    if not preset_name:
        print("No preset selected (CONFIG['preset'] is falsy); using explicit CONFIG values.")
        return config
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. Available: {list(PRESETS)}")

    print(f"Active preset: {preset_name!r}")
    for key, value in PRESETS[preset_name].items():
        if config.get(key) is None:
            config[key] = value
        else:
            print(
                f"  CONFIG[{key!r}] = {config[key]!r} explicitly set; "
                f"overriding preset default {value!r}."
            )
    return config


# --------------------------------------------------------------------------
# Path A: generate FCIDUMP from PySCF
# --------------------------------------------------------------------------


def _resolve_nelecas(n_active_electrons: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an active-electron count (int or explicit tuple) to (n_alpha, n_beta)."""
    if isinstance(n_active_electrons, (tuple, list)):
        n_alpha, n_beta = n_active_electrons
        return int(n_alpha), int(n_beta)
    n_total = int(n_active_electrons)
    n_beta = n_total // 2
    n_alpha = n_total - n_beta
    return n_alpha, n_beta


def generate_fcidump_from_pyscf(config: dict[str, Any]) -> Path:
    """Path A: build a Mole, run RHF, select an active space, write an FCIDUMP.

    Reads from config: atom, basis, charge, spin, n_frozen_core,
    n_active_orbitals, n_active_electrons, output_dir, fcidump_filename.

    Returns:
        Path to the written FCIDUMP file.
    """
    print("=== Stage 1 / Path A: generating FCIDUMP from PySCF ===")

    mol = pyscf.gto.Mole()
    mol.build(
        atom=config["atom"],
        basis=config["basis"],
        charge=config.get("charge", 0),
        spin=config.get("spin", 0),
        symmetry=False,
        verbose=0,
    )

    mf = pyscf.scf.RHF(mol).run(verbose=0)
    if not mf.converged:
        print("!" * 70)
        print("WARNING: RHF did not converge. Downstream results are unreliable.")
        print("!" * 70)

    n_frozen_core = config["n_frozen_core"]
    n_active_orbitals = config["n_active_orbitals"]
    nelecas = _resolve_nelecas(config["n_active_electrons"])

    expected_total = 2 * n_frozen_core + sum(nelecas)
    if expected_total != mol.nelectron:
        raise ValueError(
            f"Active space electron count is inconsistent with the molecule: "
            f"2*n_frozen_core + n_active_electrons = {expected_total}, "
            f"but mol.nelectron = {mol.nelectron}."
        )

    mc = pyscf.mcscf.CASCI(mf, ncas=n_active_orbitals, nelecas=nelecas)
    mc.ncore = n_frozen_core

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fcidump_path = output_dir / config["fcidump_filename"]

    fcidump.from_mcscf(mc, str(fcidump_path))

    print(f"  HF energy               = {mf.e_tot:.10f}")
    print(f"  Active space orbitals   = {n_active_orbitals}")
    print(f"  Active space electrons  = {nelecas} (alpha, beta)")
    print(f"  Frozen core orbitals    = {n_frozen_core}")
    print(f"  FCIDUMP written to      = {fcidump_path}")

    return fcidump_path


def load_existing_fcidump(config: dict[str, Any]) -> Path:
    """Path B: point at an existing FCIDUMP file and report its contents.

    Reads from config: existing_fcidump_path.

    Returns:
        Path to the (unmodified) FCIDUMP file.
    """
    print("=== Stage 1 / Path B: loading existing FCIDUMP ===")

    fcidump_path = Path(config["existing_fcidump_path"])
    if not fcidump_path.exists():
        raise FileNotFoundError(f"FCIDUMP file not found: {fcidump_path}")

    data = fcidump.read(str(fcidump_path))
    norb = data["NORB"]
    nelec_total = data["NELEC"]
    ms2 = data.get("MS2", 0)
    core_energy = data.get("ECORE", 0.0)

    n_alpha = (nelec_total + ms2) // 2
    n_beta = (nelec_total - ms2) // 2

    print(f"  norb        = {norb}")
    print(f"  nelec       = ({n_alpha}, {n_beta})")
    print(f"  core energy = {core_energy:.10f}")
    print(f"  FCIDUMP path = {fcidump_path}")

    return fcidump_path


def get_or_create_fcidump(config: dict[str, Any]) -> Path:
    """Single entry point dispatching to Path A or Path B based on config['fcidump_source']."""
    source = config["fcidump_source"]
    if source == "pyscf":
        return generate_fcidump_from_pyscf(config)
    elif source == "existing":
        return load_existing_fcidump(config)
    else:
        raise ValueError(
            f"config['fcidump_source'] must be 'pyscf' or 'existing', got {source!r}"
        )


# --------------------------------------------------------------------------
# Shared downstream representation
# --------------------------------------------------------------------------


def load_active_space_scf(fcidump_path: Path) -> ActiveSpaceSystem:
    """Reconstruct a converged SCF object living entirely in the active space.

    This is the convergence point for both Path A and Path B: regardless of
    how the FCIDUMP was produced, from here on Stage 1 is identical.
    """
    print(f"Reloading active-space Hamiltonian from {fcidump_path} ...")
    data = fcidump.read(str(fcidump_path))
    norb = data["NORB"]
    nelec_total = data["NELEC"]
    ms2 = data.get("MS2", 0)
    core_energy = data.get("ECORE", 0.0)
    nelec = ((nelec_total + ms2) // 2, (nelec_total - ms2) // 2)

    mf = fcidump.to_scf(str(fcidump_path))
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        print("!" * 70)
        print("WARNING: active-space RHF did not converge on reload.")
        print("!" * 70)

    print(f"  norb (active) = {norb}")
    print(f"  nelec (active) = {nelec}")
    print(f"  core energy = {core_energy:.10f}")
    print(f"  active-space RHF energy = {mf.e_tot:.10f}")

    return ActiveSpaceSystem(
        mf=mf, norb=norb, nelec=nelec, core_energy=core_energy, fcidump_path=fcidump_path
    )


# --------------------------------------------------------------------------
# Exact wavefunction baseline
# --------------------------------------------------------------------------


def analyse_exact_wavefunction(
    mf: Any, norb: int, nelec: tuple[int, int], config: dict[str, Any]
) -> ExactWavefunctionAnalysis | None:
    """Run CASCI, extract the CI vector, and report its concentration.

    Reports the weight on the single largest-amplitude determinant and how
    many determinants are needed to accumulate 90%/99%/99.9% of the total
    weight. This is the physical baseline for judging sampler concentration:
    a sampler landing 99% of shots on one bitstring is unsurprising if the
    exact wavefunction itself already puts that much weight there.

    Skipped (returns None) if norb exceeds config['casci_reference_max_norb'].
    """
    max_norb = config.get("casci_reference_max_norb", 12)
    if norb > max_norb:
        print(
            f"Skipping exact wavefunction analysis "
            f"(norb={norb} > casci_reference_max_norb={max_norb})."
        )
        return None

    print("Analysing exact (CASCI) wavefunction concentration ...")
    mc = pyscf.mcscf.CASCI(mf, norb, nelec)
    mc.verbose = 0
    mc.kernel()

    probs = np.abs(np.asarray(mc.ci)).ravel() ** 2
    probs_sorted = np.sort(probs)[::-1]
    cumulative = np.cumsum(probs_sorted)
    dim = int(probs_sorted.size)

    def n_dets_for(threshold: float) -> int:
        """Number of largest-amplitude determinants needed to reach `threshold` cumulative weight."""
        return int(np.searchsorted(cumulative, threshold) + 1)

    top_weight = float(probs_sorted[0])
    n90 = n_dets_for(0.90)
    n99 = n_dets_for(0.99)
    n999 = n_dets_for(0.999)

    print(f"  CI space dimension            = {dim}")
    print(f"  Weight on top determinant     = {top_weight:.6f}")
    print(f"  Determinants for 90% weight   = {n90}")
    print(f"  Determinants for 99% weight   = {n99}")
    print(f"  Determinants for 99.9% weight = {n999}")

    return ExactWavefunctionAnalysis(
        ci_space_dimension=dim,
        weight_on_top_determinant=top_weight,
        n_det_90pct=n90,
        n_det_99pct=n99,
        n_det_999pct=n999,
        casci_energy=mc.e_tot,
    )


# --------------------------------------------------------------------------
# CCSD
# --------------------------------------------------------------------------


def run_ccsd(system: ActiveSpaceSystem, config: dict[str, Any]) -> CCSDResult:
    """Run CCSD in the active space and extract t1/t2 amplitudes.

    FAILSAFE: warns loudly if CCSD does not converge, or if a CASCI (exact
    diagonalisation in the same active space) reference is available and the
    CCSD energy lies below it (a variational impossibility indicating that the
    amplitudes are unreliable).
    """
    print("Running CCSD in the active space ...")
    cc_obj = pyscf.cc.CCSD(system.mf)
    cc_obj.verbose = 0
    cc_obj.run()

    e_ccsd = cc_obj.e_tot
    e_corr = cc_obj.e_corr

    print(f"  CCSD converged      = {cc_obj.converged}")
    print(f"  CCSD total energy   = {e_ccsd:.10f}")
    print(f"  CCSD correlation E  = {e_corr:.10f}")

    if not cc_obj.converged:
        print("!" * 70)
        print("WARNING: CCSD DID NOT CONVERGE. t1/t2 amplitudes are unreliable.")
        print("!" * 70)

    e_casci = None
    max_norb = config.get("casci_reference_max_norb", 12)
    if config.get("run_casci_reference", True) and system.norb <= max_norb:
        print("Computing CASCI (exact diagonalisation) reference in the active space ...")
        mc = pyscf.mcscf.CASCI(system.mf, system.norb, system.nelec)
        mc.verbose = 0
        mc.kernel()
        e_casci = mc.e_tot
        print(f"  CASCI (exact) energy = {e_casci:.10f}")

        if e_ccsd < e_casci - 1e-6:
            print()
            print("!" * 70)
            print("!!! WARNING: CCSD HAS DIVERGED !!!")
            print("!" * 70)
            print(
                "!!! The CCSD energy is BELOW the exact (CASCI) energy computed\n"
                "!!! in the same active space. This is variationally impossible\n"
                "!!! for a well-behaved calculation.\n"
                "!!! The t1/t2 amplitudes - and therefore the LUCJ ansatz built\n"
                "!!! from them - ARE UNRELIABLE.\n"
                "!!! Results from this geometry/active space should NOT be trusted.\n"
                "!!! Continuing anyway (not aborting) so the failure mode is visible."
            )
            print("!" * 70)
            print()
    else:
        print(
            f"  Skipping CASCI reference (norb={system.norb} > "
            f"casci_reference_max_norb={max_norb}, or disabled)."
        )

    return CCSDResult(
        t1=cc_obj.t1,
        t2=cc_obj.t2,
        e_ccsd=e_ccsd,
        e_corr=e_corr,
        e_casci=e_casci,
        converged=cc_obj.converged,
    )


# --------------------------------------------------------------------------
# LUCJ operator + circuit construction
# --------------------------------------------------------------------------


def build_lucj_operator(
    ccsd_result: CCSDResult, norb: int
) -> tuple[ffsim.UCJOpSpinBalanced, list[tuple[int, int]], list[tuple[int, int]]]:
    """Build the LUCJ operator with the mandated parameter choices.

    n_reps=None (full-rank double factorisation) and optimize=False are
    MANDATORY and must not be changed: n_reps=None gives 1.2 mHa error vs
    CCSD, whereas lower ranks (e.g. n_reps=2) collapse the ansatz back toward
    Hartree-Fock (160 mHa error); optimize=True has been observed to make
    energies dramatically worse and introduces nondeterminism.

    The interaction-pair locality mask follows the standard heavy-hex
    zig-zag SQD layout: same-spin nearest-neighbour chain, opposite-spin
    on-site coupling every 4th orbital.
    """
    aa_pairs = [(p, p + 1) for p in range(norb - 1)]
    ab_pairs = [(p, p) for p in range(0, norb, 4)]

    possible_aa = norb * (norb - 1) // 2
    possible_ab = norb * norb
    print("Locality mask (heavy-hex zig-zag layout):")
    print(f"  aa (same-spin) pairs retained: {len(aa_pairs)} / {possible_aa} possible -> {aa_pairs}")
    print(f"  ab (opposite-spin) pairs retained: {len(ab_pairs)} / {possible_ab} possible -> {ab_pairs}")

    print("Building LUCJ operator (n_reps=None, optimize=False; MANDATORY, do not change) ...")
    lucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd_result.t2,
        t1=ccsd_result.t1,
        n_reps=None,
        interaction_pairs=(aa_pairs, ab_pairs),
        optimize=False,
    )
    print(f"  Actual n_reps chosen by full-rank factorisation: {lucj_op.n_reps}")

    return lucj_op, aa_pairs, ab_pairs


def build_unmasked_lucj_operator(ccsd_result: CCSDResult) -> ffsim.UCJOpSpinBalanced:
    """Build a LUCJ operator with NO locality mask (interaction_pairs=None).

    Comparison-only ansatz: quantifies what the locality mask (used by
    build_lucj_operator) costs relative to the unrestricted LUCJ form. Still
    uses the mandatory n_reps=None, optimize=False.
    """
    print("Building UNMASKED LUCJ operator (interaction_pairs=None; comparison only) ...")
    lucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd_result.t2,
        t1=ccsd_result.t1,
        n_reps=None,
        interaction_pairs=None,
        optimize=False,
    )
    print(f"  Actual n_reps chosen by full-rank factorisation: {lucj_op.n_reps}")
    return lucj_op


def compute_variational_energy(
    lucj_op: ffsim.UCJOpSpinBalanced,
    molecular_data: ffsim.MolecularData,
    norb: int,
    nelec: tuple[int, int],
) -> float:
    """Classically evaluate <HF| U(lucj_op)^dagger H U(lucj_op) |HF>.

    Used to compare ansatz quality (masked vs unmasked) against CCSD/CASCI
    without needing to sample a circuit.
    """
    reference = ffsim.hartree_fock_state(norb, nelec)
    state = ffsim.apply_unitary(reference, lucj_op, norb=norb, nelec=nelec)
    hamiltonian_op = ffsim.linear_operator(molecular_data.hamiltonian, norb=norb, nelec=nelec)
    # The Hamiltonian's underlying contraction kernel (pyscf) only accepts
    # real float64 vectors even though the LinearOperator advertises complex
    # dtype; apply it to the real and imaginary parts of `state` separately
    # (valid since H is a real linear map) rather than to `state` directly.
    if np.iscomplexobj(state):
        h_state = hamiltonian_op @ state.real + 1j * (hamiltonian_op @ state.imag)
    else:
        h_state = hamiltonian_op @ state
    return float(np.real(np.vdot(state, h_state)))


def _add_filename_suffix(filename: str, suffix: str) -> str:
    """Insert `suffix` before the file extension, e.g. ('a.qpy', '_x') -> 'a_x.qpy'."""
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix}"


def print_ansatz_comparison(
    masked: AnsatzStats,
    unmasked: AnsatzStats,
    ccsd_energy: float,
    casci_energy: float | None,
) -> None:
    """Print a side-by-side masked-vs-unmasked circuit and energy comparison table."""
    print()
    print("=== Masked vs unmasked LUCJ ansatz comparison ===")
    print(f"{'':28s}{'masked':>18s}{'unmasked':>18s}")
    print(f"{'n_reps':28s}{masked.n_reps:>18d}{unmasked.n_reps:>18d}")
    print(f"{'circuit depth':28s}{masked.circuit_depth:>18d}{unmasked.circuit_depth:>18d}")
    print(
        f"{'two-qubit gate count':28s}"
        f"{masked.two_qubit_gate_count:>18d}{unmasked.two_qubit_gate_count:>18d}"
    )
    print(
        f"{'variational energy':28s}"
        f"{masked.variational_energy:>18.10f}{unmasked.variational_energy:>18.10f}"
    )
    print(f"{'CCSD energy (reference)':28s}{ccsd_energy:>18.10f}{ccsd_energy:>18.10f}")
    if casci_energy is not None:
        print(f"{'CASCI energy (reference)':28s}{casci_energy:>18.10f}{casci_energy:>18.10f}")
    print(
        f"{'error vs CCSD (mHa)':28s}"
        f"{1000 * (masked.variational_energy - ccsd_energy):>18.4f}"
        f"{1000 * (unmasked.variational_energy - ccsd_energy):>18.4f}"
    )
    print("=" * 64)
    print()


def build_circuit(
    lucj_op: ffsim.UCJOpSpinBalanced, norb: int, nelec: tuple[int, int]
) -> QuantumCircuit:
    """Build the state-preparation + LUCJ + measurement Qiskit circuit.

    Bit ordering: Qiskit is little-endian (qubit 0 is the rightmost character
    of the measurement string). ffsim maps alpha spin-orbitals to qubits
    0..norb-1 and beta spin-orbitals to qubits norb..2*norb-1.
    """
    print("Building Qiskit circuit (HF prep + UCJ + measure_all) ...")
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)
    circuit.measure_all()
    print(f"  Circuit built: {circuit.num_qubits} qubits, depth {circuit.depth()}")
    return circuit


def _two_qubit_gate_count(circuit: QuantumCircuit) -> int:
    """Count circuit instructions acting on exactly two qubits."""
    return sum(1 for instr in circuit.data if len(instr.qubits) == 2)


def transpile_circuit(circuit: QuantumCircuit, optimization_level: int = 3) -> QuantumCircuit:
    """Transpile the circuit, using ffsim.qiskit.PRE_INIT as a pre-init pass if available.

    Transpilation targets AerSimulator's basis so that ffsim's high-level
    gates (e.g. UCJOpSpinBalancedJW, PrepareHartreeFockJW) are fully unrolled
    to gates AerSimulator recognises - required for Stage 2 sampling.
    """
    import qiskit_aer

    pre_depth = circuit.depth()
    pre_2q = _two_qubit_gate_count(circuit)
    print(f"Pre-transpilation:  depth={pre_depth}, two-qubit gates={pre_2q}")

    backend = qiskit_aer.AerSimulator()
    try:
        pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
        pm.pre_init = ffsim.qiskit.PRE_INIT
        transpiled = pm.run(circuit)
        print("Transpilation path: preset pass manager with ffsim.qiskit.PRE_INIT pre-init stage.")
    except Exception as exc:  # noqa: BLE001 - deliberate broad guard per spec
        print(f"ffsim.qiskit.PRE_INIT unavailable or failed ({exc!r}); falling back to plain qiskit.transpile.")
        transpiled = qiskit.transpile(circuit, backend=backend, optimization_level=optimization_level)
        print("Transpilation path: qiskit.transpile (no PRE_INIT).")

    post_depth = transpiled.depth()
    post_2q = _two_qubit_gate_count(transpiled)
    print(f"Post-transpilation: depth={post_depth}, two-qubit gates={post_2q}")

    return transpiled


# --------------------------------------------------------------------------
# Serialisation + validation
# --------------------------------------------------------------------------


def serialize_circuit(
    circuit: QuantumCircuit,
    metadata: dict[str, Any],
    output_dir: Path,
    circuit_filename: str,
) -> tuple[Path, Path]:
    """Write the circuit to .qpy and metadata to a sibling .json file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qpy_path = output_dir / circuit_filename
    metadata_path = qpy_path.with_suffix(".json")

    with open(qpy_path, "wb") as f:
        qiskit.qpy.dump(circuit, f)
    print(f"Circuit serialised to {qpy_path}")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata written to {metadata_path}")

    return qpy_path, metadata_path


def validate_roundtrip(qpy_path: Path, original_circuit: QuantumCircuit) -> None:
    """VALIDATION GATE: reload the .qpy and assert it matches the original circuit."""
    print(f"Validating round-trip of {qpy_path} ...")
    with open(qpy_path, "rb") as f:
        (reloaded,) = qiskit.qpy.load(f)

    assert reloaded.num_qubits == original_circuit.num_qubits, (
        f"Qubit count mismatch: reloaded={reloaded.num_qubits}, "
        f"original={original_circuit.num_qubits}"
    )
    assert reloaded.depth() == original_circuit.depth(), (
        f"Depth mismatch: reloaded={reloaded.depth()}, original={original_circuit.depth()}"
    )
    assert dict(reloaded.count_ops()) == dict(original_circuit.count_ops()), (
        f"Operation count mismatch: reloaded={dict(reloaded.count_ops())}, "
        f"original={dict(original_circuit.count_ops())}"
    )
    print("  Round-trip validation PASSED: qubits, depth, and operation counts match.")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_stage1(config: dict[str, Any]) -> Stage1Result:
    """Run the full Stage 1 pipeline: Hamiltonian -> LUCJ circuit -> .qpy + metadata.

    Also builds an unmasked comparison ansatz when
    config['build_unmasked_comparison'] is True (default), so the cost of the
    locality mask can be quantified against the exact-wavefunction baseline
    from analyse_exact_wavefunction.
    """
    import importlib.metadata as _im

    import qiskit_aer  # local import: only needed for version string

    config = apply_preset(config)

    fcidump_path = get_or_create_fcidump(config)
    system = load_active_space_scf(fcidump_path)
    exact_wf = analyse_exact_wavefunction(system.mf, system.norb, system.nelec, config)
    ccsd_result = run_ccsd(system, config)
    molecular_data = ffsim.MolecularData.from_scf(system.mf)

    common_metadata = {
        "norb": system.norb,
        "nelec": list(system.nelec),
        "optimize": False,
        "basis": config.get("basis"),
        "geometry": config.get("atom"),
        "hf_energy": system.mf.e_tot,
        "ccsd_energy": ccsd_result.e_ccsd,
        "ccsd_correlation_energy": ccsd_result.e_corr,
        "ccsd_converged": ccsd_result.converged,
        "casci_reference_energy": ccsd_result.e_casci,
        "ffsim_version": _im.version("ffsim"),
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "fcidump_path": str(fcidump_path),
        "bit_order_convention": (
            "qiskit little-endian: qubit 0 is the rightmost character of the "
            "measurement string. Alpha spin-orbitals -> qubits 0..norb-1, "
            "beta spin-orbitals -> qubits norb..2*norb-1."
        ),
    }
    if exact_wf is not None:
        common_metadata["exact_wavefunction"] = dataclasses.asdict(exact_wf)

    # --- masked ansatz (mandatory: n_reps=None, optimize=False, locality mask) ---
    lucj_op, aa_pairs, ab_pairs = build_lucj_operator(ccsd_result, system.norb)
    circuit = build_circuit(lucj_op, system.norb, system.nelec)
    transpiled = transpile_circuit(circuit, config.get("optimization_level", 3))
    masked_energy = compute_variational_energy(lucj_op, molecular_data, system.norb, system.nelec)
    print(f"Masked ansatz variational energy: {masked_energy:.10f}")

    # --- unmasked comparison ansatz (optional) ---
    unmasked_qpy_path: Path | None = None
    unmasked_metadata_path: Path | None = None
    unmasked_stats: AnsatzStats | None = None
    if config.get("build_unmasked_comparison", True):
        unmasked_op = build_unmasked_lucj_operator(ccsd_result)
        unmasked_circuit = build_circuit(unmasked_op, system.norb, system.nelec)
        unmasked_transpiled = transpile_circuit(unmasked_circuit, config.get("optimization_level", 3))
        unmasked_energy = compute_variational_energy(
            unmasked_op, molecular_data, system.norb, system.nelec
        )
        print(f"Unmasked ansatz variational energy: {unmasked_energy:.10f}")

        masked_stats_preview = AnsatzStats(
            n_reps=lucj_op.n_reps,
            circuit_depth=transpiled.depth(),
            two_qubit_gate_count=_two_qubit_gate_count(transpiled),
            variational_energy=masked_energy,
            qpy_path=Path(config["output_dir"]) / config["circuit_filename"],
            metadata_path=(Path(config["output_dir"]) / config["circuit_filename"]).with_suffix(".json"),
        )
        unmasked_filename = _add_filename_suffix(config["circuit_filename"], "_unmasked")
        unmasked_stats = AnsatzStats(
            n_reps=unmasked_op.n_reps,
            circuit_depth=unmasked_transpiled.depth(),
            two_qubit_gate_count=_two_qubit_gate_count(unmasked_transpiled),
            variational_energy=unmasked_energy,
            qpy_path=Path(config["output_dir"]) / unmasked_filename,
            metadata_path=(Path(config["output_dir"]) / unmasked_filename).with_suffix(".json"),
        )
        print_ansatz_comparison(
            masked_stats_preview, unmasked_stats, ccsd_result.e_ccsd, ccsd_result.e_casci
        )

        unmasked_metadata = dict(common_metadata)
        unmasked_metadata.update(
            {
                "n_reps_requested": None,
                "n_reps_actual": unmasked_op.n_reps,
                "interaction_pairs": None,
                "circuit_depth": unmasked_transpiled.depth(),
                "two_qubit_gate_count": _two_qubit_gate_count(unmasked_transpiled),
                "variational_energy": unmasked_energy,
                "ansatz_variant": "unmasked",
            }
        )
        unmasked_qpy_path, unmasked_metadata_path = serialize_circuit(
            unmasked_transpiled, unmasked_metadata, Path(config["output_dir"]), unmasked_filename
        )
        validate_roundtrip(unmasked_qpy_path, unmasked_transpiled)
    else:
        print("Skipping unmasked comparison ansatz (config['build_unmasked_comparison'] is False).")

    # --- masked metadata + serialisation ---
    metadata = dict(common_metadata)
    metadata.update(
        {
            "n_reps_requested": None,
            "n_reps_actual": lucj_op.n_reps,
            "interaction_pairs": {"aa": aa_pairs, "ab": ab_pairs},
            "circuit_depth": transpiled.depth(),
            "two_qubit_gate_count": _two_qubit_gate_count(transpiled),
            "variational_energy": masked_energy,
            "ansatz_variant": "masked",
            "unmasked_qpy_path": str(unmasked_qpy_path) if unmasked_qpy_path else None,
        }
    )

    qpy_path, metadata_path = serialize_circuit(
        transpiled, metadata, Path(config["output_dir"]), config["circuit_filename"]
    )
    validate_roundtrip(qpy_path, transpiled)

    masked_stats = AnsatzStats(
        n_reps=lucj_op.n_reps,
        circuit_depth=transpiled.depth(),
        two_qubit_gate_count=_two_qubit_gate_count(transpiled),
        variational_energy=masked_energy,
        qpy_path=qpy_path,
        metadata_path=metadata_path,
    )

    print("=== Stage 1 complete ===")
    return Stage1Result(
        qpy_path=qpy_path,
        metadata_path=metadata_path,
        fcidump_path=fcidump_path,
        circuit=transpiled,
        metadata=metadata,
        exact_wavefunction=exact_wf,
        masked_stats=masked_stats,
        unmasked_stats=unmasked_stats,
    )
