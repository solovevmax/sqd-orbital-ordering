# SQD Orbital Ordering

Python pipeline for constructing LUCJ ansatz circuits, sampling them, and
exporting inputs for sample-based quantum diagonalisation (SQD).

Work carried out during a summer internship at RIKEN R-CCS with
Prof. Seiji Yunoki and Dr Tomonori Shirakawa.

## Motivation

SQD uses a quantum circuit as a *sampler* to identify the electronic
configurations that matter most, then diagonalises the molecular Hamiltonian
classically in the subspace those configurations span.

The LUCJ ansatz used for this sampling is subject to a hardware locality
constraint: on a heavy-hex qubit topology, only same-spin nearest-neighbour
Jastrow elements `(p, p+1)` and opposite-spin elements at every fourth orbital
`(p, p)` for `p % 4 == 0` can be implemented without expensive SWAP networks.
Everything else is discarded.

Which elements survive that mask depends entirely on how orbitals are mapped to
qubits — and that mapping is conventionally inherited from the active-space
construction without a physical criterion. This project builds the
infrastructure needed to test whether orbital ordering measurably changes
sampling quality, and how such orderings should be scored.

## Pipeline

    Stage 1  PySCF -> FCIDUMP -> CCSD amplitudes -> LUCJ operator -> .qpy circuit
    Stage 2  .qpy -> Qiskit Aer sampling -> bitstrings with counts
    Stage 2B bitstrings -> alpha/beta determinant files in sbd format
    Stage 3  FCIDUMP + determinant files -> sbd (external C++, MPI) -> energy

Stage 3 is the `sbd` library by T. Shirakawa
(https://github.com/r-ccs-cms/sbd) and is not modified by this project.

### Files

| File | Purpose |
|---|---|
| `SQD_workflow.ipynb` | Main driver notebook; all parameters in one `CONFIG` dict |
| `stage1.py` | Hamiltonian generation, CCSD, LUCJ construction, transpilation, QPY serialisation |
| `stage2.py` | Aer sampling, particle-number diagnostics, bitstring export |
| `stage2b.py` | Alpha/beta splitting and sbd determinant-file export |
| `POC.ipynb` | Proof of concept: orbital permutation invariance and mask sensitivity |
| `sbd-build-notes/` | macOS build recipe for the external sbd library |
| `outputs/` | Generated FCIDUMPs, circuits, bitstrings, diagnostics |

## Validated interface

The Python-to-C++ interface has been checked empirically on H2/STO-3G:

| Input | Energy (Ha) | Reference |
|---|---|---|
| HF determinant only | `-1.116759307396425` | Hartree-Fock |
| Full sampled determinant set | `-1.137283834488501` | FCI |

Both match to all printed digits, confirming that the FCIDUMP format and the
bitstring convention are correct.

### Bitstring convention

Qiskit writes measurement strings with qubit 0 as the **rightmost** character.
ffsim maps alpha spin-orbitals to qubits `0..norb-1` and beta to
`norb..2*norb-1`. sbd expects each determinant with the **rightmost bit
corresponding to orbital 1**.

These conventions coincide, so the required operation is a **split, not a
reversal**:

    alpha = full_bitstring[-norb:]    # rightmost norb characters
    beta  = full_bitstring[:norb]     # leftmost norb characters

Alternative transforms are available via `CONFIG["sbd_bit_transform"]`.

## Mandatory parameter choices

Verified empirically; changing these breaks the pipeline:

- **`n_reps=None`** — full-rank double factorisation. `n_reps` is the rank of
  the factorisation of the t2 amplitudes; at low rank the ansatz collapses back
  to Hartree-Fock. For H10/STO-3G, `n_reps=2` gives 160 mHa error against CCSD
  while `n_reps=None` gives 1.2 mHa.
- **`optimize=False`** — the compressed-factorisation optimiser has been
  observed to worsen energies substantially and introduces nondeterminism that
  would confound ordering comparisons.

## Preliminary findings

N2 in a CAS(6,6) active space, 6-31g, 10^6 shots, comparing masked (locality
constrained) against unmasked (full UCJ) ansätze:

| Geometry | Exact top-1 weight | Unmasked top-1 | Masked top-1 |
|---|---|---|---|
| 1.098 Å (equilibrium) | 0.9393 | 0.9408 | 0.9940 |
| 1.55 Å (stretched) | 0.7705 | 0.7964 | 0.9800 |
| 2.00 Å (very stretched) | 0.3455 | 0.3288 | 0.8746 |

The unmasked ansatz reproduces the exact wavefunction's concentration closely
at all three geometries. The masked ansatz is systematically over-concentrated,
and the discrepancy grows with correlation strength — that is, the locality
mask does most damage precisely where multireference treatment matters most.

Because sbd forms the tensor product of the alpha and beta sectors, sampling
diversity should be assessed on the **marginals** rather than the joint
distribution: at equilibrium, 50 unique sampled bitstrings yielded 16 alpha and
15 beta determinants, giving a 240-dimensional product space, or 60% of the
full CI space.

## Caveats

- Masked and unmasked energies are those of the CCSD-initialised state, not
  variationally optimised values.
- The masked/unmasked comparison is a control experiment establishing that the
  locality mask matters. It is not itself the orbital-ordering experiment.
- CCSD diverges at 2.00 Å (energy below CASCI), so energies from that geometry
  are unreliable; the sampling distributions there are reported for interest
  only.
- Ordering comparisons require the product-space dimension to be held fixed
  across orderings; this control is not yet implemented.

## Environment

    conda create -n sqd -c conda-forge python=3.12 pyscf
    conda activate sqd
    python -m pip install ffsim qiskit qiskit-aer qiskit-addon-sqd matplotlib

For building the external sbd library on macOS, see `sbd-build-notes/`.

## References

1. J. Robledo-Moreno *et al.*, *Sci. Adv.* **11**, eadu9991 (2025);
   arXiv:2405.05068 — the SQD method.
2. T. Shirakawa *et al.*, arXiv:2511.00224 (2025) — closed-loop SQD at full
   scale on Fugaku.
3. M. Motta *et al.*, *Chem. Sci.* (2023) — the LUCJ ansatz.
4. T. Shirakawa, `sbd`: library for selected basis diagonalisation,
   https://github.com/r-ccs-cms/sbd