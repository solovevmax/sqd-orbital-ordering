# Stage 1-3 Interface Validation

System: H2 / STO-3G, CAS(2,2), norb=2, nelec=(1,1)
Date: 2026-08-20

| Subspace | sbd energy (Ha) | Reference (Ha) | Log |
|---|---|---|---|
| HF determinant only | -1.116759307396425 | -1.116759307 (RHF) | `sbd_validation_h2_hf_only.log` |
| Full sampled set | -1.137283834488501 | -1.137283834 (FCI) | `sbd_validation_h2_full.log` |

Orbital occupations: HF-only gives [2, 0] exactly; full set gives
[1.97466774704596, 0.02533225295404094].

Bit convention: split, no reversal.
  alpha = bitstring[-norb:]
  beta  = bitstring[:norb]

Validates: FCIDUMP interface, determinant file format, alpha/beta splitting,
bit ordering, tensor-product subspace construction, Davidson diagonalisation.