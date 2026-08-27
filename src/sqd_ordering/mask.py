"""src/sqd_ordering/mask.py

Single source of truth for the LUCJ heavy-hex locality mask used throughout
this project: which same-spin (aa) and opposite-spin (ab) Jastrow
(diag_coulomb_mats) entries survive, for a given orbital ordering `pos`.

`pos` convention (matches positions_from(perm, convention="layout") in
run_ordering_pipeline.py): pos[orbital] = the layout position that orbital
occupies. inv = argsort(pos) then gives inv[k] = the orbital sitting at
position k.

History: run_ordering_pipeline.py's original same_spin_pairs() omitted the
same-spin diagonal (p, p) entries that unified_run.py's fixed mask always
retained. This module is the fix, extracted so both pipelines share one
definition and cannot diverge again (see tests/test_mask_equivalence.py and
notes/PROGRESS.md, "Voided results (25 Aug)").
"""
from __future__ import annotations

import numpy as np


def _orbital_at_position(pos: np.ndarray) -> np.ndarray:
    """inv[k] = orbital occupying position k, given pos[orbital] = its position."""
    return np.argsort(pos)


def same_spin_pairs(pos: np.ndarray, norb: int) -> set[tuple[int, int]]:
    """Same-spin (aa) Jastrow entries the heavy-hex mask retains.

    Nearest-neighbour pairs along the layout (inv[k], inv[k+1]), PLUS every
    same-orbital diagonal (p, p), unconditionally and independent of
    ordering.

    The diagonal MUST be included: a same-spin diagonal term J_pp n_p n_p
    reduces by fermionic idempotency (n_p^2 = n_p for an occupation number
    operator) to J_pp n_p, a single-qubit Z rotation. Heavy-hex connectivity
    constrains two-qubit gates only - a single-qubit rotation needs no
    nearest-neighbour justification and there is no hardware reason to mask
    it out. Omitting it (as run_ordering_pipeline.py's original
    same_spin_pairs did) implements an unphysical mask, not an alternative
    one. Do not remove this without re-deriving that argument.
    """
    inv = _orbital_at_position(pos)
    pairs = {tuple(sorted((int(inv[k]), int(inv[k + 1])))) for k in range(len(inv) - 1)}
    pairs |= {(p, p) for p in range(norb)}
    return pairs


def opp_spin_pairs(
    pos: np.ndarray, norb: int, anchor_mod: int = 4, anchor_offset: int = 0,
    anchor_orbitals=None,
) -> set[tuple[int, int]]:
    """Opposite-spin (ab) Jastrow entries the mask retains: on-site (p, p)
    terms for orbitals whose LAYOUT POSITION is congruent to `anchor_offset`
    modulo `anchor_mod` (default offset 0 - the original, unchanged
    behaviour). If `anchor_orbitals` is given (an iterable of ORBITAL
    indices, not positions), it is used directly instead - free selection
    of which orbitals anchor, independent of any arithmetic-progression
    positional pattern.
    """
    if anchor_orbitals is not None:
        return {(int(p), int(p)) for p in anchor_orbitals}
    inv = _orbital_at_position(pos)
    return {(int(inv[k]), int(inv[k])) for k in range(anchor_offset % anchor_mod, norb, anchor_mod)}


def mask_matrices(
    pos: np.ndarray, norb: int, anchor_mod: int = 4, anchor_offset: int = 0,
    anchor_orbitals=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (norb, norb) same-spin and opposite-spin masks for `pos`."""
    m_aa = np.zeros((norb, norb), dtype=bool)
    for p, q in same_spin_pairs(pos, norb):
        m_aa[p, q] = m_aa[q, p] = True
    m_ab = np.zeros((norb, norb), dtype=bool)
    for p, q in opp_spin_pairs(pos, norb, anchor_mod=anchor_mod, anchor_offset=anchor_offset,
                               anchor_orbitals=anchor_orbitals):
        m_ab[p, q] = m_ab[q, p] = True
    return m_aa, m_ab


def interaction_pairs_for(
    pos: np.ndarray, norb: int, anchor_mod: int = 4, anchor_offset: int = 0,
    anchor_orbitals=None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(pairs_aa, pairs_ab) for ffsim's interaction_pairs: normalised p <= q, deduped."""
    aa = sorted(same_spin_pairs(pos, norb))
    ab = sorted(opp_spin_pairs(pos, norb, anchor_mod=anchor_mod, anchor_offset=anchor_offset,
                               anchor_orbitals=anchor_orbitals))
    return list(aa), list(ab)


def retained_J(
    pos: np.ndarray, J_aa: np.ndarray, J_ab: np.ndarray, anchor_mod: int = 4,
    anchor_offset: int = 0, anchor_orbitals=None,
) -> float:
    """Fraction of the Jastrow matrices' squared magnitude the mask retains.

    A STRUCTURAL, pre-sampling property of the ansatz. Matches
    unified_run.py's original L2 (squared-magnitude) definition exactly -
    the only change from history is that mask_matrices now includes the
    same-spin diagonal for every caller, not just unified_run.py's.
    """
    J_aa = np.asarray(J_aa)
    J_ab = np.asarray(J_ab)
    norb = J_aa.shape[-1]
    m_aa, m_ab = mask_matrices(pos, norb, anchor_mod=anchor_mod, anchor_offset=anchor_offset,
                               anchor_orbitals=anchor_orbitals)
    total = np.sum(J_aa ** 2) + np.sum(J_ab ** 2)
    if total <= 0:
        return 0.0
    kept = np.sum((J_aa * m_aa) ** 2) + np.sum((J_ab * m_ab) ** 2)
    return float(kept / total)


def retained_J_split(
    pos: np.ndarray, J_aa: np.ndarray, J_ab: np.ndarray, anchor_mod: int = 4,
    anchor_offset: int = 0, anchor_orbitals=None,
) -> tuple[float, float]:
    """retained_J's same-spin and opposite-spin sectors reported separately,
    each normalised by its OWN total squared magnitude (so both are
    fractions in [0, 1], not a further split of the combined fraction).
    """
    J_aa = np.asarray(J_aa)
    J_ab = np.asarray(J_ab)
    norb = J_aa.shape[-1]
    m_aa, m_ab = mask_matrices(pos, norb, anchor_mod=anchor_mod, anchor_offset=anchor_offset,
                               anchor_orbitals=anchor_orbitals)
    tot_aa = np.sum(J_aa ** 2)
    tot_ab = np.sum(J_ab ** 2)
    same = float(np.sum((J_aa * m_aa) ** 2) / tot_aa) if tot_aa > 0 else 0.0
    opp = float(np.sum((J_ab * m_ab) ** 2) / tot_ab) if tot_ab > 0 else 0.0
    return same, opp
