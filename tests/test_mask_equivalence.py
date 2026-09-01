"""tests/test_mask_equivalence.py

Asserts mechanism A (unified_run.py: P J P^T absorbed into orbital_rotations,
fixed mask applied after) and mechanism B (run_ordering_pipeline.py: operator
built directly with ffsim's interaction_pairs) construct entrywise-identical
LUCJ operators - after un-permuting mechanism A - for 20 random permutations
on the cached N2 CAS(6,10) reference.

Operator-level, no sampling, no sbd: this is the same test as
`experiments/preflight.py crosscheck`'s primary check, run over more
permutations, as a regression guard so the two pipelines can never again
silently diverge (see notes/PROGRESS.md, "Voided results (25 Aug)", for what
happened the one time they did).

Run: pytest tests/test_mask_equivalence.py -v
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import unified_run as U  # noqa: E402  (cached reference load; safe on import)
import run_ordering_pipeline as R  # noqa: E402  (safe on import, no side effects)
import preflight  # noqa: E402  (safe on import: heavy work is behind __main__)

N_RANDOM_PERMUTATIONS = 20
RANDOM_SEED = 20260825  # distinct from every seed used elsewhere in the pipeline


def _random_permutations(n: int, norb: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    perms: list[np.ndarray] = []
    while len(perms) < n:
        p = rng.permutation(norb)
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            perms.append(p)
    return perms


PERMUTATIONS = _random_permutations(N_RANDOM_PERMUTATIONS, U.NORB, RANDOM_SEED)


@pytest.mark.parametrize("perm", PERMUTATIONS, ids=lambda p: "".join(map(str, p)))
def test_mechanism_a_equals_mechanism_b(perm: np.ndarray) -> None:
    """Un-permuted mechanism-A operator must equal mechanism-B's operator
    entrywise (diag_coulomb_mats aa/ab, orbital_rotations), to < 1e-12.
    """
    t1, t2 = U.ref_data["t1"], U.ref_data["t2"]
    result = preflight.operator_level_check(U, R, perm, t1, t2, U.NORB)

    assert result["shape_match"], (
        f"n_reps mismatch: A={result['n_reps_A']} B={result['n_reps_B']}"
    )
    assert result["diff_aa"] < 1e-12, f"aa diff {result['diff_aa']:.3e}"
    assert result["diff_ab"] < 1e-12, f"ab diff {result['diff_ab']:.3e}"
    assert result["diff_U"] < 1e-12, f"orbital_rotations diff {result['diff_U']:.3e}"
