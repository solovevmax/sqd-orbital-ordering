#!/usr/bin/env python3
from pathlib import Path

FILE = Path("run_ordering_pipeline.py")
txt = FILE.read_text()

# 1) Add the largest-J_ab opposite-spin function after _end_weighted_opp_spin_sites
marker = "def _end_weighted_opp_spin_sites(pos, centroids, k):"
if marker not in txt:
    raise RuntimeError("Cannot find _end_weighted_opp_spin_sites to insert after")

largest_J_func = '''

def _largest_J_ab_opp_spin_sites(pos, J_ab, k):
    """
    Select k orbitals with largest |J_ab(p,p)| (largest onsite opposite-spin coupling).
    Translation-invariant and permutation-sensitive.
    """
    import numpy as np
    diag = np.abs(np.diag(J_ab))
    idx = np.argsort(-diag)  # descending
    return [int(i) for i in idx[:k]]

'''

# Insert after _end_weighted_opp_spin_sites definition
lines = txt.splitlines(keepends=True)
insert_line = None
for i, line in enumerate(lines):
    if marker in line:
        insert_line = i + 1
        break
if insert_line is None:
    raise RuntimeError("Cannot find line to insert after")
# Insert after the next blank line following the function
for i in range(insert_line, len(lines)):
    if lines[i].strip() == "":
        insert_line = i + 1
        break
lines.insert(insert_line, largest_J_func)
txt = "".join(lines)

# 2) Update opp_spin_sites dispatcher to handle "largest_J_ab"
# First, extend the function signature to accept J_ab
old_sig = "def opp_spin_sites(pos, centroids=None):"
new_sig = "def opp_spin_sites(pos, centroids=None, J_ab=None):"
if old_sig not in txt:
    raise RuntimeError("Cannot find opp_spin_sites signature to update")
txt = txt.replace(old_sig, new_sig)

old_dispatch = '''if CFG["mask_mode"] == "centered":
        assert centroids is not None, "centroids required for centered mask"
        return _centered_opp_spin_sites(pos, centroids, CFG["k_os"])
    if CFG["mask_mode"] == "end_weighted":
        assert centroids is not None, "centroids required for end_weighted mask"
        return _end_weighted_opp_spin_sites(pos, centroids, CFG["k_os"])
    # legacy anchor-based mask
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]'''

new_dispatch = '''if CFG["mask_mode"] == "centered":
        assert centroids is not None, "centroids required for centered mask"
        return _centered_opp_spin_sites(pos, centroids, CFG["k_os"])
    if CFG["mask_mode"] == "end_weighted":
        assert centroids is not None, "centroids required for end_weighted mask"
        return _end_weighted_opp_spin_sites(pos, centroids, CFG["k_os"])
    if CFG["mask_mode"] == "largest_J_ab":
        assert J_ab is not None, "J_ab required for largest_J_ab mask"
        return _largest_J_ab_opp_spin_sites(pos, J_ab, CFG["k_os"])
    # legacy anchor-based mask
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]'''

if old_dispatch not in txt:
    raise RuntimeError("Cannot find opp_spin_sites dispatcher to update")
txt = txt.replace(old_dispatch, new_dispatch)

# 3) Update interaction_pairs_for to pass J_ab to opp_spin_sites
old_ipf_body = """ab = sorted((p, p) for p in opp_spin_sites(pos, centroids))"""
new_ipf_body = """ab = sorted((p, p) for p in opp_spin_sites(pos, centroids, J_ab))"""
if old_ipf_body not in txt:
    raise RuntimeError("Cannot find interaction_pairs_for body to update")
txt = txt.replace(old_ipf_body, new_ipf_body)

# Also update the function signature to accept J_ab
old_ipf_sig = "def interaction_pairs_for(pos, centroids=None):"
new_ipf_sig = "def interaction_pairs_for(pos, centroids=None, J_ab=None):"
if old_ipf_sig not in txt:
    raise RuntimeError("Cannot find interaction_pairs_for signature to update")
txt = txt.replace(old_ipf_sig, new_ipf_sig)

print("Added largest_J_ab mask and updated opp_spin_sites + interaction_pairs_for")

FILE.write_text(txt)
print("Patch complete.")
