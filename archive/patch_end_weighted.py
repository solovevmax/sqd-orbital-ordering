#!/usr/bin/env python3
from pathlib import Path

FILE = Path("run_ordering_pipeline.py")
txt = FILE.read_text()

# 1) Add the end-weighted opposite-spin function after _centered_opp_spin_sites
marker = "def _centered_opp_spin_sites(pos, centroids, k):"
if marker not in txt:
    raise RuntimeError("Cannot find _centered_opp_spin_sites to insert after")

end_weighted_func = '''

def _end_weighted_opp_spin_sites(pos, centroids, k):
    """
    Select k orbitals closest to the chain ends (end-weighted mask).
    Translation-invariant and permutation-sensitive.
    """
    import numpy as np
    z = np.array([float(centroids[int(orb)]) for orb in range(len(pos))])
    zmin, zmax = z.min(), z.max()
    if zmax - zmin < 1e-12:
        # Degenerate case: all centroids identical; fall back to first k orbitals
        return list(range(k))
    x = (z - zmin) / (zmax - zmin)  # normalized to [0,1]
    s = np.minimum(x, 1.0 - x)       # small near ends
    idx = np.argsort(s)
    return [int(i) for i in idx[:k]]

'''

# Insert after _centered_opp_spin_sites definition
lines = txt.splitlines(keepends=True)
insert_line = None
for i, line in enumerate(lines):
    if marker in line:
        insert_line = i + 1
        break
if insert_line is None:
    raise RuntimeError("Cannot find line to insert after")
# Insert after the next blank line following the marker
for i in range(insert_line, len(lines)):
    if lines[i].strip() == "":
        insert_line = i + 1
        break
lines.insert(insert_line, end_weighted_func)
txt = "".join(lines)

# 2) Update opp_spin_sites dispatcher to handle "end_weighted"
old_dispatch = '''if CFG["mask_mode"] == "centered":
        assert centroids is not None, "centroids required for centered mask"
        return _centered_opp_spin_sites(pos, centroids, CFG["k_os"])
    # legacy anchor-based mask
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]'''

new_dispatch = '''if CFG["mask_mode"] == "centered":
        assert centroids is not None, "centroids required for centered mask"
        return _centered_opp_spin_sites(pos, centroids, CFG["k_os"])
    if CFG["mask_mode"] == "end_weighted":
        assert centroids is not None, "centroids required for end_weighted mask"
        return _end_weighted_opp_spin_sites(pos, centroids, CFG["k_os"])
    # legacy anchor-based mask
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]'''

if old_dispatch not in txt:
    raise RuntimeError("Cannot find opp_spin_sites dispatcher to update")
txt = txt.replace(old_dispatch, new_dispatch)

print("Added end_weighted mask and updated opp_spin_sites dispatcher")

FILE.write_text(txt)
print("Patch complete.")
