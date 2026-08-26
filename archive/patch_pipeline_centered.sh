#!/usr/bin/env bash
set -euo pipefail

FILE="run_ordering_pipeline.py"

# 1) Add mask_mode and k_os to CFG dict
# Find the line with "anchor_mod=4," and insert after it.
if grep -q 'mask_mode="centered"' "$FILE"; then
  echo "mask_mode and k_os already present in CFG"
else
  sed -i.bak '/anchor_mod=4,/a\
    mask_mode="centered",  # "anchor" or "centered"\
    k_os=4,                # number of opposite-spin on-site terms for centered mask' "$FILE"
  echo "Added mask_mode and k_os to CFG"
fi

# 2) Replace opp_spin_sites with version that supports centered mask
# We'll replace the function definition block.
python3 <<'PYPATCH'
import re

with open("run_ordering_pipeline.py", "r") as f:
    txt = f.read()

old_func = r"""def opp_spin_sites\(pos\):
    return \[int\(p\) for p in range\(len\(pos\)\) if pos\[p\] % CFG\["anchor_mod"\] == 0\]"""

new_func = '''def opp_spin_sites(pos, centroids=None):
    if CFG["mask_mode"] == "centered":
        assert centroids is not None, "centroids required for centered mask"
        return _centered_opp_spin_sites(pos, centroids, CFG["k_os"])
    # legacy anchor-based mask
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]


def _centered_opp_spin_sites(pos, centroids, k):
    """Select k orbitals closest to the centroid of all active orbitals."""
    center = float(np.mean(centroids))
    dists = [(abs(float(centroids[int(orb)]) - center), int(orb))
             for orb in range(len(pos))]
    dists.sort()
    return [int(orb) for _, orb in dists[:k]]'''

txt2 = re.sub(old_func, new_func, txt)
if txt2 == txt:
    raise RuntimeError("opp_spin_sites pattern not found")
with open("run_ordering_pipeline.py", "w") as f:
    f.write(txt2)

print("Replaced opp_spin_sites with centered-aware version")
PYPATCH

# 3) Update interaction_pairs_for to accept centroids
python3 <<'PYPATCH2'
import re

with open("run_ordering_pipeline.py", "r") as f:
    txt = f.read()

old_sig = r"""def interaction_pairs_for\(pos\):
    """\(pairs_aa, pairs_ab\) for ffsim, normalised to p <= q and deduped."""
    aa = sorted\({tuple\(sorted\(pq\)\) for pq in same_spin_pairs\(pos\)}\)
    ab = sorted\({\(p, p\) for p in opp_spin_sites\(pos\)}\)
    return list\(aa\), list\(ab\)"""

new_sig = '''def interaction_pairs_for(pos, centroids=None):
    """(pairs_aa, pairs_ab) for ffsim, normalised to p <= q and deduped."""
    aa = sorted({tuple(sorted(pq)) for pq in same_spin_pairs(pos)})
    ab = sorted((p, p) for p in opp_spin_sites(pos, centroids))
    return list(aa), list(ab)'''

txt2 = re.sub(old_sig, new_sig, txt)
if txt2 == txt:
    raise RuntimeError("interaction_pairs_for pattern not found")
with open("run_ordering_pipeline.py", "w") as f:
    f.write(txt2)

print("Updated interaction_pairs_for to accept centroids")
PYPATCH2

# 4) In stage3, ensure centroids is computed and passed to interaction_pairs_for
# This is a small textual patch: replace "pairs = interaction_pairs_for(pos)"
# with "centroids = orbital_centroids(...); pairs = interaction_pairs_for(pos, centroids)"
python3 <<'PYPATCH3'
import re

with open("run_ordering_pipeline.py", "r") as f:
    txt = f.read()

old_call = r"pairs = interaction_pairs_for\(pos\)"
new_call = """centroids = orbital_centroids(mol, C_loc, active)
        pairs = interaction_pairs_for(pos, centroids)"""

txt2 = re.sub(old_call, new_call, txt)
if txt2 == txt:
    raise RuntimeError("stage3 interaction_pairs_for call not found")
with open("run_ordering_pipeline.py", "w") as f:
    f.write(txt2)

print("Wired centroids into stage3 interaction_pairs_for call")
PYPATCH3

echo "Patch complete. Backup saved as ${FILE}.bak"
