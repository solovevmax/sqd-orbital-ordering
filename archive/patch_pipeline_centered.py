#!/usr/bin/env python3
from pathlib import Path

FILE = Path("run_ordering_pipeline.py")
txt = FILE.read_text()

# 1) Add mask_mode and k_os after anchor_mod line
if 'mask_mode="centered"' not in txt:
    txt = txt.replace(
        'anchor_mod=4, # opposite-spin on-site mask: position % 4 == 0',
        '''anchor_mod=4, # opposite-spin on-site mask: position % 4 == 0
    mask_mode="centered",  # "anchor" or "centered"
    k_os=4,                # number of opposite-spin on-site terms for centered mask'''
    )
    print("Added mask_mode and k_os to CFG")
else:
    print("mask_mode and k_os already present")

# 2) Replace opp_spin_sites function (exact text match)
old_opp = """def opp_spin_sites(pos):
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]"""

new_opp = '''def opp_spin_sites(pos, centroids=None):
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

if old_opp not in txt:
    raise RuntimeError("opp_spin_sites old definition not found")
txt = txt.replace(old_opp, new_opp)
print("Replaced opp_spin_sites with centered-aware version")

# 3) Replace interaction_pairs_for function (exact text match)
old_ipf = """def interaction_pairs_for(pos):
    \"\"\"(pairs_aa, pairs_ab) for ffsim, normalised to p <= q and deduped.\"\"\"
    aa = sorted({tuple(sorted(pq)) for pq in same_spin_pairs(pos)})
    ab = sorted({(p, p) for p in opp_spin_sites(pos)})
    return list(aa), list(ab)"""

new_ipf = '''def interaction_pairs_for(pos, centroids=None):
    """(pairs_aa, pairs_ab) for ffsim, normalised to p <= q and deduped."""
    aa = sorted({tuple(sorted(pq)) for pq in same_spin_pairs(pos)})
    ab = sorted((p, p) for p in opp_spin_sites(pos, centroids))
    return list(aa), list(ab)'''

if old_ipf not in txt:
    raise RuntimeError("interaction_pairs_for old definition not found")
txt = txt.replace(old_ipf, new_ipf)
print("Updated interaction_pairs_for to accept centroids")

# 4) In stage3, replace the call site
old_call = "pairs = interaction_pairs_for(pos)"
new_call = """centroids = orbital_centroids(mol, C_loc, active)
        pairs = interaction_pairs_for(pos, centroids)"""

if old_call not in txt:
    raise RuntimeError("stage3 interaction_pairs_for call not found")
txt = txt.replace(old_call, new_call)
print("Wired centroids into stage3 interaction_pairs_for call")

FILE.write_text(txt)
print("Patch complete.")
