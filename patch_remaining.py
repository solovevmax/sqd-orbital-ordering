#!/usr/bin/env python3
from pathlib import Path

FILE = Path("run_ordering_pipeline.py")
txt = FILE.read_text()

# 1) Update interaction_pairs_for signature and body
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

# 2) In stage3, replace the call site
old_call = "pairs = interaction_pairs_for(pos)"
new_call = """centroids = orbital_centroids(mol, C_loc, active)
        pairs = interaction_pairs_for(pos, centroids)"""

if old_call not in txt:
    raise RuntimeError("stage3 interaction_pairs_for call not found")
txt = txt.replace(old_call, new_call)
print("Wired centroids into stage3 interaction_pairs_for call")

FILE.write_text(txt)
print("Patch complete.")
