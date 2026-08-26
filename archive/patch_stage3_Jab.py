#!/usr/bin/env python3
from pathlib import Path

FILE = Path("run_ordering_pipeline.py")
txt = FILE.read_text()

# 1) Update interaction_pairs_for signature to accept J_ab
old_sig = "def interaction_pairs_for(pos, centroids=None, J_ab=None):"
# If this is already the case from the previous patch, we're good.
# If not, update from the simpler signature.
if old_sig not in txt:
    old_sig2 = "def interaction_pairs_for(pos, centroids=None):"
    if old_sig2 not in txt:
        raise RuntimeError("Cannot find interaction_pairs_for signature")
    txt = txt.replace(old_sig2, old_sig)
    print("Updated interaction_pairs_for signature to accept J_ab")
else:
    print("interaction_pairs_for already accepts J_ab")

# 2) Update the body to use J_ab when calling opp_spin_sites
old_body = "ab = sorted((p, p) for p in opp_spin_sites(pos, centroids, J_ab))"
if old_body not in txt:
    old_body2 = "ab = sorted((p, p) for p in opp_spin_sites(pos, centroids))"
    if old_body2 not in txt:
        raise RuntimeError("Cannot find interaction_pairs_for body")
    txt = txt.replace(old_body2, old_body)
    print("Updated interaction_pairs_for body to pass J_ab to opp_spin_sites")
else:
    print("interaction_pairs_for body already passes J_ab")

# 3) In stage3, update the call site to pass J_ab
# Find the line after "Jaa, Jab = diag_coulomb(build_ucj(t2L, t1L))"
# and then the call "pairs = interaction_pairs_for(pos, centroids, J_ab=None)" or similar.
old_call = "pairs = interaction_pairs_for(pos, centroids, J_ab=None)"
new_call = "pairs = interaction_pairs_for(pos, centroids, J_ab=Jab)"
if old_call in txt:
    txt = txt.replace(old_call, new_call)
    print("Updated stage3 call site to pass J_ab=Jab")
else:
    # Maybe the call is without J_ab yet
    old_call2 = "pairs = interaction_pairs_for(pos, centroids)"
    if old_call2 not in txt:
        raise RuntimeError("Cannot find stage3 interaction_pairs_for call")
    txt = txt.replace(old_call2, new_call)
    print("Updated stage3 call site to pass J_ab=Jab")

FILE.write_text(txt)
print("Patch complete.")
