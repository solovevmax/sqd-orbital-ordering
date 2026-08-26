#!/usr/bin/env python3
from pathlib import Path

FILE = Path("run_ordering_pipeline.py")
txt = FILE.read_text()

# Replace the retained_J_of function to use anchor-based opp_spin_sites internally
old_rj = """def retained_J_of(pos, J_aa, J_ab):
    tot = np.abs(J_aa).sum() + np.abs(J_ab).sum()
    if tot <= 0:
        return 0.0
    keep = sum(2.0 * np.abs(J_aa[:, p, q]).sum() for p, q in same_spin_pairs(pos))
    keep += sum(np.abs(J_ab[:, p, p]).sum() for p in opp_spin_sites(pos))
    return float(keep / tot)"""

new_rj = '''def _opp_spin_sites_anchor(pos):
    """Legacy anchor-based opposite-spin sites, used for retained_J scalar."""
    return [int(p) for p in range(len(pos)) if pos[p] % CFG["anchor_mod"] == 0]


def retained_J_of(pos, J_aa, J_ab):
    tot = np.abs(J_aa).sum() + np.abs(J_ab).sum()
    if tot <= 0:
        return 0.0
    keep = sum(2.0 * np.abs(J_aa[:, p, q]).sum() for p, q in same_spin_pairs(pos))
    keep += sum(np.abs(J_ab[:, p, p]).sum() for p in _opp_spin_sites_anchor(pos))
    return float(keep / tot)'''

if old_rj not in txt:
    raise RuntimeError("retained_J_of old definition not found")
txt = txt.replace(old_rj, new_rj)
print("Patched retained_J_of to use anchor-based opposite-spin sites")

FILE.write_text(txt)
print("Patch complete.")
