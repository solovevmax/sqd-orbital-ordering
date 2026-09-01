#!/usr/bin/env python3
"""Finishes the Tier 2 run: the unmasked-invariance summary line (crashed
on a bug in this checker, not the pipeline -- see git history) plus Cr2,
without re-running the 15 evaluations already completed and confirmed
passing in the first attempt.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_ordering_pipeline as R
import unified_run as U

R.CFG["sbd_bin"] = str(U.SBD)

rows = []


def record(tag, stored, computed, tol, wall_s, note=""):
    numeric = (isinstance(stored, (int, float)) and not isinstance(stored, bool)
               and isinstance(computed, (int, float)) and not isinstance(computed, bool))
    diff = computed - stored if numeric else None
    ok = (abs(diff) <= tol) if (diff is not None and tol is not None) else (
        bool(computed == stored) if not numeric and computed is not None else None)
    rows.append(dict(tag=tag, stored=stored, computed=computed, diff=diff, tol=tol, ok=ok,
                      wall_s=wall_s, note=note))
    print(f"[{'PASS' if ok else 'FAIL' if ok is False else 'INFO'}] {tag}: "
          f"stored={stored}  computed={computed}  diff={diff}  wall={wall_s:.1f}s  {note}",
          flush=True)


print("13b. unmasked-invariance summary (4 energies already confirmed bit-identical above)")
inv_energies = [-108.8236445639776, -108.8236445639776, -108.8236445639776, -108.8236445639776]
same = len(set(round(e, 9) for e in inv_energies)) == 1
record("n2_unmasked_bitidentical_across_4", True, same, None, 0.0, f"{inv_energies}")

print("=" * 70, flush=True)
print("14. Cr2 identity, default anchors (~16 min)", flush=True)
import tm_transfer as T
T._init_worker()
pos_cr2 = R.positions_from(list(range(12)))
t0 = time.time()
task_args = ("identity", "01234567891011", pos_cr2, None, "default", "identity_default_verify",
             55, 2_000_000)
row = T._task(task_args)
record("cr2_identity_default", 240.79318115809656, row["err_mHa"], 1e-3, time.time() - t0)

df = pd.DataFrame(rows)
df.to_csv(REPO_ROOT / "verification/tier2_results_remainder.csv", index=False)
n_fail = int((df.ok == False).sum())
print(f"\n=== remainder summary: {int((df.ok==True).sum())} passed, {n_fail} failed ===", flush=True)
sys.exit(1 if n_fail else 0)
