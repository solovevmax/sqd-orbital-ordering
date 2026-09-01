import sys, json
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import numpy as np
import unified_run as U
import run_ordering_pipeline as R

R.CFG["sbd_bin"] = str(U.SBD)

def load_and_sample(cachedir, seed=2026, shots=2_000_000):
    ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=cachedir)
    norb, nocc = ref["norb"], ref["nocc"]
    nelec = (nocc, nocc)
    t1L, t2L = ref["t1L"], ref["t2L"]
    centroids = ref["centroids"]
    Jaa, Jab = R.diag_coulomb(R.build_ucj(t2L, t1L))
    pos = R.positions_from(np.arange(norb))
    pairs = R.interaction_pairs_for(pos, centroids, J_ab=Jab)
    op = R.build_ucj(t2L, t1L, interaction_pairs=pairs)
    a_c, b_c, depth = R.sample_bitstrings(op, norb, nelec, shots, seed)
    return dict(ref=ref, a_c=a_c, b_c=b_c, depth=depth, norb=norb, nocc=nocc,
                t1L=t1L, t2L=t2L, Jaa=Jaa, Jab=Jab)

def report(label, a_c, b_c, norb, nocc):
    hf = R.hf_bitstring(norb, nocc)
    print(f"\n===== {label} =====")
    for name, counts in (("ALPHA", a_c), ("BETA", b_c)):
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"--- {name} top-20 (of {len(counts)} unique) ---")
        for i, (s, n) in enumerate(ranked[:20]):
            tag = " <-- HF" if s == hf else ""
            print(f"  rank{i+1:3d}  {s}  count={n}  frac={n/2_000_000:.8f}{tag}")
    a_sel, _ = R.top_dets(a_c, 15, hf)
    b_sel, _ = R.top_dets(b_c, 15, hf)
    print(f"top-15 alpha selected: {sorted(a_sel)}")
    print(f"top-15 beta selected:  {sorted(b_sel)}")
    return set(a_sel), set(b_sel)

print("Loading + sampling CACHED reference (original repo)...")
cached = load_and_sample("cache/h10_R1.6")
print("Loading + sampling FRESH reference (rebuilt from scratch)...")
fresh = load_and_sample("/private/tmp/claude-502/-Users-maxim-sqd-project/fe23625e-57cc-474d-8afb-d2c2063922ce/scratchpad/cold_start/sqd-project/cache/h10_R1.6")

a_sel_cached, b_sel_cached = report("CACHED reference", cached["a_c"], cached["b_c"], cached["norb"], cached["nocc"])
a_sel_fresh, b_sel_fresh = report("FRESH reference", fresh["a_c"], fresh["b_c"], fresh["norb"], fresh["nocc"])

print("\n===== DIFF =====")
print(f"alpha: {len(a_sel_cached ^ a_sel_fresh)} strings differ")
print(f"  only in cached: {a_sel_cached - a_sel_fresh}")
print(f"  only in fresh:  {a_sel_fresh - a_sel_cached}")
print(f"beta:  {len(b_sel_cached ^ b_sel_fresh)} strings differ")
print(f"  only in cached: {b_sel_cached - b_sel_fresh}")
print(f"  only in fresh:  {b_sel_fresh - b_sel_cached}")

# A2: is the FULL distribution different, or just the cut?
print("\n===== A2: distribution comparison =====")
for name, ca, cb in (("alpha", cached["a_c"], fresh["a_c"]), ("beta", cached["b_c"], fresh["b_c"])):
    keys = set(ca) | set(cb)
    max_frac_diff = 0.0
    max_key = None
    for k in keys:
        va, vb = ca.get(k, 0), cb.get(k, 0)
        d = abs(va - vb) / 2_000_000
        if d > max_frac_diff:
            max_frac_diff, max_key = d, k
    n_common_keys = len(set(ca) & set(cb))
    print(f"{name}: unique strings cached={len(ca)} fresh={len(cb)} common={n_common_keys}  "
          f"max |frac diff| over union = {max_frac_diff:.8f} at {max_key}")

# save raw data for later inspection
import pickle
with open("/private/tmp/claude-502/-Users-maxim-sqd-project/fe23625e-57cc-474d-8afb-d2c2063922ce/scratchpad/cold_start/task_a1_data.pkl", "wb") as f:
    pickle.dump(dict(cached_a=cached["a_c"], cached_b=cached["b_c"],
                      fresh_a=fresh["a_c"], fresh_b=fresh["b_c"],
                      cached_t1L=cached["t1L"], cached_t2L=cached["t2L"],
                      fresh_t1L=fresh["t1L"], fresh_t2L=fresh["t2L"],
                      cached_Jaa=cached["Jaa"], cached_Jab=cached["Jab"],
                      fresh_Jaa=fresh["Jaa"], fresh_Jab=fresh["Jab"]), f)
print("\nsaved task_a1_data.pkl")
