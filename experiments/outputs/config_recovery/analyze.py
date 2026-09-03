import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("recovery_results.csv")
perm = df[df.arm == "permutation"]
anchor = df[df.arm == "anchor_axis"]
PERM_LAYOUTS = perm.label.unique().tolist()

print("=" * 78)
print("A1 -- spread across the 8 permutation layouts, iteration 0 to 5")
print("=" * 78)
for it in range(6):
    sub = perm[perm.iteration == it]
    lo, hi = sub.err_mHa.min(), sub.err_mHa.max()
    ratio = hi / lo if lo > 0 else float("inf")
    print(f"  it={it}: range=[{lo:.4f}, {hi:.4f}]  spread={hi-lo:.4f} mHa  ratio={ratio:.3f}x")

it0 = perm[perm.iteration == 0]
it5 = perm[perm.iteration == 5]
spread0 = it0.err_mHa.max() - it0.err_mHa.min()
spread5 = it5.err_mHa.max() - it5.err_mHa.min()
print(f"\n  spread(it0)={spread0:.4f} mHa -> spread(it5)={spread5:.4f} mHa  "
      f"(narrows by {100*(1-spread5/spread0):.2f}%, factor {spread0/spread5 if spread5>0 else float('inf'):.1f}x)")

print()
print("=" * 78)
print("A2 -- rank stability: Kendall tau, iteration 0 ranking vs each subsequent")
print("=" * 78)
piv = perm.pivot_table(index="label", columns="iteration", values="err_mHa")
piv = piv.loc[PERM_LAYOUTS]
for it in range(1, 6):
    tau, p = stats.kendalltau(piv[0], piv[it])
    print(f"  tau(it0, it{it}) = {tau:.4f}  (p={p:.4f})")

print()
print("=" * 78)
print("A3 -- convergence: same energy or different, at iteration 5")
print("=" * 78)
print(piv[5].to_string())
n_distinct = piv[5].round(4).nunique()
print(f"\n  distinct iteration-5 energies (rounded to 1e-4 mHa): {n_distinct} of {len(piv)}")
print(f"  seed noise at 2e6 shots (established elsewhere in this project): 0.0 mHa (bit-identical)")
maxdiff_common = piv[5][piv[5].round(4) == piv[5].round(4).mode()[0]]
print(f"  {len(maxdiff_common)}/{len(piv)} layouts converge to the identical energy "
      f"({piv[5].mode()[0]:.6f} mHa); outlier(s): "
      f"{piv[5][piv[5].round(4) != piv[5].round(4).mode()[0]].to_dict()}")

print()
print("=" * 78)
print("A4 -- rho(captured, err_mHa) across the 8 layouts, at each iteration")
print("=" * 78)
for it in range(6):
    sub = perm[perm.iteration == it]
    if sub.captured.nunique() < 2 or sub.err_mHa.nunique() < 2:
        print(f"  it={it}: degenerate (captured or err_mHa constant across layouts) -- rho undefined")
        continue
    rho, p = stats.spearmanr(sub.captured, sub.err_mHa)
    print(f"  it={it}: rho={rho:.4f}  p={p:.4f}  (n={len(sub)})")

print()
print("=" * 78)
print("A5 -- anchor axis: default, best(0,1,2), no-ab -- same energy or different?")
print("=" * 78)
apiv = anchor.pivot_table(index="label", columns="iteration", values="err_mHa")
print(apiv.to_string())

print()
print("=" * 78)
print("A3b/Amendment3 -- spread of DIMENSIONS across layouts at each iteration")
print("=" * 78)
for it in range(6):
    sub = perm[perm.iteration == it]
    print(f"  it={it}: dim range=[{sub.dim.min()}, {sub.dim.max()}]  "
          f"(dim_a range=[{sub.dim_a.min()},{sub.dim_a.max()}])")

print()
print("=" * 78)
print("Amendment 3 -- traced fraction (original sample) vs iteration, permutation arm")
print("=" * 78)
tpiv = perm.pivot_table(index="label", columns="iteration", values="frac_traced_a")
print(tpiv.loc[PERM_LAYOUTS].to_string())
print(f"\n  mean traced fraction across 8 layouts, by iteration:")
print(perm.groupby("iteration").frac_traced_a.mean().to_string())

print()
print("=" * 78)
print("A6 -- cost: wall time and sbd-call count per layout, 6 iterations (0-5)")
print("=" * 78)
cost = df.groupby("label").agg(total_wall_s=("wall_s", "sum"), n_calls=("wall_s", "count"))
print(cost.to_string())
print(f"\n  total wall time, all 11 trajectories: {df.wall_s.sum():.1f}s "
      f"({df.wall_s.sum()/60:.2f} min, sbd calls only, excludes Aer sampling ~35s/layout)")

print()
print("=" * 78)
print("Amendment 4 -- did the pre-registered prediction hold?")
print("=" * 78)
print("Prediction: layout influence decays roughly in proportion to the sampled")
print("determinants' declining share of the pool (traced fraction).")
mean_traced_by_it = perm.groupby("iteration").frac_traced_a.mean()
spread_by_it = perm.groupby("iteration").err_mHa.apply(lambda s: s.max() - s.min())
print("\n  iteration | mean traced fraction | spread (mHa)")
for it in range(6):
    print(f"  {it:9d} | {mean_traced_by_it[it]:.4f}              | {spread_by_it[it]:.4f}")
corr_traced_spread = np.corrcoef(mean_traced_by_it.values, spread_by_it.values)[0, 1]
print(f"\n  Pearson correlation, traced-fraction vs spread, across the 6 iterations: {corr_traced_spread:.4f}")
