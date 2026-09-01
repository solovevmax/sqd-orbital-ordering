"""Task C: compute the anchor-optimisation compression factor from raw
per-chain CSVs, with the default anchor triple guaranteed included in the
"best anchors" candidate set at every chain (per the report's own stated
methodology in sec:compression) -- independent of any analysis code that
produced the original numbers.
"""
import pandas as pd

g1 = pd.read_csv("../experiments/outputs/g1_lite/g1_summary.csv")
h12 = pd.read_csv("../experiments/outputs/chain_aware/step2_b35_new12.csv")

g1["best_with_default"] = g1[["baseline", "best_of_40"]].min(axis=1)
g1_out = g1[["ordering", "baseline", "best_of_40", "best_with_default"]].rename(
    columns={"ordering": "chain"})

h12["best_with_default"] = h12[["baseline", "best_of_43"]].min(axis=1)
h12_out = h12[["ordering", "baseline", "best_of_43", "best_with_default"]].rename(
    columns={"ordering": "chain", "best_of_43": "best_of_40"})

combined = pd.concat([g1_out, h12_out], ignore_index=True)
assert len(combined) == 20, f"expected 20 chains, got {len(combined)}"

default_spread = combined["baseline"].max() - combined["baseline"].min()
opt_spread = combined["best_with_default"].max() - combined["best_with_default"].min()
naive_opt_spread = combined["best_of_40"].max() - combined["best_of_40"].min()

print(combined.to_string(index=False))
print()
print(f"n chains: {len(combined)}")
print(f"default spread:                          "
      f"{combined.baseline.min():.2f} - {combined.baseline.max():.2f}  = {default_spread:.2f} mHa")
print(f"optimised spread (default always incl.):  "
      f"{combined.best_with_default.min():.2f} - {combined.best_with_default.max():.2f}  = {opt_spread:.2f} mHa")
print(f"optimised spread (naive, best-of-sampled only): "
      f"{combined.best_of_40.min():.2f} - {combined.best_of_40.max():.2f}  = {naive_opt_spread:.2f} mHa")
print()
print(f"compression factor (default-included, matches the report's stated method): "
      f"{default_spread/opt_spread:.4f}x")
print(f"compression factor (naive, best-of-sampled only -- NOT the reported method): "
      f"{default_spread/naive_opt_spread:.4f}x")

n_default_not_beaten = (combined["baseline"] <= combined["best_of_40"]).sum()
print(f"\nchains where the sampled candidate set alone never beat the default "
      f"(consistent with default excluded from that chain's sample): "
      f"{n_default_not_beaten} / {len(combined)}")
