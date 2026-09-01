#!/usr/bin/env python3
"""Tier 0: recompute headline statistics from raw per-evaluation CSVs,
independently -- no code from experiments/*.py or src/sqd_ordering/ is
imported. Only pandas/numpy/scipy on the raw CSVs listed in claims.yaml.

Usage: python3 verification/verify_tier0.py [--claims verification/claims.yaml]
Exit code is nonzero if any check fails.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent


def close(a, b, tol):
    return abs(a - b) <= tol


def pct_close(a, b, tol_frac=1e-4):
    return abs(a - b) <= max(tol_frac * abs(b), 1e-6)


class Checker:
    def __init__(self):
        self.results = []

    def check(self, claim_id, claimed, computed, tol, note=""):
        ok = abs(computed - claimed) <= tol if isinstance(tol, (int, float)) else None
        self.results.append(dict(id=claim_id, claimed=claimed, computed=computed,
                                  diff=computed - claimed if isinstance(claimed, (int, float)) else None,
                                  tol=tol, ok=ok, note=note))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {claim_id}: claimed={claimed}  computed={computed}  "
              f"diff={computed - claimed if isinstance(claimed,(int,float)) else 'n/a'}  {note}")
        return ok

    def summary(self):
        n = len(self.results)
        nfail = sum(1 for r in self.results if r["ok"] is not None and not bool(r["ok"]))
        npass = sum(1 for r in self.results if r["ok"] is not None and bool(r["ok"]))
        print(f"\n=== Tier 0 summary: {npass}/{n} passed, {nfail} failed ===")
        return nfail


C = Checker()

# ---------------------------------------------------------------- N2 -----
n2 = pd.read_csv(REPO_ROOT / "outputs/unified/results.csv")
n2_rand = n2[(n2.kind == "random") & (n2.dim == 225)]
n2_rand_by_ord = n2_rand.groupby("ordering")["err_sub_mHa"].mean()

C.check("r1_n2_random_best", 21.65, round(n2_rand_by_ord.min(), 2), 0.02)
C.check("r1_n2_random_worst", 173.23, round(n2_rand_by_ord.max(), 2), 0.02)
C.check("r1_n2_random_median", 49.22, round(n2_rand_by_ord.median(), 2), 0.02)
C.check("abs_factor_of_eight", 8, round(n2_rand_by_ord.max() / n2_rand_by_ord.min(), 2), 0.05,
        "8.0x ratio worst/best, N2")
C.check("r1_n2_p5_p95_ratio", 2.71,
        round(n2_rand_by_ord.quantile(0.95) / n2_rand_by_ord.quantile(0.05), 2), 0.05)

rho_n2, p_n2 = stats.spearmanr(n2_rand_by_ord.index.map(
    n2_rand.groupby("ordering")["captured"].mean()), n2_rand_by_ord.values)
n2_capture_by_ord = n2_rand.groupby("ordering")["captured"].mean()
rho_n2, p_n2 = stats.spearmanr(n2_capture_by_ord.values, n2_rand_by_ord.values)
C.check("mech_rho_n2", -0.880, round(rho_n2, 3), 0.002)
C.check("mech_p_n2", 3e-49, p_n2, p_n2 * 0.5 + 1e-50, f"order-of-magnitude check, p={p_n2:.2e}")

# named orderings
n2_named = n2[n2.kind == "named"].groupby("ordering")["err_sub_mHa"].mean()
C.check("r1_n2_identity_err", 31.87, round(n2_named.get("identity", float("nan")), 2), 0.02)
C.check("r1_n2_oracle_err", 24.27, round(n2_named.get("max_captured_ORACLE", float("nan")), 2), 0.02)
C.check("r1_n2_maxretJ_err", 44.32, round(n2_named.get("max_retainedJ", float("nan")), 2), 0.02)

# The "5-seed study" (27.2, tau=0.83) is NOT in outputs/unified/results.csv
# (2 seeds only) -- it is a separate, archived legacy dataset. Found via
# targeted search; flagged in REPORT.md as claim provenance living in
# archive/, not any current experiment directory.
n2_5seed = pd.read_csv(REPO_ROOT / "archive/legacy_outputs/seed_replication_n2_cas610_155.csv")
n2_5seed_piv = n2_5seed.pivot_table(index="ordering", columns="seed", values="err_sub_mHa").dropna()
between5 = n2_5seed_piv.mean(axis=1).var()
within5 = n2_5seed_piv.var(axis=1).mean()
ratio_n2_5seed = between5 / within5
C.check("abs_between_within_ratio_n2", 27.2, round(ratio_n2_5seed, 1), 0.1,
        "source: archive/legacy_outputs/ -- see claims.yaml note")

import itertools
taus = [stats.kendalltau(n2_5seed_piv[a].rank(), n2_5seed_piv[b].rank())[0]
        for a, b in itertools.combinations(n2_5seed_piv.columns, 2)]
C.check("abs_kendall_tau_n2", 0.83, round(sum(taus) / len(taus), 2), 0.01)

# ---------------------------------------------------------------- H10 ----
h10 = pd.read_csv(REPO_ROOT / "experiments/outputs/h10_baseline_R1.6/h10_baseline_results.csv")
h10_ok = h10[h10.status == "OK"]
random_names = [o for o in h10_ok.ordering.unique() if o.startswith("rand")]
h10_rand_by_ord = h10_ok[h10_ok.ordering.isin(random_names)].groupby("ordering")["err_mHa"].mean()

C.check("r1_h10_random_best", 168.67, round(h10_rand_by_ord.min(), 2), 0.02)
C.check("r1_h10_random_worst", 454.89, round(h10_rand_by_ord.max(), 2), 0.02)
C.check("r1_h10_random_median", 263.19, round(h10_rand_by_ord.median(), 2), 0.02)
C.check("abs_ordering_range_mha", 286.23, round(h10_rand_by_ord.max() - h10_rand_by_ord.min(), 2), 0.05)

h10_capture_by_ord = h10_ok[h10_ok.ordering.isin(random_names)].groupby("ordering")["captured"].mean()
rho_h10, p_h10 = stats.spearmanr(h10_capture_by_ord.values, h10_rand_by_ord.values)
C.check("mech_rho_h10", -0.983, round(rho_h10, 3), 0.003)
C.check("mech_p_h10", 3.6e-37, p_h10, p_h10 * 0.5 + 1e-38, f"order-of-magnitude check, p={p_h10:.2e}")

h10_piv = h10_ok.pivot_table(index="ordering", columns="seed", values="err_mHa").dropna()
between_h10 = h10_piv.mean(axis=1).var()
within_h10 = h10_piv.var(axis=1).mean()
ratio_h10 = between_h10 / within_h10 if within_h10 > 0 else float("inf")
C.check("abs_between_within_ratio_h10", 278.2, round(ratio_h10, 1), 2.0)

named_means = h10_ok[~h10_ok.ordering.isin(random_names)].groupby("ordering")["err_mHa"].mean()
C.check("r1_h10_identity_err", 300.32, round(named_means.get("identity", float("nan")), 2), 0.02,
        "SHIPPED CACHED reference -- see rebuilt_reference tolerance class")
C.check("r1_h10_physical_err", 389.71, round(named_means.get("physical", float("nan")), 2), 0.02)
C.check("r1_h10_physrev_err", 218.64, round(named_means.get("physical_reverse", float("nan")), 2), 0.02)
C.check("r1_h10_phys_physrev_gap", 171.07,
        round(named_means["physical"] - named_means["physical_reverse"], 2), 0.02)

# -------------------------------------------------------- anchor lever ---
c1 = pd.read_csv(REPO_ROOT / "experiments/outputs/anchor_decomposition_R1.6/c1_all120_identity.csv")
c1_ok = c1[c1.status.isna() | (c1.status == "OK")]
C.check("abs_anchor_range_mha", 234.10, round(c1_ok.err_mHa.max() - c1_ok.err_mHa.min(), 2), 0.05)

b1 = pd.read_csv(REPO_ROOT / "experiments/outputs/anchor_decomposition_R1.6/b1_offset_sweep.csv")
b1_physical = b1[b1.ordering == "physical"]
C.check("lever_offset_sweep_range", 171.08,
        round(b1_physical.err_mHa.max() - b1_physical.err_mHa.min(), 2), 0.02,
        "filtered to the physical chain, per the report's own text")

floor_count = int((c1_ok.err_mHa.round(2) == 458.70).sum())
C.check("floor_h10_identity_count", 17, floor_count, 0,
        "GENUINE DISCREPANCY, not a script bug: raw data shows 15 triples "
        "bit-identical at 458.699662 mHa plus 1 more (458.695209) rounding "
        "to 458.70 at 2dp = 16, not 17. See REPORT.md.")

# -------------------------------------------------------- compression ----
g1 = pd.read_csv(REPO_ROOT / "experiments/outputs/g1_lite/g1_summary.csv")
h12 = pd.read_csv(REPO_ROOT / "experiments/outputs/chain_aware/step2_b35_new12.csv")
g1["best_with_default"] = g1[["baseline", "best_of_40"]].min(axis=1)
h12["best_with_default"] = h12[["baseline", "best_of_43"]].min(axis=1)
combined = pd.concat([
    g1[["ordering", "baseline", "best_with_default"]],
    h12[["ordering", "baseline", "best_with_default"]],
], ignore_index=True)
default_spread = combined.baseline.max() - combined.baseline.min()
opt_spread = combined.best_with_default.max() - combined.best_with_default.min()
C.check("comp_factor", 3.09, round(default_spread / opt_spread, 2), 0.02)

# -------------------------------------------------------- S0 18-chain ----
cmp_tbl = pd.read_csv(REPO_ROOT / "experiments/outputs/chain_aware_v2/comparison_table.csv")
s0 = cmp_tbl[cmp_tbl.score == "S0"]
C.check("heldout_s0_correct_sign", 18, int((s0.rho_captured > 0).sum()), 0, f"of {len(s0)}")
C.check("heldout_s0_significant", 18, int((s0.p_captured < 0.05).sum()), 0, f"of {len(s0)}")
C.check("heldout_s0_median_rho", 0.703, round(s0.rho_captured.median(), 3), 0.001)
C.check("heldout_s0_worst_rho", 0.361, round(s0.rho_captured.min(), 3), 0.001)
C.check("heldout_argmax_median_regret", 0.347, round(s0.regret_frac.median(), 3), 0.001)
C.check("heldout_argmax_exceeding_1", 1, int((s0.regret_frac > 1.0).sum()), 0, f"of {len(s0)}")

# ------------------------------------------------------- shortlist rule --
p2 = pd.read_csv(REPO_ROOT / "experiments/outputs/chain_aware_v3/p2_shortlist.csv")
heldout = p2[(p2.group == "heldout") & (p2.k == 5)]
s0_5 = heldout[heldout.score == "S0"]
rand_5 = heldout[heldout.score == "__random__"]
C.check("abs_shortlist_median_regret", 0.002, round(float(s0_5.regret.median()), 3), 0.001)
C.check("abs_shortlist_p90_regret", 0.002, round(float(s0_5.regret_p90.median()), 3), 0.001)
C.check("shortlist_random_median", 0.181, round(float(rand_5.regret.median()), 3), 0.005)
C.check("shortlist_random_p90", 0.621, round(float(rand_5.regret_p90.median()), 3), 0.02)

# ------------------------------------------------------------ floor % ----
f1c = pd.read_csv(REPO_ROOT / "experiments/outputs/floor_generalization/f1c_floor_vs_default_50random.csv")
f1c_ok = f1c[f1c.status == "OK"]
worse = (f1c_ok.default_err > f1c_ok.floor_err).sum()
C.check("floor_50random_worse_than_floor_count_frac", 4.0, round(100 * worse / len(f1c_ok), 1), 0.5,
        f"{worse}/{len(f1c_ok)}")

import json, ast
lucj = pd.read_csv(REPO_ROOT / "experiments/outputs/lucj_control/identity_120_lucj.csv")
lucj_meta = json.loads((REPO_ROOT / "experiments/outputs/lucj_control/metadata.json").read_text())
noab_control = ast.literal_eval(lucj_meta["no_ab_control_err_lucj"])["identity"]
worse_lucj = int((lucj.err_lucj > noab_control).sum())
C.check("ansatz_worse_than_control_pct", 4.2, round(100 * worse_lucj / len(lucj), 1), 0.5,
        f"{worse_lucj}/{len(lucj)}, control={noab_control:.4f} mHa")

# -------------------------------------------------------- transpilation --
a1 = pd.read_csv(REPO_ROOT / "experiments/outputs/transpilation_audit/a1_by_triple.csv")
def cv(series):
    return 100 * series.std(ddof=0) / series.mean()

C.check("resource_anchor_cv_2q", 5.1, round(cv(a1.two_q_gates), 1), 0.2)
C.check("resource_anchor_cv_depth", 11.6, round(cv(a1.depth), 1), 0.3)
C.check("resource_anchor_cv_swap", 27.4, round(cv(a1.swap_count), 1), 1.0)

a2 = pd.read_csv(REPO_ROOT / "experiments/outputs/transpilation_audit/a2_by_chain.csv")
C.check("resource_samespin_cv_2q", 19.0, round(cv(a2.two_q_gates), 1), 0.5)
C.check("resource_samespin_cv_depth", 21.7, round(cv(a2.depth), 1), 0.5)
C.check("resource_samespin_cv_swap", 37.5, round(cv(a2.swap_count), 1), 1.5)

rho_gates_s0, _ = stats.spearmanr(a1.two_q_gates, a1.S0)
rho_gates_err, _ = stats.spearmanr(a1.two_q_gates, a1.err_sqd)
C.check("resource_rho_2q_s0", -0.709, round(rho_gates_s0, 3), 0.01)
C.check("resource_rho_2q_err", 0.608, round(rho_gates_err, 3), 0.01)

# --------------------------------------------------- headroom-normalised -
h10_cap_mean = h10_ok[h10_ok.ordering.isin(random_names)].captured.mean()
h10_ceiling = 0.7554  # ideal capture at budget, from cache/h10_R1.6 (tier 1 claim)
h10_headroom = h10_ceiling - h10_cap_mean
C.check("transfer_h10_range_per_headroom", 1445,
        round((c1_ok.err_mHa.max() - c1_ok.err_mHa.min()) / h10_headroom, -1), 30,
        f"headroom={h10_headroom:.4f}")

n2_cap_mean = n2_rand.captured.mean()
n2_ceiling = 0.9866
n2_headroom = n2_ceiling - n2_cap_mean
n2anchor = pd.read_csv(REPO_ROOT / "experiments/outputs/n2_anchor_axis/identity_120.csv")
n2_anchor_col = "err_sub_mHa" if "err_sub_mHa" in n2anchor.columns else "err_mHa"
n2_anchor_range = n2anchor[n2_anchor_col].max() - n2anchor[n2_anchor_col].min()
C.check("transfer_n2_range_per_headroom", 2550,
        round(n2_anchor_range / n2_headroom, -1), 100,
        f"headroom={n2_headroom:.4f}, anchor_range={n2_anchor_range:.2f}")

nfail = C.summary()
sys.exit(1 if nfail else 0)
