#!/usr/bin/env python3
"""
experiments/figures.py
=======================

Publication figure set for the report / final presentation. Analysis only:
reads cached CSVs already produced by prior experiments, does not sample,
call sbd, or recompute any reference. Writes results/figures/.

Every figure is emitted in two physical sizes (paper: 85mm single-column,
slide: 16:9) and two formats (vector PDF, 300dpi PNG) -- six figures x two
sizes x two formats = 24 image files, plus results/figures/README.md.

Data provenance note: five of the six figures draw only from
experiments/outputs/. Figures 1 (bottom panel) and 2 (panel A) need the
149-random-ordering N2 dataset; the only cached CSV at that exact sample
size is outputs/stage1/nonoracle_scores.csv (root-level outputs/, not
experiments/outputs/). It is read as-is -- no recomputation -- and the
exception is called out explicitly in results/figures/README.md.
"""
from __future__ import annotations

import csv
import sys
import warnings
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

# A handful of score variants (e.g. s2_os) saturate at a ceiling value for
# most orderings, so a minority of bootstrap resamples in figure 6 are
# constant by chance; nanpercentile in bootstrap_ci() already discards
# those draws, so the per-resample warning is expected noise, not a bug.
warnings.filterwarnings("ignore", category=ConstantInputWarning)

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "results" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Okabe-Ito colourblind-safe palette
# ----------------------------------------------------------------------
OI = dict(
    black="#000000",
    orange="#E69F00",
    sky_blue="#56B4E9",
    green="#009E73",
    yellow="#F0E442",
    blue="#0072B2",
    vermillion="#D55E00",
    purple="#CC79A7",
)

ORDER_COLOR = {
    "identity": OI["blue"],
    "physical": OI["vermillion"],
    "physical_reverse": OI["purple"],
    "s2_max": OI["orange"],
    "retainedJ_max": OI["sky_blue"],
    "rand007": OI["green"],
    "reverse": OI["yellow"],
    "max_captured_ORACLE": OI["vermillion"],
    "max_retainedJ": OI["green"],
}
RANDOM_COLOR = OI["black"]

MM_TO_IN = 1.0 / 25.4
PAPER_W = 85 * MM_TO_IN

STYLE = {
    "paper": dict(font=9, tick=8.5, lw=1.0, ms=10, marker_lw=1.0, dpi=300),
    "slide": dict(font=16, tick=14, lw=1.8, ms=28, marker_lw=1.8, dpi=300),
}


def rc(style):
    s = STYLE[style]
    return {
        "font.size": s["font"],
        "axes.labelsize": s["font"],
        "axes.titlesize": s["font"],
        "xtick.labelsize": s["tick"],
        "ytick.labelsize": s["tick"],
        "legend.fontsize": s["tick"],
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "svg.fonttype": "none",
    }


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name, style):
    s = STYLE[style]
    pdf = OUTDIR / f"{name}_{style}.pdf"
    png = OUTDIR / f"{name}_{style}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=s["dpi"])
    plt.close(fig)
    return [pdf.name, png.name]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


ALL_WRITTEN = []
CAPTIONS = []


def emit(fig, name, style):
    ALL_WRITTEN.extend(save(fig, name, style))


def caption(fig_id, text):
    CAPTIONS.append(f"[{fig_id}] {text}")


# ========================================================================
# Data loading
# ========================================================================

H10_SCORES = REPO / "experiments" / "outputs" / "score_audit_R1.6" / "all_scores.csv"
N2_SCORES = REPO / "outputs" / "stage1" / "nonoracle_scores.csv"
ANCHOR_REANALYSIS = REPO / "experiments" / "outputs" / "anchor_reanalysis" / "anchor_reanalysis.csv"
C1_ALL120 = REPO / "experiments" / "outputs" / "anchor_decomposition_R1.6" / "c1_all120_identity.csv"
B1_OFFSET = REPO / "experiments" / "outputs" / "anchor_decomposition_R1.6" / "b1_offset_sweep.csv"
F1B_NAMED = REPO / "experiments" / "outputs" / "floor_generalization" / "f1b_named_orderings.csv"


def h10_rows():
    return read_csv(H10_SCORES)


def h10_named_rand():
    rows = h10_rows()
    rand = [r for r in rows if r["ordering"].startswith("rand")]
    named = {r["ordering"]: r for r in rows if not r["ordering"].startswith("rand")}
    return rand, named


def n2_named_rand():
    rows = read_csv(N2_SCORES)
    rand = [r for r in rows if r["ordering"][0] == "r" and r["ordering"][1:].isdigit()]
    named = {r["ordering"]: r for r in rows if not (r["ordering"][0] == "r" and r["ordering"][1:].isdigit())}
    return rand, named


# ========================================================================
# FIGURE 1 -- the effect exists
# ========================================================================

def draw_fig1(style):
    s = STYLE[style]
    figsize = (PAPER_W, PAPER_W * 1.05) if style == "paper" else (10.0, 5.625)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, layout="constrained")

    rng = np.random.default_rng(0)

    def strip(ax, rand_vals, named_vals, xlabel, title):
        xspan = max(rand_vals) - min(rand_vals)
        y = rng.uniform(-0.12, 0.35, size=len(rand_vals))
        ax.scatter(rand_vals, y, s=s["ms"] * 0.5, c=RANDOM_COLOR, alpha=0.35,
                   linewidths=0, zorder=2)
        med = float(np.median(rand_vals))
        ax.axvline(med, color=RANDOM_COLOR, lw=s["lw"] * 0.8, ls=":", zorder=1)
        ax.text(0.02, 0.0, f"random median = {med:.1f} mHa",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=s["tick"], color=RANDOM_COLOR)

        # merge named orderings that land on (near-)identical values so
        # markers/labels never sit exactly on top of one another
        groups = []
        for label, val in sorted(named_vals.items(), key=lambda kv: kv[1]):
            if groups and abs(val - groups[-1][1]) < 1e-6:
                groups[-1] = (groups[-1][0] + " = " + label, groups[-1][1])
            else:
                groups.append((label, val))

        # greedily assign each item to one of a few y-tiers, each time
        # picking the tier whose most-recent x is farthest from this x, to
        # keep near-x items from landing in visually adjacent tiers
        n_tiers = min(4, len(groups)) if groups else 1
        tier_y = np.linspace(0.55, 1.55, n_tiers) if n_tiers > 1 else [1.0]
        tier_last_x = [None] * n_tiers
        for label, val in groups:
            if any(t is None for t in tier_last_x):
                ti = next(i for i, t in enumerate(tier_last_x) if t is None)
            else:
                ti = int(np.argmax([abs(val - t) for t in tier_last_x]))
            tier_last_x[ti] = val
            c = ORDER_COLOR.get(label.split(" = ")[0], OI["black"])
            ax.scatter([val], [tier_y[ti]], s=s["ms"] * 3.2, c=c, marker="D",
                       edgecolors="white", linewidths=s["marker_lw"] * 0.5, zorder=3)
            ax.plot([val, val], [0.35, tier_y[ti] - 0.12], color=c,
                    lw=s["lw"] * 0.5, alpha=0.5, zorder=1)
            ax.annotate(label, xy=(val, tier_y[ti]), xytext=(4, 0),
                        textcoords="offset points",
                        fontsize=s["tick"], color=c, va="center", ha="left")

        ax.set_yticks([])
        ax.set_ylim(-0.55, 2.05)
        ax.set_xlim(min(rand_vals) - 0.06 * xspan, max(rand_vals) + 0.34 * xspan)
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=s["font"])
        return med

    h10_rand, h10_named = h10_named_rand()
    h10_wanted = ["identity", "physical", "physical_reverse", "s2_max", "retainedJ_max"]
    h10_named_use = {k: float(h10_named[k]["err_mHa"]) for k in h10_wanted if k in h10_named}
    h10_rand_vals = [float(r["err_mHa"]) for r in h10_rand]
    strip(ax1, h10_rand_vals, h10_named_use, "SQD subspace error (mHa)",
          f"H10, CAS(10,10) -- {len(h10_rand_vals)} random orderings")

    n2_rand, n2_named = n2_named_rand()
    n2_wanted = ["identity", "reverse", "max_captured_ORACLE", "max_retainedJ"]
    n2_named_use = {k: float(n2_named[k]["err_mHa"]) for k in n2_wanted if k in n2_named}
    n2_rand_vals = [float(r["err_mHa"]) for r in n2_rand]
    strip(ax2, n2_rand_vals, n2_named_use, "SQD subspace error (mHa)",
          f"N2, CAS(6,10) -- {len(n2_rand_vals)} random orderings")

    return fig, len(h10_rand_vals), len(n2_rand_vals), np.median(h10_rand_vals), np.median(n2_rand_vals)


def figure1():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, n_h10, n_n2, med_h10, med_n2 = draw_fig1(style)
            emit(fig, "fig1_effect_exists", style)
    caption(
        "Figure 1",
        f"The subspace-error effect exists on both search axes. Top: err_mHa for "
        f"n={n_h10} random same-spin orderings of H10 CAS(10,10) at fixed default "
        f"anchors (random median {med_h10:.1f} mHa), with identity, physical, "
        f"physical_reverse, s2_max and retainedJ_max marked. Bottom: n={n_n2} random "
        f"orderings of N2 CAS(6,10) (random median {med_n2:.1f} mHa), with identity, "
        f"reverse, max_captured_ORACLE and max_retainedJ marked. Panels use "
        f"independent x-scales. Source: score_audit_R1.6/all_scores.csv (H10), "
        f"outputs/stage1/nonoracle_scores.csv (N2)."
    )


# ========================================================================
# FIGURE 2 -- the mechanism
# ========================================================================

def scatter_panel(ax, x, y, s, color, label=None, fit=False):
    ax.scatter(x, y, s=s["ms"], c=color, alpha=0.55, linewidths=0, zorder=2)
    rho, p = spearmanr(x, y)
    txt = f"$\\rho$={rho:+.3f}\np={p:.1e}\nn={len(x)}"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=s["tick"])
    return rho, p


def draw_fig2(style):
    s = STYLE[style]
    n2_rand, _ = n2_named_rand()
    n2_cap = [float(r["captured"]) for r in n2_rand]
    n2_err = [float(r["err_mHa"]) for r in n2_rand]

    h10_rand, _ = h10_named_rand()
    h10_cap = [float(r["captured"]) for r in h10_rand]
    h10_err = [float(r["err_mHa"]) for r in h10_rand]

    anchor_rows = read_csv(ANCHOR_REANALYSIS)
    per_ord = {}
    for ordn in ("identity", "physical", "rand007"):
        sub = [r for r in anchor_rows if r["ordering"] == ordn]
        per_ord[ordn] = ([float(r["captured"]) for r in sub], [float(r["err_mHa"]) for r in sub])

    if style == "paper":
        # A and B stacked full-width; C as one overlaid scatter (small
        # multiples do not fit legibly at 85mm with three extra axes).
        figsize = (PAPER_W, PAPER_W * 2.35)
        fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=figsize, layout="constrained")

        rhoA, pA = scatter_panel(axA, n2_cap, n2_err, s, OI["blue"])
        axA.set_xlabel("captured weight")
        axA.set_ylabel("SQD subspace error (mHa)")
        axA.set_title("A -- N2, CAS(6,10)", loc="left", fontsize=s["font"])

        rhoB, pB = scatter_panel(axB, h10_cap, h10_err, s, OI["vermillion"])
        axB.set_xlabel("captured weight")
        axB.set_ylabel("SQD subspace error (mHa)")
        axB.set_title("B -- H10, CAS(10,10)", loc="left", fontsize=s["font"])

        rhosC = {}
        y0 = 0.97
        for i, ordn in enumerate(("identity", "physical", "rand007")):
            cap, err = per_ord[ordn]
            c = ORDER_COLOR[ordn]
            axC.scatter(cap, err, s=s["ms"], c=c, alpha=0.6, linewidths=0)
            rho, p = spearmanr(cap, err)
            rhosC[ordn] = (rho, p, len(cap))
            axC.text(0.97, y0 - i * 0.10, f"{ordn}: $\\rho$={rho:+.3f}",
                     transform=axC.transAxes, ha="right", va="top",
                     fontsize=s["tick"], color=c)
        axC.set_xlabel("captured weight")
        axC.set_ylabel("SQD subspace error (mHa)")
        axC.set_title("C -- anchor-selection axis\n(3 fixed orderings, n=40 each)",
                      loc="left", fontsize=s["font"])
        for a in (axA, axB, axC):
            clean(a)
        return fig, (rhoA, pA, len(n2_cap)), (rhoB, pB, len(h10_cap)), rhosC

    figsize = (12.2, 6.2)
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.08, wspace=0.5)
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2])
    axC1 = fig.add_subplot(gs[1, 0])
    axC2 = fig.add_subplot(gs[1, 1], sharey=axC1)
    axC3 = fig.add_subplot(gs[1, 2], sharey=axC1)

    rhoA, pA = scatter_panel(axA, n2_cap, n2_err, s, OI["blue"])
    axA.set_xlabel("captured weight")
    axA.set_ylabel("SQD subspace error (mHa)")
    axA.set_title("A -- N2, CAS(6,10)", loc="left", fontsize=s["font"])

    rhoB, pB = scatter_panel(axB, h10_cap, h10_err, s, OI["vermillion"])
    axB.set_xlabel("captured weight")
    axB.set_title("B -- H10, CAS(10,10)", loc="left", fontsize=s["font"])

    rhosC = {}
    panel_titles = {"identity": "C -- identity", "physical": "physical", "rand007": "rand007"}
    for ax, ordn in zip((axC1, axC2, axC3), ("identity", "physical", "rand007")):
        cap, err = per_ord[ordn]
        rho, p = scatter_panel(ax, cap, err, s, ORDER_COLOR[ordn])
        rhosC[ordn] = (rho, p, len(cap))
        ax.set_xlabel("captured weight")
        ax.set_title(panel_titles[ordn], loc="left", fontsize=s["font"], color=ORDER_COLOR[ordn])
    axC1.set_ylabel("err (mHa)")
    plt.setp(axC2.get_yticklabels(), visible=False)
    plt.setp(axC3.get_yticklabels(), visible=False)

    for a in (axA, axB, axC1, axC2, axC3):
        clean(a)

    return fig, (rhoA, pA, len(n2_cap)), (rhoB, pB, len(h10_cap)), rhosC


def figure2():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, A, B, C = draw_fig2(style)
            emit(fig, "fig2_mechanism", style)
    caption(
        "Figure 2",
        f"Captured weight predicts subspace error on both search axes, both "
        f"systems. A: N2 same-spin ordering axis, n={A[2]}, rho={A[0]:+.3f} "
        f"(p={A[1]:.1e}). B: H10 same-spin ordering axis, n={B[2]}, "
        f"rho={B[0]:+.3f} (p={B[1]:.1e}). C: H10 anchor-selection axis at three "
        f"fixed orderings (n=40 triples each): identity rho={C['identity'][0]:+.3f} "
        f"(p={C['identity'][1]:.1e}), physical rho={C['physical'][0]:+.3f} "
        f"(p={C['physical'][1]:.1e}), rand007 rho={C['rand007'][0]:+.3f} "
        f"(p={C['rand007'][1]:.1e}). Source: outputs/stage1/nonoracle_scores.csv (A), "
        f"score_audit_R1.6/all_scores.csv (B), anchor_reanalysis/anchor_reanalysis.csv (C)."
    )


# ========================================================================
# FIGURE 3 -- two levers
# ========================================================================

def figure3_data():
    h10_rand, _ = h10_named_rand()
    err = [float(r["err_mHa"]) for r in h10_rand]
    span_ordering = max(err) - min(err)

    rows = read_csv(B1_OFFSET)
    phys = [r for r in rows if r["ordering"] == "physical"]
    off_err = [float(r["err_mHa"]) for r in phys]
    span_offset = max(off_err) - min(off_err)

    c1 = read_csv(C1_ALL120)
    c1_err = [float(r["err_mHa"]) for r in c1]
    span_anchor = max(c1_err) - min(c1_err)

    return [
        ("same-spin ordering", span_ordering, "10! = 3,628,800", len(err)),
        ("free anchor selection", span_anchor, "C(10,3) = 120", len(c1)),
        ("anchor offset", span_offset, "4", len(off_err)),
    ]


def draw_fig3(style):
    s = STYLE[style]
    figsize = (PAPER_W, PAPER_W * 0.85) if style == "paper" else (10.0, 5.0)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    data = sorted(figure3_data(), key=lambda d: -d[1])
    labels = [d[0] for d in data]
    spans = [d[1] for d in data]
    colors = [OI["blue"], OI["vermillion"], OI["green"]]

    short_space = {"10! = 3,628,800": "10!", "C(10,3) = 120": "C(10,3)", "4": "4"}

    y = np.arange(len(labels))[::-1]
    ax.barh(y, spans, color=colors, height=0.55)
    xlim = max(spans) * (1.55 if style == "paper" else 1.75)
    for yy, d in zip(y, data):
        label, span, space, n = d
        space_txt = short_space[space] if style == "paper" else space
        ax.text(span + max(spans) * 0.02, yy,
                f"{span:.2f} mHa\n({space_txt})" if style == "paper" else
                f"{span:.2f} mHa  (search space: {space_txt}, n={n})",
                va="center", ha="left", fontsize=s["tick"], linespacing=1.15)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("err_mHa span (max − min)")
    ax.set_xlim(0, xlim)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    return fig, data


def figure3():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, data = draw_fig3(style)
            emit(fig, "fig3_two_levers", style)
    parts = "; ".join(f"{d[0].split(chr(10))[0]}: span={d[1]:.2f} mHa over {d[2]} (n={d[3]})" for d in data)
    caption(
        "Figure 3",
        f"A tiny search space buys a large error span. {parts}. All spans on H10 "
        f"CAS(10,10) at the fixed default 225-dim budget. Source: "
        f"score_audit_R1.6/all_scores.csv (ordering lever), "
        f"anchor_decomposition_R1.6/c1_all120_identity.csv (free anchor lever), "
        f"anchor_decomposition_R1.6/b1_offset_sweep.csv (anchor-offset lever, "
        f"physical ordering)."
    )


# ========================================================================
# FIGURE 4 -- the anchor rule and its limits
# ========================================================================

FIG4_ORDERINGS = ["identity", "rand007", "physical"]
FIG4_FRAC_REGRET = {"identity": 0.217, "rand007": 0.513, "physical": 0.696}


def draw_fig4(style):
    s = STYLE[style]
    figsize = (PAPER_W, PAPER_W * 1.55) if style == "paper" else (12.0, 4.6)
    if style == "paper":
        fig, axes = plt.subplots(3, 1, figsize=figsize, layout="constrained")
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")

    rows = read_csv(ANCHOR_REANALYSIS)
    results = {}
    for ax, ordn in zip(axes, FIG4_ORDERINGS):
        sub = [r for r in rows if r["ordering"] == ordn]
        x = np.array([float(r["retained_J_oppspin"]) for r in sub])
        y = np.array([float(r["err_mHa"]) for r in sub])
        c = ORDER_COLOR[ordn]
        ax.scatter(x, y, s=s["ms"], c=c, alpha=0.6, linewidths=0, zorder=2)

        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color=c, lw=s["lw"], ls="--", zorder=1)

        i_top = int(np.argmax(x))
        i_best = int(np.argmin(y))
        ax.scatter([x[i_top]], [y[i_top]], marker="^", s=s["ms"] * 3.5,
                   facecolors="none", edgecolors=OI["black"], linewidths=s["marker_lw"], zorder=3)
        ax.scatter([x[i_best]], [y[i_best]], marker="*", s=s["ms"] * 5,
                   facecolors=OI["yellow"], edgecolors=OI["black"], linewidths=s["marker_lw"] * 0.6, zorder=4)

        rho, p = spearmanr(x, y)
        frac = FIG4_FRAC_REGRET[ordn]
        ax.text(0.97, 0.95,
                f"$\\rho$={rho:+.3f}\np={p:.1e}\nregret={frac:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=s["tick"])
        ax.set_title(ordn, loc="left", fontsize=s["font"], color=c)
        ax.set_xlabel("retained_J_oppspin")
        clean(ax)
        results[ordn] = (rho, p, frac, len(sub))
    axes[0].set_ylabel("SQD subspace error (mHa)")

    handles = [
        mlines.Line2D([], [], marker="^", color=OI["black"], linestyle="none",
                      markerfacecolor="none", markersize=7, label="rule's top pick"),
        mlines.Line2D([], [], marker="*", color=OI["black"], linestyle="none",
                      markerfacecolor=OI["yellow"], markersize=9, label="true best"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False,
               fontsize=s["tick"])
    return fig, results


def figure4():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, results = draw_fig4(style)
            emit(fig, "fig4_anchor_rule_limits", style)
    parts = "; ".join(
        f"{o} rho={results[o][0]:+.3f} (p={results[o][1]:.1e}), fractional regret={results[o][2]:.3f}"
        for o in FIG4_ORDERINGS
    )
    n = results["identity"][3]
    caption(
        "Figure 4",
        f"The retained_J_oppspin anchor-selection rule degrades as the ordering "
        f"moves away from identity. Each panel: n={n} anchor triples at a fixed "
        f"ordering, dashed line is a linear trend, triangle marks the rule's top "
        f"pick (max retained_J_oppspin), star marks the true best (min err_mHa). "
        f"{parts}. Source: anchor_reanalysis/anchor_reanalysis.csv."
    )


# ========================================================================
# FIGURE 5 -- the no-alpha-beta floor
# ========================================================================

def draw_fig5(style):
    s = STYLE[style]
    figsize = (PAPER_W, PAPER_W * 0.95) if style == "paper" else (10.0, 5.625)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    rows = {r["ordering"]: r for r in read_csv(F1B_NAMED)}
    orderings = ["identity", "rand007", "physical"]
    bar_labels = ["floor\n(no opp-spin)", "default\n(p%4==0 anchors)", "best\n(anchor search)"]
    bar_colors = [OI["black"], OI["orange"], OI["blue"]]

    group_w = 0.75
    bw = group_w / 3
    for gi, ordn in enumerate(orderings):
        r = rows[ordn]
        floor = float(r["floor"])
        default = float(r["default"])
        best = float(r["best"])
        vals = [floor, default, best]
        xs = [gi - group_w / 2 + bw * (k + 0.5) for k in range(3)]
        ax.bar(xs, vals, width=bw * 0.92, color=bar_colors)
        ax.hlines(floor, gi - group_w / 2 - 0.03, gi + group_w / 2 + 0.03,
                  color=OI["black"], lw=s["lw"], ls="--", zorder=3)
        if default > floor:
            gap = default - floor
            ax.annotate(f"+{gap:.2f} mHa", xy=(xs[1], (floor + default) / 2),
                        xytext=(xs[1] + bw * 1.15, (floor + default) / 2),
                        fontsize=s["tick"], color=OI["vermillion"], va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=OI["vermillion"], lw=s["lw"] * 0.8))

    ax.set_xticks(range(len(orderings)))
    ax.set_xticklabels(orderings)
    ax.set_ylabel("SQD subspace error (mHa)")
    clean(ax)

    handles = [mlines.Line2D([0], [0], color=c, lw=6, label=l.split("\n")[0])
               for c, l in zip(bar_colors, bar_labels)]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
               fontsize=s["tick"])
    return fig, rows


def figure5():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, rows = draw_fig5(style)
            emit(fig, "fig5_floor_and_limits", style)
    penalty = float(rows["physical"]["default"]) - float(rows["physical"]["floor"])
    caption(
        "Figure 5",
        f"The opposite-spin coupling mask is not always a net benefit. Grouped "
        f"bars per ordering (n=3: identity, rand007, physical) show floor "
        f"(same-spin only), default (p%4==0 anchors) and best (anchor search) "
        f"err_mHa; the dashed line spans each group's own floor. At physical "
        f"the default sits above its own floor by {penalty:+.2f} mHa -- the "
        f"opposite-spin mask actively hurts there. Source: "
        f"floor_generalization/f1b_named_orderings.csv."
    )


# ========================================================================
# FIGURE 6 -- the null results
# ========================================================================

VARIANTS = ["s1_amp", "s1_ampJ", "s1_amp_ss", "s1_amp_os", "s1_ampJ_ss",
            "s1_ampJ_os", "s2", "s2_ss", "s2_os", "s2_soft_ss", "retained_J"]

VARIANT_LABEL = {
    "s1_amp": "s1_amp",
    "s1_ampJ": "s1_ampJ",
    "s1_amp_ss": "s1_amp_ss",
    "s1_amp_os": "s1_amp_os",
    "s1_ampJ_ss": "s1_ampJ_ss",
    "s1_ampJ_os": "s1_ampJ_os",
    "s2": "s2",
    "s2_ss": "s2_ss",
    "s2_os": "s2_os",
    "s2_soft_ss": "s2_soft_ss",
    "retained_J": "retained_J",
    "retained_J_samespin": "  ↳ same-spin",
    "retained_J_oppspin": "  ↳ opp-spin",
}


def bootstrap_ci(x, y, seed=0, n_resamples=2000):
    # Manual paired bootstrap (not scipy.stats.bootstrap): several score
    # variants are saturated at a ceiling value for most orderings (e.g.
    # s2_os == 1.0 for 44/50 rows), so a nontrivial fraction of resamples
    # are constant and spearmanr returns NaN for them. nanpercentile drops
    # those degenerate resamples instead of propagating NaN into the CI.
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore"):
            stats[i] = spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def figure6_data():
    rand, _ = h10_named_rand()
    err = np.array([float(r["err_mHa"]) for r in rand])
    results = {}
    for v in VARIANTS + ["retained_J_samespin", "retained_J_oppspin"]:
        x = np.array([float(r[v]) for r in rand])
        rho, p = spearmanr(x, err)
        lo, hi = bootstrap_ci(x, err, seed=zlib.crc32(v.encode()))
        results[v] = (rho, p, lo, hi)
    return results, len(rand)


def draw_fig6(style):
    s = STYLE[style]
    results, n = figure6_data()

    order = sorted([v for v in VARIANTS if v != "retained_J"],
                    key=lambda v: -abs(results[v][0]))
    rj_pos = min(len(order), sum(1 for v in order if abs(results[v][0]) > abs(results["retained_J"][0])))
    rows = order[:rj_pos] + ["retained_J", "retained_J_samespin", "retained_J_oppspin"] + order[rj_pos:]

    figsize = (PAPER_W, PAPER_W * 1.55) if style == "paper" else (10.0, 6.5)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    y = np.arange(len(rows))[::-1]
    for yy, v in zip(y, rows):
        rho, p, lo, hi = results[v]
        is_decomp = v in ("retained_J_samespin", "retained_J_oppspin")
        c = OI["orange"] if is_decomp else OI["blue"]
        ax.plot([lo, hi], [yy, yy], color=c, lw=s["lw"] * 1.3, zorder=1)
        ax.scatter([rho], [yy], s=s["ms"] * 1.6, c=c, zorder=2)

    ax.axvline(0, color=OI["black"], lw=s["lw"] * 0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_LABEL[v] for v in rows])
    ax.set_xlabel("Spearman $\\rho$ vs err_mHa")
    ax.set_title(f"H10, n={n} random orderings (95% CI)", loc="left", fontsize=s["tick"])
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    return fig, rows, results, n


def figure6():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, rows, results, n = draw_fig6(style)
            emit(fig, "fig6_null_results", style)
    rj = results["retained_J"]
    ss = results["retained_J_samespin"]
    os_ = results["retained_J_oppspin"]
    caption(
        "Figure 6",
        f"Eleven score1/score2/retained_J variants (n={n} random H10 orderings), "
        f"Spearman rho vs err_mHa with bootstrap 95% CI, ordered by |rho|; none "
        f"reach the predictive bar reliably except captured itself (Figure 2), "
        f"which is diagnostic-only since it requires the sampling it would "
        f"replace. retained_J's near-zero combined effect (rho={rj[0]:+.3f}) is "
        f"a sign cancellation of its same-spin (rho={ss[0]:+.3f}) and "
        f"opposite-spin (rho={os_[0]:+.3f}) components, shown as adjacent rows. "
        f"Source: score_audit_R1.6/all_scores.csv."
    )


# ========================================================================
# README + main
# ========================================================================

README = """\
# results/figures/ -- regeneration index

Generated by `experiments/figures.py` (analysis only -- no sampling, no sbd
calls, no new reference computation). Each figure below can be regenerated
independently given the listed CSV(s) and columns.

## Figure 1 -- the effect exists
- `experiments/outputs/score_audit_R1.6/all_scores.csv`
  columns: `ordering`, `err_mHa` (top panel, H10: 50 random `rand###`
  orderings, named `identity`/`physical`/`physical_reverse`/`s2_max`/
  `retainedJ_max`)
- `outputs/stage1/nonoracle_scores.csv`
  columns: `ordering`, `err_mHa` (bottom panel, N2: 149 random `r###`
  orderings, named `identity`/`reverse`/`max_captured_ORACLE`/
  `max_retainedJ`). NOTE: this file lives outside `experiments/outputs/` --
  it is the only cached CSV at the required 149-random-ordering sample size.

## Figure 2 -- the mechanism
- Panel A (N2, n=149): `outputs/stage1/nonoracle_scores.csv`,
  columns `captured`, `err_mHa` (random `r###` rows only)
- Panel B (H10, n=50): `experiments/outputs/score_audit_R1.6/all_scores.csv`,
  columns `captured`, `err_mHa` (random `rand###` rows only)
- Panel C (3x n=40): `experiments/outputs/anchor_reanalysis/anchor_reanalysis.csv`,
  columns `ordering`, `captured`, `err_mHa`, filtered to
  `identity`/`physical`/`rand007`

## Figure 3 -- two levers
- Same-spin ordering lever: `experiments/outputs/score_audit_R1.6/all_scores.csv`,
  column `err_mHa` (50 random `rand###` rows)
- Free anchor-selection lever: `experiments/outputs/anchor_decomposition_R1.6/c1_all120_identity.csv`,
  column `err_mHa` (all 120 rows)
- Anchor-offset lever: `experiments/outputs/anchor_decomposition_R1.6/b1_offset_sweep.csv`,
  columns `ordering`, `anchor_offset`, `err_mHa`, filtered to `ordering=='physical'`

## Figure 4 -- the anchor rule and its limits
- `experiments/outputs/anchor_reanalysis/anchor_reanalysis.csv`,
  columns `ordering`, `triple`, `retained_J_oppspin`, `err_mHa`, filtered to
  `identity`/`rand007`/`physical` (n=40 each). Fractional regret values
  (0.217 / 0.513 / 0.696) are taken from
  `experiments/outputs/anchor_reanalysis/report.txt` (section D2).

## Figure 5 -- the no-alpha-beta floor
- `experiments/outputs/floor_generalization/f1b_named_orderings.csv`,
  columns `ordering`, `floor`, `default`, `best`, for
  `identity`/`rand007`/`physical`

## Figure 6 -- the null results
- `experiments/outputs/score_audit_R1.6/all_scores.csv`,
  columns `err_mHa` and the eleven score-variant columns: `s1_amp`,
  `s1_ampJ`, `s1_amp_ss`, `s1_amp_os`, `s1_ampJ_ss`, `s1_ampJ_os`, `s2`,
  `s2_ss`, `s2_os`, `s2_soft_ss`, `retained_J` (plus its decomposition
  `retained_J_samespin` / `retained_J_oppspin`), computed over the 50 random
  `rand###` H10 orderings. Spearman rho is exact; 95% CIs are a paired
  bootstrap (2000 resamples, `scipy.stats.bootstrap`, method='basic') over
  those same 50 rows, recomputed by `figures.py` at generation time (not
  cached).

## Regenerating
```
python3 experiments/figures.py
```
Outputs `<name>_paper.pdf` / `.png` (85mm width) and `<name>_slide.pdf` /
`.png` (16:9) per figure into this directory, plus prints all six suggested
captions to stdout.
"""


def main():
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()

    (OUTDIR / "README.md").write_text(README)

    print(f"\nWrote {len(ALL_WRITTEN)} figure files to {OUTDIR}:")
    for name in sorted(ALL_WRITTEN):
        print(f"  {name}")

    print("\nSuggested captions:\n")
    for c in CAPTIONS:
        print(c)
        print()


if __name__ == "__main__":
    sys.exit(main())
