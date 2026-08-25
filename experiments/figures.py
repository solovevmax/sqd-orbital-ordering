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

Every render is checked before it is written: check_clipping() draws the
figure at its final save dpi and flags any Text/Legend artist whose window
extent falls outside the canvas, and save() re-opens the written PNG with
PIL to confirm its pixel dimensions match the requested physical size. Both
are reported per-file at the end of main().
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
import matplotlib.patches as mpatches
import matplotlib.text as mtext
import matplotlib.legend as mlegend
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
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

# Canonical ordering-name vocabulary, used verbatim everywhere an ordering
# is named on a figure: identity, reverse, physical, physical_reverse,
# s1_max, s2_max, retainedJ_max, rand007. Some cached CSVs spell these
# differently (e.g. N2's "max_retainedJ"); RENAME_ORDERING maps those onto
# the canonical spelling before anything is drawn or printed.
RENAME_ORDERING = {"max_retainedJ": "retainedJ_max"}

ORDER_COLOR = {
    "identity": OI["blue"],
    "physical": OI["vermillion"],
    "physical_reverse": OI["purple"],
    "s1_max": OI["yellow"],
    "s2_max": OI["orange"],
    "retainedJ_max": OI["sky_blue"],
    "rand007": OI["green"],
    "reverse": OI["black"],
    "max_captured_ORACLE": OI["vermillion"],
}
RANDOM_COLOR = OI["black"]

# retained_J's readable name is kept as "retained_J" throughout (an
# established symbol in this project's own vocabulary, on a par with "s1"
# and "s2"), per the explicit figure-6 labelling request; G1's "no column
# names" rule targets ad hoc identifiers like err_mHa, which do get a
# plain-English replacement everywhere below.
RETAINED_J_OS_LABEL = "retained_J (opposite-spin only)"

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


# ----------------------------------------------------------------------
# Render verification (G3 / G5): clipping check + PNG dimension read-back
# ----------------------------------------------------------------------

RENDER_REPORT = []


def check_clipping(fig, dpi):
    """Draw fig at `dpi` and return text of any Text/Legend artist whose
    window extent falls outside the canvas -- i.e. would be visibly cut
    off in the saved raster. Not a check against the axes box: titles,
    tick labels and axis labels legitimately sit outside the axes but
    must still land inside the canvas."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w_px, h_px = fig.canvas.get_width_height()
    bad = []
    for artist in fig.findobj():
        if isinstance(artist, mtext.Text):
            # matplotlib's tick-label pooling leaves stale, detached Text
            # instances behind after a locator recompute (axes=None, not
            # a figure-level text either) -- get_visible() still reports
            # True on these, but they are never actually drawn. Only text
            # still attached to an axes, or a genuine figure-level text
            # (fig.text / suptitle), is a real candidate for clipping.
            attached = artist.axes is not None or artist in fig.texts
            if not attached or not artist.get_visible():
                continue
            txt = artist.get_text()
            if not txt or not txt.strip():
                continue
            try:
                bbox = artist.get_window_extent(renderer)
            except Exception:
                continue
            if bbox.width == 0 and bbox.height == 0:
                continue
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > w_px + 1 or bbox.y1 > h_px + 1:
                bad.append(txt.replace("\n", " / "))
        elif isinstance(artist, mlegend.Legend):
            try:
                bbox = artist.get_window_extent(renderer)
            except Exception:
                continue
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > w_px + 1 or bbox.y1 > h_px + 1:
                bad.append("<legend>")
    return bad, (w_px, h_px)


def save(fig, name, style):
    s = STYLE[style]
    pdf = OUTDIR / f"{name}_{style}.pdf"
    png = OUTDIR / f"{name}_{style}.png"

    # Save first (this is what a reader actually sees), then run the
    # clipping check as the last thing done on this figure, at the same
    # dpi as the PNG save -- constrained_layout can shift slightly between
    # successive draws at different dpi, so checking *before* saving can
    # flag a transient layout that was never actually written to disk.
    w_in, h_in = fig.get_size_inches()
    fig.savefig(pdf)
    fig.savefig(png, dpi=s["dpi"])
    clipped, (w_px, h_px) = check_clipping(fig, dpi=s["dpi"])
    plt.close(fig)

    with Image.open(png) as im:
        actual_px = im.size
    expected_px = (round(w_in * s["dpi"]), round(h_in * s["dpi"]))

    RENDER_REPORT.append(dict(
        name=png.name, actual_px=actual_px, expected_px=expected_px, clipped=clipped,
    ))
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


def h10_named_rand():
    rows = read_csv(H10_SCORES)
    rand = [r for r in rows if r["ordering"].startswith("rand")]
    named = {RENAME_ORDERING.get(r["ordering"], r["ordering"]): r
             for r in rows if not r["ordering"].startswith("rand")}
    return rand, named


def n2_named_rand():
    rows = read_csv(N2_SCORES)
    rand = [r for r in rows if r["ordering"][0] == "r" and r["ordering"][1:].isdigit()]
    named = {RENAME_ORDERING.get(r["ordering"], r["ordering"]): r
             for r in rows if not (r["ordering"][0] == "r" and r["ordering"][1:].isdigit())}
    return rand, named


# ========================================================================
# FIGURE 1 -- the effect exists
# ========================================================================

def _label_groups(named_vals):
    """Merge named orderings landing on (near-)identical values, so two
    markers never sit exactly on top of one another, and return groups
    sorted by value ascending."""
    groups = []
    for label, val in sorted(named_vals.items(), key=lambda kv: kv[1]):
        if groups and abs(val - groups[-1][1]) < 1e-6:
            groups[-1] = (groups[-1][0] + " = " + label, groups[-1][1])
        else:
            groups.append((label, val))
    return groups


def draw_fig1(style):
    s = STYLE[style]

    h10_rand, h10_named = h10_named_rand()
    h10_wanted = ["identity", "physical", "physical_reverse", "s1_max", "s2_max", "retainedJ_max"]
    h10_named_use = {k: float(h10_named[k]["err_mHa"]) for k in h10_wanted if k in h10_named}
    h10_rand_vals = [float(r["err_mHa"]) for r in h10_rand]
    h10_groups = _label_groups(h10_named_use)

    n2_rand, n2_named = n2_named_rand()
    n2_wanted = ["identity", "reverse", "max_captured_ORACLE", "retainedJ_max"]
    n2_named_use = {k: float(n2_named[k]["err_mHa"]) for k in n2_wanted if k in n2_named}
    n2_rand_vals = [float(r["err_mHa"]) for r in n2_rand]
    n2_groups = _label_groups(n2_named_use)

    # every named ordering gets its own label row (a "staircase"), so
    # near-coincident values never collide -- unlike a fixed small number
    # of tiers, this scales with however many names a panel needs. row_h
    # is in data-y units (used to place rows within each axes); row_in /
    # overhead_in are physical inches (used to size the figure so however
    # many rows a panel needs, it actually has room to render them).
    row_h = 0.34
    base_y = 0.55
    row_in, overhead_in, width_in = (0.145, 1.05, PAPER_W) if style == "paper" \
        else (0.30, 1.9, 11.5)

    def panel_rows(n_groups):
        return base_y + row_h * (n_groups - 1) + 0.75  # + bracket + headroom

    h_ratio = panel_rows(len(h10_groups))
    n_ratio = panel_rows(len(n2_groups))
    h_in = overhead_in + row_in * len(h10_groups)
    n_in = overhead_in + row_in * len(n2_groups)
    figsize = (width_in, h_in + n_in)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, layout="constrained",
        gridspec_kw=dict(height_ratios=[h_ratio, n_ratio]),
    )

    rng = np.random.default_rng(0)

    def strip(ax, rand_vals, groups, xlabel, title, fold_label):
        vmin, vmax = min(rand_vals), max(rand_vals)
        xspan = vmax - vmin
        n_rows = len(groups)
        top_row_y = base_y + row_h * (n_rows - 1)
        bracket_y = top_row_y + 0.5

        # shaded band across the full random min-to-max range, the
        # headline number made visually legible without arithmetic
        ax.axvspan(vmin, vmax, color=OI["black"], alpha=0.06, zorder=0)
        fold = vmax / vmin
        ax.annotate("", xy=(vmax, bracket_y), xytext=(vmin, bracket_y),
                    arrowprops=dict(arrowstyle="<->", color=RANDOM_COLOR, lw=s["lw"]))
        ax.text((vmin + vmax) / 2, bracket_y + 0.14, f"{fold:.1f}$\\times$",
                ha="center", va="bottom", fontsize=s["font"], color=RANDOM_COLOR,
                fontweight="bold")

        y_pts = rng.uniform(-0.20, 0.20, size=len(rand_vals))
        ax.scatter(rand_vals, y_pts, s=s["ms"] * 0.5, c=RANDOM_COLOR, alpha=0.35,
                   linewidths=0, zorder=2)
        med = float(np.median(rand_vals))
        ax.axvline(med, color=RANDOM_COLOR, lw=s["lw"] * 0.8, ls=":", zorder=1)
        ax.text(0.02, 0.0, f"random median = {med:.1f} mHa",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=s["tick"], color=RANDOM_COLOR)

        # named markers sit ON the strip baseline (y=0), same line as the
        # random points, distinguished by shape/size/colour; labels are
        # staircased above with a thin leader line back to the marker
        for i, (label, val) in enumerate(groups):
            row_y = base_y + row_h * i
            c = ORDER_COLOR.get(label.split(" = ")[0], OI["black"])
            ax.scatter([val], [0], s=s["ms"] * 3.4, c=c, marker="D",
                       edgecolors="white", linewidths=s["marker_lw"] * 0.6, zorder=4)
            ax.plot([val, val], [0.16, row_y - 0.08], color=c,
                    lw=s["lw"] * 0.6, alpha=0.6, zorder=1)
            ax.annotate(label, xy=(val, row_y), xytext=(0, 2),
                        textcoords="offset points",
                        fontsize=s["tick"], color=c, va="bottom", ha="center")

        ax.set_yticks([])
        ax.set_ylim(-0.45, bracket_y + 0.5)
        ax.set_xlim(vmin - 0.09 * xspan, vmax + 0.09 * xspan)
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontsize=s["font"])
        return med, fold

    med_h10, fold_h10 = strip(ax1, h10_rand_vals, h10_groups, "SQD subspace error (mHa)",
                               f"H10, CAS(10,10) -- {len(h10_rand_vals)} random orderings",
                               "H10")
    med_n2, fold_n2 = strip(ax2, n2_rand_vals, n2_groups, "SQD subspace error (mHa)",
                             f"N2, CAS(6,10) -- {len(n2_rand_vals)} random orderings",
                             "N2")

    return fig, len(h10_rand_vals), len(n2_rand_vals), med_h10, med_n2, fold_h10, fold_n2


def figure1():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, n_h10, n_n2, med_h10, med_n2, fold_h10, fold_n2 = draw_fig1(style)
            emit(fig, "fig1_effect_exists", style)
    caption(
        "Figure 1",
        f"The subspace-error effect exists on both search axes. Top: SQD subspace "
        f"error for n={n_h10} random same-spin orderings of H10 CAS(10,10) at fixed "
        f"default anchors (random median {med_h10:.1f} mHa; max/min = {fold_h10:.1f}x, "
        f"shaded band), with identity, physical, physical_reverse, s1_max, s2_max and "
        f"retainedJ_max marked on the strip. Bottom: n={n_n2} random orderings of N2 "
        f"CAS(6,10) (random median {med_n2:.1f} mHa; max/min = {fold_n2:.1f}x), with "
        f"identity, reverse, max_captured_ORACLE and retainedJ_max marked (identity and "
        f"reverse coincide exactly and are shown as one marker). Panels use independent "
        f"x-scales. Source: score_audit_R1.6/all_scores.csv (H10), "
        f"outputs/stage1/nonoracle_scores.csv (N2)."
    )


# ========================================================================
# FIGURE 2 -- the mechanism
# ========================================================================

CEILING = {"N2": 0.9866, "H10": 0.7554}


def scatter_panel(ax, x, y, s, color, box_loc="upper_right", show_n=True):
    ax.scatter(x, y, s=s["ms"], c=color, alpha=1.0, linewidths=0, zorder=2)
    rho, p = spearmanr(x, y)
    parts = [f"$\\rho$={rho:+.3f}", f"p={p:.1e}"]
    if show_n:
        parts.append(f"n={len(x)}")
    txt = "\n".join(parts)
    xy = dict(upper_right=(0.97, 0.95, "right", "top"),
              upper_left=(0.03, 0.95, "left", "top"))[box_loc]
    ax.text(xy[0], xy[1], txt, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=s["tick"])
    return rho, p


def _ceiling_line(ax, value, s):
    ax.axvline(value, color=OI["black"], lw=s["lw"] * 0.9, ls="--", zorder=1, alpha=0.7)
    ax.annotate("ceiling", xy=(value, 0.97), xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points",
                fontsize=s["tick"] * 0.92, color=OI["black"], ha="right", va="top",
                rotation=90, alpha=0.8)


def _sparse_xticks(ax, n=3, fmt="%.2f"):
    """Explicit, evenly-spaced tick positions from the axes' own current
    xlim -- unlike MaxNLocator, this guarantees exactly n ticks, which
    matters in the narrow multi-column panels below where the default
    locator's "nice number" ticks can run into each other."""
    lo, hi = ax.get_xlim()
    ticks = np.linspace(lo, hi, n)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))


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
        # A and B stacked full-width (each needs its own y-ticks, title,
        # rho/p/n box and a ceiling line -- a narrow 1-of-3 gridspec
        # column has no room for all of that at 85mm). C is a row of 3
        # narrow shared-y panels below, same as the slide layout.
        figsize = (PAPER_W, PAPER_W * 2.55)
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.15, wspace=0.15)
        axA = fig.add_subplot(gs[0, :])
        axB = fig.add_subplot(gs[1, :])
        axC1 = fig.add_subplot(gs[2, 0])
        axC2 = fig.add_subplot(gs[2, 1], sharey=axC1)
        axC3 = fig.add_subplot(gs[2, 2], sharey=axC1)
    else:
        figsize = (12.4, 6.4)
        fig = plt.figure(figsize=figsize, layout="constrained")
        gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.1, wspace=0.55)
        axA = fig.add_subplot(gs[0, 0:2])
        axB = fig.add_subplot(gs[0, 2])
        axC1 = fig.add_subplot(gs[1, 0])
        axC2 = fig.add_subplot(gs[1, 1], sharey=axC1)
        axC3 = fig.add_subplot(gs[1, 2], sharey=axC1)

    rhoA, pA = scatter_panel(axA, n2_cap, n2_err, s, OI["blue"], box_loc="upper_left")
    _ceiling_line(axA, CEILING["N2"], s)
    axA.set_xlabel("captured weight")
    axA.set_ylabel("SQD subspace error (mHa)")
    axA.set_title("A -- N2, CAS(6,10)", loc="left", fontsize=s["font"])

    rhoB, pB = scatter_panel(axB, h10_cap, h10_err, s, OI["vermillion"], box_loc="upper_left")
    _ceiling_line(axB, CEILING["H10"], s)
    axB.set_xlabel("captured weight")
    axB.set_title("B -- H10, CAS(10,10)" if style == "paper" else "B -- H10", loc="left", fontsize=s["font"])

    rhosC = {}
    panel_titles = {"identity": "C -- identity", "physical": "physical", "rand007": "rand007"}
    for ax, ordn in zip((axC1, axC2, axC3), ("identity", "physical", "rand007")):
        cap, err = per_ord[ordn]
        rho, p = scatter_panel(ax, cap, err, s, ORDER_COLOR[ordn], show_n=False)
        rhosC[ordn] = (rho, p, len(cap))
        ax.set_xlabel("captured weight")
        ax.set_title(panel_titles[ordn], loc="left", fontsize=s["font"], color=ORDER_COLOR[ordn])
        _sparse_xticks(ax, 3)
    axC1.set_ylabel("SQD subspace error (mHa)")
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
        f"Captured weight predicts SQD subspace error on both search axes, both "
        f"systems, with each system's ideal capture ceiling marked (N2: "
        f"{CEILING['N2']}, H10: {CEILING['H10']}; provided ansatz-capacity values, "
        f"not fit to these points). A: N2 same-spin ordering axis, n={A[2]}, "
        f"rho={A[0]:+.3f} (p={A[1]:.1e}). B: H10 same-spin ordering axis, n={B[2]}, "
        f"rho={B[0]:+.3f} (p={B[1]:.1e}). C: H10 anchor-selection axis at three fixed "
        f"orderings (n=40 triples each, shared y-axis): identity rho={C['identity'][0]:+.3f} "
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
        ("same-spin ordering", span_ordering, 3_628_800, len(err)),
        ("free anchor selection", span_anchor, 120, len(c1)),
        ("anchor offset", span_offset, 4, len(off_err)),
    ]


def draw_fig3(style):
    s = STYLE[style]
    figsize = (PAPER_W, PAPER_W * 0.85) if style == "paper" else (10.0, 5.0)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    data = sorted(figure3_data(), key=lambda d: -d[1])
    labels = [d[0] for d in data]
    spans = [d[1] for d in data]
    colors = [OI["blue"], OI["vermillion"], OI["green"]]

    y = np.arange(len(labels))[::-1]
    ax.barh(y, spans, color=colors, height=0.55, zorder=2)
    # data-driven cap: the longest bar plus a small margin, not a round
    # number -- leaves no large empty region
    xlim = max(spans) * 1.12
    margin = xlim * 0.025
    # the search-space exponent sits inside the bar (fixed absolute
    # margin from its start); the mHa value sits just past the bar's own
    # tip, in the shared axis margin -- anchoring both labels to a bar's
    # own (possibly short) length would crowd them together on the
    # shortest bar, since the two would compete for the same interior
    for yy, d, col in zip(y, data, colors):
        label, span, space, n = d
        exponent = np.log10(space)
        ax.text(margin, yy, f"$10^{{{exponent:.1f}}}$",
                va="center", ha="left", fontsize=s["font"], color="white",
                fontweight="bold")
        ax.text(span + margin, yy, f"{span:.2f} mHa",
                va="center", ha="left", fontsize=s["font"], color=col,
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("SQD subspace error span (mHa)")
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
    parts = "; ".join(
        f"{d[0]}: span={d[1]:.2f} mHa over a search space of {d[2]:,} "
        f"(10^{np.log10(d[2]):.1f}; n={d[3]} sampled)"
        for d in data
    )
    caption(
        "Figure 3",
        f"A tiny search space buys a large SQD subspace error span, across four "
        f"orders of magnitude of search-space size. {parts}. All spans on H10 "
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
FULL120_BEST = 224.60  # c1_all120_identity.csv min err_mHa, all 120 triples


def draw_fig4(style):
    s = STYLE[style]
    figsize = (PAPER_W, PAPER_W * 1.75) if style == "paper" else (12.5, 4.8)
    if style == "paper":
        fig, axes = plt.subplots(3, 1, figsize=figsize, layout="constrained", sharey=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained", sharey=True)

    rows = read_csv(ANCHOR_REANALYSIS)
    results = {}
    for ax, ordn in zip(axes, FIG4_ORDERINGS):
        sub = [r for r in rows if r["ordering"] == ordn]
        x = np.array([float(r["retained_J_oppspin"]) for r in sub])
        y = np.array([float(r["err_mHa"]) for r in sub])
        c = ORDER_COLOR[ordn]
        ax.scatter(x, y, s=s["ms"], c=c, alpha=0.6, linewidths=0, zorder=2)

        i_top = int(np.argmax(x))
        i_best = int(np.argmin(y))
        rank_from_bottom = int(np.argsort(x).tolist().index(i_best)) + 1
        ax.scatter([x[i_top]], [y[i_top]], marker="^", s=s["ms"] * 3.5,
                   facecolors="none", edgecolors=OI["black"], linewidths=s["marker_lw"], zorder=3)
        ax.scatter([x[i_best]], [y[i_best]], marker="*", s=s["ms"] * 5,
                   facecolors=OI["yellow"], edgecolors=OI["black"], linewidths=s["marker_lw"] * 0.6, zorder=4)

        rho, p = spearmanr(x, y)
        frac = FIG4_FRAC_REGRET[ordn]
        ax.text(0.97, 0.95,
                f"$\\rho$={rho:+.3f}\np={p:.1e}\nregret={frac:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=s["tick"])
        ax.set_title(f"{ordn}  (n={len(sub)})", loc="left", fontsize=s["font"], color=c)
        ax.set_xlabel(RETAINED_J_OS_LABEL)
        clean(ax)
        results[ordn] = (rho, p, frac, len(sub), rank_from_bottom, float(y[i_best]))

        if ordn == "physical":
            ax.annotate(
                f"true best ranks {rank_from_bottom}/{len(sub)} from the\n"
                f"bottom on this score -- the rule points\nthe wrong way here",
                xy=(x[i_best], y[i_best]), xycoords="data",
                xytext=(0.30, 0.32), textcoords="axes fraction",
                fontsize=s["tick"], color=OI["vermillion"], ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=OI["vermillion"], lw=s["lw"]),
            )
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
    identity_paired_best = results["identity"][5]
    caption(
        "Figure 4",
        f"The {RETAINED_J_OS_LABEL} anchor-selection rule degrades as the ordering "
        f"moves away from identity. All three panels use the SAME paired set of "
        f"n={n} anchor triples (out of the full C(10,3)=120), so the comparison "
        f"is not confounded by which triples were sampled; over the full 120 "
        f"triples, identity's true best is {FULL120_BEST:.2f} mHa, slightly better "
        f"than the {identity_paired_best:.2f} mHa true best within the paired-40 "
        f"subset plotted here. Triangle marks "
        f"the rule's top pick (max {RETAINED_J_OS_LABEL}), star marks the true best "
        f"(min SQD subspace error); no trend line is drawn -- a straight fit would "
        f"imply a linear model this figure does not claim, so the scatter plus "
        f"Spearman rho stand on their own. In the physical panel the true best sits "
        f"near the bottom of the {RETAINED_J_OS_LABEL} distribution (annotated on "
        f"the panel) -- the rule points in exactly the wrong direction there. "
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
    default_c, best_c = OI["orange"], OI["blue"]

    group_w = 0.75
    bw = group_w / 3
    penalty = None
    for gi, ordn in enumerate(orderings):
        r = rows[ordn]
        floor = float(r["floor"])
        default = float(r["default"])
        best = float(r["best"])
        x_floor = gi - group_w / 2 + bw * 0.5
        x_default = gi - group_w / 2 + bw * 1.5
        x_best = gi - group_w / 2 + bw * 2.5

        ax.bar([x_floor], [floor], width=bw * 0.92, facecolor="none",
               edgecolor=OI["black"], hatch="////", lw=s["lw"], zorder=2)
        ax.bar([x_default], [default], width=bw * 0.92, color=default_c, zorder=2)
        ax.bar([x_best], [best], width=bw * 0.92, color=best_c, zorder=2)

        ax.hlines(floor, gi - group_w / 2 - 0.02, gi + group_w / 2 + 0.02,
                  color=OI["black"], lw=s["lw"], ls="--", zorder=3)

        if default > floor:
            penalty = default - floor
            # the penalty sliver is the top of the default bar itself
            # (already opaque orange), so a *fill* placed behind it would
            # be invisible -- outline it instead, drawn on top of the bar
            rect = mpatches.Rectangle(
                (x_default - bw * 0.46, floor), bw * 0.92, penalty,
                facecolor="none", edgecolor=OI["vermillion"], lw=s["lw"] * 1.8,
                zorder=4,
            )
            ax.add_patch(rect)
            ax.annotate(f"+{penalty:.2f} mHa", xy=(x_default, floor + penalty / 2),
                        xytext=(x_default + bw * 1.05, floor + penalty / 2),
                        fontsize=s["tick"], color=OI["vermillion"], va="center", ha="left",
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=OI["vermillion"], lw=s["lw"] * 0.8))

    ax.set_xticks(range(len(orderings)))
    ax.set_xticklabels(orderings)
    ax.set_ylabel("SQD subspace error (mHa)")
    clean(ax)

    handles = [
        mpatches.Patch(facecolor="none", edgecolor=OI["black"], hatch="////", label="floor"),
        mpatches.Patch(facecolor=default_c, label="default"),
        mpatches.Patch(facecolor=best_c, label="best"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3,
               frameon=False, fontsize=s["tick"])
    return fig, rows, penalty


def figure5():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, rows, penalty = draw_fig5(style)
            emit(fig, "fig5_floor_and_limits", style)
    caption(
        "Figure 5",
        f"The opposite-spin coupling mask is not always a net benefit. Grouped "
        f"bars per ordering (n=3: identity, rand007, physical) show floor "
        f"(same-spin only, hatched outline), default (anchors at p mod 4 = 0) "
        f"and best (anchor search) SQD subspace error; the dashed line spans "
        f"only its own group's floor. At physical the default sits ABOVE its "
        f"own floor by +{penalty:.2f} mHa (shaded region) -- the opposite-spin "
        f"mask actively hurts there. Source: "
        f"floor_generalization/f1b_named_orderings.csv."
    )


# ========================================================================
# FIGURE 6 -- the null results
# ========================================================================

VARIANTS = ["s1_amp", "s1_ampJ", "s1_amp_ss", "s1_amp_os", "s1_ampJ_ss",
            "s1_ampJ_os", "s2", "s2_ss", "s2_os", "s2_soft_ss", "retained_J"]

# Readable tick labels, mirrored in results/figures/README.md
VARIANT_LABEL = {
    "s1_amp": "s1, amplitude",
    "s1_ampJ": "s1, amplitude × J",
    "s1_amp_ss": "s1, amplitude, same-spin",
    "s1_amp_os": "s1, amplitude, opposite-spin",
    "s1_ampJ_ss": "s1, amplitude × J, same-spin",
    "s1_ampJ_os": "s1, amplitude × J, opposite-spin",
    "s2": "s2, pairwise score",
    "s2_ss": "s2, same-spin",
    "s2_os": "s2, opposite-spin",
    "s2_soft_ss": "s2, soft same-spin",
    "retained_J": "retained_J (combined)",
    "retained_J_samespin": "retained_J (same-spin only)",
    "retained_J_oppspin": "retained_J (opposite-spin only)",
}

# The project's own predictive-correlation bar (score_audit.py A2:
# abs(rho) >= 0.5 and p < 1e-3), reused here as the shaded "not yet useful"
# band rather than an unexplained round number.
PREDICTIVE_RHO = 0.5


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

    figsize = (PAPER_W, PAPER_W * 1.65) if style == "paper" else (11.5, 6.5)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.axvspan(-PREDICTIVE_RHO, PREDICTIVE_RHO, color=OI["black"], alpha=0.06, zorder=0)

    y = np.arange(len(rows))[::-1]
    for yy, v in zip(y, rows):
        rho, p, lo, hi = results[v]
        is_decomp = v in ("retained_J_samespin", "retained_J_oppspin")
        c = OI["orange"] if is_decomp else OI["blue"]
        excludes_zero = lo > 0 or hi < 0
        ax.plot([lo, hi], [yy, yy], color=c, lw=s["lw"] * 1.3, zorder=1)
        marker = "o" if excludes_zero else "o"
        face = c if excludes_zero else "white"
        ax.scatter([rho], [yy], s=s["ms"] * 1.6, c=face, edgecolors=c,
                   linewidths=s["marker_lw"], zorder=3, marker=marker)
        if excludes_zero:
            ax.annotate("*", xy=(hi, yy), xytext=(4, -1), textcoords="offset points",
                        fontsize=s["font"] * 1.3, color=c, va="center", ha="left",
                        fontweight="bold")

    def wrap(label):
        # at 85mm width, the long prose labels (F6d) need more left margin
        # than the panel can spare on one line; wrap at the last comma so
        # the same readable text fits in two shorter lines instead.
        if style != "paper" or len(label) <= 16:
            return label
        idx = label.rfind(", ")
        return label if idx < 0 else label[:idx] + ",\n" + label[idx + 2:]

    ax.axvline(0, color=OI["black"], lw=s["lw"] * 0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(VARIANT_LABEL[v]) for v in rows])
    xlabel = "Spearman $\\rho$ (score vs SQD subspace error)" if style == "slide" else "Spearman $\\rho$"
    ax.set_xlabel(xlabel)
    title = f"H10, n={n} random orderings (95% CI)" if style == "slide" else f"H10, n={n}"
    ax.set_title(title, loc="left", fontsize=s["tick"])
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
    excl = [VARIANT_LABEL[v] for v in rows if results[v][2] > 0 or results[v][3] < 0]
    caption(
        "Figure 6",
        f"Eleven score1/score2/retained_J variants (n={n} random H10 orderings), "
        f"Spearman rho vs SQD subspace error with bootstrap 95% CI, ordered by "
        f"|rho|. The shaded band marks |rho| < {PREDICTIVE_RHO}, the predictive-"
        f"correlation threshold already used in score_audit.py's own screen "
        f"(|rho| >= 0.5 and p < 1e-3); every variant here falls inside it except "
        f"captured itself (Figure 2), which is diagnostic-only since it requires "
        f"the sampling it would replace. Filled markers with an asterisk have a "
        f"CI excluding zero ({'; '.join(excl) if excl else 'none'}) but still fall "
        f"short of the predictive band. retained_J's near-zero combined effect "
        f"(rho={rj[0]:+.3f}) is a sign cancellation of its same-spin "
        f"(rho={ss[0]:+.3f}) and opposite-spin (rho={os_[0]:+.3f}) components, "
        f"shown as adjacent rows. Source: score_audit_R1.6/all_scores.csv."
    )


# ========================================================================
# README + main
# ========================================================================

def _variant_mapping_table():
    lines = ["| column | label used in Figure 6 |", "|---|---|"]
    for v in VARIANTS:
        lines.append(f"| `{v}` | {VARIANT_LABEL[v]} |")
    lines.append(f"| `retained_J_samespin` | {VARIANT_LABEL['retained_J_samespin']} |")
    lines.append(f"| `retained_J_oppspin` | {VARIANT_LABEL['retained_J_oppspin']} |")
    return "\n".join(lines)


README = """\
# results/figures/ -- regeneration index

Generated by `experiments/figures.py` (analysis only -- no sampling, no sbd
calls, no new reference computation). Each figure below can be regenerated
independently given the listed CSV(s) and columns.

Ordering names are normalised to a single canonical vocabulary everywhere a
name is drawn on a figure: `identity`, `reverse`, `physical`,
`physical_reverse`, `s1_max`, `s2_max`, `retainedJ_max`, `rand007`. The
cached N2 CSV spells one of these `max_retainedJ`; `RENAME_ORDERING` in
`figures.py` maps it to `retainedJ_max` before drawing.

## Figure 1 -- the effect exists
- `experiments/outputs/score_audit_R1.6/all_scores.csv`
  columns: `ordering`, `err_mHa` (top panel, H10: 50 random `rand###`
  orderings, named `identity`/`physical`/`physical_reverse`/`s1_max`/
  `s2_max`/`retainedJ_max`)
- `outputs/stage1/nonoracle_scores.csv`
  columns: `ordering`, `err_mHa` (bottom panel, N2: 149 random `r###`
  orderings, named `identity`/`reverse`/`max_captured_ORACLE`/
  `retainedJ_max` (renamed from `max_retainedJ`)). NOTE: this file lives
  outside `experiments/outputs/` -- it is the only cached CSV at the
  required 149-random-ordering sample size.
- The shaded band and "Nx" annotation are the random set's own max/min
  ratio, computed at generation time (not cached).

## Figure 2 -- the mechanism
- Panel A (N2, n=149): `outputs/stage1/nonoracle_scores.csv`,
  columns `captured`, `err_mHa` (random `r###` rows only)
- Panel B (H10, n=50): `experiments/outputs/score_audit_R1.6/all_scores.csv`,
  columns `captured`, `err_mHa` (random `rand###` rows only)
- Panel C (3x n=40, shared y-axis): `experiments/outputs/anchor_reanalysis/anchor_reanalysis.csv`,
  columns `ordering`, `captured`, `err_mHa`, filtered to
  `identity`/`physical`/`rand007`
- Ceiling reference lines (N2: 0.9866, H10: 0.7554) are supplied ansatz-
  capacity constants, not derived from the plotted CSVs -- they are not
  present in any cached file and are hardcoded in `figures.py` as `CEILING`.

## Figure 3 -- two levers
- Same-spin ordering lever: `experiments/outputs/score_audit_R1.6/all_scores.csv`,
  column `err_mHa` (50 random `rand###` rows)
- Free anchor-selection lever: `experiments/outputs/anchor_decomposition_R1.6/c1_all120_identity.csv`,
  column `err_mHa` (all 120 rows)
- Anchor-offset lever: `experiments/outputs/anchor_decomposition_R1.6/b1_offset_sweep.csv`,
  columns `ordering`, `anchor_offset`, `err_mHa`, filtered to `ordering=='physical'`
- Search-space sizes (10! = 3,628,800; C(10,3) = 120; 4) are combinatorial
  facts about the respective search, not read from a CSV.

## Figure 4 -- the anchor rule and its limits
- `experiments/outputs/anchor_reanalysis/anchor_reanalysis.csv`,
  columns `ordering`, `triple`, `retained_J_oppspin`, `err_mHa`, filtered to
  `identity`/`rand007`/`physical`. All three orderings use the SAME paired
  set of 40 triples (verified: identical `triple` sets across the three
  filters), so the panel-to-panel comparison is not confounded by sampling.
  Fractional regret values (0.217 / 0.513 / 0.696) and the full-120-triple
  best (224.60 mHa, `anchor_decomposition_R1.6/c1_all120_identity.csv`) are
  taken from `experiments/outputs/anchor_reanalysis/report.txt` (section D2).

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
  `rand###` H10 orderings. Spearman rho is exact; 95% CIs are a manual
  paired bootstrap (2000 resamples per variant, seeded deterministically
  from `zlib.crc32` of the column name so re-running reproduces the same
  figure bit-for-bit) over those same 50 rows, recomputed by `figures.py`
  at generation time (not cached). The shaded band (|rho| < 0.5) is
  `score_audit.py`'s own predictive-correlation threshold, reused here
  rather than an unexplained round number.

### Score-variant label mapping (F6d)

{variant_table}

## Regenerating
```
python3 experiments/figures.py
```
Outputs `<name>_paper.pdf` / `.png` (85mm width) and `<name>_slide.pdf` /
`.png` (widescreen, ~16:9 -- a few figures are sized modestly wider or
taller than exactly 16:9 where that many rows/panels needed it to stay
legible and clipping-free) per figure into this directory. Prints all six suggested
captions to stdout, followed by a render-verification report: for every
file, its pixel dimensions (read back from the saved PNG with PIL) and
whether any annotation was found to extend beyond the saved canvas.
""".format(variant_table=_variant_mapping_table())


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

    print("Render verification (pixel dimensions read back from the saved PNG; "
          "clipping checked at save dpi against the full canvas):")
    for r in RENDER_REPORT:
        dims_ok = r["actual_px"] == r["expected_px"]
        dims = f"{r['actual_px'][0]}x{r['actual_px'][1]} px"
        dims_note = "" if dims_ok else f" (expected {r['expected_px'][0]}x{r['expected_px'][1]})"
        if r["clipped"]:
            status = f"CLIPPED: {r['clipped']}"
        else:
            status = "no annotation extends beyond the saved canvas"
        print(f"  {r['name']}: {dims}{dims_note} -- {status}")


if __name__ == "__main__":
    sys.exit(main())
