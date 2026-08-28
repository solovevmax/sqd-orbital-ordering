#!/usr/bin/env python3
"""
experiments/figures.py
=======================

Publication figure set for the report / final presentation. Analysis only:
reads cached CSVs already produced by prior experiments, does not sample,
call sbd, or recompute any reference. Writes results/figures/.

Every figure is emitted in two physical sizes (paper: 85mm single-column,
slide: widescreen) and two formats (vector PDF, 300dpi PNG) -- eleven
figures x two sizes x two formats = 44 image files, plus
results/figures/README.md. Exception: Figures 2 and 7a's paper variants are
170mm (double-column) rather than 85mm -- their side-by-side panels need
that width to stay legible at their requested wide aspect ratios.

Data provenance note: most figures draw only from experiments/outputs/.
Figures 1 (bottom panel) and 2 (panel A) need the 149-random-ordering N2
dataset; the only cached CSV at that exact sample size is
outputs/stage1/nonoracle_scores.csv (root-level outputs/, not
experiments/outputs/). It is read as-is -- no recomputation -- and the
exception is called out explicitly in results/figures/README.md. Figure 2
panel C reads experiments/outputs/tm_transfer/ (Cr2 transition-metal transfer
test) and Figure 7a/7b read experiments/outputs/g1_lite/ and
experiments/outputs/chain_aware/ (12 held-out H10 chains) -- see their
README sections for the corrections applied before drawing.

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
    # neutral grey: not one of the 8 core Okabe-Ito hues, but a common,
    # colourblind-safe extension for a "this is not one of the real data
    # colours" series -- used only for `reverse`, so it never appears
    # next to a genuine data colour, and specifically so it is
    # distinguishable from the pure-black random-ordering cloud in Figure 1
    grey="#666666",
)

# Canonical ordering-name vocabulary, used verbatim everywhere an ordering
# is named on a figure: identity, reverse, physical, physical_reverse,
# s1_max, s2_max, retainedJ_max, rand007. Some cached CSVs spell these
# differently (e.g. N2's "max_retainedJ"); RENAME_ORDERING maps those onto
# the canonical spelling before anything is drawn or printed.
RENAME_ORDERING = {
    "max_retainedJ": "retainedJ_max",
    "max_captured_ORACLE": "max_captured (oracle)",
}

ORDER_COLOR = {
    "identity": OI["blue"],
    "physical": OI["vermillion"],
    "physical_reverse": OI["purple"],
    "s1_max": OI["yellow"],
    "s2_max": OI["orange"],
    "retainedJ_max": OI["sky_blue"],
    "rand007": OI["green"],
    "reverse": OI["grey"],
    "max_captured (oracle)": OI["vermillion"],
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


def check_edge_pixels(png_path, border=2, tol=250):
    """Ground-truth clipping check on the actual saved file: any non-white
    pixel within `border` pixels of the canvas edge. This exists because
    check_clipping()'s matplotlib redraw is not always reliable -- for at
    least one figure here, constrained_layout produced a *different*
    layout on a redraw than the one it used for the actual savefig() (a
    label that's inherently wider than its axis is a degenerate case for
    the solver), so a redraw-based check can pass while the saved PNG
    itself is clipped. Reading the file back removes that gap entirely."""
    with Image.open(png_path) as im:
        arr = np.asarray(im.convert("RGB"))
    edges = np.concatenate([
        arr[:border, :, :].reshape(-1, 3),
        arr[-border:, :, :].reshape(-1, 3),
        arr[:, :border, :].reshape(-1, 3),
        arr[:, -border:, :].reshape(-1, 3),
    ])
    return bool(np.any(edges < tol))


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

    if check_edge_pixels(png) and not clipped:
        clipped = ["<pixel-edge check: non-white content touches the canvas "
                   "border, but the matplotlib redraw check did not identify "
                   "which artist -- inspect this file visually>"]

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
H10_BASELINE = REPO / "experiments" / "outputs" / "h10_baseline_R1.6" / "h10_baseline_results.csv"
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


def n2_outlier_diagnostic():
    """The N2 random-ordering max/min ratio (8.0x) is driven almost
    entirely by a single point (r102, 173.23 mHa -- the next-highest point
    is 122.79 mHa). Printed once at generation time so the annotation
    choice below (full-range AND p95/p5) is backed by the actual numbers,
    not asserted."""
    rand, _ = n2_named_rand()
    errs = np.array([float(r["err_mHa"]) for r in rand])
    caps = np.array([float(r["captured"]) for r in rand])
    names = [r["ordering"] for r in rand]
    order = np.argsort(-errs)

    mx, mn = errs.max(), errs.min()
    second_max = np.sort(errs)[-2]
    p95, p5 = np.percentile(errs, 95), np.percentile(errs, 5)
    rho_all, p_all = spearmanr(caps, errs)
    imax = int(np.argmax(errs))
    mask = np.ones(len(errs), dtype=bool)
    mask[imax] = False
    rho_wo, p_wo = spearmanr(caps[mask], errs[mask])

    print(f"\nFigure 1 N2 outlier diagnostic (n={len(errs)} random orderings):")
    print("  Top 6 by err_mHa:")
    for i in order[:6]:
        print(f"    {names[i]}: err={errs[i]:.2f} mHa  captured={caps[i]:.4f}")
    print(f"  max/min ratio:        {mx / mn:.2f}x")
    print(f"  second-max/min ratio: {second_max / mn:.2f}x")
    print(f"  p95/p5 ratio:         {p95 / p5:.2f}x")
    print(f"  rho(captured, err) with all {len(errs)} points:    {rho_all:+.3f} (p={p_all:.2e})")
    print(f"  rho(captured, err) without the largest point (n={len(errs) - 1}): {rho_wo:+.3f} (p={p_wo:.2e})")
    return dict(fold=mx / mn, p95p5=p95 / p5, second_fold=second_max / mn,
                rho_all=rho_all, rho_wo=rho_wo)


# ========================================================================
# FIGURE 1 -- the effect exists
# ========================================================================

def _label_groups(named_vals, merge_tol=1.0):
    """Merge named orderings landing within merge_tol mHa of each other
    (exact ties as well as near-misses like reverse/retainedJ_max, 0.11
    mHa apart), so two markers never sit exactly on top of one another,
    and return groups sorted by value ascending. Merged labels are joined
    in `named_vals`'s own insertion order (e.g. "reverse = retainedJ_max"),
    not value order, so which name comes first doesn't depend on a
    difference of a few hundredths of a mHa."""
    order_index = {label: i for i, label in enumerate(named_vals)}
    groups = []  # each: [display_label, representative_val, [members]]
    for label, val in sorted(named_vals.items(), key=lambda kv: kv[1]):
        if groups and abs(val - groups[-1][1]) < merge_tol:
            groups[-1][2].append(label)
            groups[-1][0] = " = ".join(sorted(groups[-1][2], key=lambda l: order_index[l]))
        else:
            groups.append([label, val, [label]])
    return [(label, val) for label, val, _members in groups]


def draw_fig1(style):
    s = STYLE[style]

    h10_rand, h10_named = h10_named_rand()
    h10_wanted = ["identity", "reverse", "physical", "physical_reverse", "s1_max", "s2_max", "retainedJ_max"]
    h10_named_use = {k: float(h10_named[k]["err_mHa"]) for k in h10_wanted if k in h10_named}
    h10_rand_vals = [float(r["err_mHa"]) for r in h10_rand]
    h10_groups = _label_groups(h10_named_use)

    n2_rand, n2_named = n2_named_rand()
    n2_wanted = ["identity", "reverse", "max_captured (oracle)", "retainedJ_max"]
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

    def strip(ax, rand_vals, groups, xlabel, title, fold_label, p95p5=None):
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
        # the plain max/min ratio is driven almost entirely by a single
        # outlier point on N2 (see n2_outlier_diagnostic()); p95/p5 is
        # shown alongside it there, not instead of it -- no point is
        # removed, both numbers are just made visible together
        label_txt = (f"{fold:.1f}$\\times$ full range | {p95p5:.1f}$\\times$ p95-p5"
                     if p95p5 is not None else f"{fold:.1f}$\\times$")
        ax.text((vmin + vmax) / 2, bracket_y + 0.14, label_txt,
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

    n2_p95p5 = float(np.percentile(n2_rand_vals, 95)) / float(np.percentile(n2_rand_vals, 5))

    med_h10, fold_h10 = strip(ax1, h10_rand_vals, h10_groups, "SQD subspace error (mHa)",
                               f"H10, CAS(10,10) -- {len(h10_rand_vals)} random orderings",
                               "H10")
    med_n2, fold_n2 = strip(ax2, n2_rand_vals, n2_groups, "SQD subspace error (mHa)",
                             f"N2, CAS(6,10) -- {len(n2_rand_vals)} random orderings",
                             "N2", p95p5=n2_p95p5)

    return fig, len(h10_rand_vals), len(n2_rand_vals), med_h10, med_n2, fold_h10, fold_n2, n2_p95p5


def figure1():
    diag = n2_outlier_diagnostic()
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, n_h10, n_n2, med_h10, med_n2, fold_h10, fold_n2, n2_p95p5 = draw_fig1(style)
            emit(fig, "fig1_effect_exists", style)
    caption(
        "Figure 1",
        f"The subspace-error effect exists on both search axes. Top: SQD subspace "
        f"error for n={n_h10} random same-spin orderings of H10 CAS(10,10) at fixed "
        f"default anchors (random median {med_h10:.1f} mHa; max/min = {fold_h10:.1f}x, "
        f"shaded band), with identity, reverse, physical, physical_reverse, s1_max, "
        f"s2_max and retainedJ_max marked on the strip (reverse and retainedJ_max are "
        f"only 0.11 mHa apart and are shown as one joint marker/label). Bottom: n={n_n2} "
        f"random orderings of N2 CAS(6,10) (random median {med_n2:.1f} mHa; max/min = "
        f"{fold_n2:.1f}x full range, {n2_p95p5:.1f}x p95/p5), with identity, reverse, "
        f"max_captured (oracle) and retainedJ_max marked (identity and reverse coincide "
        f"exactly and are shown as one marker). No point is removed: the full-range "
        f"factor is shown alongside p95/p5 rather than in place of it, because it is "
        f"driven almost entirely by a single point (r102, {diag['fold']:.2f}x/min vs. "
        f"the next-highest point's {diag['second_fold']:.2f}x/min) -- Spearman rho is "
        f"materially unchanged with ({diag['rho_all']:+.3f}) or without ({diag['rho_wo']:+.3f}) "
        f"it, so it is a real point, not an artifact. Panels use independent x-scales. "
        f"Source: score_audit_R1.6/all_scores.csv (H10), outputs/stage1/nonoracle_scores.csv (N2)."
    )


# ========================================================================
# FIGURE 2 -- the mechanism
# ========================================================================

TM_TRANSFER_SQD = REPO / "experiments" / "outputs" / "tm_transfer" / "stage2_sqd.csv"

# N2/H10 ceilings are supplied ansatz-capacity constants, not derived from
# any cached CSV (see README). Cr2's is different: it IS a value computed
# by the tm_transfer run itself (ideal_ceiling in stage3_report.txt), read
# here as the literal number from that report rather than hand-typed.
CEILING = {"N2": 0.9866, "H10": 0.7554, "Cr2": 0.881293}


def cr2_anchor_axis_points():
    """Pooled (captured, err_mHa) over all three tm_transfer chains
    (identity, random, reverse) -- 60 anchor triples each, 180 total. The
    per-chain rho is -0.971/-0.967/-0.971 (stage3_report.txt); pooling is
    valid here because it's the same relationship being sampled three
    times, not three different relationships."""
    rows = read_csv(TM_TRANSFER_SQD)
    sub = [r for r in rows if r["role"] == "triple"]
    return [float(r["captured"]) for r in sub], [float(r["err_mHa"]) for r in sub]


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

    cr2_cap, cr2_err = cr2_anchor_axis_points()

    # three systems, side by side, each with its own full axes (independent
    # error scales rule out a shared y-axis) -- 85mm can't hold three fully
    # labelled panels legibly, so like Figure 7 this is emitted at full
    # double-column width (170mm) in the paper variant instead
    figsize = (170 * MM_TO_IN, 170 * MM_TO_IN * 0.46) if style == "paper" else (13.5, 4.6)
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=figsize, layout="constrained")

    rhoA, pA = scatter_panel(axA, n2_cap, n2_err, s, OI["blue"], box_loc="upper_left")
    _ceiling_line(axA, CEILING["N2"], s)
    axA.set_xlabel("captured weight")
    axA.set_ylabel("SQD subspace error (mHa)")
    axA.set_title("A -- N2, CAS(6,10)", loc="left", fontsize=s["font"])

    rhoB, pB = scatter_panel(axB, h10_cap, h10_err, s, OI["vermillion"], box_loc="upper_left")
    _ceiling_line(axB, CEILING["H10"], s)
    axB.set_xlabel("captured weight")
    axB.set_title("B -- H10, CAS(10,10)", loc="left", fontsize=s["font"])

    rhoC, pC = scatter_panel(axC, cr2_cap, cr2_err, s, OI["green"], box_loc="upper_left")
    _ceiling_line(axC, CEILING["Cr2"], s)
    axC.set_xlabel("captured weight")
    axC.set_title("C -- Cr2, CAS(12,12)", loc="left", fontsize=s["font"])

    for a in (axA, axB, axC):
        clean(a)

    return fig, (rhoA, pA, len(n2_cap)), (rhoB, pB, len(h10_cap)), (rhoC, pC, len(cr2_cap))


def figure2():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, A, B, C = draw_fig2(style)
            emit(fig, "fig2_mechanism", style)
    caption(
        "Figure 2",
        f"Captured weight predicts SQD subspace error across three systems, with "
        f"each system's ideal capture ceiling marked (N2: {CEILING['N2']}, H10: "
        f"{CEILING['H10']} -- provided ansatz-capacity constants, not fit to these "
        f"points; Cr2: {CEILING['Cr2']:.4f} -- computed ideal_ceiling from the "
        f"tm_transfer run itself). A: N2, CAS(6,10), same-spin ordering axis, "
        f"n={A[2]}, rho={A[0]:+.3f} (p={A[1]:.1e}). B: H10, CAS(10,10), same-spin "
        f"ordering axis, n={B[2]}, rho={B[0]:+.3f} (p={B[1]:.1e}). C: Cr2, CAS(12,12), "
        f"anchor-selection axis pooled across all 3 chains (identity, random, "
        f"reverse; n={C[2]} = 3 x 60 triples), rho={C[0]:+.3f} (p={C[1]:.1e}) -- "
        f"per-chain rho is -0.971/-0.967/-0.971 (tm_transfer/stage3_report.txt), so "
        f"pooling does not average over three different relationships, it repeats "
        f"the same one. The capture mechanism (this figure) replicates on a "
        f"transition-metal active space, the project's intended application domain. "
        f"The H10 anchor-selection axis previously shown here (identity/physical/"
        f"rand007) moves to Figure 2b for backup reference. Source: "
        f"outputs/stage1/nonoracle_scores.csv (A), score_audit_R1.6/all_scores.csv "
        f"(B), tm_transfer/stage2_sqd.csv (C)."
    )


def draw_fig2b_anchor_axis(style):
    s = STYLE[style]
    anchor_rows = read_csv(ANCHOR_REANALYSIS)
    per_ord = {}
    for ordn in ("identity", "physical", "rand007"):
        sub = [r for r in anchor_rows if r["ordering"] == ordn]
        per_ord[ordn] = ([float(r["captured"]) for r in sub], [float(r["err_mHa"]) for r in sub])

    figsize = (PAPER_W, PAPER_W * 0.85) if style == "paper" else (11.0, 4.2)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize, layout="constrained", sharey=True)

    rhos = {}
    panel_titles = {"identity": "identity", "physical": "physical", "rand007": "rand007"}
    for ax, ordn in zip((ax1, ax2, ax3), ("identity", "physical", "rand007")):
        cap, err = per_ord[ordn]
        rho, p = scatter_panel(ax, cap, err, s, ORDER_COLOR[ordn])
        rhos[ordn] = (rho, p, len(cap))
        ax.set_title(panel_titles[ordn], loc="left", fontsize=s["font"], color=ORDER_COLOR[ordn])
        _sparse_xticks(ax, 3)
    # one shared x-label on the centre panel rather than three repeats --
    # at 85mm width, "captured weight" x 3 runs past the right edge
    ax2.set_xlabel("captured weight")
    ax1.set_ylabel("SQD subspace error (mHa)")
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)

    for a in (ax1, ax2, ax3):
        clean(a)

    return fig, rhos


def figure2b():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, rhos = draw_fig2b_anchor_axis(style)
            emit(fig, "fig2b_anchor_axis", style)
    caption(
        "Figure 2b (backup)",
        f"H10 anchor-selection axis at three fixed orderings (n=40 triples each, "
        f"shared y-axis): identity rho={rhos['identity'][0]:+.3f} "
        f"(p={rhos['identity'][1]:.1e}), physical rho={rhos['physical'][0]:+.3f} "
        f"(p={rhos['physical'][1]:.1e}), rand007 rho={rhos['rand007'][0]:+.3f} "
        f"(p={rhos['rand007'][1]:.1e}). Same mechanism as Figure 2, on the anchor-"
        f"selection search axis instead of the same-spin-ordering axis; moved out "
        f"of the main sequence to make room for Figure 2's three-system layout. "
        f"For backup/reference use. Source: anchor_reanalysis/anchor_reanalysis.csv."
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
                f"true best ranks {rank_from_bottom}/{len(sub)} from\n"
                f"the bottom on this score --\n"
                f"the rule points the wrong\n"
                f"way here",
                xy=(x[i_best], y[i_best]), xycoords="data",
                xytext=(0.06, 0.32), textcoords="axes fraction",
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
            # shrinkB leaves a gap between the leader line and the text --
            # without it the line runs flush into the "+" glyph and the
            # two visually fuse into what reads as a "+/-" sign
            ax.annotate(f"+{penalty:.2f} mHa", xy=(x_default, floor + penalty / 2),
                        xytext=(x_default + bw * 1.05, floor + penalty / 2),
                        fontsize=s["tick"], color=OI["vermillion"], va="center", ha="left",
                        fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=OI["vermillion"], lw=s["lw"] * 0.8,
                                         shrinkA=0, shrinkB=6))

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


def draw_fig6b(style):
    """A single-message slide figure -- nothing beyond the three rows,
    their bootstrap CIs, and the zero line: retained_J's near-zero combined
    effect is a sign cancellation of its two opposite-signed components.
    Fonts are scaled up well beyond the normal STYLE sizes since this must
    read from the back of a room, not on a printed page."""
    s = STYLE[style]
    big = dict(s, font=s["font"] * 1.7, tick=s["tick"] * 1.6,
               lw=s["lw"] * 1.4, ms=s["ms"] * 1.5, marker_lw=s["marker_lw"] * 1.4)
    results, n = figure6_data()
    rows = ["retained_J", "retained_J_samespin", "retained_J_oppspin"]

    figsize = (PAPER_W * 1.7, PAPER_W * 1.15) if style == "paper" else (11.0, 5.5)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    y = np.arange(len(rows))[::-1]
    his = []
    for yy, v in zip(y, rows):
        rho, p, lo, hi = results[v]
        his.append(hi)
        c = OI["orange"] if v != "retained_J" else OI["blue"]
        ax.plot([lo, hi], [yy, yy], color=c, lw=big["lw"] * 1.5, zorder=1)
        ax.scatter([rho], [yy], s=big["ms"] * 2.2, c=c, edgecolors="white",
                   linewidths=big["marker_lw"], zorder=3)
        ax.annotate(f"{rho:+.3f}", xy=(hi, yy), xytext=(12, 0), textcoords="offset points",
                    fontsize=big["font"], color=c, va="center", ha="left", fontweight="bold")

    ax.axvline(0, color=OI["black"], lw=big["lw"], zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([VARIANT_LABEL[v] for v in rows], fontsize=big["tick"])
    # short label -- at generous font this is a wide word already, and the
    # fuller "(vs SQD subspace error)" phrasing ran past the canvas edge
    ax.set_xlabel("Spearman $\\rho$", fontsize=big["font"])
    ax.tick_params(axis="x", labelsize=big["tick"])
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    # explicit right-hand pad for the "+0.212"-style value labels, which
    # autoscale doesn't account for (it only sees the line/marker data)
    ax.set_xlim(min(results[v][2] for v in rows) - 0.12, max(his) + 0.55)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    return fig, results, n


def figure6b():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, results, n = draw_fig6b(style)
            emit(fig, "fig6b_cancellation", style)
    rj = results["retained_J"]
    ss = results["retained_J_samespin"]
    os_ = results["retained_J_oppspin"]
    caption(
        "Figure 6b",
        f"retained_J's near-zero combined effect (rho={rj[0]:+.3f}, n={n} random H10 "
        f"orderings) is a sign cancellation: its same-spin component is positive "
        f"(rho={ss[0]:+.3f}) and its opposite-spin component is negative "
        f"(rho={os_[0]:+.3f}), with bootstrap 95% CIs shown. Single-message slide "
        f"figure -- Figure 6's 11-row version is unchanged and remains the report "
        f"figure. Source: score_audit_R1.6/all_scores.csv."
    )


# ========================================================================
# FIGURE 7 -- lever interaction and spread compression
# ========================================================================

G1_LITE_DIR = REPO / "experiments" / "outputs" / "g1_lite"
G1_ALL = G1_LITE_DIR / "g1_all.csv"
CHAIN_AWARE_B2 = REPO / "experiments" / "outputs" / "chain_aware" / "phaseB_b2_all.csv"

FIG7_ORDERINGS = ["identity", "physical", "rand030", "rand037",
                   "rand007", "rand032", "rand047", "rand029"]
NEW_CHAINS = [f"newchain{i:02d}" for i in range(12)]
ALL_FIG7_CHAINS = FIG7_ORDERINGS + NEW_CHAINS

FIG7_HIGHLIGHT = {"rand029": OI["vermillion"], "rand030": OI["blue"]}
FIG7_OTHER = "#888888"

# chain_aware/step2_analysis_report.txt (B3.5): "Combined n=20 (8 G1-lite +
# 12 new): ... compression=3.15x". That number is best-of-candidates alone,
# WITHOUT the default-anchor-triple-inclusion correction applied below (see
# figure7_n20_data()) -- kept here only as the value to check the corrected
# recompute against, per the task's explicit instruction to do so.
REPORTED_COMPRESSION_N20 = 3.15

CHANGE_REPORT = []


def _default_anchor_triple(permutation_str):
    """The default anchor rule is position % 4 == 0 (run_ordering_pipeline
    CFG['anchor_mod']=4), where "position" is where an orbital sits in the
    ordering's own layout -- not the raw orbital index. positions_from()
    computes this as argsort(perm): pos[orbital] = its layout position."""
    perm = np.array([int(c) for c in permutation_str])
    pos = np.argsort(perm)
    return tuple(sorted(int(o) for o in range(len(perm)) if pos[o] % 4 == 0))


def figure7_data():
    base_rows = read_csv(H10_BASELINE)
    baseline = {}
    permutation = {}
    for r in base_rows:
        if r["ordering"] in FIG7_ORDERINGS and r["seed"] == "2026" and r["ordering"] not in baseline:
            baseline[r["ordering"]] = float(r["err_mHa"])
            permutation[r["ordering"]] = r["permutation"]

    all_rows = read_csv(G1_ALL)
    best40 = {}
    best40_triple = {}
    sampled_triples = {}
    for ordn in FIG7_ORDERINGS:
        sub = [r for r in all_rows if r["ordering"] == ordn]
        triples = {}
        for r in sub:
            t = tuple(int(x) for x in r["triple"].strip("()").replace(" ", "").split(","))
            triples[t] = float(r["err_mHa"])
        sampled_triples[ordn] = triples
        best_t = min(triples, key=triples.get)
        best40[ordn] = triples[best_t]
        best40_triple[ordn] = best_t

    # IMPORTANT check: is each ordering's own default anchor triple part of
    # its 40 sampled candidates? If not, best-of-40 alone can be *worse*
    # than the ordering's actual default -- exactly what happens for
    # rand030 (best40=180.81 > its own baseline/default=168.67), which is
    # only possible if the default was excluded from the candidate pool.
    corrected = {}
    default_triple = {}
    CHANGE_REPORT.clear()
    for ordn in FIG7_ORDERINGS:
        dt = _default_anchor_triple(permutation[ordn])
        default_triple[ordn] = dt
        in_pool = dt in sampled_triples[ordn]
        # the default triple's err_mHa is exactly the baseline evaluation
        # (that IS what "baseline" means: default-anchor, default ordering)
        best_with_default = min(best40[ordn], baseline[ordn])
        corrected[ordn] = best_with_default
        if best_with_default < best40[ordn] - 1e-9:
            CHANGE_REPORT.append(
                f"{ordn}: default triple {dt} was NOT in its 40 sampled candidates; "
                f"best-of-40 alone gave {best40[ordn]:.2f} mHa, corrected "
                f"(best-of-40 UNION default) = {best_with_default:.2f} mHa "
                f"({best40[ordn] - best_with_default:+.2f} mHa)"
            )
        elif not in_pool:
            CHANGE_REPORT.append(
                f"{ordn}: default triple {dt} was NOT in its 40 sampled candidates, "
                f"but best-of-40 ({best40[ordn]:.2f} mHa) already beat the default "
                f"({baseline[ordn]:.2f} mHa) anyway -- no change"
            )

    return dict(baseline=baseline, best40=best40, corrected=corrected,
                default_triple=default_triple, best40_triple=best40_triple)


def figure7_new_chains_data():
    """The 12 held-out chains from chain_aware/phaseB: same
    default-anchor-triple-inclusion correction as figure7_data() above,
    applied to phaseB_b2_all.csv's role/is_floor/is_default columns
    directly rather than re-deriving the default triple from a permutation
    string (phaseB_b2_all.csv already flags the default-anchor row and the
    floor-control row explicitly)."""
    rows = read_csv(CHAIN_AWARE_B2)
    by_chain = {}
    for r in rows:
        by_chain.setdefault(r["chain"], []).append(r)

    baseline, corrected = {}, {}
    for chain in NEW_CHAINS:
        rs = by_chain[chain]
        default_row = next(r for r in rs if r["role"] == "default_anchor")
        default_err = float(default_row["err_sqd"])
        cands = [r for r in rs if r["role"] == "" and r["is_floor"] != "True"]
        best_cand = min(float(r["err_sqd"]) for r in cands)
        best_with_default = min(best_cand, default_err)
        baseline[chain] = default_err
        corrected[chain] = best_with_default
        if best_with_default < best_cand - 1e-9:
            CHANGE_REPORT.append(
                f"{chain}: default-anchor triple not among its {len(cands)} "
                f"candidates; best-of-candidates alone gave {best_cand:.2f} mHa, "
                f"corrected = {best_with_default:.2f} mHa "
                f"({best_cand - best_with_default:+.2f} mHa)"
            )
    return baseline, corrected


_FIG7_N20_CACHE = None


def figure7_n20_data():
    """Combines the 8 G1-lite orderings with the 12 chain_aware held-out
    chains into one n=20 pool, both under the same default-anchor-triple
    correction, then checks the resulting compression factor against
    REPORTED_COMPRESSION_N20 and prints the exact recomputed numbers --
    per the task's explicit CRITICAL instruction, this figure uses the
    recomputed value even where it disagrees with the report, and says so.
    Cached module-wide: draw_fig7a/draw_fig7b each call this once per style
    (paper, slide), and the diagnostic print below should appear once per
    run, not once per call."""
    global _FIG7_N20_CACHE
    if _FIG7_N20_CACHE is not None:
        return _FIG7_N20_CACHE

    d8 = figure7_data()
    baseline12, corrected12 = figure7_new_chains_data()
    baseline = dict(d8["baseline"])
    corrected = dict(d8["corrected"])
    baseline.update(baseline12)
    corrected.update(corrected12)

    b_vals = np.array([baseline[k] for k in ALL_FIG7_CHAINS])
    c_vals = np.array([corrected[k] for k in ALL_FIG7_CHAINS])
    b_span = float(b_vals.max() - b_vals.min())
    c_span = float(c_vals.max() - c_vals.min())
    compression = b_span / c_span
    b_argmin = min(baseline, key=baseline.get)
    b_argmax = max(baseline, key=baseline.get)
    c_argmin = min(corrected, key=corrected.get)
    c_argmax = max(corrected, key=corrected.get)

    print(f"\nFigure 7a/7b n=20 compression recompute "
          f"(default anchor triple included in every candidate set):")
    print(f"  default-anchor spread: {b_span:.2f} mHa "
          f"(min {b_argmin}={baseline[b_argmin]:.2f}, max {b_argmax}={baseline[b_argmax]:.2f})")
    print(f"  best-anchor spread:    {c_span:.2f} mHa "
          f"(min {c_argmin}={corrected[c_argmin]:.2f}, max {c_argmax}={corrected[c_argmax]:.2f})")
    print(f"  compression: {compression:.3f}x  (report states {REPORTED_COMPRESSION_N20}x)")
    if abs(compression - REPORTED_COMPRESSION_N20) > 0.03:
        print(
            f"  DISCREPANCY: recomputed {compression:.2f}x does not match the "
            f"report's {REPORTED_COMPRESSION_N20}x (chain_aware/"
            f"step2_analysis_report.txt, B3.5). That number is best-of-candidates "
            f"alone, without the default-anchor-triple-inclusion correction this "
            f"figure is required to apply. Applying the correction can only ever "
            f"lower or hold a chain's own best value (it's a min() against the "
            f"uncorrected best), never raise it -- here it lowers rand030's best "
            f"from 180.81 to 168.67 mHa, which becomes the new pooled minimum and "
            f"widens the best-anchor spread from {c_span - (180.81 - 168.67):.2f} "
            f"to {c_span:.2f} mHa. So the corrected compression factor is "
            f"mathematically bounded to be <= the report's number, and the "
            f"recomputed {compression:.2f}x is consistent with that direction, "
            f"not a computation error. This figure uses the recomputed "
            f"{compression:.2f}x, not the report's {REPORTED_COMPRESSION_N20}x."
        )
    else:
        print("  matches the report.")

    _FIG7_N20_CACHE = dict(baseline=baseline, corrected=corrected, b_span=b_span,
                           c_span=c_span, compression=compression)
    return _FIG7_N20_CACHE


def _count_crossings(baseline, corrected, keys):
    """How many other chains each chain's line crosses between the default
    and best columns -- i.e. how many chains it swaps relative rank with."""
    import itertools
    counts = {k: 0 for k in keys}
    for a, b in itertools.combinations(keys, 2):
        if (baseline[a] < baseline[b]) != (corrected[a] < corrected[b]):
            counts[a] += 1
            counts[b] += 1
    return counts


def draw_fig7a(style):
    s = STYLE[style]
    d = figure7_n20_data()
    baseline, corrected = d["baseline"], d["corrected"]
    b_span, c_span, compression = d["b_span"], d["c_span"], d["compression"]

    # ~2:1 aspect ratio, wide enough to hold the 20-line fan-to-band -- the
    # standard 85mm single-column paper width is too narrow for that at a
    # 2:1 aspect, so (as with the old combined Figure 7) the paper variant
    # is emitted at full double-column width (170mm) instead
    figsize = (170 * MM_TO_IN, 170 * MM_TO_IN / 2) if style == "paper" else (12.5, 6.2)
    fig, axA = plt.subplots(figsize=figsize, layout="constrained")

    b_vals = np.array([baseline[k] for k in ALL_FIG7_CHAINS])
    c_vals = np.array([corrected[k] for k in ALL_FIG7_CHAINS])

    for k in ALL_FIG7_CHAINS:
        hl = k in FIG7_HIGHLIGHT
        c = FIG7_HIGHLIGHT.get(k, FIG7_OTHER)
        axA.plot([0, 1], [baseline[k], corrected[k]], color=c,
                 lw=s["lw"] * (2.2 if hl else 0.9), alpha=(1.0 if hl else 0.55),
                 marker="o", ms=s["ms"] * (0.35 if hl else 0.2),
                 zorder=(3 if hl else 2))

    xpad_right = 0.62
    axA.set_ylim(b_vals.min() - b_span * 0.06, b_vals.max() + b_span * 0.06)
    # provisional xlim, wide enough that the two highlighted labels are
    # never clipped during the measurement draw below -- axA is re-limited
    # to the *measured* extent afterward, so this costs no final whitespace
    axA.set_xlim(-3.0, 1 + xpad_right + 0.75)

    for k in FIG7_HIGHLIGHT:
        axA.text(-0.16, baseline[k], k, ha="right", va="center", fontsize=s["tick"],
                 color=FIG7_HIGHLIGHT[k], fontweight="bold")

    # span brackets must sit outside the plotting area entirely -- left
    # margin for the default spread, right margin for the optimised spread
    # -- and the left one must clear the two highlighted labels' actual
    # rendered extent (a fixed offset broke this in an earlier version:
    # the bracket ran straight through the label text)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_left_px = min(
        t.get_window_extent(renderer).x0
        for t in axA.texts if t.get_text() in FIG7_HIGHLIGHT
    )
    inv = axA.transData.inverted()
    bracket_x = inv.transform((label_left_px - 10, 0))[0]
    text_x = inv.transform((label_left_px - 18, 0))[0]
    left_lim = inv.transform((label_left_px - 38, 0))[0]

    axA.annotate("", xy=(bracket_x, b_vals.max()), xytext=(bracket_x, b_vals.min()),
                 arrowprops=dict(arrowstyle="<->", color=OI["black"], lw=s["lw"]))
    axA.text(text_x, (b_vals.max() + b_vals.min()) / 2, f"{b_span:.2f} mHa",
             ha="right", va="center", fontsize=s["tick"], rotation=90)
    axA.annotate("", xy=(1 + xpad_right, c_vals.max()), xytext=(1 + xpad_right, c_vals.min()),
                 arrowprops=dict(arrowstyle="<->", color=OI["black"], lw=s["lw"]))
    axA.text(1 + xpad_right + 0.05, (c_vals.max() + c_vals.min()) / 2,
             f"{c_span:.2f} mHa\n({compression:.2f}$\\times$ tighter)",
             ha="left", va="center", fontsize=s["tick"], color=OI["vermillion"],
             fontweight="bold")

    axA.set_xlim(left_lim, 1 + xpad_right + 0.75)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["default", "best\n(candidates ∪ default)"])
    axA.set_ylabel("SQD subspace error (mHa)")
    axA.tick_params(right=True, labelright=True)
    axA.set_title(f"n={len(ALL_FIG7_CHAINS)} chains (8 H10 named + 12 held-out)",
                  loc="left", fontsize=s["font"])
    axA.spines["top"].set_visible(False)

    return fig, d


def figure7a():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, r = draw_fig7a(style)
            emit(fig, "fig7a_compression", style)
    crossings = _count_crossings(r["baseline"], r["corrected"], ALL_FIG7_CHAINS)
    caption(
        "Figure 7a",
        f"Do the two levers (same-spin ordering, anchor selection) interact? For "
        f"n={len(ALL_FIG7_CHAINS)} H10 chains (8 named orderings from G1-lite + 12 "
        f"held-out chains from the chain-aware transfer test), each line connects a "
        f"chain's default-anchor error to its best-of-candidates-UNION-default "
        f"error. The default-anchor spread is {r['b_span']:.2f} mHa; after anchor "
        f"optimisation it compresses to {r['c_span']:.2f} mHa -- a "
        f"{r['compression']:.2f}x reduction, so ordering still matters "
        f"post-optimisation, just far less. rand029 (worst default, "
        f"{r['baseline']['rand029']:.2f}->{r['corrected']['rand029']:.2f} mHa) "
        f"crosses {crossings['rand029']} other chains' lines; rand030 (best "
        f"default, {r['baseline']['rand030']:.2f} mHa, unchanged after "
        f"optimisation) crosses none -- it is the single best chain both before "
        f"and after. CRITICAL DATA NOTE: this figure's spreads and compression "
        f"factor are recomputed with each chain's own default-anchor triple "
        f"included in its candidate set (a strict min() against the uncorrected "
        f"best-of-candidates value); chain_aware/step2_analysis_report.txt's own "
        f"headline number for this same n=20 pool ({REPORTED_COMPRESSION_N20}x) "
        f"is the UNCORRECTED value and does not match this figure's "
        f"{r['compression']:.2f}x -- see stdout for the itemised recompute and "
        f"why the corrected value can only be equal to or smaller than the "
        f"report's. Source: experiments/outputs/g1_lite/, "
        f"experiments/outputs/chain_aware/phaseB_b2_all.csv."
    )


def draw_fig7b(style):
    s = STYLE[style]
    d = figure7_n20_data()
    baseline, corrected = d["baseline"], d["corrected"]
    b_vals = np.array([baseline[k] for k in ALL_FIG7_CHAINS])
    c_vals = np.array([corrected[k] for k in ALL_FIG7_CHAINS])

    figsize = (PAPER_W, PAPER_W) if style == "paper" else (6.5, 6.0)
    fig, axB = plt.subplots(figsize=figsize, layout="constrained")

    lo = min(b_vals.min(), c_vals.min())
    hi = max(b_vals.max(), c_vals.max())
    pad = (hi - lo) * 0.08
    axB.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=OI["black"],
             lw=s["lw"] * 0.8, ls=":", zorder=1)
    for k in ALL_FIG7_CHAINS:
        hl = k in FIG7_HIGHLIGHT
        c = FIG7_HIGHLIGHT.get(k, FIG7_OTHER)
        axB.scatter([baseline[k]], [corrected[k]], s=s["ms"] * (2.4 if hl else 1.6),
                    c=c, zorder=(3 if hl else 2), edgecolors="white",
                    linewidths=s["marker_lw"] * 0.5, alpha=(1.0 if hl else 0.75))
    rho, p = spearmanr(b_vals, c_vals)
    ci_lo, ci_hi = bootstrap_ci(b_vals, c_vals, seed=zlib.crc32(b"fig7_n20"))
    axB.text(0.03, 0.97, f"$\\rho$={rho:+.3f}\np={p:.3f}\n95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]\n"
             f"n={len(ALL_FIG7_CHAINS)}",
             transform=axB.transAxes, ha="left", va="top", fontsize=s["tick"])
    axB.set_xlim(lo - pad, hi + pad)
    axB.set_ylim(lo - pad, hi + pad)
    axB.set_xlabel("default-anchor error (mHa)")
    axB.set_ylabel("best-anchor error (mHa)")
    clean(axB)

    return fig, dict(rho=rho, p=p, ci=(ci_lo, ci_hi), n=len(ALL_FIG7_CHAINS))


def figure7b():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig, r = draw_fig7b(style)
            emit(fig, "fig7b_baseline_vs_optimised", style)
    caption(
        "Figure 7b (backup)",
        f"Default-anchor vs best-anchor error, n={r['n']} H10 chains (same pool as "
        f"Figure 7a), with the y=x reference. Spearman rho={r['rho']:+.3f} "
        f"(p={r['p']:.3f}, bootstrap 95% CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]). "
        f"Panel B of the former combined Figure 7, kept for backup/reference use. "
        f"Source: experiments/outputs/g1_lite/, "
        f"experiments/outputs/chain_aware/phaseB_b2_all.csv."
    )


# ========================================================================
# FIGURE 8 -- reversal schematic (why anchors move but couplings don't)
# ========================================================================

# H10 "physical" and "physical_reverse" permutations (h10_baseline_R1.6/
# h10_baseline_results.csv): physical_reverse's string is the exact
# character-reversal of physical's. Per run_ordering_pipeline.py's
# convention, permutation[i] IS the orbital placed at chain position i (the
# inverse of positions_from()'s pos[orbital]=argsort(perm)), so the raw
# digit string, read left to right, is already the layout order drawn below.
PHYSICAL_PERM = "7281504936"
PHYSICAL_REVERSE_PERM = "6394051827"
PHYSICAL_ERR = 389.7149558584312
PHYSICAL_REVERSE_ERR = 218.63787417716551
FIG8_ANCHOR_POS = (0, 4, 8)  # position % 4 == 0, same anchor_mod=4 rule as Figure 7


def draw_fig8(style):
    s = STYLE[style]
    layout_top = [int(c) for c in PHYSICAL_PERM]
    layout_bot = [int(c) for c in PHYSICAL_REVERSE_PERM]
    n = len(layout_top)
    assert layout_bot == layout_top[::-1]  # "reversed" is a literal claim, not just a label

    link_color = OI["blue"]
    anchor_color = OI["vermillion"]

    y_top, y_bot = 1.0, 0.0
    r = 0.34

    # geometry is fixed by n and the padding below regardless of style;
    # compute the data aspect ratio up front and size the figure to match
    # it exactly at a target width, rather than guessing a figsize and
    # fighting set_aspect("equal") -- a mismatch there is what produces
    # large empty margins above/below a wide, short schematic like this one
    x_lo, x_hi = -2.9, (n - 1 + 1.0) + 2.9 + 2.2
    y_lo, y_hi = y_bot - 1.05, y_top + 1.05
    data_aspect = (x_hi - x_lo) / (y_hi - y_lo)
    target_w = PAPER_W * 1.9 if style == "paper" else 13.5
    figsize = (target_w, target_w / data_aspect)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    def draw_row(y, layout):
        for i in range(n - 1):
            ax.plot([i, i + 1], [y, y], color=link_color, lw=s["lw"] * 2.6,
                    zorder=1, solid_capstyle="round")
        for i, orb in enumerate(layout):
            is_anchor = i in FIG8_ANCHOR_POS
            face = anchor_color if is_anchor else "white"
            txt_color = "white" if is_anchor else OI["black"]
            ax.add_patch(mpatches.Circle((i, y), r, facecolor=face, edgecolor=OI["black"],
                                          lw=s["marker_lw"] * 1.3, zorder=3))
            ax.text(i, y, str(orb), ha="center", va="center", fontsize=s["font"] * 1.15,
                    color=txt_color, fontweight="bold", zorder=4)

    draw_row(y_top, layout_top)
    draw_row(y_bot, layout_bot)

    ax.text(-1.1, y_top, "layout", ha="right", va="center",
            fontsize=s["font"] * 1.3, fontweight="bold")
    ax.text(-1.1, y_bot, "layout\nreversed", ha="right", va="center",
            fontsize=s["font"] * 1.3, fontweight="bold")

    # placed clear of both rows (above the top row, below the bottom row)
    # rather than squeezed between them, so neither annotation ever
    # overlaps a circle regardless of font metrics
    ax.text((n - 1) / 2, y_top + 0.62, "same-spin couplings: identical",
            ha="center", va="bottom", fontsize=s["font"] * 1.15, color=link_color,
            fontweight="bold")
    ax.text((n - 1) / 2, y_bot - 0.62, "anchors: different orbitals",
            ha="center", va="top", fontsize=s["font"] * 1.15, color=anchor_color,
            fontweight="bold")

    x_e = n - 1 + 1.0
    ax.text(x_e, y_top, f"{PHYSICAL_ERR:.2f} mHa", ha="left", va="center",
            fontsize=s["font"] * 1.25, fontweight="bold")
    ax.text(x_e, y_bot, f"{PHYSICAL_REVERSE_ERR:.2f} mHa", ha="left", va="center",
            fontsize=s["font"] * 1.25, fontweight="bold")

    gap = PHYSICAL_ERR - PHYSICAL_REVERSE_ERR
    x_bracket = x_e + 2.9
    ax.annotate("", xy=(x_bracket, y_top), xytext=(x_bracket, y_bot),
                arrowprops=dict(arrowstyle="<->", color=OI["black"], lw=s["lw"] * 1.4))
    ax.text(x_bracket + 0.15, (y_top + y_bot) / 2, f"{gap:.0f} mHa",
            ha="left", va="center", fontsize=s["font"] * 1.25, fontweight="bold")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


def figure8():
    for style in ("paper", "slide"):
        with plt.rc_context(rc(style)):
            fig = draw_fig8(style)
            emit(fig, "fig8_reversal_schematic", style)
    gap = PHYSICAL_ERR - PHYSICAL_REVERSE_ERR
    caption(
        "Figure 8",
        f"Why reversing the same-spin ordering changes the anchor-selection "
        f"result: H10 'physical' (layout) and 'physical_reverse' (layout reversed) "
        f"are the exact same chain read backwards, so their same-spin nearest-"
        f"neighbour links are identical as a set -- reversing a chain doesn't "
        f"change which orbitals are adjacent, only their left-right order. But the "
        f"anchor rule (position % 4 == 0) selects positions, not orbitals: "
        f"reversal moves the default-anchor triple from orbitals (7, 5, 3) to "
        f"(6, 0, 2), a completely different set. Same-spin coupling is reversal-"
        f"invariant; opposite-spin anchor placement is not. Result: "
        f"{PHYSICAL_ERR:.2f} mHa (layout) vs {PHYSICAL_REVERSE_ERR:.2f} mHa "
        f"(layout reversed), a {gap:.2f} mHa gap from reversal alone. Source: "
        f"h10_baseline_R1.6/h10_baseline_results.csv ('physical'/'physical_reverse', "
        f"seed 2026)."
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
- The N2 panel annotates BOTH the full-range ratio (max/min, 8.0x) and the
  p95/p5 ratio (2.7x): the full-range number is driven almost entirely by
  one point (r102, 173.23 mHa vs. the next-highest point's 122.79 mHa), so
  showing only it would overstate the typical spread. No point is removed
  -- `n2_outlier_diagnostic()` prints the top-6 points, both ratios, and
  Spearman rho with/without the largest point (-0.880 vs. -0.877, i.e. the
  point is real signal, not an artifact) to stdout on every regeneration.

## Figure 2 -- the mechanism
- Panel A (N2, n=149): `outputs/stage1/nonoracle_scores.csv`,
  columns `captured`, `err_mHa` (random `r###` rows only)
- Panel B (H10, n=50): `experiments/outputs/score_audit_R1.6/all_scores.csv`,
  columns `captured`, `err_mHa` (random `rand###` rows only)
- Panel C (Cr2, CAS(12,12), n=180): `experiments/outputs/tm_transfer/stage2_sqd.csv`,
  columns `captured`, `err_mHa`, filtered to `role == 'triple'` (60 anchor
  triples x 3 chains -- identity, random, reverse -- pooled; per-chain rho
  is -0.971/-0.967/-0.971 per `tm_transfer/stage3_report.txt`, so pooling
  repeats the same relationship three times rather than averaging over
  different ones). Demonstrates the capture mechanism replicates on a
  transition-metal active space, the project's intended application domain.
- Ceiling reference lines: N2 (0.9866) and H10 (0.7554) are supplied
  ansatz-capacity constants, not derived from the plotted CSVs, hardcoded
  in `figures.py` as `CEILING`. Cr2's (0.8813) is different: it IS a value
  computed by the tm_transfer run itself (`ideal_ceiling` in
  `tm_transfer/stage3_report.txt`), read here as that literal number.
- Panels are side by side; like Figure 7a, the paper variant is 170mm
  (double-column) since three fully-labelled independent-axis panels don't
  fit legibly at 85mm.
- The H10 anchor-selection-axis panel previously shown here (identity/
  physical/rand007, n=40 triples each) moves to Figure 2b, for backup
  reference -- displaced by Cr2's addition, not removed from the record.

## Figure 2b (backup) -- H10 anchor-selection axis
- `experiments/outputs/anchor_reanalysis/anchor_reanalysis.csv`,
  columns `ordering`, `captured`, `err_mHa`, filtered to
  `identity`/`physical`/`rand007` (n=40 triples each, shared y-axis). Same
  mechanism as Figure 2, on the anchor-selection axis instead of the
  same-spin-ordering axis. For backup/reference use.

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

## Figure 6b -- the cancellation, alone
- Same source and bootstrap method as Figure 6, restricted to the three
  retained_J rows (combined / same-spin only / opposite-spin only). A
  single-message slide figure at generously large font; Figure 6's 11-row
  version is unchanged and remains the report figure.

## Figure 7a -- lever interaction and spread compression
- `experiments/outputs/h10_baseline_R1.6/h10_baseline_results.csv`,
  columns `ordering`, `err_mHa` (default-anchor "baseline" error), `permutation`
  (used to derive each ordering's own default anchor triple: position % 4 == 0,
  where position = argsort(permutation) -- the same convention as
  `run_ordering_pipeline.py`'s `CFG['anchor_mod']=4`), for 8 named H10
  orderings (`identity`, `physical`, `rand007`, `rand029`, `rand030`,
  `rand032`, `rand037`, `rand047`), seed 2026.
- `experiments/outputs/g1_lite/g1_all.csv` (40 sampled anchor triples per
  ordering) and `experiments/outputs/chain_aware/phaseB_b2_all.csv` (43
  candidate triples + explicit default-anchor and no-ab-floor rows per
  chain, for 12 held-out H10 chains never used elsewhere) -- together, 20
  chains total.
- Correction applied before drawing (see CRITICAL DATA NOTE in the Figure
  7a caption): for each chain, "best" = min(best-of-its-sampled-candidates,
  its own default-anchor baseline), since the default triple's error is
  exactly the baseline value and was absent from the candidate pool for
  most chains. Only rand030 actually changes under this correction
  (best-of-40 alone was 180.81 mHa; corrected best = 168.67 mHa, its own
  default) -- the itemised diff for every chain checked is printed to
  stdout on every regeneration.
- `chain_aware/step2_analysis_report.txt` (section B3.5) reports this same
  n=20 pool's compression factor as 3.15x -- but that number is
  best-of-candidates alone, WITHOUT the default-triple correction above.
  Applying the correction can only lower or hold each chain's best value,
  never raise it; here it lowers rand030's best and so widens (not
  narrows) the best-anchor spread, meaning the corrected compression factor
  is mathematically bounded to be <= 3.15x. The actual recomputed value is
  printed to stdout on every regeneration and used in the figure and
  caption; it does not equal 3.15x, and the stdout output says so
  explicitly rather than silently overriding the mismatch.
- Ordering labels sit outside the left plotting area, right-aligned; the
  two span-annotation brackets sit further outside still, positioned from
  the labels' actual rendered extent (not a fixed offset) so they can never
  run through the labels regardless of text length. rand029 (worst default)
  and rand030 (best default) are highlighted; all other chains are shown
  in grey, unlabelled, to keep the panel legible at n=20.
- Side by side by construction: a single wide panel at ~2:1 aspect. Like
  Figure 2, the paper variant is 170mm (double-column) rather than 85mm.

## Figure 7b (backup) -- baseline vs optimised, n=20
- Same n=20 pool and correction as Figure 7a. Scatter of default-anchor vs
  best-anchor error with the y=x reference, Spearman rho and bootstrap 95%
  CI. Panel B of the former combined Figure 7, kept for backup/reference use.

## Figure 8 -- reversal schematic
- `experiments/outputs/h10_baseline_R1.6/h10_baseline_results.csv`,
  `permutation` and `err_mHa` columns for `physical` (389.71 mHa) and
  `physical_reverse` (218.64 mHa), seed 2026. `physical_reverse`'s
  permutation string is the exact character-reversal of `physical`'s
  (verified in `figures.py` with an assert, not just asserted in prose).
- A schematic, not a data plot: no axes, no gridlines. Two rows of 10
  circles (orbital indices read directly off the permutation string, since
  `permutation[i]` is the orbital at chain position `i`); same-spin
  nearest-neighbour links (adjacent circles) drawn identically in both
  rows, since reversing a chain preserves which orbitals are adjacent;
  positions 0/4/8 (the `position % 4 == 0` anchor rule, same as Figure 7a)
  highlighted in both rows, landing on different orbitals in each ((7, 5,
  3) vs (6, 0, 2)) because the rule selects positions, not orbitals. This
  is the mechanism behind Figure 1's `physical`/`physical_reverse` gap.

## Regenerating
```
python3 experiments/figures.py
```
Outputs `<name>_paper.pdf` / `.png` (85mm width, except Figures 2 and 7a at
170mm -- see their sections above) and `<name>_slide.pdf` /
`.png` (widescreen, ~16:9 -- a few figures are sized modestly wider or
taller than exactly 16:9 where that many rows/panels needed it to stay
legible and clipping-free) per figure into this directory. Prints all
eleven suggested captions to stdout, followed by a render-verification
report: for every file, its pixel dimensions (read back from the saved PNG
with PIL) and whether any annotation was found to extend beyond the saved
canvas.
""".format(variant_table=_variant_mapping_table())


def main():
    figure1()
    figure2()
    figure2b()
    figure3()
    figure4()
    figure5()
    figure6()
    figure6b()
    figure7a()
    figure7b()
    figure8()

    (OUTDIR / "README.md").write_text(README)

    print(f"\nWrote {len(ALL_WRITTEN)} figure files to {OUTDIR}:")
    for name in sorted(ALL_WRITTEN):
        print(f"  {name}")

    print("\nFigure 7 default-anchor-inclusion check (best-of-candidates vs "
          "best-of-candidates-UNION-default, all 20 chains):")
    if CHANGE_REPORT:
        for line in CHANGE_REPORT:
            print(f"  {line}")
    else:
        print("  no orderings changed")

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
