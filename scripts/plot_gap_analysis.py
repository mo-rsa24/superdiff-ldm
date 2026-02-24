#!/usr/bin/env python3
"""
Gap analysis visualisation suite — Plots 00 through 13.

Each plot builds on the previous to form a progressive statistical narrative:

  Terminal distributions  (data: per_seed_distances.json)
    00  Strip by pair     — raw dots (one per seed) + bold mean tick, 4 panels  ← foundation
    01  Grouped bar       — mean ± std per pair per condition
    02  Hist grouped      — x=distance, grouped bars per bin (colour=pair), 3 condition panels
    03  Pooled strip      — all seeds pooled across pairs, bold tick=mean, colour=condition
    04  KDE pooled        — smooth of plot 03; all pairs combined, colour=condition
    05  KDE by pair       — 4 panels, one per concept pair, colour=condition

  Temporal dynamics  (data: trajectory_distances.json)
    06  Stacked bar pooled  — incremental divergence per time bin, pooled across all pairs
    07  Stacked bar by pair — same, one group per concept pair
    08  Per-condition       — 3 subplots, 4 pair-coloured mean ± std bands
    09  Individual seeds    — same as 08, thin per-seed traces + bold mean
    10  Generalised         — 3 subplots (one per condition), all pooled,
                              mean with ±1 std outer band and 95 % CI inner band

  Closing the gap  (data: per_seed_distances.json + all_pairs_gap.json)
    11  p* strip          — terminal distance from AND: p*, mono, solo c₁, solo c₂
    12  CLIP comparison   — per-pair CLIP sim AND↔p* vs AND↔mono
    13  JS divergence     — JS²(P_p*, P_mono) per pair; lower = p* and mono are more alike

Usage
-----
# All plots:
conda run -n superdiff python scripts/plot_gap_analysis.py

# Single plot (both forms accepted):
conda run -n superdiff python scripts/plot_gap_analysis.py --plot 06
conda run -n superdiff python scripts/plot_gap_analysis.py --plot 11

# Custom data / output directories:
conda run -n superdiff python scripts/plot_gap_analysis.py \\
    --data-dir   experiments/inversion/gap_analysis \\
    --output-dir experiments/inversion/gap_analysis/figures \\
    --plot all

Notes
-----
Plots 11 and 13 require d_T_pstar in per_seed_distances.json.
Re-run measure_composability_gap.py after the code update to populate that column.
Plot 12 uses all_pairs_gap.json (CLIP scores already computed in the main run).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("Warning: scipy not found — plots 04, 05, 13 require it (pip install scipy).")


# ---------------------------------------------------------------------------
# Colour / label constants
# ---------------------------------------------------------------------------

TERM_COLOR = {
    "d_T_mono":  "#E15759",   # red
    "d_T_c1":    "#4E79A7",   # blue
    "d_T_c2":    "#59A14F",   # green
    "d_T_pstar": "#9467BD",   # purple — the "close it" condition
}
TERM_LABEL = {
    "d_T_mono":  "Monolithic",
    "d_T_c1":    "Solo c₁",
    "d_T_c2":    "Solo c₂",
    "d_T_pstar": "p* (inverter)",
}
TRAJ_COLOR = {
    "d_t_mono":  "#E15759",
    "d_t_c1":    "#4E79A7",
    "d_t_c2":    "#59A14F",
    "d_t_pstar": "#9467BD",
}
TRAJ_LABEL = {
    "d_t_mono":  "Monolithic",
    "d_t_c1":    "Solo c₁",
    "d_t_c2":    "Solo c₂",
    "d_t_pstar": "p* (inverter)",
}

TERM_CONDITIONS = ["d_T_mono", "d_T_c1", "d_T_c2"]
TRAJ_CONDITIONS = ["d_t_mono", "d_t_c1", "d_t_c2"]

# Four visually distinct hues — one per concept pair
PAIR_PALETTE = ["#F28E2B", "#76B7B2", "#B07AA1", "#9C755F"]

# Time bins for stacked temporal bar charts
# 5 equal-width windows; sum of incremental heights = terminal MSE (since d_0 = 0)
STEP_BINS  = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
BIN_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50"]
BIN_ALPHAS = [0.92, 0.74, 0.55, 0.36, 0.18]   # opaque = early, transparent = late


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pair_color_map(pairs):
    return {p: PAIR_PALETTE[i % len(PAIR_PALETTE)] for i, p in enumerate(sorted(pairs))}


def short_pair(pair: str) -> str:
    """'a cat + a dog'  →  'cat + dog'  (strips leading articles)."""
    def strip_article(s):
        for art in ("a ", "an ", "the "):
            if s.lower().startswith(art):
                return s[len(art):]
        return s
    return " + ".join(strip_article(p) for p in pair.split(" + "))


def kde_pmf(vals: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE (Scott bandwidth) normalised to a sum-to-one PMF."""
    d = np.maximum(stats.gaussian_kde(vals, bw_method="scott")(x_grid), 1e-12)
    return d / d.sum()


def traj_stats(df: pd.DataFrame, cond: str, steps):
    """Return (means, stds) arrays aligned to `steps` for a trajectory column."""
    agg = df.groupby("step")[cond].agg(["mean", "std"]).reindex(steps)
    return agg["mean"].values, agg["std"].values


def bin_increments(df_traj: pd.DataFrame, cond: str) -> list:
    """
    For each bin in STEP_BINS compute mean(d_at_t2) - mean(d_at_t1) across all
    rows in df_traj.  Since d_0 = 0 (shared noise seed), the sum of all
    increments equals the mean terminal distance d_T.
    """
    increments = []
    for (t1, t2) in STEP_BINS:
        rows1 = df_traj[df_traj["step"] == t1]
        rows2 = df_traj[df_traj["step"] == t2]
        d1 = rows1[cond].mean() if len(rows1) > 0 else 0.0
        d2 = rows2[cond].mean() if len(rows2) > 0 else 0.0
        increments.append(max(d2 - d1, 0.0))
    return increments


def hide_top_right(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_terminal(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "per_seed_distances.json"
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun measure_composability_gap.py first.")
    return pd.DataFrame(json.loads(p.read_text()))


def load_trajectory(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "trajectory_distances.json"
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun measure_composability_gap.py first.")
    return pd.DataFrame(json.loads(p.read_text()))


def load_all_pairs_gap(data_dir: Path):
    p = data_dir / "all_pairs_gap.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ===========================================================================
# Plot 00 — Strip chart: raw dots (one per seed) + bold mean tick, per pair
#           Foundation — shows raw data before any aggregation.
# ===========================================================================

def plot_00(df_term, df_traj, out_dir, **kw):
    pairs = sorted(df_term["pair"].unique())
    rng   = np.random.default_rng(42)
    x_pos = {c: float(i) for i, c in enumerate(TERM_CONDITIONS)}

    y_max = df_term[TERM_CONDITIONS].values.max()
    y_top = max(y_max * 1.08, 2.15)

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 5.5), sharey=True)

    for ax, pair in zip(axes, pairs):
        sub = df_term[df_term["pair"] == pair]

        for cond in TERM_CONDITIONS:
            vals   = sub[cond].values
            x      = x_pos[cond]
            jitter = rng.uniform(-0.13, 0.13, len(vals))
            c      = TERM_COLOR[cond]

            ax.scatter(x + jitter, vals, color=c, s=55, alpha=0.72,
                       linewidths=1.2, edgecolors="white", zorder=3)
            ax.plot([x - 0.24, x + 0.24], [vals.mean(), vals.mean()],
                    color=c, lw=4.5, solid_capstyle="butt", zorder=4)

        ax.axhline(2.0, color="#CCCCCC", lw=1.2, ls=":", zorder=1)
        ax.set_title(short_pair(pair), fontsize=11)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Mono", "Solo c₁", "Solo c₂"], fontsize=10)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    axes[0].set_ylabel("Distance from AND  (per-element MSE)", fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=TERM_COLOR[c], markersize=9,
               markeredgecolor="white", markeredgewidth=1.2,
               label=TERM_LABEL[c])
        for c in TERM_CONDITIONS
    ]
    fig.legend(handles=legend_handles, title="Condition",
               loc="upper right", fontsize=10, title_fontsize=11,
               frameon=True, bbox_to_anchor=(1.0, 0.93))

    fig.suptitle(
        "Plot 00 — Terminal Latent Distance from SuperDiff-AND  (per Concept Pair)\n"
        "Bold tick = mean across seeds.  Each dot = one seed.",
        fontsize=13,
    )
    fig.text(0.5, -0.02,
             "Scale:  0 = identical to AND  │  ≈0.25 subtle  │  ≈0.60 noticeable  "
             "│  ≈2.0 unrelated latents (ceiling)",
             ha="center", fontsize=9, color="#888888")

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_00_strip_by_pair.png")


# ===========================================================================
# Plot 01 — Grouped bar: mean ± std, x = pair, grouped by condition
# ===========================================================================

def plot_01(df_term, df_traj, out_dir, **kw):
    pairs   = sorted(df_term["pair"].unique())
    x       = np.arange(len(pairs))
    width   = 0.24
    offsets = np.array([-1, 0, 1]) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, cond in enumerate(TERM_CONDITIONS):
        means = [df_term[df_term["pair"] == p][cond].mean() for p in pairs]
        stds  = [df_term[df_term["pair"] == p][cond].std()  for p in pairs]
        ax.bar(x + offsets[i], means, width * 0.9,
               yerr=stds, capsize=4, error_kw={"lw": 1.5, "ecolor": "#444"},
               color=TERM_COLOR[cond], label=TERM_LABEL[cond], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_pair(p) for p in pairs], fontsize=11)
    ax.set_ylabel("Per-element MSE  (z_T^cond − z_T^AND)", fontsize=11)
    ax.set_title(
        "Plot 01 — Terminal Latent Distance from SuperDiff-AND\n"
        "Grouped bar: mean ± 1 std  (16 seeds per pair)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_01_terminal_bars.png")


# ===========================================================================
# Plot 02 — Grouped bar histogram: x=distance bins, bars grouped by pair,
#           3 condition panels.  Shifts x-axis from pair → distance; no overlap.
# ===========================================================================

def plot_02(df_term, df_traj, out_dir, **kw):
    pairs   = sorted(df_term["pair"].unique())
    pcmap   = pair_color_map(pairs)
    n_pairs = len(pairs)

    all_vals  = df_term[TERM_CONDITIONS].values.flatten()
    bin_edges = np.linspace(all_vals.min() * 0.85, all_vals.max() * 1.08, 10)  # 9 bins
    bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_w     = bin_edges[1] - bin_edges[0]

    # Each pair gets an equal sub-bar within the bin; 5 % gap between bins
    bar_w    = bin_w * 0.92 / n_pairs
    p_offset = (np.arange(n_pairs) - (n_pairs - 1) / 2) * bar_w

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, sharex=True)

    for ax, cond in zip(axes, TERM_CONDITIONS):
        for pi, pair in enumerate(pairs):
            vals   = df_term[df_term["pair"] == pair][cond].values
            counts, _ = np.histogram(vals, bins=bin_edges)
            ax.bar(bin_ctrs + p_offset[pi], counts, bar_w,
                   color=pcmap[pair], label=short_pair(pair),
                   alpha=0.87, edgecolor="white", linewidth=0.6)

        ax.set_title(TERM_LABEL[cond], fontsize=12, color=TERM_COLOR[cond])
        ax.set_xlabel("Per-element MSE", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    axes[0].set_ylabel("Count  (seeds)", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper right", title="Concept pair")

    fig.suptitle(
        "Plot 02 — Terminal Distance: Grouped Bar Histogram  (colour = concept pair)\n"
        "One panel per condition,  9 bins,  16 seeds per pair",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_02_histogram_grouped.png")


# ===========================================================================
# Plot 03 — Pooled strip: all seeds from all pairs, bold tick = pooled mean,
#           colour = condition.  Direct bridge from per-pair view to KDE.
# ===========================================================================

def plot_03(df_term, df_traj, out_dir, **kw):
    rng   = np.random.default_rng(42)
    x_pos = {c: float(i) for i, c in enumerate(TERM_CONDITIONS)}

    y_max = df_term[TERM_CONDITIONS].values.max()
    y_top = max(y_max * 1.08, 2.15)

    n_pairs = df_term["pair"].nunique()
    n_seeds = df_term["seed"].nunique()
    n_total = len(df_term)   # dots per condition

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cond in TERM_CONDITIONS:
        vals   = df_term[cond].values   # all n_total values (all pairs pooled)
        x      = x_pos[cond]
        jitter = rng.uniform(-0.22, 0.22, len(vals))
        c      = TERM_COLOR[cond]

        ax.scatter(x + jitter, vals, color=c, s=45, alpha=0.50,
                   linewidths=1.0, edgecolors="white", zorder=3)

        # Bold mean tick (pooled across all pairs)
        ax.plot([x - 0.32, x + 0.32], [vals.mean(), vals.mean()],
                color=c, lw=5.5, solid_capstyle="butt", zorder=4)

    ax.axhline(2.0, color="#CCCCCC", lw=1.2, ls=":", zorder=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Mono", "Solo c₁", "Solo c₂"], fontsize=12)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel("Distance from AND  (per-element MSE)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    ax.set_title(
        "Plot 03 — Pooled Terminal Distance  (all concept pairs combined)\n"
        f"Bold tick = mean across all seeds & pairs.  "
        f"N = {n_total} dots per condition  ({n_pairs} pairs × {n_seeds} seeds)",
        fontsize=12,
    )
    ax.text(0.5, -0.08,
            "Scale:  0 = identical to AND  │  ≈0.25 subtle  │  ≈0.60 noticeable  "
            "│  ≈2.0 ceiling",
            ha="center", transform=ax.transAxes, fontsize=9, color="#888888")

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_03_pooled_strip.png")


# ===========================================================================
# Plot 04 — KDE pooled: smooth of plot 03; all pairs combined, colour=condition
# ===========================================================================

def plot_04(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 04 (scipy required).")
        return

    all_vals = df_term[TERM_CONDITIONS].values.flatten()
    x_grid   = np.linspace(all_vals.min() * 0.85, all_vals.max() * 1.08, 600)
    n        = len(df_term)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in TERM_CONDITIONS:
        vals = df_term[cond].values
        dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
        ax.plot(x_grid, dens, color=TERM_COLOR[cond], label=TERM_LABEL[cond], lw=2.5)
        ax.fill_between(x_grid, dens, alpha=0.13, color=TERM_COLOR[cond])
        ax.axvline(vals.mean(), color=TERM_COLOR[cond], lw=1.2, ls="--", alpha=0.7)

    ax.set_xlabel("Per-element MSE  (z_T^cond − z_T^AND)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Plot 04 — Terminal Distance Distribution (KDE, pooled)\n"
        f"All concept pairs combined  (N={n} per condition, Scott bandwidth)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_04_kde_pooled.png")


# ===========================================================================
# Plot 05 — KDE faceted: one panel per concept pair, colour = condition
# ===========================================================================

def plot_05(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 05 (scipy required).")
        return

    pairs = sorted(df_term["pair"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.5), sharey=False)

    for ax, pair in zip(axes, pairs):
        sub      = df_term[df_term["pair"] == pair]
        all_vals = sub[TERM_CONDITIONS].values.flatten()
        x_grid   = np.linspace(all_vals.min() * 0.85, all_vals.max() * 1.08, 400)

        for cond in TERM_CONDITIONS:
            vals = sub[cond].values
            dens = stats.gaussian_kde(vals, bw_method="scott")(x_grid)
            ax.plot(x_grid, dens, color=TERM_COLOR[cond], label=TERM_LABEL[cond], lw=2)
            ax.fill_between(x_grid, dens, alpha=0.12, color=TERM_COLOR[cond])

        ax.set_title(short_pair(pair), fontsize=11)
        ax.set_xlabel("MSE", fontsize=10)
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel("Density", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "Plot 05 — Per-Pair Terminal Distance Distribution (KDE)\n"
        "One panel per concept pair,  colour = condition",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_05_kde_by_pair.png")


# ===========================================================================
# Plot 06 — Stacked temporal bar (pooled across all pairs + seeds).
#
#   x = condition (Mono / Solo c₁ / Solo c₂)
#   Stack = 5 time bins (0-10, 10-20, 20-30, 30-40, 40-50)
#   Each segment height = mean incremental divergence Δd over that 10-step window
#   Bar total height = terminal distance d_T  (because d_0 = 0)
#   Alpha gradient: opaque bottom (early) → transparent top (late)
# ===========================================================================

def plot_06(df_term, df_traj, out_dir, **kw):
    conditions = TRAJ_CONDITIONS
    x_pos = {c: float(i) for i, c in enumerate(conditions)}
    bar_width = 0.52

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for cond in conditions:
        x   = x_pos[cond]
        c   = TRAJ_COLOR[cond]
        inc = bin_increments(df_traj, cond)

        bottom = 0.0
        for bi, height in enumerate(inc):
            ax.bar(x, height, bar_width, bottom=bottom,
                   color=c, alpha=BIN_ALPHAS[bi],
                   edgecolor="white", linewidth=0.8)
            bottom += height

        # Annotate total (= mean terminal MSE)
        ax.text(x, bottom + ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 0 else bottom + 0.005,
                f"{bottom:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=c, fontweight="bold")

    # Recompute y-limit now that bars are placed
    ax.relim(); ax.autoscale_view()
    ymax = ax.get_ylim()[1]
    # Patch the annotations — rewrite them at correct y offset
    for text in ax.texts:
        text.remove()
    for cond in conditions:
        x   = x_pos[cond]
        c   = TRAJ_COLOR[cond]
        inc = bin_increments(df_traj, cond)
        total = sum(inc)
        ax.text(x, total + ymax * 0.015, f"{total:.3f}",
                ha="center", va="bottom", fontsize=9.5, color=c, fontweight="bold")

    ax.set_xticks([x_pos[c] for c in conditions])
    ax.set_xticklabels([TRAJ_LABEL[c] for c in conditions], fontsize=12)
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylabel("Cumulative MSE from AND  (incremental per time bin)", fontsize=11)
    ax.set_title(
        "Plot 06 — Temporal Divergence from AND  (stacked bar, pooled)\n"
        "Each segment = mean incremental divergence Δd over a 10-step window.\n"
        "Opaque = early denoising steps;  transparent = late.",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    # Legend for time bins
    bin_handles = [
        Patch(facecolor="#777777", alpha=BIN_ALPHAS[bi],
              edgecolor="white", label=f"Steps {BIN_LABELS[bi]}")
        for bi in range(len(STEP_BINS))
    ]
    ax.legend(handles=bin_handles, title="Time bin", fontsize=9,
              loc="upper right", title_fontsize=10)

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_06_stacked_bar_pooled.png")


# ===========================================================================
# Plot 07 — Stacked temporal bar by concept pair.
#   4 groups (one per pair), 3 bars each (conditions), 5 stacked time-bin segments.
#   Inter-group spacing keeps groups visually separated.
# ===========================================================================

def plot_07(df_term, df_traj, out_dir, **kw):
    pairs      = sorted(df_traj["pair"].unique())
    conditions = TRAJ_CONDITIONS
    n_cond     = len(conditions)
    n_pairs    = len(pairs)

    bar_width   = 0.20
    cond_offsets = np.array([-1, 0, 1]) * bar_width   # 3 bars per group

    fig, ax = plt.subplots(figsize=(4.5 * n_pairs, 5.5))

    group_x = np.arange(n_pairs, dtype=float)

    for pi, pair in enumerate(pairs):
        sub = df_traj[df_traj["pair"] == pair]
        for ci, cond in enumerate(conditions):
            x   = group_x[pi] + cond_offsets[ci]
            c   = TRAJ_COLOR[cond]
            inc = bin_increments(sub, cond)

            bottom = 0.0
            for bi, height in enumerate(inc):
                ax.bar(x, height, bar_width * 0.92, bottom=bottom,
                       color=c, alpha=BIN_ALPHAS[bi],
                       edgecolor="white", linewidth=0.5)
                bottom += height

    ax.set_xticks(group_x)
    ax.set_xticklabels([short_pair(p) for p in pairs], fontsize=11)
    ax.set_xlim(-0.5, n_pairs - 0.5)
    ax.set_ylabel("Cumulative MSE from AND  (incremental per time bin)", fontsize=11)
    ax.set_title(
        "Plot 07 — Temporal Divergence from AND  (stacked bar, by concept pair)\n"
        "Opaque = early denoising steps;  transparent = late.  "
        "Bar height = mean terminal MSE d_T.",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    # Condition colour legend
    cond_handles = [
        Patch(facecolor=TRAJ_COLOR[c], alpha=0.88,
              edgecolor="white", label=TRAJ_LABEL[c])
        for c in conditions
    ]
    # Time bin alpha legend
    bin_handles = [
        Patch(facecolor="#777777", alpha=BIN_ALPHAS[bi],
              edgecolor="white", label=f"Steps {BIN_LABELS[bi]}")
        for bi in range(len(STEP_BINS))
    ]

    leg1 = ax.legend(handles=cond_handles, title="Condition",
                     fontsize=9, loc="upper right", title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(handles=bin_handles, title="Time bin",
              fontsize=9, loc="upper left", title_fontsize=10)

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_07_stacked_bar_by_pair.png")


# ===========================================================================
# Plot 08 — Per-condition: 3 subplots, pair-coloured mean ± std bands
# ===========================================================================

def plot_08(df_term, df_traj, out_dir, **kw):
    pairs = sorted(df_traj["pair"].unique())
    pcmap = pair_color_map(pairs)
    steps = sorted(df_traj["step"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, cond in zip(axes, TRAJ_CONDITIONS):
        for pair in pairs:
            c   = pcmap[pair]
            sub = df_traj[df_traj["pair"] == pair]
            means, stds = traj_stats(sub, cond, steps)
            ax.plot(steps, means, color=c, label=short_pair(pair), lw=2)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=c)
        ax.set_title(TRAJ_LABEL[cond], fontsize=12, color=TRAJ_COLOR[cond])
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel("Per-element MSE", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper left", title="Concept pair")
    fig.suptitle(
        "Plot 08 — Per-Condition Temporal Detail  (Mean ± Std,  colour = pair)",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_08_per_condition_temporal.png")


# ===========================================================================
# Plot 09 — Individual seed trajectories (thin α) + bold mean, colour = pair
# ===========================================================================

def plot_09(df_term, df_traj, out_dir, **kw):
    pairs = sorted(df_traj["pair"].unique())
    pcmap = pair_color_map(pairs)
    steps = sorted(df_traj["step"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, cond in zip(axes, TRAJ_CONDITIONS):
        for (pair, _seed), grp in df_traj.groupby(["pair", "seed"]):
            grp = grp.sort_values("step")
            ax.plot(grp["step"], grp[cond], color=pcmap[pair], lw=0.6, alpha=0.22)

        for pair in pairs:
            c     = pcmap[pair]
            sub   = df_traj[df_traj["pair"] == pair]
            means = sub.groupby("step")[cond].mean().reindex(steps).values
            ax.plot(steps, means, color=c, lw=2.5, label=short_pair(pair))

        ax.set_title(TRAJ_LABEL[cond], fontsize=12, color=TRAJ_COLOR[cond])
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel("Per-element MSE", fontsize=11)
    axes[-1].legend(fontsize=9, loc="upper left", title="Concept pair")
    fig.suptitle(
        "Plot 09 — Individual Seed Trajectories (thin, α=0.22) + Mean (bold)\n"
        "Colour = concept pair",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_09_individual_seeds.png")


# ===========================================================================
# Plot 10 — Generalised: 3 subplots (one per condition), all pairs+seeds pooled
# ===========================================================================

def plot_10(df_term, df_traj, out_dir, **kw):
    steps      = sorted(df_traj["step"].unique())
    n_per_step = int(df_traj[df_traj["step"] == steps[0]].shape[0])
    n_pairs    = df_traj["pair"].nunique()
    n_seeds    = df_traj["seed"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, cond in zip(axes, TRAJ_CONDITIONS):
        means, stds = traj_stats(df_traj, cond, steps)
        ci95 = 1.96 * stds / np.sqrt(n_per_step)
        c    = TRAJ_COLOR[cond]

        ax.plot(steps, means, color=c, lw=2.5)
        ax.fill_between(steps, means - stds,  means + stds,
                        alpha=0.12, color=c, label="±1 std")
        ax.fill_between(steps, means - ci95,  means + ci95,
                        alpha=0.30, color=c, label="95 % CI")

        ax.set_title(TRAJ_LABEL[cond], fontsize=13, color=c, fontweight="bold")
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    axes[0].set_ylabel("Per-element MSE from AND", fontsize=11)
    fig.suptitle(
        "Plot 10 — Generalised Temporal Distance (All Pairs + Seeds Pooled)\n"
        f"N = {n_pairs} pairs × {n_seeds} seeds = {n_pairs * n_seeds} trajectories per condition",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_10_generalised.png")


# ===========================================================================
# Plot 11 — p* terminal strip: terminal distance from AND for ALL conditions
#            including p* (the inverter output).
#            Gracefully skipped if d_T_pstar is absent — re-run measure script.
# ===========================================================================

def plot_11(df_term, df_traj, out_dir, **kw):
    if "d_T_pstar" not in df_term.columns:
        print("  Skipping plot 11: 'd_T_pstar' column not found in per_seed_distances.json.")
        print("  Re-run measure_composability_gap.py with the updated code to generate it.")
        return

    all_conds = ["d_T_pstar", "d_T_mono", "d_T_c1", "d_T_c2"]
    pairs  = sorted(df_term["pair"].unique())
    rng    = np.random.default_rng(42)
    x_pos  = {c: float(i) for i, c in enumerate(all_conds)}

    all_vals = df_term[all_conds].values
    y_top    = max(all_vals.max() * 1.12, 0.60)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for cond in all_conds:
        vals   = df_term[cond].values
        x      = x_pos[cond]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        c      = TERM_COLOR[cond]

        ax.scatter(x + jitter, vals, color=c, s=50, alpha=0.55,
                   linewidths=1.1, edgecolors="white", zorder=3)
        ax.plot([x - 0.30, x + 0.30], [vals.mean(), vals.mean()],
                color=c, lw=5.0, solid_capstyle="butt", zorder=4)

    # Annotate mean values above each tick
    for cond in all_conds:
        x = x_pos[cond]
        m = df_term[cond].mean()
        ax.text(x, m + y_top * 0.03, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9, color=TERM_COLOR[cond])

    ax.set_xticks([x_pos[c] for c in all_conds])
    ax.set_xticklabels(["p* (inverter)", "Monolithic", "Solo c₁", "Solo c₂"], fontsize=11)
    ax.set_xlim(-0.6, len(all_conds) - 0.4)
    ax.set_ylim(bottom=0, top=y_top)
    ax.set_ylabel("Distance from AND  (per-element MSE)", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)

    n_pairs = df_term["pair"].nunique()
    n_seeds = df_term["seed"].nunique()
    ax.set_title(
        "Plot 11 — Closing the Gap: p* vs Baselines  (terminal latent distance from AND)\n"
        f"Bold tick = mean.  N = {len(df_term)} ({n_pairs} pairs × {n_seeds} seeds)",
        fontsize=12,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=TERM_COLOR[c], markersize=9,
               markeredgecolor="white", markeredgewidth=1.2,
               label=TERM_LABEL[c])
        for c in all_conds
    ]
    ax.legend(handles=legend_handles, title="Condition",
              fontsize=10, title_fontsize=11, frameon=True,
              loc="upper right")

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_11_pstar_strip.png")


# ===========================================================================
# Plot 12 — CLIP similarity comparison: AND↔p* vs AND↔mono, per concept pair.
#            Uses all_pairs_gap.json — no re-run needed.
# ===========================================================================

def plot_12(df_term, df_traj, out_dir, data_dir=None, **kw):
    if data_dir is None:
        print("  Skipping plot 12: data_dir not provided.")
        return

    gap_data = load_all_pairs_gap(data_dir)
    if gap_data is None:
        print("  Skipping plot 12: all_pairs_gap.json not found in data_dir.")
        return

    # "pair" field in JSON is a list [c1, c2] (serialised from a Python tuple)
    pairs      = [f"{r['pair'][0]} + {r['pair'][1]}" for r in gap_data]
    clip_pstar = [r["gap_and_pstar"]["clip_cos"] for r in gap_data]
    clip_mono  = [r["gap_and_mono"]["clip_cos"]  for r in gap_data]

    x     = np.arange(len(pairs))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_p = ax.bar(x - width / 2, clip_pstar, width,
                    color=TERM_COLOR["d_T_pstar"], label="AND ↔ p* (inverter)", alpha=0.85)
    bars_m = ax.bar(x + width / 2, clip_mono,  width,
                    color=TERM_COLOR["d_T_mono"],  label="AND ↔ Monolithic",    alpha=0.85)

    for bar in list(bars_p) + list(bars_m):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([short_pair(p) for p in pairs], fontsize=11)
    ax.set_ylabel("CLIP cosine similarity  (↑ better)", fontsize=11)
    ax.set_ylim(top=min(1.0, max(clip_pstar + clip_mono) * 1.10))
    ax.set_title(
        "Plot 12 — CLIP Similarity to AND Reference: p* vs Monolithic  (per concept pair)\n"
        "Higher = generated image closer to SuperDiff-AND in CLIP space",
        fontsize=13,
    )
    ax.legend(fontsize=11)

    # Horizontal dashed lines at mean
    ax.axhline(np.mean(clip_pstar), color=TERM_COLOR["d_T_pstar"],
               lw=1.5, ls="--", alpha=0.55)
    ax.axhline(np.mean(clip_mono), color=TERM_COLOR["d_T_mono"],
               lw=1.5, ls="--", alpha=0.55)
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_12_clip_comparison.png")


# ===========================================================================
# Plot 13 — JS² divergence: p* vs mono terminal distance distributions.
#            Lower JS² = p* and mono arrive at more similar latent neighbourhoods.
#            Gracefully skipped if d_T_pstar is absent.
# ===========================================================================

def plot_13(df_term, df_traj, out_dir, **kw):
    if not SCIPY_OK:
        print("  Skipping plot 13 (scipy required).")
        return
    if "d_T_pstar" not in df_term.columns:
        print("  Skipping plot 13: 'd_T_pstar' column not found in per_seed_distances.json.")
        print("  Re-run measure_composability_gap.py with the updated code to generate it.")
        return

    pairs    = sorted(df_term["pair"].unique())
    all_vals = np.concatenate([df_term["d_T_pstar"].values,
                                df_term["d_T_mono"].values])
    x_grid   = np.linspace(all_vals.min() * 0.70, all_vals.max() * 1.15, 1000)

    jsd = []
    for pair in pairs:
        sub     = df_term[df_term["pair"] == pair]
        p_pstar = kde_pmf(sub["d_T_pstar"].values, x_grid)
        p_mono  = kde_pmf(sub["d_T_mono"].values,  x_grid)
        jsd.append(jensenshannon(p_pstar, p_mono) ** 2)

    x     = np.arange(len(pairs))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(x, jsd, width,
                  color=TERM_COLOR["d_T_pstar"], alpha=0.85, edgecolor="white")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([short_pair(p) for p in pairs], fontsize=11)
    ax.set_ylabel("JS²(P_p*,  P_mono)  (↓ better)", fontsize=11)
    ax.set_title(
        "Plot 13 — Closing the Gap: Distribution Similarity of p* vs Monolithic\n"
        "JS²(P_p*, P_mono) per concept pair — lower = inverter produces similar latent "
        "distribution to the naive monolithic baseline",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    hide_top_right(ax)

    mean_jsd = np.mean(jsd)
    ax.axhline(mean_jsd, color=TERM_COLOR["d_T_pstar"], lw=1.5, ls="--", alpha=0.7)
    ax.text(len(pairs) - 0.5, mean_jsd + ax.get_ylim()[1] * 0.01,
            f"mean = {mean_jsd:.3f}", ha="right", va="bottom",
            fontsize=9, color=TERM_COLOR["d_T_pstar"])

    fig.tight_layout()
    save_fig(fig, out_dir / "plot_13_js_pstar_vs_mono.png")


# ===========================================================================
# Registry and CLI
# ===========================================================================

PLOTS = {
    "00": ("Strip chart by pair (foundation)",              plot_00),
    "01": ("Terminal grouped bars",                         plot_01),
    "02": ("Grouped bar histogram by pair",                 plot_02),
    "03": ("Pooled strip (all pairs combined)",             plot_03),
    "04": ("KDE pooled",                                    plot_04),
    "05": ("KDE by pair",                                   plot_05),
    "06": ("Stacked temporal bar — pooled",                 plot_06),
    "07": ("Stacked temporal bar — by pair",                plot_07),
    "08": ("Per-condition temporal detail",                 plot_08),
    "09": ("Individual seed traces + mean",                 plot_09),
    "10": ("Generalised per-condition",                     plot_10),
    "11": ("p* terminal strip  [close-it]",                 plot_11),
    "12": ("CLIP comparison bar  [close-it]",               plot_12),
    "13": ("JS² divergence: p* vs mono  [close-it]",        plot_13),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Gap analysis figures 00–13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir", default="experiments/inversion/gap_analysis",
        help="Directory containing per_seed_distances.json, trajectory_distances.json, "
             "and all_pairs_gap.json",
    )
    p.add_argument(
        "--output-dir", default="experiments/inversion/gap_analysis/figures",
        help="Directory to write PNG files (created if absent)",
    )
    p.add_argument(
        "--plot", default="all",
        help="Which plot to generate: 'all' (default) or a number like '06' / '13'",
    )
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir} ...")
    df_term = load_terminal(data_dir)
    df_traj = load_trajectory(data_dir)
    print(f"  Terminal:   {len(df_term)} records "
          f"({df_term['pair'].nunique()} pairs × {df_term['seed'].nunique()} seeds)")
    print(f"  Trajectory: {len(df_traj)} records")
    if "d_T_pstar" in df_term.columns:
        print("  d_T_pstar column present — plots 11 and 13 will run.")
    else:
        print("  d_T_pstar column absent  — plots 11 and 13 will be skipped.")
        print("  (Re-run measure_composability_gap.py to populate it.)")

    target = args.plot.strip().lower()
    if target == "all":
        keys = sorted(PLOTS.keys())
    else:
        key = target.zfill(2)
        if key not in PLOTS:
            print(f"Unknown plot '{target}'.\nValid options: {list(PLOTS.keys())} or 'all'")
            sys.exit(1)
        keys = [key]

    for key in keys:
        label, fn = PLOTS[key]
        print(f"\n[{key}] {label}")
        fn(df_term, df_traj, out_dir, data_dir=data_dir)

    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
