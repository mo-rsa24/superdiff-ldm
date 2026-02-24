#!/usr/bin/env python3
"""
Trajectory analysis visualisation suite — Plots 00 through 13.

Scaffolded EDA that builds a traceable, auditable chain of evidence for the
claim that SuperDiff-AND produces more balanced denoising trajectories than
monolithic or PoE baselines.

Layer 0 — Identity anchor (what are we looking at?)
    00  Decoded images      — load existing decoded_images.png as the result anchor

Layer 1 — Raw scalar summaries (no projection needed)
    01  Latent norms        — ||x_t||₂ per condition over denoising steps
    02  Velocity magnitudes — ||v_t||₂ per condition over denoising steps
    03  Pairwise L2         — distance between key condition pairs over steps

Layer 2 — Structure within the divergence
    04  Divergence onset    — first step at which each pair separates (bar chart)
    05  Path lengths        — total L2 traversed per condition (bar chart)
    06  Terminal heatmap    — N×N endpoint distance matrix (heat map)

Layer 3 — Geometry (projection introduced only after raw evidence is established)
    07  PCA scree           — variance explained by each PC (trust calibration)
    08  Manifold overlaid   — all conditions on one PCA/MDS plot, uniform time colour
    09  Per-condition panels — one subplot per condition, shared axis

Layer 4 — Mechanism (unique SuperDiff evidence)
    10  Kappa dynamics      — per-concept composition weight κ(t) over steps
    11  Log-likelihood      — log q_cᵢ(x_t) per concept over steps

Layer 5 — Closing (semantic evaluation)
    12  CLIP terminal       — cosine similarity to class prompts per condition
    13  CLIP intermediate   — concept coverage at intermediate steps (if available)

Usage
-----
# All plots:
conda run -n superdiff python scripts/plot_trajectory_analysis.py \\
    --data-dir experiments/trajectory_dynamics/<timestamp>

# Single plot:
conda run -n superdiff python scripts/plot_trajectory_analysis.py \\
    --data-dir experiments/trajectory_dynamics/<timestamp> --plot 10

# Custom output directory:
conda run -n superdiff python scripts/plot_trajectory_analysis.py \\
    --data-dir experiments/trajectory_dynamics/<timestamp> \\
    --output-dir experiments/trajectory_dynamics/<timestamp>/figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("Warning: scipy not found — KDE in plot 01/02 will fall back to plain lines.")


# ---------------------------------------------------------------------------
# Colour / style constants  (mirrors trajectory_dynamics_experiment.py)
# ---------------------------------------------------------------------------

CONDITION_COLORS = {
    "prompt_a":          "#e63946",
    "prompt_b":          "#457b9d",
    "monolithic":        "#2a9d8f",
    "superdiff":         "#e9c46a",
    "superdiff_det":     "#f4a261",
    "superdiff_fm_ode":  "#d4a373",
    "superdiff_multi":   "#e76f51",
    "superdiff_guided":  "#264653",
    "poe":               "#9b5de5",
    "composable_not":    "#c1121f",
    "superdiff_not":     "#780000",
}

CONDITION_CMAPS = {
    "prompt_a":         "Reds",
    "prompt_b":         "Blues",
    "monolithic":       "Greens",
    "superdiff":        "Oranges",
    "superdiff_det":    "YlOrBr",
    "superdiff_fm_ode": "copper",
    "superdiff_multi":  "RdPu",
    "superdiff_guided": "BuGn",
    "poe":              "Purples",
    "composable_not":   "RdPu",
    "superdiff_not":    "Reds",
}

_FALLBACK_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
    "#9b5de5", "#f4a261", "#264653", "#d62828",
]
_FALLBACK_CMAPS = ["Reds", "Blues", "Greens", "Oranges", "Purples", "copper", "RdPu", "BuGn"]

# Colours for concept lines inside kappa / log-likelihood plots
_CONCEPT_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#9b5de5", "#f4a261"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cond_color(name: str, idx: int) -> str:
    return CONDITION_COLORS.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _cond_cmap(name: str, idx: int) -> str:
    return CONDITION_CMAPS.get(name, _FALLBACK_CMAPS[idx % len(_FALLBACK_CMAPS)])


def hide_top_right(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path) -> dict:
    p = data_dir / "trajectory_data.json"
    if not p.exists():
        sys.exit(
            f"Not found: {p}\n"
            "Run trajectory_dynamics_experiment.py first (it saves trajectory_data.json)."
        )
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _all_steps(data: dict) -> list:
    return list(range(data["n_steps"]))


def _shared_axis_limits(data: dict) -> tuple:
    """Return (xlim, ylim) covering all projected trajectories with 10 % margin."""
    all_pts = []
    for cond in data["per_condition"].values():
        x, y = cond.get("projected_x", []), cond.get("projected_y", [])
        if x and y:
            all_pts.append(np.column_stack([x, y]))
    if not all_pts:
        return ((-1, 1), (-1, 1))
    pts = np.vstack(all_pts)
    margin = 0.10 * (pts.max(0) - pts.min(0) + 1e-8)
    return (
        (pts[:, 0].min() - margin[0], pts[:, 0].max() + margin[0]),
        (pts[:, 1].min() - margin[1], pts[:, 1].max() + margin[1]),
    )


# ===========================================================================
# Plot 00 — Decoded images (load existing decoded_images.png)
# ===========================================================================

def plot_00(data: dict, data_dir: Path, out_dir: Path):
    img_name = data.get("decoded_images_path", "decoded_images.png")
    img_path = data_dir / img_name
    if not img_path.exists():
        print(f"  Skipping plot 00: {img_path} not found.")
        return

    img = plt.imread(str(img_path))
    cfg = data.get("config", {})
    n_steps = data.get("num_inference_steps", data.get("n_steps", "?"))

    fig, ax = plt.subplots(figsize=(min(img.shape[1] / 100, 18), min(img.shape[0] / 100, 8)))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        f"Plot 00 — Decoded Final Latents  "
        f"(seed={cfg.get('seed', '?')},  steps={cfg.get('num_inference_steps', n_steps)},  "
        f"gs={cfg.get('guidance_scale', '?')})\n"
        f"Left = concept A  ·  Right = concept B  ·  "
        f"Middle = composition methods",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_00_decoded_images.png")


# ===========================================================================
# Plot 01 — Latent norms: ||x_t||₂ per condition over denoising steps
# ===========================================================================

def plot_01(data: dict, data_dir: Path, out_dir: Path):
    conditions = data["conditions"]
    labels     = data["labels"]
    steps      = _all_steps(data)

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, name in enumerate(conditions):
        norms = data["per_condition"][name].get("latent_norms", [])
        if not norms:
            continue
        c = _cond_color(name, idx)
        ax.plot(steps[:len(norms)], norms, color=c,
                label=labels.get(name, name), lw=2.2)

    ax.set_xlabel("Denoising step  (0 = shared noise  x_T → 50 = clean image)", fontsize=12)
    ax.set_ylabel("||x_t||₂  (latent L2 norm)", fontsize=12)
    ax.set_title(
        "Plot 01 — Latent Norm Over Time\n"
        "All conditions start from the same x_T.  Convergence rate reveals conditioning strength.",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_01_latent_norms.png")


# ===========================================================================
# Plot 02 — Velocity magnitudes: ||v_t||₂ per condition
# ===========================================================================

def plot_02(data: dict, data_dir: Path, out_dir: Path):
    conditions = data["conditions"]
    labels     = data["labels"]
    steps      = _all_steps(data)

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, name in enumerate(conditions):
        vmags = data["per_condition"][name].get("velocity_magnitudes", [])
        if not vmags:
            continue
        c = _cond_color(name, idx)
        # Drop the trailing zero (store_final stores no velocity)
        effective = vmags[:-1]
        ax.plot(steps[:len(effective)], effective, color=c,
                label=labels.get(name, name), lw=2.2)

    ax.set_xlabel("Denoising step", fontsize=12)
    ax.set_ylabel("||v_t||₂  (velocity field magnitude)", fontsize=12)
    ax.set_title(
        "Plot 02 — Velocity Magnitude Over Time\n"
        "SuperDiff's kappa reweighting produces a distinctive velocity profile vs fixed-guidance baselines.",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_02_velocity_magnitudes.png")


# ===========================================================================
# Plot 03 — Pairwise L2 distances (all pairs shown with thin lines;
#           pairs involving a superdiff condition highlighted bold)
# ===========================================================================

def plot_03(data: dict, data_dir: Path, out_dir: Path):
    pairwise = data.get("pairwise_l2", {})
    if not pairwise:
        print("  Skipping plot 03: no pairwise_l2 data.")
        return

    steps = _all_steps(data)
    labels = data["labels"]
    conditions = data["conditions"]
    superdiff_keys = {n for n in conditions
                      if any(s in n for s in ("superdiff", "poe"))}

    fig, ax = plt.subplots(figsize=(11, 5.5))

    pair_colors = plt.cm.Set2(np.linspace(0, 1, max(len(pairwise), 1)))

    for idx, (pair_key, dists) in enumerate(pairwise.items()):
        ni, nj = pair_key.split("|", 1)
        lbl_i  = labels.get(ni, ni)
        lbl_j  = labels.get(nj, nj)
        is_highlighted = bool(superdiff_keys & {ni, nj})

        ax.plot(steps[:len(dists)], dists,
                color=pair_colors[idx % len(pair_colors)],
                lw=2.5 if is_highlighted else 1.0,
                alpha=0.95 if is_highlighted else 0.45,
                label=f"{lbl_i}  vs  {lbl_j}",
                zorder=4 if is_highlighted else 2)

    ax.set_xlabel("Denoising step", fontsize=12)
    ax.set_ylabel("L2 distance (latent space)", fontsize=12)
    ax.set_title(
        "Plot 03 — Pairwise Latent Trajectory Distances Over Time\n"
        "Bold = pairs involving SuperDiff/PoE.  "
        "Distances start at 0 (shared x_T) and accumulate.",
        fontsize=13,
    )
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_03_pairwise_l2.png")


# ===========================================================================
# Plot 04 — Divergence onset: first step at which each pair separates
# ===========================================================================

def plot_04(data: dict, data_dir: Path, out_dir: Path):
    onset = data.get("summary", {}).get("divergence_onset_step", {})
    if not onset:
        print("  Skipping plot 04: divergence_onset_step not in summary.")
        return

    pairs  = list(onset.keys())
    values = [onset[p] for p in pairs]
    y_pos  = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(pairs))))
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.85, len(pairs)))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.65)

    for bar, v in zip(bars, values):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"step {int(v)}", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel("First denoising step where pair distance > 1 % of max", fontsize=11)
    ax.set_title(
        "Plot 04 — Divergence Onset: When Do Trajectories Begin to Separate?\n"
        "Earlier onset = conditioning difference is felt sooner in the denoising process.",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_04_divergence_onset.png")


# ===========================================================================
# Plot 05 — Path lengths: total L2 traversed per condition
# ===========================================================================

def plot_05(data: dict, data_dir: Path, out_dir: Path):
    path_lengths = data.get("summary", {}).get("path_lengths", {})
    if not path_lengths:
        print("  Skipping plot 05: path_lengths not in summary.")
        return

    labels_raw = list(path_lengths.keys())
    values     = [path_lengths[l] for l in labels_raw]
    x          = np.arange(len(labels_raw))

    # Colour by condition name (best-effort match)
    conditions = data["conditions"]
    lbl_to_cond = {data["labels"].get(n, n): n for n in conditions}
    colors = [
        _cond_color(lbl_to_cond.get(lbl, ""), i)
        for i, lbl in enumerate(labels_raw)
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels_raw) * 1.8), 5))
    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_raw, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Total path length  (cumulative L2)", fontsize=11)
    ax.set_title(
        "Plot 05 — Total Latent Path Length per Condition\n"
        "Shorter = more direct trajectory.  Longer = more correction / wandering.",
        fontsize=13,
    )
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_05_path_lengths.png")


# ===========================================================================
# Plot 06 — Terminal endpoint distance heatmap: N×N pairwise L2 at step T
# ===========================================================================

def plot_06(data: dict, data_dir: Path, out_dir: Path):
    ep_dists = data.get("summary", {}).get("endpoint_distances_l2", {})
    conditions = data["conditions"]
    labels = data["labels"]

    if not ep_dists:
        print("  Skipping plot 06: endpoint_distances_l2 not in summary.")
        return

    N = len(conditions)
    mat = np.zeros((N, N))

    for i, ni in enumerate(conditions):
        for j, nj in enumerate(conditions):
            if i == j:
                continue
            key1 = f"{labels.get(ni, ni)} vs {labels.get(nj, nj)}"
            key2 = f"{labels.get(nj, nj)} vs {labels.get(ni, ni)}"
            val = ep_dists.get(key1, ep_dists.get(key2, 0.0))
            mat[i, j] = val

    short_labels = [labels.get(n, n).replace("SuperDIFF ", "SD-")
                    .replace("FM-ODE", "FMO")[:20]
                    for n in conditions]

    fig, ax = plt.subplots(figsize=(max(5, N), max(4, N - 1)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="L2 distance at step T")

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Annotate cells
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if val < mat.max() * 0.6 else "white")

    ax.set_title(
        "Plot 06 — Terminal Endpoint Distance Heatmap\n"
        "Pairwise L2 distance at final denoising step T.  "
        "Low = similar endpoints.",
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_06_endpoint_heatmap.png")


# ===========================================================================
# Plot 07 — PCA scree / MDS stress
# ===========================================================================

def plot_07(data: dict, data_dir: Path, out_dir: Path):
    method  = data.get("projection_method", "pca")
    pca_var = data.get("pca_variance_explained", [])
    stress  = data.get("mds_stress")

    if method == "mds":
        # Just show a text summary
        fig, ax = plt.subplots(figsize=(5, 3))
        msg = f"MDS stress: {stress:.4f}" if stress is not None else "MDS stress: not recorded"
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14,
                transform=ax.transAxes)
        ax.set_title("Plot 07 — MDS Stress (projection quality)", fontsize=13)
        ax.axis("off")
        fig.tight_layout()
        save_fig(fig, out_dir / "plot_07_projection_quality.png")
        return

    if not pca_var:
        print("  Skipping plot 07: pca_variance_explained not recorded.")
        return

    cumvar = np.cumsum(pca_var)
    n_shown = min(len(pca_var), 20)
    x = np.arange(1, n_shown + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Bar: individual variance
    ax1.bar(x, [v * 100 for v in pca_var[:n_shown]],
            color="#457b9d", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Variance explained  (%)", fontsize=11)
    ax1.set_title("Individual PC variance", fontsize=12)
    ax1.grid(axis="y", alpha=0.25)
    hide_top_right(ax1)

    # Line: cumulative
    ax2.plot(x, cumvar[:n_shown] * 100, color="#e63946", lw=2.5, marker="o", ms=5)
    ax2.axhline(80, color="#888", lw=1, ls="--")
    ax2.axhline(95, color="#444", lw=1, ls="--")
    ax2.text(n_shown * 0.98, 81, "80 %", ha="right", fontsize=9, color="#888")
    ax2.text(n_shown * 0.98, 96, "95 %", ha="right", fontsize=9, color="#444")
    ax2.set_xlabel("Number of PCs", fontsize=11)
    ax2.set_ylabel("Cumulative variance  (%)", fontsize=11)
    ax2.set_title("Cumulative explained variance", fontsize=12)
    ax2.set_ylim(0, 101)
    ax2.grid(alpha=0.25)
    hide_top_right(ax2)

    top2 = float(np.sum(pca_var[:2])) * 100
    fig.suptitle(
        f"Plot 07 — PCA Scree Plot  (PC1+PC2 = {top2:.1f} % of variance)\n"
        "This establishes how much information the 2D manifold preserves.",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_07_pca_scree.png")


# ===========================================================================
# Plot 08 — Trajectory manifold: all conditions, uniform time colour (viridis)
# ===========================================================================

def plot_08(data: dict, data_dir: Path, out_dir: Path):
    conditions = data["conditions"]
    labels     = data["labels"]
    n_steps    = data["n_steps"]
    method     = data.get("projection_method", "pca")
    ax_prefix  = "MDS" if method == "mds" else "PC"

    if not any(data["per_condition"][n].get("projected_x") for n in conditions):
        print("  Skipping plot 08: no projected coordinates in trajectory_data.json.")
        return

    norm      = Normalize(vmin=0, vmax=n_steps - 2)
    cmap_name = "viridis"
    cmap      = cm.get_cmap(cmap_name)

    _MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
    _EP_COLORS = _FALLBACK_COLORS

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, name in enumerate(conditions):
        pc = data["per_condition"][name]
        x, y = pc.get("projected_x", []), pc.get("projected_y", [])
        if not x:
            continue
        pts = np.column_stack([x, y])

        # Time-gradient line (uniform viridis)
        points   = pts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap_name, norm=norm,
                            linewidths=2.2, alpha=0.88)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

        # Endpoint marker (differentiated by shape + colour)
        marker   = _MARKERS[idx % len(_MARKERS)]
        ep_color = _EP_COLORS[idx % len(_EP_COLORS)]
        ax.plot(pts[-1, 0], pts[-1, 1], marker=marker, color=ep_color,
                markersize=11, markeredgecolor="black", markeredgewidth=0.9,
                zorder=6, linestyle="none")
        ax.annotate(labels.get(name, name)[:18],
                    xy=(pts[-1, 0], pts[-1, 1]),
                    fontsize=8, fontweight="bold", color=ep_color,
                    textcoords="offset points", xytext=(6, 3))

    # Shared origin
    first_name = next(n for n in conditions if data["per_condition"][n].get("projected_x"))
    origin = np.array([data["per_condition"][first_name]["projected_x"][0],
                       data["per_condition"][first_name]["projected_y"][0]])
    ax.plot(origin[0], origin[1], "ko", markersize=8, zorder=7)
    ax.annotate(r"$x_T$", xy=(origin[0], origin[1]), fontsize=11, fontweight="bold",
                textcoords="offset points", xytext=(-14, -12))

    # Colorbar for time
    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, label="Denoising step")

    legend_handles = [
        Line2D([0], [0], marker=_MARKERS[i % len(_MARKERS)],
               color=_EP_COLORS[i % len(_EP_COLORS)],
               markeredgecolor="black", markersize=9,
               linestyle="none", label=labels.get(n, n))
        for i, n in enumerate(conditions)
        if data["per_condition"][n].get("projected_x")
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="best")
    ax.autoscale()
    ax.set_xlabel(f"{ax_prefix} 1", fontsize=12)
    ax.set_ylabel(f"{ax_prefix} 2", fontsize=12)
    ax.set_title(
        f"Plot 08 — Latent Trajectory Manifold ({method.upper()}, uniform time colour)\n"
        "Same colour = same denoising step.  Endpoints differentiated by marker shape + colour.",
        fontsize=13,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_08_manifold_overlaid.png")


# ===========================================================================
# Plot 09 — Per-condition trajectory subplots (shared axis)
# ===========================================================================

def plot_09(data: dict, data_dir: Path, out_dir: Path):
    conditions = data["conditions"]
    labels     = data["labels"]
    n_steps    = data["n_steps"]
    method     = data.get("projection_method", "pca")
    ax_prefix  = "MDS" if method == "mds" else "PC"

    active = [n for n in conditions if data["per_condition"][n].get("projected_x")]
    if not active:
        print("  Skipping plot 09: no projected coordinates.")
        return

    xlim, ylim = _shared_axis_limits(data)
    norm       = Normalize(vmin=0, vmax=n_steps - 2)
    n_cond     = len(active)

    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5))
    if n_cond == 1:
        axes = [axes]

    for ax, name in zip(axes, active):
        pc   = data["per_condition"][name]
        x, y = pc["projected_x"], pc["projected_y"]
        pts  = np.column_stack([x, y])
        cmap = _cond_cmap(name, active.index(name))

        points   = pts.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.0)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

        ax.plot(pts[0, 0],  pts[0, 1],  "ko", markersize=6, zorder=5)
        ax.plot(pts[-1, 0], pts[-1, 1], "s",
                color=cm.get_cmap(cmap)(0.85), markersize=8, zorder=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(labels.get(name, name), fontsize=10, fontweight="bold")
        ax.set_xlabel(f"{ax_prefix} 1", fontsize=10)
        ax.set_ylabel(f"{ax_prefix} 2", fontsize=10)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Plot 09 — Per-Condition Trajectory Subplots  ({method.upper()})\n"
        "Shared axis — spatial positions are directly comparable across panels.",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_09_per_condition_subplots.png")


# ===========================================================================
# Plot 10 — Kappa dynamics: per-concept weight κ(t) for SuperDiff conditions
# ===========================================================================

def plot_10(data: dict, data_dir: Path, out_dir: Path):
    conditions = data["conditions"]
    labels     = data["labels"]
    steps      = _all_steps(data)

    superdiff_conds = [
        n for n in conditions
        if data["per_condition"][n].get("kappa") is not None
    ]

    if not superdiff_conds:
        print("  Skipping plot 10: no kappa data (only SuperDiff conditions have kappa).")
        return

    n_sd = len(superdiff_conds)
    fig, axes = plt.subplots(1, n_sd, figsize=(5.5 * n_sd, 5), sharey=False)
    if n_sd == 1:
        axes = [axes]

    for ax, name in zip(axes, superdiff_conds):
        kappa_series = data["per_condition"][name]["kappa"]   # (T+1, M) list of lists
        concept_labels = data["per_condition"][name].get("concept_labels", [])
        M = len(kappa_series[0]) if kappa_series else 0

        for m in range(M):
            k_m = [row[m] for row in kappa_series]
            clbl = concept_labels[m] if m < len(concept_labels) else f"concept {m+1}"
            ax.plot(steps[:len(k_m)], k_m,
                    color=_CONCEPT_COLORS[m % len(_CONCEPT_COLORS)],
                    label=clbl, lw=2.2)

        ax.axhline(0.5, color="#AAAAAA", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlabel("Denoising step", fontsize=11)
        ax.set_ylabel("κ  (composition weight)", fontsize=11)
        ax.set_title(labels.get(name, name), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)
        ax.set_ylim(bottom=None)
        hide_top_right(ax)

    fig.suptitle(
        "Plot 10 — Kappa Dynamics: Per-Concept Composition Weights Over Time\n"
        "κ = 0.5 (dashed) = equal weight.  Deviations reflect density imbalance being corrected.",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_10_kappa_dynamics.png")


# ===========================================================================
# Plot 11 — Log-likelihood evolution per concept
# ===========================================================================

def plot_11(data: dict, data_dir: Path, out_dir: Path):
    conditions = data["conditions"]
    labels     = data["labels"]
    steps      = _all_steps(data)

    sd_conds = [
        n for n in conditions
        if data["per_condition"][n].get("log_likelihoods") is not None
    ]

    if not sd_conds:
        print("  Skipping plot 11: no log_likelihoods data.")
        return

    n_sd = len(sd_conds)
    fig, axes = plt.subplots(1, n_sd, figsize=(5.5 * n_sd, 5), sharey=False)
    if n_sd == 1:
        axes = [axes]

    for ax, name in zip(axes, sd_conds):
        ll_series = data["per_condition"][name]["log_likelihoods"]   # (T+1, M) list
        concept_labels = data["per_condition"][name].get("concept_labels", [])
        M = len(ll_series[0]) if ll_series else 0

        for m in range(M):
            ll_m = [row[m] for row in ll_series]
            clbl = concept_labels[m] if m < len(concept_labels) else f"concept {m+1}"
            ax.plot(steps[:len(ll_m)], ll_m,
                    color=_CONCEPT_COLORS[m % len(_CONCEPT_COLORS)],
                    label=clbl, lw=2.2)

        ax.set_xlabel("Denoising step", fontsize=11)
        ax.set_ylabel("Approx. log q_c(x_t)", fontsize=11)
        ax.set_title(labels.get(name, name), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)
        hide_top_right(ax)

    fig.suptitle(
        "Plot 11 — Log-Likelihood Evolution per Concept\n"
        "Density equalization claim: log q_c₁ and log q_c₂ should evolve at the same rate under AND.",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_11_log_likelihoods.png")


# ===========================================================================
# Plot 12 — CLIP terminal: cosine similarity to class prompts, per condition
# ===========================================================================

def plot_12(data: dict, data_dir: Path, out_dir: Path):
    clip = data.get("clip", {})
    if not clip:
        print("  Skipping plot 12: no CLIP data (run without --no-clip-probe).")
        return

    conditions = data["conditions"]
    labels     = data["labels"]

    # Build class list from first condition's keys
    first = next(iter(clip.values()), {})
    class_keys = list(first.keys())
    if not class_keys:
        print("  Skipping plot 12: CLIP similarity dict is empty.")
        return

    n_cond   = len(conditions)
    n_class  = len(class_keys)
    x        = np.arange(n_class)
    width    = 0.8 / max(n_cond, 1)
    offsets  = (np.arange(n_cond) - (n_cond - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(max(9, n_class * 2), 5))

    for idx, name in enumerate(conditions):
        sims = clip.get(name, {})
        vals = [sims.get(k, 0.0) for k in class_keys]
        ax.bar(x + offsets[idx], vals, width * 0.9,
               color=_cond_color(name, idx),
               label=labels.get(name, name), alpha=0.85,
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(class_keys, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("CLIP cosine similarity  (↑ better)", fontsize=11)
    ax.set_title(
        "Plot 12 — CLIP Zero-Shot Classification: Terminal Latents\n"
        "Higher similarity to a class prompt = stronger concept presence in the generated image.",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.25)
    hide_top_right(ax)
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_12_clip_terminal.png")


# ===========================================================================
# Plot 13 — CLIP intermediate: concept coverage at multiple denoising stages
#           Requires clip_intermediate data in trajectory_data.json (optional).
# ===========================================================================

def plot_13(data: dict, data_dir: Path, out_dir: Path):
    clip_inter = data.get("clip_intermediate", {})
    if not clip_inter:
        print("  Skipping plot 13: no clip_intermediate data.")
        print("  (Add intermediate CLIP decoding to save_trajectory_data() to enable this.)")
        return

    # clip_intermediate format: {step_label: {condition: {class: sim}}}
    step_labels = list(clip_inter.keys())
    conditions  = data["conditions"]
    labels      = data["labels"]

    first_step = next(iter(clip_inter.values()), {})
    first_cond = next(iter(first_step.values()), {})
    class_keys = list(first_cond.keys())

    n_steps = len(step_labels)
    n_class = len(class_keys)
    n_cond  = len(conditions)

    fig, axes = plt.subplots(1, n_steps, figsize=(4.5 * n_steps, 5), sharey=True)
    if n_steps == 1:
        axes = [axes]

    for ax, slbl in zip(axes, step_labels):
        step_data = clip_inter[slbl]
        x        = np.arange(n_class)
        width    = 0.8 / max(n_cond, 1)
        offsets  = (np.arange(n_cond) - (n_cond - 1) / 2) * width

        for idx, name in enumerate(conditions):
            sims = step_data.get(name, {})
            vals = [sims.get(k, 0.0) for k in class_keys]
            ax.bar(x + offsets[idx], vals, width * 0.9,
                   color=_cond_color(name, idx),
                   label=labels.get(name, name) if slbl == step_labels[0] else "_nolegend_",
                   alpha=0.85, edgecolor="white")

        ax.set_title(f"Step {slbl}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(class_keys, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        hide_top_right(ax)

    axes[0].set_ylabel("CLIP cosine similarity", fontsize=11)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Plot 13 — CLIP Concept Coverage at Intermediate Denoising Steps\n"
        "Does AND maintain both concepts throughout?  Does mono commit to one early?",
        fontsize=13,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "plot_13_clip_intermediate.png")


# ===========================================================================
# Registry and CLI
# ===========================================================================

PLOTS = {
    "00": ("Decoded images  [identity anchor]",                plot_00),
    "01": ("Latent norms over time",                           plot_01),
    "02": ("Velocity magnitudes over time",                    plot_02),
    "03": ("Pairwise L2 distances",                            plot_03),
    "04": ("Divergence onset bar chart",                       plot_04),
    "05": ("Path length bar chart",                            plot_05),
    "06": ("Terminal endpoint distance heatmap",               plot_06),
    "07": ("PCA scree / MDS stress",                           plot_07),
    "08": ("Manifold overlaid  [uniform time colour]",         plot_08),
    "09": ("Per-condition trajectory subplots",                plot_09),
    "10": ("Kappa dynamics  [SuperDiff only]",                 plot_10),
    "11": ("Log-likelihood evolution  [SuperDiff only]",       plot_11),
    "12": ("CLIP terminal similarity",                         plot_12),
    "13": ("CLIP intermediate  [requires clip_intermediate]",  plot_13),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Trajectory analysis figures 00–13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing trajectory_data.json (the experiment output directory)",
    )
    p.add_argument(
        "--output-dir", default="",
        help="Directory to write PNG files (default: <data-dir>/figures/)",
    )
    p.add_argument(
        "--plot", default="all",
        help="Which plot to generate: 'all' (default) or a number like '08' / '10'",
    )
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir) if args.output_dir else data_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectory_data.json from {data_dir} ...")
    data = load_data(data_dir)

    conditions = data.get("conditions", [])
    print(f"  Conditions : {conditions}")
    print(f"  Steps      : {data.get('n_steps', '?')}")
    print(f"  Projection : {data.get('projection_method', '?')}")

    kappa_conds = [n for n in conditions
                   if data["per_condition"].get(n, {}).get("kappa") is not None]
    if kappa_conds:
        print(f"  Kappa data : {kappa_conds}")
    else:
        print("  Kappa data : none  (plots 10 & 11 will be skipped)")

    target = args.plot.strip().lower()
    if target == "all":
        keys = sorted(PLOTS.keys())
    else:
        key = target.zfill(2)
        if key not in PLOTS:
            print(f"Unknown plot '{target}'.  Valid: {list(PLOTS.keys())} or 'all'")
            sys.exit(1)
        keys = [key]

    for key in keys:
        label, fn = PLOTS[key]
        print(f"\n[{key}] {label}")
        fn(data, data_dir, out_dir)

    print(f"\nDone. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
